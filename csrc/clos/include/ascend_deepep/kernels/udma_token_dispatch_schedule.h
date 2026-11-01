#pragma once

#include "ascend_deepep/kernels/udma_dispatch_sender.h"
#include "ascend_deepep/kernels/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep::udma {
// Separate list windows from the sender's in-flight WQE staging region.
constexpr uint32_t kTokenListsOffset = kDispatchSenderBytes;
static_assert(kTokenListsOffset + kMaxPeers * kTileTokens * sizeof(int32_t) <= 216 * 1024);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

template <class Resource>
__aicore__ inline void DispatchTokenRemote(
    Resource& resource, GM_ADDR x, GM_ADDR scales, GM_ADDR plan, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const DispatchTilingData& td, uint32_t core, uint32_t cores)
{
    auto wq = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
    auto lists = resource.ubBuf.template GetBufferByByte<int32_t>(kTokenListsOffset);
    const uint32_t workers = TokenWorkers(cores);
    if (core < workers)
        for (uint32_t first = 0; first < td.world_size - 1; first += kMaxPeers) {
            DispatchSender sender(
                td, core, wq, workspace, x, scales, first, core);
            AscendC::GlobalTensor<int32_t> plan_gm;
            plan_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
            AscendC::GlobalTensor<int32_t> rank_gm, channel_gm;
            rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
            channel_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix));
            const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1) / td.num_channels;
            const uint32_t max_tile = 65536U / td.world_size < kTileTokens ? 65536U / td.world_size : kTileTokens;
            const uint32_t plan_stride = (channel_tokens + 7U) / 8U * 8U;
            const uint32_t scale_bytes = td.num_scales * td.scale_element_bytes;
            const uint32_t fields = td.has_x_scales && scale_bytes ? 2U : 1U;
            for (uint32_t ch = core; ch < td.num_channels; ch += workers) {
                const uint32_t begin = ch * channel_tokens < td.num_tokens ? ch * channel_tokens : td.num_tokens;
                const uint32_t end = begin + channel_tokens < td.num_tokens ? begin + channel_tokens : td.num_tokens;
                uint32_t row[kMaxPeers]{}, consumed[kMaxPeers]{}, total[kMaxPeers]{};
                for (uint32_t p = 0; p < sender.Count(); ++p) {
                    const uint32_t peer = sender.Rank(p);
                    const uint32_t previous = ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0;
                    total[p] = channel_gm.GetValue(peer * td.num_channels + ch) - previous;
                    row[p] = (td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0) +
                             (ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0);
                }
                for (uint32_t start = begin; start < end; start += max_tile) {
                    const uint32_t size = end - start < max_tile ? end - start : max_tile;
                    uint32_t counts[kMaxPeers]{}, cursor[kMaxPeers]{}, field[kMaxPeers]{}, remaining = 0;
                    // Keep the original input-token tile and peer visit boundaries.
                    // A sorted list can contribute at most `size` hits to this tile.
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    for (uint32_t p = 0; p < sender.Count(); ++p) {
                        const uint32_t left = total[p] - consumed[p];
                        const uint32_t load = left < size ? left : size;
                        if (load == 0)
                            continue;
                        const uint64_t base = (uint64_t(sender.Rank(p)) * td.num_channels + ch) * plan_stride;
                        AscendC::DataCopyExtParams copy{1, load * uint32_t(sizeof(int32_t)), 0, 0, 0};
                        AscendC::DataCopyPad(
                            lists[p * kTileTokens], plan_gm[base + consumed[p]], copy,
                            AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                    for (uint32_t p = 0; p < sender.Count(); ++p) {
                        const uint32_t left = total[p] - consumed[p];
                        const uint32_t load = left < size ? left : size;
                        while (counts[p] < load && uint32_t(lists.GetValue(p * kTileTokens + counts[p])) < start + size)
                            ++counts[p];
                        remaining += counts[p];
                        consumed[p] += counts[p];
                    }
                    while (remaining) {
                        for (uint32_t p = 0; p < sender.Count(); ++p) {
                            uint32_t visit = td.udma_wqes_per_visit;
                            while (visit && cursor[p] < counts[p]) {
                                const uint64_t base = (uint64_t(sender.Rank(p)) * td.num_channels + ch) * plan_stride;
                                const uint64_t index = base + consumed[p] - counts[p] + cursor[p];
                                const uint32_t available = (counts[p] - cursor[p]) * fields - field[p];
                                const uint32_t count = sender.PutRows(
                                    p, reinterpret_cast<const __gm__ int32_t*>(plan) + index, row[p], field[p],
                                    available < visit ? available : visit);
                                const uint32_t rows = (field[p] + count) / fields;
                                field[p] = (field[p] + count) % fields;
                                cursor[p] += rows;
                                row[p] += rows;
                                remaining -= rows;
                                visit -= count;
                            }
                        }
                    }
                }
            }
            // Empty channel owners still publish to every peer. A marker is private
            // to this worker; it never stands for another worker's queue completion.
            sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core);
            sender.Quiet(); // Drain every physical SQ state before reusing this window.
        }
    WaitIncoming<true>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, cores, workers);
    // Also includes the parallel SIMT metadata writers on every rank.
    aclshmemi_sync_npu_v4<true>(0);
}
#endif
} // namespace ascend_deepep::udma
