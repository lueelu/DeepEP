#pragma once

#include "kernels/dispatch/dispatch_device_trace.h"

#include "kernels/dispatch/udma_dispatch_sender.h"
#include "kernels/dispatch/channel_metadata.h"
#include "common/platform.h"
#include "common/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep::udma {
// Separate list windows from the sender's in-flight WQE staging region.
constexpr uint32_t kListsOffset = kDispatchSenderBytes;
static_assert(kListsOffset + kMaxPeers * kTileTokens * sizeof(int32_t) <= platform::kKernelUbBytes);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

// Each producer owns one 512-byte local flag slot. Before Begin, zero is
// drained by every core. After both SIMT outputs drain, producers publish one.
template <class Resource>
__aicore__ inline void StorePackedScaleReady(Resource& resource, const DispatchTilingData& td,
    uint32_t core, uint64_t value)
{
    auto scratch = resource.template GetBufferByByte<uint64_t>(0);
    scratch.SetValue(0, value);
    AscendC::GlobalTensor<uint64_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(td.packed_scale_ready + uint64_t(core) * 512U));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::DataCopyExtParams copy{1, 8U, 0, 0, 0};
    AscendC::DataCopyPad(flag, scratch, copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
}

__aicore__ inline void WaitPackedScaleReady(const DispatchTilingData& td, uint32_t core, uint32_t cores)
{
    const uint32_t first = MetadataFirst(td.world_size, td.udma_peers_per_worker, cores,
        td.udma_peer_order, td.udma_ranks_per_group, td.udma_workers_per_peer);
    const uint64_t started = static_cast<uint64_t>(AscendC::GetSystemCycle());
    for (uint32_t producer = first; producer < cores; ++producer) {
        auto* address = reinterpret_cast<__gm__ uint64_t*>(td.packed_scale_ready + uint64_t(producer) * 512U);
        AscendC::GlobalTensor<uint64_t> flag;
        flag.SetGlobalBuffer(address);
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
            const uint64_t value = flag.GetValue(0);
            if (value == 1U) break;
            const uint64_t elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - started;
            if (elapsed >= td.npu_timeout_cycles) {
                AscendC::printf(
                    "[CLOS_SCALE_TIMEOUT] op=dispatch stage=scale_ready rank=%u aiv_core_idx=%u "
                    "producer_aiv_core_idx=%u instance_id=%u cached=%u expected=1 observed=%llu "
                    "elapsed_us=%llu timeout_s=%u\n",
                    td.rank, core, producer, td.instance_id, td.cached_mode, static_cast<unsigned long long>(value),
                    static_cast<unsigned long long>(elapsed / kNpuSystemCyclesPerUs),
                    static_cast<uint32_t>(td.npu_timeout_cycles / kNpuSystemCyclesPerSecond));
                trap(); return;
            }
        }
    }
}

struct NoDispatchOverlapWork {
    __aicore__ void BeforePublish(uint32_t) {}
    __aicore__ void operator()(uint32_t) {}
};

template <uint32_t Capacity, uint32_t Qps, class Resource, class Trace, class PostSubmit = NoDispatchOverlapWork>
__aicore__ inline void DispatchRemoteImpl(
    Resource& resource, GM_ADDR x, GM_ADDR scales, GM_ADDR plan, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const DispatchTilingData& td, uint32_t core, uint32_t cores, Trace& trace, PostSubmit post_submit = PostSubmit())
{
    constexpr uint32_t sender_bytes = kHeadersOffset + Capacity * Qps * (2U * kWqeBytes + kOutputRecordBytes) + 32;
    static_assert(sender_bytes + Capacity * kTileTokens * sizeof(int32_t) <= platform::kKernelUbBytes);
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    auto lists = resource.template GetBufferByByte<int32_t>(sender_bytes);
    DispatchSenderT<Capacity, Qps, Trace> sender(td, core, wq, workspace, x, scales);
    sender.SetTrace(&trace);
    AscendC::GlobalTensor<int32_t> plan_gm;
    plan_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
    AscendC::GlobalTensor<int32_t> rank_gm, channel_gm;
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    channel_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix));
    const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1) / td.num_channels;
    const bool cached_channel = td.cached_mode && !td.has_x_scales && td.cached_channel && td.x_element_bytes == 2;
    trace.Set(DispatchDeviceTrace::PayloadPeers, sender.Count());
    trace.Set(DispatchDeviceTrace::WholeChannel, cached_channel);
    const uint32_t max_tile = cached_channel ? channel_tokens :
        (65536U / td.world_size < kTileTokens ? 65536U / td.world_size : kTileTokens);
    const uint32_t plan_stride = (channel_tokens + 7U) / 8U * 8U;
    const uint32_t scale_bytes = td.num_scales * td.scale_element_bytes;
    const uint32_t fields = td.has_x_scales && scale_bytes && !td.packed_peer_scales ? 2U : 1U;
    // Lanes own disjoint existing notify channels; destination row order is unchanged.
    for (uint32_t ch = core % td.udma_workers_per_peer; sender.Count() && ch < td.num_channels; ch += td.udma_workers_per_peer) {
        const uint32_t begin = ch * channel_tokens < td.num_tokens ? ch * channel_tokens : td.num_tokens;
        const uint32_t end = begin + channel_tokens < td.num_tokens ? begin + channel_tokens : td.num_tokens;
        uint32_t row[Capacity]{}, consumed[Capacity]{}, total[Capacity]{};
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            const uint32_t peer = sender.Rank(p);
            const uint32_t previous = ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0;
            total[p] = channel_gm.GetValue(peer * td.num_channels + ch) - previous;
            row[p] = (td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0) +
                     (ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0);
        }
        for (uint32_t start = begin; start < end; start += max_tile) {
            const uint32_t size = end - start < max_tile ? end - start : max_tile;
            uint32_t counts[Capacity]{}, cursor[Capacity]{}, field[Capacity]{}, remaining = 0;
            if (cached_channel) {
                // The handle already contains compact lists and the exact per-peer
                // count for this channel. PutRows reads token indices from that list.
                for (uint32_t p = 0; p < sender.Count(); ++p) {
                    counts[p] = consumed[p] = total[p];
                    remaining += total[p];
                }
            } else {
                const uint64_t list_started = trace.Now();
                trace.Count(DispatchDeviceTrace::ListWindows);
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
                trace.Add(DispatchDeviceTrace::ListCycles, list_started);
            }
            while (remaining) {
                for (uint32_t p = 0; p < sender.Count(); ++p) {
                    uint32_t visit = td.udma_wqes_per_visit;
                    if (cursor[p] < counts[p]) sender.BeginVisit(p);
                    while (visit && cursor[p] < counts[p]) {
                        const uint64_t base = (uint64_t(sender.Rank(p)) * td.num_channels + ch) * plan_stride;
                        const uint64_t index = base + consumed[p] - counts[p] + cursor[p];
                        const uint32_t available = (counts[p] - cursor[p]) * fields - field[p];
                        trace.Count(DispatchDeviceTrace::PutRowsCalls);
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
    if (td.packed_peer_scales) {
        // Both builders reuse the same UB: drain payload requests first.
        sender.Flush();
        if (td.packed_scale_ready && sender.Count())
            WaitPackedScaleReady(td, core, cores);
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            const uint32_t peer = sender.Rank(p);
            const uint32_t base = td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0;
            for (uint32_t ch = core % td.udma_workers_per_peer; ch < td.num_channels;
                 ch += td.udma_workers_per_peer) {
                const uint32_t first = ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0;
                const uint32_t last = channel_gm.GetValue(peer * td.num_channels + ch);
                sender.PutPackedScales(p, first, last - first, base);
            }
        }
    }
    if (td.channel_metadata) {
        // Start payload transfer before preparing metadata. Every peer/channel
        // keeps its original sender owner; packing only touches private UB and
        // that owner's immutable GM staging. No extra collective is required.
        sender.Flush();
        const uint64_t pack_started = trace.Now();
        ascend_deepep::PackRemoteChannelMetadata<Capacity>(resource, sender, plan, channel_prefix,
            td, core, DispatchLiveSendUbBytes(Capacity, Qps));
        trace.Add(Trace::TopkCycles, pack_started);
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            const uint32_t peer = sender.Rank(p);
            const uint32_t rank_base = td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0;
            for (uint32_t ch = core % td.udma_workers_per_peer; ch < td.num_channels; ch += td.udma_workers_per_peer) {
                const uint32_t first = ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0;
                const uint32_t count = channel_gm.GetValue(peer * td.num_channels + ch) - first;
                if (!count) continue;
                const uint64_t slot = (uint64_t(peer) * td.num_channels + ch) * plan_stride;
                const uint64_t row = uint64_t(rank_base) + first;
                if (!td.cached_mode) {
                    const uint64_t source = reinterpret_cast<uint64_t>(plan) + slot * sizeof(int32_t);
                    if (td.metadata_direct_output) sender.PutChannelOutput(p, kOutputSrcWord, row, source, count);
                    else sender.PutChannelField(p, workspace, td.src_idx_buffer_offset + row * sizeof(int32_t),
                        source, count * uint32_t(sizeof(int32_t)));
                }
                sender.PutChannelField(p, workspace, td.topk_idx_buffer_offset + row * td.num_topk * sizeof(topk_idx_t),
                    td.packed_channel_indices + slot * td.num_topk * sizeof(topk_idx_t),
                    count * td.num_topk * uint32_t(sizeof(topk_idx_t)));
                if (td.has_topk_weights) {
                    const uint64_t source = td.packed_channel_weights + slot * td.num_topk * sizeof(float);
                    if (td.metadata_direct_output) sender.PutChannelOutput(p, kOutputWeightWord, row, source, count);
                    else sender.PutChannelField(p, workspace, td.topk_weights_buffer_offset + row * td.num_topk * sizeof(float),
                        source, count * td.num_topk * uint32_t(sizeof(float)));
                }
            }
        }
    }
    // Submit the short tail before same-core SIMT metadata. Neither this callback
    // nor the local work below may overwrite live sender headers/descriptors.
    sender.Flush();
    trace.Mark(DispatchDeviceTrace::PayloadQueued);
    constexpr uint32_t self_offset = DispatchLiveSendUbBytes(Capacity, Qps);
    static_assert(self_offset == sender_bytes + Capacity * kTileTokens * sizeof(int32_t));
    post_submit.BeforePublish(self_offset);
    sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core % td.udma_workers_per_peer);
    trace.Mark(DispatchDeviceTrace::Published);
    post_submit(self_offset);
    trace.Mark(DispatchDeviceTrace::IncomingBegin);
    WaitIncoming<true>(workspace, td, core, cores, td.udma_workers_per_peer * Qps);
    trace.Mark(DispatchDeviceTrace::IncomingDone);
    sender.Quiet();
    trace.Mark(DispatchDeviceTrace::QuietDone);
    // Default SO markers cover payload only; channel_metadata also covers the
    // UDMA metadata above. Keep the collective for default SIMT metadata and
    // for self-copy on every path before outputs or workspace are consumed.
    aclshmemx_sync_vec_all();
    trace.Mark(DispatchDeviceTrace::SyncDone);
}
template <class Resource, class Trace, class PostSubmit = NoDispatchOverlapWork>
__aicore__ inline void DispatchRemote(
    Resource& r, GM_ADDR x, GM_ADDR scales, GM_ADDR plan, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const DispatchTilingData& td, uint32_t core, uint32_t cores, Trace& trace, PostSubmit post_submit = PostSubmit())
{
    if (td.udma_qps_per_worker == 2) {
        if (td.udma_peer_order == kGroupRoundRobinOrder)
            DispatchRemoteImpl<kGroupPeerCapacity, 2>(r, x, scales, plan, rp, cp, workspace, td, core, cores, trace, post_submit);
        else
            DispatchRemoteImpl<kMaxPeers, 2>(r, x, scales, plan, rp, cp, workspace, td, core, cores, trace, post_submit);
    } else if (td.udma_peer_order == kGroupRoundRobinOrder)
        DispatchRemoteImpl<kGroupPeerCapacity, 1>(r, x, scales, plan, rp, cp, workspace, td, core, cores, trace, post_submit);
    else
        DispatchRemoteImpl<kMaxPeers, 1>(r, x, scales, plan, rp, cp, workspace, td, core, cores, trace, post_submit);
}
#endif
} // namespace ascend_deepep::udma
