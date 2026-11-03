#pragma once

#include "kernels/combine/udma_combine_sender.h"
#include "common/types.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Keep the payload row residue unchanged, but pack only this worker's indices.
// The source is immutable for the call; count is bounded by the peer's end row.
__simt_vf__ __launch_bounds__(32) inline void GatherTokenIndices(
    __ubuf__ int32_t* output, const __gm__ int32_t* input, uint32_t row, uint32_t step, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += 32U)
        output[i] = input[uint64_t(row) + uint64_t(i) * step];
}

template <uint32_t Phase = 0, class Resource>
__aicore__ inline void CombineTokenRemote(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core)
{
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    auto indices = resource.template GetBufferByByte<int32_t>(kCombineSenderBytes);
    const uint32_t workers = TokenWorkers(td.num_channels);
    if (core < workers)
        for (uint32_t first = 0; first < td.world_size - 1; first += kMaxPeers) {
            CombineSender sender(
                td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()), workspace, x, weights, first, core,
                workers, 1U);
            AscendC::GlobalTensor<int32_t> rank_gm;
            rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
            // Strided rows split each peer across independent QP owners; stagger the
            // starting row and peer order to avoid all workers targeting one peer first.
            (void)channel_prefix;
            const uint32_t fields = td.has_topk_weights ? 2U : 1U;
            uint32_t row[kMaxPeers]{}, end[kMaxPeers]{}, loaded[kMaxPeers]{}, cursor[kMaxPeers]{}, field[kMaxPeers]{};
            uint32_t remaining = 0;
            for (uint32_t p = 0; p < sender.Count(); ++p) {
                const uint32_t peer = sender.Rank(p);
                row[p] = peer ? rank_gm.GetValue((peer - 1) * td.world_size + td.rank) : 0;
                end[p] = rank_gm.GetValue(peer * td.world_size + td.rank);
                row[p] += (core + peer * ((workers + td.world_size - 1) / td.world_size)) % workers;
                if (row[p] < end[p])
                    remaining += (end[p] - row[p] + workers - 1) / workers;
            }
            while (remaining) {
                for (uint32_t p = 0; p < sender.Count(); ++p) {
                    uint32_t visit = td.udma_wqes_per_visit;
                    while (visit && row[p] < end[p]) {
                        if (cursor[p] >= loaded[p]) {
                            // Pending spans can still read this window in SIMT.
                            sender.Flush();
                            const uint32_t rows_left = (end[p] - row[p] + workers - 1) / workers;
                            const uint32_t count = rows_left < kTileTokens ? rows_left : kTileTokens;
                            // Flush has completed the previous WQE VF's index reads.
                            // V_S then S_V orders this gather before any later reader.
                            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                            asc_vf_call<GatherTokenIndices>(
                                dim3(32), reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()) + p * kTileTokens,
                                reinterpret_cast<const __gm__ int32_t*>(src_idx), row[p], workers, count);
                            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                            cursor[p] = 0;
                            loaded[p] = count;
                        }
                        const uint32_t available = (loaded[p] - cursor[p]) * fields - field[p];
                        auto* window =
                            reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()) + p * kTileTokens + cursor[p];
                        const uint32_t count = sender.template PutSelectedRows<Phase>(
                            p, window, row[p], field[p], available < visit ? available : visit);
                        const uint32_t rows = (field[p] + count) / fields;
                        field[p] = (field[p] + count) % fields;
                        cursor[p] += rows;
                        row[p] += rows * workers;
                        remaining -= rows;
                        visit -= count;
                    }
                }
            }
            if constexpr (Phase == 1) {
                sender.Flush();
            } else {
                sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core);
                sender.Quiet();
            }
        }
    if constexpr (Phase != 1) {
        WaitIncoming<false>(
            workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, td.num_channels, workers);
    }
    // WaitIncoming's all-core barrier includes drained local self copies.
}
#endif
} // namespace ascend_deepep::udma
