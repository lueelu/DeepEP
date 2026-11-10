#pragma once

#include "kernels/combine/udma_combine_sender.h"
#include "common/types.h"
#include "kernels/aggregate/aggregate_pipeline.h"
#include "kernels/combine/udma_combine_weights.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
template <uint32_t Capacity, uint32_t Phase, uint32_t Qps, class Resource>
__aicore__ inline void CombineRemoteImpl(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core)
{
    constexpr uint32_t sender_bytes = kHeadersOffset + Capacity * Qps * (2U * kWqeBytes + sizeof(uint64_t));
    static_assert(sender_bytes + Capacity * kTileTokens * sizeof(int32_t) <= 216 * 1024);
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    auto indices = resource.template GetBufferByByte<int32_t>(sender_bytes);
    const bool simt_weights = Phase == 0 && CombineSimtWeightsEnabled(td);
    if (simt_weights)
        CombineRemoteWeights(weights, src_idx, rank_prefix, workspace, td, core);
    CombineSenderT<Capacity, Qps> sender(td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()), workspace,
        x, weights, UINT32_MAX, 0, 1, 0, !simt_weights);
    AscendC::GlobalTensor<int32_t> src_gm, rank_gm;
    src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    // Dispatch stores all channels of one source rank contiguously. Ownership
    // here is per peer, so channel boundaries need not split the return stream.
    (void)channel_prefix;
    const uint32_t fields = td.has_topk_weights && !simt_weights ? 2U : 1U;
    uint32_t row[Capacity]{}, end[Capacity]{}, loaded[Capacity]{}, cursor[Capacity]{}, field[Capacity]{};
    uint32_t ready_chunk[Capacity];
    if constexpr (Phase == 3) {
        for (uint32_t p = 0; p < Capacity; ++p) ready_chunk[p] = UINT32_MAX;
    }
    uint32_t remaining = 0;
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        const uint32_t peer = sender.Rank(p);
        row[p] = peer ? rank_gm.GetValue((peer - 1) * td.world_size + td.rank) : 0;
        end[p] = rank_gm.GetValue(peer * td.world_size + td.rank);
        // Split the original peer row interval, retaining src_idx and output addresses.
        const uint32_t begin = row[p], length = end[p] - begin;
        const uint32_t lane = core % td.udma_workers_per_peer;
        row[p] = begin + uint64_t(length) * lane / td.udma_workers_per_peer;
        end[p] = begin + uint64_t(length) * (lane + 1U) / td.udma_workers_per_peer;
        remaining += end[p] - row[p];
    }
    while (remaining) {
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            uint32_t visit = td.udma_wqes_per_visit;
            if (row[p] < end[p]) sender.BeginVisit(p);
            while (visit && row[p] < end[p]) {
                if constexpr (Phase == 3) {
                    const uint32_t chunk = row[p] / td.pipeline_chunk_tokens;
                    if (ready_chunk[p] != chunk) {
                        sender.Flush(); // Publish earlier rows before waiting; never quiet per chunk.
                        WaitAggregateReady(td, core, row[p]);
                        ready_chunk[p] = chunk;
                    }
                }
                if (cursor[p] == loaded[p]) {
                    // Pending spans can still read this window in SIMT.
                    sender.Flush();
                    const uint32_t aligned = row[p] / 8 * 8;
                    const uint32_t leading = row[p] - aligned;
                    const uint32_t count =
                        end[p] - row[p] < kTileTokens - leading ? end[p] - row[p] : kTileTokens - leading;
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::DataCopyExtParams copy{1, (count + leading) * uint32_t(sizeof(int32_t)), 0, 0, 0};
                    AscendC::DataCopyPad(
                        indices[p * kTileTokens], src_gm[aligned], copy,
                        AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                    cursor[p] = leading;
                    loaded[p] = leading + count;
                }
                uint32_t available = (loaded[p] - cursor[p]) * fields - field[p];
                if constexpr (Phase == 3) {
                    const uint32_t in_chunk = td.pipeline_chunk_tokens - row[p] % td.pipeline_chunk_tokens;
                    const uint32_t ready_fields = in_chunk * fields - field[p];
                    available = available < ready_fields ? available : ready_fields;
                }
                auto* window = reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()) + p * kTileTokens + cursor[p];
                const uint32_t count =
                    sender.template PutSelectedRows<Phase == 3 ? 0 : Phase>(
                        p, window, row[p], field[p], available < visit ? available : visit);
                const uint32_t rows = (field[p] + count) / fields;
                field[p] = (field[p] + count) % fields;
                cursor[p] += rows;
                row[p] += rows;
                remaining -= rows;
                visit -= count;
            }
        }
    }
    if constexpr (Phase == 1) {
        sender.Flush(); // Drain WQE staging, not the asynchronous payload reads.
    } else {
        sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core % td.udma_workers_per_peer);
        WaitIncoming<false>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, td.num_channels, td.udma_workers_per_peer * Qps);
        sender.Quiet();
        // UDMA SO markers do not cover SIMT stores issued by other AIVs.
        // Match Dispatch's full-rank completion before Reduce/workspace reuse.
        if (simt_weights)
            aclshmemi_sync_npu_v4<true>(0);
        else
            AscendC::SyncAll<true>();
    }
}
template <uint32_t Phase = 0, class Resource>
__aicore__ inline void CombineRemote(
    Resource& r, GM_ADDR x, GM_ADDR weights, GM_ADDR idx, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core)
{
    if (td.udma_qps_per_worker == 2) {
        if (td.udma_peer_order == kGroupRoundRobinOrder)
            CombineRemoteImpl<kGroupPeerCapacity, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core);
        else
            CombineRemoteImpl<kMaxPeers, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core);
    } else if (td.udma_peer_order == kGroupRoundRobinOrder)
        CombineRemoteImpl<kGroupPeerCapacity, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core);
    else
        CombineRemoteImpl<kMaxPeers, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core);
}
#endif
} // namespace ascend_deepep::udma
