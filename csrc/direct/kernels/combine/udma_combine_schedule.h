#pragma once

#include "kernels/combine/udma_combine_sender.h"
#include "kernels/combine/udma_combine_self_copy.h"
#include "common/platform.h"
#include "common/types.h"
#include "kernels/aggregate/aggregate_pipeline.h"
#include "kernels/combine/udma_combine_weights.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
template <uint32_t Capacity, uint32_t Phase, uint32_t Qps, class Resource, class Trace>
__aicore__ inline void CombineRemoteImpl(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core, Trace& trace)
{
    // Separate VF-produced count from descriptors, indices and pressure scratch.
    constexpr uint32_t sender_base_bytes = kHeadersOffset + Capacity * Qps * (2U * kWqeBytes + sizeof(uint64_t));
    const uint32_t sender_bytes = sender_base_bytes + (td.combine_payload_max_rows > 1U ? 32U : 0U);
    static_assert(sender_base_bytes + 32U + Capacity * kTileTokens * sizeof(int32_t) <= platform::kKernelUbBytes);
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    auto indices = resource.template GetBufferByByte<int32_t>(sender_bytes);
    const bool compact_weights = Phase == 0 && td.combine_compact_weights;
    const bool simt_weights = !compact_weights && Phase == 0 && CombineSimtWeightsEnabled(td);
    if (simt_weights) {
        const uint64_t started = trace.Now();
        CombineRemoteWeights(weights, src_idx, rank_prefix, workspace, td, core);
        trace.Add(Trace::TopkCycles, started);
    }
    CombineSenderT<Capacity, Qps, Trace> sender(td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()), workspace,
        x, weights, UINT32_MAX, 0, 1, 0, !simt_weights && !compact_weights);
    sender.SetTrace(&trace);
    trace.Set(Trace::PayloadPeers, sender.Count());
    AscendC::GlobalTensor<int32_t> src_gm, rank_gm;
    src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    // Dispatch stores all channels of one source rank contiguously. Ownership
    // here is per peer, so channel boundaries need not split the return stream.
    const uint32_t fields = td.has_topk_weights && !simt_weights && !compact_weights ? 2U : 1U;
    uint32_t row[Capacity]{}, end[Capacity]{}, loaded[Capacity]{}, cursor[Capacity]{}, field[Capacity]{};
    uint32_t peer_begin[Capacity]{};
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
        peer_begin[p] = begin;
        if (td.combine_payload_max_rows > 1U &&
            (end[p] < begin || length > td.token_capacity || end[p] > td.num_recv_tokens)) {
            trap(); return;
        }
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
                if constexpr (Phase == 0) {
                    if (td.combine_payload_max_rows > 1U) {
                        // The whole peer input is already continuous. Preserve
                        // logical-row visits and worker slices, bypass index DMA.
                        const uint32_t available = end[p] - row[p];
                        trace.Count(Trace::PutRowsCalls);
                        const uint32_t count = sender.PutRows(
                            p, nullptr, row[p], 0U, available < visit ? available : visit, peer_begin[p]);
                        row[p] += count;
                        remaining -= count;
                        visit -= count;
                        continue;
                    }
                }
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
                    sender.FlushForIndex();
                    const uint64_t loaded_at = trace.Now();
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
                    if constexpr (Trace::enabled) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                    }
                    trace.Add(Trace::ListCycles, loaded_at);
                    trace.Count(Trace::ListWindows);
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
                trace.Count(Trace::PutRowsCalls);
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
        if (compact_weights) {
            // All payload is submitted first. Each worker owns a disjoint
            // contiguous slice of the peer's dispatch-ordered input rows.
            sender.Flush();
            for (uint32_t p = 0; p < sender.Count(); ++p) {
                const uint32_t peer = sender.Rank(p);
                const uint32_t base = peer ? rank_gm.GetValue((peer - 1) * td.world_size + td.rank) : 0;
                const uint32_t length = rank_gm.GetValue(peer * td.world_size + td.rank) - base;
                const uint32_t lane = core % td.udma_workers_per_peer;
                const uint32_t begin = base + uint64_t(length) * lane / td.udma_workers_per_peer;
                const uint32_t finish = base + uint64_t(length) * (lane + 1U) / td.udma_workers_per_peer;
                if (begin < finish) {
                    sender.BeginVisit(p);
                    sender.PutCompactWeights(p, begin, base, finish - begin);
                }
            }
        }
        trace.Mark(Trace::PayloadQueued);
        sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core % td.udma_workers_per_peer);
        trace.Mark(Trace::Published);
        if constexpr (Phase == 0 || Phase == 2) {
            const uint64_t self_started = trace.Now();
            // SIMT weight AIVs finish CombineRemoteWeights (V_S) above before
            // copying their self channels. Payload AIVs have posted all
            // UDMA requests, including weights when no spare AIV exists.
            // Self writes use this rank's capacity-sized partition (with
            // compact row indices when compact_self_rows is enabled);
            // remote contributors use their own rank slots, so neither
            // payload nor weight self stores collide with incoming writes.
            // The SQ now owns the WQEs. Finish MTE3's reads of the reused
            // UB before self-copy loads use it, while UDMA reads GM payload.
            if (!td.combine_direct_self) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                CombineSelfCopy(resource, x, weights, src_idx, rank_prefix, channel_prefix, workspace, td, core);
            }
            trace.SelfDone(self_started);
        }
        trace.Mark(Trace::IncomingBegin);
        WaitIncoming<false>(workspace, td, core, td.block_dim, td.udma_workers_per_peer * Qps);
        trace.Mark(Trace::IncomingDone);
        sender.Quiet();
        trace.Mark(Trace::QuietDone);
        // UDMA SO markers do not cover SIMT stores issued by other AIVs.
        // Match Dispatch's full-rank completion before Reduce/workspace reuse.
        if (simt_weights)
            aclshmemx_sync_vec_all();
        else
            AscendC::SyncAll<true>();
        trace.Mark(Trace::SyncDone);
    }
}
template <uint32_t Phase = 0, class Resource, class Trace>
__aicore__ inline void CombineRemote(
    Resource& r, GM_ADDR x, GM_ADDR weights, GM_ADDR idx, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core, Trace& trace)
{
    if (td.udma_qps_per_worker == 2) {
        if (td.udma_peer_order == kGroupRoundRobinOrder)
            CombineRemoteImpl<kGroupPeerCapacity, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
        else if (td.udma_peers_per_worker > kMaxPeers)
            CombineRemoteImpl<kMaxRankPeers, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
        else
            CombineRemoteImpl<kMaxPeers, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
    } else if (td.udma_peer_order == kGroupRoundRobinOrder)
        CombineRemoteImpl<kGroupPeerCapacity, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
    else if (td.udma_peers_per_worker > kMaxPeers)
        CombineRemoteImpl<kMaxRankPeers, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
    else
        CombineRemoteImpl<kMaxPeers, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
}
template <uint32_t Phase = 0, class Resource>
__aicore__ inline void CombineRemote(
    Resource& r, GM_ADDR x, GM_ADDR weights, GM_ADDR idx, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core)
{
    DisabledDispatchDeviceTrace trace(0);
    CombineRemote<Phase>(r, x, weights, idx, rp, cp, workspace, td, core, trace);
}
#endif
} // namespace ascend_deepep::udma
