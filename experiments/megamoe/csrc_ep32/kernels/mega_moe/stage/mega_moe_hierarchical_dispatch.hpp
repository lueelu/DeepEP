// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_HIERARCHICAL_DISPATCH_H
#define MEGA_MOE_HIERARCHICAL_DISPATCH_H

#include "../common/mega_moe_roce.hpp"
#include "mega_moe_send_mask.hpp"

namespace MegaMoeImpl {

__aicore__ inline __ubuf__ uint8_t* HierarchicalRoceScratch(const Params& params)
{
    return reinterpret_cast<__ubuf__ uint8_t*>(static_cast<uint64_t>(params.tilingData->roceScratchUbOffset));
}

__aicore__ inline void PublishHierarchicalLocalSignal(const Params& params, HierarchicalSignal signal)
{
    WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, signal),
                        static_cast<int32_t>(params.tilingData->hierarchicalSequence));
}

__aicore__ inline void WaitHierarchicalLocalSignal(const Params& params, HierarchicalSignal signal, uint32_t peer = 0U,
                                                   __ubuf__ uint8_t* scratch = nullptr)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceWaitFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal), params.tilingData->hierarchicalSequence,
                       peer, scratch, RoceWaitKind::LocalReadiness);
#else
    (void)peer;
    (void)scratch;
    WaitUntilGmFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal),
                          static_cast<int32_t>(params.tilingData->hierarchicalSequence));
#endif
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
}

// The packer owns this core's reserved UB. Weight storage precedes the active
// route packets, so one metadata Put and its original tail marker cover both.
__aicore__ inline void PackHierarchicalCombineWeights(const MoeStageCommonConfig& common, const Params& params)
{
    if (!UseHierarchicalCombineReduction(params.tilingData)) {
        return;
    }
    GlobalTensor<float> source, destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.probsGmAddr));
    const uint64_t slot = HierarchicalHost(common.rankId) * CalcRailTopkIdsSlotBytes(params.tilingData);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.peermemInfo.railTopkIdsRecvPtr + slot -
                                                                HierarchicalMetadataPrefixBytes(params.tilingData)));
    constexpr uint32_t batch = HIERARCHICAL_ROCE_UB_BYTES / sizeof(float);
    LocalTensor<float> weights(TPosition::VECCALC, params.tilingData->roceScratchUbOffset, batch);
    const uint32_t total = common.tokenNum * common.topK;
    for (uint32_t offset = 0U; offset < total; offset += batch) {
        const uint32_t count = total - offset < batch ? total - offset : batch;
        DataCopyPad(weights, source[offset], {1U, count * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_MTE3, MEGAMOE_ROCE_EVENT_ID>();
        DataCopyPad(destination[offset], weights, {1U, count * 4U, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE3_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    }
    SyncFuncStatic<HardEvent::MTE3_S, MEGAMOE_ROCE_EVENT_ID>();
}

// Local packing preserves the existing rail format. RDMA reads its stable GM
// copy, never the packer's reusable UB. Payload rows retain their original token
// indices at the gateway; route metadata determines which rows consumers read.
__aicore__ inline void PostHierarchicalDispatch(const MoeStageCommonConfig& common, const Params& params,
                                                const SendMaskConfig& config, AscendTimerDevice* timer = nullptr)
{
    const uint32_t myHost = HierarchicalHost(common.rankId);
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    auto* roceScratch = HierarchicalRoceScratch(params);
    // Put metadata first on the single owned QP. Its last packet carries a
    // full sequence: HNS can release Relay while the following payload is
    // still in flight, without a separate metadata commit WQE.
    WaitHierarchicalLocalSignal(params, HierarchicalSignal::LocalRailPacked, gateway, roceScratch);
    const auto prefixBytes = HierarchicalMetadataPrefixBytes(params.tilingData);
    GM_ADDR packedRail = params.peermemInfo.railTopkIdsRecvPtr + myHost * config.railTopkIdsSlotBytes - prefixBytes;
    const uint32_t railBytes =
        static_cast<uint32_t>(prefixBytes + CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK));
    RocePutBytes(packedRail, packedRail, railBytes, gateway, roceScratch, timer, AscendTimer::RdmaPhase::DispatchMeta);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // XSCALE has no established packet/SO contract. Preserve its explicit
    // quiet/commit/quiet protocol before posting the independent payload.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        roceScratch, timer, AscendTimer::RdmaPhase::DispatchMeta);
#endif
}

__aicore__ inline __gm__ int32_t* HierarchicalPayloadCommit(const Params& params, uint32_t lane)
{
    return reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.hierarchicalControlPtr +
                                             HierarchicalSignalOffset(HierarchicalSignal::NetPayloadCommit) +
                                             lane * HIERARCHICAL_FLAG_BYTES);
}

__aicore__ inline void PostHierarchicalPayloadLane(const MoeStageCommonConfig& common, const Params& params,
                                                   uint32_t lane, AscendTimerDevice* timer)
{
    const uint32_t begin = HierarchicalPayloadTokenBegin(common.tokenNum, lane);
    const uint32_t end = HierarchicalPayloadTokenBegin(common.tokenNum, lane + 1U);
    const uint32_t rowBytes =
        DispatchPacketCount(CalcQuantTokenScaleBytes(common.tokenHiddenDim, 1U, common.topK, false)) *
        RAIL_PACKET_BYTES;
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    auto* scratch = HierarchicalRoceScratch(params);
    RocePutBytes(params.peermemInfo.relayPayloadPtr + static_cast<uint64_t>(begin) * rowBytes,
                 params.peermemInfo.dispatchPacketPtr + static_cast<uint64_t>(begin) * rowBytes,
                 (end - begin) * rowBytes, gateway, scratch, timer, AscendTimer::RdmaPhase::DispatchPayload, false,
                 lane + 1U, begin, end - begin);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // Empty lanes still commit. Only this QP can certify its own preceding Put.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalPayloadCommit(params, lane)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        scratch, timer, AscendTimer::RdmaPhase::DispatchPayload, lane + 1U);
#endif
}

// Slots are lane-major within each source host. The four same-lane mesh
// records form a contiguous stripe for the second exchange.
// The per-core stripe and Dispatch probe staging have disjoint lifetimes.
// Keep the pre-zeroed reset buffer and rail count accumulators intact.
__aicore__ inline void BroadcastRouteBlock(GM_ADDR* peers, uint64_t offset, GM_ADDR source, uint32_t bytes,
                                           uint32_t first, uint32_t stride, uint32_t skip, const Params& params)
{
    GlobalTensor<uint8_t> src, dst;
    src.SetGlobalBuffer(source);
    LocalTensor<uint8_t> block(TPosition::VECCALC, params.tilingData->roceScratchUbOffset + HIERARCHICAL_ROCE_UB_BYTES,
                               bytes);
    DataCopy(block, src[0], bytes);
    SyncFuncStatic<HardEvent::MTE2_MTE3, MEGAMOE_ROCE_EVENT_ID>();
    for (uint32_t i = 0; i < 4U; ++i) {
        if (i == skip) continue;
        dst.SetGlobalBuffer(peers[first + i * stride] + offset);
        DataCopy(dst[0], block, bytes);
    }
    SyncFuncStatic<HardEvent::MTE3_MTE2, MEGAMOE_ROCE_EVENT_ID>();
}

__aicore__ inline void BuildCompactPlan(const MoeStageCommonConfig& common, const Params& params, GM_ADDR source,
                                        LocalTensor<int32_t>* compactIds = nullptr)
{
    if (compactIds != nullptr) {
        StoreCompactPlan(common, params, *compactIds);
        return;
    }
    auto* ids = reinterpret_cast<__gm__ int32_t*>(source);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.compactPtr);
    uint32_t count = 0;
    for (uint32_t t = 0; t < common.tokenNum; ++t) {
        uint32_t mask = 0;
        for (uint32_t slot = 0; slot < common.topK; ++slot) {
            const uint32_t i = t * common.topK + slot;
            const int32_t id = ReadGmByPassDCache(ids + i + i / RAIL_PACKET_DATA_WORDS);
            if (id >= 0 &&
                static_cast<uint32_t>(id) / (16U * common.moeExpertPerRank) == HierarchicalHost(common.rankId))
                mask |= 1U << slot;
        }
        WriteGmByPassDCache(plan + 1 + t, static_cast<int32_t>(mask));
        if (mask) WriteGmByPassDCache(plan + 1 + common.tokenNum + count++, static_cast<int32_t>(t));
    }
    WriteGmByPassDCache(plan, static_cast<int32_t>(count));
}

__aicore__ inline void GatherRouteStripes(const AivJobContext& job, const MoeStageCommonConfig& common,
                                          const Params& params, GM_ADDR* peers, const SendMaskConfig& config,
                                          SendMaskScratch& scratch, AscendTimerDevice* timer)
{
    const uint32_t host = HierarchicalHost(common.rankId);
    const uint32_t local = common.rankId % 16U, mesh = local / 4U, lane = local % 4U;
    const uint32_t bytes = CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK);
    const uint64_t offset = params.peermemInfo.gatherPtr - params.peermemInfo.rankSyncInWorldPtr;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t sourceHost = pass ^ host;
        auto* ids = params.peermemInfo.railTopkIdsRecvPtr + sourceHost * config.railTopkIdsSlotBytes;
        if (pass && params.tilingData->bf16Bs96) {
            // Four independent workers each broadcast the gateway record to one local mesh.
            if (job.jobIndex >= 16U && job.jobIndex < 20U) {
                const uint32_t targetMesh = job.jobIndex - 16U;
                auto* marker = reinterpret_cast<__gm__ int32_t*>(ids + bytes) - 1;
                RoceWaitCommit(marker, params.tilingData->hierarchicalSequence, timer,
                               AscendTimer::RdmaPhase::DispatchMeta, HierarchicalGateway(common.rankId));
                AcquireRoutePackets(ids, bytes / RAIL_PACKET_BYTES, params.tilingData->hierarchicalSequence,
                                    scratch.topkIndexTensor);
                const uint32_t index = RouteGatherSlot(sourceHost * 16U + local);
                BroadcastRouteBlock(peers, offset + index * bytes, ids, bytes, host * 16U + targetMesh * 4U,
                                    1U, 4U, params);
            }
            continue;
        }
        if (job.jobIndex == 0U) {
            if (pass) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
                auto* marker = reinterpret_cast<__gm__ int32_t*>(ids + bytes) - 1;
#else
                auto* marker = HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit);
#endif
                RoceWaitCommit(marker, params.tilingData->hierarchicalSequence, timer,
                               AscendTimer::RdmaPhase::DispatchMeta, HierarchicalGateway(common.rankId));
            }

            const uint32_t index = RouteGatherSlot(sourceHost * HIERARCHICAL_RANKS_PER_HOST + local);
#ifdef DEEPEP_MEGAMOE_TIMER
            AscendTimerRelayScope exchange(timer, AscendTimer::RelayStep::Clos, sourceHost, 0U, bytes);
#endif
            BroadcastRouteBlock(peers, offset + index * bytes, ids, bytes, host * 16U + lane, 4U, 4U, params);
        } else if (job.jobIndex == 4U) {
            const uint32_t first = sourceHost * 16U + lane * 4U;
#ifdef DEEPEP_MEGAMOE_TIMER
            AscendTimerRelayScope exchange(timer, AscendTimer::RelayStep::Mesh, sourceHost, 0U, 4U * bytes);
#endif
            AcquireRoutePackets(params.peermemInfo.gatherPtr + first * bytes, 4U * bytes / RAIL_PACKET_BYTES,
                                params.tilingData->hierarchicalSequence, scratch.topkIndexTensor);
            BroadcastRouteBlock(peers, offset + first * bytes, params.peermemInfo.gatherPtr + first * bytes, 4U * bytes,
                                host * 16U + mesh * 4U, 1U, lane, params);
        }
    }
    for (uint32_t rank = job.jobIndex; rank < common.worldSize; rank += job.totalJobs) {
        const uint32_t index = RouteGatherSlot(rank);
        auto* record = params.peermemInfo.gatherPtr + index * bytes;
        {
#ifdef DEEPEP_MEGAMOE_TIMER
            AscendTimerRelayScope wait(timer, AscendTimer::RelayStep::SourceWait, rank, 0U, bytes);
#endif
            AcquireRoutePackets(record, bytes / RAIL_PACKET_BYTES, params.tilingData->hierarchicalSequence,
                                scratch.topkIndexTensor);
        }
        bool compactIds = false;
        {
#ifdef DEEPEP_MEGAMOE_TIMER
            AscendTimerRelayScope route(timer, AscendTimer::RelayStep::LocalRoute, rank, 0U, common.tokenNum);
#endif
            RelayRailTopkIdsToExperts(common, peers, config, params, scratch, rank / 16U, job.jobIndex,
                                      common.rankId * common.moeExpertPerRank, common.moeExpertPerRank, timer, record,
                                      rank, &compactIds);
        }
        if (rank == HierarchicalGateway(common.rankId)) {
#ifdef DEEPEP_MEGAMOE_TIMER
            AscendTimerRelayScope plan(timer, AscendTimer::RelayStep::CompactPlan, rank, 0U, common.tokenNum);
#endif
            BuildCompactPlan(common, params, record, compactIds ? &scratch.topkIdsTensor : nullptr);
        }
    }
}

__aicore__ inline void GatherAndPostHierarchicalDispatch(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                         const Params& params, GM_ADDR* winRankAddr,
                                                         const SendMaskConfig& config, SendMaskScratch& scratch,
                                                         AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.jobIndex >= job.totalJobs) {
        return;
    }
    const uint32_t myHost = HierarchicalHost(common.rankId);
    const uint32_t otherHost = 1U - myHost;
    const uint32_t lane = HierarchicalDispatchLane(job.jobIndex);
    if (lane < HIERARCHICAL_DISPATCH_QPS) PostHierarchicalPayloadLane(common, params, lane, timer);
    if (UseRouteAllGather(params.tilingData)) {
        GatherRouteStripes(job, common, params, winRankAddr, config, scratch, timer);
        return;
    }
    // Only g1..last AIV0 relay. Each keeps its former group's expert range
    // and handles BOTH source hosts, locally packed metadata first.
    if (job.jobIndex == 0U || (job.jobIndex & 1U) != 0U || job.totalJobs <= 2U) {
        return;
    }
    const uint32_t relayCore = job.jobIndex / 2U - 1U;
    const uint32_t relayCores = job.totalJobs / 2U - 1U;
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    uint32_t firstExpert;
    uint32_t expertCount;
    CalcRailRelayExpertRange(expertsPerHost, relayCore, relayCores, 1U, 0U, firstExpert, expertCount);
    for (uint32_t pass = 0U; pass < 2U; ++pass) {
        const uint32_t sourceHost = pass == 0U ? myHost : otherHost;
        if (pass == 1U && relayCore == 0U) {
            // g1 AIV0 is the receive leader on BOTH hosts. Do useful local
            // Relay before waiting for the independent incoming metadata.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
            const uint64_t markerOffset = otherHost * config.railTopkIdsSlotBytes +
                                          CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK) - sizeof(int32_t);
            auto* marker = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + markerOffset);
#else
            auto* marker = HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit);
#endif
            RoceWaitCommit(marker, params.tilingData->hierarchicalSequence, timer, AscendTimer::RdmaPhase::DispatchMeta,
                           HierarchicalGateway(common.rankId));
            PublishHierarchicalLocalSignal(params, HierarchicalSignal::MetaReady);
            BuildCompactPlan(common, params,
                             params.peermemInfo.railTopkIdsRecvPtr + otherHost * config.railTopkIdsSlotBytes);
        }
        WaitHierarchicalLocalSignal(params,
                                    pass == 0U ? HierarchicalSignal::LocalRailPacked : HierarchicalSignal::MetaReady,
                                    HierarchicalGateway(common.rankId));
        if (expertCount != 0U) {
            RelayRailTopkIdsToExperts(common, winRankAddr, config, params, scratch, sourceHost, job.jobIndex,
                                      static_cast<int32_t>(myHost * expertsPerHost + firstExpert),
                                      static_cast<int32_t>(expertCount), timer);
        }
    }
}

__aicore__ inline uint32_t ProbePayloadLanes(const MoeStageCommonConfig& common, const Params& params, uint32_t ready)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    const uint32_t packets =
        DispatchPacketCount(CalcQuantTokenScaleBytes(common.tokenHiddenDim, 1U, common.topK, false));
    // Between Dispatch slices the helper owns the same aligned tail reservation as the row probe.
    constexpr uint32_t words = 32U / sizeof(uint32_t);
    LocalTensor<uint32_t> snapshot(TPosition::VECCALC,
                                   params.tilingData->roceScratchUbOffset + HIERARCHICAL_ROCE_UB_BYTES,
                                   HIERARCHICAL_DISPATCH_QPS * words);
    uint32_t submitted = 0;
    for (uint32_t lane = 0; lane < HIERARCHICAL_DISPATCH_QPS; ++lane) {
        const uint32_t bit = 1U << lane;
        if (ready & bit) continue;
        const uint32_t begin = HierarchicalPayloadTokenBegin(common.tokenNum, lane);
        const uint32_t end = HierarchicalPayloadTokenBegin(common.tokenNum, lane + 1U);
        if (begin == end) {
            ready |= bit;
            continue;
        }
        GlobalTensor<uint32_t> tail;
        tail.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            params.peermemInfo.relayPayloadPtr + static_cast<uint64_t>(end) * packets * RAIL_PACKET_BYTES - 32U));
        DataCopy(snapshot[lane * words], tail[0], words);
        submitted |= bit;
    }
    if (submitted) {
        SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
        for (uint32_t lane = 0; lane < HIERARCHICAL_DISPATCH_QPS; ++lane)
            if ((submitted & (1U << lane)) &&
                snapshot.GetValue((lane + 1U) * words - 1U) == params.tilingData->hierarchicalSequence)
                ready |= 1U << lane;
        SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    }
#else
    (void)common;
    for (uint32_t lane = 0; lane < HIERARCHICAL_DISPATCH_QPS; ++lane)
        if (!(ready & (1U << lane)) &&
            RoceTestFlagEquals(HierarchicalPayloadCommit(params, lane), params.tilingData->hierarchicalSequence))
            ready |= 1U << lane;
#endif
    return ready;
}

__aicore__ inline void FinishHierarchicalDispatch(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                  const Params& params, AscendTimerDevice* timer = nullptr,
                                                  uint32_t* remembered = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Leave the AIV1 count-table builder and the first Dispatch wave free to run.
    const uint32_t firstHelper = job.totalJobs - 3U;
    if (job.jobIndex != firstHelper && job.jobIndex != firstHelper + 2U) {
        return;
    }
    const uint32_t half = (job.jobIndex - firstHelper) / 2U;
    // Helpers only observe incoming payloads. The sender retains CQ ownership.
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    const bool profile = timer != nullptr && timer->Enabled();
    const int64_t start = profile ? AscendC::GetSystemCycle() : 0;
    constexpr uint32_t fullMask = (1U << HIERARCHICAL_DISPATCH_QPS) - 1U;
    uint32_t readyMask = remembered ? *remembered : 0U;
    if (readyMask == fullMask) return;
    uint32_t initialMissing = 0U;
    uint32_t polls = 0U;
    RoceWaitState state;
    do {
        readyMask = ProbePayloadLanes(common, params, readyMask);
        if (profile) {
            if (polls == 0U) initialMissing = fullMask ^ readyMask;
            ++polls;
        }
        if (remembered) {
            const bool advanced = *remembered != readyMask;
            if (readyMask != fullMask) {
                *remembered = readyMask;
                if (profile && advanced)
                    timer->RecordDetail(AscendTimer::AIV_PAYLOAD_ALL_LANES_WAIT, start, AscendC::GetSystemCycle(),
                                        (static_cast<uint64_t>(initialMissing) << 32) | polls,
                                        params.tilingData->hierarchicalSequence);
                return;
            }
        }
        if (readyMask != fullMask) {
            // Combine may already own these queues after the input barrier.
            RoceCheckWait(state, gateway, nullptr, RoceWaitKind::Commit, fullMask, readyMask);
        }
    } while (readyMask != fullMask);
    AscendTimerFixedScope broadcast(timer, AscendTimer::AIV_GATEWAY_PAYLOAD_BROADCAST);
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    if (profile) {
        timer->RecordDetail(AscendTimer::AIV_PAYLOAD_ALL_LANES_WAIT, start, AscendC::GetSystemCycle(),
                            (static_cast<uint64_t>(initialMissing) << 32) | polls,
                            params.tilingData->hierarchicalSequence);
    }
    const uint32_t firstRank = HierarchicalHost(common.rankId) * HIERARCHICAL_RANKS_PER_HOST;
    const uint32_t slot = common.rankId - firstRank;
    const uint64_t offset = params.peermemInfo.gatewayPayloadReadyPtr - params.peermemInfo.rankSyncInWorldPtr +
                            static_cast<uint64_t>(slot) * HIERARCHICAL_FLAG_BYTES;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (UseRouteAllGather(params.tilingData)) {
        // The eight lane snapshots have retired. The following aligned line belongs to this helper
        // until all eight MTE3 writes retire. Receivers acquire word zero of their gateway slot.
        constexpr uint32_t words = 32U / sizeof(uint32_t);
        LocalTensor<int32_t> notice(
            TPosition::VECCALC,
            params.tilingData->roceScratchUbOffset + HIERARCHICAL_ROCE_UB_BYTES + HIERARCHICAL_DISPATCH_QPS * 32U,
            words);
        for (uint32_t i = 0; i < words; ++i)
            notice.SetValue(i, static_cast<int32_t>(params.tilingData->hierarchicalSequence));
        SyncFuncStatic<HardEvent::S_MTE3, MEGAMOE_ROCE_EVENT_ID>();
        for (uint32_t localRank = half * 8U; localRank < (half + 1U) * 8U; ++localRank) {
            GlobalTensor<int32_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                (localRank == slot ? params.peermemInfo.rankSyncInWorldPtr : g_winRankAddr_[firstRank + localRank]) +
                offset));
            DataCopy(dst[0], notice, words);
        }
        SyncFuncStatic<HardEvent::MTE3_S, MEGAMOE_ROCE_EVENT_ID>();
    } else
#endif
    {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        for (uint32_t localRank = half * 8U; localRank < (half + 1U) * 8U; ++localRank) {
            const uint32_t peer = firstRank + localRank;
            auto* ready = reinterpret_cast<__gm__ int32_t*>(
                (peer == common.rankId ? params.peermemInfo.rankSyncInWorldPtr : g_winRankAddr_[peer]) + offset);
            WriteGmByPassDCache(ready, static_cast<int32_t>(params.tilingData->hierarchicalSequence));
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    if (remembered) *remembered = readyMask;
}

// [0,32) reader flags, [32,64) producer flags, [64,96) per-rank
// broadcasts and [96,98) host closure flags. Every line has one writer.
__aicore__ inline __gm__ int32_t* ClosureSlot(const Params& params, uint32_t slot)
{
    return reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.closurePtr + slot * HIERARCHICAL_FLAG_BYTES);
}

__aicore__ inline void PublishCoreClosure(const Params& params, uint32_t block, bool producers)
{
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    WriteGmByPassDCache(ClosureSlot(params, block + (producers ? 32U : 0U)),
                        static_cast<int32_t>(params.tilingData->hierarchicalSequence));
}

// The reader coordinator can poll between foreground tasks. Retain cursors:
// each closed core/rank has one writer and cannot change before this invocation retires.
struct HostClosureCursor {
    uint32_t core = 0;
    uint32_t peer = 0;
    bool broadcast = false;
    bool closed = false;
};

__aicore__ inline bool ProgressHostClosure(const Params& params, uint32_t rank, bool producers,
                                           HostClosureCursor& cursor)
{
    if (cursor.closed) return true;
    const uint32_t seq = params.tilingData->hierarchicalSequence;
    while (cursor.core < params.tilingData->aicNum) {
        if (static_cast<uint32_t>(ReadGmByPassDCache(ClosureSlot(params, cursor.core + (producers ? 32U : 0U)))) != seq)
            return false;
        ++cursor.core;
    }
    const uint32_t host = HierarchicalHost(rank) * 16U;
    const uint32_t first = producers ? 80U : 64U;
    if (!cursor.broadcast) {
        DataSyncBarrier<MemDsbT::DDR>();
        const uint64_t offset = params.peermemInfo.closurePtr - params.peermemInfo.rankSyncInWorldPtr +
                                (first + rank - host) * HIERARCHICAL_FLAG_BYTES;
        for (uint32_t r = host; r < host + 16U; ++r)
            WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[r] + offset), static_cast<int32_t>(seq));
        cursor.broadcast = true;
    }
    while (cursor.peer < 16U) {
        if (static_cast<uint32_t>(ReadGmByPassDCache(ClosureSlot(params, first + cursor.peer))) != seq) return false;
        ++cursor.peer;
    }
    DataSyncBarrier<MemDsbT::DDR>();
    WriteGmByPassDCache(ClosureSlot(params, producers ? 97U : 96U), static_cast<int32_t>(seq));
    cursor.closed = true;
    return true;
}

__aicore__ inline void CloseHost(const Params& params, uint32_t rank, bool producers, HostClosureCursor& cursor)
{
    while (!ProgressHostClosure(params, rank, producers, cursor)) {
        const auto start = GetSystemCycle();
        while (GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

// Output barrier closes every local reader before the next invocation reuses GM.
__aicore__ inline void CrossRankSyncInHierarchicalHost(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                       const Params& params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    auto* counter =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(job.jobIndex) * RANK_SYNC_COUNTER_SLOT_BYTES);
    const int32_t epoch = ReadGmByPassDCache(counter) + 1;
    const uint32_t firstRank = HierarchicalHost(common.rankId) * HIERARCHICAL_RANKS_PER_HOST;
    auto* localFlags = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr);
    for (uint32_t localRank = job.jobIndex; localRank < HIERARCHICAL_RANKS_PER_HOST; localRank += job.totalJobs) {
        const uint32_t peer = firstRank + localRank;
        auto* remoteFlag = reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[peer]) + common.rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteFlag, epoch);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        // Dispatch's QP1 access ended before the input barrier. Only the
        // current Combine QP1 owner may progress its CQ here (AIV0 for QP=8).
        auto* scratch =
            HierarchicalCombineLane(params.tilingData, job.jobIndex) == 0U ? HierarchicalRoceScratch(params) : nullptr;
        RoceWaitFlagEquals(localFlags + peer * INT_CACHELINE, static_cast<uint32_t>(epoch),
                           HierarchicalGateway(common.rankId), scratch, RoceWaitKind::HostProducerBarrier);
#else
        GmSignalWaitBarrier(localFlags + peer * INT_CACHELINE, epoch);
#endif
    }
    WriteGmByPassDCache(counter, epoch);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

}  // namespace MegaMoeImpl
#endif
