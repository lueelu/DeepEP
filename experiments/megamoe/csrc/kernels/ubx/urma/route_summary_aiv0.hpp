// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "device_routes.hpp"
#include "route_summary_parallel.hpp"

namespace MegaMoeUrma {
__aicore__ inline __gm__ int32_t* RoutePhaseFlag(const Arguments& a, uint32_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(a.localFlags + uint64_t(index) * kFlagBytes);
}
__aicore__ inline void WaitRoutePhase(const Arguments& a, uint32_t index, uint32_t value)
{
    while (AscendC::ReadGmByPassDCache(RoutePhaseFlag(a, index)) != static_cast<int32_t>(value)) {
        const uint64_t tick = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - tick < 64U) {
        }
    }
    RouteFence<AscendC::HardEvent::S_MTE2>();
}

// Each active source owner snapshots one mailbox. MTE stores are byte-exact:
// adjacent incoming rows/returnBase entries may share GM cache lines, so no
// concurrent Scalar cached writes or DCCI are allowed on these output arrays.
__aicore__ inline void ReadRouteSource(const Arguments& a, uint32_t source)
{
    if (source >= a.config.world) return;
    WaitLocalDone(reinterpret_cast<__gm__ uint8_t*>(a.symmetric + a.config.routeReadyOffset + source * kFlagBytes),
                  a.config.epoch);
    RouteLoad(a.symmetric + a.config.routeInboxOffset + uint64_t(source) * kRouteMailboxBytes, 0U,
              8U + (a.config.localExperts + 1U) * 4U);
    LocalTensor<uint32_t> inbox(TPosition::VECCALC, 0U, kRouteMailboxBytes / 4U);
    LocalTensor<uint32_t> prefix(TPosition::VECCALC, 2048U, kMaxLocalExperts + 1U);
    LocalTensor<uint64_t> base(TPosition::VECCALC, 4096U, 4U);
    for (uint32_t e = 0; e <= a.config.localExperts; ++e) prefix.SetValue(e, inbox.GetValue(e + 2U));
    base.SetValue(0, uint64_t(inbox.GetValue(0)) | (uint64_t(inbox.GetValue(1)) << 32U));
    RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, incoming) + uint64_t(source) * (kMaxLocalExperts + 1U) * 4U,
              2048U, (a.config.localExperts + 1U) * 4U);
    RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, returnBase) + uint64_t(source) * 8U, 4096U, 8U);
    AscendC::AtomicAdd(RoutePhaseFlag(a, kLocalFlagRouteSourcesDone), 1);
}

__aicore__ inline void BuildOwnedRouteExperts(const Arguments& a, uint32_t core)
{
    const uint32_t begin = uint64_t(a.config.localExperts) * core / kAic;
    const uint32_t end = uint64_t(a.config.localExperts) * (core + 1U) / kAic;
    if (begin != end) {
        // Only incoming is needed and it is immutable after SourcesDone.
        // Do not read the per-expert output arrays while other workers write.
        constexpr uint32_t incomingOffset = offsetof(DeviceRouteInfo, incoming);
        constexpr uint32_t incomingBytes = kWorld * (kMaxLocalExperts + 1U) * sizeof(uint32_t);
        RouteLoad(a.routeInfo + incomingOffset, kRouteInfoUb + incomingOffset, incomingBytes);
        auto* r = reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
        SummarizeRouteExperts(a.config, *r, begin, end);
        constexpr uint32_t quota = (kMaxLocalExperts + kAic - 1U) / kAic;
        LocalTensor<uint32_t> sources(TPosition::VECCALC, 0U, quota * (kWorld + 1U));
        LocalTensor<uint32_t> groups(TPosition::VECCALC, 4096U, quota);
        LocalTensor<uint32_t> rows(TPosition::VECCALC, 6144U, quota);
        LocalTensor<uint64_t> tiles(TPosition::VECCALC, 8192U, quota);
        for (uint32_t e = begin; e < end; ++e) {
            for (uint32_t s = 0; s <= kWorld; ++s) sources.SetValue((e - begin) * (kWorld + 1U) + s, r->sources[e][s]);
            groups.SetValue(e - begin, r->groupBase[e + 1U]);
            rows.SetValue(e - begin, r->compactBase[e + 1U]);
            tiles.SetValue(e - begin, r->gmm1Base[e + 1U]);
        }
        RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, sources) + uint64_t(begin) * (kWorld + 1U) * 4U, 0U,
                  (end - begin) * (kWorld + 1U) * 4U);
        RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, groupBase) + uint64_t(begin + 1U) * 4U, 4096U,
                  (end - begin) * 4U);
        RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, compactBase) + uint64_t(begin + 1U) * 4U, 6144U,
                  (end - begin) * 4U);
        RouteSave(a.routeInfo + offsetof(DeviceRouteInfo, gmm1Base) + uint64_t(begin + 1U) * 8U, 8192U,
                  (end - begin) * 8U);
    }
    AscendC::AtomicAdd(RoutePhaseFlag(a, kLocalFlagRouteExpertsDone), 1);
}

// Called only by block0 AIV0, after all 28 expert shards are committed.
__aicore__ inline void FinishParallelRouteSummary(Arguments& a)
{
    RouteLoad(a.routeInfo, kRouteInfoUb, sizeof(DeviceRouteInfo));
    auto* r = reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    const bool wave12 = a.config.world == 4U && (!MEGAMOE_URMA_ROUTE_REORDER || a.sharedExperts != 0U);
    FinishRouteSummaryPrefix(a.config, *r, wave12);
    LocalTensor<int32_t> exported(TPosition::VECCALC, 0U, kMaxLocalExperts);
    for (uint32_t e = 0; e < a.config.localExperts; ++e)
        exported.SetValue(e, static_cast<int32_t>(r->sources[e][a.config.world]));
    RouteSave(a.expertCountsOutput, 0U, a.config.localExperts * sizeof(int32_t));
    constexpr uint32_t receiveOffset = offsetof(DeviceRouteInfo, incoming);
    RouteSave(a.routeInfo + receiveOffset, kRouteInfoUb + receiveOffset, sizeof(DeviceRouteInfo) - receiveOffset);
    RouteScheduleOutput out(a);
    EmitRouteTasks(a.config, *r, out, wave12);
    out.Finish();
    RouteStore(a.routeConfig, 0U, a.config);
    RouteFlush(a.routeConfig, sizeof(Config));
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::WriteGmByPassDCache(RoutePhaseFlag(a, kLocalFlagRouteSummaryDone), int32_t{1});
}

// No SyncAll: AIV1 continues Pack/Dispatch throughout these AIV0-only phases.
// S0 invokes this AFTER the existing startup Pack join, not before it.
__aicore__ inline void RunAiv0RouteSummary(Arguments& a, uint32_t core, Timer& timer)
{
    timer.CoarseBegin(CoarsePhase::RouteSummary);
    const uint64_t start = timer.TraceBegin();
    ReadRouteSource(a, core);
    WaitRoutePhase(a, kLocalFlagRouteSourcesDone, a.config.world);
    BuildOwnedRouteExperts(a, core);
    if (core == 0U) {
        WaitRoutePhase(a, kLocalFlagRouteExpertsDone, kAic);
        FinishParallelRouteSummary(a);
    }
    timer.TraceEnd(TraceEvent::RouteSummary, start);
    timer.CoarseEnd(CoarsePhase::RouteSummary);
}
}  // namespace MegaMoeUrma
