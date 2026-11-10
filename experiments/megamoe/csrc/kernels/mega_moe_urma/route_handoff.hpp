// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/device_routes.hpp"
#include "kernels/mega_moe_urma/pack_round.hpp"
#include "kernels/mega_moe_urma/route_summary_aiv0.hpp"

namespace MegaMoeUrma {
__aicore__ inline __gm__ int32_t* RouteSummaryFlag(const Arguments& a)
{
    return reinterpret_cast<__gm__ int32_t*>(a.localFlags + kLocalFlagRouteSummaryDone * kFlagBytes);
}
__aicore__ inline bool RouteSummaryReady(const Arguments& a)
{
    return AscendC::ReadGmByPassDCache(RouteSummaryFlag(a)) == 1;
}
__aicore__ inline void AcquireRouteSummary(Arguments& a)
{
    // The producer never changes outgoing's cache lines. Only invalidate the
    // newly published receive portion, including lines fetched speculatively.
    constexpr uint32_t offset = offsetof(DeviceRouteInfo, incoming);
    RouteFlush(a.routeInfo + offset, sizeof(DeviceRouteInfo) - offset);
    RouteReloadConfig(a);
    RouteFence<AscendC::HardEvent::S_MTE2>();
}
__aicore__ inline void WaitRouteSummary(Arguments& a, Timer& timer)
{
    const uint64_t start = timer.TraceWaitBegin();
    while (!RouteSummaryReady(a)) {
        const uint64_t tick = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - tick < 64U) {
        }
    }
    AcquireRouteSummary(a);
}

// Called after the existing startup quotas. Pack assignments never change:
// this loop advances the SAME nextPackRound cursor used by normal activation.
// AIV0 builds the summary; helpers never wait for a remote source while they
// still have local Pack work. No low-UB ownership is transferred to this loop.
__aicore__ inline void PrepareActivationRoutes(Arguments& a, uint32_t helper, uint32_t& nextPackRound, PackBatch& pack,
                                               Timer& timer)
{
    const uint32_t packOwner = helper + CommunicationWorkers(a.config.world);
    while (!RouteSummaryReady(a)) {
        if (nextPackRound < a.config.packRoundCount) {
            PackOneRound(a, nextPackRound++, packOwner, kAic, pack);
        } else {
            pack.Close();
            const uint64_t tick = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - tick < 64U) {
            }
        }
    }
    // No UB payload is touched here; non-owner helpers may retain their Pack
    // ring until the first actual activation/scale-gather task closes it.
    WaitRouteSummary(a, timer);
}
}  // namespace MegaMoeUrma
