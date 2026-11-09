// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "route_runtime.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// All 28 steady Pack owners publish this existing counter only after Close().
// S1 also counts 28 startup-only AIV0 owners (no startup join). No producer
// waits for Reduce, so consumers may wait here without delaying a producer.
__aicore__ inline void WaitPackInverseReady(const Arguments& args)
{
    using namespace AscendC;
    auto* done = reinterpret_cast<__gm__ int32_t*>(args.localFlags + kLocalFlagPackDone * kFlagBytes);
    while (ReadGmByPassDCache(done) != static_cast<int32_t>(args.sharedExperts ? kLocalWorkers : kAic)) {
        const uint64_t start = GetSystemCycle();
        while (GetSystemCycle() - start < 200U) {
        }
    }
    // Acquire the Pack completion publication before loading inverse entries.
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
}

// The data pipeline has not submitted a token yet: low UB is temporary ID
// scratch. Only the final dependency array survives initialization, at its
// existing non-overlapping metadata offset. Each route slot stays distinct,
// including repeated expert choices for one token.
__aicore__ inline void BuildReduceDependencies(const Arguments& args, uint64_t firstRoute, uint32_t routes,
                                               AscendC::LocalTensor<int64_t>& inverse,
                                               AscendC::LocalTensor<uint64_t>& dependencies)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> ids;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.topkIds));
    LocalTensor<uint8_t> raw(TPosition::VECCALC, 0U, routes * args.topkIdBytes);
    DataCopyPad(raw, ids[firstRoute * args.topkIdBytes], {1U, routes * args.topkIdBytes, 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    auto words = raw.ReinterpretCast<uint32_t>();
    const auto* info = RouteInfo(args);
    const uint32_t wordStride = args.topkIdBytes / sizeof(uint32_t);
    for (uint32_t i = 0U; i < routes; ++i) {
        const uint32_t expert = words.GetValue(i * wordStride);
        const uint32_t peer = expert / args.config.localExperts;
        const uint64_t row = static_cast<uint64_t>(inverse.GetValue(i));
        uint64_t dependency;
        // Unitflag self is protected by the final Reorder CV, not AIC GMM2.
        const uint64_t peerRow = row - info->outgoing[peer * args.config.localExperts];
        dependency = uint64_t(peer) * args.config.maxPeerChunks + peerRow / args.config.cvRows;
        dependencies.SetValue(i, dependency);
    }
}

}  // namespace MegaMoeUrma
