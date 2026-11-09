// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {

__aicore__ inline __gm__ int32_t* LocalCompletionFlag(const Arguments& args, uint32_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(args.localFlags + static_cast<uint64_t>(index) * kFlagBytes);
}

__aicore__ inline uint64_t PeerPhaseSlot(const Arguments& args, uint32_t phase, uint32_t source)
{
    return args.config.barrierOffset + (static_cast<uint64_t>(phase) * args.config.world + source) * kFlagBytes;
}

// AIC completion certifies contiguous scratch, not self returnY. Communication
// owners must also finish their self-CV stores before the aggregate bypass.
__aicore__ inline void MarkCubeComplete(const Arguments& args)
{
    AscendC::AtomicAdd(LocalCompletionFlag(args, kLocalFlagAicDone), 1);
}

__aicore__ inline void MarkReorderComplete(const Arguments& args)
{
    AscendC::AtomicAdd(LocalCompletionFlag(args, kLocalFlagReorderDone), 1);
}

// Nonblocking: call once before each Combine quiet, then poll at the tail.
// Each peer has exactly one local observer. Once counted, it never reads that
// flag again in this invocation. Outbound completion alone is NOT incoming
// completion: inspect the peer-owned slot in this rank's symmetric window.
// These observations also permit single-slot route-inbox reuse: all peers
// publish finished only after acquiring their incoming summary, even for
// empty streams. Keep the observers and final local join before returning.
__aicore__ inline bool ObserveIncomingCompletion(const Arguments& args, uint32_t owner, bool& peerCounted,
                                                 bool& selfCounted)
{
    using namespace AscendC;
    if (!peerCounted) {
        const uint32_t peer = PeerForOwner(args.config.rank, owner);
        auto* finished = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + PeerPhaseSlot(args, 1U, peer));
        // Empty peer streams can finish a later invocation before this owner
        // observes the prior one. A later finished epoch also certifies the
        // current return data; per-CV payload flags still require exact epochs.
        if (ReadGmByPassDCache(finished) >= args.config.epoch) {
            AtomicAdd(LocalCompletionFlag(args, kLocalFlagIncomingDone), 1);
            peerCounted = true;
        }
    }
    // Only owner 0 contributes self. Empty AIC/AIV1 workloads still report.
    // All remote finished flags plus AIC28 alone must NOT bypass self reads:
    // scratch can be complete while the last AIV1's self scatter is pending.
    if (owner == 0U && !selfCounted &&
        ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagAicDone)) == static_cast<int32_t>(kAic) &&
        ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagReorderDone)) ==
            static_cast<int32_t>(CommunicationWorkers(args.config.world))) {
        AtomicAdd(LocalCompletionFlag(args, kLocalFlagIncomingDone), 1);
        selfCounted = true;
    }
    return peerCounted && (owner != 0U || selfCounted);
}

// The caller has already submitted SO finished and completed its final quiet.
// Only incoming observations remain; Reduce gets no additional timer/poll work.
__aicore__ inline void FinishCombinePeer(const Arguments& args, uint32_t owner, bool& peerCounted, bool& selfCounted)
{
    using namespace AscendC;
    while (!ObserveIncomingCompletion(args, owner, peerCounted, selfCounted)) {
        const uint64_t start = GetSystemCycle();
        while (GetSystemCycle() - start < 200U) {
        }
    }
}

}  // namespace MegaMoeUrma
