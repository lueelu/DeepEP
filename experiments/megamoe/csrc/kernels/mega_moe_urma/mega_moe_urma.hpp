// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

// Device execution entry: ProcessRouted (S0) and ProcessShared (S1).
// AIC computes; AIV0 prepares routes/weights; AIV1 overlaps communication,
// activation and reorder. Stage headers contain the individual mechanisms.
// Read these two entry functions for sequencing; the stages are not serial.

#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_gmm.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_shared_expert.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_pack.hpp"

#include "kernels/mega_moe_urma/stage/mega_moe_reduce.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_reorder.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_transport.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_timer.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_route_prepare.hpp"
#include "kernels/mega_moe_urma/stage/mega_moe_dispatch_combine.hpp"

namespace MegaMoeUrma {
using namespace AscendC;

// Peer-start publication uses scratch above Pack's live buffers.
constexpr uint32_t kCommunicationSignalOffset = kPackUbBase + kPackUbBytes;
static_assert(kCommunicationSignalOffset % 32U == 0U);

__aicore__ inline void TraceLocalSync(Timer& timer, uint32_t stage)
{
    TraceMeta meta;
    meta.cv = stage;  // 0=two-round startup pack, 3=local invocation completion.
    const uint64_t start = timer.TraceBegin();
    const auto coarsePhase = stage == 0U ? CoarsePhase::SyncStart : CoarsePhase::SyncEnd;
    timer.CoarseBegin(coarsePhase);
    SyncAll<false>();
    timer.CoarseEnd(coarsePhase);
    timer.TraceEnd(TraceEvent::LocalSync, start, meta);
}

// Publish before packing so remote startup can overlap local DMA. Do not
// wait here: all workers must reach the two-round startup join. Previous-call
// receive/return consumers finished at the prior local join, and this call's
// counter resets completed before the first local route join. Self pack and remote
// PUTs address disjoint source intervals in receive storage.
template <class S>
__aicore__ inline void PublishPeerStart(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    TraceMeta meta;
    meta.peer = peer;
    timer.Begin(Phase::Barrier);
    const uint64_t start = timer.TraceBegin();
    PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot<S>(args, 0U, args.config.rank), peer,
                args.config.epoch, signal, EVENT_ID0);
    timer.TraceEnd(TraceEvent::PublishStart, start, meta);
    timer.End(Phase::Barrier);
}

template <class S>
__aicore__ inline void RunProfiledCompute(const Arguments& args, uint32_t block, Timer& timer)
{
    // Select once, outside the loop. NoopTimer has no state, clocks, stores,
    // cache operations or fences; both coarse and formal use this instantiation.
    if (args.coarseTrace != 0U || (args.trace == 0U && args.timer == 0U)) {
        const auto phase = g_coreType == AIC ? CoarsePhase::Gmm : CoarsePhase::Dequant;
        timer.CoarseBegin(phase);
        NoopTimer disabled;
        RunCubeAndPrologue<S>(args, block, disabled);
        timer.CoarseEnd(phase);
    } else {
        RunCubeAndPrologue<S>(args, block, timer);
    }
}

template <class S>
__aicore__ inline void ProcessRouted(Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity, args.coarseTrace != 0U);
    // 1. Reset flags and prepare local routes; incoming summaries arrive later.
    PrepareDeviceRoutes<S>(args, timer);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const uint32_t communicationWorkers = CommunicationWorkers<S>();
    const int64_t originalOverflow =
        GetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    // 2. Publish readiness and fill the startup Pack rounds.
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block < communicationWorkers) {
            PublishPeerStart<S>(args, block, timer);
        }
        timer.CoarseBegin(CoarsePhase::Pack);
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        for (uint32_t round = 0U; round < kPackStartupRounds; ++round) {
            PackOneRound<S>(args, round, worker, kLocalWorkers, timer);
        }
    }
    if constexpr (g_coreType == AIV) timer.CoarseEnd(CoarsePhase::Pack);
    // No CV loop contains a rank-wide or local full-core synchronization.
    TraceLocalSync(timer, 0U);
    // 3. Each engine advances its role concurrently.
    if constexpr (g_coreType == AIC) {
        WaitRouteSummary(args, timer);
        RunProfiledCompute<S>(args, block, timer);
        MarkCubeComplete(args);
    } else if (GetSubBlockIdx() == 0U) {
        RunAiv0RouteSummary<S>(args, block, timer);
        WaitRouteSummary(args, timer);
        RunProfiledCompute<S>(args, block, timer);
    } else {
        SharedIncoming<S> incoming{args, block};
        RunCooperativeAiv1<S>(args, block, timer, incoming);
    }
    if constexpr (g_coreType == AIV) {
        // 4. Reduce returned rows after releasing this engine's scratch.
        // Only this engine releases its phase-private UB/events. Other
        // compute and communication engines may still be producing rows.
        PipeBarrier<PIPE_ALL>();
        const uint32_t worker = ReduceWorkerIndex<S>(block, GetSubBlockIdx());
        if (worker < ReduceWorkers<S>()) {
            timer.Begin(Phase::ReducePhase);
            timer.CoarseBegin(CoarsePhase::Reduce);
            // The legacy pipeline only exists for the H3584/TopK8 geometry; other
            // specializations never instantiate its fixed UB contract.
            if constexpr (S::kHidden == 3584U && S::kTopK == 8U) {
                if (UseLegacyReduce<S>(args.config))
                    Reduce<S>(args, worker, timer);
                else
                    ReduceGeneric<S>(args, worker, ReduceWorkers<S>(), timer);
            } else {
                ReduceGeneric<S>(args, worker, ReduceWorkers<S>(), timer);
            }
            timer.CoarseEnd(CoarsePhase::Reduce);
            const uint64_t owned = ReduceOwnedRows<S>(args.config.bs, worker);
            timer.End(Phase::ReducePhase, owned * S::kHidden * sizeof(bfloat16_t));
            PipeBarrier<PIPE_ALL>();
        }
    }
    // A local MIX join, not a rank barrier: every reducer, peer owner and AIC
    // must finish before returning and reusing this invocation's workspaces.
    TraceLocalSync(timer, 3U);
    timer.FinishTrace();
    SetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>(originalOverflow);
}

template <class S, class T>
__aicore__ inline void RunSharedComputeSequence(const Arguments& a, uint32_t block, T& timer)
{
    RunSharedGmm<S, true>(a, block, timer);
    if constexpr (g_coreType == AIC) {
        // SharedMxGmm (including its destructor) no longer owns L1/events.
        WriteGmByPassDCache(LocalCompletionFlag(a, kLocalFlagSharedL1Released + block), 1);
        // RunSharedGmm has drained this core's shared GMM1 output and retired
        // its context. Count also cores with zero assigned shared tiles.
        // Do not wait for routed compute/shared GMM2: that would create a cycle
        // with the remaining Dispatch data needed by the routed experts.
        if (a.config.bs < 4096U) AtomicAdd(LocalCompletionFlag(a, kLocalFlagSharedGmm1Done), 1);
    }
    RunCubeAndPrologue<S>(a, block, timer);
    if constexpr (g_coreType == AIC) MarkCubeComplete(a);  // routed self-return is ready before Shared GMM2
    // AIV0 must finish/release routed B slots before the W8 AIC context
    // initializes those same L1 addresses. AIV0 then proceeds to Reduce.
    WaitComputeFlag(a.localFlags, kLocalFlagRoutedPrologueReleased + block, 1);
    RunSharedGmm<S, false>(a, block, timer);
}
template <class S>
__aicore__ inline void ProcessShared(Arguments& a)
{
    Timer timer;
    timer.Init(a.timer, a.trace, a.traceCapacity, a.coarseTrace != 0);
    timer.SharedMode();
    // S1 overlaps shared GMM, routed work and communication by engine role.
    PrepareDeviceRoutes<S>(a, timer);
    const uint32_t block = GetBlockIdx() / GetTaskRation(), comm = CommunicationWorkers<S>();
    const int64_t overflow = GetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>(0);
    if constexpr (g_coreType == AIC) {
        WaitRouteSummary(a, timer);
        timer.CoarseBegin(CoarsePhase::Gmm);
        if (a.coarseTrace || (!a.trace && !a.timer)) {
            NoopTimer t;
            RunSharedComputeSequence<S>(a, block, t);
        } else
            RunSharedComputeSequence<S>(a, block, timer);
        timer.CoarseEnd(CoarsePhase::Gmm);
    } else if (GetSubBlockIdx() == 0) {
        RunAiv0RouteSummary<S>(a, block, timer);
        WaitRouteSummary(a, timer);
        timer.CoarseBegin(CoarsePhase::Pack);
        PackBatch<S> startup(a, timer);
        for (uint32_t round = 0; round < kPackStartupRounds; ++round)
            PackOneRound<S>(a, round, LocalWorkerIndex(block, 0U), kLocalWorkers, startup);
        startup.Close();
        MarkPackWorkerComplete(a);
        timer.CoarseEnd(CoarsePhase::Pack);
        // AIC may still be in shared GMM1 while this AIV0 finishes Pack.
        // Do not initialize the routed W4 B ring until its L1 owner releases it.
        WaitComputeFlag(a.localFlags, kLocalFlagSharedL1Released + block, 1);
        timer.CoarseBegin(CoarsePhase::Dequant);
        if (a.coarseTrace || (!a.trace && !a.timer)) {
            NoopTimer t;
            RunCubeAndPrologue<S>(a, block, t);
        } else
            RunCubeAndPrologue<S>(a, block, timer);
        timer.CoarseEnd(CoarsePhase::Dequant);
        // This AIV0 has finished its own dequant contexts. Release its local
        // UB/event use before Reduce; do not wait for other AIC/AIV workers.
        PipeBarrier<PIPE_ALL>();
        WriteGmByPassDCache(LocalCompletionFlag(a, kLocalFlagRoutedPrologueReleased + block), 1);
        NoReduceObserver observer;  // incoming counters remain AIV1-owned
        timer.CoarseBegin(CoarsePhase::Reduce);
        ReduceGeneric<S, true>(a, LocalWorkerIndex(block, 0U), kLocalWorkers, timer, observer);
        timer.CoarseEnd(CoarsePhase::Reduce);
    } else {
        if (block < comm) PublishPeerStart<S>(a, block, timer);
        timer.CoarseBegin(CoarsePhase::Pack);
        PackBatch<S> pack(a, timer);
        for (uint32_t round = 0; round < kPackStartupRounds; ++round)
            PackOneRound<S>(a, round, LocalWorkerIndex(block, 1U), kLocalWorkers, pack);
        pack.Close();
        timer.CoarseEnd(CoarsePhase::Pack);
        SharedIncoming<S> incoming{a, block};
        RunCooperativeAiv1<S>(a, block, timer, incoming);
        PipeBarrier<PIPE_ALL>();
        timer.CoarseBegin(CoarsePhase::Reduce);
        ReduceGeneric<S, true>(a, LocalWorkerIndex(block, 1U), kLocalWorkers, timer, incoming);
        timer.CoarseEnd(CoarsePhase::Reduce);
        if (block < comm) {
            const auto start = timer.TraceBegin();
            Quiet(PeerForOwner(a.config.rank, block));
            TraceMeta completion;
            completion.peer = PeerForOwner(a.config.rank, block);
            timer.TraceEnd(TraceEvent::CombineSendComplete, start, completion);
            while (!ObserveIncomingCompletion<S>(a, block, incoming.peer, incoming.self)) {
                const auto t = GetSystemCycle();
                while (GetSystemCycle() - t < 200U) {
                }
            }
        }
    }
    // Join routed work before adding the shared-expert result.
    TraceLocalSync(timer, 3U);
    if constexpr (g_coreType == AIV) {
        timer.CoarseBegin(CoarsePhase::SharedSum);
        SharedAdd<S>(a, LocalWorkerIndex(block, GetSubBlockIdx()), timer);
        timer.CoarseEnd(CoarsePhase::SharedSum);
    }
    const auto join = timer.TraceBegin();
    SyncAll<false>();
    TraceMeta m;
    m.cv = 4;
    timer.TraceEnd(TraceEvent::LocalSync, join, m);
    timer.FinishTrace();
    SetCtrlSpr<UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL, UrmaMegaMoeImpl::OVERFLOW_MODE_CTRL>(overflow);
}

}  // namespace MegaMoeUrma
