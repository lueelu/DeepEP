// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute.hpp"
#include "shared.hpp"
#include "communication_schedule.hpp"
#include "pack_reduce.hpp"
#include "reduce_generic.hpp"
#include "pack_round.hpp"
#include "peer_reorder_progress.hpp"
#include "tail_completion.hpp"
#include "timer.hpp"
#include "device_routes.hpp"
#include "transport.hpp"
#include "cooperative_aiv1.hpp"

namespace MegaMoeUrma {
using namespace AscendC;

// Reorder owns low 128KiB and Pack owns the following 58,880B. Keep the
// communication owner's persistent signal/CV metadata above BOTH regions:
// reorder can execute with submitted Combine PUTs still in flight.
constexpr uint32_t kCommunicationSignalOffset = kPackUbBase + kPackUbBytes;
// Counters become live only after Pack closes. Reuse the region after both
// span bitmaps, below the signal slot, so BS16384 also fits 192KiB UB.
constexpr uint32_t kCommunicationSubmittedOffset = kPackUbBase + 2U * kPeerReorderBitmapBytes;
constexpr uint32_t kCommunicationPendingOffset = kCommunicationSubmittedOffset + kMaxPeerChunks * sizeof(uint32_t);
static_assert(kCommunicationSignalOffset % 32U == 0U);
static_assert(kCommunicationPendingOffset + kMaxPeerChunks * sizeof(uint32_t) <= kRequiredUbBytes);
// After the local Dispatch join, this owner's Pack UB is dead. Reuse the
// remote/self first-row bitmaps, away from both DMA slots and CV counters.
constexpr uint32_t kRemoteReorderBitmapOffset = kPackUbBase;
constexpr uint32_t kSelfReorderBitmapOffset = kRemoteReorderBitmapOffset + kPeerReorderBitmapBytes;
static_assert(kSelfReorderBitmapOffset + kPeerReorderBitmapBytes <= kCommunicationSubmittedOffset);
static_assert(kCommunicationPendingOffset + kMaxPeerChunks * sizeof(uint32_t) <= kCommunicationSignalOffset);

template <typename T>
__aicore__ inline void WaitProgressFlag(__gm__ T* flag, T expected, Timer& timer, TraceEvent event,
                                        const TraceMeta& meta)
{
    while (ReadGmByPassDCache(flag) < expected) {
        const uint64_t backoff = GetSystemCycle();
        while (GetSystemCycle() - backoff < 64U) {
        }
    }
    (void)timer;
    (void)event;
    (void)meta;
}

__aicore__ inline TraceMeta ChunkTraceMeta(const Chunk& chunk)
{
    TraceMeta meta;
    meta.peer = chunk.peer;
    meta.cv = chunk.cv;
    meta.row = chunk.localRow;
    meta.rows = chunk.rows;
    return meta;
}

__aicore__ inline uint32_t PeerDispatchCvCount(const Arguments& args, uint32_t peer)
{
    return RuntimeSendCvCount(args, peer);
}

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
__aicore__ inline void PublishPeerStart(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    TraceMeta meta;
    meta.peer = peer;
    timer.Begin(Phase::Barrier);
    const uint64_t start = timer.TraceBegin();
    PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot(args, 0U, args.config.rank), peer,
                args.config.epoch, signal, EVENT_ID0);
    timer.TraceEnd(TraceEvent::PublishStart, start, meta);
    timer.End(Phase::Barrier);
}

// One owner, one QP. Data and SO notifications remain in flight while this
// worker prepares other disjoint GM rows. No Combine is submitted in Dispatch.

__aicore__ inline void CommunicatePeer(Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    const uint32_t workers = CommunicationWorkers(args.config.world);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    auto* startFlag = reinterpret_cast<__gm__ uint64_t*>(symmetric + PeerPhaseSlot(args, 0U, peer));
    WaitProgressFlag(startFlag, args.config.epoch, timer, TraceEvent::WaitStartPeer, peerMeta);
    timer.End(Phase::Barrier);

    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    const uint32_t dispatchCvs = PeerDispatchCvCount(args, peer);
    for (uint32_t round = 0U; round < args.config.packRoundCount; ++round) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        if (chunk.rows != 0U) {
            auto meta = ChunkTraceMeta(chunk);
            WaitProgressFlag(PackReadyFlag(args, peer, round), static_cast<int32_t>(chunk.rows), timer,
                             TraceEvent::WaitDispatch, meta);
            timer.CoarseBegin(CoarsePhase::Dispatch);
            timer.Begin(Phase::Dispatch);
            meta.bytes = chunk.rows * h;
            uint64_t start = timer.TraceBegin();
            Put(symmetric, args.config.receiveXOffset + chunk.remoteRow * h,
                reinterpret_cast<__gm__ uint8_t*>(args.expandedX + chunk.localRow * h),
                static_cast<uint32_t>(meta.bytes), peer);
            timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
            meta.bytes = chunk.rows * scales;
            start = timer.TraceBegin();
            Put(symmetric, args.config.receiveScaleOffset + chunk.remoteRow * scales,
                reinterpret_cast<__gm__ uint8_t*>(args.expandedScale + chunk.localRow * scales),
                static_cast<uint32_t>(meta.bytes), peer);
            timer.TraceEnd(TraceEvent::DispatchPutScale, start, meta);
            meta.bytes = 0U;
            start = timer.TraceBegin();
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            PublishOrdered(symmetric, args.config.dispatchReadyOffset + ready * kFlagBytes, peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
            timer.End(Phase::Dispatch, chunk.rows * (h + scales), true);
        }
        // Even an empty peer keeps its static quota for other peers/self.
        // Pack's GM writes are complete before it increments packReady.
        PackOneRound(args, round + kPackStartupRounds, owner, kAic, timer);
        if (chunk.rows != 0U) {
            // Prepare future rows while this CV is in flight, but retire its
            // X/scale/SO operations before submitting the next CV to this peer.
            const auto meta = ChunkTraceMeta(chunk);
            const uint64_t start = timer.TraceBegin();
            Quiet(peer);
            if (round + 1U == dispatchCvs) timer.CoarseEnd(CoarsePhase::Dispatch);
            timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
            timer.MarkSendComplete(Phase::Dispatch);
        }
    }
    MarkPackWorkerComplete(args);
    WaitRouteSummary(args, timer);
    AtomicAdd(LocalCompletionFlag(args, kLocalFlagDispatchDone), 1);
    if (owner == 0U) {
        WaitProgressFlag(LocalCompletionFlag(args, kLocalFlagPackDone), static_cast<int32_t>(kAic), timer,
                         TraceEvent::WaitLocalDispatch, peerMeta);
        AtomicAdd(LocalCompletionFlag(args, kLocalFlagDispatchDone), 1);
    }
    // Every nonempty CV has completed its peer quiet before this local join.
    // This is not a global rank barrier.
    WaitProgressFlag(LocalCompletionFlag(args, kLocalFlagDispatchDone), static_cast<int32_t>(args.config.world), timer,
                     TraceEvent::WaitLocalDispatch, peerMeta);

    const uint32_t begin = RouteInfo(args)->recvCvBase[peer];
    const uint32_t end = RouteInfo(args)->recvCvBase[peer + 1U];
    const uint32_t selfBegin = RouteInfo(args)->recvCvBase[args.config.rank];
    const uint32_t selfEnd = RouteInfo(args)->recvCvBase[args.config.rank + 1U];
    // Remote CVs hold submitted ROW counts; self CVs hold copied SPAN counts.
    // Both become the submitted sentinel only after the full CV ready flag.
    // Each row is written once, with no GM reuse before final quiet.
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kCommunicationSubmittedOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> selfCopied(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> remoteSpanCopied(TPosition::VECCALC, kRemoteReorderBitmapOffset, kPeerReorderBitmapWords);
    LocalTensor<uint32_t> selfSpanCopied(TPosition::VECCALC, kSelfReorderBitmapOffset, kPeerReorderBitmapWords);
    const auto remoteSpans = PeerReorderSpanRange(args, begin, end);
    const auto selfSpans = PeerReorderSpanRange(args, selfBegin, selfEnd);
    ClearReorderBitmap(remoteSpanCopied, remoteSpans.spanCount);
    ClearReorderBitmap(selfSpanCopied, selfSpans.spanCount);
    uint32_t remaining = end - begin;
    uint32_t selfRemaining = 0U;
    for (uint32_t j = 0U; j < remaining; ++j) submitted.SetValue(j, 0U);
    for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
        selfCopied.SetValue(j, 0U);
        ++selfRemaining;
    }
    bool sentComplete = false;
    bool finishedPublished = false;
    bool selfComplete = false;
    bool peerCounted = false;
    bool selfCounted = false;
    bool waiting = false;
    uint64_t waitStart = 0U;
    while (!sentComplete || !selfComplete) {
        bool progress = false;
        for (uint32_t j = 0U; j < end - begin; ++j) {
            uint32_t sentRows = submitted.GetValue(j);
            if (sentRows == kReorderCvSubmitted) continue;
            const Chunk chunk = RuntimeReceiveChunkByIndex(args, begin + j);
            // Remote progress only needs the CV key; avoid a separate count pass.
            const PeerReorderRange range{begin + j, 0U};
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, remoteSpans.spanBegin, remoteSpanCopied,
                                                            timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            const uint32_t readyRows = ReorderedPeerPrefixRows(args, range, remoteSpans.spanBegin, remoteSpanCopied);
            while (sentRows < chunk.rows) {
                const uint32_t left = chunk.rows - sentRows;
                const uint32_t putRows = left < kCombinePutRows ? left : kCombinePutRows;
                if (sentRows + putRows > readyRows) break;
                auto meta = ChunkTraceMeta(chunk);
                meta.row = chunk.localRow + sentRows;
                meta.rows = putRows;
                meta.bytes = putRows * rowBytes;
                timer.CoarseBegin(CoarsePhase::Combine);
                timer.Begin(Phase::Combine);
                const uint64_t start = timer.TraceBegin();
                Put(symmetric, args.config.returnYOffset + (chunk.remoteRow + sentRows) * rowBytes,
                    reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + (chunk.localRow + sentRows) * rowBytes),
                    static_cast<uint32_t>(meta.bytes), peer);
                timer.TraceEnd(TraceEvent::CombinePut, start, meta);
                timer.End(Phase::Combine, meta.bytes, true);
                sentRows += putRows;
            }
            submitted.SetValue(j, sentRows);
            if (sentRows != chunk.rows) continue;
            // Payloads may leave in 128-row pieces, but Reduce still sees one
            // ready flag per original CV (256 rows by default). Same-QP SO
            // orders the flag after EVERY preceding data PUT, without quiet.
            auto meta = ChunkTraceMeta(chunk);
            meta.bytes = 0U;
            timer.Begin(Phase::Combine);
            const uint64_t start = timer.TraceBegin();
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            PublishOrdered(symmetric, args.config.combineReadyOffset + ready * kFlagBytes, peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
            timer.End(Phase::Combine, 0U, true);
            submitted.SetValue(j, kReorderCvSubmitted);
            --remaining;
            progress = true;
        }
        if (remaining == 0U && !finishedPublished) {
            if (waiting) {
                waiting = false;
            }
            // Publish remote finished promptly, including empty edges. Defer
            // the blocking quiet until this owner's self copies are complete.
            const uint64_t publishStart = timer.TraceBegin();
            PublishOrdered(symmetric, PeerPhaseSlot(args, 1U, args.config.rank), peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, publishStart, peerMeta);
            finishedPublished = true;
            progress = true;
        }
        for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
            const uint32_t copied = selfCopied.GetValue(j);
            if (copied == kReorderCvSubmitted) continue;
            const Chunk chunk = RuntimeReceiveChunkByIndex(args, selfBegin + j);
            const auto range = LoadPeerReorderRange(args, selfBegin + j);
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, selfSpans.spanBegin, selfSpanCopied,
                                                            timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            selfCopied.SetValue(j, copied + advanced);
            if (copied + advanced != range.spanCount) continue;
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            // Reorder drained local stores. Self requires no URMA operation.
            WriteGmByPassDCache(
                reinterpret_cast<__gm__ uint64_t*>(symmetric + args.config.combineReadyOffset + ready * kFlagBytes),
                args.config.epoch);
            selfCopied.SetValue(j, kReorderCvSubmitted);
            --selfRemaining;
            progress = true;
        }
        if (selfRemaining == 0U && !selfComplete) {
            MarkReorderComplete(args);
            selfComplete = true;
            progress = true;
        }
        if (finishedPublished && selfComplete && !sentComplete) {
            if (waiting) {
                waiting = false;
            }
            // Self ready must never depend on a remote final quiet. Unready
            // self CVs are rescanned above while remote operations progress.
            // Final quiet is mandatory even when timer is disabled.
            const uint64_t completeStart = timer.TraceBegin();
            Quiet(peer);
            timer.CoarseEnd(CoarsePhase::Combine);
            timer.TraceEnd(TraceEvent::CombineSendComplete, completeStart, peerMeta);
            if (end != begin) timer.MarkSendComplete(Phase::Combine);
            sentComplete = true;
            progress = true;
        }
        ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
        if (!progress) {
            if (!waiting) {
                waitStart = timer.TraceWaitBegin();
                waiting = true;
            }
            const uint64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 200U) {
            }
        }
    }
    FinishCombinePeer(args, owner, peerCounted, selfCounted);
}

__aicore__ inline void CommunicatePeerShared(Arguments& args, uint32_t owner, Timer& timer, SharedIncoming& incoming)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    uint64_t start = timer.TraceWaitBegin();
    WaitStartAtLeast(symmetric + PeerPhaseSlot(args, 0U, peer), args.config.epoch);

    timer.End(Phase::Barrier);
    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    const uint32_t dispatchCvs = PeerDispatchCvCount(args, peer);
    uint32_t nextPackRound = kPackStartupRounds;
    PackBatch pack(args, timer);
    uint32_t nextDispatch = 0U, step = 0U;
    uint64_t dispatchBytes = 0U;
    bool dispatchStarted = false;
    // Small S1 cases must not flood the link while the shared head is running.
    // A wave is one merged X+Scale submission, not one individual PUT or CV.
    const uint32_t startupWaves = args.config.bs / 1024U > 0U ? args.config.bs / 1024U : 1U;
    bool sharedHeadReady = args.config.bs >= 4096U;
    while (nextDispatch < dispatchCvs) {
        if (!sharedHeadReady) {
            sharedHeadReady =
                ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagSharedGmm1Done)) == static_cast<int32_t>(kAic);
            if (!sharedHeadReady && step >= startupWaves) {
                // Retire the preceding static Pack quota before parking this
                // sender; do not withhold its ready notifications from others.
                pack.FlushReady();
                const uint64_t waitStart = timer.TraceWaitBegin();
                while (ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagSharedGmm1Done)) !=
                       static_cast<int32_t>(kAic)) {
                    const uint64_t backoff = GetSystemCycle();
                    while (GetSystemCycle() - backoff < 200U) {
                    }
                }

                sharedHeadReady = true;
            }
        }
        const uint32_t target = DispatchBatchLimit(step, dispatchCvs - nextDispatch,
                                                   sharedHeadReady && nextPackRound >= args.config.packRoundCount);
        const Chunk first = LoadPackChunk(args, nextDispatch, peer);
        uint32_t count = 1U;
        uint64_t rows = first.rows;
        while (count < target) {
            const Chunk next = LoadPackChunk(args, nextDispatch + count, peer);
            if (!AdjacentChunks(first, rows, count, next) || (rows + next.rows) * h > kMaxPutBytes) break;
            rows += next.rows;
            ++count;
        }
        const uint32_t end = nextDispatch + count;
        // Startup/the previous iteration already prepared our current quota.
        // This batch may depend on the previous pending quota: publish before
        // polling PackReady. Keep the six event credits/cursor across scalar PUTs.
        pack.FlushReady();
        TraceMeta meta = ChunkTraceMeta(first);
        meta.rows = rows;
        start = timer.TraceWaitBegin();
        for (uint32_t cv = nextDispatch; cv < end; ++cv) {
            const Chunk chunk = LoadPackChunk(args, cv, peer);
            while (ReadGmByPassDCache(PackReadyFlag(args, peer, cv)) < static_cast<int32_t>(chunk.rows)) {
                const uint64_t backoff = GetSystemCycle();
                while (GetSystemCycle() - backoff < 64U) {
                }
            }
        }

        if (!dispatchStarted) {
            timer.CoarseBegin(CoarsePhase::Dispatch);
            timer.Begin(Phase::Dispatch);
            dispatchStarted = true;
        }
        meta.bytes = rows * h;
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveXOffset + first.remoteRow * h,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedX + first.localRow * h), static_cast<uint32_t>(meta.bytes),
            peer);
        timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
        meta.bytes = rows * scales;
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveScaleOffset + first.remoteRow * scales,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedScale + first.localRow * scales),
            static_cast<uint32_t>(meta.bytes), peer);
        timer.TraceEnd(TraceEvent::DispatchPutScale, start, meta);
        meta.bytes = 0U;
        start = timer.TraceBegin();
        // Same-QP SO orders each CV epoch after both merged payload PUTs.
        // No per-batch quiet and no MTE flag write; flags are zeroed by the
        // receiver before its start handshake. Each slot is added exactly once.
        for (uint32_t cv = nextDispatch; cv < end; ++cv) {
            const uint64_t index = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + cv;
            PublishOrdered(symmetric, args.config.dispatchReadyOffset + index * kFlagBytes, peer, args.config.epoch);
        }
        timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
        dispatchBytes += rows * (h + scales);
        nextDispatch = end;
        ++step;
        if (nextDispatch == dispatchCvs) {
            // Submission-only endpoint must precede the final post-PUT Pack.
            timer.CoarseEnd(CoarsePhase::Dispatch);
            timer.End(Phase::Dispatch, dispatchBytes, true);
        }
        // Issue this round FIRST, then prepare future data while it is in flight.
        // Keep every original static quota, extending Pack when the next 8-CV
        // batch would otherwise outrun this owner's prepared prefix.
        const uint32_t packLimit =
            DispatchPackAfterLimit(count, nextDispatch, step, dispatchCvs, nextPackRound, args.config.packRoundCount);
        while (nextPackRound < packLimit) PackOneRound(args, nextPackRound++, owner, kAic, pack);
    }
    // Submission boundary only: Dispatch may still be in flight when this
    // owner starts Combine on the same QP. Do not introduce a quiet here.
    // A short/empty peer still owns quotas belonging to other destinations.
    while (nextPackRound < args.config.packRoundCount) PackOneRound(args, nextPackRound++, owner, kAic, pack);
    pack.Close();
    MarkPackWorkerComplete(args);
    WaitRouteSummary(args, timer);
    // This owner has completed its original static Pack quota. Do not wait
    // for other owners: each Combine CV is protected by GMM2 readiness.

    const uint32_t workers = CommunicationWorkers(args.config.world);
    SharedActivationState shared(args, owner, timer);
    const uint32_t begin = RouteInfo(args)->recvCvBase[peer];
    const uint32_t end = RouteInfo(args)->recvCvBase[peer + 1U];
    const uint32_t selfBegin = RouteInfo(args)->recvCvBase[args.config.rank];
    const uint32_t selfEnd = RouteInfo(args)->recvCvBase[args.config.rank + 1U];
    // Remote CVs hold submitted ROW counts; self CVs hold copied SPAN counts.
    // Both become the submitted sentinel only after the full CV ready flag.
    // Each row is written once, with no GM reuse before final quiet.
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kCommunicationSubmittedOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> selfCopied(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> remoteSpanCopied(TPosition::VECCALC, kRemoteReorderBitmapOffset, kPeerReorderBitmapWords);
    LocalTensor<uint32_t> selfSpanCopied(TPosition::VECCALC, kSelfReorderBitmapOffset, kPeerReorderBitmapWords);
    const auto remoteSpans = PeerReorderSpanRange(args, begin, end);
    const auto selfSpans = PeerReorderSpanRange(args, selfBegin, selfEnd);
    ClearReorderBitmap(remoteSpanCopied, remoteSpans.spanCount);
    ClearReorderBitmap(selfSpanCopied, selfSpans.spanCount);
    uint32_t remaining = end - begin;
    uint32_t selfRemaining = 0U;
    for (uint32_t j = 0U; j < remaining; ++j) submitted.SetValue(j, 0U);
    for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
        selfCopied.SetValue(j, 0U);
        ++selfRemaining;
    }
    bool sentComplete = false;
    bool finishedPublished = false;
    bool selfComplete = false;
    bool waiting = false;
    uint64_t waitStart = 0U;
    while (!sentComplete || !selfComplete || !shared.Done()) {
        bool progress = false;
        for (uint32_t j = 0U; j < end - begin; ++j) {
            uint32_t sentRows = submitted.GetValue(j);
            if (sentRows == kReorderCvSubmitted) continue;
            const Chunk chunk = RuntimeReceiveChunkByIndex(args, begin + j);
            const PeerReorderRange range{begin + j, 0U};
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, remoteSpans.spanBegin, remoteSpanCopied,
                                                            timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            const uint32_t readyRows = ReorderedPeerPrefixRows(args, range, remoteSpans.spanBegin, remoteSpanCopied);
            while (sentRows < chunk.rows) {
                const uint32_t left = chunk.rows - sentRows;
                const uint32_t putRows = left < kCombinePutRows ? left : kCombinePutRows;
                if (sentRows + putRows > readyRows) break;
                auto meta = ChunkTraceMeta(chunk);
                meta.row = chunk.localRow + sentRows;
                meta.rows = putRows;
                meta.bytes = putRows * rowBytes;
                timer.CoarseBegin(CoarsePhase::Combine);
                timer.Begin(Phase::Combine);
                const uint64_t start = timer.TraceBegin();
                Put(symmetric, args.config.returnYOffset + (chunk.remoteRow + sentRows) * rowBytes,
                    reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + (chunk.localRow + sentRows) * rowBytes),
                    static_cast<uint32_t>(meta.bytes), peer);
                timer.TraceEnd(TraceEvent::CombinePut, start, meta);
                timer.End(Phase::Combine, meta.bytes, true);
                sentRows += putRows;
            }
            submitted.SetValue(j, sentRows);
            if (sentRows != chunk.rows) continue;
            // Payloads may leave in 128-row pieces, but Reduce still sees one
            // ready flag per original CV (256 rows by default). Same-QP SO
            // orders the flag after EVERY preceding data PUT, without quiet.
            auto meta = ChunkTraceMeta(chunk);
            meta.bytes = 0U;
            timer.Begin(Phase::Combine);
            const uint64_t start = timer.TraceBegin();
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            PublishOrdered(symmetric, args.config.combineReadyOffset + ready * kFlagBytes, peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
            timer.End(Phase::Combine, 0U, true);
            submitted.SetValue(j, kReorderCvSubmitted);
            --remaining;
            progress = true;
        }
        if (remaining == 0U && !finishedPublished) {
            if (waiting) {
                waiting = false;
            }
            // Publish remote finished promptly, including empty edges. Defer
            // the blocking quiet until this owner's self copies are complete.
            const uint64_t publishStart = timer.TraceBegin();
            PublishOrdered(symmetric, PeerPhaseSlot(args, 1U, args.config.rank), peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, publishStart, peerMeta);
            finishedPublished = true;
            progress = true;
        }
        for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
            const uint32_t copied = selfCopied.GetValue(j);
            if (copied == kReorderCvSubmitted) continue;
            const Chunk chunk = RuntimeReceiveChunkByIndex(args, selfBegin + j);
            const auto range = LoadPeerReorderRange(args, selfBegin + j);
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, selfSpans.spanBegin, selfSpanCopied,
                                                            timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            selfCopied.SetValue(j, copied + advanced);
            if (copied + advanced != range.spanCount) continue;
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            // Reorder drained local stores. Self requires no URMA operation.
            WriteGmByPassDCache(
                reinterpret_cast<__gm__ uint64_t*>(symmetric + args.config.combineReadyOffset + ready * kFlagBytes),
                args.config.epoch);
            selfCopied.SetValue(j, kReorderCvSubmitted);
            --selfRemaining;
            progress = true;
        }
        if (selfRemaining == 0U && !selfComplete) {
            MarkReorderComplete(args);
            selfComplete = true;
            progress = true;
        }
        if (finishedPublished && selfComplete && !sentComplete) {
            if (waiting) {
                waiting = false;
            }
            // Return to Reduce without blocking on outgoing traffic. The caller
            // performs the mandatory final quiet after its local Reduce quota.
            timer.CoarseEnd(CoarsePhase::Combine);
            sentComplete = true;
            progress = true;
        }
        if (!shared.Done()) {
            SharedWaitScope before{timer, waiting, waitStart, peerMeta, TraceEvent::WaitCombineReady};
            progress = shared.TryOne(before) || progress;
        }

        incoming.Poll();
        if (!progress) {
            if (!waiting) {
                waitStart = timer.TraceWaitBegin();
                waiting = true;
            }
            const uint64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 200U) {
            }
        }
    }
}

__aicore__ inline void RunProfiledCompute(const Arguments& args, uint32_t block, Timer& timer)
{
    // Select once, outside the loop. NoopTimer has no state, clocks, stores,
    // cache operations or fences; both coarse and formal use this instantiation.
    if (args.coarseTrace != 0U || (args.trace == 0U && args.timer == 0U)) {
        const auto phase = g_coreType == AIC ? CoarsePhase::Gmm : CoarsePhase::Dequant;
        timer.CoarseBegin(phase);
        NoopTimer disabled;
        RunCubeAndPrologue(args, block, disabled);
        timer.CoarseEnd(phase);
    } else {
        RunCubeAndPrologue(args, block, timer);
    }
}

__aicore__ inline void ProcessRouted(Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity, args.coarseTrace != 0U);
    PrepareDeviceRoutes(args, timer);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    const int64_t originalOverflow =
        GetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block < communicationWorkers) {
            PublishPeerStart(args, block, timer);
        }
        timer.CoarseBegin(CoarsePhase::Pack);
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        for (uint32_t round = 0U; round < kPackStartupRounds; ++round) {
            PackOneRound(args, round, worker, kLocalWorkers, timer);
        }
    }
    if constexpr (g_coreType == AIV) timer.CoarseEnd(CoarsePhase::Pack);
    // No CV loop contains a rank-wide or local full-core synchronization.
    TraceLocalSync(timer, 0U);
    if constexpr (g_coreType == AIC) {
        WaitRouteSummary(args, timer);
        RunProfiledCompute(args, block, timer);
        MarkCubeComplete(args);
    } else if (GetSubBlockIdx() == 0U) {
        RunAiv0RouteSummary(args, block, timer);
        WaitRouteSummary(args, timer);
        RunProfiledCompute(args, block, timer);
    } else {
        SharedIncoming incoming{args, block};
        RunCooperativeAiv1(args, block, timer, incoming);
    }
    if constexpr (g_coreType == AIV) {
        // Only this engine releases its phase-private UB/events. Other
        // compute and communication engines may still be producing rows.
        PipeBarrier<PIPE_ALL>();
        const uint32_t worker = ReduceWorkerIndex(block, GetSubBlockIdx(), args.config.world);
        if (worker < ReduceWorkers(args.config.world)) {
            timer.Begin(Phase::ReducePhase);
            timer.CoarseBegin(CoarsePhase::Reduce);
            if (UseLegacyReduce(args.config))
                Reduce(args, worker, timer);
            else
                ReduceGeneric(args, worker, ReduceWorkers(args.config.world), timer);
            timer.CoarseEnd(CoarsePhase::Reduce);
            const uint64_t owned = ReduceOwnedRows(args.config.bs, worker, args.config.world);
            timer.End(Phase::ReducePhase, owned * args.config.hidden * sizeof(bfloat16_t));
            PipeBarrier<PIPE_ALL>();
        }
    }
    // A local MIX join, not a rank barrier: every reducer, peer owner and AIC
    // must finish before returning and reusing this invocation's workspaces.
    TraceLocalSync(timer, 3U);
    timer.FinishTrace();
    SetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>(originalOverflow);
}

template <class T>
__aicore__ inline void RunSharedComputeSequence(const Arguments& a, uint32_t block, T& timer)
{
    RunSharedGmm<true>(a, block, timer);
    if constexpr (g_coreType == AIC) {
        // SharedMxGmm (including its destructor) no longer owns L1/events.
        WriteGmByPassDCache(LocalCompletionFlag(a, kLocalFlagSharedL1Released + block), 1);
        // RunSharedGmm has drained this core's shared GMM1 output and retired
        // its context. Count also cores with zero assigned shared tiles.
        // Do not wait for routed compute/shared GMM2: that would create a cycle
        // with the remaining Dispatch data needed by the routed experts.
        if (a.config.bs < 4096U) AtomicAdd(LocalCompletionFlag(a, kLocalFlagSharedGmm1Done), 1);
    }
    RunCubeAndPrologue(a, block, timer);
    if constexpr (g_coreType == AIC) MarkCubeComplete(a);  // routed self-return is ready before Shared GMM2
    // AIV0 must finish/release routed B slots before the W8 AIC context
    // initializes those same L1 addresses. AIV0 then proceeds to Reduce.
    WaitComputeFlag(a.localFlags, kLocalFlagRoutedPrologueReleased + block, 1);
    RunSharedGmm<false>(a, block, timer);
}
__aicore__ inline void ProcessShared(Arguments& a)
{
    Timer timer;
    timer.Init(a.timer, a.trace, a.traceCapacity, a.coarseTrace != 0);
    timer.SharedMode();
    PrepareDeviceRoutes(a, timer);
    const uint32_t block = GetBlockIdx() / GetTaskRation(), comm = CommunicationWorkers(a.config.world);
    const int64_t overflow = GetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>(0);
    if constexpr (g_coreType == AIC) {
        WaitRouteSummary(a, timer);
        timer.CoarseBegin(CoarsePhase::Gmm);
        if (a.coarseTrace || (!a.trace && !a.timer)) {
            NoopTimer t;
            RunSharedComputeSequence(a, block, t);
        } else
            RunSharedComputeSequence(a, block, timer);
        timer.CoarseEnd(CoarsePhase::Gmm);
    } else if (GetSubBlockIdx() == 0) {
        RunAiv0RouteSummary(a, block, timer);
        WaitRouteSummary(a, timer);
        timer.CoarseBegin(CoarsePhase::Pack);
        PackBatch startup(a, timer);
        for (uint32_t round = 0; round < kPackStartupRounds; ++round)
            PackOneRound(a, round, LocalWorkerIndex(block, 0U), kLocalWorkers, startup);
        startup.Close();
        MarkPackWorkerComplete(a);
        timer.CoarseEnd(CoarsePhase::Pack);
        // AIC may still be in shared GMM1 while this AIV0 finishes Pack.
        // Do not initialize the routed W4 B ring until its L1 owner releases it.
        WaitComputeFlag(a.localFlags, kLocalFlagSharedL1Released + block, 1);
        timer.CoarseBegin(CoarsePhase::Dequant);
        if (a.coarseTrace || (!a.trace && !a.timer)) {
            NoopTimer t;
            RunCubeAndPrologue(a, block, t);
        } else
            RunCubeAndPrologue(a, block, timer);
        timer.CoarseEnd(CoarsePhase::Dequant);
        // This AIV0 has finished its own dequant contexts. Release its local
        // UB/event use before Reduce; do not wait for other AIC/AIV workers.
        PipeBarrier<PIPE_ALL>();
        WriteGmByPassDCache(LocalCompletionFlag(a, kLocalFlagRoutedPrologueReleased + block), 1);
        NoReduceObserver observer;  // incoming counters remain AIV1-owned
        timer.CoarseBegin(CoarsePhase::Reduce);
        ReduceGeneric<true>(a, LocalWorkerIndex(block, 0U), kLocalWorkers, timer, observer);
        timer.CoarseEnd(CoarsePhase::Reduce);
    } else {
        if (block < comm) PublishPeerStart(a, block, timer);
        timer.CoarseBegin(CoarsePhase::Pack);
        PackBatch pack(a, timer);
        for (uint32_t round = 0; round < kPackStartupRounds; ++round)
            PackOneRound(a, round, LocalWorkerIndex(block, 1U), kLocalWorkers, pack);
        pack.Close();
        timer.CoarseEnd(CoarsePhase::Pack);
        SharedIncoming incoming{a, block};
        RunCooperativeAiv1(a, block, timer, incoming);
        PipeBarrier<PIPE_ALL>();
        timer.CoarseBegin(CoarsePhase::Reduce);
        ReduceGeneric<true>(a, LocalWorkerIndex(block, 1U), kLocalWorkers, timer, incoming);
        timer.CoarseEnd(CoarsePhase::Reduce);
        if (block < comm) {
            const auto start = timer.TraceBegin();
            Quiet(PeerForOwner(a.config.rank, block));
            TraceMeta completion;
            completion.peer = PeerForOwner(a.config.rank, block);
            timer.TraceEnd(TraceEvent::CombineSendComplete, start, completion);
            while (!ObserveIncomingCompletion(a, block, incoming.peer, incoming.self)) {
                const auto t = GetSystemCycle();
                while (GetSystemCycle() - t < 200U) {
                }
            }
        }
    }
    TraceLocalSync(timer, 3U);
    if constexpr (g_coreType == AIV) {
        timer.CoarseBegin(CoarsePhase::SharedSum);
        SharedAdd(a, LocalWorkerIndex(block, GetSubBlockIdx()), timer);
        timer.CoarseEnd(CoarsePhase::SharedSum);
    }
    const auto join = timer.TraceBegin();
    SyncAll<false>();
    TraceMeta m;
    m.cv = 4;
    timer.TraceEnd(TraceEvent::LocalSync, join, m);
    timer.FinishTrace();
    SetCtrlSpr<UbxMegaMoeImpl::OVERFLOW_MODE_CTRL, UbxMegaMoeImpl::OVERFLOW_MODE_CTRL>(overflow);
}

}  // namespace MegaMoeUrma
