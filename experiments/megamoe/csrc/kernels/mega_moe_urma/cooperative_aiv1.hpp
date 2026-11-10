// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/shared.hpp"
#include "kernels/mega_moe_urma/peer_reorder_progress.hpp"
#include "kernels/mega_moe_urma/consumer_completion.hpp"
#include "kernels/mega_moe_urma/completion_batch.hpp"
#include "kernels/mega_moe_urma/activation_ub.hpp"
#include "kernels/mega_moe_urma/reorder.hpp"
#include "kernels/mega_moe_urma/communication_schedule.hpp"

namespace MegaMoeUrma {
// Persistent per-peer send/self-ready counters use high UB only after this
// owner's Pack is finished. Paired tile Reorder uses low 128KiB temporarily.
struct CooperativeUb {
    static constexpr uint32_t submitted = kAiv1PersistentUbBase;
    static constexpr uint32_t selfCounts = submitted + kMaxPeerChunks * 4U;
};
static_assert(CooperativeUb::selfCounts + kMaxPeerChunks * 4U <= kRequiredUbBytes);

// A contiguous row prefix is safe only after every N tile of each intersecting
// M-group has reached its final layout. Counters are published after MTE3 drain.
__aicore__ inline uint32_t PairedReorderedPrefix(const Arguments& a, const Chunk& c)
{
    RuntimePeerSpans cursor(a, c);
    PeerReorderSpan span{};
    uint32_t bit, rows = 0U;
    const auto required = static_cast<int32_t>(a.config.hidden / kComputeTileN);
    while (cursor.Next(span, bit)) {
        if (ReadGmByPassDCache(ComputeFlag(a.gmm2Done, span.fragmentIndex)) < required) break;
        rows += span.rows;
    }
    return rows;
}
struct DirectReorderWork {
    const Arguments& a;
    PackBatch& pack;
    uint32_t core;
    UbRoutedActivation& activation;
    template <class TimerT>
    __aicore__ inline void Activate(const RoutedCompletionItem& item, TimerT& timer)
    {
        activation.Run(a, item.fragment, item.group, item.tile, timer);
    }
    __aicore__ inline void Before()
    {
        pack.Close();
    }
    template <class TimerT>
    __aicore__ inline void Reorder(const RoutedCompletionItem& item, TimerT& timer)
    {
        ReorderGmm2Tile(a, item.fragment, item.group, item.tile, timer);
    }
};
struct CooperativeNoWait {
    __aicore__ inline void Before() {}
};

__aicore__ inline TraceMeta CooperativeChunkMeta(const Chunk& c, uint32_t offset, uint32_t rows)
{
    TraceMeta m;
    m.peer = c.peer;
    m.cv = c.cv;
    m.row = c.localRow + offset;
    m.rows = rows;
    return m;
}

// One QP owner advances at most one Dispatch CV or one remote/self reorder CV
// per turn. No blocking PeerStart/PackReady wait and no per-CV quiet.
class CooperativePeer {
public:
    __aicore__ CooperativePeer(Arguments& a, uint32_t core, Timer& timer, DispatchReceiveObserver& observer)
        : a_(a),
          core_(core),
          timer_(timer),
          dispatchObserver_(observer),
          active_(core < CommunicationWorkers(a.config.world)),
          peer_(active_ ? PeerForOwner(a.config.rank, core) : 0U),
          submitted_(TPosition::VECCALC, CooperativeUb::submitted, kMaxPeerChunks),
          selfCounts_(TPosition::VECCALC, CooperativeUb::selfCounts, kMaxPeerChunks)
    {
        dispatchCount_ = active_ ? RuntimeSendCvCount(a, peer_) : 0U;
    }
    __aicore__ inline bool Active() const
    {
        return active_;
    }
    __aicore__ inline bool DispatchDone() const
    {
        return nextDispatch_ == dispatchCount_;
    }
    __aicore__ inline bool Done() const
    {
        return !active_ || (initialized_ && finished_ && selfFinished_);
    }

    __aicore__ inline bool TryDispatch(PackBatch& pack, bool packDone)
    {
        if (!active_ || !TryStart() || DispatchDone()) return false;
        auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(a_.symmetric);
        if (a_.sharedExperts && a_.config.bs < 4096U && !sharedHeadReady_) {
            sharedHeadReady_ =
                ReadGmByPassDCache(LocalCompletionFlag(a_, kLocalFlagSharedGmm1Done)) == static_cast<int32_t>(kAic);
            const uint32_t startup = RouteMax(1U, a_.config.bs / 1024U);
            if (!sharedHeadReady_ && dispatchSteps_ >= startup) return false;
        }
        const Chunk c = RuntimeSendChunk(a_, peer_, nextDispatch_);
        pack.FlushReady();
        if (ReadGmByPassDCache(PackReadyFlag(a_, peer_, c.cv)) < static_cast<int32_t>(c.rows)) return false;
        // Retain S1's 1/1/2/4/8 ramp and merged-batch startup limit. Bound a
        // cooperative turn to eight CVs so FIX consumption gets a fair chance.
        const bool unrestricted = packDone && (sharedHeadReady_ || a_.config.bs >= 4096U);
        const uint32_t target =
            a_.sharedExperts
                ? RouteMin(8U, DispatchBatchLimit(dispatchSteps_, dispatchCount_ - nextDispatch_, unrestricted))
                : 1U;
        uint32_t count = 1U, rows = c.rows;
        while (count < target) {
            const auto next = RuntimeSendChunk(a_, peer_, nextDispatch_ + count);
            if (!AdjacentChunks(c, rows, count, next) || uint64_t(rows + next.rows) * a_.config.hidden > kMaxPutBytes ||
                ReadGmByPassDCache(PackReadyFlag(a_, peer_, next.cv)) < static_cast<int32_t>(next.rows))
                break;
            rows += next.rows;
            ++count;
        }
        timer_.CoarseBegin(CoarsePhase::Dispatch);
        timer_.Begin(Phase::Dispatch);
        auto m = CooperativeChunkMeta(c, 0U, rows);
        m.bytes = uint64_t(rows) * a_.config.hidden;
        auto start = timer_.TraceBegin();
        Put(symmetric, a_.config.receiveXOffset + c.remoteRow * a_.config.hidden,
            reinterpret_cast<__gm__ uint8_t*>(a_.expandedX + c.localRow * a_.config.hidden),
            static_cast<uint32_t>(m.bytes), peer_);
        timer_.TraceEnd(TraceEvent::DispatchPutX, start, m);
        const uint32_t scales = a_.config.hidden / 32U;
        m.bytes = uint64_t(rows) * scales;
        start = timer_.TraceBegin();
        Put(symmetric, a_.config.receiveScaleOffset + c.remoteRow * scales,
            reinterpret_cast<__gm__ uint8_t*>(a_.expandedScale + c.localRow * scales), static_cast<uint32_t>(m.bytes),
            peer_);
        timer_.TraceEnd(TraceEvent::DispatchPutScale, start, m);
        m.bytes = 0;
        start = timer_.TraceBegin();
        for (uint32_t cv = c.cv; cv < c.cv + count; ++cv)
            PublishOrdered(
                symmetric,
                a_.config.dispatchReadyOffset + (uint64_t(a_.config.rank) * a_.config.maxPeerChunks + cv) * kFlagBytes,
                peer_, a_.config.epoch);
        timer_.TraceEnd(TraceEvent::DispatchPublish, start, m);
        timer_.End(Phase::Dispatch, uint64_t(rows) * (a_.config.hidden + scales), true);
        nextDispatch_ += count;
        ++dispatchSteps_;
        if (DispatchDone()) timer_.CoarseEnd(CoarsePhase::Dispatch);
        return true;
    }

    __aicore__ inline bool TryCombine(bool packDone)
    {
        // Even an empty outgoing Dispatch edge must observe the receiver's
        // reset/start before sending Combine or the empty finished marker.
        if (!active_ || !packDone || !DispatchDone() || !TryStart()) return false;
        if (!initialized_) InitReceive();
        bool progress = TryRemote();
        progress = TrySelf() || progress;
        if (remaining_ == 0U && !finished_) {
            const uint64_t start = timer_.TraceBegin();
            PublishOrdered(reinterpret_cast<__gm__ uint8_t*>(a_.symmetric), PeerPhaseSlot(a_, 1U, a_.config.rank),
                           peer_, a_.config.epoch);
            TraceMeta m;
            m.peer = peer_;
            timer_.TraceEnd(TraceEvent::CombinePublish, start, m);
            finished_ = true;
            progress = true;
            if (a_.sharedExperts) timer_.CoarseEnd(CoarsePhase::Combine);
        }
        if (selfRemaining_ == 0U && !selfFinished_) {
            MarkReorderComplete(a_);
            selfFinished_ = true;
            progress = true;
        }
        return progress;
    }

    __aicore__ inline void FinishRouted(SharedIncoming& incoming)
    {
        if (!active_) return;
        const uint64_t start = timer_.TraceBegin();
        Quiet(peer_);
        incoming.dispatch.Poll();  // Final nonblocking receive observation; quiet is not receive evidence.
        TraceMeta m;
        m.peer = peer_;
        timer_.TraceEnd(TraceEvent::CombineSendComplete, start, m);
        timer_.CoarseEnd(CoarsePhase::Combine);
        if (receives_) timer_.MarkSendComplete(Phase::Combine);
        FinishCombinePeer(a_, core_, incoming.peer, incoming.self);
    }

private:
    __aicore__ inline bool TryStart()
    {
        if (started_) return true;
        auto* flag = reinterpret_cast<__gm__ uint64_t*>(a_.symmetric + PeerPhaseSlot(a_, 0U, peer_));
        started_ = ReadGmByPassDCache(flag) >= a_.config.epoch;
        return started_;
    }
    __aicore__ inline void InitReceive()
    {
        const auto* r = RouteInfo(a_);
        begin_ = r->recvCvBase[peer_];
        receives_ = r->recvCvBase[peer_ + 1U] - begin_;
        dispatchObserver_.Arm(a_, peer_, r->incoming[peer_][a_.config.localExperts]);
        selfBegin_ = r->recvCvBase[a_.config.rank];
        selfCount_ = r->recvCvBase[a_.config.rank + 1U] - selfBegin_;
        remaining_ = receives_;
        selfRemaining_ = 0U;
        nextSelf_ = core_;
        for (uint32_t i = 0; i < receives_; ++i) submitted_.SetValue(i, 0U);
        for (uint32_t i = 0; i < selfCount_; ++i) selfCounts_.SetValue(i, 0U);
        for (uint32_t i = core_; i < selfCount_; i += CommunicationWorkers(a_.config.world)) ++selfRemaining_;
        initialized_ = true;
    }
    __aicore__ inline bool TryRemote()
    {
        if (!remaining_) return false;
        for (uint32_t skipped = 0; skipped < receives_; ++skipped) {
            const uint32_t j = nextRemote_;
            nextRemote_ = (nextRemote_ + 1U) % receives_;
            uint32_t sent = submitted_.GetValue(j);
            if (sent == kReorderCvSubmitted) continue;
            const Chunk c = RuntimeReceiveChunkByIndex(a_, begin_ + j);
            const uint32_t ready = PairedReorderedPrefix(a_, c);
            const uint32_t n = RouteMin(kCombinePutRows, c.rows - sent);
            if (sent + n > ready) return false;
            timer_.CoarseBegin(CoarsePhase::Combine);
            timer_.Begin(Phase::Combine);
            auto m = CooperativeChunkMeta(c, sent, n);
            m.bytes = uint64_t(n) * a_.config.hidden * 2U;
            auto start = timer_.TraceBegin();
            Put(reinterpret_cast<__gm__ uint8_t*>(a_.symmetric),
                a_.config.returnYOffset + (c.remoteRow + sent) * a_.config.hidden * 2U,
                reinterpret_cast<__gm__ uint8_t*>(a_.gmm2Output + (c.localRow + sent) * a_.config.hidden * 2U),
                static_cast<uint32_t>(m.bytes), peer_);
            timer_.TraceEnd(TraceEvent::CombinePut, start, m);
            timer_.End(Phase::Combine, m.bytes, true);
            sent += n;
            submitted_.SetValue(j, sent);
            if (sent == c.rows) {
                start = timer_.TraceBegin();
                m.bytes = 0U;
                PublishOrdered(reinterpret_cast<__gm__ uint8_t*>(a_.symmetric),
                               a_.config.combineReadyOffset +
                                   (uint64_t(a_.config.rank) * a_.config.maxPeerChunks + c.cv) * kFlagBytes,
                               peer_, a_.config.epoch);
                timer_.TraceEnd(TraceEvent::CombinePublish, start, m);
                submitted_.SetValue(j, kReorderCvSubmitted);
                --remaining_;
            }
            return true;
        }
        return false;
    }
    __aicore__ inline bool TrySelf()
    {
        if (!selfRemaining_) return false;
        const uint32_t workers = CommunicationWorkers(a_.config.world);
        const uint32_t owned = (selfCount_ - 1U - core_) / workers + 1U;
        for (uint32_t skipped = 0; skipped < owned; ++skipped) {
            const uint32_t j = nextSelf_;
            nextSelf_ += workers;
            if (nextSelf_ >= selfCount_) nextSelf_ = core_;
            const uint32_t before = selfCounts_.GetValue(j);
            if (before == kReorderCvSubmitted) continue;
            const Chunk c = RuntimeReceiveChunkByIndex(a_, selfBegin_ + j);
            if (PairedReorderedPrefix(a_, c) != c.rows) return false;
            WriteGmByPassDCache(reinterpret_cast<__gm__ uint64_t*>(
                                    a_.symmetric + a_.config.combineReadyOffset +
                                    (uint64_t(a_.config.rank) * a_.config.maxPeerChunks + c.cv) * kFlagBytes),
                                a_.config.epoch);
            selfCounts_.SetValue(j, kReorderCvSubmitted);
            --selfRemaining_;
            return true;
        }
        return false;
    }
    Arguments& a_;
    uint32_t core_;
    Timer& timer_;
    DispatchReceiveObserver& dispatchObserver_;
    bool active_;
    uint32_t peer_;
    bool started_ = false, initialized_ = false, finished_ = false, selfFinished_ = false, sharedHeadReady_ = false;
    uint32_t dispatchCount_ = 0U, nextDispatch_ = 0U, dispatchSteps_ = 0U, begin_ = 0U, receives_ = 0U, selfBegin_ = 0U,
             selfCount_ = 0U;
    uint32_t remaining_ = 0U, selfRemaining_ = 0U, nextRemote_ = 0U, nextSelf_ = 0U;
    LocalTensor<uint32_t> submitted_, selfCounts_;
};

__aicore__ inline void RunCooperativeAiv1(Arguments& a, uint32_t core, Timer& timer, SharedIncoming& incoming)
{
    incoming.dispatch.Init(timer);
    RoutedCompletionConsumer completion(a, core);
    completion.Init();  // even zero-task AICs need the single credit
    PackBatch pack(a, timer);
    CooperativePeer peer(a, core, timer, incoming.dispatch);
    SharedActivationState shared(a, core, timer);
    if (!peer.Active()) timer.CoarseBegin(CoarsePhase::Activation);
    UbRoutedActivation activation;
    activation.Init(a);
    const uint32_t comm = CommunicationWorkers(a.config.world), helpers = kAic - comm;
    uint32_t nextPack = kPackStartupRounds;
    uint32_t nextGather = peer.Active() ? UINT32_MAX : core - comm;
    bool ready = false, packDone = false;
    DirectReorderWork directReorder{a, pack, core, activation};
    while (true) {
        bool progress = false;
        // A bounded Pack quantum each turn, never a Pack-until-empty loop.
        if (!packDone && nextPack < a.config.packRoundCount) {
            PackOneRound(a, nextPack++, core, kAic, pack);
            progress = true;
        }
        if (!packDone && nextPack >= a.config.packRoundCount) {
            pack.Close();
            MarkPackWorkerComplete(a);
            packDone = true;
            progress = true;
        }
        if (!ready && RouteSummaryReady(a)) {
            AcquireRouteSummary(a);
            ready = true;
            progress = true;
        }
        if (peer.Active())
            progress = peer.TryDispatch(pack, packDone) || progress;
        else if (ready && nextGather < a.config.fragmentCount) {
            const auto f = LoadFragment(a, nextGather);
            if (AreGroupInputsReady(a, f)) {
                pack.Close();
                GatherInputScale(a, f, nextGather, timer);
                nextGather += helpers;
                progress = true;
            }
        }
        // The single RX mailbox must be consumed in exact producer order.
        // Never wait for a later GMM1 while an earlier GMM2 owns the mailbox.
        if (ready) progress = (ConsumeIssuedReorders(completion, directReorder, timer) != 0U) || progress;
        if (ready && packDone) progress = peer.TryCombine(true) || progress;
        incoming.Poll();
        if (ready && !progress && !shared.Done()) {
            pack.Close();
            CooperativeNoWait before;
            progress = shared.TryOne(before);
        }
        if (ready && packDone && peer.Done() && completion.Done() && nextGather >= a.config.fragmentCount &&
            shared.Done())
            break;
        if (!progress) {
            const uint64_t tick = GetSystemCycle();
            while (GetSystemCycle() - tick < 64U) {
            }
        }
    }
    pack.Close();
    if (!a.sharedExperts) peer.FinishRouted(incoming);
    if (!peer.Active()) timer.CoarseEnd(CoarsePhase::Activation);
    // S1 preserves the outer Reduce -> final quiet/finished-observe ordering.
}
}  // namespace MegaMoeUrma
