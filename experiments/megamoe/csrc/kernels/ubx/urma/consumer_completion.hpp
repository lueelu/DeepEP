// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"
#include "owned_compute_cursor.hpp"
#include "compute_contract.hpp"
#include "lazy_compute_routes.hpp"
#include "timer.hpp"
#include "routed_fix_channel.hpp"

namespace MegaMoeUrma {
struct DeviceRoutePlanView {
    const Arguments& a;
    __aicore__ inline uint32_t TaskCount() const
    {
        return a.config.computeTaskCount;
    }
    __aicore__ inline ComputeBatchTask Task(uint32_t i) const
    {
        const auto* p = reinterpret_cast<const __gm__ ComputeBatchTask*>(a.computeTasks) + i;
        return {p->stage, p->batchIndex};
    }
    __aicore__ inline ComputeBatch Batch(uint32_t i) const
    {
        const auto* p = reinterpret_cast<const __gm__ ComputeBatch*>(a.computeBatches) + i;
        return {p->fragmentBegin, p->fragmentCount};
    }
    __aicore__ inline uint32_t Tiles(bool g1) const
    {
        return g1 ? Gmm1ComputeTileCount(kComputeTileM, a.config) : ComputeTileCount(kComputeTileM, a.config.hidden);
    }
    __aicore__ inline Fragment Group(uint32_t group) const
    {
        return RuntimeFragment(a, group);
    }
};
class OwnedRouteTileCursor {
public:
    __aicore__ OwnedRouteTileCursor(const Arguments& a, uint32_t core) : plan_{a}, sequence_(plan_, core) {}
    __aicore__ inline bool Next(RoutedCompletionItem& item)
    {
        return sequence_.Next(item);
    }

private:
    DeviceRoutePlanView plan_;
    OwnedTileSequence<DeviceRoutePlanView> sequence_;
};
__aicore__ inline __gm__ int32_t* RoutedIssued(const Arguments& a, uint32_t core)
{
    return reinterpret_cast<__gm__ int32_t*>(a.gmmIssued + uint64_t(core) * kFlagBytes);
}

class RoutedCompletionProducer {
public:
    __aicore__ RoutedCompletionProducer(const Arguments& a, uint32_t core) : a_(a), core_(core) {}
    __aicore__ inline void Acquire()
    {
        RoutedFixChannel::Acquire(sequence_);
    }
    __aicore__ inline void Commit()
    {
        RoutedFixChannel::Publish(sequence_);
        // Issued only means the FIX full has been enqueued. The consumer must
        // wait for that hardware event before setting any compute-ready flag.
        AscendC::WriteGmByPassDCache(RoutedIssued(a_, core_), static_cast<int32_t>(++sequence_));
    }
    template <class TimerT>
    __aicore__ inline void Finish(TimerT& timer)
    {
        const auto start = timer.TraceBegin();
        RoutedFixChannel::DrainProducer();
        // AIC final completion remains visible in the unified GMM envelope.
        timer.TraceEnd(TraceEvent::Gmm2Drain, start);
    }

private:
    const Arguments& a_;
    uint32_t core_, sequence_ = 0U;
};

class RoutedCompletionConsumer {
public:
    __aicore__ RoutedCompletionConsumer(const Arguments& a, uint32_t core) : a_(a), core_(core), cursor_(a, core) {}
    __aicore__ inline void Init()
    {
        RoutedFixChannel::InitConsumer();
    }
    // Caller closes Pack/scale/Reorder DMA lifetimes before invoking this.
    // Route summary must be acquired before using cursor/Done/IssuedCount.
    __aicore__ inline bool Done()
    {
        Prepare();
        return exhausted_;
    }
    __aicore__ inline uint32_t IssuedCount()
    {
        Prepare();
        if (exhausted_) return 0U;
        if (consumed_ >= issued_) issued_ = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(RoutedIssued(a_, core_)));
        return consumed_ < issued_ ? issued_ - consumed_ : 0U;
    }
    __aicore__ inline bool NextIsGmm1()
    {
        Prepare();
        return !exhausted_ && next_.gmm1;
    }
    // IssuedCount captured the budget; this method does not poll GM again.
    template <class TimerT>
    __aicore__ inline void CompleteIssuedOne(RoutedCompletionItem& item, TimerT& timer)
    {
        Prepare();
        const auto start = timer.TraceBegin();
        RoutedFixChannel::Wait(consumed_);
        item = next_;
        // Both stages go directly to same-core UB consumers. There is no
        // intermediate per-tile GM-ready write. Activation/Reorder publish
        // their downstream group-ready counts after final MTE3 completion.
        ++consumed_;
        prepared_ = false;
        TraceMeta m;
        m.fragment = item.group;
        m.expert = item.fragment.expert;
        m.row = item.fragment.rowBegin;
        m.rows = item.fragment.rows;
        m.tileN = uint64_t(item.tile) * (item.gmm1 ? kGmm1LogicalTileN : kComputeTileN);
        m.columns = item.gmm1 ? kGmm1LogicalTileN : kComputeTileN;
        timer.TraceEnd(item.gmm1 ? TraceEvent::Gmm1TileDrain : TraceEvent::Gmm2Drain, start, m);
    }
    __aicore__ inline void ReleaseCompleted()
    {
        RoutedFixChannel::Release(consumed_ - 1U);
    }

private:
    __aicore__ inline void Prepare()
    {
        if (!prepared_ && !exhausted_) {
            exhausted_ = !cursor_.Next(next_);
            prepared_ = !exhausted_;
        }
    }
    const Arguments& a_;
    uint32_t core_, consumed_ = 0U, issued_ = 0U;
    OwnedRouteTileCursor cursor_;
    RoutedCompletionItem next_{};
    bool prepared_ = false, exhausted_ = false;
};
}  // namespace MegaMoeUrma
