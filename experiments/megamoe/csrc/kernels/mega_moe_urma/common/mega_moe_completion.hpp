// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernel_operator.h"
#include <c_api/asc_simd.h>
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_route_views.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_timer.hpp"

// FIX ownership, task cursors and completion consumption.

namespace MegaMoeUrma {
// Protocol source: Yancy/DeepEP (source revision was not recorded).
// AIC and its paired AIV1 exchange ownership of the FIX output mailbox.
// B supply uses AIV0 full4..7, empty8..11; these channels use AIV1 full12,
// empty14. One low-UB mailbox stays owned until activation/scatter completes.
struct RoutedFixChannel {
    static constexpr uint32_t Slots = 1U, Full = 12U, Empty = 14U, Aiv1Target = 16U;
    __aicore__ static inline void InitConsumer()
    {
        for (uint32_t i = 0; i < Slots; ++i) asc_sync_intra_arrive(PIPE_MTE3, Aiv1Target | (Empty + i));
    }
    __aicore__ static inline void Acquire(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_FIX, Aiv1Target | (Empty + sequence % Slots));
    }
    __aicore__ static inline void Publish(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_FIX, Aiv1Target | (Full + sequence % Slots));
    }
    __aicore__ static inline void Wait(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_MTE2, Full + sequence % Slots);
        // S must not publish a ready flag before the MTE2 hardware wait passes.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0U));
    }
    __aicore__ static inline void Release(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_MTE3, Aiv1Target | (Empty + sequence % Slots));
    }
    __aicore__ static inline void DrainProducer()
    {
        for (uint32_t i = 0; i < Slots; ++i) asc_sync_intra_wait(PIPE_FIX, Aiv1Target | (Empty + i));
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(static_cast<AscendC::TEventID>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(static_cast<AscendC::TEventID>(0U));
    }
};

}  // namespace MegaMoeUrma

#if defined(__NPU_DEVICE__)
#define URMA_CURSOR_INLINE __aicore__ inline
#else
#define URMA_CURSOR_INLINE inline
#endif
namespace MegaMoeUrma {
struct RoutedCompletionItem {
    bool gmm1 = false;
    uint32_t group = 0U, tile = 0U;
    uint64_t logicalTile = 0U;
    Fragment fragment{};
};

// Production ownership iterator, independent of storage access. Tests compare
// this cursor against a naive complete-task enumeration, including idle cores.
template <class Plan>
class OwnedTileSequence {
public:
    URMA_CURSOR_INLINE OwnedTileSequence(const Plan& p, uint32_t core) : plan_(p), core_(core) {}
    URMA_CURSOR_INLINE bool Next(RoutedCompletionItem& item)
    {
        while (step_ < plan_.TaskCount()) {
            if (!active_) {
                const auto task = plan_.Task(step_);
                const auto batch = plan_.Batch(task.batchIndex);
                gmm1_ = task.stage == ComputeBatchStage::Gmm1;
                begin_ = batch.fragmentBegin;
                tiles_ = plan_.Tiles(gmm1_);
                limit_ = uint64_t(batch.fragmentCount) * tiles_;
                if (limit_ == 0U) {
                    ++step_;
                    continue;
                }
                const auto first = plan_.Group(begin_);
                const uint64_t base = gmm1_ ? first.gmm1TileBegin : first.gmm2TileBegin;
                next_ = FirstOwnedComputeTile(base, RoutedOwnerResidue(gmm1_, core_), kAic);
                active_ = true;
            }
            if (next_ < limit_) {
                item.gmm1 = gmm1_;
                item.group = begin_ + static_cast<uint32_t>(next_ / tiles_);
                item.tile = static_cast<uint32_t>(next_ % tiles_);
                item.fragment = plan_.Group(item.group);
                item.logicalTile = (gmm1_ ? item.fragment.gmm1TileBegin : item.fragment.gmm2TileBegin) + item.tile;
                next_ += kAic;
                return true;
            }
            active_ = false;
            ++step_;
        }
        return false;
    }

private:
    const Plan& plan_;
    uint32_t core_, step_ = 0U, begin_ = 0U, tiles_ = 0U;
    uint64_t next_ = 0U, limit_ = 0U;
    bool active_ = false, gmm1_ = false;
};
}  // namespace MegaMoeUrma
#undef URMA_CURSOR_INLINE

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_OUTPUT_INLINE __aicore__ inline
#else
#define URMA_OUTPUT_INLINE inline
#endif

// Backend::Arm captures only work already enqueued. At most one pending
// completion owns FIX_S1. Submit the next ready tile BEFORE AfterSubmit;
// flush BEFORE any blocking dependency wait and at kernel termination.
template <class Backend>
class DeferredOutput {
public:
    URMA_OUTPUT_INLINE explicit DeferredOutput(Backend& backend) : backend_(backend) {}

    URMA_OUTPUT_INLINE void BeforeDependency(bool ready)
    {
        if (!ready) Flush();
    }

    URMA_OUTPUT_INLINE void AfterSubmit(bool gmm1, uint64_t tile, const TraceMeta& meta)
    {
        Flush();
        backend_.Arm();
        pending_ = true;
        gmm1_ = gmm1;
        tile_ = tile;
        meta_ = meta;
    }

    URMA_OUTPUT_INLINE void Flush()
    {
        if (!pending_) return;
        backend_.Complete(gmm1_, tile_, meta_);
        pending_ = false;
    }

private:
    Backend& backend_;
    bool pending_ = false;
    bool gmm1_ = false;
    uint64_t tile_ = 0U;
    TraceMeta meta_{};
};

#undef URMA_OUTPUT_INLINE
}  // namespace MegaMoeUrma

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_TILE_INLINE __aicore__ inline
#else
#define URMA_TILE_INLINE inline
#endif

// One captured FIX fence. Same-type work may be submitted before retiring
// it. Type changes, unsatisfied input dependencies and final exit must Flush.
template <class Backend>
class TileCompletion {
public:
    URMA_TILE_INLINE explicit TileCompletion(Backend& backend) : backend_(backend) {}
    URMA_TILE_INLINE void BeforeStage(bool gmm1)
    {
        if (pending_ && gmm1_ != gmm1) Flush();
    }
    URMA_TILE_INLINE void BeforeDependency(bool ready)
    {
        if (!ready) Flush();
    }
    URMA_TILE_INLINE void AfterSubmit(bool gmm1, uint64_t tile, const TraceMeta& meta)
    {
        // The old fence was captured before the tile just submitted.
        Flush();
        backend_.Arm();
        pending_ = true;
        gmm1_ = gmm1;
        tile_ = tile;
        meta_ = meta;
    }
    URMA_TILE_INLINE void Flush()
    {
        if (!pending_) return;
        backend_.Complete(gmm1_, tile_, meta_);
        pending_ = false;
    }

private:
    Backend& backend_;
    bool pending_ = false, gmm1_ = false;
    uint64_t tile_ = 0U;
    TraceMeta meta_{};
};
#undef URMA_TILE_INLINE
}  // namespace MegaMoeUrma

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

#if defined(__NPU_DEVICE__)
#define URMA_COMPLETION_INLINE __aicore__ inline
#else
#define URMA_COMPLETION_INLINE inline
#endif

namespace MegaMoeUrma {
// Bound one cooperative turn to the single hardware UB ownership credit. An issued
// tile can still be waiting on hardware: never grow this batch while waiting,
// and never cross into GMM1 when a Reorder batch is already in progress.
constexpr uint32_t kDirectReorderBatch = 1U;
template <class Consumer, class Work, class TimerT>
URMA_COMPLETION_INLINE uint32_t ConsumeIssuedReorders(Consumer& completion, Work& work, TimerT& timer)
{
    const uint32_t available = completion.IssuedCount();
    if (!available) return 0U;
    const uint32_t limit = available < kDirectReorderBatch ? available : kDirectReorderBatch;
    work.Before();
    uint32_t count = 0U;
    while (count < limit) {
        if (count && completion.NextIsGmm1()) break;
        RoutedCompletionItem item;
        completion.CompleteIssuedOne(item, timer);  // real hardware wait; retain UB ownership
        ++count;
        if (item.gmm1)
            work.Activate(item, timer);
        else
            work.Reorder(item, timer);
        // Single RX mailbox stays owned through all V/MTE3 reads, including
        // GM-backed GMM1 activation, which also uses this low-UB workspace.
        completion.ReleaseCompleted();
        if (item.gmm1) break;
    }
    return count;
}
}  // namespace MegaMoeUrma
#undef URMA_COMPLETION_INLINE
