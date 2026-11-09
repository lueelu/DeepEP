// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"
#include "compute_contract.hpp"
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
