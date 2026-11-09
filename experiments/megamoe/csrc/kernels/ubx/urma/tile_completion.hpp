// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"

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
