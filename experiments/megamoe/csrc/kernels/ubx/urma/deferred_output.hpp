// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"

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
