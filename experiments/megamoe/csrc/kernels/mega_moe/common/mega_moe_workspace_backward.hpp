// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once
#include "mega_moe_bf16_layout.hpp"
#include "mega_moe_bf16_wave_plan.hpp"
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#define MEGA_MOE_WORKSPACE_HD __host_aicore__
#else
#define MEGA_MOE_WORKSPACE_HD
#endif

namespace MegaMoeBF16Training {
// Byte offsets shared by Host allocation and Device addressing.
struct BackwardWorkspaceLayout {
    uint64_t doOffset, recomputeGateOffset, recomputeUpOffset;
    uint64_t dzOffset, hpOffset, dpOffset, dispatchReadyOffset, aivReadyOffset, hDoneOffset;
    uint64_t dhpReadyOffset, dzReadyOffset, dxReadyOffset, doReleasedOffset, progressOffset;
    uint64_t controlOffset, totalBytes;
    uint64_t stagedReadyOffset;
    bool gmDataGrad, recompute;
    MEGA_MOE_WORKSPACE_HD constexpr BackwardWorkspaceLayout(uint64_t e, uint64_t r, uint64_t h, uint64_t n,
                                                            uint64_t world, uint64_t producers, uint32_t splitExperts,
                                                            uint32_t savedMode)
        : doOffset(0),
          recomputeGateOffset(0),
          recomputeUpOffset(0),
          dzOffset(0),
          hpOffset(0),
          dpOffset(0),
          dispatchReadyOffset(0),
          aivReadyOffset(0),
          hDoneOffset(0),
          dhpReadyOffset(0),
          dzReadyOffset(0),
          dxReadyOffset(0),
          doReleasedOffset(0),
          progressOffset(0),
          controlOffset(0),
          totalBytes(0),
          stagedReadyOffset(0),
          gmDataGrad(UseGmForBackwardDataGrad(savedMode)),
          recompute(savedMode == kSaveNone)
    {
        uint64_t offset = AlignUp(2 * r * h, 512);
        recomputeGateOffset = offset;
        if (recompute) offset += AlignUp(r * n, 512);
        recomputeUpOffset = offset;
        if (recompute) offset += AlignUp(r * n, 512);
        dzOffset = offset;
        offset += AlignUp(2 * r * n, 512);
        hpOffset = offset;
        offset += AlignUp(r * n, 512);
        dpOffset = offset;
        offset += AlignUp(4 * r * BackwardTopkPartialCount(savedMode, n / 2), 512);
        controlOffset = offset;
        dispatchReadyOffset = offset;
        offset += AlignUp(4 * e * 16, 128);
        aivReadyOffset = offset;
        offset += AlignUp(4, 32);
        hDoneOffset = offset;
        offset += AlignUp(4, 32);
        dhpReadyOffset = offset;
        offset += AlignUp(4 * e * 16, 128);
        dzReadyOffset = offset;
        offset += AlignUp(4 * e * 16, 128);
        dxReadyOffset = offset;
        offset += AlignUp(4 * e * 16, 128);
        doReleasedOffset = offset;
        if (gmDataGrad) offset += AlignUp(4 * e * 16, 128);
        progressOffset = AlignUp(offset, 64);
        totalBytes = BackwardDispatchProgressLayout::TotalBytes(offset, splitExperts, world, producers, savedMode);
        if (splitExperts && UseBackwardStagedWave(savedMode, n / 2)) {
            stagedReadyOffset = AlignUp(totalBytes, 64);
            totalBytes = stagedReadyOffset + ((r + 127U) / 128U + e) * 64;
        }
    }
    template <typename Tiling>
    MEGA_MOE_WORKSPACE_HD explicit BackwardWorkspaceLayout(const Tiling& t, uint32_t splitExperts)
        : BackwardWorkspaceLayout(t.expertPerRank, t.maxOutputSize, t.k, t.n, t.epWorldSize, t.blockNumPerEP,
                                  splitExperts, t.savedGateUpMode)
    {
    }
};
}  // namespace MegaMoeBF16Training
#undef MEGA_MOE_WORKSPACE_HD
