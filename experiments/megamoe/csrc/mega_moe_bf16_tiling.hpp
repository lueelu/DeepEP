// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_bf16_tiling.hpp
 * \brief Shared tiling parameters and constants for MegaMoE BF16 training.
 */

#ifndef MEGA_MOE_BF16_TILING_HPP
#define MEGA_MOE_BF16_TILING_HPP

#include <cstdint>

// P1 GMM2-grad writes one FP32 dP partial per packed row and F tile.
// P6 reduces the partials for each row.
// Host workspace sizing and the device epilogue use the same F-tile width.
constexpr uint32_t BACKWARD_DP_PARTIAL_TILE_N = 256U;

// SwiGLU activation clamp limit (dsv4 swiglu_limit): gate = min(gate, L),
// up = clamp(up, -L, L). FLT_MAX disables the clamp.
static constexpr float kMegaMoeSwigluLimitDisabled = 3.402823466e+38F;

struct MegaMoeBf16TilingData {
    uint32_t expertPerRank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;  // if blockNumPerEP < dispatchRows, then dispatchRows is invalid
    uint32_t maxOutputSize;
    uint32_t rankId;
    uint32_t topK;
    int32_t dispatchRows;
    uint8_t groupListType;
    bool transX;
    bool transW;
    bool transW2;
    // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
    float swigluLimit;
    // Effective BF16 save policy: 1=none, 2=BF16, 3=MXFP8.
    uint32_t savedGateUpMode{2};
    // The Host selects the token-count exchange from EP size and the number of local experts.
    // Dense exchanges the full count table. Compact sends peer-specific counts and prefixes.
    // 0=dense, 1=compact. Experimental variants always keep this at zero.
    uint32_t compactTokenCountExchange{0};
    // TopK weights and their gradients: 0=BF16, 1=FP32. GMM operands stay BF16.
    uint32_t topkWeightType{0};
    uint32_t mGroupsPerWave{0};
    // 0: interleave GMM1/GMM2 Waves; 1: finish local GMM1 before GMM2.
    uint32_t forwardSchedule{0};
    // Training multiplies p before GMM2; inference weights each route during Unpermute.
    uint32_t routeWeightBeforeGmm2{1};
    uint32_t backwardSplitExperts{0};
};
#endif
