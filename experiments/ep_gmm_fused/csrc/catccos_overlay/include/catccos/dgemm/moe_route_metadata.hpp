/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/catccos_overlay/LICENSE in the source tree and licenses/catccos/LICENSE in the wheel.
 */
// Modified by zhu-mingzhe71 2026
#ifndef CATCCOS_DGEMM_MOE_ROUTE_METADATA_HPP
#define CATCCOS_DGEMM_MOE_ROUTE_METADATA_HPP

#include <cstdint>

namespace Catccos::DGemm {

inline constexpr uint32_t MOE_ROUTE_MAX_RANKS = 8;
inline constexpr uint32_t MOE_ROUTE_MAX_LOCAL_EXPERTS = 64;
inline constexpr uint32_t MOE_ROUTE_MAX_SEGMENTS = MOE_ROUTE_MAX_RANKS * MOE_ROUTE_MAX_LOCAL_EXPERTS;

// One contiguous source-rank/expert run. sourceRowOffset is relative to the
// source-rank stream in symmetric staging; outputRowOffset is relative to the
// exported expert-major [expert][source rank][row, K] tensor.
struct MoeRouteSegment {
    uint64_t sourceRowOffset{0};
    uint64_t outputRowOffset{0};
    uint32_t rows{0};
    uint16_t sourceRank{0};
    uint16_t localExpert{0};
};

// Host-prepared route prefixes.  The kernel treats this as an optimization
// descriptor only; the original route tables remain the source of truth and
// provide the compatibility fallback when this pointer is null.
struct MoeRouteMetadata {
    uint32_t rankSize{0};
    uint32_t localExpertNum{0};
    uint32_t segmentCount{0};
    uint32_t reserved{0};
    uint64_t peerRows[MOE_ROUTE_MAX_RANKS]{};
    uint64_t expertRows[MOE_ROUTE_MAX_LOCAL_EXPERTS]{};
    MoeRouteSegment segments[MOE_ROUTE_MAX_SEGMENTS]{};
};

}  // namespace Catccos::DGemm

#endif  // CATCCOS_DGEMM_MOE_ROUTE_METADATA_HPP
