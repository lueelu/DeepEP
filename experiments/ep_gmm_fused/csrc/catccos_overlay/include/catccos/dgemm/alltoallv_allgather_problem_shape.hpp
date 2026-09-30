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

#ifndef CATCCOS_DGEMM_ALLTOALLV_ALLGATHER_PROBLEM_SHAPE_HPP
#define CATCCOS_DGEMM_ALLTOALLV_ALLGATHER_PROBLEM_SHAPE_HPP

#include <cstdint>

#include "catlass/gemm_coord.hpp"

namespace Catccos::DGemm {

using Catlass::GemmCoord;

template <uint32_t TP_SIZE_LIMIT_, uint32_t EP_SIZE_LIMIT_, uint32_t LOCAL_EXPERT_NUM_LIMIT_>
struct MoeConstraints {
    static constexpr uint32_t TP_SIZE_LIMIT = TP_SIZE_LIMIT_;
    static constexpr uint32_t EP_SIZE_LIMIT = EP_SIZE_LIMIT_;
    static constexpr uint32_t LOCAL_EXPERT_NUM_LIMIT = LOCAL_EXPERT_NUM_LIMIT_;

    static constexpr uint32_t EXPERT_NUM_LIMIT = LOCAL_EXPERT_NUM_LIMIT * EP_SIZE_LIMIT;
    static constexpr uint32_t RANK_SIZE_LIMIT = TP_SIZE_LIMIT * EP_SIZE_LIMIT;
};

using DefaultMoeConstraints = MoeConstraints<2, 8, 8>;

struct AllToAllVAllGatherProblemShape {
    using UnderlyingProblemShape = GemmCoord;

    CATLASS_HOST_DEVICE
    AllToAllVAllGatherProblemShape() = default;

    CATLASS_HOST_DEVICE
    AllToAllVAllGatherProblemShape(UnderlyingProblemShape const originProblemShape, uint32_t rankSize, uint32_t rankIdx,
                                   uint32_t epSize, uint32_t expertNum, GM_ADDR ptrLocalTokensPerExpert,
                                   GM_ADDR ptrGlobalTokensPerLocalExpert)
        : originProblemShape_(originProblemShape),
          rankSize_(rankSize),
          rankIdx_(rankIdx),
          epSize_(epSize),
          expertNum_(expertNum),
          ptrLocalTokensPerExpert_(reinterpret_cast<__gm__ int64_t*>(ptrLocalTokensPerExpert)),
          ptrGlobalTokensPerLocalExpert_(reinterpret_cast<__gm__ int64_t*>(ptrGlobalTokensPerLocalExpert))
    {
    }

    CATLASS_HOST_DEVICE
    uint32_t k() const
    {
        return originProblemShape_.k();
    }

    CATLASS_HOST_DEVICE
    uint32_t n() const
    {
        return originProblemShape_.n();
    }

    CATLASS_HOST_DEVICE
    uint32_t rankSize() const
    {
        return rankSize_;
    }

    CATLASS_HOST_DEVICE
    uint32_t rankIdx() const
    {
        return rankIdx_;
    }

    CATLASS_HOST_DEVICE
    uint32_t epSize() const
    {
        return epSize_;
    }

    CATLASS_HOST_DEVICE
    uint32_t tpSize() const
    {
        return rankSize_ / epSize_;
    }

    CATLASS_HOST_DEVICE
    uint32_t localExpertNum() const
    {
        return expertNum_ / epSize_;
    }

    CATLASS_HOST_DEVICE
    uint32_t expertNum() const
    {
        return expertNum_;
    }

    // PyTorch exposes routing counts as signed int64 tensors. Kernel geometry
    // remains uint32 because m and the CatCCOS schedulers use uint32 extents.
    CATLASS_HOST_DEVICE
    uint32_t localTokensPerExpert(uint32_t expertIdx) const
    {
        return static_cast<uint32_t>(ptrLocalTokensPerExpert_[expertIdx]);
    }

    CATLASS_HOST_DEVICE
    uint32_t localTokensPerExpert(uint32_t epIdx, uint32_t localExpertIdx) const
    {
        return localTokensPerExpert(epIdx * localExpertNum() + localExpertIdx);
    }

    CATLASS_HOST_DEVICE
    uint32_t globalTokensPerLocalExpert(uint32_t tokenIdx) const
    {
        return static_cast<uint32_t>(ptrGlobalTokensPerLocalExpert_[tokenIdx]);
    }

    CATLASS_HOST_DEVICE
    uint32_t globalTokensPerLocalExpert(uint32_t rankIdx, uint32_t localExpertIdx) const
    {
        return globalTokensPerLocalExpert(rankIdx * localExpertNum() + localExpertIdx);
    }

private:
    UnderlyingProblemShape originProblemShape_;
    uint32_t rankSize_;
    uint32_t rankIdx_;
    uint32_t epSize_;
    uint32_t expertNum_;

    __gm__ int64_t* ptrLocalTokensPerExpert_;
    __gm__ int64_t* ptrGlobalTokensPerLocalExpert_;
};

}  // namespace Catccos::DGemm

#endif  // CATCCOS_DGEMM_ALLTOALLV_ALLGATHER_PROBLEM_SHAPE_HPP
