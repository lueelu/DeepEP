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

#ifndef CATCCOS_COMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP
#define CATCCOS_COMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"

namespace Catccos::CommEpilogue::Block {

using Catlass::MatrixCoord;

/// Consumer-side scheduler paired with
/// BlockMmadSchedulerGmmAllToAllVTargetOrder. For one source rank and one
/// communication round it describes a single contiguous [rows, N] transfer.
template <class MoeConstraints_ = DGemm::DefaultMoeConstraints, bool IsDynamic_ = true>
struct BlockCommSchedulerGmmAllToAllVTargetOrder {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;
    static constexpr bool IsDynamic = IsDynamic_;

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() = default;

        CATLASS_DEVICE
        static MatrixCoord CoreSplit()
        {
            return MatrixCoord{uint32_t{1}, uint32_t{1}};
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord coreSplit;

        CATLASS_HOST_DEVICE
        ParamsBase() = default;

        CATLASS_HOST_DEVICE
        explicit ParamsBase(MatrixCoord coreSplit_) : coreSplit(coreSplit_) {}

        CATLASS_DEVICE
        MatrixCoord CoreSplit() const
        {
            return coreSplit;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    ProblemShape problemShape;
    uint32_t groupsPerCommInRank{1};
    uint32_t srcEpIdx{0};
    uint32_t blockM{1};
    uint32_t commLoops{0};

    uint32_t inputGroupNum[MoeConstraints::EP_SIZE_LIMIT]{};
    uint32_t inputGroupNumInLocalExpert[MoeConstraints::EP_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT]{};
    uint32_t inputRowOffset[MoeConstraints::EP_SIZE_LIMIT]{};
    uint32_t outputGroupNum[MoeConstraints::RANK_SIZE_LIMIT]{};

    uint32_t actualRows{0};
    uint32_t outputRowOffset{0};
    uint32_t consumedRows{0};

    CATLASS_DEVICE
    BlockCommSchedulerGmmAllToAllVTargetOrder() = default;

    CATLASS_DEVICE
    BlockCommSchedulerGmmAllToAllVTargetOrder(ProblemShape const& problemShape_, uint32_t groupsPerCommInRank_,
                                              uint32_t srcEpIdx_, MatrixCoord const& mmadBlockShape_,
                                              MatrixCoord const&, MatrixCoord const&)
        : problemShape(problemShape_),
          groupsPerCommInRank(Max<uint32_t>(1U, groupsPerCommInRank_)),
          srcEpIdx(srcEpIdx_),
          blockM(mmadBlockShape_.row())
    {
        uint32_t inputRows = 0;
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            inputRowOffset[epIdx] = inputRows;
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                uint32_t tokens = problemShape.localTokensPerExpert(epIdx, localExpertIdx);
                uint32_t groups = CeilDiv(tokens, blockM);
                inputGroupNumInLocalExpert[epIdx][localExpertIdx] = groups;
                inputGroupNum[epIdx] += groups;
                inputRows += tokens;
            }
            commLoops = Max(commLoops, CeilDiv(inputGroupNum[epIdx], groupsPerCommInRank));
        }

        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                outputGroupNum[dstRankIdx] +=
                    CeilDiv(problemShape.globalTokensPerLocalExpert(dstRankIdx, localExpertIdx), blockM);
            }
            commLoops = Max(commLoops, CeilDiv(outputGroupNum[dstRankIdx], groupsPerCommInRank));
        }
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const
    {
        return commLoops;
    }

    CATLASS_DEVICE
    void UpdateCommContext(uint32_t commIdx)
    {
        if (commIdx != 0) {
            consumedRows += actualRows;
        }
        outputRowOffset = inputRowOffset[srcEpIdx] + consumedRows;
        actualRows = 0;

        uint32_t skipGroups = commIdx * groupsPerCommInRank;
        uint32_t remainingGroups = Min(groupsPerCommInRank, ClipSub(inputGroupNum[srcEpIdx], skipGroups));
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum() && remainingGroups != 0;
             ++localExpertIdx) {
            uint32_t expertGroups = inputGroupNumInLocalExpert[srcEpIdx][localExpertIdx];
            if (skipGroups >= expertGroups) {
                skipGroups -= expertGroups;
                continue;
            }

            uint32_t groups = Min(remainingGroups, expertGroups - skipGroups);
            uint32_t tokens = problemShape.localTokensPerExpert(srcEpIdx, localExpertIdx);
            uint32_t startRow = skipGroups * blockM;
            actualRows += Min(groups * blockM, ClipSub(tokens, startRow));
            remainingGroups -= groups;
            skipGroups = 0;
        }
    }

    CATLASS_DEVICE
    uint32_t GetActualRows() const
    {
        return actualRows;
    }

    CATLASS_DEVICE
    uint32_t GetOutputRowOffset() const
    {
        return outputRowOffset;
    }

    CATLASS_DEVICE
    bool HasInput() const
    {
        return actualRows != 0;
    }

    CATLASS_DEVICE
    uint32_t GetActualReceiveAccum(uint32_t commIdx) const
    {
        uint32_t commGroupOffset = commIdx * groupsPerCommInRank;
        uint32_t receiveCount = 0;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            receiveCount += static_cast<uint32_t>(commGroupOffset < outputGroupNum[dstRankIdx]);
        }
        return receiveCount;
    }
};

}  // namespace Catccos::CommEpilogue::Block

#endif  // CATCCOS_COMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP
