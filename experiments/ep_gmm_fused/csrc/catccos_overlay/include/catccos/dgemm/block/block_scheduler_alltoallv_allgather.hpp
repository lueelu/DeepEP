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

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::DGemm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <class MoeConstraints_ = DefaultMoeConstraints, uint32_t M_SPLIT_ = 1>
struct BlockMmadSchedulerAllToAllVAllGather {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;
    static constexpr uint32_t M_SPLIT = M_SPLIT_;
    static_assert(M_SPLIT > 0, "A2AV-GMM M swizzle split must be positive");

    struct CommContext {
        uint32_t tokenNumList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t tokenOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t inGroupOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
    };

    struct MmadContext {
        GemmCoord problemShapes[MoeConstraints::RANK_SIZE_LIMIT];
        GemmCoord blockGrids[MoeConstraints::RANK_SIZE_LIMIT];
        uint32_t taskPrefix[MoeConstraints::RANK_SIZE_LIMIT + 1]{0};
        uint32_t rankSize{0};
        uint32_t taskNum{0};
    };

    ProblemShape problemShape;
    uint32_t commShapeM;
    GemmCoord blockShape;
    uint32_t commLoops;

    uint32_t outputSplitList[MoeConstraints::RANK_SIZE_LIMIT]{0};
    uint64_t outputOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};

    CommContext commContext{};
    MmadContext mmadContext{};

    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGather() = default;

    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGather(ProblemShape const& problemShape_, uint32_t commShapeM_,
                                         MatrixCoord const& blockShapeMN_)
        : problemShape(problemShape_), commShapeM(commShapeM_)
    {
        blockShape = {blockShapeMN_.row(), blockShapeMN_.column(), problemShape.k()};
        commLoops = 0;

        uint64_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
                uint32_t tokens = problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx);
                outputSplitList[rankIdx] += tokens;
                outputOffsetList[localExpertIdx][rankIdx] = outputOffset;
                outputOffset += tokens;
            }
        }

        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            // 统计从每个 rank 接收所需的通信轮次
            auto receiveLoops = CeilDiv(outputSplitList[rankIdx], commShapeM_);
            commLoops = Max(commLoops, receiveLoops);
        }

        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            uint32_t inputSplit = 0;
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                inputSplit += problemShape.localTokensPerExpert(epIdx, localExpertIdx);
            }
            // 统计发送到每个 EP 的通信轮次
            auto sendLoops = CeilDiv(inputSplit, commShapeM_);
            commLoops = Max(commLoops, sendLoops);
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
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            auto actualCommTokens = Min(commShapeM, ClipSub(outputSplitList[srcRankIdx], commIdx * commShapeM));

            uint64_t tokenOffset{0};
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                auto tokenNumInLastGroup = commContext.tokenNumList[localExpertIdx][srcRankIdx];
                commContext.inGroupOffsetList[localExpertIdx][srcRankIdx] += tokenNumInLastGroup;

                auto globalTokens = problemShape.globalTokensPerLocalExpert(srcRankIdx, localExpertIdx);
                auto residueTokens = globalTokens - commContext.inGroupOffsetList[localExpertIdx][srcRankIdx];
                auto tokens = Min<uint64_t>(actualCommTokens - tokenOffset, residueTokens);
                commContext.tokenNumList[localExpertIdx][srcRankIdx] = tokens;
                commContext.tokenOffsetList[localExpertIdx][srcRankIdx] = tokenOffset;
                tokenOffset += tokens;
            }
        }
    }

    CATLASS_DEVICE
    void UpdateMmadContext(uint32_t localExpertIdx)
    {
        mmadContext.rankSize = problemShape.rankSize();
        mmadContext.taskNum = 0;
        mmadContext.taskPrefix[0] = 0;
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            mmadContext.problemShapes[srcRankIdx] = {commContext.tokenNumList[localExpertIdx][srcRankIdx],
                                                     problemShape.n(), problemShape.k()};
            mmadContext.blockGrids[srcRankIdx] = CeilDiv(mmadContext.problemShapes[srcRankIdx], blockShape);
            mmadContext.taskNum += Numel(mmadContext.blockGrids[srcRankIdx]);
            mmadContext.taskPrefix[srcRankIdx + 1] = mmadContext.taskNum;
        }
    }

    // Compatibility path for kernels that intentionally submit one source
    // rank at a time.  Empty prefixes preserve the actual rank id in the
    // distributed block coordinate while retaining the legacy task domain.
    CATLASS_DEVICE
    void UpdateMmadContext(uint32_t localExpertIdx, uint32_t selectedSrcRankIdx)
    {
        mmadContext.rankSize = problemShape.rankSize();
        mmadContext.taskNum = 0;
        mmadContext.taskPrefix[0] = 0;
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            uint32_t tokens =
                srcRankIdx == selectedSrcRankIdx ? commContext.tokenNumList[localExpertIdx][srcRankIdx] : 0;
            mmadContext.problemShapes[srcRankIdx] = {tokens, problemShape.n(), problemShape.k()};
            mmadContext.blockGrids[srcRankIdx] = CeilDiv(mmadContext.problemShapes[srcRankIdx], blockShape);
            mmadContext.taskNum += Numel(mmadContext.blockGrids[srcRankIdx]);
            mmadContext.taskPrefix[srcRankIdx + 1] = mmadContext.taskNum;
        }
    }

    uint32_t startTask = AscendC::GetBlockIdx();

    struct Iter {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, M_SPLIT>;
        Scheduler* const scheduler;
        uint32_t taskIdx;
        uint32_t coreLoops;

        CATLASS_DEVICE
        void Next()
        {
            taskIdx += AscendC::GetBlockNum();
        }

        CATLASS_DEVICE
        bool End()
        {
            if (taskIdx >= coreLoops) {
                scheduler->startTask = taskIdx - coreLoops;
            }
            return taskIdx >= coreLoops;
        }
    };

    CATLASS_DEVICE
    Iter Begin()
    {
        uint32_t taskIdx = startTask;
        uint32_t coreLoops = mmadContext.taskNum;
        return Iter{this, taskIdx, coreLoops};
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockOffset(Iter const& iter) const
    {
        uint32_t srcRankIdx = 0;
        while (srcRankIdx + 1 < mmadContext.rankSize && iter.taskIdx >= mmadContext.taskPrefix[srcRankIdx + 1]) {
            ++srcRankIdx;
        }
        uint32_t taskIdxInRank = iter.taskIdx - mmadContext.taskPrefix[srcRankIdx];
        auto blockCoordMN =
            MatrixSwizzle<M_SPLIT, 0>::GetCoord(mmadContext.blockGrids[srcRankIdx].GetCoordMN(), taskIdxInRank);
        return Catlass::MakeCoord<uint32_t>(blockCoordMN.row() * blockShape.m(), blockCoordMN.column() * blockShape.n(),
                                            0, srcRankIdx);
    }

    struct RemapperA {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, M_SPLIT>;
        const Scheduler* scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        DistMatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            auto tokenOffset = scheduler->commContext.tokenOffsetList[localExpertIdx][blockOffset.rank()];
            return {blockOffset.GetCoordMK() + Catlass::MakeCoord<uint32_t>(tokenOffset, 0), blockOffset.rank()};
        }

        CATLASS_DEVICE
        GemmCoord GetResidueShape(DistGemmCoord const& blockOffset) const
        {
            return ClipSub(scheduler->mmadContext.problemShapes[blockOffset.rank()],
                           GemmCoord{blockOffset.GetCoordMNK()});
        }
    };

    CATLASS_DEVICE
    const RemapperA GetRemapperA(uint32_t commIdx, uint32_t localExpertIdx) const
    {
        return {this, commIdx, localExpertIdx};
    }

    CATLASS_DEVICE
    const RemapperA GetRemapperA(uint32_t commIdx, uint32_t localExpertIdx, uint32_t) const
    {
        return {this, commIdx, localExpertIdx};
    }

    struct RemapperC {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, M_SPLIT>;
        const Scheduler* scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            auto outputOffset = scheduler->outputOffsetList[localExpertIdx][blockOffset.rank()];
            auto inGroupOffset = scheduler->commContext.inGroupOffsetList[localExpertIdx][blockOffset.rank()];
            return blockOffset.GetCoordMN() + Catlass::MakeCoord<uint32_t>(outputOffset + inGroupOffset, 0);
        }

        CATLASS_DEVICE
        GemmCoord GetResidueShape(DistGemmCoord const& blockOffset) const
        {
            return ClipSub(scheduler->mmadContext.problemShapes[blockOffset.rank()],
                           GemmCoord{blockOffset.GetCoordMNK()});
        }
    };

    CATLASS_DEVICE
    const RemapperC GetRemapperC(uint32_t commIdx, uint32_t localExpertIdx) const
    {
        return {this, commIdx, localExpertIdx};
    }

    CATLASS_DEVICE
    const RemapperC GetRemapperC(uint32_t commIdx, uint32_t localExpertIdx, uint32_t) const
    {
        return {this, commIdx, localExpertIdx};
    }

    CATLASS_DEVICE
    GemmCoord RemapActualBlockShape(DistGemmCoord const& blockOffset, RemapperA const& remapperA,
                                    RemapperC const& remapperC) const
    {
        auto remapABlockShape = Min(blockShape, remapperA.GetResidueShape(blockOffset));
        return Min(remapABlockShape, remapperC.GetResidueShape(blockOffset));
    }
};

}  // namespace Catccos::DGemm::Block

#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP
