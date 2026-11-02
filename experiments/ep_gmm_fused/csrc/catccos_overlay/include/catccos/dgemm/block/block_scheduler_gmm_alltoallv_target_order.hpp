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

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::DGemm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/// GMM scheduler for destination-major, full-N AllToAllV staging.
///
/// Communication rounds are cut only between complete M blocks. Each selected
/// M block contributes every N tile, so the producer can store it into a
/// row-major [rows, N] stage and the consumer can issue one contiguous UDMA Get.
/// N tiles remain striped over all AICs; an M block is not pinned to one core.
template <class MoeConstraints_ = DefaultMoeConstraints>
struct BlockMmadSchedulerGmmAllToAllVTargetOrder {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;

    struct CommContext {
        uint32_t groupNumList[MoeConstraints::RANK_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT]{};
        uint32_t finishedGroupList[MoeConstraints::RANK_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT]{};
        uint32_t rowOffsetList[MoeConstraints::RANK_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT]{};
        uint32_t actualRows[MoeConstraints::RANK_SIZE_LIMIT]{};
    };

    struct LocalExpertContext {
        uint32_t localExpertIdx{0};
        uint32_t dstRankIdx{0};
        uint32_t rankSize{0};
        uint32_t blockAccumList[MoeConstraints::RANK_SIZE_LIMIT + 1]{};
        uint32_t blockNum{0};

        CATLASS_DEVICE
        void UpdateDstRank(uint32_t taskIdx)
        {
            while (dstRankIdx + 1 < rankSize && taskIdx >= blockAccumList[dstRankIdx + 1]) {
                ++dstRankIdx;
            }
        }

        CATLASS_DEVICE
        uint32_t GetBlockIdxInRank(uint32_t taskIdx) const
        {
            return taskIdx - blockAccumList[dstRankIdx];
        }
    };

    ProblemShape problemShape;
    GemmCoord blockShape;
    uint32_t groupsPerCommInRank{1};
    uint32_t nLoops{1};
    uint32_t commLoops{0};

    uint32_t outputGroupNum[MoeConstraints::RANK_SIZE_LIMIT]{};
    uint32_t outputGroupNumInLocalExpert[MoeConstraints::RANK_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT]{};
    uint32_t outputOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{};
    GemmCoord problemShapes[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT];

    CommContext commContext{};
    LocalExpertContext localExpertCtx{};

    uint32_t startTask = AscendC::GetBlockIdx();

    CATLASS_DEVICE
    BlockMmadSchedulerGmmAllToAllVTargetOrder() = default;

    CATLASS_DEVICE
    BlockMmadSchedulerGmmAllToAllVTargetOrder(ProblemShape const& problemShape_, uint32_t groupsPerCommInRank_,
                                              MatrixCoord const& blockShapeMN_)
        : problemShape(problemShape_), groupsPerCommInRank(Max<uint32_t>(1U, groupsPerCommInRank_))
    {
        blockShape = {blockShapeMN_.row(), blockShapeMN_.column(), problemShape.k()};
        nLoops = CeilDiv(problemShape.n(), blockShape.n());

        uint32_t maxInputGroups = 0;
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            uint32_t inputGroups = 0;
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                inputGroups += CeilDiv(problemShape.localTokensPerExpert(epIdx, localExpertIdx), blockShape.m());
            }
            maxInputGroups = Max(maxInputGroups, inputGroups);
        }
        commLoops = CeilDiv(maxInputGroups, groupsPerCommInRank);

        uint32_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
                uint32_t tokens = problemShape.globalTokensPerLocalExpert(dstRankIdx, localExpertIdx);
                uint32_t groups = CeilDiv(tokens, blockShape.m());
                outputGroupNumInLocalExpert[dstRankIdx][localExpertIdx] = groups;
                outputGroupNum[dstRankIdx] += groups;
                outputOffsetList[localExpertIdx][dstRankIdx] = outputOffset;
                outputOffset += tokens;
                problemShapes[localExpertIdx][dstRankIdx] = {tokens, problemShape.n(), problemShape.k()};
            }
        }

        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            commLoops = Max(commLoops, CeilDiv(outputGroupNum[dstRankIdx], groupsPerCommInRank));
        }
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const
    {
        return commLoops;
    }

    CATLASS_DEVICE
    uint32_t GetGroupsPerCommInRank() const
    {
        return groupsPerCommInRank;
    }

    CATLASS_DEVICE
    uint32_t GetActualRows(uint32_t dstRankIdx) const
    {
        return commContext.actualRows[dstRankIdx];
    }

    CATLASS_DEVICE
    void UpdateCommContext(uint32_t commIdx)
    {
        uint32_t commGroupOffset = commIdx * groupsPerCommInRank;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            uint32_t actualGroups = Min(groupsPerCommInRank, ClipSub(outputGroupNum[dstRankIdx], commGroupOffset));
            uint32_t selectedGroups = 0;
            uint32_t selectedRows = 0;

            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                auto& previousGroups = commContext.groupNumList[dstRankIdx][localExpertIdx];
                auto& finishedGroups = commContext.finishedGroupList[dstRankIdx][localExpertIdx];
                finishedGroups += previousGroups;

                uint32_t expertGroups = outputGroupNumInLocalExpert[dstRankIdx][localExpertIdx];
                uint32_t residueGroups = ClipSub(expertGroups, finishedGroups);
                uint32_t groups = Min(ClipSub(actualGroups, selectedGroups), residueGroups);
                previousGroups = groups;
                commContext.rowOffsetList[dstRankIdx][localExpertIdx] = selectedRows;

                uint32_t tokens = problemShape.globalTokensPerLocalExpert(dstRankIdx, localExpertIdx);
                uint32_t startRow = finishedGroups * blockShape.m();
                uint32_t rows = Min(groups * blockShape.m(), ClipSub(tokens, startRow));
                selectedGroups += groups;
                selectedRows += rows;
            }
            commContext.actualRows[dstRankIdx] = selectedRows;
        }
    }

    CATLASS_DEVICE
    void UpdateLocalExpertContext(uint32_t localExpertIdx)
    {
        localExpertCtx.localExpertIdx = localExpertIdx;
        localExpertCtx.dstRankIdx = 0;
        localExpertCtx.rankSize = problemShape.rankSize();

        uint32_t blockAccum = 0;
        localExpertCtx.blockAccumList[0] = 0;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            blockAccum += commContext.groupNumList[dstRankIdx][localExpertIdx] * nLoops;
            localExpertCtx.blockAccumList[dstRankIdx + 1] = blockAccum;
        }
        localExpertCtx.blockNum = blockAccum;
    }

    struct Iter {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
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
        return {this, startTask, localExpertCtx.blockNum};
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockOffset(Iter const& iter)
    {
        localExpertCtx.UpdateDstRank(iter.taskIdx);
        uint32_t blockIdxInRank = localExpertCtx.GetBlockIdxInRank(iter.taskIdx);
        uint32_t dstRankIdx = localExpertCtx.dstRankIdx;
        uint32_t localExpertIdx = localExpertCtx.localExpertIdx;
        uint32_t groupNum = commContext.groupNumList[dstRankIdx][localExpertIdx];
        MatrixCoord selectedGrid{groupNum, nLoops};
        auto selectedCoord = MatrixSwizzle<7, 1>::GetCoord(selectedGrid, blockIdxInRank);

        uint32_t globalGroupIdx = commContext.finishedGroupList[dstRankIdx][localExpertIdx] + selectedCoord.row();
        return Catlass::MakeCoord(globalGroupIdx * blockShape.m(), selectedCoord.column() * blockShape.n(), 0U,
                                  dstRankIdx);
    }

    struct RemapperA {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
        Scheduler const* scheduler;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            uint32_t tokenOffset = scheduler->outputOffsetList[localExpertIdx][blockOffset.rank()];
            return blockOffset.GetCoordMK() + Catlass::MakeCoord<uint32_t>(tokenOffset, 0);
        }

        CATLASS_DEVICE
        GemmCoord GetResidueShape(DistGemmCoord const& blockOffset) const
        {
            return ClipSub(scheduler->problemShapes[localExpertIdx][blockOffset.rank()],
                           GemmCoord{blockOffset.GetCoordMNK()});
        }
    };

    CATLASS_DEVICE
    RemapperA GetRemapperA(uint32_t, uint32_t localExpertIdx) const
    {
        return {this, localExpertIdx};
    }

    struct RemapperC {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
        Scheduler const* scheduler;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        DistMatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            uint32_t rowOffset = scheduler->commContext.rowOffsetList[blockOffset.rank()][localExpertIdx];
            uint32_t selectedGroupIdx = blockOffset.m() / scheduler->blockShape.m() -
                                        scheduler->commContext.finishedGroupList[blockOffset.rank()][localExpertIdx];
            MatrixCoord localOffset{rowOffset + selectedGroupIdx * scheduler->blockShape.m(), blockOffset.n()};
            return {localOffset, blockOffset.rank()};
        }
    };

    CATLASS_DEVICE
    RemapperC GetRemapperC(uint32_t, uint32_t localExpertIdx) const
    {
        return {this, localExpertIdx};
    }

    CATLASS_DEVICE
    GemmCoord RemapActualBlockShape(DistGemmCoord const& blockOffset, RemapperA const& remapperA,
                                    RemapperC const&) const
    {
        return Min(blockShape, remapperA.GetResidueShape(blockOffset));
    }
};

}  // namespace Catccos::DGemm::Block

#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_HPP
