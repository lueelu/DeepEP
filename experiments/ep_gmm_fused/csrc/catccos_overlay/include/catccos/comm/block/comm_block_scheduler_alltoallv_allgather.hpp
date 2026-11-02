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

#ifndef CATCCOS_COMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER
#define CATCCOS_COMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/symm_coord.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"

namespace Catccos::Comm::Block {

using Catlass::MatrixCoord;

template <class MoeConstraints_ = DGemm::DefaultMoeConstraints>
struct BlockCommSchedulerAllToAllVAllGather {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;

    ProblemShape problemShape;
    MatrixCoord commShape;
    MatrixCoord blockShape;
    MatrixCoord blockGrid;

    uint32_t commLoops;
    uint32_t coreLoops;

    uint32_t inputSplitList[MoeConstraints::EP_SIZE_LIMIT]{0};     // Tokens sent to each EP
    uint64_t inputOffsetList[MoeConstraints::EP_SIZE_LIMIT]{0};    // Data offset for each EP
    uint32_t outputSplitList[MoeConstraints::RANK_SIZE_LIMIT]{0};  // Tokens received from each rank

    CATLASS_DEVICE
    BlockCommSchedulerAllToAllVAllGather() = default;

    CATLASS_DEVICE
    BlockCommSchedulerAllToAllVAllGather(ProblemShape const& problemShape_, uint32_t commShapeM_,
                                         MatrixCoord const& blockShape_)
        : problemShape(problemShape_), blockShape(blockShape_)
    {
        commShape = Catlass::MakeCoord<uint32_t>(commShapeM_, problemShape.k());
        blockGrid = CeilDiv(commShape, blockShape);
        coreLoops = Numel(blockGrid) * problemShape.rankSize();
        commLoops = 0;

        int64_t tokenOffset = 0;
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                inputSplitList[epIdx] += problemShape.localTokensPerExpert(epIdx, localExpertIdx);
            }
            inputOffsetList[epIdx] = tokenOffset;
            tokenOffset += inputSplitList[epIdx];
            auto sendLoops = CeilDiv(inputSplitList[epIdx], commShapeM_);
            commLoops = Max(commLoops, sendLoops);
        }

        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                outputSplitList[rankIdx] += problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx);
            }
            auto receiveLoops = CeilDiv(outputSplitList[rankIdx], commShapeM_);
            commLoops = Max(commLoops, receiveLoops);
        }
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const
    {
        return commLoops;
    }

    CATLASS_DEVICE
    uint32_t GetActualReceiveAccum(uint32_t commIdx) const
    {
        auto commTokens = commShape.row();
        int64_t actualReceivedAccum = 0;
        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            auto actualTokens = Min(commTokens, ClipSub(outputSplitList[rankIdx], commIdx * commTokens));
            MatrixCoord actualReceiveShape{actualTokens, problemShape.k()};
            actualReceivedAccum += Numel(CeilDiv(actualReceiveShape, blockShape));
        }
        return actualReceivedAccum;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops(uint32_t commIdx) const
    {
        (void)commIdx;
        return coreLoops;  // Upper bound independent of commIdx
    }

    CATLASS_DEVICE
    uint32_t GetBlockCount() const
    {
        return Numel(blockGrid);
    }

    struct Iter {
        uint32_t taskIdx{0};
        uint32_t coreLoops{0};

        CATLASS_DEVICE
        void Next()
        {
            taskIdx += AscendC::GetBlockNum();
        }

        CATLASS_DEVICE
        bool End()
        {
            return (AscendC::GetSubBlockIdx() == 1) || (taskIdx >= coreLoops);
        }
    };

    CATLASS_DEVICE
    Iter Begin(uint32_t commIdx) const
    {
        uint32_t taskIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        return {taskIdx, GetCoreLoops(commIdx)};
    }

    CATLASS_DEVICE
    SymmMatrixCoord GetBlockOffset(Iter const& iter) const
    {
        auto dstRankIdx = iter.taskIdx % problemShape.rankSize();
        auto blockIdx = iter.taskIdx / problemShape.rankSize();
        return GetBlockOffset(blockIdx, dstRankIdx);
    }

    CATLASS_DEVICE
    SymmMatrixCoord GetBlockOffset(uint32_t blockIdx, uint32_t dstRankIdx) const
    {
        auto blockCoord = Catlass::MakeCoord(blockIdx / blockGrid.column(), blockIdx % blockGrid.column());
        return SymmMatrixCoord{blockCoord * blockShape, problemShape.rankIdx(), dstRankIdx};
    }

    struct RemapperSrc {
        using Scheduler = BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;

        Scheduler* scheduler;
        uint32_t commIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(SymmMatrixCoord const& blockOffset) const
        {
            auto dstRankIdx = blockOffset.remote();
            auto dstEpIdx = dstRankIdx % scheduler->problemShape.epSize();
            auto epOffset = Catlass::MakeCoord<uint32_t>(scheduler->inputOffsetList[dstEpIdx], 0);
            auto commOffset = Catlass::MakeCoord<uint32_t>(commIdx, 0) * scheduler->commShape;
            return epOffset + commOffset + blockOffset.GetMatrixCoord();
        }

        CATLASS_DEVICE
        MatrixCoord GetResidueShape(SymmMatrixCoord const& blockOffset) const
        {
            auto dstRankIdx = blockOffset.remote();
            auto dstEpIdx = dstRankIdx % scheduler->problemShape.epSize();
            auto epShape = Catlass::MakeCoord(scheduler->inputSplitList[dstEpIdx], scheduler->problemShape.k());
            auto commOffset = Catlass::MakeCoord<uint32_t>(commIdx, 0) * scheduler->commShape;
            auto actualCommShape = Min(scheduler->commShape, ClipSub(epShape, commOffset));
            return ClipSub(actualCommShape, blockOffset.GetMatrixCoord());
        }
    };

    CATLASS_DEVICE
    const RemapperSrc GetRemapperSrc(uint32_t commIdx)
    {
        return {this, commIdx};
    }

    CATLASS_DEVICE
    MatrixCoord RemapActualBlockShape(SymmMatrixCoord const& blockOffset, RemapperSrc const& remapperSrc) const
    {
        return Min(blockShape, remapperSrc.GetResidueShape(blockOffset));
    }
};

}  // namespace Catccos::Comm::Block

#endif
