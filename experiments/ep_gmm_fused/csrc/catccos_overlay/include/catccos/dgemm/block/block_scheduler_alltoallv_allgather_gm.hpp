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

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_GM_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::DGemm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

// Each AIC owns a separate, 64-byte-aligned scratch slice. The complete
// expert/rank schedule stays in GM and is prepared at the same point as the
// local-table scheduler. Only the currently submitted expert is cached locally;
// tile remappers never access GM. No AIC reads another AIC's scratch slice.
template <class MoeConstraints_ = DefaultMoeConstraints, uint32_t M_SPLIT_ = 1>
struct BlockMmadSchedulerAllToAllVAllGatherGm {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;
    static constexpr uint32_t M_SPLIT = M_SPLIT_;
    static_assert(M_SPLIT > 0, "A2AV-GMM M swizzle split must be positive");

    static constexpr uint64_t TABLE_ELEMENTS =
        static_cast<uint64_t>(MoeConstraints::LOCAL_EXPERT_NUM_LIMIT) * MoeConstraints::RANK_SIZE_LIMIT;
    static constexpr uint64_t OFFSET_TABLE_START = (TABLE_ELEMENTS * sizeof(uint32_t) + 7U) / 8U * 8U;
    static constexpr uint64_t WORKSPACE_BYTES =
        (OFFSET_TABLE_START + 3U * TABLE_ELEMENTS * sizeof(uint64_t) + 63U) / 64U * 64U;

    struct CommContext {
        __gm__ uint32_t* tokenNumList{nullptr};
        __gm__ uint64_t* tokenOffsetList{nullptr};
        __gm__ uint64_t* inGroupOffsetList{nullptr};
    };

    struct ExpertCache {
        uint64_t tokenOffsetList[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint64_t inGroupOffsetList[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint64_t outputOffsetList[MoeConstraints::RANK_SIZE_LIMIT]{};
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
    __gm__ uint64_t* outputOffsetList{nullptr};

    CommContext commContext{};
    ExpertCache expertCache{};
    MmadContext mmadContext{};

    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGatherGm() = default;

    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGatherGm(ProblemShape const& problemShape_, uint32_t commShapeM_,
                                         MatrixCoord const& blockShapeMN_, GM_ADDR scratch)
        : problemShape(problemShape_), commShapeM(commShapeM_)
    {
        // The caller has already selected this AIC's private slice. Initialize
        // every active slot before reading it; the allocation may be reused or
        // contain arbitrary values from torch::empty.
        commContext.tokenNumList = reinterpret_cast<__gm__ uint32_t*>(scratch);
        commContext.tokenOffsetList = reinterpret_cast<__gm__ uint64_t*>(scratch + OFFSET_TABLE_START);
        commContext.inGroupOffsetList = commContext.tokenOffsetList + TABLE_ELEMENTS;
        outputOffsetList = commContext.inGroupOffsetList + TABLE_ELEMENTS;
        blockShape = {blockShapeMN_.row(), blockShapeMN_.column(), problemShape.k()};
        commLoops = 0;

        uint64_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
                uint32_t tokens = problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx);
                outputSplitList[rankIdx] += tokens;
                auto index = TableIndex(localExpertIdx, rankIdx);
                outputOffsetList[index] = outputOffset;
                commContext.tokenNumList[index] = 0;
                commContext.tokenOffsetList[index] = 0;
                commContext.inGroupOffsetList[index] = 0;
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
    static uint64_t TableIndex(uint32_t localExpertIdx, uint32_t rankIdx)
    {
        return static_cast<uint64_t>(localExpertIdx) * MoeConstraints::RANK_SIZE_LIMIT + rankIdx;
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
                auto index = TableIndex(localExpertIdx, srcRankIdx);
                auto tokenNumInLastGroup = commContext.tokenNumList[index];
                auto inGroupOffset = commContext.inGroupOffsetList[index] + tokenNumInLastGroup;
                commContext.inGroupOffsetList[index] = inGroupOffset;

                auto globalTokens = problemShape.globalTokensPerLocalExpert(srcRankIdx, localExpertIdx);
                auto residueTokens = globalTokens - inGroupOffset;
                auto tokens = Min<uint64_t>(actualCommTokens - tokenOffset, residueTokens);
                commContext.tokenNumList[index] = tokens;
                commContext.tokenOffsetList[index] = tokenOffset;
                tokenOffset += tokens;
            }
        }
    }

    CATLASS_DEVICE
    void UpdateMmadContext(uint32_t localExpertIdx)
    {
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            auto index = TableIndex(localExpertIdx, srcRankIdx);
            expertCache.tokenOffsetList[srcRankIdx] = commContext.tokenOffsetList[index];
            expertCache.inGroupOffsetList[srcRankIdx] = commContext.inGroupOffsetList[index];
            expertCache.outputOffsetList[srcRankIdx] = outputOffsetList[index];
        }
        mmadContext.rankSize = problemShape.rankSize();
        mmadContext.taskNum = 0;
        mmadContext.taskPrefix[0] = 0;
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            mmadContext.problemShapes[srcRankIdx] = {commContext.tokenNumList[TableIndex(localExpertIdx, srcRankIdx)],
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
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            auto index = TableIndex(localExpertIdx, srcRankIdx);
            expertCache.tokenOffsetList[srcRankIdx] = commContext.tokenOffsetList[index];
            expertCache.inGroupOffsetList[srcRankIdx] = commContext.inGroupOffsetList[index];
            expertCache.outputOffsetList[srcRankIdx] = outputOffsetList[index];
        }
        mmadContext.rankSize = problemShape.rankSize();
        mmadContext.taskNum = 0;
        mmadContext.taskPrefix[0] = 0;
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            uint32_t tokens =
                srcRankIdx == selectedSrcRankIdx ? commContext.tokenNumList[TableIndex(localExpertIdx, srcRankIdx)] : 0;
            mmadContext.problemShapes[srcRankIdx] = {tokens, problemShape.n(), problemShape.k()};
            mmadContext.blockGrids[srcRankIdx] = CeilDiv(mmadContext.problemShapes[srcRankIdx], blockShape);
            mmadContext.taskNum += Numel(mmadContext.blockGrids[srcRankIdx]);
            mmadContext.taskPrefix[srcRankIdx + 1] = mmadContext.taskNum;
        }
    }

    uint32_t startTask = AscendC::GetBlockIdx();

    struct Iter {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGatherGm<MoeConstraints, M_SPLIT>;
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
        using Scheduler = BlockMmadSchedulerAllToAllVAllGatherGm<MoeConstraints, M_SPLIT>;
        const Scheduler* scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        DistMatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            auto tokenOffset = scheduler->expertCache.tokenOffsetList[blockOffset.rank()];
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
        using Scheduler = BlockMmadSchedulerAllToAllVAllGatherGm<MoeConstraints, M_SPLIT>;
        const Scheduler* scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            auto outputOffset = scheduler->expertCache.outputOffsetList[blockOffset.rank()];
            auto inGroupOffset = scheduler->expertCache.inGroupOffsetList[blockOffset.rank()];
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

#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_GM_HPP
