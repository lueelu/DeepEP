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

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_GM_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::DGemm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

// Full destination-major schedule backed by one private GM slice per AIC.
// UpdateCommContext prepares every expert before the stage wait, as in the
// local-table scheduler. Selecting an expert copies its ready entries into
// O(rankSize) scalar storage, so tile traversal does not access the GM tables.
// Only this AIC's scalar pipeline reads or writes its slice; it is neither a
// communication stage nor a table shared with another AIC/AIV.
template <class MoeConstraints_ = DefaultMoeConstraints>
struct BlockMmadSchedulerGmmAllToAllVTargetOrderGm {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;

    static constexpr uint64_t TABLE_ELEMENTS =
        static_cast<uint64_t>(MoeConstraints::RANK_SIZE_LIMIT) * MoeConstraints::LOCAL_EXPERT_NUM_LIMIT;
    static constexpr uint64_t WORKSPACE_BYTES = ((6 * TABLE_ELEMENTS * sizeof(uint32_t) + 63) / 64) * 64;

    enum Table : uint32_t {
        GROUP_NUM = 0,
        FINISHED_GROUP = 1,
        ROW_OFFSET = 2,
        OUTPUT_GROUP_NUM = 3,
        OUTPUT_OFFSET = 4,
        TOKENS = 5,
    };

    struct CommContext {
        uint32_t actualRows[MoeConstraints::RANK_SIZE_LIMIT]{};
    };

    struct ExpertCache {
        uint32_t groupNum[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint32_t finishedGroup[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint32_t rowOffset[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint32_t outputOffset[MoeConstraints::RANK_SIZE_LIMIT]{};
        uint32_t tokens[MoeConstraints::RANK_SIZE_LIMIT]{};
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
    __gm__ uint32_t* tables{nullptr};

    CommContext commContext{};
    ExpertCache expertCache{};
    LocalExpertContext localExpertCtx{};
    uint32_t startTask = AscendC::GetBlockIdx();

    CATLASS_DEVICE
    uint64_t TableIndex(Table table, uint32_t dstRankIdx, uint32_t localExpertIdx) const
    {
        return static_cast<uint64_t>(table) * TABLE_ELEMENTS +
               static_cast<uint64_t>(dstRankIdx) * MoeConstraints::LOCAL_EXPERT_NUM_LIMIT + localExpertIdx;
    }

    CATLASS_DEVICE
    uint32_t Read(Table table, uint32_t dstRankIdx, uint32_t localExpertIdx) const
    {
        return tables[TableIndex(table, dstRankIdx, localExpertIdx)];
    }

    CATLASS_DEVICE
    void Write(Table table, uint32_t dstRankIdx, uint32_t localExpertIdx, uint32_t value)
    {
        tables[TableIndex(table, dstRankIdx, localExpertIdx)] = value;
    }

    CATLASS_DEVICE
    BlockMmadSchedulerGmmAllToAllVTargetOrderGm() = default;

    // scratch already points to this AIC's 64-byte-aligned WORKSPACE_BYTES
    // slice. The caller owns its lifetime through kernel completion.
    CATLASS_DEVICE
    BlockMmadSchedulerGmmAllToAllVTargetOrderGm(ProblemShape const& problemShape_,
                                               uint32_t groupsPerCommInRank_,
                                               MatrixCoord const& blockShapeMN_, GM_ADDR scratch)
        : problemShape(problemShape_), groupsPerCommInRank(Max<uint32_t>(1U, groupsPerCommInRank_)),
          tables(reinterpret_cast<__gm__ uint32_t*>(scratch))
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
                Write(OUTPUT_GROUP_NUM, dstRankIdx, localExpertIdx, groups);
                Write(OUTPUT_OFFSET, dstRankIdx, localExpertIdx, outputOffset);
                Write(TOKENS, dstRankIdx, localExpertIdx, tokens);
                // Scratch may contain data from a previous invocation. Reset
                // every live state entry before UpdateCommContext reads it.
                Write(GROUP_NUM, dstRankIdx, localExpertIdx, 0);
                Write(FINISHED_GROUP, dstRankIdx, localExpertIdx, 0);
                Write(ROW_OFFSET, dstRankIdx, localExpertIdx, 0);
                outputGroupNum[dstRankIdx] += groups;
                outputOffset += tokens;
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
                uint32_t finishedGroups = Read(FINISHED_GROUP, dstRankIdx, localExpertIdx) +
                                          Read(GROUP_NUM, dstRankIdx, localExpertIdx);
                uint32_t expertGroups = Read(OUTPUT_GROUP_NUM, dstRankIdx, localExpertIdx);
                uint32_t residueGroups = ClipSub(expertGroups, finishedGroups);
                uint32_t groups = Min(ClipSub(actualGroups, selectedGroups), residueGroups);
                Write(FINISHED_GROUP, dstRankIdx, localExpertIdx, finishedGroups);
                Write(GROUP_NUM, dstRankIdx, localExpertIdx, groups);
                Write(ROW_OFFSET, dstRankIdx, localExpertIdx, selectedRows);

                uint32_t tokens = Read(TOKENS, dstRankIdx, localExpertIdx);
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
            expertCache.groupNum[dstRankIdx] = Read(GROUP_NUM, dstRankIdx, localExpertIdx);
            expertCache.finishedGroup[dstRankIdx] = Read(FINISHED_GROUP, dstRankIdx, localExpertIdx);
            expertCache.rowOffset[dstRankIdx] = Read(ROW_OFFSET, dstRankIdx, localExpertIdx);
            expertCache.outputOffset[dstRankIdx] = Read(OUTPUT_OFFSET, dstRankIdx, localExpertIdx);
            expertCache.tokens[dstRankIdx] = Read(TOKENS, dstRankIdx, localExpertIdx);
            blockAccum += expertCache.groupNum[dstRankIdx] * nLoops;
            localExpertCtx.blockAccumList[dstRankIdx + 1] = blockAccum;
        }
        localExpertCtx.blockNum = blockAccum;
    }

    struct Iter {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrderGm<MoeConstraints>;
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
        MatrixCoord selectedGrid{expertCache.groupNum[dstRankIdx], nLoops};
        auto selectedCoord = MatrixSwizzle<7, 1>::GetCoord(selectedGrid, blockIdxInRank);

        uint32_t globalGroupIdx = expertCache.finishedGroup[dstRankIdx] + selectedCoord.row();
        return Catlass::MakeCoord(globalGroupIdx * blockShape.m(), selectedCoord.column() * blockShape.n(), 0U,
                                  dstRankIdx);
    }

    // Remappers use the current expert's cache. Finish using them before the
    // next UpdateLocalExpertContext or UpdateCommContext call.
    struct RemapperA {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrderGm<MoeConstraints>;
        Scheduler const* scheduler;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            uint32_t tokenOffset = scheduler->expertCache.outputOffset[blockOffset.rank()];
            return blockOffset.GetCoordMK() + Catlass::MakeCoord<uint32_t>(tokenOffset, 0);
        }

        CATLASS_DEVICE
        GemmCoord GetResidueShape(DistGemmCoord const& blockOffset) const
        {
            GemmCoord problem{scheduler->expertCache.tokens[blockOffset.rank()], scheduler->problemShape.n(),
                              scheduler->problemShape.k()};
            return ClipSub(problem, GemmCoord{blockOffset.GetCoordMNK()});
        }
    };

    CATLASS_DEVICE
    RemapperA GetRemapperA(uint32_t, uint32_t localExpertIdx) const
    {
        return {this, localExpertIdx};
    }

    struct RemapperC {
        using Scheduler = BlockMmadSchedulerGmmAllToAllVTargetOrderGm<MoeConstraints>;
        Scheduler const* scheduler;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        DistMatrixCoord operator()(DistGemmCoord const& blockOffset) const
        {
            uint32_t rowOffset = scheduler->expertCache.rowOffset[blockOffset.rank()];
            uint32_t selectedGroupIdx = blockOffset.m() / scheduler->blockShape.m() -
                                        scheduler->expertCache.finishedGroup[blockOffset.rank()];
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

#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_GMM_ALLTOALLV_TARGET_ORDER_GM_HPP
