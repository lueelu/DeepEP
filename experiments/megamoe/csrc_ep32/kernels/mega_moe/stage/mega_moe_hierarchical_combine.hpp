/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */

#ifndef MEGA_MOE_HIERARCHICAL_COMBINE_H
#define MEGA_MOE_HIERARCHICAL_COMBINE_H

#include "csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"
#include "kernels/mega_moe_common/common/mega_moe_utils.hpp"

namespace MegaMoeImpl {

// Private scalar state of one of the eight per-rank QP owners. One uint32 mask covers
// every supported top-k slot of a token. Shared epoch flags are never cleared:
// only this private pending mask changes as rows are submitted to the NIC.
struct HierarchicalCombineProgress {
    uint32_t pendingTopkMasks[HIERARCHICAL_MAX_TOKENS + 1U]{};
    // Last observed ready mask, used only by trace diagnostics.
    uint32_t deferredTopkMasks[HIERARCHICAL_MAX_TOKENS + 1U]{};
    uint32_t expectedRows = 0U;
    uint32_t sentRows = 0U;
    uint32_t nextRow = 0U;
    uint32_t owned = 0U;
    uint32_t tokens[HIERARCHICAL_MAX_TOKENS]{};
    bool initialized = false;
};

// The caller has already acquired the paired source's metadata during Dispatch.
// Count only experts on this gateway's host. Original (token, top-k) coordinates
// are retained, including holes and duplicate destination ranks.
__aicore__ inline void InitializeHierarchicalCombineProgressRange(const MoeStageCommonConfig& common,
                                                                  const Params& params,
                                                                  HierarchicalCombineProgress& progress,
                                                                  uint32_t firstToken, uint32_t endToken,
                                                                  AscendTimerDevice* timer = nullptr)
{
    if (progress.initialized) {
        return;
    }
    AscendTimerFixedScope initialization(timer, AscendTimer::AIV_COMBINE_INIT);
    const uint32_t localHost = HierarchicalHost(common.rankId);
    const uint32_t sourceHost = HierarchicalHost(HierarchicalGateway(common.rankId));
    const uint64_t sourceSlotOffset = static_cast<uint64_t>(sourceHost) * CalcRailTopkIdsSlotBytes(params.tilingData);
    GlobalTensor<int32_t> sourceTopk;
    sourceTopk.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + sourceSlotOffset));
    constexpr uint32_t scratchWords = MEGAMOE_ROCE_SCRATCH_BYTES / sizeof(int32_t);
    LocalTensor<int32_t> topkScratch(TPosition::VECCALC, params.tilingData->roceScratchUbOffset, scratchWords);
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    progress.expectedRows = 0U;
    for (uint32_t token = firstToken; token < endToken; ++token) {
        progress.pendingTopkMasks[token] = 0U;
        progress.deferredTopkMasks[token] = 0U;
    }
    // Dispatch transferred whole 512B packets. Read each in two aligned chunks
    // using the owner's reserved 256B UB; the last word is a marker, not an ID.
    // This runs only before the first Combine Post, after Dispatch is drained.
    const uint32_t rowBegin = firstToken * common.topK;
    const uint32_t rowEnd = endToken * common.topK;
    for (uint32_t packet = rowBegin / RAIL_PACKET_DATA_WORDS;
         rowBegin < rowEnd && packet * RAIL_PACKET_DATA_WORDS < rowEnd; ++packet) {
        const uint32_t firstRow = packet * RAIL_PACKET_DATA_WORDS;
        const uint32_t remaining = rowEnd - firstRow;
        const uint32_t validPacket = remaining < RAIL_PACKET_DATA_WORDS ? remaining : RAIL_PACKET_DATA_WORDS;
        const uint32_t beginWord = rowBegin > firstRow ? rowBegin - firstRow : 0U;
        // Read only aligned 256B chunks intersecting this lane. Never interpret
        // the packet's sequence marker or neighbouring lanes' IDs as routes.
        for (uint32_t firstWord = beginWord / scratchWords * scratchWords; firstWord < validPacket;
             firstWord += scratchWords) {
            SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
            DataCopy(topkScratch, sourceTopk[packet * RAIL_PACKET_WORDS + firstWord], scratchWords);
            SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
            const uint32_t left = validPacket - firstWord;
            const uint32_t valid = left < scratchWords ? left : scratchWords;
            const uint32_t firstValid = beginWord > firstWord ? beginWord - firstWord : 0U;
            for (uint32_t i = firstValid; i < valid; ++i) {
                const int32_t expert = topkScratch.GetValue(i);
                if (expert >= 0 && static_cast<uint32_t>(expert) / expertsPerHost == localHost) {
                    const uint32_t row = firstRow + firstWord + i;
                    progress.pendingTopkMasks[row / common.topK] |= uint32_t(1) << (row % common.topK);
                    ++progress.expectedRows;
                }
            }
        }
    }
    // Finish scalar reads before the RDMA backend reuses this UB for its WQE.
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    progress.initialized = true;
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_HIERARCHICAL_COMBINE_H
