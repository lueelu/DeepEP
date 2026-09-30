/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_TOKEN_DISPATCH_H
#define MEGA_MOE_TOKEN_DISPATCH_H

/*
 * 功能：按真实路由搬运 Dispatch 数据并发布 GMM1 ready；输入为专家 slice 和本核行范围。
 * 输出：与原 balance+rotate 一致的 token/scale/metadata 落位。
 * 保留 Prefill 的规则配额快路径；索引批量预取、token 环跨 peer 复用，不增加 UB 或核间同步。
 */

#include "../common/mega_moe_utils.hpp"
#include "../common/mega_moe_peer_rows.hpp"
#include "mega_moe_token_quant.hpp"
#include "mega_moe_count_arrival.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch 的阶段私有配置。任务分工(BlockJobContext)与 count 槽位(BlockWorkspaceContext)
// 与 GMM 阶段共用编排层的同一份实例，不在此重复持有。
struct TokenDispatchConfig {
    uint64_t maxOutputSize;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint64_t routeIndexAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t ranksPerNode;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    PeerRowLimitsCache peerRowLimitsCache{};  // 标量对象状态；与可复用的 UB Tensor 内容分离。
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
};

// 当前物理 AIV1 在单个专家内负责的左闭右开 row 区间。
struct ExpertDispatchCoreRange {
    uint32_t expertGlobalRowBegin;
    uint32_t localExpertRowBegin;
    uint32_t localExpertRowEnd;
};

// 状态仅活到当前专家的 owner 范围结束；发布 ready 前必须排空，不能带入 SwiGLU 的 UB。
struct DispatchCopyProgress {
    uint32_t issuedRows = 0U;
    int32_t pendingRow = -1;
};

struct DispatchRouteSegment {
    uint32_t sourceRank;
    int32_t dstRow;
    uint32_t indexOffset;
    int32_t rowCount;
};

struct DispatchRouteBatch {
    DispatchRouteSegment segments[MAX_DISPATCH_BUFFER_COUNT];
    uint32_t count = 0U;
    uint32_t usedItems = 0U;
};

// 由 tiling/peermem/quant 配置装配 Token Dispatch 阶段配置（普通与 wave 编排模板共用）。
__aicore__ inline TokenDispatchConfig CreateTokenDispatchConfig(const Params& params,
                                                                const QuantProcessConfig& quantProcessConfig)
{
    uint64_t quantWinOffset =
        static_cast<uint64_t>(params.peermemInfo.quantTokenScalePtr - params.peermemInfo.rankSyncInWorldPtr);
    return {.maxOutputSize = params.tilingData->maxOutputSize,
            .bufferConfig = params.tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .quantWinOffset = quantWinOffset,
            .quantTokenAlignBytes = quantProcessConfig.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig.quantTokenScaleAlignBytes,
            .ranksPerNode = params.tilingData->dispatchRanksPerNode};
}

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(const TokenDispatchConfig& context,
                                                                    const TokenDispatchScratch<ActivationType>& scratch,
                                                                    int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig& context, const Params& params,
                                                     TokenDispatchScratch<ActivationType>& scratch, int32_t bufferIdx,
                                                     int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType>& remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(params.tilingData->topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(
        TOPK_INDEX, topkIndex % static_cast<int32_t>(params.tilingData->topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// copyIdx 是 workspace 绝对行号；待写行可能来自上一个 peer，不能再读取已复用的 route-index 缓冲。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig& context, const Params& params,
                                                     TokenDispatchScratch<ActivationType>& scratch, int32_t bufferIdx,
                                                     int32_t copyIdx)
{
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType*>(params.workspaceInfo.dispatchRevDataPtr));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType*>(params.workspaceInfo.dispatchRevScalePtr));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.metaInfoPtr));
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkSlot = scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].GetValue(TOPK_INDEX);
        int32_t weightBits = weightBitsTensor.GetValue(static_cast<uint32_t>(topkSlot));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[copyIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[copyIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[copyIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// 环形槽持续跨 peer 使用：先发起下一行读，再写出上一行，避免每个小 peer 段都排空 MTE。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(const TokenDispatchConfig& context, const Params& params,
                                                    GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                                    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx,
                                                    int32_t copyNum, DispatchCopyProgress& progress)
{
    const MegaMoeDispatchBufferConfig& bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType*>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    for (int32_t issueIdx = 0; issueIdx < copyNum; ++issueIdx) {
        // 单槽没有读写重叠空间，必须先写出；否则 reuse 会等一个尚未发布的 MTE3 事件。
        if (bufferCount == 1 && progress.pendingRow >= 0) {
            StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, scratch,
                                                                                               0, progress.pendingRow);
            progress.pendingRow = -1;
        }
        int32_t issueBufferIdx = progress.issuedRows % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        if (progress.issuedRows < static_cast<uint32_t>(bufferCount)) {
            FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx,
                                                                      topkIndex, remoteRankIdx, remoteRankGlobalTensor);
        } else {
            FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx,
                                                                     topkIndex, remoteRankIdx, remoteRankGlobalTensor);
        }
        if (progress.pendingRow >= 0) {
            StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, params, scratch, (progress.issuedRows - 1U) % bufferCount, progress.pendingRow);
        }
        progress.pendingRow = rowDstOffset + issueIdx;
        ++progress.issuedRows;
    }
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void DrainDispatchCopyBuffers(const TokenDispatchConfig& context, const Params& params,
                                                TokenDispatchScratch<ActivationType>& scratch,
                                                const DispatchCopyProgress& progress)
{
    uint32_t bufferCount = context.bufferConfig.bufferCount;
    if (progress.pendingRow >= 0) {
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, (progress.issuedRows - 1U) % bufferCount, progress.pendingRow);
    }
    uint32_t used = progress.issuedRows < bufferCount ? progress.issuedRows : bufferCount;
    for (uint32_t bufferIdx = 0U; bufferIdx < used; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 发布一个 source-rank 段覆盖到的 GMM1 tile ready 计数。
__aicore__ inline void PublishGmm1TileReady(const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                            uint32_t localExpertId, int32_t gmm1TileRowCount,
                                            int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{
    if (segmentExpertRowBegin >= segmentExpertRowEnd) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t* gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * syncLayout.dispatchFlagSlotCountPerExpert;
    int32_t firstTileIdx = segmentExpertRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentExpertRowEnd - 1) / gmm1TileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileExpertRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileExpertRowEnd = tileExpertRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin =
            segmentExpertRowBegin > tileExpertRowBegin ? segmentExpertRowBegin : tileExpertRowBegin;
        int32_t overlapRowEnd = segmentExpertRowEnd < tileExpertRowEnd ? segmentExpertRowEnd : tileExpertRowEnd;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE, overlapRowEnd - overlapRowBegin);
    }
}

// route-index 使用独立的 MTE2_S 事件 6，避免与持续存活的 token 槽 0..5 冲突。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void FlushDispatchRouteBatch(const TokenDispatchConfig& context, const Params& params,
                                               GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                               DispatchRouteBatch& batch, DispatchCopyProgress& progress)
{
    if (batch.count == 0U) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, MAX_DISPATCH_BUFFER_COUNT>();
    for (uint32_t i = 0U; i < batch.count; ++i) {
        const auto& segment = batch.segments[i];
        CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, segment.dstRow, segment.sourceRank, segment.indexOffset,
            segment.rowCount, progress);
    }
    batch.count = 0U;
    batch.usedItems = 0U;
}

/*
 * 保持 Prefill 的 peer 选择、源 ordinal 和目标行不变，只合并多个 peer 的索引读取屏障。
 * 每段 UB 起点按 32B 对齐；容不下完整段时先消费旧 batch，不为填满 UB 切碎小 peer。
 */
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline int32_t DispatchRankTokens(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                             const Params& params, GM_ADDR* winRankAddr,
                                             TokenDispatchScratch<ActivationType>& scratch, uint32_t localExpertId,
                                             int32_t expertGlobalRowBegin, uint32_t remoteRankIdx,
                                             int32_t destinationRowBegin, int32_t segmentMatchOrdinalBegin,
                                             int32_t segmentMatchOrdinalEnd, DispatchRouteBatch& batch,
                                             DispatchCopyProgress& progress)
{
    constexpr uint32_t itemsPerBlock = ALIGN_32 / sizeof(int32_t);
    const uint32_t capacity = static_cast<uint32_t>(context.bufferConfig.routeItemsPerBatch);
    int32_t processed = 0;
    const int32_t rowCount = segmentMatchOrdinalEnd - segmentMatchOrdinalBegin;
    while (processed < rowCount) {
        const uint32_t remaining = static_cast<uint32_t>(rowCount - processed);
        const uint32_t take = remaining < capacity ? remaining : capacity;
        const uint32_t padded = Ops::Base::CeilAlign(take, itemsPerBlock);
        if (batch.count == MAX_DISPATCH_BUFFER_COUNT || batch.usedItems + padded > capacity) {
            FlushDispatchRouteBatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, winRankAddr,
                                                                                         scratch, batch, progress);
        }
        const uint64_t slotOffset =
            (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * context.routeIndexAlignSize;
        GlobalTensor<int32_t> remoteRouteIndexGlobal;
        remoteRouteIndexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            params.peermemInfo.maskRecvPtr + slotOffset +
            static_cast<uint64_t>(segmentMatchOrdinalBegin + processed) * sizeof(int32_t)));
        DataCopyPad(scratch.validTopkIndexTensor[batch.usedItems], remoteRouteIndexGlobal,
                    {1U, take * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        batch.segments[batch.count++] = {remoteRankIdx, expertGlobalRowBegin + destinationRowBegin + processed,
                                         batch.usedItems, static_cast<int32_t>(take)};
        batch.usedItems += padded;
        processed += static_cast<int32_t>(take);
    }
    return processed;
}

/*
 * 在当前专家的 source-rank 累计行数中二分查找 globalRowIdx 所属的首个 rank。
 * 返回值作为 DispatchOwnedExpertRows 的扫描起点，后续只访问与本核范围相交的连续 rank 段。
 */
template <typename ActivationType>
__aicore__ inline uint32_t FindDispatchSourceRank(const MoeStageCommonConfig& common,
                                                  const TokenDispatchScratch<ActivationType>& scratch,
                                                  uint32_t localExpertId, uint32_t globalRowIdx)
{
    uint32_t expertRankBegin = localExpertId * common.worldSize;
    uint32_t searchBegin = expertRankBegin;
    uint32_t searchEnd = expertRankBegin + common.worldSize;
    while (searchBegin < searchEnd) {
        uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(middle));
        if (rankGlobalRowEnd <= globalRowIdx) {
            searchBegin = middle + 1U;
        } else {
            searchEnd = middle;
        }
    }
    return searchBegin - expertRankBegin;
}

/*
 * 搬运当前物理 AIV1 在一个专家内负责的连续 row。
 * 该范围先按 source-rank prefix 拆段，再从 compact index 槽搬运 token/metadata；全部 rank 段完成后，
 * 按实际覆盖行数累加 GMM1 tile ready。AtomicAdd 允许同一 tile 被多个 range 分段发布。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                               const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                               GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                               uint32_t localExpertId, const ExpertDispatchCoreRange& coreRange)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    // prefix 单调递增，二分找到与本核首行相交的第一个 source rank，避免从 rank 0 线性扫描。
    uint32_t sourceRankIdx = FindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);
    int32_t dispatchedRowCount = 0;
    DispatchRouteBatch batch{};
    DispatchCopyProgress progress{};

    while (sourceRankIdx < common.worldSize) {
        uint32_t prefixIndex = localExpertId * common.worldSize + sourceRankIdx;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex));
        uint32_t rankGlobalRowBegin =
            prefixIndex == 0U ? 0U : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex - 1U));
        if (rankGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }

        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > rankGlobalRowBegin ? coreGlobalRowBegin : rankGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < rankGlobalRowEnd ? coreGlobalRowEnd : rankGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            int32_t rankDispatchedRowCount = DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, params, winRankAddr, scratch, localExpertId,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                static_cast<int32_t>(overlapGlobalRowBegin - coreRange.expertGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin), batch, progress);
            dispatchedRowCount += rankDispatchedRowCount;
        }
        ++sourceRankIdx;
    }
    FlushDispatchRouteBatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, winRankAddr, scratch,
                                                                                 batch, progress);
    DrainDispatchCopyBuffers<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, scratch, progress);
    // 所有 source-rank 子段完成后统一发布本核贡献；GMM1 按 tile 累计到完整行数后即可启动。
    PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount,
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowBegin) + dispatchedRowCount);
}

// 原 prefix 仍表示 [expert][source rank] 的真实路由数量，只改变 Dispatch 的目标行排列。
struct DispatchPeerCounts {
    LocalTensor<int32_t> prefix;
    uint32_t expertRankBegin;

    __aicore__ inline uint32_t Get(uint32_t peer) const
    {
        uint32_t index = expertRankBegin + peer;
        int32_t begin = index == 0U ? 0 : prefix.GetValue(index - 1U);
        return static_cast<uint32_t>(prefix.GetValue(index) - begin);
    }
};

// 同一 slice 内各 peer 分段取数，直接落入原连续工作区；GMM/timer 继续使用原 slice 边界。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchPeerBalancedRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                                const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                                GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                                uint32_t expertIdx, const ExpertDispatchCoreRange& coreRange,
                                                uint32_t sliceBegin, uint32_t sliceEnd)
{
    DispatchRouteBatch batch{};
    DispatchCopyProgress progress{};
    DispatchPeerCounts counts{scratch.cumsumInfoTensor, expertIdx * common.worldSize};
#if MEGAMOE_PREFILL_PEER_FAST_PATH
    const PeerRowLimits& limits =
        GetCachedPeerRowLimits(scratch.peerRowLimitsCache, counts, expertIdx, common.worldSize);
    bool regular = CanUseRegularPeerSlice(limits, common.worldSize, sliceBegin, sliceEnd);
#else
    PeerRowLimits limits = GetPeerRowLimits(counts, common.worldSize);
    bool regular = false;
#endif
    PeerRowCut beginCut{}, endCut{};
    RegularPeerSlice regularSlice{};
    uint32_t firstOrdinal = 0U;
    if (regular) {
        regularSlice = CreateRegularPeerSlice(common.worldSize, sliceBegin, sliceEnd);
        firstOrdinal =
            FindRegularPeerOrdinal(regularSlice, common.worldSize, coreRange.localExpertRowBegin - sliceBegin);
    } else {
        // 真实 peer 已耗尽：保留原配额算法及余数消耗顺序，不能套用均匀配额假设。
        beginCut = GetPeerRowCut(counts, common.worldSize, limits, sliceBegin);
        endCut = GetPeerRowCut(counts, common.worldSize, limits, sliceEnd);
    }
    uint32_t destination = sliceBegin + (regular ? regularSlice.Prefix(firstOrdinal) : 0U);
    for (uint32_t ordinal = firstOrdinal; ordinal < common.worldSize; ++ordinal) {
        uint32_t peer = GetDispatchPeer(ordinal, common.rankId, common.worldSize, context.ranksPerNode);
        uint32_t sourceBegin, sourceEnd;
        if (regular) {
            // 直接恢复本 peer 的源区间，不再读取无关 peer 的 count 或从 ordinal=0 扫描。
            sourceBegin = regularSlice.SourceBegin(ordinal);
            sourceEnd = regularSlice.SourceEnd(ordinal);
        } else {
            uint32_t count = counts.Get(peer);
            sourceBegin = TakePeerRowPrefix(count, beginCut);
            sourceEnd = TakePeerRowPrefix(count, endCut);
        }
        uint32_t destinationEnd = destination + sourceEnd - sourceBegin;
        uint32_t overlapBegin =
            destination > coreRange.localExpertRowBegin ? destination : coreRange.localExpertRowBegin;
        uint32_t overlapEnd =
            destinationEnd < coreRange.localExpertRowEnd ? destinationEnd : coreRange.localExpertRowEnd;
        if (overlapBegin < overlapEnd) {
            uint32_t first = sourceBegin + overlapBegin - destination;
            DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, params, winRankAddr, scratch, expertIdx,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), peer, static_cast<int32_t>(overlapBegin),
                static_cast<int32_t>(first), static_cast<int32_t>(first + overlapEnd - overlapBegin), batch, progress);
        }
        destination = destinationEnd;
        if (destination >= coreRange.localExpertRowEnd) {
            break;
        }
    }
    FlushDispatchRouteBatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, winRankAddr, scratch,
                                                                                 batch, progress);
    DrainDispatchCopyBuffers<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, scratch, progress);
    // 每个 owner 仍独占连续目标行；本核所有 peer 段落地后按目标行发布一次贡献。
    PublishGmm1TileReady(syncLayout, params, expertIdx, static_cast<int32_t>(PipelineTileM),
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowEnd));
}

/*
 * Dispatch 唯一执行入口：搬运调用方给定的左闭右开专家 token 范围，并发布 GMM1 tile ready。
 * 调用方负责提前规划范围、推进 WAVE/专家位置以及在必要时恢复 prefix；本函数只做范围裁剪、AIV1 分工、
 * 专家/source-rank 拆段和数据发送。共享底层搬运与同步，仅 W4 非量化 token 路径选择 peer 均衡落位。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch,
          bool PeerBalanced = false>
__aicore__ inline void DispatchTokenRange(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                          const BlockJobContext& blockJob, const MoeSyncWorkspaceLayout& syncLayout,
                                          const Params& params, GM_ADDR* winRankAddr,
                                          TokenDispatchScratch<ActivationType>& scratch, const ExpertTokenRange& range)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || blockJob.totalJobs == 0U) {
        return;
    }

    // maxOutputSize 是 Dispatch workspace 的硬边界，先统一裁剪再做分核。
    uint32_t dispatchGlobalRowBegin = static_cast<uint32_t>(
        range.begin.globalTokenIndex < context.maxOutputSize ? range.begin.globalTokenIndex : context.maxOutputSize);
    uint32_t dispatchGlobalRowEnd = static_cast<uint32_t>(
        range.end.globalTokenIndex < context.maxOutputSize ? range.end.globalTokenIndex : context.maxOutputSize);
    if (dispatchGlobalRowEnd <= dispatchGlobalRowBegin) {
        return;
    }

    uint32_t dispatchRowCount = dispatchGlobalRowEnd - dispatchGlobalRowBegin;
    // 对整个输入范围一次连续均分；按全局起点轮转首 owner，避免余数长期集中在低编号 AIV1。
    WorkRange ownedRange =
        GetRotatedBalancedTokenRange(dispatchRowCount, blockJob.jobIndex, blockJob.totalJobs, dispatchGlobalRowBegin);
    if (ownedRange.count == 0U) {
        return;
    }
    uint32_t coreGlobalRowBegin = dispatchGlobalRowBegin + ownedRange.start;
    uint32_t coreGlobalRowEnd = coreGlobalRowBegin + ownedRange.count;

    // end 位于专家内部时需要包含该专家；恰好位于专家边界时 end.expertIdx 已指向下一专家。
    uint32_t lastExpertExclusive = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    if (lastExpertExclusive > common.moeExpertPerRank) {
        lastExpertExclusive = common.moeExpertPerRank;
    }
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < lastExpertExclusive; ++expertIdx) {
        uint32_t expertGlobalRowBegin =
            expertIdx == 0U
                ? 0U
                : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U));
        uint32_t expertGlobalRowEnd =
            static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue((expertIdx + 1U) * common.worldSize - 1U));
        if (expertGlobalRowEnd <= coreGlobalRowBegin) {
            continue;
        }
        if (expertGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }
        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > expertGlobalRowBegin ? coreGlobalRowBegin : expertGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < expertGlobalRowEnd ? coreGlobalRowEnd : expertGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            // 将本核的全局连续区间投影为当前专家内的局部 row 区间。
            ExpertDispatchCoreRange expertRange{expertGlobalRowBegin, overlapGlobalRowBegin - expertGlobalRowBegin,
                                                overlapGlobalRowEnd - expertGlobalRowBegin};
            if constexpr (PeerBalanced) {
                uint32_t sliceBegin =
                    dispatchGlobalRowBegin > expertGlobalRowBegin ? dispatchGlobalRowBegin - expertGlobalRowBegin : 0U;
                uint32_t sliceEnd = dispatchGlobalRowEnd < expertGlobalRowEnd
                                        ? dispatchGlobalRowEnd - expertGlobalRowBegin
                                        : expertGlobalRowEnd - expertGlobalRowBegin;
                DispatchPeerBalancedRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                    context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange, sliceBegin,
                    sliceEnd);
            } else {
                DispatchOwnedExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                    context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
            }
        }
    }
}

template <typename ActivationType>
__aicore__ inline void LoadMteExpertCounts(const MoeStageCommonConfig& common, const Params& params,
                                           TokenDispatchScratch<ActivationType>& scratch, uint32_t expectedEpoch)
{
    const uint32_t stride = static_cast<uint32_t>(CalcMteExpertCountRankStride(common.moeExpertPerRank));
    const uint32_t wireWords = common.worldSize * stride;
    const auto& buffers = params.tilingData->dispatchBufferConfig;
    const uint64_t availableBytes = static_cast<uint64_t>(buffers.routeItemsPerBatch) * sizeof(int32_t) +
                                    static_cast<uint64_t>(buffers.bufferCount) * buffers.copyBufferBytes;
    __gm__ int32_t* wire = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.expertCountRecvPtr);
    if (static_cast<uint64_t>(wireWords) * sizeof(int32_t) <= availableBytes) {
        const uint32_t stagingAddr = static_cast<uint32_t>(scratch.validTopkIndexTensor.GetPhyAddr());
        LocalTensor<int32_t> snapshot(TPosition::VECCALC, stagingAddr, wireWords);
        GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(wire);
        uint32_t stale = 0;
        do {
            SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
            DataCopyPad(snapshot, global, {1U, static_cast<uint32_t>(wireWords * sizeof(int32_t)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
            asc_vf_call<CheckTransposeExpertCountsVF>(
                reinterpret_cast<__ubuf__ uint32_t*>(snapshot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.cumsumInfoTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.expertTokenNumsOutTensor.GetPhyAddr()), common.worldSize,
                common.moeExpertPerRank, stride, expectedEpoch);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            stale = static_cast<uint32_t>(scratch.expertTokenNumsOutTensor.GetValue(0));
            if (stale != 0U) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
                }
            }
        } while (stale != 0U);
        return;
    }
    // Unusual small-token-buffer / large-expert-table shapes retain a bounded-UB path.
    for (uint32_t expert = 0; expert < common.moeExpertPerRank; ++expert) {
        for (uint32_t source = 0; source < common.worldSize; ++source) {
            __gm__ int32_t* address = wire + source * stride + expert;
            int32_t value = ReadGmByPassDCache(address);
            while ((static_cast<uint32_t>(value) >> 24) != expectedEpoch) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
                }
                value = ReadGmByPassDCache(address);
            }
            scratch.cumsumInfoTensor.SetValue(expert * common.worldSize + source, value & 0x00FFFFFF);
        }
    }
}

// Wave 进入逐专家流水前一次准备完整 count 表；每个物理 block 只发布一次 ready。
// W4 Wave-ahead 路径同时将 prefix 持久化到 GM，避免后续 Activation 覆盖 UB 后丢失 Dispatch 状态。
template <bool NeedCumsumReload = false, typename ActivationType>
__aicore__ inline void PrepareMoeExpertTokenCountTable(const MoeStageCommonConfig& common,
                                                       const BlockWorkspaceContext& countWorkspace,
                                                       const Params& params,
                                                       TokenDispatchScratch<ActivationType>& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;
    // 每次重新准备输入路由都失效缓存。不能仅靠 expertId，否则重复 launch 可能复用旧计数。
    scratch.peerRowLimitsCache = {};
    __gm__ int32_t* launchCountSlot0 =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES);
    int32_t expectEpoch = (ReadGmByPassDCache(launchCountSlot0) & 0x7F) | 0x80;
    if (UseDecodeRailRouting(params.tilingData)) {
        LoadMteExpertCounts(common, params, scratch, static_cast<uint32_t>(expectEpoch));
    } else {
        GlobalTensor<int32_t> expertCountGlobal;
        expertCountGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.expertCountRecvPtr));
        DataCopyPad(scratch.cumsumInfoTensor, expertCountGlobal,
                    {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();

        /*
         * 到达校验：count 槽高 8 位为发送侧写入的 launch epoch(与 rankSync 计数槽同源,见
         * send_mask.h PublishExpertCounts)。跨卡同步信号与 count 数据跨源/跨通道无到达序,
         * 同步放行不代表 count 已落地——先向量校验整个快照，失败则按原协议从 GM 轮询。
         */
        asc_vf_call<CheckExpertCountEpochVF>(
            reinterpret_cast<__ubuf__ uint32_t*>(scratch.cumsumInfoTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(scratch.expertTokenNumsOutTensor.GetPhyAddr()), rawCountElementCount,
            static_cast<uint32_t>(expectEpoch));
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        if (scratch.expertTokenNumsOutTensor.GetValue(0) != 0) {
            // 快照尚未全部到达：保留原 bypass-DCache 轮询，不引入反复 DMA 的可见性假设。
            // VF 已剥离 UB epoch，因此回退必须从 GM 重读，不能再次判断 UB 高位。
            __gm__ int32_t* rawCountGm = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.expertCountRecvPtr);
            for (uint32_t slotIdx = 0U; slotIdx < rawCountElementCount; ++slotIdx) {
                int32_t slotValue = AscendC::ReadGmByPassDCache(rawCountGm + slotIdx);
                while (((slotValue >> 24) & 0xFF) != expectEpoch) {
                    int64_t startCycle = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
                    }
                    slotValue = AscendC::ReadGmByPassDCache(rawCountGm + slotIdx);
                }
                scratch.cumsumInfoTensor.SetValue(slotIdx, slotValue & 0x00FFFFFF);
            }
        }
    }
    SyncFuncStatic<AscendC::HardEvent::S_V, SYNC_EVENT_ID2>();

    ComputeExpertCountTables(scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor, common.moeExpertPerRank,
                             common.worldSize, params.tilingData->maxOutputSize);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    if constexpr (NeedCumsumReload) {
        DataCopyPad(scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
                    {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();

    __gm__ int32_t* countTableReady =
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WriteGmByPassDCache(countTableReady, static_cast<int32_t>(1));
}

/*
 * W4 的 Activation/SwiGLU 会覆盖 AIV1 UB 0 开始的 prefix 表。调用方在执行 DispatchTokenRange 前，
 * 从当前物理 block 的 GM 备份恢复“首专家前缀 + 本次范围覆盖的专家前缀”。
 */
template <typename ActivationType>
__aicore__ inline void ReloadDispatchCumsumRange(const MoeStageCommonConfig& common,
                                                 TokenDispatchScratch<ActivationType>& scratch, uint32_t firstExpertIdx,
                                                 uint32_t lastExpertIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    if (firstExpertIdx >= common.moeExpertPerRank || firstExpertIdx > lastExpertIdx) {
        return;
    }
    if (lastExpertIdx >= common.moeExpertPerRank) {
        lastExpertIdx = common.moeExpertPerRank - 1U;
    }
    constexpr uint32_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
    uint32_t requiredBeginIndex = firstExpertIdx == 0U ? 0U : firstExpertIdx * common.worldSize - 1U;
    uint32_t alignedBeginIndex = requiredBeginIndex / INT32_PER_32B * INT32_PER_32B;
    uint32_t requiredEndIndex = (lastExpertIdx + 1U) * common.worldSize;
    uint32_t reloadElementCount = requiredEndIndex - alignedBeginIndex;
    DataCopyPad(scratch.cumsumInfoTensor[alignedBeginIndex], scratch.cumsumInfoGlobalTensor[alignedBeginIndex],
                {1U, reloadElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
}

template <typename ActivationType>
__aicore__ inline void ExportCompactExpertTokenCounts(const MoeStageCommonConfig& common,
                                                      const BlockWorkspaceContext& countWorkspace, const Params& params,
                                                      TokenDispatchScratch<ActivationType>& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || countWorkspace.blockIdx != 0U) {
        return;
    }

    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    GlobalTensor<int32_t> expertRevTokenNums;
    expertRevTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.expertRevTokenNumsPtr));
    DataCopyPad(scratch.expertTokenNumsOutTensor, expertRevTokenNums[countOffset],
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID2>();

    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID2>();
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TOKEN_DISPATCH_H
