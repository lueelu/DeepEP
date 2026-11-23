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

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H
#ifdef DEEPEP_MEGAMOE_ROCE

#include "csrc_ep32/kernels/mega_moe/stage/mega_moe_route_host_masks.hpp"
#endif

#include "../common/mega_moe_mxfp8_utils.hpp"
#include "kernels/mega_moe_common/common/mega_moe_utils.hpp"
#ifdef DEEPEP_MEGAMOE_ROCE
#include "csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

struct TokenUnpermuteConfig {
    AivJobContext job;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteScratch {
#ifdef DEEPEP_MEGAMOE_ROCE
    uint32_t acquiredQps = 0;
    int32_t incomingQp[4]{};
    uint32_t localMasks[4]{};
    uint32_t firstOutput = 0;
#endif
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
};

// 加载并转换一批 top-k 权重。
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const MoeStageCommonConfig& common, const Params& params,
                                                 TokenUnpermuteScratch& scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsTensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(float)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsBf16Tensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(bfloat16_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(scratch.topKWeightsTensor, scratch.topKWeightsBf16Tensor, RoundMode::CAST_NONE,
             batchTokenCount * common.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

// 将一个 MoE 专家的输入搬入 UB，并按 Combine 模式完成 Cast 或 MXFP8 反量化。
template <uint8_t CombineMode>
__aicore__ inline void LoadMoeExpertInput(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                          TokenUnpermuteScratch& scratch, const GlobalTensor<bfloat16_t>& expandedX,
                                          uint64_t expertInputIndex, TEventID event,
                                          LocalTensor<bfloat16_t>& dataInBf16, LocalTensor<float>& dataInFp32)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        DataCopy(dataInBf16, expandedX[expertInputIndex * common.tokenHiddenDim], common.tokenHiddenDim);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    } else {
        uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
        DataCopy(dataInBf16, expandedX[expertInputIndex * quantTokenElementCount], quantTokenElementCount);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, scratch.bf16ScaleTensor,
                                                 scratch.fp32ScaleTensor, nScale, common.tokenHiddenDim);
    }
}

// 累加一个 token 对应的全部 MoE 专家输入。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(const TokenUnpermuteConfig& context,
                                                    const MoeStageCommonConfig& common, TokenUnpermuteScratch& scratch,
                                                    int32_t tokenIdx, int32_t localIdx,
                                                    const GlobalTensor<bfloat16_t>& expandedX,
                                                    const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, common, scratch, expandedX, expertInputIndex, event, dataInBf16,
                                        dataInFp32);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if (expertIdx == 0) {
            if constexpr (TopkWeightsPrefetch) {
                DataCopy(scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
            } else {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(scratch.dataResFp32Tensor, dataInFp32, expertScale, common.tokenHiddenDim);
            }
        } else {
            if constexpr (!TopkWeightsPrefetch) {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(dataInFp32, dataInFp32, expertScale, common.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(event);
    }
}

#ifdef DEEPEP_MEGAMOE_ROCE
// Weighted gateway mode receives one FP32 sum for the remote host. Raw remote
// top-k slots remain untouched and must never be read or weighted a second time.
__aicore__ inline void AccumulateHierarchicalReducedToken(const TokenUnpermuteConfig& context,
                                                          const MoeStageCommonConfig& common, const Params& params,
                                                          TokenUnpermuteScratch& scratch, int32_t tokenIdx,
                                                          int32_t localIdx, const GlobalTensor<bfloat16_t>& expandedX,
                                                          const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    // Borrow an input slot, not the BF16 output slot: the previous token's MTE3
    // may still be reading that output. Return the slot event before normal use.
    uint32_t localMask = 0U;
    bool hasRemote = false;
    if (UseInputEarlyClosure(params.tilingData, params.inputScaleGmAddr != nullptr) && params.tilingData->aicNum == 28) {
        localMask = scratch.localMasks[tokenIdx - scratch.firstOutput];
        hasRemote = scratch.incomingQp[tokenIdx - scratch.firstOutput] >= 0;
    } else {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        auto ids = scratch.dataResFp32Tensor[bufferConfig.fp32SlotElementCount].template ReinterpretCast<int32_t>();
        GlobalTensor<int32_t> idsGm;
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdxGmAddr));
        DataCopyPad(ids, idsGm[static_cast<uint64_t>(tokenIdx) * common.topK], {1U, common.topK * 4U, 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();
        const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
        for (uint32_t k = 0U; k < common.topK; ++k) {
            const uint32_t host = static_cast<uint32_t>(ids.GetValue(k)) / expertsPerHost;
            hasRemote |= host == (1U - HierarchicalHost(common.rankId));
            if (host == HierarchicalHost(common.rankId)) {
                localMask |= uint32_t{1} << k;
            }
        }
        SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    const uint32_t allMask = 0xffffffffU >> (32U - common.topK);
    if (localMask == allMask) {
        AccumulateMoeExpertsForToken<COMBINE_NO_QUANT, false>(context, common, scratch, tokenIdx, localIdx, expandedX,
                                                              bufferConfig);
        return;
    }
    bool first = true;
    for (uint32_t k = 0U; k < common.topK; ++k) {
        if ((localMask & (uint32_t{1} << k)) == 0U) {
            continue;
        }
        const uint32_t index = (localIdx * (common.topK + common.sharedExpertNum) + k) % bufferConfig.inputBufferCount;
        const TEventID event = static_cast<TEventID>(index);
        auto inputBf16 = scratch.dataResTensor[(index + 1U) * bufferConfig.bf16SlotElementCount];
        auto inputFp32 = scratch.dataResFp32Tensor[(index + 1U) * bufferConfig.fp32SlotElementCount];
        LoadMoeExpertInput<COMBINE_NO_QUANT>(context, common, scratch, expandedX,
                                             static_cast<uint64_t>(tokenIdx) * common.topK + k, event, inputBf16,
                                             inputFp32);
        const float weight = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + k);
        SyncFuncStatic<HardEvent::S_V, EVENT_ID0>();
        PipeBarrier<PIPE_V>();
        if (first) {
            Muls(scratch.dataResFp32Tensor, inputFp32, weight, common.tokenHiddenDim);
            first = false;
        } else {
            Muls(inputFp32, inputFp32, weight, common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, inputFp32, common.tokenHiddenDim);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(event);
    }
    if (!hasRemote) {
        if (first) Duplicate(scratch.dataResFp32Tensor, 0.0F, common.tokenHiddenDim);
        PipeBarrier<PIPE_V>();
        return;
    }
    // At least one remote expert exists here. Its received sum already includes
    // every remote weight. The per-QP commit join precedes this whole stage.
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (UseInputEarlyClosure(params.tilingData, params.inputScaleGmAddr != nullptr) && params.tilingData->aicNum == 28) {
        const uint32_t qp = scratch.incomingQp[tokenIdx - scratch.firstOutput];
        if ((scratch.acquiredQps & (1U << qp)) == 0) {
            auto* commit = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.combineReduceCommitPtr +
                                                             qp * HIERARCHICAL_FLAG_BYTES);
            RoceWaitCommit(commit, params.tilingData->hierarchicalSequence, nullptr, AscendTimer::RdmaPhase::Combine,
                           HierarchicalGateway(common.rankId), nullptr, qp + 1U);
            scratch.acquiredQps |= 1U << qp;
        }
    }
    auto partial = scratch.dataResFp32Tensor[bufferConfig.fp32SlotElementCount];
    GlobalTensor<float> partialGm;
    partialGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.peermemInfo.combinePartialRecvPtr));
    DataCopy(partial, partialGm[static_cast<uint64_t>(tokenIdx) * common.tokenHiddenDim], common.tokenHiddenDim);
    SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
    if (first) {
        DataCopy(scratch.dataResFp32Tensor, partial, common.tokenHiddenDim);
    } else {
        Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, partial, common.tokenHiddenDim);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

#endif
// 等待该共享专家的 GMM2 结果就绪后累加进当前 token（等待与累加是一体动作，调用方无需分两步）。
template <uint32_t Gmm1TileM>
__aicore__ inline void AccumulateSharedExpertForToken(const MoeStageCommonConfig& common, const Params& params,
                                                      TokenUnpermuteScratch& scratch, int32_t tokenIdx,
                                                      int32_t localIdx, uint32_t sharedExpertIdx,
                                                      const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(common.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t* counterAddr = GetCombineSyncCounterAddress(
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
        tokenGroupIndex);
    const auto sharedPolicy = SelectSharedExpertPolicy(params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4,
                                                       common.sharedExpertNum, common.tokenNum);
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(common.tokenHiddenDim, sharedPolicy.gmm2TileN);
    WaitUntilGmFlagEquals(counterAddr, static_cast<int32_t>(gmm2NTilesPerGroup));

    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.sharedExpertResultPtr));
    int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) +
                                  static_cast<int32_t>(common.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 = scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(
        dataInBf16,
        sharedResult[(static_cast<uint64_t>(sharedExpertIdx) * common.tokenNum + tokenIdx) * common.tokenHiddenDim],
        common.tokenHiddenDim);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(
    const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common, const Params& params,
    TokenUnpermuteScratch& scratch, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    const GlobalTensor<bfloat16_t>& expandedX, GlobalTensor<bfloat16_t>& output, int32_t jobOffset,
    int32_t batchTokenOffset, int32_t batchTokenCount, TEventID outputBufferEvent)
{
    if constexpr (!TopkWeightsPrefetch) {
        LoadTokenUnpermuteWeights<TopkWeightsType>(common, params, scratch, jobOffset, batchTokenOffset,
                                                   batchTokenCount);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
#ifdef DEEPEP_MEGAMOE_ROCE
        if (UseHierarchicalCombineReduction(params.tilingData)) {
            AccumulateHierarchicalReducedToken(context, common, params, scratch, tokenIdx, localIdx, expandedX,
                                               bufferConfig);
        } else {
            AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, scratch, tokenIdx, localIdx,
                                                                           expandedX, bufferConfig);
        }
#else
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, scratch, tokenIdx, localIdx,
                                                                       expandedX, bufferConfig);
#endif
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
        }
        WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
        Cast(scratch.dataResTensor, scratch.dataResFp32Tensor, RoundMode::CAST_RINT, common.tokenHiddenDim);
        SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[static_cast<uint64_t>(tokenIdx) * common.tokenHiddenDim], scratch.dataResTensor,
                 common.tokenHiddenDim);
        SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

/*
 * 构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置（普通/wave/a8w4 编排共用）。
 * UB 从地址 0 起按下列顺序连续排布（N = inputBufferCount，每段起点=上一段末尾）：
 *   [dataRes    ] (N+1) 个 bf16 槽 × bf16SlotElementCount —— 累加结果 bf16 输出
 *   [dataResFp32] (N+1) 个 fp32 槽 × fp32SlotElementCount —— fp32 累加中间态
 *   [topKWeights] topKWeightsBufferBytes —— 本批 token 的 topk 权重（fp32）
 *   [权重转换区 ] 仅 TopkWeightsType==bf16 时存在（topKWeightsConversionBufferBytes）
 *   [bf16Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·2B·扩展系数)
 *   [fp32Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·4B·扩展系数)
 */
template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline MegaMoeUnpermuteBufferConfig CreateTokenUnpermuteBuffers(const TokenUnpermuteConfig& context,
                                                                           uint32_t tokenHiddenDim,
                                                                           TokenUnpermuteScratch& scratch)
{
    MegaMoeUnpermuteBufferConfig bufferConfig = context.job.jobIndex < context.fullTokenChunkJobCount
                                                    ? context.fullTokenChunkConfig
                                                    : context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) * static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) * static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor =
        LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor =
        LocalTensor<float>(TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
    }
    return bufferConfig;
}

// 原型：MegaMoe::Unpermute。执行当前 token 分区的完整 Unpermute 流水。
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                       const Params& params, TokenUnpermuteScratch& scratch,
                                       const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, context.job.jobIndex, context.job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
#ifdef DEEPEP_MEGAMOE_ROCE
    if (UseInputEarlyClosure(params.tilingData, params.inputScaleGmAddr != nullptr) && params.tilingData->aicNum == 28) {
        // Borrow a full FP32 input slot before output DMA starts. Both masks and
        // the dense ID snapshot fit into H words for the three gated shapes.
        auto ids = scratch.dataResFp32Tensor[bufferConfig.fp32SlotElementCount].template ReinterpretCast<uint32_t>();
        const uint32_t words = common.tokenNum * common.topK;
        auto masks = ids[(words + 7U) / 8U * 8U];
        GlobalTensor<uint32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(params.expertIdxGmAddr));
        DataCopy(ids, source, words);
        SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
        const uint32_t width = common.moeExpertPerRank * HIERARCHICAL_RANKS_PER_HOST;
        auto* ptr = reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
        auto* out = reinterpret_cast<__ubuf__ uint32_t*>(masks.GetPhyAddr());
        asc_vf_call<RouteHostMasksVF<false>>(ptr, out, nullptr, nullptr, common.tokenNum, common.topK,
                                             (1U - HierarchicalHost(common.rankId)) * width, width);
        SyncFuncStatic<HardEvent::V_S, EVENT_ID0>();
        uint32_t count = 0;
        scratch.firstOutput = tokenRange.start;
        for (uint32_t t = 0; t < common.tokenNum; ++t) {
            const uint32_t remoteMask = masks.GetValue(t);
            const bool remote = remoteMask != 0U;
            if (t >= tokenRange.start && t < tokenRange.start + tokenRange.count) {
                scratch.incomingQp[t - tokenRange.start] = remote ? static_cast<int32_t>(count) : -1;
                scratch.localMasks[t - tokenRange.start] = masks.GetValue((common.tokenNum + 7U) / 8U * 8U + t);
            }
            count += remote;
        }
        SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
        for (uint32_t i = 0; i < tokenRange.count; ++i)
            if (scratch.incomingQp[i] >= 0)
                scratch.incomingQp[i] = CompactOwner(params.tilingData, scratch.incomingQp[i], count);
    }
#endif
    int32_t jobLen = static_cast<int32_t>(tokenRange.count);
    int32_t jobOffset = static_cast<int32_t>(tokenRange.start);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.peermemInfo.combineSendPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.y2GmAddr));

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen; batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount = batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ? jobLen - batchTokenOffset
                                                                                          : bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, common, params, scratch, bufferConfig, expandedX, output, jobOffset, batchTokenOffset,
            batchTokenCount, outputBufferEvent);
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_UNPERMUTE_H
