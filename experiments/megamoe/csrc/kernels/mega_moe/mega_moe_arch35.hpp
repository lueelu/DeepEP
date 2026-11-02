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

/*!
 * \file mega_moe_arch35.h
 * \brief
 */

#ifndef MEGA_MOE_ARCH35_H
#define MEGA_MOE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.hpp"
#include "common/mega_moe_workspace.hpp"
#include "common/mega_moe_utils.hpp"
#include "common/ascend_timer_v2_device.hpp"
#include "common/mega_moe_exception_dump_policy.hpp"
#include "epilogue/block_epilogue_activation_mx_quant.hpp"
#include "stage/mega_moe_token_quant.hpp"
#include "stage/mega_moe_send_mask.hpp"
#include "stage/mega_moe_workspace_reset.hpp"
#include "stage/mega_moe_shared_expert_input.hpp"
#include "stage/mega_moe_token_dispatch.hpp"
#include "stage/mega_moe_gmm1_activation.hpp"
#include "stage/mega_moe_gmm2_combine.hpp"
#include "stage/mega_moe_unpermute.hpp"
#include "mc2_common/quantize_functions.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass                                                                            \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData* tilingData,
                                GM_ADDR tilingGM, GM_ADDR timerGM = nullptr);

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();
    __aicore__ inline void InitTokenUnpermuteConfig();

protected:
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void InitQuantTokenBufferConfig();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void ProcessInputPreparationStage();
    template <bool SourceGmm1Fused>
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState& state, GMMAddrInfo& gmmAddrInfo,
                                                   uint32_t& startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                   uint32_t sliceTokenCount,
                                                   WaveCombineBufferConfig& combineBufferConfig,
                                                   uint32_t& combineRowSequence, int32_t& pairwiseTileSequence,
                                                   bool isFinalCombine, bool reverseBlocks = false);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState& state, uint32_t expertIdx);
    __aicore__ inline void ProcessSharedExpertGmm1(int32_t& gmm1TileReadySequence);
    __aicore__ inline void ProcessSharedExpertGmm2();
    __aicore__ inline void ProcessSharedExpertActivation();
    __aicore__ inline void ProcessSharedExpertTail();
    struct SharedActivationCursor {
        uint32_t expert = 0U;
        uint32_t nextTile = 0U;
        uint32_t tileCount = 0U;
        bool gmm1Ready = false;
        bool done = false;
    };
    __aicore__ inline bool TryRunSharedExpertActivation(SharedActivationCursor& cursor);
    template <typename Derived>
    __aicore__ inline void ProcessWave(Derived& derived);

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    MoeStageCommonConfig commonConfig_{};
    GmmExecutionConfig gmmExecutionConfig_{};
    BlockWorkspaceContext countWorkspace_{};
    MoeSyncWorkspaceLayout syncWorkspaceLayout_{};
    // 单次 reset 批量元素数（与 syncWorkspaceLayout_ 描述的清零区域配套）。
    int32_t resetBatchElementCount_ = 0;
    // 输入准备各 stage（quant/mask/reset/shared-prepare/unpermute）共用的逐 AIV 任务分工。
    AivJobContext aivJob_{};
    TokenDispatchConfig tokenDispatchConfig_;
    SendMaskConfig sendMaskConfig_;
    QuantProcessConfig quantProcessConfig_;
    QuantTokenBufferConfig quantTokenBufferConfig_;
    AivJobContext waveCombineJob_{};
    TokenUnpermuteConfig tokenUnpermuteConfig_;

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint16_t gmm2PingPongIdx_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    SharedExpertPolicy sharedPolicy_{};
    bool decodeInputReady_ = false;
    uint32_t moeExpertPerRank_ = 0;
    uint32_t mGroupsPerWave_ = 1U;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;
    QuantProcessScratch<ActivationType> quantScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，ActivationQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using ActivationQuantOutType =
        typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // ActivationQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;

    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                                         TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_M_256, L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    WaveCombineScratch waveCombineScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
    MegaMoeImpl::ExceptionDumpEngine exceptionDump_;
    AscendTimerDevice timer_;
    __gm__ MegaMoeImpl::GmmLoopCount* gmmLoopCount_{nullptr};
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<ActivationType, QuantScaleOutType, TopkWeightsPrefetch, A_ELEMS_PER_BYTE>(k_, params_);
    sendMaskConfig_ = CreateSendMaskConfig(params_, sharedPolicy_.routeAiv1Only ? blockIdx_ : aivCoreIdx_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    waveCombineJob_ = {.jobIndex = blockIdx_, .totalJobs = blockNum_};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteConfig()
{
    tokenUnpermuteConfig_ = {.job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
                             .quantTokenSizeBytes = quantTokenBufferConfig_.quantTokenSizeBytes,
                             .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
                             .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
                             .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData* tilingData, GM_ADDR tilingGM, GM_ADDR timerGM)
{
    timer_.Init(timerGM);
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    sharedPolicy_ = SelectSharedExpertPolicy(ENABLE_A8W4, sharedExpertNum_, tilingData->bs);
    decodeInputReady_ = UseDecodeInputReady(ENABLE_A8W4, tilingData->bs);
    mGroupsPerWave_ = tilingData->mGroupsPerWave;
    gmm2PingPongIdx_ = 0;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    GM_ADDR dumpBase = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[rankId_]);
    for (int i = 0; i < worldSize_; i++) {
        g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]) + EXCEPTION_DUMP_REGION_SIZE;
    }
    params_.aGmAddr = x;
    params_.inputScaleGmAddr = scales;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1, .weightScales1 = weightScales1, .weight2 = weight2, .weightScales2 = weightScales2};
    if (sharedExpertNum_ > 0U) {
        sharedWeightTensorListAddrs_ = {.weight1 = sharedWeight1,
                                        .weightScales1 = sharedWeightScales1,
                                        .weight2 = sharedWeight2,
                                        .weightScales2 = sharedWeightScales2};
    }
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = tilingData->bs,
                     .topK = tilingData->topK,
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = tilingData->hiddenDim};
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    int32_t activationFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, static_cast<int64_t>(L1_TILE_M_256))) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig(params_, quantProcessConfig_);
    InitSyncWorkspaceConfigs(dispatchFlagSlotsPerExpert, activationFlagSlotsPerExpert);
    InitGmmConfigs();
    InitQuantTokenBufferConfig();
    InitTokenUnpermuteConfig();

#if !defined(ASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP)
    gmmLoopCount_ =
        MegaMoeImpl::RegisterMegaMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData, params_.peermemInfo,
                                                  reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
#else
    (void)dumpBase;
    (void)tilingGM;
    gmmLoopCount_ = nullptr;
#endif
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    const TokenDispatchConfig& context = tokenDispatchConfig_;
    TokenDispatchScratch<ActivationType>& scratch = tokenDispatchScratch_;
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.expertRevTokenNumsPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    const MegaMoeDispatchBufferConfig& bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = commonConfig_.tokenHiddenDim / A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt = Ops::Base::CeilDiv(static_cast<int64_t>(commonConfig_.tokenHiddenDim),
                                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                              MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.worldSize * commonConfig_.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.cumsumInfoPtr +
                                              static_cast<uint64_t>(cumsumInfoTensorSize) * countWorkspace_.blockIdx));
    }

    uint32_t cumsumInfoTensorAddr = 0U;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // compact route 已消除 mask 扫描，只保留一份接收 index batch。
    uint32_t validTopkIndexTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    scratch.copyTmpBaseAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;
    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.moeExpertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoTensorSize / sizeof(int32_t));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitQuantTokenBufferConfig()
{
    quantTokenBufferConfig_ = {.quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantTokenBufferConfig_ = CreateQuantTokenBufferConfig(k_);
    }
}

// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(params_.workspaceInfo.flagResetElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount = static_cast<uint64_t>(params_.workspaceInfo.gmm1TileStatusElementCount);
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH)
                                         ? static_cast<int32_t>(resetElementCountPerCore)
                                         : DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t routeCoreCount = sharedPolicy_.routeAiv1Only ? blockNum_ : blockAivNum_;
    uint32_t expertPerCoreMax = CalcSendMaskExpertCapacity(params_.tilingData, routeCoreCount);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // Prefill retains quotient/remainder assignment; decode reserves the maximum
    // relay expert range. Host SetSendMaskConfigs uses the same capacity helper.
    const SendMaskBufferConfig& bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t topkIndexTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    sendMaskScratch_.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIndexTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetBatchElementCount_ = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    if (sharedExpertNum_ > 0U) {
        sharedExpertPrepareScratch_.copyBuffer0 = quantScratch_.xOutTensor0;
        sharedExpertPrepareScratch_.copyBuffer1 = quantScratch_.xOutTensor1;
    }

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t routeRingAddr = xInAlignAddr2 + xInAlignSize;
    /*
     * h%64==32（scale 组数为奇数）时，量化链路存在三处"计算不覆盖、却进入定长通信记录或参与
     * 计算"的跨 launch UB 残留：xIn 尾部（进 ComputeMaxExp 尾块 mask 内 lane）、xOut 记录的
     * scale 偶数补齐槽（ComputeScale 掩码写不到）、mxTemp 的 halfScale 补偶槽（被
     * ComputeFp8Data 尾块 E2B 广播进乘法，0×NaN 仍为 NaN）。残留呈 NaN/大指数位型时整行
     * GMM 输出被污染为 NaN，最终 combine 输出成块清零（首轮 UB 干净故仅多轮调用时显形）。
     * 此处对 [mxTempTensorAddr, routeRingAddr) 连续 span（mxTemp/xOut0/xOut1/xIn0/xIn1 五段
     * 量化 scratch）一次性清零：span 边界取 routeRingAddr、与本函数的地址推进公式同源，
     * 中间插入新 buffer 时范围自动跟随；有效区随后每 token 均被完整覆写，残留位恒为良性 0。
     * h%64==0 时不存在上述缝隙，本清零不改变任何可观测行为。
     */
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (routeRingAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((routeRingAddr - mxTempTensorAddr) / sizeof(int16_t)));
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    uint32_t routeRingBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.routeRingTensor = LocalTensor<uint8_t>(TPosition::VECCALC, routeRingAddr, routeRingBytes);
    uint32_t sendCntAccAddr = routeRingAddr + routeRingBytes;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    return CreateTokenUnpermuteBuffers<TopkWeightsType, CombineQuantMode>(
        tokenUnpermuteConfig_, commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1(int32_t& gmm1TileReadySequence)
{
    if constexpr (ENABLE_A8W4 && g_coreType == AIV) {
        if (GetSubBlockIdx() != 0U) {
            return;
        }
    }
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);

    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t pingpongIdx = 0U;
    GmmRuntimeState runtimeState{startBlockIdx, vecSetSyncCom, pingpongIdx};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                           ENABLE_A8W4>(commonConfig_, gmmExecutionConfig_, params_.workspaceInfo,
                                                        sharedWeightTensorListAddrs_, sharedEpilogueOp_, gmmAddrInfo,
                                                        sharedExpertIdx);
        if constexpr (ENABLE_A8W4) {
            // Keep shared metadata and event counts separate from routed expert tiles.
            timer_.SetWave(AscendTimer::NO_WAVE);
            timer_.SetSlice(sharedExpertIdx, 0U, commonConfig_.tokenNum);
            gmmAddrInfo.timer = timer_.Enabled() ? &timer_ : nullptr;
            // Shared activation uses its own completion protocol, never the routed tile sequence.
            gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        }
        RunSharedExpertGmm1ActivationStage<QuantOutType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                           ENABLE_A8W4, GMM1_TILE_M>(
            commonConfig_, gmmExecutionConfig_, params_, sharedEpilogueOp_, gmmAddrInfo, problemShape, runtimeState,
            sharedExpertIdx, &gmm1TileReadySequence);
        if constexpr (ENABLE_A8W4) {
            timer_.ClearSlice();
        }
    }
    EndSync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    if constexpr (ENABLE_A8W4 && CombineQuantMode == COMBINE_NO_QUANT && g_coreType == AIC) {
        // All shared outputs owned by this AIC are visible, including the zero-tile case.
        SyncFuncStatic<HardEvent::FIX_S, SYNC_EVENT_ID0>();
        WriteGmByPassDCache(
            reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagSharedGmm1DonePtr) + blockIdx_ * INT_CACHELINE,
            1);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertActivation()
{
    // Routed stages have reused UB: reconstruct activation scratch before consuming the stored shared GMM1 output.
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);
    ProblemShape shape;
    Get<M_VALUE>(shape) = commonConfig_.tokenNum;
    Get<N_VALUE>(shape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(shape) = commonConfig_.tokenHiddenDim;
    const uint32_t tileCount =
        Ops::Base::CeilDiv(commonConfig_.tokenNum, GMM1_TILE_M) *
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    GMMAddrInfo addr{};
    for (uint32_t expert = 0; expert < sharedExpertNum_; ++expert) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                           ENABLE_A8W4>(commonConfig_, gmmExecutionConfig_, params_.workspaceInfo,
                                                        sharedWeightTensorListAddrs_, sharedEpilogueOp_, addr, expert);
        timer_.SetWave(AscendTimer::NO_WAVE);
        timer_.SetSlice(expert, 0U, commonConfig_.tokenNum);
        addr.timer = timer_.Enabled() ? &timer_ : nullptr;
        for (uint32_t tile = blockIdx_; tile < tileCount; tile += blockNum_) {
            RunReadyGmm1ActivationTileA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                           GMM1_TILE_M, false>(
                sharedEpilogueOp_, shape, addr, gmmExecutionConfig_.blockJob, tile, 0U, AscendTimer::AIV_SHARED_SWIGLU);
        }
        timer_.ClearSlice();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::TryRunSharedExpertActivation(SharedActivationCursor& cursor)
{
    if (cursor.done) {
        return false;
    }
    if (!cursor.gmm1Ready) {
        // Never block AIV1: routed activation/Dispatch/Combine must still make progress.
        auto* flags = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagSharedGmm1DonePtr);
        for (uint32_t core = 0U; core < blockNum_; ++core) {
            if (AscendC::ReadGmByPassDCache(flags + core * INT_CACHELINE) < 1) {
                return false;
            }
        }
        cursor.gmm1Ready = true;
        cursor.nextTile = blockIdx_;
        cursor.tileCount =
            Ops::Base::CeilDiv(commonConfig_.tokenNum, GMM1_TILE_M) *
            Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    }
    if (cursor.nextTile < cursor.tileCount) {
        // AIV1 did not enter shared GMM1. Rebind its separate epilogue and shared GM views;
        // UB is reused sequentially by Dispatch, routed/shared activation and Combine.
        typename SharedBlockEpilogue::Params epilogueParams{
            .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
            .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
            .clampLimit = params_.tilingData->clampLimit,
            .actMode = params_.tilingData->actMode,
            .actSubMode = params_.tilingData->actSubMode,
            .activationAlpha = params_.tilingData->activationAlpha,
            .activationBeta = params_.tilingData->activationBeta};
        sharedEpilogueOp_.Init(epilogueParams);
        ProblemShape shape;
        Get<M_VALUE>(shape) = commonConfig_.tokenNum;
        Get<N_VALUE>(shape) = commonConfig_.gmm1OutputDim;
        Get<K_VALUE>(shape) = commonConfig_.tokenHiddenDim;
        GMMAddrInfo addr{};
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                           ENABLE_A8W4>(commonConfig_, gmmExecutionConfig_, params_.workspaceInfo,
                                                        sharedWeightTensorListAddrs_, sharedEpilogueOp_, addr,
                                                        cursor.expert);
        timer_.SetWave(AscendTimer::NO_WAVE);
        timer_.SetSlice(cursor.expert, 0U, commonConfig_.tokenNum);
        addr.timer = timer_.Enabled() ? &timer_ : nullptr;
        RunReadyGmm1ActivationTileA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                       GMM1_TILE_M, false>(sharedEpilogueOp_, shape, addr, gmmExecutionConfig_.blockJob,
                                                           cursor.nextTile, 0U, AscendTimer::AIV_SHARED_SWIGLU);
        timer_.ClearSlice();
        cursor.nextTile += blockNum_;
        if (cursor.nextTile >= cursor.tileCount) {
            ++cursor.expert;
            cursor.nextTile = blockIdx_;
        }
    } else {
        // This AIV1 owns no tile for any shared expert, but must publish its completion.
        cursor.expert = sharedExpertNum_;
    }
    if (cursor.expert >= sharedExpertNum_) {
        PipeBarrier<PIPE_ALL>();
        SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID0>();
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagSharedActivationDonePtr) +
                                blockIdx_ * INT_CACHELINE,
                            1);
        cursor.done = true;
    }
    // One tile at most; recheck higher-priority tasks before consuming the next tile.
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertTail()
{
    if constexpr (ENABLE_A8W4) {
        const auto& workspace = params_.workspaceInfo;
        if constexpr (g_coreType == AIC) {
            // Includes the deferred final routed GMM2 wave and cores without any owned tiles.
            SyncFuncStatic<HardEvent::FIX_S, SYNC_EVENT_ID0>();
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                WriteGmByPassDCache(
                    reinterpret_cast<__gm__ int32_t*>(workspace.flagRoutedFfnDonePtr) + blockIdx_ * INT_CACHELINE, 1);
            }
        } else {
            if (GetSubBlockIdx() != 0U) {
                return;  // AIV1 continues Combine independently.
            }
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                WaitForAllCoreFlags(workspace.flagRoutedFfnDonePtr, blockNum_);
                PipeBarrier<PIPE_ALL>();
                ProcessSharedExpertActivation();
                // Both activation data and MX scales must be in GM before any shared GMM2 reader starts.
                PipeBarrier<PIPE_ALL>();
                SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID0>();
                WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace.flagSharedActivationDonePtr) +
                                        blockIdx_ * INT_CACHELINE,
                                    1);
            }
        }
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            WaitForAllCoreFlags(workspace.flagRoutedFfnDonePtr, blockNum_);
        }
        if constexpr (g_coreType == AIC || CombineQuantMode != COMBINE_NO_QUANT) {
            // Only AIC consumes shared activation, whose K tiles may come from any AIV1.
            WaitForAllCoreFlags(workspace.flagSharedActivationDonePtr, blockNum_);
        }
        // Non-quantized AIV0 enters the weight prologue directly after the previous
        // prologue drains. Existing AIC/AIV0 credits protect L1 reuse without a GM gate.
        ProcessSharedExpertGmm2();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2()
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    GmmRuntimeState runtimeState{startBlockIdx, vecSetSyncCom, gmm2PingPongIdx_};
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<ActivationQuantOutType, Weight1Type, QuantScaleOutType, GMM1_TILE_M>(
            commonConfig_, gmmExecutionConfig_, params_.workspaceInfo, sharedWeightTensorListAddrs_, gmmAddrInfo,
            sharedExpertIdx);
        if constexpr (ENABLE_A8W4) {
            timer_.SetWave(AscendTimer::NO_WAVE);
            timer_.SetSlice(sharedExpertIdx, 0U, commonConfig_.tokenNum);
            gmmAddrInfo.timer = timer_.Enabled() ? &timer_ : nullptr;
        }
        RunGmm2ByMode<COMBINE_NO_QUANT, QuantOutType, ActivationQuantOutType, Weight1Type, QuantScaleOutType,
                      ENABLE_A8W4, ENABLE_A4W4, GMM1_TILE_M, false, true, false, false>(
            gmmExecutionConfig_, gmmAddrInfo, problemShape, runtimeState);
        if constexpr (ENABLE_A8W4) {
            timer_.ClearSlice();
        }
    }
}

// W4 基类统一执行原 A8W4/A4W4 的专家 GMM2/Combine；派生模板只提供当前专家 slice。
// A8W4 非量化路径在当前 slice 的完整行就绪后发送 token；其他模式保持原分工。
template <TemplateMegaMoeTypeClass>
template <bool SourceGmm1Fused>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState& state, GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, WaveCombineBufferConfig& combineBufferConfig, uint32_t& combineRowSequence,
    int32_t& pairwiseTileSequence, bool isFinalCombine, bool reverseBlocks)
{
    constexpr bool tokenCombine = ENABLE_A8W4 && CombineQuantMode == COMBINE_NO_QUANT;
    constexpr bool pairwiseCombine = !tokenCombine && CombineQuantMode == COMBINE_NO_QUANT;
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm2NTileCount;
    if (!HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx, reverseBlocks)) {
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo,
            state, tokenStartIndexInExpert);
        if constexpr (pairwiseCombine) {
            // 非量化 Combine 按 GMM2 tile 配对消费，需要为每个 AIC 配置独立完成标记。
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        }
        gmmAddrInfo.timer = timer_.Enabled() ? &timer_ : nullptr;
        ProblemShape sliceProblemShape = state.problemShape;
        Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
        int32_t* gmm2TileSequence = nullptr;
        if constexpr (pairwiseCombine) {
            gmm2TileSequence = &pairwiseTileSequence;
        }
        timer_.SetSlice(state.expertIdx, tokenStartIndexInExpert, sliceTokenCount);
        // 非量化路径保留逐 tile FIX 完成边界；其他路径保留原有 slice 完成边界。
        if constexpr (g_coreType == AIC && CombineQuantMode != COMBINE_NO_QUANT) {
            timer_.StartSpan(AscendTimer::AIC_GMM2);
        }
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                timer_.StartSpan(AscendTimer::AIV_GMM2_PROLOGUE);
            }
        }
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    TopkWeightsPrefetch, false, false, true, pairwiseCombine, tokenCombine, SourceGmm1Fused>(
            sliceProblemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, expertTokenCount,
            tokenStartIndexInExpert, &params_, gmm2TileSequence, reverseBlocks);
        if constexpr (g_coreType == AIC && CombineQuantMode != COMBINE_NO_QUANT) {
            timer_.EndSpan(AscendTimer::AIC_GMM2);
        }
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                timer_.EndSpan(AscendTimer::AIV_GMM2_PROLOGUE);
            }
        }
        timer_.ClearSlice();
        if constexpr (tokenCombine && g_coreType == AIC) {
            // 只在“有 GEMM2 tile”的分支通知：一个实际生产核完成本 slice 的全部
            // tile 后加一。空核不通知；同一核有多个 tile 也只通知一次。
            // helper 等待 FIX 写出后再发布，避免 AIV1 读取尚未完成的 GEMM2 输出。
            NotifyTokenCombineSliceReady(params_, state.expertIdx, tokenStartIndexInExpert);
        }
    }

    if constexpr (tokenCombine) {
        // 发送行归属与 GEMM2 tile 分核独立：无配对计算 tile 的 AIV1 仍可能有行要发。
        // AIC 通知已在上方完成，这里的通用发送入口对 AIC 直接返回。
        RunTokenCombineSlice(commonConfig_, waveCombineJob_, params_, state, tokenStartIndexInExpert, sliceTokenCount,
                             timer_, waveCombineScratch_);
        return;
    }

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        // AIV1 已在上述 GMM2 调用中消费当前 slice；无需在外部重建 scheduler。
        return;
    }

    // 量化 Combine 保持原 W4 语义：跨 WAVE 的专家必须等最后一个 slice 完成后再消费。
    if (tokenStartIndexInExpert + sliceTokenCount < expertTokenCount) {
        return;
    }

    NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, state);
    if (isFinalCombine && !(ENABLE_A8W4 && sharedExpertNum_ > 0U)) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                // W4 prologue 完成后，末轮 AIV0 才能通过 MTE2 复用同一 UB 区域。
                SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                timer_.StartSpan(AscendTimer::AIV_COMBINE);
            }
        }
        RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, params_, gmmAddrInfo, state, state.expertIdx,
                                                    combineRowSequence);
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                timer_.EndSpan(AscendTimer::AIV_COMBINE);
            }
        }
        return;
    }
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer_.StartSpan(AscendTimer::AIV_COMBINE);
        }
    }
    RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_,
                                          params_, gmmAddrInfo, state, state.expertIdx, combineRowSequence);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer_.EndSpan(AscendTimer::AIV_COMBINE);
        }
    }
}

// 从 workspace 读取 token 数并准备 GMM 专家状态；GMM1 可按需先等待 token count ready。
template <TemplateMegaMoeTypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareGmmExpertState(ExpertLoopState& state,
                                                                               uint32_t expertIdx)
{
    if constexpr (WaitForTokenCountReady) {
        if (GetSubBlockIdx() == 0U && !state.expertCountTableReady) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
            state.expertCountTableReady = true;
        }
    }
    uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessInputPreparationStage()
{
    if constexpr (g_coreType == AIV) {
        timer_.Start(AscendTimer::AIV_INPUT_BUFFER_INIT);
    }
    SendAndQuantBuffInit();
    if constexpr (g_coreType == AIV) {
        timer_.End(AscendTimer::AIV_INPUT_BUFFER_INIT);
        timer_.Start(AscendTimer::AIV_QUANT);
    }
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        aivJob_, commonConfig_, params_, quantProcessConfig_, quantScratch_);
    if constexpr (g_coreType == AIV) {
        if (decodeInputReady_) {
            // Route/count owners differ from quant owners. Count readiness must
            // certify every local token/scale before any remote reader proceeds.
            PipeBarrier<PIPE_ALL>();
            SyncAll<true>();
        }
    }
    if constexpr (g_coreType == AIV) {
        timer_.End(AscendTimer::AIV_QUANT);
    }
    if (!sharedPolicy_.routeAiv1Only) {
        if constexpr (g_coreType == AIV) {
            timer_.Start(AscendTimer::AIV_ROUTE_MASK);
        }
        GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
                                         sendMaskScratch_);
        if constexpr (g_coreType == AIV) {
            timer_.End(AscendTimer::AIV_ROUTE_MASK);
        }
    }
    if constexpr (g_coreType == AIV) {
        timer_.Start(AscendTimer::AIV_RESET);
    }
    ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
    if constexpr (g_coreType == AIV) {
        timer_.End(AscendTimer::AIV_RESET);
    }
    if (sharedExpertNum_ > 0) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType, A_ELEMS_PER_BYTE>(
            aivJob_, commonConfig_, params_, quantProcessConfig_, sharedExpertPrepareScratch_);
    }
    if (decodeInputReady_) {
        // Route published old+1. The public barrier below makes slot 0's
        // matching epoch visible to every local count consumer.
        AdvanceInputEpochLocally(params_.peermemInfo.rankSyncInWorldPtr, aivJob_);
    }
}

/*
 * 所有 MTE Wave 路径共用相同的阶段边界，派生类只负责 MoE 专家 Wave 的具体编排。
 */
template <TemplateMegaMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessWave(Derived& derived)
{
    timer_.Start(AscendTimer::KERNEL_TIMING);
    // 保存入口时的溢出模式，计算期间关闭溢出检查。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    // Decode: dual-AIV Quant/Route/reset; prefill shared: Route follows the public barrier.
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::INPUT_PREPARE);
    ProcessInputPreparationStage();
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    } else {
        timer_.Start(AscendTimer::AIC_WAIT_INPUT_PREP);
    }
    SyncAll<false>();  // AIC 等待 AIV 完成输入准备后再进入 GMM。
    if constexpr (g_coreType == AIC) {
        timer_.End(AscendTimer::AIC_WAIT_INPUT_PREP);
    }

    int32_t gmm1TileReadySequence = 0;
    if (sharedExpertNum_ > 0U) {
        // Shared AIC/AIV0 start GMM1; AIV1 enters count/Dispatch (decode) or Route/handshake (prefill).
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM1);
        ProcessSharedExpertGmm1(gmm1TileReadySequence);
    }

    // Decode uses per-slot epoch/count readiness; prefill retains its rank handshake.
    if (!decodeInputReady_) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_INPUT);
    }
    if (sharedPolicy_.routeAiv1Only) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                const AivJobContext routeJob{blockIdx_, blockNum_};
                timer_.Start(AscendTimer::AIV_ROUTE_MASK);
                GatherAndSendExpertCompactRoutes(routeJob, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
                                                 sendMaskScratch_);
                PipeBarrier<PIPE_ALL>();
                timer_.End(AscendTimer::AIV_ROUTE_MASK);
                timer_.Start(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);
                CrossRankSyncInWorldSize<true>(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, routeJob,
                                               aivCoreIdx_ - GetSubBlockIdx(),
                                               params_.workspaceInfo.flagTopkValidIndexSyncPtr);
                timer_.End(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);
            }
        }
    } else if (!decodeInputReady_) {
        if constexpr (g_coreType == AIV) {
            timer_.Start(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);
        }
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        if constexpr (g_coreType == AIV) {
            timer_.End(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);
        }
    }
    // 阶段 3：由派生类编排 MoE 专家的 Dispatch、GMM1/Activation 和 GMM2/Combine。
    if constexpr (ENABLE_A8W4) {
        derived.ProcessMoeExpertStages(gmm1TileReadySequence);
    } else {
        derived.ProcessMoeExpertStages();
    }
    if (ENABLE_A8W4 && sharedExpertNum_ > 0U) {
        // AIC/AIV0 enter the shared tail without waiting for AIV1's final Combine.
        ProcessSharedExpertTail();
    }
    if constexpr (g_coreType == AIV) {
        timer_.Start(AscendTimer::AIV_EXPORT_SYNC);
        ExportCompactExpertTokenCounts(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        timer_.End(AscendTimer::AIV_EXPORT_SYNC);
    }
    if (sharedExpertNum_ > 0U && !ENABLE_A8W4) {
        // 阶段 4：可选的共享专家 GMM2，其结果由输出聚合阶段合并。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
        ProcessSharedExpertGmm2();
    }

    // 阶段 5：所有 rank 完成 Combine 结果发送后，AIV 执行输出聚合。
    if constexpr (g_coreType == AIV) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_OUTPUT);
        timer_.Start(AscendTimer::AIV_FINAL_CROSS_RANK_WAIT);
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        timer_.End(AscendTimer::AIV_FINAL_CROSS_RANK_WAIT);
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::UNPERMUTE);
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        timer_.Start(AscendTimer::AIV_UNPERMUTE);
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
        timer_.End(AscendTimer::AIV_UNPERMUTE);
    }

    // 恢复入口时的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
    timer_.End(AscendTimer::KERNEL_TIMING);
    timer_.FlushDynamicCounts();
}

}  // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
