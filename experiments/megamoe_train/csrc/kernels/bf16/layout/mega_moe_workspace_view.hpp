// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_workspace_view.hpp
 * \brief Device views of forward, backward and SHMEM workspace layouts.
 */

#ifndef MEGA_MOE_WORKSPACE_VIEW_HPP
#define MEGA_MOE_WORKSPACE_VIEW_HPP

#include "kernel_operator.h"
#include "mega_moe_tiling.hpp"
#include "layout/mega_moe_layout.hpp"

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t FLAG_ID_MAX = 16;
// GMM1(dispatch/swiglu)阶段结束、GMM2开始前，AIV1->AIC的一次性同步：
// 用一个独立的flag槽位(避开combine使用的pingpong 0/1)，值2 => AIV1本地槽8，
// cube侧WaitForVector(2+FLAG_ID_MAX)路由到AIV1。
constexpr uint16_t GMM2_UB_READY_FLAG = 2;
constexpr uint32_t SWIGLU_N_HALF = MegaMoeBF16Training::kSwigluNHalf;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = static_cast<int64_t>(MegaMoeBF16Training::kAlignment32);
constexpr int64_t ALIGN_128 = static_cast<int64_t>(MegaMoeBF16Training::kAlignment128);
constexpr int64_t ALIGN_256 = static_cast<int64_t>(MegaMoeBF16Training::kAlignment256);
constexpr int64_t ALIGN_512 = static_cast<int64_t>(MegaMoeBF16Training::kAlignment512);
constexpr int32_t INT_CACHELINE = static_cast<int32_t>(MegaMoeBF16Training::kIntCacheline);
constexpr int32_t MAX_AICORE_NUM = 36;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_BF_16 = static_cast<int64_t>(MegaMoeBF16Training::kBf16Bytes);
constexpr int64_t SIZE_INT_32 = static_cast<int64_t>(MegaMoeBF16Training::kInt32Bytes);
}  // namespace

struct ForwardWorkspaceInfo {
    GM_ADDR gmCtx;
    GM_ADDR ptrA;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrFlagDispatchToGmm1;
    GM_ADDR ptrDispatchedTopkWeights;
    GM_ADDR ptrSavedGate{nullptr};
    GM_ADDR ptrSavedUp{nullptr};
    GM_ADDR ptrSavedGateScale{nullptr};
    GM_ADDR ptrSavedUpScale{nullptr};
    GM_ADDR ptrSavedGateHigh{nullptr};
    GM_ADDR ptrSavedUpLow{nullptr};
    GM_ADDR ptrSavedUpHigh{nullptr};
    GM_ADDR ptrSavedTokenPerExpert;
    int64_t workspaceSize;
    int64_t savedStateSize;
    int64_t transientWorkspaceSize;
    __aicore__ inline ForwardWorkspaceInfo() = default;
    __aicore__ inline ForwardWorkspaceInfo(GM_ADDR base, const MegaMoeTilingData* tilingData)
    {
        workspaceSize = 0;
        ptrA = base;
        workspaceSize += Ops::Base::CeilAlign(SIZE_BF_16 * tilingData->maxOutputSize * tilingData->k, ALIGN_512);
        ptrAScale = base + workspaceSize;
        ptrA2 = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_BF_16 * tilingData->maxOutputSize * tilingData->n / SWIGLU_N_HALF, ALIGN_512);
        ptrA2Scale = base + workspaceSize;
        ptrcumsumMM = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize * tilingData->expertPerRank, ALIGN_512);
        expandedRowIdx = base + workspaceSize;
        workspaceSize +=
            SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->m), ALIGN_256) * tilingData->topK;
        expandedRowIdxGather = base + workspaceSize;
        workspaceSize +=
            SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->m), ALIGN_256) * tilingData->topK;
        ptrSumBeforeRank = base + workspaceSize;
        workspaceSize +=
            SIZE_INT_32 *
            Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->expertPerRank, ALIGN_128);
        ptrFlagSwiGluToGmm2 = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE, ALIGN_128);
        ptrFlagDispatchToGmm1 = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE, ALIGN_128);
        ptrDispatchedTopkWeights = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            static_cast<int64_t>(MegaMoeBF16Training::TopkWeightBytes(tilingData->topkWeightType)) *
                tilingData->maxOutputSize,
            ALIGN_512);
        ptrSavedTokenPerExpert = nullptr;
        savedStateSize = 0;
        transientWorkspaceSize = workspaceSize;
    }

    // Persistent view shared by forward and backward. No transient pointer
    // arithmetic is performed when backward constructs this view.
    __aicore__ inline void InitSavedState(GM_ADDR savedStateBase, const MegaMoeTilingData* tilingData,
                                          bool embeddedDispatchOutput = true)
    {
        const MegaMoeBF16Training::TrainingSavedStateLayout savedLayout(
            tilingData->m, tilingData->k, tilingData->topK, tilingData->epWorldSize, tilingData->expertPerRank,
            tilingData->maxOutputSize, tilingData->n,
            MegaMoeBF16Training::SaveGateUpAsMxFp8(tilingData->swigluLimit, tilingData->savedGateUpMode),
            tilingData->savedGateUpMode != MegaMoeBF16Training::kSaveNone,
            MegaMoeBF16Training::SaveGateUpClipMasks(tilingData->swigluLimit, tilingData->savedGateUpMode),
            embeddedDispatchOutput, MegaMoeBF16Training::TopkWeightBytes(tilingData->topkWeightType));
        ptrA = embeddedDispatchOutput ? savedStateBase + savedLayout.ptrAOffset : nullptr;
        ptrcumsumMM = savedStateBase + savedLayout.cumsumOffset;
        expandedRowIdx = savedStateBase + savedLayout.expandedRowIdxOffset;
        expandedRowIdxGather = savedStateBase + savedLayout.expandedRowIdxGatherOffset;
        ptrSumBeforeRank = savedStateBase + savedLayout.sumBeforeRankOffset;
        ptrSavedTokenPerExpert = savedStateBase + savedLayout.savedTokenCountsOffset;
        // Packed TopK weights have exactly the same row order and lifetime as
        // packed X. Compact ordinary state keeps both in dispatch_output.
        ptrDispatchedTopkWeights =
            embeddedDispatchOutput ? savedStateBase + savedLayout.dispatchedTopkWeightsOffset : nullptr;
        ptrSavedGate = savedLayout.gateUpSaved ? savedStateBase + savedLayout.gateOffset : nullptr;
        ptrSavedUp = savedLayout.gateUpSaved ? savedStateBase + savedLayout.upOffset : nullptr;
        ptrSavedGateScale = savedLayout.gateUpMxFp8 ? savedStateBase + savedLayout.gateScaleOffset : nullptr;
        ptrSavedUpScale = savedLayout.gateUpMxFp8 ? savedStateBase + savedLayout.upScaleOffset : nullptr;
        ptrSavedGateHigh = savedLayout.clipMaskKind ? savedStateBase + savedLayout.gateHighOffset : nullptr;
        ptrSavedUpLow = savedLayout.clipMaskKind ? savedStateBase + savedLayout.upLowOffset : nullptr;
        ptrSavedUpHigh = savedLayout.clipMaskKind ? savedStateBase + savedLayout.upHighOffset : nullptr;

        ptrA2 = nullptr;
        ptrAScale = nullptr;
        ptrA2Scale = ptrcumsumMM;
        ptrFlagSwiGluToGmm2 = nullptr;
        ptrFlagDispatchToGmm1 = nullptr;
        savedStateSize = static_cast<int64_t>(savedLayout.totalBytes);
        transientWorkspaceSize = 0;
        workspaceSize = savedStateSize;
    }

    // Forward additionally owns temporary GMM2 input and readiness flags.
    __aicore__ inline ForwardWorkspaceInfo(GM_ADDR savedStateBase, GM_ADDR transientWorkspaceBase,
                                           const MegaMoeTilingData* tilingData)
    {
        InitSavedState(savedStateBase, tilingData);
        const MegaMoeBF16Training::ForwardTransientWorkspaceLayout transientLayout(
            tilingData->n, tilingData->expertPerRank, tilingData->maxOutputSize);
        ptrA2 = transientWorkspaceBase + transientLayout.ptrA2Offset;
        ptrFlagSwiGluToGmm2 = transientWorkspaceBase + transientLayout.swigluToGmm2FlagOffset;
        ptrFlagDispatchToGmm1 = transientWorkspaceBase + transientLayout.dispatchToGmm1FlagOffset;
        ptrAScale = ptrA2;
        transientWorkspaceSize = static_cast<int64_t>(transientLayout.totalBytes);
        workspaceSize = savedStateSize + transientWorkspaceSize;
    }

    // Ordinary BF16 keeps packed X in an independent tensor. The framework
    // may retain that tensor for the existing backward or release it when a
    // backward implementation performs its own Dispatch.
    __aicore__ inline ForwardWorkspaceInfo(GM_ADDR savedStateBase, GM_ADDR dispatchOutputBase,
                                           GM_ADDR transientWorkspaceBase, const MegaMoeTilingData* tilingData)
    {
        InitSavedState(savedStateBase, tilingData, false);
        const MegaMoeBF16Training::DispatchOutputLayout dispatchLayout(
            tilingData->k, tilingData->maxOutputSize, MegaMoeBF16Training::TopkWeightBytes(tilingData->topkWeightType));
        ptrA = dispatchOutputBase + dispatchLayout.tokensOffset;
        ptrDispatchedTopkWeights = dispatchOutputBase + dispatchLayout.topkWeightsOffset;
        const MegaMoeBF16Training::ForwardTransientWorkspaceLayout transientLayout(
            tilingData->n, tilingData->expertPerRank, tilingData->maxOutputSize);
        ptrA2 = transientWorkspaceBase + transientLayout.ptrA2Offset;
        ptrFlagSwiGluToGmm2 = transientWorkspaceBase + transientLayout.swigluToGmm2FlagOffset;
        ptrFlagDispatchToGmm1 = transientWorkspaceBase + transientLayout.dispatchToGmm1FlagOffset;
        ptrAScale = ptrA2;
        transientWorkspaceSize = static_cast<int64_t>(transientLayout.totalBytes);
        workspaceSize = savedStateSize + transientWorkspaceSize;
    }
};
// Persistent view of the state retained by forward for the matching backward.
// The saved data fields are read-only.
// Forward-only buffers deliberately remain null so backward cannot accidentally
// depend on transient storage whose lifetime ended when forward returned.
struct BackwardSavedStateInfo : ForwardWorkspaceInfo {
    __aicore__ inline BackwardSavedStateInfo() = default;
    __aicore__ inline BackwardSavedStateInfo(GM_ADDR savedStateBase, GM_ADDR dispatchOutputBase,
                                             const MegaMoeTilingData* tilingData)
    {
        InitSavedState(savedStateBase, tilingData, false);
        const MegaMoeBF16Training::DispatchOutputLayout dispatchLayout(
            tilingData->k, tilingData->maxOutputSize, MegaMoeBF16Training::TopkWeightBytes(tilingData->topkWeightType));
        ptrA = dispatchOutputBase + dispatchLayout.tokensOffset;
        ptrDispatchedTopkWeights = dispatchOutputBase + dispatchLayout.topkWeightsOffset;
        // Match the persistent-only backward view: no forward scratch aliases.
        gmCtx = nullptr;
        ptrA2Scale = nullptr;
    }
};

struct BackwardWorkspaceInfo {
    GM_ADDR ptrDO;
    GM_ADDR ptrRecomputeGate;
    GM_ADDR ptrRecomputeUp;
    GM_ADDR ptrGradZ;
    GM_ADDR ptrHp;
    GM_ADDR ptrDpProducts;
    GM_ADDR ptrDX;
    GM_ADDR ptrBwFlagDOReleased;
    GM_ADDR ptrBwFlagDZReady;
    GM_ADDR ptrBwFlagDXReady;
    GM_ADDR ptrBwFlagDHpReady;
    GM_ADDR ptrBwFlagDispatchToGmm2Grad;
    GM_ADDR ptrBwFlagAiv1Ready;
    GM_ADDR ptrBwFlagHDone;
    GM_ADDR ptrBwDispatchProgress;
    int64_t workspaceSize;

    __aicore__ inline BackwardWorkspaceInfo() = default;

    __aicore__ inline BackwardWorkspaceInfo(GM_ADDR base, const MegaMoeTilingData* tilingData, uint32_t splitExperts)
    {
        const int64_t e = tilingData->expertPerRank;
        const int64_t r = tilingData->maxOutputSize;
        const int64_t k = tilingData->k;
        const int64_t n = tilingData->n;
        const int64_t f = n / SWIGLU_N_HALF;

        int64_t offset = 0;
        ptrDO = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_BF_16 * r * k, ALIGN_512);
        const uint32_t savedMode =
            MegaMoeBF16Training::ResolveSavedGateUpMode(tilingData->savedGateUpMode, tilingData->swigluLimit);
        if (savedMode == MegaMoeBF16Training::kSaveNone) {
            ptrRecomputeGate = base + offset;
            offset += Ops::Base::CeilAlign(SIZE_BF_16 * r * f, ALIGN_512);
            ptrRecomputeUp = base + offset;
            offset += Ops::Base::CeilAlign(SIZE_BF_16 * r * f, ALIGN_512);
        } else {
            ptrRecomputeGate = nullptr;
            ptrRecomputeUp = nullptr;
        }
        ptrGradZ = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_BF_16 * r * n, ALIGN_512);
        ptrHp = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_BF_16 * r * f, ALIGN_512);
        // One FP32 partial sum per [packed row, 256-wide F tile]. Different F
        // tiles may be owned by different AIVs, so P1 writes disjoint partials
        // and P6 performs the small final reduction without cross-core atomics.
        const int64_t dpPartialTiles =
            Ops::Base::CeilDiv(static_cast<uint64_t>(f), static_cast<uint64_t>(BACKWARD_DP_PARTIAL_TILE_N));
        ptrDpProducts = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32 * r * dpPartialTiles, ALIGN_512);
        // Saved-mode dX aliases dO after all expert dW2 readers retire.
        // none writes dX through UB and never accesses ptrDX.
        ptrDX = MegaMoeBF16Training::UseGmForBackwardDataGrad(savedMode) ? ptrDO : nullptr;
        ptrBwFlagDispatchToGmm2Grad = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32 * e * INT_CACHELINE, ALIGN_128);
        ptrBwFlagAiv1Ready = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32, ALIGN_32);
        ptrBwFlagHDone = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32, ALIGN_32);
        ptrBwFlagDHpReady = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32 * e * INT_CACHELINE, ALIGN_128);
        ptrBwFlagDZReady = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32 * e * INT_CACHELINE, ALIGN_128);
        ptrBwFlagDXReady = base + offset;
        offset += Ops::Base::CeilAlign(SIZE_INT_32 * e * INT_CACHELINE, ALIGN_128);
        if (MegaMoeBF16Training::UseGmForBackwardDataGrad(savedMode)) {
            ptrBwFlagDOReleased = base + offset;
            offset += Ops::Base::CeilAlign(SIZE_INT_32 * e * INT_CACHELINE, ALIGN_128);
        } else {
            ptrBwFlagDOReleased = nullptr;
        }
        ptrBwDispatchProgress = MegaMoeBF16Training::UseGmForBackwardDataGrad(savedMode) && splitExperts != 0
                                    ? base + MegaMoeBF16Training::AlignUp(offset, 64)
                                    : nullptr;
        workspaceSize = MegaMoeBF16Training::BackwardDispatchProgressLayout::TotalBytes(
            offset, splitExperts, tilingData->epWorldSize, tilingData->blockNumPerEP, savedMode);
    }
};

struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrCompactTokenCount;
    GM_ADDR ptrCompactTokenCountBlockSums;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrA0;  // initrouting结果，包括data+scale
    GM_ADDR ptrRoutedTopkWeights;
    GM_ADDR ptrD;
    uint64_t tokenCountsOffset;
    uint64_t payloadOffset;
    bool compactTokenCountExchange;
    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData* tilingData,
                                  bool allowCompactTokenCountExchange = false)
    {
        ptrBase = base;
        const bool compactEnabled = allowCompactTokenCountExchange && tilingData->compactTokenCountExchange != 0;
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
            tilingData->epWorldSize, tilingData->expertPerRank, compactEnabled);
        compactTokenCountExchange = countLayout.enabled;
        ptrCompactTokenCount = countLayout.enabled ? base + countLayout.packetOffset : nullptr;
        ptrCompactTokenCountBlockSums = countLayout.enabled ? base + countLayout.blockSumsOffset : nullptr;
        tokenCountsOffset = countLayout.tokenCountsOffset;
        payloadOffset = countLayout.payloadOffset;
        int64_t offset = static_cast<int64_t>(tokenCountsOffset);
        ptrTokenPerExpert = base + offset;
        offset = static_cast<int64_t>(payloadOffset);
        ptrA0 = base + offset;
        offset += static_cast<uint64_t>(tilingData->m) * tilingData->topK * tilingData->k * sizeof(bfloat16_t);
        ptrRoutedTopkWeights = base + offset;
        offset += Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->m) * tilingData->topK *
                static_cast<int64_t>(MegaMoeBF16Training::TopkWeightBytes(tilingData->topkWeightType)),
            static_cast<int64_t>(ALIGN_512));
        ptrD = base + offset;
    }
};

#endif  // MEGA_MOE_WORKSPACE_VIEW_HPP
