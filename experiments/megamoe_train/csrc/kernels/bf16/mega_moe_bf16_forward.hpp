// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*!
 * \file mega_moe_bf16_forward.hpp
 * \brief
 */

#ifndef MEGA_MOE_BF16_FORWARD_HPP
#define MEGA_MOE_BF16_FORWARD_HPP

#include "kernel_operator.h"
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "layout/mega_moe_context.hpp"
#include "routing/mega_moe_combine_send.hpp"
#include "tla/layout.hpp"
#include "epilogue/block_epilogue_swiglu_forward_bf16.hpp"
#include "catlass/mega_moe_bf16_forward_catlass_impl.hpp"
#include "common/mega_moe_forward_save.hpp"
#include "routing/mega_moe_topk_weight_gather.hpp"
#include "routing/mega_moe_unpermute.hpp"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t>;

constexpr uint32_t TOPK_WEIGHT_COPY_BUFFER_BYTES = MegaMoeBF16Training::ForwardDispatchUbLayout::kTopkWeightBytes;

template <typename TopkWeightT = bfloat16_t>
class MegaMoeBF16Forward {
public:
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    using TopkWeightBitsT = std::conditional_t<sizeof(TopkWeightT) == sizeof(uint16_t), uint16_t, uint32_t>;
    static constexpr uint32_t TOPK_WEIGHT_COPY_ELEMENTS = TOPK_WEIGHT_COPY_BUFFER_BYTES / sizeof(TopkWeightT);
    using QuantOutType = bfloat16_t;
    using QuantScaleOutType = float;
    __aicore__ inline MegaMoeBF16Forward(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR permutedInputOut,
                                GM_ADDR actualRecvTokensOut, GM_ADDR savedMetaOut, GM_ADDR workspaceGM,
                                MegaMoeTilingData* tilingData, GM_ADDR transientWorkspaceGM = nullptr,
                                GM_ADDR dispatchOutputGM = nullptr, bool allowCompactTokenCountExchange = true);
    __aicore__ inline void Process();
    template <uint32_t SaveMode, bool TransW>
    __aicore__ inline void ProcessSelected();
    template <uint32_t SaveMode, bool TransW, bool GmSwiglu = false>
    __aicore__ inline void ProcessWithSaveMode();
    __aicore__ inline void EnableDispatchTileReady(GM_ADDR transientWorkspaceGM);

private:
    __aicore__ inline void FirstBuffInit(TPipe* tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void CrossRankSyncCntPerExpertCompact();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void SaveForwardState();
    __aicore__ inline void GatherRoutedTopkWeights();
    template <bool TileReady = false, bool WaitForFullExpert = false>
    __aicore__ inline void DispatchTokens(GMMAddrInfo& gmmAddrInfo, uint32_t expertIdx,
                                          GlobalTensor<QuantOutType> aGlobalTensor,
                                          bool useTileReady = TileReady && !WaitForFullExpert);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo& gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void UnpermuteV2();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                                                GlobalTensor<QuantScaleOutType> dstScale,
                                                GlobalTensor<QuantOutType> src, int32_t rows, int32_t hiddenSize,
                                                int32_t& pingpongId);
    __aicore__ inline void CopyGMToGMTopkWeights(GlobalTensor<TopkWeightT> dst, GlobalTensor<TopkWeightT> src,
                                                 int32_t rows);

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t& coreLen, int32_t& coreOffset,
                                        int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2;  // 取到vec核总数
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    __aicore__ inline void GmSignalWait(__gm__ int32_t* sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t* sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    TBuf<> CrossRankSyncBuf_;
    TBuf<> PrevSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> TopkWeightCopyBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> SaveStateBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t gmm1VecSetSyncCom_ = 0;
    int32_t gmm2VecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    int32_t targetFlag_;

    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint32_t dispatchCopyElementsPerPing_ = 0;
    bool useSplitTrainingState_ = false;
    GM_ADDR dispatchTileReadyBase_ = nullptr;
    uint64_t dispatchTileSlotsPerExpert_ = 0;
    bool preparedPhaseEnabled_ = false;

    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using BlockEpilogue =
        MegaMoeBF16Impl::BlockEpilogueSwigluForwardBF16<MegaMoeBF16ForwardCatlassImpl::Gmm1L1TileShape, true,
                                                        TopkWeightT>;
    BlockEpilogue epilogueOp_;  // GMMswiglu调用
};

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR permutedInputOut, GM_ADDR actualRecvTokensOut, GM_ADDR savedMetaOut,
    GM_ADDR workspaceGM, MegaMoeTilingData* tilingData, GM_ADDR transientWorkspaceGM, GM_ADDR dispatchOutputGM,
    bool allowCompactTokenCountExchange)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = 1;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    // DeepEP passes each local expert's weights in one contiguous tensor.
    // Unlike the original aclTensorList ABI, these are already raw GM addresses.
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr = weightScales1;
    params_.b2ScaleGmAddr = weightScales2;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.permutedInputOutGmAddr = permutedInputOut;
    params_.actualRecvTokensOutGmAddr = actualRecvTokensOut;
    params_.savedMetaOutGmAddr = savedMetaOut;
    params_.probsGmAddr = topkWeights;
    useSplitTrainingState_ = transientWorkspaceGM != nullptr;
    if (useSplitTrainingState_ && dispatchOutputGM != nullptr) {
        params_.workspaceInfo = ForwardWorkspaceInfo(workspaceGM, dispatchOutputGM, transientWorkspaceGM, tilingData);
    } else if (useSplitTrainingState_) {
        params_.workspaceInfo = ForwardWorkspaceInfo(workspaceGM, transientWorkspaceGM, tilingData);
    } else {
        params_.workspaceInfo = ForwardWorkspaceInfo(workspaceGM, tilingData);
    }
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, allowCompactTokenCountExchange);
    params_.tilingData = tilingData;
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
                      params_.workspaceInfo.ptrFlagSwiGluToGmm2, params_.workspaceInfo.ptrDispatchedTopkWeights,
                      nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256, tilingData->swigluLimit,
                      params_.workspaceInfo.ptrSavedGate, params_.workspaceInfo.ptrSavedUp,
                      params_.workspaceInfo.ptrSavedGateScale, params_.workspaceInfo.ptrSavedUpScale,
                      params_.workspaceInfo.ptrSavedGateHigh, params_.workspaceInfo.ptrSavedUpLow,
                      params_.workspaceInfo.ptrSavedUpHigh});
    perRankStridesInTokenPerExpert_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ =
        params_.peermemInfo.ptrTokenPerExpert + TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::EnableDispatchTileReady(GM_ADDR transientWorkspaceGM)
{
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout layout(params_.tilingData->n, expertPerRank_,
                                                                     maxOutputSize_);
    dispatchTileReadyBase_ = transientWorkspaceGM + layout.offset;
    dispatchTileSlotsPerExpert_ = layout.flagSlotsPerExpert;
    params_.workspaceInfo.transientWorkspaceSize = layout.totalBytes;
    params_.workspaceInfo.workspaceSize += layout.bytes;
    if constexpr (DEEPEP_BF16_FORCE_EXPERT_READY != 0) {
        preparedPhaseEnabled_ = false;
        return;
    }
    MegaMoeBF16ForwardSchedule::View view;
    view.Init(reinterpret_cast<__gm__ int32_t*>(dispatchTileReadyBase_),
              reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM),
              reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrFlagDispatchToGmm1), maxOutputSize_,
              expertPerRank_, worldSize_, blockNum_, dispatchBlockNumPerEP_, k_);
    preparedPhaseEnabled_ = view.enabled && MegaMoeBF16Training::ForwardDispatchTileReadyLayout::UseTileReadyForShape(
                                                m_, topK_, expertPerRank_);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::FirstBuffInit(TPipe* tPipe)
{
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    const MegaMoeBF16Training::ForwardDispatchUbLayout layout(worldSize_, expertPerRank_, k_, dispatchRows_);
    tPipe->InitBuffer(CrossRankSyncBuf_, layout.syncBytes);
    tPipe->InitBuffer(PrevSumBuf_, layout.prevSumBytes);
    tPipe->InitBuffer(CumsumBuf_, layout.cumsumBytes);
    dispatchCopyElementsPerPing_ = layout.copyElementsPerPing;
    tPipe->InitBuffer(CopyTempBuf_, layout.copyBytes);
    tPipe->InitBuffer(TopkWeightCopyBuf_, layout.kTopkWeightBytes);
    tPipe->InitBuffer(ResetFlagBuf_, layout.resetFlagBytes);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ResetWorkSpaceFlagList()
{
    // Host leaves forward-transient storage uninitialized. Reset every
    // control flag here on each invocation before the pre-Dispatch SyncAll.
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    if constexpr (g_coreType == AIV) {
        if (aivCoreIdx_ == 0) {
            int32_t flagNum = expertPerRank_ * INT_CACHELINE * 2;
            // LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
            LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
            Duplicate(initTensor, 0, flagNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm_, initTensor, flagNum);
        }
        if (dispatchTileReadyBase_ != nullptr) {
            // The legacy flag regions have independent alignment padding.
            // Reset expert completion at its actual address (also for E=1).
            __gm__ int32_t* expertFlags =
                reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrFlagDispatchToGmm1);
            for (uint32_t expert = aivCoreIdx_; expert < expertPerRank_; expert += blockAivNum_) {
                WriteGmByPassDCache(expertFlags + expert * INT_CACHELINE, 0);
            }
            // Static metadata and all per-expert row words are prepared by
            // AIV0 during the existing cumsum scan, before pre-Dispatch SyncAll.
            // Other modes keep the original one-word startup reset here.
            if (!preparedPhaseEnabled_) {
                __gm__ int32_t* flags = reinterpret_cast<__gm__ int32_t*>(dispatchTileReadyBase_);
                for (uint64_t slot = aivCoreIdx_; slot < dispatchTileSlotsPerExpert_; slot += blockAivNum_)
                    WriteGmByPassDCache(flags + slot * INT_CACHELINE, 0);
            }
            // Process()'s existing pre-Dispatch SyncAll also orders these
            // per-launch resets before any AIV1 publication or AIC polling.
        }
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer1 = CumsumBuf_.Get<int32_t>();
    DataCopyPad(
        tmpBuffer1, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, ALIGN_128) - expertPerRank_) * sizeof(int32_t)),
         0},
        {});
    if (preparedPhaseEnabled_) {
        using PhaseView = MegaMoeBF16ForwardSchedule::View;
        PhaseView view;
        view.Init(reinterpret_cast<__gm__ int32_t*>(dispatchTileReadyBase_),
                  reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM),
                  reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrFlagDispatchToGmm1), maxOutputSize_,
                  expertPerRank_, worldSize_, blockNum_, dispatchBlockNumPerEP_, k_);
        LocalTensor<int32_t> nodes = CopyTempBuf_.Get<int32_t>();
        Duplicate(nodes, 0, view.words);
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        // Same single source scan as the original prefix sum. No second
        // peer pass, runtime sorting, payload-side atomics or group arrival.
        for (uint32_t source = 0; source < worldSize_; ++source) {
            for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
                const uint32_t offset = source * expertPerRankAligned + expert;
                const uint32_t begin = source ? tmpBuffer1.GetValue(offset - expertPerRankAligned) : 0U;
                const uint32_t end = begin + tmpBuffer1.GetValue(offset);
                tmpBuffer1.SetValue(offset, static_cast<int32_t>(end));
                view.Accumulate(nodes, expert, source, begin, end);
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(cumsumInWorkSpace_, tmpBuffer1,
                    {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
        // Count GM output and earlier control clears must retire before we
        // reuse count UB and publish mode word1. Payload has not started.
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        uint64_t remaining = maxOutputSize_, largest = 0;
        for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
            const uint64_t raw = tmpBuffer1.GetValue((worldSize_ - 1U) * expertPerRankAligned + expert);
            const uint64_t take = raw < remaining ? raw : remaining;
            remaining -= take;
            largest = take > largest ? take : largest;
            // Packed capacity truncation changes producer ownership. Keep
            // original scheduler for that expert; data-ready remains exact.
            WriteGmByPassDCache(view.modes + expert * INT_CACHELINE + 1U, raw == take ? 1 : 0);
        }
        const uint32_t countElems = worldSize_ * expertPerRankAligned;
        Duplicate(tmpBuffer1, 0, countElems);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GlobalTensor<int32_t> controls;
        controls.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dispatchTileReadyBase_));
        const uint64_t clearWords = ((largest + PhaseView::kRows - 1U) / PhaseView::kRows) * INT_CACHELINE;
        for (uint64_t offset = 0; offset < clearWords; offset += countElems) {
            const uint64_t left = clearWords - offset;
            DataCopy(controls[offset], tmpBuffer1, static_cast<uint32_t>(left < countElems ? left : countElems));
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        // Dense UB trees -> spare GM columns. GM stride skips all E data
        // counters. UB pitch is rounded to 32B by DataCopyPad, including E=7.
        const uint32_t nodeAligned = (view.words + 7U) / 8U * 8U;
        const uint32_t copyElems = 2U * dispatchCopyElementsPerPing_ * sizeof(QuantOutType) / sizeof(int32_t);
        const uint32_t tailElems = copyElems > nodeAligned ? copyElems - nodeAligned : 0U;
        const bool useTail = tailElems > countElems;
        LocalTensor<int32_t> stage = useTail ? nodes[nodeAligned] : tmpBuffer1;
        const uint32_t stageElems = useTail ? tailElems : countElems;
        const uint32_t pitch = (view.freeWords + 7U) / 8U * 8U;
        const uint32_t lines = (view.words + view.freeWords - 1U) / view.freeWords;
        const uint32_t batch = stageElems / pitch;
        for (uint32_t first = 0; first < lines; first += batch) {
            const uint32_t rows = (lines - first) < batch ? (lines - first) : batch;
            for (uint32_t row = 0; row < rows; ++row)
                for (uint32_t col = 0; col < view.freeWords; ++col) {
                    const uint32_t index = (first + row) * view.freeWords + col;
                    stage.SetValue(row * pitch + col, index < view.words ? nodes.GetValue(index) : 0);
                }
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(
                controls[first * INT_CACHELINE + expertPerRank_], stage,
                DataCopyExtParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(view.freeWords * sizeof(int32_t)),
                                  0, static_cast<uint32_t>(expertPerRank_ * sizeof(int32_t)), 0});
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
        return;  // Existing caller pre-Dispatch SyncAll makes everything visible.
    }
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
            tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(cumsumInWorkSpace_, tmpBuffer1,
                {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(expertTokenNumsOut_, tmpCopyBuffer,
                DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::SaveForwardState()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint64_t perRankStride = Ops::Base::CeilAlign(static_cast<uint64_t>(worldSize_) * expertPerRank_,
                                                        static_cast<uint64_t>(MegaMoeMeta::kTokenCountStrideAlignment));
    const uint64_t countsBytes = worldSize_ * perRankStride * sizeof(int32_t);
    auto copyBuffer = SaveStateBuf_.Get<uint8_t>();

    if (useSplitTrainingState_) {
        // Packed X/TopK weights are already in their final storage (embedded
        // for raw BF16, independent dispatch_output for training). Snapshot
        // only the SHMEM token-count table because SHMEM is mutable across
        // subsequent forward calls.
        MegaMoeForwardSave::CopyContiguous(params_.workspaceInfo.ptrSavedTokenPerExpert,
                                           params_.peermemInfo.ptrTokenPerExpert, countsBytes, blockIdx_, blockNum_,
                                           copyBuffer);
        return;
    }

    uint64_t actualRows = 0;
    if (groupListType_ == 0) {
        actualRows = static_cast<uint64_t>(groupList_.GetValue(expertPerRank_ - 1));
    } else {
        for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
            actualRows += static_cast<uint64_t>(groupList_.GetValue(expertIdx));
        }
    }
    if (actualRows > maxOutputSize_) {
        actualRows = maxOutputSize_;
    }
    if (blockIdx_ == 0) {
        AscendC::WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(params_.actualRecvTokensOutGmAddr),
                                     static_cast<int32_t>(actualRows));
    }

    MegaMoeForwardSave::CopyContiguous(params_.permutedInputOutGmAddr, params_.workspaceInfo.ptrA,
                                       actualRows * k_ * sizeof(bfloat16_t), blockIdx_, blockNum_, copyBuffer);

    const uint64_t cumsumOffset = countsBytes;
    const uint64_t cumsumBytes =
        Ops::Base::CeilAlign(static_cast<uint64_t>(worldSize_) * expertPerRank_ * sizeof(int32_t),
                             static_cast<uint64_t>(MegaMoeMeta::kMetaAlignment));
    const uint64_t rowIdxGatherOffset = cumsumOffset + cumsumBytes;
    const uint64_t rowIdxGatherBytes =
        Ops::Base::CeilAlign(static_cast<uint64_t>(m_), static_cast<uint64_t>(MegaMoeMeta::kRoutingRowAlignment)) *
        topK_ * sizeof(int32_t);
    const uint64_t sumBeforeRankOffset = rowIdxGatherOffset + rowIdxGatherBytes;
    MegaMoeForwardSave::CopyContiguous(params_.savedMetaOutGmAddr, params_.peermemInfo.ptrTokenPerExpert, countsBytes,
                                       blockIdx_, blockNum_, copyBuffer);
    MegaMoeForwardSave::CopyContiguous(params_.savedMetaOutGmAddr + cumsumOffset, params_.workspaceInfo.ptrcumsumMM,
                                       cumsumBytes, blockIdx_, blockNum_, copyBuffer);
    MegaMoeForwardSave::CopyContiguous(params_.savedMetaOutGmAddr + rowIdxGatherOffset,
                                       params_.workspaceInfo.expandedRowIdxGather, rowIdxGatherBytes, blockIdx_,
                                       blockNum_, copyBuffer);
    MegaMoeForwardSave::CopyContiguous(params_.savedMetaOutGmAddr + sumBeforeRankOffset,
                                       params_.workspaceInfo.ptrSumBeforeRank, perRankStride * sizeof(int32_t),
                                       blockIdx_, blockNum_, copyBuffer);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::EndSync()
{
    if (gmm1VecSetSyncCom_ == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        WaitForVector();
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::EndGMM2Sync()
{
    if constexpr (g_coreType == AIV) {
        return;
    }
    if (gmm2VecSetSyncCom_ <= 0) {
        return;
    } else if (gmm2VecSetSyncCom_ == 1) {
        WaitForVector(FLAG_ID_MAX);
    } else {
        WaitForVector(gmm2PingPongIdx_ + FLAG_ID_MAX);
        WaitForVector(1 - gmm2PingPongIdx_ + FLAG_ID_MAX);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CrossRankSyncCntPerExpert()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint32_t compactDestinationsPerCore = Ops::Base::CeilDiv(worldSize_, blockAivNum_);
    const uint32_t compactPacketOffset =
        Ops::Base::CeilAlign(2U * compactDestinationsPerCore * expertPerRank_, static_cast<uint32_t>(8));
    const uint32_t compactWorkElements =
        compactPacketOffset +
        static_cast<uint32_t>(MegaMoeBF16Training::kCompactTokenCountPacketBytes / sizeof(int32_t));
    if (params_.peermemInfo.compactTokenCountExchange &&
        blockAivNum_ <= MegaMoeBF16Training::kCompactTokenCountMaxAivCores &&
        compactWorkElements <= static_cast<uint32_t>(perRankStridesInTokenPerExpert_)) {
        CrossRankSyncCntPerExpertCompact();
        return;
    }
    uint32_t numPerCore =
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    LocalTensor<int32_t> tmpBuffer = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> prevSumBuf = PrevSumBuf_.Get<int32_t>();
    // 给对端发送
    uint64_t offsetInDstWinAddr =
        perRankStridesInTokenPerExpert_ * sizeof(int32_t) * rankId_ + params_.peermemInfo.tokenCountsOffset;
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    // 本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx != rankId_) {
            for (int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert) +
                                            TokenPerExpertLayout(dstRankIdx, 0, checkIdx);
                GmSignalWait(syncCheck, 0);
            }
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstRankIdx * expertPerRank_], prevSumBuf,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CrossRankSyncCntPerExpertCompact()
{
    using namespace MegaMoeBF16Training;
    constexpr uint32_t packetElements = kCompactTokenCountPacketBytes / sizeof(int32_t);
    constexpr uint32_t prefixElements = kCompactTokenCountPrefixOffset / sizeof(int32_t);
    constexpr int32_t arrivalTag = 0x800000;

    const uint32_t destinationsPerCore = Ops::Base::CeilDiv(worldSize_, blockAivNum_);
    const uint32_t firstDestination = aivCoreIdx_ * destinationsPerCore;
    const uint32_t localDestinations =
        firstDestination < worldSize_
            ? ((worldSize_ - firstDestination) < destinationsPerCore ? worldSize_ - firstDestination
                                                                     : destinationsPerCore)
            : 0;
    const uint32_t localElements = localDestinations * expertPerRank_;

    LocalTensor<int32_t> work = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> localCounts = work;
    LocalTensor<int32_t> localPrefixes = work[localElements];
    const uint32_t packetOffset = Ops::Base::CeilAlign(2U * localElements, static_cast<uint32_t>(8));
    LocalTensor<int32_t> packet = work[packetOffset];

    GlobalTensor<int32_t> sourceCounts;
    sourceCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
    int32_t localSum = 0;
    if (localElements > 0) {
        DataCopyPad(localCounts, sourceCounts[firstDestination * expertPerRank_],
                    {1, static_cast<uint16_t>(localElements * sizeof(int32_t)), 0, 0}, {});
        SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t index = 0; index < localElements; ++index) {
            localPrefixes.SetValue(index, localSum);
            localSum += localCounts.GetValue(index);
        }
    }
    auto blockSums = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrCompactTokenCountBlockSums);
    WriteGmByPassDCache(blockSums + aivCoreIdx_, localSum);

    // Publish every core's contiguous destination-range sum. The following
    // scalar scan is at most 71 int32 values on A5 and yields this core's
    // exclusive prefix in the flattened destination/expert route order.
    SyncAll<true>();
    int32_t blockPrefix = 0;
    for (uint32_t previous = 0; previous < aivCoreIdx_; ++previous) {
        blockPrefix += ReadGmByPassDCache(blockSums + previous);
    }

    for (uint32_t localDestination = 0; localDestination < localDestinations; ++localDestination) {
        for (uint32_t index = 0; index < packetElements; ++index) {
            packet.SetValue(index, 0);
        }
        const uint32_t countBase = localDestination * expertPerRank_;
        for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
            const uint32_t index = countBase + expert;
            int32_t count = localCounts.GetValue(index);
            if (expert == 0) {
                count += arrivalTag;
            }
            packet.SetValue(expert, count);
            packet.SetValue(prefixElements + expert, blockPrefix + localPrefixes.GetValue(index));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        const uint32_t destination = firstDestination + localDestination;
        GlobalTensor<int32_t> remotePacket;
        remotePacket.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetRankWinAddrWithOffset(
            destination,
            MegaMoeBF16Training::kPeerDataOffset + static_cast<uint64_t>(rankId_) * kCompactTokenCountPacketBytes)));
        DataCopy(remotePacket, packet, packetElements);
        // Complete the remote write before scalar stores build the next packet.
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    GlobalTensor<int32_t> localPackets;
    localPackets.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrCompactTokenCount));
    for (uint32_t source = aivCoreIdx_; source < worldSize_; source += blockAivNum_) {
        const uint64_t packetBase = static_cast<uint64_t>(source) * packetElements;
        GmSignalWait(reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrCompactTokenCount) + packetBase, 0);
        DataCopy(packet, localPackets[packetBase], packetElements);
        SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        packet.SetValue(0, packet.GetValue(0) - arrivalTag);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(tokenPerExpertWin_[TokenPerExpertLayout(source, rankId_, 0)], packet,
                    {1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
        DataCopyPad(preSumBeforeRankWorkSpace_[source * expertPerRank_], packet[prefixElements],
                    {1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
        // Both GM copies consume packet. Drain them before the next MTE2 fill.
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::GatherRoutedTopkWeights()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    const int64_t totalRoutes = static_cast<int64_t>(m_) * topK_;
    const int64_t routesPerCore = Ops::Base::CeilDiv(totalRoutes, static_cast<int64_t>(blockAivNum_));
    const int64_t indexBase = static_cast<int64_t>(aivCoreIdx_) * routesPerCore;
    const int64_t elements = indexBase >= totalRoutes                  ? 0
                             : indexBase + routesPerCore > totalRoutes ? totalRoutes - indexBase
                                                                       : routesPerCore;
    if (elements == 0) {
        return;
    }

    // Do not publish the permutation with SIMT scalar stores.  A peer may
    // observe the following count notification before those cached stores
    // have become visible to its MTE2 reader.  Gather each core's contiguous
    // span into UB and publish it with MTE3 instead, matching the established
    // GM -> UB -> GM visibility model used by the normal Dispatch payload.
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.expandedRowIdx));
    GlobalTensor<TopkWeightBitsT> topkWeights;
    topkWeights.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightBitsT*>(params_.probsGmAddr));
    GlobalTensor<TopkWeightBitsT> routedTopkWeights;
    routedTopkWeights.SetGlobalBuffer(
        reinterpret_cast<__gm__ TopkWeightBitsT*>(params_.peermemInfo.ptrRoutedTopkWeights));
    LocalTensor<TopkWeightBitsT> buffer = TopkWeightCopyBuf_.Get<TopkWeightBitsT>();

    const int64_t processCount = Ops::Base::CeilDiv(elements, static_cast<int64_t>(TOPK_WEIGHT_COPY_ELEMENTS));
    for (int64_t processIndex = 0; processIndex < processCount; ++processIndex) {
        const int64_t chunkOffset = processIndex * TOPK_WEIGHT_COPY_ELEMENTS;
        const int64_t chunkElements =
            chunkOffset + TOPK_WEIGHT_COPY_ELEMENTS > elements ? elements - chunkOffset : TOPK_WEIGHT_COPY_ELEMENTS;
        for (int64_t index = 0; index < chunkElements; ++index) {
            const int64_t sortedRow = indexBase + chunkOffset + index;
            const int32_t originalRoute = expandedRowIdx.GetValue(sortedRow);
            buffer.SetValue(index, topkWeights.GetValue(originalRoute));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(chunkElements * sizeof(TopkWeightBitsT)), 0U, 0U, 0U};
        DataCopyPad(routedTopkWeights[indexBase + chunkOffset], buffer, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                                                                             GlobalTensor<QuantScaleOutType> dstScale,
                                                                             GlobalTensor<QuantOutType> src,
                                                                             int32_t rows, int32_t hiddenSize,
                                                                             int32_t& pingpongId)
{
    (void)dstScale;
    constexpr int32_t BufferNum = 2;
    LocalTensor<QuantOutType> tmpBuffer1 = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[dispatchCopyElementsPerPing_];
    auto processCount = Ops::Base::CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }
        const int64_t elementOffset = static_cast<int64_t>(processIndex) * dispatchRows_ * hiddenSize;
        const int64_t elementCount = static_cast<int64_t>(rowNum) * hiddenSize;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(buf, src[elementOffset], elementCount);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        DataCopy(dst[elementOffset], buf, elementCount);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CopyGMToGMTopkWeights(GlobalTensor<TopkWeightT> dst,
                                                                              GlobalTensor<TopkWeightT> src,
                                                                              int32_t rows)
{
    LocalTensor<TopkWeightT> buffer = TopkWeightCopyBuf_.Get<TopkWeightT>();
    constexpr int32_t chunkElements = static_cast<int32_t>(TOPK_WEIGHT_COPY_ELEMENTS);
    const int32_t processCount = Ops::Base::CeilDiv(rows, chunkElements);
    for (int32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        const int32_t offset = processIndex * chunkElements;
        const int32_t elements = offset + chunkElements > rows ? rows - offset : chunkElements;
        const uint32_t bytes = static_cast<uint32_t>(elements * sizeof(TopkWeightT));
        DataCopyExtParams copyParams{1U, bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<TopkWeightT> padParams{false, 0U, 0U, 0U};
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        DataCopyPad(buffer, src[offset], copyParams, padParams);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        DataCopyPad(dst[offset], buffer, copyParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
    }
}

template <typename TopkWeightT>
template <bool TileReady, bool WaitForFullExpert>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::DispatchTokens(GMMAddrInfo& gmmAddrInfo, uint32_t expertIdx,
                                                                       GlobalTensor<QuantOutType> aGlobalTensor,
                                                                       bool useTileReady)
{
    int64_t widthA = Get<K_VALUE>(problemShape_);  // H 长度
    int64_t widthAScale = 0;
    int32_t pingpongId = 0;
    const uint64_t expertCopyLimit = TileReady ? static_cast<uint64_t>(Get<M_VALUE>(problemShape_)) : maxOutputSize_;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;
        uint32_t rowStartInDst =
            (dstRankIdx == 0 ? 0 : cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        if (rowStartInDst < expertCopyLimit) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > expertCopyLimit) {
                rowsCopyCnt = expertCopyLimit - rowStartInDst;
            }
            // Count exchange has already stored the exclusive source prefix
            // for every local expert. Do not cache only ten source slots or
            // advance the source cursor by a capacity-clipped receive count.
            const int32_t rowSrcOffset = preSumBeforeRankWorkSpace_(dstRankIdx * expertPerRank_ + expertIdx);
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor<QuantOutType> gmRemoteA;
                GlobalTensor<QuantScaleOutType> aScaleGlobalTensor;
                GlobalTensor<TopkWeightT> gmRemoteTopkWeights;
                GlobalTensor<TopkWeightT> dispatchedTopkWeights;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetAScale = rowDstOffsetPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * (widthA + widthAScale);
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                    GetRankWinAddrWithOffset(dstRankIdx, params_.peermemInfo.payloadOffset)));
                aScaleGlobalTensor.SetGlobalBuffer(
                    reinterpret_cast<__gm__ QuantScaleOutType*>(gmmAddrInfo.aScaleGlobal));
                if (!useTileReady) {
                    // Expert mode submits the complete peer/core slice. The
                    // existing two-slot DMA ping-pong still bounds UB usage.
                    CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale],
                                         gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
                }
                const uint64_t routedWeightOffset =
                    params_.peermemInfo.ptrRoutedTopkWeights - params_.peermemInfo.ptrBase;
                gmRemoteTopkWeights.SetGlobalBuffer(
                    reinterpret_cast<__gm__ TopkWeightT*>(GetRankWinAddrWithOffset(dstRankIdx, routedWeightOffset)));
                // rowDstOffsetPerCore is relative to the current expert.
                // Match gmmAddrInfo.aGlobal by applying the expert-row
                // prefix before copying its dispatched route weights.
                dispatchedTopkWeights.SetGlobalBuffer(
                    reinterpret_cast<__gm__ TopkWeightT*>(params_.workspaceInfo.ptrDispatchedTopkWeights) +
                    Get<IDX_M_OFFSET>(baseOffset_));
                if (useTileReady) {
                    using ReadyLayout = MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
                    int32_t copiedRows = 0;
                    while (copiedRows < rowsPerCore) {
                        const uint32_t expertRow = rowDstOffsetPerCore + copiedRows;
                        const uint32_t fragmentRows = ReadyLayout::FragmentRows(expertRow, rowsPerCore - copiedRows);
                        CopyGMToGMPerTokenV2(aGlobalTensor[static_cast<int64_t>(expertRow) * widthA],
                                             aScaleGlobalTensor[gmOffsetAScale],
                                             gmRemoteA[gmOffsetPeer + static_cast<int64_t>(copiedRows) * widthA],
                                             fragmentRows, widthA, pingpongId);
                        CopyGMToGMTopkWeights(dispatchedTopkWeights[expertRow],
                                              gmRemoteTopkWeights[rowSrcOffsetPerCore + copiedRows], fragmentRows);
                        // Both payload and early-TopK weights must be visible
                        // before the corresponding GMM1/SwiGLU can consume them.
                        SyncFunc<AscendC::HardEvent::MTE3_S>();
                        AtomicAdd(gmmAddrInfo.groupFlagList2 + (expertRow / ReadyLayout::kReadyRows) * INT_CACHELINE,
                                  static_cast<int32_t>(fragmentRows));
                        copiedRows += fragmentRows;
                    }
                } else {
                    CopyGMToGMTopkWeights(dispatchedTopkWeights[rowDstOffsetPerCore],
                                          gmRemoteTopkWeights[rowSrcOffsetPerCore], rowsPerCore);
                }
            }
        }
        // Publish partition completion only after both payload and routed
        // weights are visible. Empty partitions also contribute to the target.
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        if constexpr (TileReady) {
            // Every source/core partition, including empty partitions, retires
            // once. AIC can stop polling tile flags after the whole expert lands.
            __gm__ int32_t* expertFlag =
                reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrFlagDispatchToGmm1) +
                expertIdx * INT_CACHELINE;
            AtomicAdd(expertFlag, int32_t(1));
        } else {
            AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
}

template <typename TopkWeightT>
__aicore__ inline bool MegaMoeBF16Forward<TopkWeightT>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    if (Get<IDX_M_OFFSET>(baseOffset_) + Get<M_VALUE>(problemShape_) > maxOutputSize_) {
        Get<M_VALUE>(problemShape_) = maxOutputSize_ - Get<IDX_M_OFFSET>(baseOffset_);
    }

    // split_m，when m=0, skip
    if (Get<M_VALUE>(problemShape_) <= 0) {
        return false;
    }

    return true;
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::UpdateGlobalBuffer(GMMAddrInfo& gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal =
            params_.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr (g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal =
                params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_),
                Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_),
                Get<IDX_M_OFFSET>(baseOffset_),
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);  // GMMSwiglu调用
        }
    } else if constexpr (g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal =
            params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal =
            params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList =
        (__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2 + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 +
                                 Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    if (dispatchTileReadyBase_ != nullptr && gmmIndex == 0) {
        // Existing cacheline has sixteen words; at most seven experts are
        // eligible under the unchanged shape policy. Distinct words prevent
        // run-ahead producer reuse across experts, including empty experts.
        const uint32_t readyWord =
            preparedPhaseEnabled_ ? static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(baseOffset_)) : 0U;
        gmmAddrInfo.groupFlagList2 = reinterpret_cast<__gm__ int32_t*>(dispatchTileReadyBase_) + readyWord;
    }
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    const uint64_t countElements =
        static_cast<uint64_t>(worldSize_) *
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    constexpr uint32_t chunkElements =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes / sizeof(int32_t);
    const uint32_t zeroElements = countElements < chunkElements ? static_cast<uint32_t>(countElements) : chunkElements;
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp{AscendC::TPosition::VECIN, 0, zeroElements};
    Duplicate(tmp, 0, zeroElements);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    for (uint64_t offset = 0; offset < countElements; offset += chunkElements) {
        const uint32_t elements =
            countElements - offset < chunkElements ? static_cast<uint32_t>(countElements - offset) : chunkElements;
        DataCopy(tokenPerExpertWin_[offset], tmp, elements);
    }
    // Unpermute reuses UB address zero after the existing all-core barrier.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::UnpermuteV2()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (MegaMoeBF16Routing::TryUnpermute(params_.workspaceInfo.expandedRowIdxGather, params_.peermemInfo.ptrD,
                                         params_.y2GmAddr, k_, topK_, coreOffset, coreLen,
                                         Catlass::Arch::Ascend950::UB_SIZE)) {
        return;
    }
    MegaMoeBF16Routing::UnpermuteByHiddenChunk(params_.workspaceInfo.expandedRowIdxGather, params_.peermemInfo.ptrD,
                                               params_.y2GmAddr, k_, topK_, coreOffset, coreLen);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount =
        (__gm__ int32_t*)(params_.peermemInfo.ptrBase + MegaMoeBF16Training::PeerSyncCountOffset(worldSize_));
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Process()
{
    using namespace MegaMoeBF16Training;
    const bool transW = params_.tilingData->transW;
    switch (ResolveSavedGateUpMode(params_.tilingData->savedGateUpMode, params_.tilingData->swigluLimit)) {
        case kSaveNone:
            if (transW) {
                ProcessSelected<kSaveNone, true>();
            } else {
                ProcessSelected<kSaveNone, false>();
            }
            break;
        case kSaveBf16:
            if (transW) {
                ProcessSelected<kSaveBf16, true>();
            } else {
                ProcessSelected<kSaveBf16, false>();
            }
            break;
        default:
            if (transW) {
                ProcessSelected<kSaveMxFp8, true>();
            } else {
                ProcessSelected<kSaveMxFp8, false>();
            }
            break;
    }
}

template <typename TopkWeightT>
template <uint32_t SaveMode, bool TransW>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ProcessSelected()
{
    if (MegaMoeBF16Training::ForwardUsesGmSwiglu(k_, SaveMode)) {
        ProcessWithSaveMode<SaveMode, TransW, true>();
    } else {
        ProcessWithSaveMode<SaveMode, TransW, false>();
    }
}

template <typename TopkWeightT>
template <uint32_t SaveMode, bool TransW, bool GmSwiglu>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ProcessWithSaveMode()
{
    // The Host enqueues the local token-table memset on this same stream and
    // launches immediately. Reaching this barrier proves
    // every rank's local clear has completed before any peer starts routing.
    // The 72 KiB synchronization header is persistent in this mode, so the
    // monotonically increasing generation counters remain valid across calls.
    if constexpr (g_coreType == AscendC::AIV) {
        CrossRankSyncInWorldSize();
    }
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    // 1. Initialize routing and pack BF16 tokens without quantization.
    moe_init_routing_v3<bfloat16_t, QuantOutType>(
        params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr, params_.peermemInfo.ptrA0,
        params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRankAddr_, nullptr,
        params_.workspaceInfo.expandedRowIdxGather, params_.workspaceInfo.ptrA,
        &params_.tilingData->moeInitRoutingTilingData, params_.tilingData->moeInitRoutingTilingData.tilingKey);

    // Finish and release the optional gather-only UB phase before Dispatch
    // buffer allocation.
    bool routedWeightsReady = false;
    if constexpr (g_coreType == AscendC::AIV) {
        routedWeightsReady = MegaMoeBF16Routing::TryGatherTopkWeights<TopkWeightT>(
            params_.probsGmAddr, params_.workspaceInfo.expandedRowIdx, params_.peermemInfo.ptrRoutedTopkWeights,
            static_cast<uint64_t>(m_) * topK_, aivCoreIdx_, blockAivNum_, true);
    }

    // 2. buffer申请 & workspace Init
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);

    // Use the same sorted-row order as ptrA0 so Dispatch can move each token
    // row and its BF16 routing weight with identical source/destination spans.
    // The large-table fallback uses FirstBuffInit's UB staging buffer and
    // the same MTE3 publication boundary as the batched path above.
    if (!routedWeightsReady) {
        GatherRoutedTopkWeights();
    }
    SyncAll<true>();

    ResetWorkSpaceFlagList();

    // 3. cnt值全ep域内的发送以及本端接收整理
    CrossRankSyncCntPerExpert();
    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }
    // Split training state only snapshots the stable token-count table. AIV0
    // has released firstPipe above, so copy it now with a short-lived pipe.
    // The existing pre-GMM1 SyncAll below also proves the save UB is released
    // before the AIC may reuse overlapping storage for GMM1 fixpipe.
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 0 && useSplitTrainingState_) {
            TPipe savePipe;
            savePipe.InitBuffer(SaveStateBuf_, MegaMoeForwardSave::kCopyBufferBytes);
            SaveForwardState();
            PipeBarrier<PIPE_ALL>();
            savePipe.Destroy();
        }
    }
    SyncAll<false>();

    // 4. for循环按照本卡专家id进行Dispatch，内部gmm1SwigluQuant按照基本块进行流水
    GMMAddrInfo gmmAddrInfo{};
    {
        constexpr bool waitForFullExpert = DEEPEP_BF16_FORCE_EXPERT_READY != 0;
        const bool shapeTileReadyCandidate =
            !waitForFullExpert &&
            MegaMoeBF16Training::ForwardDispatchTileReadyLayout::UseTileReadyForShape(m_, topK_, expertPerRank_);
        const uint32_t readyBlocksPerProducerPartition =
            MegaMoeBF16Training::ForwardDispatchTileReadyLayout::ReadyBlocksPerProducerPartition(
                m_, topK_, expertPerRank_, worldSize_, dispatchBlockNumPerEP_);
        bool startupExpertHandled = false;
        uint32_t gmm1N = Get<N_VALUE>(problemShape_);
        uint32_t gmm1K = Get<K_VALUE>(problemShape_);
        epilogueOp_.UpdateNextProblem({0, gmm1N / SWIGLU_N_HALF, gmm1K, 0});
        constexpr bool useGmSwiglu = GmSwiglu;
        MegaMoeBF16ForwardCatlassImpl::ForwardGroupMatmulParam<0, TransW, useGmSwiglu> gmm1Params(
            gmm1N, gmm1K, params_.tilingData->blockNumPerEP);
        for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
            if (!UpdateGroupParams(expertIdx)) {
                continue;
            }
            const bool useTileReady =
                preparedPhaseEnabled_
                    ? shapeTileReadyCandidate &&
                          Get<M_VALUE>(problemShape_) >=
                              MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kMinReadyRowsPerExpert
                    : MegaMoeBF16Training::ForwardDispatchTileReadyLayout::UseTileReadyForStartupExpert(
                          shapeTileReadyCandidate, startupExpertHandled, Get<M_VALUE>(problemShape_));
            // Each actual non-empty expert independently retains the
            // existing small-expert fallback. No runtime sorting/search.
            startupExpertHandled = true;
            UpdateGlobalBuffer(gmmAddrInfo, 0);
            if constexpr (g_coreType == AscendC::AIV) {
                if (subBlockIdx_ == 1) {
                    // Bypass L2 only for ordinary BF16 Dispatch packed-X stores.
                    // GMM1 creates its own input tensor with its original read policy.
                    GlobalTensor<QuantOutType> dispatchedX;
                    dispatchedX.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(gmmAddrInfo.aGlobal));
                    dispatchedX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    DispatchTokens<true, waitForFullExpert>(gmmAddrInfo, expertIdx, dispatchedX, useTileReady);
                }
            }
            // Producer and consumer share one invocation-wide protocol
            // choice. Both retain the expert-completion counter so auto
            // mode can fall back without changing workspace layout.
            if constexpr (useGmSwiglu) {
                MegaMoeBF16ForwardCatlassImpl::GroupMatmulSwigluGm<true, waitForFullExpert, SaveMode, TransW>(
                    gmm1Params, epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, expertIdx,
                    useTileReady, readyBlocksPerProducerPartition, Get<IDX_C_OFFSET>(baseOffset_));
            } else {
                MegaMoeBF16ForwardCatlassImpl::GroupMatmulSwiglu<true, waitForFullExpert, SaveMode, TransW>(
                    gmm1Params, epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, gmm1VecSetSyncCom_,
                    expertIdx, useTileReady, readyBlocksPerProducerPartition);
            }
        }
    }

    if constexpr (g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
            if (useSplitTrainingState_) {
                // GMM2 only needs this core's AIV1 to release the overlapping
                // Dispatch UB. Do not wait for unrelated GMM1 cores.
                NotifyCube(GMM2_UB_READY_FLAG);
            }
        }
    }

    // Legacy embedded-state callers also save packed X and routing metadata.
    // Retain their historical all-core snapshot boundary; moving that large
    // copy before Dispatch would serialize their main path.
    if (!useSplitTrainingState_) {
        SyncAll<false>();
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1) {
                TPipe savePipe;
                savePipe.InitBuffer(SaveStateBuf_, MegaMoeForwardSave::kCopyBufferBytes);
                SaveForwardState();
                PipeBarrier<PIPE_ALL>();
                savePipe.Destroy();
                NotifyCube(GMM2_UB_READY_FLAG);
            }
        }
    }
    // cube仅等待本核AIV1(dispatch)释放UB后再进入GMM2 fixpipe；不等待AIV0(swiglu)，
    // 从而保留“gmm2与swiglu尾块并行”，swiglu逐专家由groupFlagList继续把关。
    if constexpr (g_coreType == AscendC::AIC) {
        WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    {
        uint32_t gmm2N = Get<K_VALUE>(problemShape_);
        uint32_t gmm2K = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;
        MegaMoeBF16ForwardCatlassImpl::ForwardGroupMatmulParam<1, TransW> gmm2Params(gmm2N, gmm2K);
        bool runGmm2 = true;
        if (runGmm2) {
            for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
                if (!UpdateGroupParams(expertIdx)) {
                    continue;
                }
                UpdateGlobalBuffer(gmmAddrInfo, 1);
                MegaMoeBF16ForwardCatlassImpl::GroupMatmul2(gmm2Params, params_, problemShape_, gmmAddrInfo,
                                                            startBlockIdx_, gmm2VecSetSyncCom_, expertIdx, 0,
                                                            gmm2PingPongIdx_, rankId_);
            }
        }
    }
    EndSync();
    EndGMM2Sync();
    SyncAll<true>();

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr (g_coreType == AIV) {
        ResetTokenPerExpert();
        SyncAll<true>();
        CrossRankSyncInWorldSize();  // 全卡软同步，确认combine send完成
        UnpermuteV2();
        PipeBarrier<PIPE_ALL>();
        // Publish a safe-to-reuse boundary only after the last SHMEM payload
        // reader has drained. A faster rank may enqueue the next invocation as
        // soon as it leaves this barrier without racing a slower peer.
        CrossRankSyncInWorldSize();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}  // namespace MegaMoeImpl
#endif
