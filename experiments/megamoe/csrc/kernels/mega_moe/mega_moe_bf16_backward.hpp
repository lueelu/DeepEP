// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*
 * DeepEP MegaMoe BF16 training backward kernel.
 *
 * Notation: R rounds to BF16; h is the unweighted FP32 SwiGLU activation
 * with the configured clipping, Hp = R(h*p), and dh = FP32(p)*FP32(dHp). Neither
 * h nor dh denotes the hidden width. For each expert, canonical weights
 * are W1=[2F,K] and W2=[K,F], where K=k_ is the hidden width and n_=2F.
 * The formulas below use that layout; transW=false uses the transposed
 * weight storage and matching gradient layouts.
 *
 * Stage pipeline (saved modes compute each expert's dW2 immediately before P2):
 *
 *   P0-A  route gather           publish source-token dY and saved route indices (AIV)
 *   P0-B  restore count table + Dispatch dO from peers -> ptrDO (AIV0)
 *   P0-C  save=none recompute    Z = Xpacked @ W1.T -> gate/up GM planes (AIC)
 *   P1    GMM2 Grad + epilogue   dHp = dO @ W2; h, Hp, dp partials, dh -> dZ
 *   P2    GMM1 Grad + Combine    packed dX = dZ @ W1 -> peers' ptrD
 *   P3    GMM1 dW                dW1 = dZ.T @ Xpacked -> grad_weight1 (AIC)
 *   P4    save=none GMM2 dW      dW2 = dO.T @ Hp -> grad_weight2 (AIC)
 *   P5    Unpermute              FP32 sum_slot(ptrD) -> BF16 grad_x (AIV)
 *   P6    Reduce/route/unpermute  P1 dp partials -> grad_topk (AIV)
 *
 * P1 writes row-owned dHp to GM. Saved-mode Wave waits for each dO block;
 * recompute waits for complete-expert dO and Gate/Up. Row-ready consumers
 * may replace a 128-row block with Hp after its last dHp column is visible.
 * The expert-ready GM path waits for all producers. The tiled recompute path
 * hands dHp through the paired AIV UB slots instead.
 * Whole-expert Dispatch publishes bwFlagDispatch_[expert]; Wave Dispatch
 * publishes 128-row ready counters. P0 staging occupies a fixed UB window above the
 * P1/P2 work tiles (see StageUb), so dispatch can overlap earlier experts'
 * P1 work without a whole-stage UB ownership handover.
 *
 * Saved forward context (routing metadata, packed X and count-table snapshot)
 * and backward-only scratch are passed as separate buffers.
 */
#ifndef MEGA_MOE_BF16_BACKWARD_HPP
#define MEGA_MOE_BF16_BACKWARD_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "common/mega_moe_utils.hpp"
#include "stage/mega_moe_combine_send.hpp"
#include "stage/mega_moe_route_bf16.hpp"
#include "common/mega_moe_bf16_peer_row_view.hpp"
#include "stage/mega_moe_unpermute_bf16.hpp"
#include "epilogue/block_epilogue_swiglu_bf16_backward.hpp"
#include "catlass_gmm/mega_moe_bf16_catlass_impl_backward.hpp"

using namespace AscendC;

namespace MegaMoeImpl {

using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// With tiled save=none output, P1 alternates between the paired AIVs: AIC event 0 targets
// AIV0 and event FLAG_ID_MAX targets AIV1; each AIV consumes local event 0.
// Its P2 uses AIV0 with local ping-pong events 0/1. Saved modes use GM
// completion counters instead of these per-tile output events.
constexpr uint16_t BW_P1_EVENT = 0;  // save=none P1 AIC <-> AIV0/AIV1
constexpr uint32_t BW_TOPK_ROUTE_BATCH_ROWS = 1024;
constexpr uint32_t BW_TOPK_PARTIAL_UB_BYTES = 128 * 1024;

enum BackwardPhase {
    BW_PHASE_RECOMPUTE = 0,
    BW_PHASE_GMM2GRAD = 1,
    BW_PHASE_GMM1GRAD = 2,
    BW_PHASE_DW1 = 3,
    BW_PHASE_DW2 = 4
};

template <typename TopkWeightT = bfloat16_t>
class MegaMoeBF16Backward {
public:
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    using QuantOutType = bfloat16_t;
    __aicore__ inline MegaMoeBF16Backward(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR gradX, GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
                                GM_ADDR trainingSavedState, GM_ADDR dispatchOutput, GM_ADDR backwardWorkspace,
                                MegaMoeBf16TilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SetTimer(MegaMoeTrainTimerDevice* timer)
    {
        timer_ = timer;
        params_.timer = timer;
    }

private:
    __aicore__ inline void InitStageLayout();
    __aicore__ inline AscendC::LocalTensor<QuantOutType> StageUb(uint32_t bytes);
    __aicore__ inline void UnpermuteBuffInit(TPipe* tPipe);
    __aicore__ inline void TopkGradBuffInit(TPipe* tPipe);
    __aicore__ inline void RestoreTokenPerExpert();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void ClearEmptyExpertGradWeights();
    __aicore__ inline void ClearGradWeightPlane(GM_ADDR base, uint64_t planeElements, uint32_t expertIdx);
    __aicore__ inline void DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx);
    using WavePlan = MegaMoeBF16Impl::BackwardWavePlan;
    __aicore__ inline bool UseStagedWave(uint32_t ordinal) const
    {
        return MegaMoeBF16Impl::UseBackwardStagedWave(savedGateUpMode_, n_ / 2) && params_.peerRowsGroups == 0U &&
               ordinal < splitExperts_;
    }
    __aicore__ inline bool UseRecomputeDhpRows(uint32_t ordinal) const
    {
        return ordinal < splitExperts_ && MegaMoeBF16Impl::UseRecomputeDhpRowReady(savedGateUpMode_, n_ / 2);
    }
    // For row-ready recompute, the expert counter covers Gate/Up; dHp uses per-row slots.
    __aicore__ inline __gm__ int32_t* RecomputeReady(uint32_t expert) const
    {
        return reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + expert * INT_CACHELINE;
    }
    // The host clears this slot for each backward. One AIV per expert publishes
    // a positive chunk size before dispatch; all three Wave stages read it.
    __aicore__ inline __gm__ int32_t* WaveChunkSlot(uint32_t expert) const
    {
        return reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + expert * INT_CACHELINE + 2;
    }
    __aicore__ inline void PrepareWaveChunks()
    {
        if (!UseStagedWave(0) || aivCoreIdx_ >= expertPerRank_) return;
        const MegaMoeBF16Impl::Wave::GmPeerPrefix prefix{
            reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM)};
        MegaMoeBF16Impl::PreparedExpertCursor cursor;
        for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
            uint32_t rows = 0, ordinal = 0;
            if (cursor.Next(groupList_.GetValue(expert), groupListType_, maxOutputSize_, rows, ordinal) &&
                expert % blockAivNum_ == aivCoreIdx_ && UseStagedWave(ordinal)) {
                const WavePlan plan(rows, n_ / 2, blockNum_, 0, 0);
                AtomicAdd(WaveChunkSlot(expert),
                          static_cast<int32_t>(plan.SelectChunkTiles(prefix, worldSize_, expertPerRank_, expert)));
            }
        }
    }
    __aicore__ inline WavePlan MakeWavePlan(uint32_t expert, uint32_t rows, uint32_t precedingTiles) const
    {
        const MegaMoeBF16Impl::Wave::GmPeerPrefix prefix{
            reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM)};
        const uint32_t selfBegin = rankId_ ? prefix.Get((rankId_ - 1) * expertPerRank_ + expert) : 0;
        int32_t chunk;
        do {
            chunk = ReadGmByPassDCache(WaveChunkSlot(expert));
        } while (chunk == 0);
        return WavePlan(rows, n_ / 2, blockNum_, selfBegin, precedingTiles, static_cast<uint32_t>(chunk));
    }
    __aicore__ inline void DispatchStagedBackward(GM_ADDR dstBase, uint32_t expert, const WavePlan& plan);
    template <class Prefix>
    __aicore__ inline void DispatchWaveTokensBackward(GM_ADDR dstBase, uint32_t expertIdx, const Prefix& prefix);
    __aicore__ inline void CopyDispatchedGradRows(GlobalTensor<QuantOutType> dst, uint32_t source, uint32_t sourceRow,
                                                  uint32_t rows, int32_t& ping);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(Bf16GMMAddrInfo& gmmAddrInfo, uint32_t phase, uint32_t expertIdx,
                                              int64_t rowBase = -1);
    __aicore__ inline void UnpermuteBackward();
    __aicore__ inline void ReduceDpPartialBatch(uint16_t batchRows, uint16_t partialTiles, uint16_t partialStride);
    __aicore__ inline void ReduceAndRouteTopkGrad();
    __aicore__ inline void CopyTopkGradBatch(uint64_t packedRow, uint64_t remoteRow, uint32_t rank, uint32_t rows,
                                             uint32_t partialTiles, uint32_t partialStride, uint32_t expert,
                                             uint32_t expertRow);
    __aicore__ inline void UnpermuteTopkGrad();
    __aicore__ inline void CrossRankSyncInWorldSize();

    template <bool TransW>
    __aicore__ inline void GMM1RecomputePhase();
    template <bool TransW, bool GmOutput>
    __aicore__ inline void GMM2GradPhase();
    template <bool TransW, bool ExpertReady>
    __aicore__ inline void GMM1GradPhase();
    template <bool TransW>
    __aicore__ inline void GMM1GradReuseDOPhase();
    template <bool TransW>
    __aicore__ inline void GMM1DWPhase();
    template <bool TransW>
    __aicore__ inline void GMM2DWPhase();

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    // P1/P2 own three work tiles at UB offset zero. P0 reuses the disjoint
    // staging region above them, sequentially on each AIV.
    static constexpr uint32_t BW_TILE_BYTES =
        MegaMoeBF16BackwardCatlassImpl::L1_TILE_M * MegaMoeBF16BackwardCatlassImpl::L1_TILE_N * sizeof(bfloat16_t);
    static constexpr uint32_t BW_STAGE_BASE_BYTES = 3 * BW_TILE_BYTES;
    static constexpr uint32_t BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES =
        MegaMoeBF16Impl::ForwardDispatchUbLayout::kCountResetChunkBytes;
    static_assert(BW_STAGE_BASE_BYTES + BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES <=
                      MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE,
                  "the static staging window must hold the gradient-zero tile");

    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> TopkGradRouteBuf_;
    TBuf<> TopkGradPartialBuf_;
    TBuf<> TopkGradPeerPrefixBuf_;

    __gm__ Mc2Aclnn::Bf16MoeContext* mc2Context_{nullptr};
    Bf16Params params_{};
    MegaMoeTrainTimerDevice* timer_ = nullptr;
    BackwardSavedStateInfo savedState_{};
    BackwardWorkspaceInfo bwWs_{};
    uint32_t splitExperts_ = 0;
    TupleShape problemShape_{};
    using BlockEpilogue =
        MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<MegaMoeBF16BackwardCatlassImpl::L1TileShape, TopkWeightT>;
    BlockEpilogue epilogueOp_;

    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> savedTokenPerExpert_;
    __gm__ int32_t* bwFlagDispatch_{nullptr};
    __gm__ int32_t* bwFlagAiv1Ready_{nullptr};
    __gm__ int32_t* bwFlagHDone_{nullptr};

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t n_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint32_t savedGateUpMode_ = MegaMoeBF16Impl::kSaveNone;
    uint64_t maxOutputSize_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 0;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    int32_t dispatchCopyElementsPerPing_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint32_t borrowedDw2Tiles_ = 0;
    int64_t cumMRows_ = 0;
    int64_t preOffset_ = 0;
    GM_ADDR gradWeight1GmAddr_{nullptr};
    GM_ADDR gradWeight2GmAddr_{nullptr};
    GM_ADDR gradTopkWeightsGmAddr_{nullptr};
};

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::Init(GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights,
                                                              GM_ADDR weight1, GM_ADDR weight2, GM_ADDR gradX,
                                                              GM_ADDR gradTopkWeights, GM_ADDR gradWeight1,
                                                              GM_ADDR gradWeight2, GM_ADDR trainingSavedState,
                                                              GM_ADDR dispatchOutput, GM_ADDR backwardWorkspace,
                                                              MegaMoeBf16TilingData* tilingData)
{
    splitExperts_ = tilingData->backwardSplitExperts;
    m_ = tilingData->m;
    k_ = tilingData->k;
    n_ = tilingData->n;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    rankId_ = tilingData->rankId;
    savedGateUpMode_ = tilingData->savedGateUpMode;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::Bf16MoeContext*>(context);
    for (int i = 0; i < worldSize_; i++) {
        bf16WinRankAddr[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    params_.aGmAddr = gradY;
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.probsGmAddr = topkWeights;
    params_.y2GmAddr = gradX;
    savedState_ = BackwardSavedStateInfo(trainingSavedState, dispatchOutput, tilingData);
    params_.workspaceInfo = savedState_;
    params_.peermemInfo = Bf16PeermemInfo(bf16WinRankAddr[rankId_], tilingData, true);
    params_.tilingData = tilingData;
    params_.peerRowsGroups = 0;
    gradWeight1GmAddr_ = gradWeight1;
    gradWeight2GmAddr_ = gradWeight2;
    gradTopkWeightsGmAddr_ = gradTopkWeights;
    bwWs_ = BackwardWorkspaceInfo(backwardWorkspace, tilingData, splitExperts_);
    bwFlagDispatch_ = (__gm__ int32_t*)bwWs_.ptrBwFlagDispatchToGmm2Grad;
    bwFlagAiv1Ready_ = (__gm__ int32_t*)bwWs_.ptrBwFlagAiv1Ready;
    bwFlagHDone_ = (__gm__ int32_t*)bwWs_.ptrBwFlagHDone;

    perRankStridesInTokenPerExpert_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    savedTokenPerExpert_.SetGlobalBuffer((__gm__ int32_t*)savedState_.ptrSavedTokenPerExpert);

    GM_ADDR gatePlane =
        (savedGateUpMode_ == MegaMoeBF16Impl::kSaveNone) ? bwWs_.ptrRecomputeGate : savedState_.ptrSavedGate;
    GM_ADDR upPlane = (savedGateUpMode_ == MegaMoeBF16Impl::kSaveNone) ? bwWs_.ptrRecomputeUp : savedState_.ptrSavedUp;
    epilogueOp_.Init({gatePlane, upPlane, savedState_.ptrSavedGateScale, savedState_.ptrSavedUpScale,
                      savedState_.ptrSavedGateHigh, savedState_.ptrSavedUpLow, savedState_.ptrSavedUpHigh,
                      bwWs_.ptrGradZ, bwWs_.ptrHp, bwWs_.ptrDpProducts, params_.workspaceInfo.ptrDispatchedTopkWeights,
                      0, 0, tilingData->swigluLimit, savedGateUpMode_, bwWs_.ptrWideDp});
    epilogueOp_.UpdateNextProblem({0, n_, 0, 0});
}

// Fixed-position P0 staging window at the top of UB, above the P1/P2 tile
// region [0, BW_STAGE_BASE_BYTES). P0-A gather, count restore and P0-B
// dispatch run sequentially on each lane and share this region.
// Saved Wave consumes dO by canonical M block. Other dHp paths wait for the expert.
// Raw fixed-position binding, same pattern as the epilogue's own tile
// members (VECIN raw tiles at fixed byte offsets, fed by plain DataCopy in
// both directions).  The return type spells bfloat16_t, NOT the class alias
// QuantOutType: in an out-of-line member definition the return type sits
// before the qualified MegaMoeBF16Backward<TopkWeightT>:: name, where class-scope names
// are not visible yet (parameters and the body are fine).
template <typename TopkWeightT>
__aicore__ inline AscendC::LocalTensor<bfloat16_t> MegaMoeBF16Backward<TopkWeightT>::StageUb(uint32_t bytes)
{
    return AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECIN, BW_STAGE_BASE_BYTES, bytes / sizeof(bfloat16_t));
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::InitStageLayout()
{
    constexpr uint32_t available = MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE - BW_STAGE_BASE_BYTES;
    constexpr uint32_t indexBytes = Bf16Route::kIndexBatch * sizeof(int32_t);
    constexpr uint32_t cols = (available - indexBytes) / (2U * sizeof(QuantOutType)) / 32U * 32U;
    static_assert(cols > 0 && available >= MegaMoeImpl::kCopyBufferBytes);
    dispatchCopyElementsPerPing_ = k_ < cols ? k_ : cols;
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::UnpermuteBuffInit(TPipe* tPipe)
{
    uint32_t unpermuteBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, unpermuteBufAlign);
    tPipe->InitBuffer(DataResFp32Buf_, unpermuteBufAlign * 2);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::TopkGradBuffInit(TPipe* tPipe)
{
    tPipe->InitBuffer(TopkGradRouteBuf_,
                      Ops::Base::CeilAlign(BW_TOPK_ROUTE_BATCH_ROWS * static_cast<uint32_t>(sizeof(TopkWeightT)),
                                           static_cast<uint32_t>(ALIGN_32)));
    tPipe->InitBuffer(TopkGradPartialBuf_, BW_TOPK_PARTIAL_UB_BYTES);
    if (params_.peerRowsGroups != 0U) {
        const uint32_t prefixBytes = worldSize_ * expertPerRank_ * sizeof(int32_t);
        tPipe->InitBuffer(TopkGradPeerPrefixBuf_, Ops::Base::CeilAlign(prefixBytes, static_cast<uint32_t>(ALIGN_32)));
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::RestoreTokenPerExpert()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BW_RESTORE_TABLE);
    int32_t perRank = static_cast<int32_t>(perRankStridesInTokenPerExpert_);
    LocalTensor<int32_t> tmpBuffer = StageUb(perRank * sizeof(int32_t)).template ReinterpretCast<int32_t>();
    // P0 has no TPipe; SyncFunc would access an invalid pipe. Use explicit events.
    for (int32_t sourceRank = static_cast<int32_t>(aivCoreIdx_); sourceRank < static_cast<int32_t>(worldSize_);
         sourceRank += static_cast<int32_t>(blockAivNum_)) {
        DataCopy(tmpBuffer, savedTokenPerExpert_[sourceRank * perRank], perRank);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(tokenPerExpertWin_[sourceRank * perRank], tmpBuffer, perRank);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ResetTokenPerExpert()
{
    MegaMoeTrainTimerFixedScope timerReset(timer_, MegaMoeTrainTimer::AIV_RESET);
    // Restored counts must be cleared before the next forward's atomic count exchange.
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    const uint64_t countElements =
        static_cast<uint64_t>(worldSize_) * static_cast<uint32_t>(perRankStridesInTokenPerExpert_);
    constexpr uint32_t chunkElements =
        MegaMoeBF16Impl::ForwardDispatchUbLayout::kCountResetChunkBytes / sizeof(int32_t);
    const uint32_t zeroElements = countElements < chunkElements ? static_cast<uint32_t>(countElements) : chunkElements;
    // The P6 pipe is destroyed; pair events without fetching a TPipe event.
    SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    LocalTensor<int32_t> tmp{AscendC::TPosition::VECIN, 0, zeroElements};
    Duplicate(tmp, 0, zeroElements);
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (uint64_t offset = 0; offset < countElements; offset += chunkElements) {
        const uint32_t elements =
            countElements - offset < chunkElements ? static_cast<uint32_t>(countElements - offset) : chunkElements;
        DataCopy(tokenPerExpertWin_[offset], tmp, elements);
    }
    SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::CopyDispatchedGradRows(GlobalTensor<QuantOutType> dst,
                                                                                uint32_t source, uint32_t sourceRow,
                                                                                uint32_t rows, int32_t& ping)
{
    const auto& peer = params_.peermemInfo;
    GlobalTensor<int32_t> routeIds;
    routeIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                                 GetRankWinAddrWithOffset(bf16WinRankAddr, source, peer.routeIndices - peer.ptrBase)) +
                             sourceRow);
    GlobalTensor<QuantOutType> input;
    input.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantOutType*>(GetRankWinAddrWithOffset(bf16WinRankAddr, source, peer.payloadOffset)));
    const uint32_t cols = dispatchCopyElementsPerPing_;
    const uint32_t ringBytes = 2U * cols * sizeof(QuantOutType);
    auto slots = StageUb(ringBytes);
    LocalTensor<int32_t> indices{TPosition::VECIN, BW_STAGE_BASE_BYTES + ringBytes, Bf16Route::kIndexBatch};
    Bf16Route::ForEachSourceToken(routeIds, indices, rows, topK_, [&](uint32_t row, uint32_t token) __aicore__ {
        for (uint32_t col = 0; col < k_; col += cols) {
            const uint32_t count = k_ - col < cols ? k_ - col : cols;
            ping ^= 1;
            const auto event = ping ? EVENT_ID1 : EVENT_ID0;
            auto buf = slots[ping * cols];
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopy(buf, input[static_cast<uint64_t>(token) * k_ + col], count);
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            DataCopy(dst[static_cast<uint64_t>(row) * k_ + col], buf, count);
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
    });
}

template <typename TopkWeightT>
template <class Prefix>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::DispatchWaveTokensBackward(GM_ADDR dstBase, uint32_t expertIdx,
                                                                                    const Prefix& prefix)
{
    using namespace MegaMoeBF16Impl::Wave;
    const auto info =
        DescribePeerExpert(prefix, worldSize_, expertPerRank_, expertIdx, maxOutputSize_, params_.peerRowsGroups);
    const PeerCounts<Prefix> counts{prefix, expertPerRank_, expertIdx, info.rows};
    int32_t ping = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        const uint32_t source = idx / dispatchBlockNumPerEP_;
        const uint32_t producer = idx % dispatchBlockNumPerEP_;
        const uint32_t rows = counts.Get(source);
        const uint32_t bandRows = (rows + dispatchBlockNumPerEP_ - 1U) / dispatchBlockNumPerEP_;
        const uint32_t ownBegin = producer * bandRows;
        const uint32_t ownEnd = ownBegin + bandRows < rows ? ownBegin + bandRows : rows;
        const uint32_t sourceBase = preSumBeforeRankWorkSpace_(source * expertPerRank_ + expertIdx);
        GlobalTensor<QuantOutType> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstBase));
        PeerRowCursor cursor(counts, worldSize_, rankId_, params_.peerRowsGroups, info.rows, info.firstSliceRows, 0U,
                             ownBegin < ownEnd ? info.rows : 0U);
        PeerTransfer transfer{};
        while (cursor.Next(transfer)) {
            if (transfer.sourceRank != source) continue;
            const uint32_t begin = ownBegin > transfer.sourceOrdinalBegin ? ownBegin : transfer.sourceOrdinalBegin;
            const uint32_t ordinalEnd = transfer.sourceOrdinalBegin + transfer.rowCount;
            const uint32_t end = ownEnd < ordinalEnd ? ownEnd : ordinalEnd;
            if (begin < end) {
                const uint32_t target = transfer.destinationRow + begin - transfer.sourceOrdinalBegin;
                CopyDispatchedGradRows(output[static_cast<uint64_t>(target) * k_], source, sourceBase + begin,
                                       end - begin, ping);
            }
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        AtomicAdd(bwFlagDispatch_ + expertIdx * INT_CACHELINE, int32_t(1));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::DispatchStagedBackward(GM_ADDR dstBase, uint32_t expert,
                                                                                const WavePlan& plan)
{
    const MegaMoeBF16Impl::Wave::GmPeerPrefix prefix{
        reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM)};
    const RowSliceReady ready(bwWs_.ptrStagedReady, cumMRows_, expert, WavePlan::tileM);
    GlobalTensor<QuantOutType> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstBase));
    int32_t ping = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    plan.ForEach(blockIdx_, [&](uint32_t row, uint32_t rows, uint32_t member, uint32_t owners) __aicore__ {
        const auto band = GetBalancedTokenRange(rows, member, owners);
        if (!band.count) return;
        if (timer_) {
            timer_->SetWave(plan.WaveId(row));
            timer_->SetSlice(expert, row + band.start, band.count);
        }
        MegaMoeTrainTimerSpanScope span(timer_, MegaMoeTrainTimer::AIV_BW_DISPATCH);
        MegaMoeBF16Impl::VisitCanonicalPeerRows(
            prefix, worldSize_, expertPerRank_, expert, row + band.start, row + band.start + band.count,
            [&](uint32_t peer, uint32_t ordinal, uint32_t target, uint32_t take) __aicore__ {
                const uint32_t source = preSumBeforeRankWorkSpace_(peer * expertPerRank_ + expert);
                CopyDispatchedGradRows(output[uint64_t(target) * k_], peer, source + ordinal, take, ping);
            });
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        AtomicAdd(ready.Slot(row), int32_t(1));
    });
    if (timer_) timer_->SetWave(MegaMoeTrainTimerSchema::NO_WAVE);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx)
{
    if (timer_ != nullptr) {
        timer_->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape_));
        timer_->SetTaskShape(128, 256);
    }
    MegaMoeTrainTimerSpanScope timerDispatch(timer_, MegaMoeTrainTimer::AIV_BW_DISPATCH);
    if (params_.peerRowsGroups != 0U) {
        using namespace MegaMoeBF16Impl::Wave;
        if (savedGateUpMode_ == MegaMoeBF16Impl::kSaveMxFp8) {
            // Saved MXFP8 writes dHp to GM. AIV0 can use its first tile for the
            // read-only prefix until Dispatch ends and its SwiGLU epilogue starts.
            static_assert(MegaMoeBF16Impl::kMaxTotalExperts * sizeof(int32_t) <= BW_TILE_BYTES);
            const uint32_t countElements = worldSize_ * expertPerRank_;
            LocalTensor<int32_t> prefixUb{TPosition::VECIN, 0, (countElements + 7U) / 8U * 8U};
            DataCopyPad(prefixUb, cumsumInWorkSpace_,
                        DataCopyExtParams{1U, countElements * uint32_t{sizeof(int32_t)}, 0U, 0U, 0U},
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
            DispatchWaveTokensBackward(dstBase, expertIdx, UbPeerPrefix{prefixUb});
        } else {
            const GmPeerPrefix prefix{reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM)};
            DispatchWaveTokensBackward(dstBase, expertIdx, prefix);
        }
        return;
    }
    int64_t widthA = k_;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;

        uint32_t rowStartInDst =
            (dstRankIdx == 0 ? 0 : cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            // Forward saved the exclusive source offset for each local
            // expert; this offset is independent of receive-capacity clipping.
            const int32_t rowSrcOffset = preSumBeforeRankWorkSpace_(dstRankIdx * expertPerRank_ + expertIdx);
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor<QuantOutType> aGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstBase));
                CopyDispatchedGradRows(aGlobalTensor[gmOffsetA], dstRankIdx, rowSrcOffsetPerCore, rowsPerCore,
                                       pingpongId);
            }
        }
        // Complete MTE3 payload writes before publishing readiness on PIPE_S.
        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AtomicAdd((__gm__ int32_t*)(bwFlagDispatch_ + expertIdx * INT_CACHELINE), int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename TopkWeightT>
__aicore__ inline bool MegaMoeBF16Backward<TopkWeightT>::UpdateGroupParams(uint32_t expertIdx)
{
    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    if (cumMRows_ + Get<M_VALUE>(problemShape_) > maxOutputSize_) {
        Get<M_VALUE>(problemShape_) = maxOutputSize_ - cumMRows_;
    }
    if (Get<M_VALUE>(problemShape_) <= 0) {
        return false;
    }
    return true;
}

// Use aligned int32 stores; the caller supplies a zero tile with its V -> MTE3 dependency.
template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ClearGradWeightPlane(GM_ADDR base, uint64_t planeElements,
                                                                              uint32_t expertIdx)
{
    constexpr uint64_t WORD_BYTES = sizeof(uint32_t);
    constexpr uint64_t WORDS_PER_32_BYTES = ALIGN_32 / WORD_BYTES;
    const uint64_t planeBytes = planeElements * sizeof(bfloat16_t);
    const uint64_t planeWords = planeBytes / WORD_BYTES;
    const uint64_t wordsPerCoreUnaligned = (planeWords + blockAivNum_ - 1) / blockAivNum_;
    const uint64_t wordsPerCore =
        ((wordsPerCoreUnaligned + WORDS_PER_32_BYTES - 1) / WORDS_PER_32_BYTES) * WORDS_PER_32_BYTES;
    const uint64_t coreOffset = static_cast<uint64_t>(aivCoreIdx_) * wordsPerCore;
    if (coreOffset >= planeWords) {
        return;
    }
    const uint64_t remainingWords = planeWords - coreOffset;
    const uint64_t coreWords = wordsPerCore < remainingWords ? wordsPerCore : remainingWords;
    const uint64_t expertOffset = static_cast<uint64_t>(expertIdx) * planeWords;

    GlobalTensor<int32_t> zeroGm;
    zeroGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base) + expertOffset + coreOffset, coreWords);
    LocalTensor<int32_t> zeroUb = StageUb(BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES).template ReinterpretCast<int32_t>();
    constexpr uint32_t ZERO_CHUNK_WORDS = BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES / sizeof(int32_t);
    for (uint64_t offset = 0; offset < coreWords; offset += ZERO_CHUNK_WORDS) {
        const uint32_t words =
            coreWords - offset < ZERO_CHUNK_WORDS ? static_cast<uint32_t>(coreWords - offset) : ZERO_CHUNK_WORDS;
        DataCopy(zeroGm[offset], zeroUb, words);
    }
}

// Non-empty experts are fully overwritten by the dW1/dW2 GEMMs. Empty experts
// do not launch either GEMM, so all AIVs cooperatively clear only those output
// planes while the AICs compute weight gradients for the non-empty experts.
// Reproduce UpdateGroupParams' maxOutputSize clamp: an expert beyond the
// receive-capacity boundary is also skipped by both GEMMs and must be zero.
template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ClearEmptyExpertGradWeights()
{
    int64_t previousGroupEnd = 0;
    uint64_t consumedRows = 0;
    bool zeroUbReady = false;
    const uint64_t gradWeight1PlaneElements = static_cast<uint64_t>(n_) * k_;
    const uint64_t gradWeight2PlaneElements =
        static_cast<uint64_t>(k_) * (n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF);

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        const int64_t groupValue = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        const int64_t routedRows = groupListType_ == 0 ? groupValue - previousGroupEnd : groupValue;
        if (groupListType_ == 0) {
            previousGroupEnd = groupValue;
        }

        uint64_t effectiveRows = 0;
        if (routedRows > 0 && consumedRows < maxOutputSize_) {
            const uint64_t availableRows = maxOutputSize_ - consumedRows;
            const uint64_t positiveRows = static_cast<uint64_t>(routedRows);
            effectiveRows = positiveRows < availableRows ? positiveRows : availableRows;
        }
        if (effectiveRows == 0) {
            if (!zeroUbReady) {
                // Reuse drained staging storage; InitGlobalMemory cannot follow TPipe allocation.
                LocalTensor<int32_t> zeroUb =
                    StageUb(BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES).template ReinterpretCast<int32_t>();
                constexpr uint32_t ZERO_CHUNK_WORDS = BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES / sizeof(int32_t);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                Duplicate(zeroUb, static_cast<int32_t>(0), ZERO_CHUNK_WORDS);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
                zeroUbReady = true;
            }
            ClearGradWeightPlane(gradWeight1GmAddr_, gradWeight1PlaneElements, expertIdx);
            ClearGradWeightPlane(gradWeight2GmAddr_, gradWeight2PlaneElements, expertIdx);
        } else {
            consumedRows += effectiveRows;
        }
    }
    if (zeroUbReady) {
        // Retire zero-tile reads before P5 reuses staging storage.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::UpdateGlobalBuffer(Bf16GMMAddrInfo& gmmAddrInfo,
                                                                            uint32_t phase, uint32_t expertIdx,
                                                                            int64_t rowBase)
{
    gmmAddrInfo.timer = timer_;
    const int64_t base = rowBase < 0 ? cumMRows_ : rowBase;
    const uint64_t bf16 = sizeof(bfloat16_t);
    if (phase == BW_PHASE_RECOMPUTE) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
        gmmAddrInfo.bGlobal = params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16;
    } else if (phase == BW_PHASE_GMM2GRAD) {
        gmmAddrInfo.aGlobal = bwWs_.ptrDO + base * k_ * bf16;
        gmmAddrInfo.bGlobal = params_.b2GmAddr + static_cast<uint64_t>(expertIdx) * k_ * (n_ / 2) * bf16;
    } else if (phase == BW_PHASE_GMM1GRAD) {
        gmmAddrInfo.aGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
        gmmAddrInfo.bGlobal = params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16;
    } else if (phase == BW_PHASE_DW1) {
        GM_ADDR x = params_.workspaceInfo.ptrA;
        if (params_.tilingData->transW) {
            gmmAddrInfo.aGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
            gmmAddrInfo.bGlobal = x + base * k_ * bf16;
        } else {
            gmmAddrInfo.aGlobal = x + base * k_ * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
        }
        gmmAddrInfo.cGlobal = gradWeight1GmAddr_ + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16;
    } else if (phase == BW_PHASE_DW2) {
        if (params_.tilingData->transW) {
            gmmAddrInfo.aGlobal = bwWs_.ptrDO + base * k_ * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrHp + base * (n_ / 2) * bf16;
        } else {
            gmmAddrInfo.aGlobal = bwWs_.ptrHp + base * (n_ / 2) * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrDO + base * k_ * bf16;
        }
        gmmAddrInfo.cGlobal = gradWeight2GmAddr_ + static_cast<uint64_t>(expertIdx) * k_ * (n_ / 2) * bf16;
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::CrossRankSyncInWorldSize()
{
    MegaMoeTrainTimerWaitScope timerWait(g_coreType == AscendC::AIV ? timer_ : nullptr,
                                         MegaMoeTrainTimer::AIV_SYNC_WAIT);
    Bf16CrossRankSyncInWorldSize(bf16WinRankAddr, params_.peermemInfo.ptrBase, MegaMoeBF16Impl::kPeerSyncCountOffset,
                                 rankId_, worldSize_, {aivCoreIdx_, blockAivNum_});
}

template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM1RecomputePhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, ColumnMajor, RowMajor>;
    Bf16GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> gmm1Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    // P0-C uses GEMM {m=Me, n=2F, k=K}: Xpacked @ W1.T for canonical
    // weights, or Xpacked @ W1 for transposed storage. Me is set per expert
    // by UpdateGroupParams; K=k_ and 2F=n_ are fixed here.
    Get<N_VALUE>(problemShape_) = n_;
    Get<K_VALUE>(problemShape_) = k_;
    uint32_t ordinal = 0;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        const int64_t expertRowBase = cumMRows_;
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_RECOMPUTE, expertIdx);
        const int64_t planeOffset =
            expertRowBase * (n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF) * sizeof(bfloat16_t);
        GroupMatmulRecompute(gmm1Params, problemShape_, gmmAddrInfo, bwWs_.ptrRecomputeGate + planeOffset,
                             bwWs_.ptrRecomputeUp + planeOffset, startBlockIdx, expertIdx);
        if constexpr (g_coreType == AscendC::AIC) {
            if (UseRecomputeDhpRows(ordinal)) {
                SetFlag<HardEvent::FIX_S>(EVENT_ID0);
                WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
                AtomicAdd(RecomputeReady(expertIdx), int32_t(1));
            }
        }
        ++ordinal;
        cumMRows_ += Get<M_VALUE>(problemShape_);
    }
}

template <typename TopkWeightT>
template <bool TransW, bool GmOutput>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM2GradPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
    Bf16GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    vecSetSyncCom_ = 0;
    uint32_t nextTargetAiv = 0;
    const int32_t dispatchTarget = static_cast<int32_t>(worldSize_ * dispatchBlockNumPerEP_);
    // GmOutput selects GM dHp for saved Gate/Up or a row-owned recompute epilogue. Otherwise the dual
    // target copier selects a paired AIV UB slot through Fixpipe subBlockId;
    // the AIC keeps one BlockMmad instance and L1/L0 event state machine.
    BackwardGroupMatmulParam<RowMajor, WeightLayout, GmOutput ? 0U : DUAL_AIV_TARGET, GmOutput> gmm2Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    // P1 GEMM produces dHp, not dh: canonical dHp = dO @ W2 is
    // [Me,K] x [K,F], with GEMM N=F=n_/2 and K=k_. Transposed storage
    // uses W2.T. The epilogue forms dH=FP32(dHp)*FP32(p), then
    // rounds dGate and dUp to BF16 dZ.
    Get<N_VALUE>(problemShape_) = n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
    Get<K_VALUE>(problemShape_) = k_;
    if constexpr (g_coreType == AscendC::AIV) {
        uint64_t routedRows = 0;
        for (uint32_t expert = 0; expert < expertPerRank_; ++expert) routedRows += groupList_.GetValue(expert);
        epilogueOp_.SetReductionRows(routedRows < maxOutputSize_ ? routedRows : maxOutputSize_);
    }
    uint32_t nonemptyOrdinal = 0;
    uint32_t precedingTiles = 0;
    uint32_t donor = 0, donorStart = 0, donorDone = 0;
    int64_t donorBase = 0;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        const int64_t expertRowBase = cumMRows_;
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_GMM2GRAD, expertIdx);
        __gm__ int32_t* dispatchFlag = bwFlagDispatch_ + expertIdx * INT_CACHELINE;
        if constexpr (GmOutput) {
            const bool recomputeRows = UseRecomputeDhpRows(nonemptyOrdinal);
            if (recomputeRows || UseStagedWave(nonemptyOrdinal)) {
                const auto rows = Get<M_VALUE>(problemShape_);
                const auto plan = recomputeRows ? WavePlan(rows, n_ / 2, blockNum_, 0, precedingTiles)
                                                : MakeWavePlan(expertIdx, rows, precedingTiles);
                if (recomputeRows) {
                    if (timer_) timer_->SetSlice(expertIdx, 0, rows);
                    if constexpr (g_coreType == AscendC::AIC) {
                        MegaMoeTrainTimerWaitScope wait(timer_, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                        while (ReadGmByPassDCache(dispatchFlag) < dispatchTarget) {
                        }
                    } else {
                        MegaMoeTrainTimerWaitScope wait(timer_, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD_WAIT);
                        while (ReadGmByPassDCache(RecomputeReady(expertIdx)) < static_cast<int32_t>(blockNum_)) {
                        }
                    }
                }
                // Each AIC takes a prefix of its own P2 output tiles, preserving the complete expert K.
                auto borrow = [&](__gm__ int32_t* priority, uint32_t arrivals) __aicore__ {
                    if constexpr (g_coreType == AscendC::AIC) {
                        while (donor < expertIdx) {
                            const int64_t end = groupList_.GetValue(donor);
                            const int64_t prev = groupListType_ == 0 && donor ? groupList_.GetValue(donor - 1) : 0;
                            const int64_t available = static_cast<int64_t>(maxOutputSize_) - donorBase;
                            const int64_t rows = end - prev < available ? end - prev : available;
                            if (rows <= 0) {
                                ++donor;
                                continue;
                            }
                            auto shape = problemShape_;
                            Get<M_VALUE>(shape) = TransW ? k_ : n_ / 2;
                            Get<N_VALUE>(shape) = TransW ? n_ / 2 : k_;
                            Get<K_VALUE>(shape) = rows;
                            BlockScheduler sched(
                                {static_cast<uint32_t>(Get<M_VALUE>(shape)), static_cast<uint32_t>(Get<N_VALUE>(shape)),
                                 static_cast<uint32_t>(rows)},
                                Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
                            const uint32_t total = sched.GetCoreLoops();
                            const uint32_t offset = (gmm2Params.blockIdx + blockNum_ - donorStart) % blockNum_;
                            const uint32_t owned = offset < total ? 1 + (total - 1 - offset) / blockNum_ : 0;
                            if (donorDone == owned) {
                                ++donor;
                                donorBase += rows;
                                donorDone = 0;
                                donorStart = (donorStart + total) % blockNum_;
                                continue;
                            }
                            auto* hpReady =
                                reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDZReady) + donor * INT_CACHELINE + 1;
                            if (ReadGmByPassDCache(hpReady) < static_cast<int32_t>(blockNum_)) return false;
                            if (ReadGmByPassDCache(priority) >= static_cast<int32_t>(arrivals)) return false;
                            Bf16GMMAddrInfo addr{};
                            UpdateGlobalBuffer(addr, BW_PHASE_DW2, donor, donorBase);
                            using DhpMmad = typename decltype(gmm2Params)::BlockMmad;
                            gmm2Params.blockMmad.~DhpMmad();
                            uint32_t start = donorStart, completed = 0;
                            {
                                BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> dw;
                                GMM2DW(dw, shape, addr, start, donor, donorDone, 1, priority, arrivals, &completed);
                            }
                            new (&gmm2Params.blockMmad) DhpMmad(gmm2Params.resource);
                            donorDone += completed;
                            borrowedDw2Tiles_ += completed;
                            return completed != 0;
                        }
                    }
                    return false;
                };
                GMM2GradStagedWave(gmm2Params, epilogueOp_, problemShape_, gmmAddrInfo,
                                   bwWs_.ptrHp + expertRowBase * (n_ / 2) * sizeof(bfloat16_t), expertIdx,
                                   expertRowBase, plan,
                                   RowSliceReady(bwWs_.ptrStagedReady, expertRowBase, expertIdx, L1_TILE_M),
                                   startBlockIdx, recomputeRows, borrow);
                if (timer_) timer_->SetWave(MegaMoeTrainTimerSchema::NO_WAVE);
                precedingTiles += plan.tiles;
                const uint32_t columns = (n_ / 2 + L1_TILE_N - 1U) / L1_TILE_N;
                startBlockIdx = (startBlockIdx + uint64_t(plan.tiles) * columns) % blockNum_;
            } else {
                GMM2GradSwiGLUGradGM(
                    gmm2Params, epilogueOp_, problemShape_, gmmAddrInfo,
                    bwWs_.ptrHp + expertRowBase * (n_ / 2) * sizeof(bfloat16_t), startBlockIdx, nextTargetAiv,
                    expertIdx, expertRowBase, dispatchFlag, dispatchTarget,
                    reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + expertIdx * INT_CACHELINE);
            }
            if constexpr (g_coreType == AscendC::AIV) {
                // Idle AIVs must also publish completion after dZ/Hp/partial stores retire.
                PipeBarrier<PIPE_ALL>();
                if (UseStagedWave(nonemptyOrdinal) && subBlockIdx_ == 1U) {
                    AtomicAdd(reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDZReady) + expertIdx * INT_CACHELINE + 1,
                              int32_t(1));
                }
                AtomicAdd(reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDZReady) + expertIdx * INT_CACHELINE,
                          int32_t(1));
            }
        } else {
            GMM2GradSwiGLUGradDualAiv(gmm2Params, epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx,
                                      vecSetSyncCom_, nextTargetAiv, expertIdx, expertRowBase, dispatchFlag,
                                      dispatchTarget, BW_P1_EVENT);
        }
        ++nonemptyOrdinal;
        cumMRows_ += Get<M_VALUE>(problemShape_);
    }
}

template <typename TopkWeightT>
template <bool TransW, bool ExpertReady>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM1GradPhase()
{
    if constexpr (ExpertReady) {
        GMM1GradReuseDOPhase<TransW>();
    } else {
        using namespace MegaMoeBF16BackwardCatlassImpl;
        using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
        Bf16GMMAddrInfo gmmAddrInfo{};
        uint32_t startBlockIdx = 0;
        vecSetSyncCom_ = 0;
        gmm2PingPongIdx_ = 0;
        BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, false> gmm1Params;
        cumMRows_ = 0;
        preOffset_ = 0;
        // P2 GEMM produces packed dX: canonical dZ @ W1 is
        // [Me,2F] x [2F,K], with GEMM N=k_ and K=n_. Transposed storage
        // uses W1.T; Combine sends the resulting rows to source-rank ptrD.
        Get<N_VALUE>(problemShape_) = k_;
        Get<K_VALUE>(problemShape_) = n_;
        for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
            if (!UpdateGroupParams(expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_GMM1GRAD, expertIdx);
            // mLoc is expert-relative; keep Combine's slice offset at zero.
            GMM1GradCombine<0>(gmm1Params, params_, problemShape_, gmmAddrInfo, startBlockIdx, vecSetSyncCom_,
                               expertIdx, 0, gmm2PingPongIdx_);
            cumMRows_ += Get<M_VALUE>(problemShape_);
        }
    }
}

// Saved-mode dO and dX share the same packed-row/K GM band. Complete
// dW2=dO.T@Hp and all AIC dO reads before P2 overwrites that band with dX;
// other experts occupy disjoint bands.
template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM1GradReuseDOPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
    uint32_t dxStartBlockIdx = 0;
    uint32_t dw2StartBlockIdx = 0;
    cumMRows_ = 0;
    preOffset_ = 0;
    Get<N_VALUE>(problemShape_) = k_;
    Get<K_VALUE>(problemShape_) = n_;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        auto* dzReady = reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDZReady) + expertIdx * INT_CACHELINE;
        auto* doReleased = reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDOReleased) + expertIdx * INT_CACHELINE;
        auto* dxReady = reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDXReady) + expertIdx * INT_CACHELINE;
        const int64_t rows = Get<M_VALUE>(problemShape_);
        if (timer_ != nullptr) {
            timer_->SetSlice(expertIdx, 0, rows);
            timer_->SetTaskContext(-1, 2U);
        }
        if constexpr (g_coreType == AscendC::AIC) {
            // Wait for all AIV dZ/Hp stores, which also retire the corresponding AIC dO reads.

            {
                MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_WAIT);
                while (ReadGmByPassDCache(dzReady) < 2 * static_cast<int32_t>(blockNum_)) {
                }
            }

            {
                // Drain each CATLASS handle before another reuses its L1/L0 storage.
                BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> dw2Params;
                auto dwShape = problemShape_;
                Get<M_VALUE>(dwShape) = TransW ? k_ : n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
                Get<N_VALUE>(dwShape) = TransW ? n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF : k_;
                Get<K_VALUE>(dwShape) = rows;
                Bf16GMMAddrInfo dwAddr{};
                UpdateGlobalBuffer(dwAddr, BW_PHASE_DW2, expertIdx);
                uint32_t skip = 0;
                if (borrowedDw2Tiles_ != 0) {
                    const uint32_t mm = Get<M_VALUE>(dwShape), nn = Get<N_VALUE>(dwShape);
                    BlockScheduler sched({mm, nn, static_cast<uint32_t>(rows)},
                                         Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
                    const uint32_t total = sched.GetCoreLoops();
                    const uint32_t offset = (dw2Params.blockIdx + blockNum_ - dw2StartBlockIdx) % blockNum_;
                    const uint32_t local = offset < total ? 1 + (total - 1 - offset) / blockNum_ : 0;
                    skip = borrowedDw2Tiles_ < local ? borrowedDw2Tiles_ : local;
                    borrowedDw2Tiles_ -= skip;
                }
                GMM2DW(dw2Params, dwShape, dwAddr, dw2StartBlockIdx, expertIdx, skip);
                PipeBarrier<PIPE_ALL>();  // retire this core's dO reads and dW2 writes
            }

            // Count idle AICs too; no core may overwrite dO before all dW2 readers retire.
            AtomicAdd(doReleased, int32_t(1));
            {
                if (timer_ != nullptr) {
                    timer_->SetSlice(expertIdx, 0, rows);
                }
                MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DO_REUSE_WAIT);
                while (ReadGmByPassDCache(doReleased) < static_cast<int32_t>(blockNum_)) {
                }
            }
        }
        {
            BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> dxParams;
            Bf16GMMAddrInfo dxAddr{};
            UpdateGlobalBuffer(dxAddr, BW_PHASE_GMM1GRAD, expertIdx);
            GMM1GradCombineGM<WeightLayout>(dxParams, params_, problemShape_, dxAddr,
                                            bwWs_.ptrDX + cumMRows_ * k_ * sizeof(bfloat16_t), dxStartBlockIdx,
                                            expertIdx, dxReady);
        }
        cumMRows_ += rows;
    }
}

template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM1DWPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    Bf16GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<ColumnMajor, RowMajor, 0, true> gmm1Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        // Save Me before remapping GEMM dimensions; address bookkeeping still uses token rows.
        const int64_t me = Get<M_VALUE>(problemShape_);
        if constexpr (TransW) {
            Get<M_VALUE>(problemShape_) = n_;
            Get<N_VALUE>(problemShape_) = k_;
        } else {
            Get<M_VALUE>(problemShape_) = k_;
            Get<N_VALUE>(problemShape_) = n_;
        }
        Get<K_VALUE>(problemShape_) = me;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_DW1, expertIdx);
        GMM1DW(gmm1Params, problemShape_, gmmAddrInfo, startBlockIdx, expertIdx);
        cumMRows_ += me;
    }
}

template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM2DWPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    Bf16GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> gmm2Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        const int64_t me = Get<M_VALUE>(problemShape_);
        if constexpr (TransW) {
            Get<M_VALUE>(problemShape_) = k_;
            Get<N_VALUE>(problemShape_) = n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
        } else {
            Get<M_VALUE>(problemShape_) = n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
            Get<N_VALUE>(problemShape_) = k_;
        }
        Get<K_VALUE>(problemShape_) = me;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_DW2, expertIdx);
        GMM2DW(gmm2Params, problemShape_, gmmAddrInfo, startBlockIdx, expertIdx);
        cumMRows_ += me;
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::UnpermuteBackward()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BW_UNPERMUTE);
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    MegaMoeImpl::TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdxGather;
    expandedRowIdxGather.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<QuantOutType> ptrD;
    ptrD.SetGlobalBuffer((__gm__ QuantOutType*)params_.peermemInfo.ptrD);
    GlobalTensor<QuantOutType> gradX;
    gradX.SetGlobalBuffer((__gm__ QuantOutType*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        // Retire the previous MTE3 before V overwrites its result buffer.
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        LocalTensor<QuantOutType> dataResBf16 = DataResBuf_.Get<QuantOutType>();
        LocalTensor<QuantOutType> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<QuantOutType> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = DataResFp32Buf_.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];

        // Accumulate in FP32 and round once; BF16 scatter_add may use a different rounding tree.
        for (int32_t expId = 0; expId < static_cast<int32_t>(topK_); ++expId) {
            int32_t rowIdx = expandedRowIdxGather(tokenIdx * static_cast<int32_t>(topK_) + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            MegaMoeBF16Routing::LoadUnpermuteInput(dataInBf16, ptrD, rowIdx, k_, expId % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            if (expId == 0) {
                // dX includes +0.0F in the first slot, including for a negative zero input.
                Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
                PipeBarrier<PIPE_V>();
                Adds(dataResFp32, dataInFp32, 0.0F, k_);
            } else {
                MegaMoeBF16Routing::AccumulateUnpermuteRoute(dataResFp32, dataInFp32, dataInBf16, k_, false);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(gradX[tokenIdx * static_cast<int64_t>(k_)], dataResBf16, k_);
    }
    // Retire the final GM write before the UB pipe is destroyed.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// Reduce each FP32 partial row into the routing-weight gradient dtype.
template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ReduceDpPartialBatch(uint16_t batchRows, uint16_t partialTiles,
                                                                              uint16_t partialStride)
{
    constexpr uint16_t fp32PerReg = 256U / sizeof(float);
    const uint16_t repeatsPerRow = static_cast<uint16_t>(
        Ops::Base::CeilDiv(static_cast<uint32_t>(partialTiles), static_cast<uint32_t>(fp32PerReg)));
    LocalTensor<float> partialBuf = TopkGradPartialBuf_.Get<float>();
    LocalTensor<TopkWeightT> routeBuf = TopkGradRouteBuf_.template Get<TopkWeightT>();
    __ubuf__ float* partialData = reinterpret_cast<__ubuf__ float*>(partialBuf.GetPhyAddr());
    __ubuf__ TopkWeightT* routeData = reinterpret_cast<__ubuf__ TopkWeightT*>(routeBuf.GetPhyAddr());

    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < batchRows; ++row) {
            AscendC::Reg::RegTensor<float> accumulated;
            AscendC::Reg::MaskReg oneLane = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            if (partialTiles == 1U) {
                AscendC::Reg::DataCopy(accumulated, partialData + static_cast<uint32_t>(row) * partialStride);
            } else {
                AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
                uint16_t remaining = partialTiles;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    const uint16_t lanes = remaining > fp32PerReg ? fp32PerReg : remaining;
                    uint32_t maskLanes = lanes;
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(maskLanes);
                    AscendC::Reg::RegTensor<float> values;
                    AscendC::Reg::RegTensor<float> rowSum;
                    const uint32_t offset =
                        static_cast<uint32_t>(row) * partialStride + static_cast<uint32_t>(repeat) * fp32PerReg;
                    AscendC::Reg::DataCopy(values, partialData + offset);
                    AscendC::Reg::ReduceSum(rowSum, values, mask);
                    AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
                    remaining -= lanes;
                }
            }
            if constexpr (std::is_same_v<TopkWeightT, bfloat16_t>) {
                AscendC::Reg::RegTensor<bfloat16_t> result;
                AscendC::Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(result, accumulated,
                                                                                                oneLane);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(routeData + row,
                                                                                                    result, oneLane);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(routeData + row,
                                                                                               accumulated, oneLane);
            }
        }
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::CopyTopkGradBatch(uint64_t packedRow, uint64_t remoteRow,
                                                                           uint32_t rank, uint32_t rows,
                                                                           uint32_t partialTiles,
                                                                           uint32_t partialStride, uint32_t expert,
                                                                           uint32_t expertRow)
{
    GlobalTensor<float> partials;
    partials.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bwWs_.ptrDpProducts));
    auto partialBuf = TopkGradPartialBuf_.template Get<float>();
    auto routeBuf = TopkGradRouteBuf_.template Get<TopkWeightT>();
    const DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(partialTiles * sizeof(float)), 0U,
                                 0U, 0U};
    DataCopyPad(partialBuf, partials[packedRow * partialTiles], copy, DataCopyPadExtParams<float>{false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_V>();
    ReduceDpPartialBatch(rows, partialTiles, partialStride);
    SyncFunc<HardEvent::V_MTE2>();
    SyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<TopkWeightT> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightT*>(GetRankWinAddrWithOffset(
        bf16WinRankAddr, rank, params_.peermemInfo.ptrTopkGrad - params_.peermemInfo.ptrBase)));
    MegaMoeTrainTimerPassiveScope send(timer_, MegaMoeTrainTimer::AIV_BW_DP_SEND, expert, expertRow, rows, uint32_t(-1),
                                       rows, 1U, partialTiles, 1U, 0U, rank);
    const DataCopyExtParams batchCopy{1U, static_cast<uint32_t>(rows * sizeof(TopkWeightT)), 0U, 0U, 0U};
    output.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyPad(output[remoteRow], routeBuf, batchCopy);
    if (rank != rankId_) {
        // The AIV exchanging barrier signals can differ from the one posting this DMA.
        // Acknowledge the bytes at the destination before this writer joins the barrier.
        using Bits = std::conditional_t<std::is_same_v<TopkWeightT, float>, uint32_t, uint16_t>;
        auto received = partialBuf.template ReinterpretCast<TopkWeightT>();
        auto receivedBits = received.template ReinterpretCast<Bits>();
        auto expectedBits = routeBuf.template ReinterpretCast<Bits>();
        SyncFunc<HardEvent::MTE3_MTE2>();
        bool arrived;
        do {
            DataCopyPad(received, output[remoteRow], batchCopy, DataCopyPadExtParams<TopkWeightT>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_S>();
            arrived = true;
            for (uint32_t i = 0; i < rows; ++i) {
                if (receivedBits.GetValue(i) != expectedBits.GetValue(i)) {
                    arrived = false;
                    break;
                }
            }
            SyncFunc<HardEvent::S_MTE2>();
        } while (!arrived);
    }
    SyncFunc<HardEvent::MTE3_V>();
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ReduceAndRouteTopkGrad()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BW_DP_REDUCE);
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    const int32_t f = static_cast<int32_t>(n_ / SWIGLU_N_HALF);
    const int32_t partialTiles = static_cast<int32_t>(MegaMoeBF16Impl::BackwardTopkPartialCount(savedGateUpMode_, f));
    const int32_t partialCapacity = static_cast<int32_t>(BW_TOPK_PARTIAL_UB_BYTES / sizeof(float));
    // Masked arithmetic does not mask Reg::DataCopy's full-register load.
    // Reserve one register minus one element as a UB tail guard.
    constexpr int32_t fp32PerReg = 256U / sizeof(float);
    const int32_t partialStride = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint32_t>(partialTiles), static_cast<uint32_t>(ALIGN_32 / sizeof(float))));
    if (partialTiles <= 0 || partialStride > partialCapacity - (fp32PerReg - 1)) {
        return;
    }
    const int32_t rowsByPartialUb = (partialCapacity - (fp32PerReg - 1)) / partialStride;
    const int32_t batchCapacity = rowsByPartialUb < static_cast<int32_t>(BW_TOPK_ROUTE_BATCH_ROWS)
                                      ? rowsByPartialUb
                                      : static_cast<int32_t>(BW_TOPK_ROUTE_BATCH_ROWS);
    if (params_.peerRowsGroups != 0U) {
        using namespace MegaMoeBF16Impl::Wave;
        auto prefixUb = TopkGradPeerPrefixBuf_.Get<int32_t>();
        const uint32_t prefixBytes = worldSize_ * expertPerRank_ * sizeof(int32_t);
        DataCopyPad(prefixUb, cumsumInWorkSpace_, DataCopyExtParams{1U, prefixBytes, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        const UbPeerPrefix prefix{prefixUb};
        constexpr uint32_t rowsPerBlock = ALIGN_32 / sizeof(TopkWeightT);
        for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
            const auto info =
                DescribePeerExpert(prefix, worldSize_, expertPerRank_, expert, maxOutputSize_, params_.peerRowsGroups);
            if (info.rows == 0U) continue;
            const PeerCounts<UbPeerPrefix> counts{prefix, expertPerRank_, expert, info.rows};
            // Visit each physical fragment once. Ownership stays in 32-byte blocks of source order.
            PeerRowCursor cursor(counts, worldSize_, rankId_, params_.peerRowsGroups, info.rows, info.firstSliceRows,
                                 0U, info.rows);
            PeerTransfer transfer{};
            while (cursor.Next(transfer)) {
                const uint32_t source = transfer.sourceRank;
                const uint32_t rows = counts.Get(source);
                const uint32_t blocks = (rows + rowsPerBlock - 1U) / rowsPerBlock;
                const uint32_t quotient = blocks / blockAivNum_, extra = blocks % blockAivNum_;
                const uint32_t ownBegin =
                    (aivCoreIdx_ * quotient + (aivCoreIdx_ < extra ? aivCoreIdx_ : extra)) * rowsPerBlock;
                const uint32_t limit = ownBegin + (quotient + (aivCoreIdx_ < extra ? 1U : 0U)) * rowsPerBlock;
                const uint32_t ownEnd = limit < rows ? limit : rows;
                const uint32_t ordinalEnd = transfer.sourceOrdinalBegin + transfer.rowCount;
                const uint32_t begin = ownBegin > transfer.sourceOrdinalBegin ? ownBegin : transfer.sourceOrdinalBegin;
                const uint32_t end = ownEnd < ordinalEnd ? ownEnd : ordinalEnd;
                if (begin >= end) continue;
                const uint32_t remoteBase = preSumBeforeRankWorkSpace_(source * expertPerRank_ + expert);
                for (uint32_t row = begin; row < end; row += batchCapacity) {
                    const uint32_t take = end - row < static_cast<uint32_t>(batchCapacity) ? end - row : batchCapacity;
                    const uint32_t physicalRow = transfer.destinationRow + row - transfer.sourceOrdinalBegin;
                    CopyTopkGradBatch(info.base + physicalRow, remoteBase + row, source, take, partialTiles,
                                      partialStride, expert, physicalRow);
                }
            }
        }
        return;
    }
    int64_t previousGroupEnd = 0;
    int64_t expertBase = 0;
    for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
        const int64_t groupEnd = static_cast<int32_t>(groupList_.GetValue(expert));
        int64_t expertRows = groupListType_ == 0 ? groupEnd - previousGroupEnd : groupEnd;
        previousGroupEnd = groupListType_ == 0 ? groupEnd : previousGroupEnd;
        if (expertBase + expertRows > static_cast<int32_t>(maxOutputSize_)) {
            expertRows = static_cast<int64_t>(maxOutputSize_) - expertBase;
        }
        if (expertRows <= 0) {
            continue;
        }
        int64_t rankRowBegin = 0;
        for (uint32_t dstRank = 0; dstRank < worldSize_; ++dstRank) {
            int64_t rankRows = tokenPerExpertWin_(TokenPerExpertLayout(dstRank, rankId_, expert));
            if (rankRowBegin + rankRows > expertRows) {
                rankRows = expertRows - rankRowBegin;
            }
            if (rankRows <= 0) {
                continue;
            }
            // Assign whole 32-byte blocks to AIVs so adjacent MTE3 writes cannot overlap.
            constexpr int32_t rowsPerBlock = ALIGN_32 / static_cast<int32_t>(sizeof(TopkWeightT));
            const int32_t ownershipBlocks = static_cast<int32_t>(
                Ops::Base::CeilDiv(static_cast<uint32_t>(rankRows), static_cast<uint32_t>(rowsPerBlock)));
            const int32_t blocksPerCore = ownershipBlocks / static_cast<int32_t>(blockAivNum_);
            const int32_t extraBlocks = ownershipBlocks % static_cast<int32_t>(blockAivNum_);
            const int32_t coreBlocks = blocksPerCore + (static_cast<int32_t>(aivCoreIdx_) < extraBlocks ? 1 : 0);
            const int64_t coreBlockBegin =
                static_cast<int64_t>(aivCoreIdx_) * blocksPerCore +
                (static_cast<int32_t>(aivCoreIdx_) < extraBlocks ? static_cast<int32_t>(aivCoreIdx_) : extraBlocks);
            const int64_t coreRowBegin = coreBlockBegin * rowsPerBlock;
            int64_t coreRows = static_cast<int64_t>(coreBlocks) * rowsPerBlock;
            if (coreRowBegin + coreRows > rankRows) {
                coreRows = rankRows - coreRowBegin;
            }
            if (coreRows > 0) {
                const int64_t dstExpertOffset =
                    static_cast<int64_t>(preSumBeforeRankWorkSpace_(dstRank * expertPerRank_ + expert));
                for (int64_t batchBegin = 0; batchBegin < coreRows; batchBegin += batchCapacity) {
                    const int32_t batchRows = coreRows - batchBegin > batchCapacity
                                                  ? batchCapacity
                                                  : static_cast<int32_t>(coreRows - batchBegin);
                    const int64_t packedRow = expertBase + rankRowBegin + coreRowBegin + batchBegin;
                    CopyTopkGradBatch(packedRow, dstExpertOffset + coreRowBegin + batchBegin, dstRank, batchRows,
                                      partialTiles, partialStride, expert,
                                      static_cast<uint32_t>(packedRow - expertBase));
                }
            }
            rankRowBegin += rankRows;
        }
        expertBase += expertRows;
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::UnpermuteTopkGrad()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BW_TOPK_UNPERMUTE);
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    MegaMoeImpl::TilingByCore(static_cast<int32_t>(m_ * topK_), coreLen, coreOffset, 1);
    __gm__ int32_t* routeToPacked = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.expandedRowIdxGather);
    using GradientBits = std::conditional_t<std::is_same_v<TopkWeightT, float>, uint32_t, uint16_t>;
    __gm__ GradientBits* routedDp = reinterpret_cast<__gm__ GradientBits*>(params_.peermemInfo.ptrTopkGrad);
    __gm__ GradientBits* gradTopk = reinterpret_cast<__gm__ GradientBits*>(gradTopkWeightsGmAddr_);
    for (int32_t route = coreOffset; route < coreOffset + coreLen; ++route) {
        const GradientBits dpBits = ReadGmByPassDCache(routedDp + routeToPacked[route]);
        // Each slot has one owner. Preserve its bits and bypass cross-AIV DCache writeback.
        WriteGmByPassDCache(gradTopk + route, dpBits);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::Process()
{
    // P0 staging (gather slots / count-table restore / dispatch ring) is a
    // fixed static UB window above the P1/P2 tile region (StageUb), so P0
    // needs no TPipe and no ownership handover into P1.  P5/P6 allocate
    // their reduction buffers in local pipes.
    InitStageLayout();
    // The Host-side token-table clear is ordered before this kernel on the
    // same stream. The entry device synchronization makes that local
    // ordering visible to every peer without a Host stream synchronize or
    // aclshmem_barrier_all.
    CrossRankSyncInWorldSize();

    // ---- P0-A: publish source-token dY and this microbatch's route table (all AIV) ----
    if constexpr (g_coreType == AscendC::AIV) {
        PrepareWaveChunks();
        {
            MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BW_GATHER);
            auto scratch = StageUb(MegaMoeImpl::kCopyBufferBytes).template ReinterpretCast<uint8_t>();
            Bf16Route::PublishDispatchInput(params_.aGmAddr, params_.peermemInfo.sourceTokens,
                                            params_.workspaceInfo.expandedRowIdx, params_.peermemInfo.routeIndices, m_,
                                            k_, topK_, aivCoreIdx_, blockAivNum_, scratch);
        }

        SyncAll<true>();
        CrossRankSyncInWorldSize();

        RestoreTokenPerExpert();
        SyncAll<true>();

        if (subBlockIdx_ == 0) {
            // AIV0s pull peers' raw route dY into packed dO for all experts.
            // Whole-expert Dispatch signals the expert; Wave Dispatch signals
            // each canonical block. P1 AICs wait for
            // the selected readiness before reading dO, allowing earlier
            // experts' work to overlap later experts' dispatch.
            cumMRows_ = 0;
            preOffset_ = 0;
            uint32_t nonemptyOrdinal = 0;
            uint32_t precedingTiles = 0;
            for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
                const int64_t expertRowBase = cumMRows_;
                if (!UpdateGroupParams(expertIdx)) {
                    continue;
                }
                GM_ADDR dst = bwWs_.ptrDO + expertRowBase * k_ * sizeof(bfloat16_t);
                if (UseStagedWave(nonemptyOrdinal)) {
                    const auto plan = MakeWavePlan(expertIdx, Get<M_VALUE>(problemShape_), precedingTiles);
                    DispatchStagedBackward(dst, expertIdx, plan);
                    precedingTiles += plan.tiles;
                } else {
                    DispatchTokensBackward(dst, expertIdx);
                }
                ++nonemptyOrdinal;
                cumMRows_ += Get<M_VALUE>(problemShape_);
            }
        }
    }

    // ---- P0-C: save=none GMM1 recompute ----
    // BF16/MXFP8 modes consume the matching forward-persistent gate/up in P1.
    // Only save=none recomputes GMM1 into backward GM scratch. The AIC does
    // not hand these results through AIV UB, so it can overlap P0-A/P0-B.
    if (savedGateUpMode_ == MegaMoeBF16Impl::kSaveNone) {
        if (params_.tilingData->transW) {
            GMM1RecomputePhase<true>();
        } else {
            GMM1RecomputePhase<false>();
        }
    }

    // ---- P1: GMM2 dHp + SwiGLU backward epilogue ----
    // GM output: AICs wait for a Wave block or a complete expert before reading dO.
    // AIVs wait for the dHp rows they consume when slice readiness is enabled;
    // otherwise they wait for all expert arrivals. Each owner replaces dHp
    // with Hp after its dP/Hp/dZ computation has consumed the original value.
    // The save=none GM path also waits for the corresponding recompute rows.
    // UB output: AIC alternates dHp tiles between paired AIV slots, with a
    // completion handshake before reuse. Gate/Up comes from P0-C scratch.
    // P0 staging is disjoint from both work paths.
    const bool useGmDataGrad = MegaMoeBF16Impl::UseGmForBackwardDataGrad(savedGateUpMode_);
    // For recompute, only P1 needs GM dHp so one AIV can reduce a full row.
    // The later dX/Combine and dW phases keep their existing ownership.
    if (useGmDataGrad || MegaMoeBF16Impl::UseWholeRowTopkGrad(savedGateUpMode_, n_ / 2)) {
        if (params_.tilingData->transW) {
            GMM2GradPhase<true, true>();
        } else {
            GMM2GradPhase<false, true>();
        }
    } else {
        if (params_.tilingData->transW) {
            GMM2GradPhase<true, false>();
        } else {
            GMM2GradPhase<false, false>();
        }
    }
    if (!useGmDataGrad) {
        const int32_t p1OutstandingMask = vecSetSyncCom_;
        if constexpr (g_coreType == AscendC::AIC) {
            if ((p1OutstandingMask & 1) != 0) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(BW_P1_EVENT);
                }  // AIV0 wrote its last dZ tile
            }
            if ((p1OutstandingMask & 2) != 0) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(BW_P1_EVENT + FLAG_ID_MAX);
                }  // AIV1 wrote the last dZ tile
            }
            // Pair-local events cover only local tiles; P2/P3 require every AIV's dZ stores.
            const int32_t p1ReadyTarget = 2 * static_cast<int32_t>(blockNum_);
            {
                MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_SYNC_WAIT);
                while (ReadGmByPassDCache(bwFlagAiv1Ready_) < p1ReadyTarget) {
                    const int64_t start = GetSystemCycle();
                    while (GetSystemCycle() - start < 100) {
                    }
                }
            }
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        // Every AIV retires its dZ/Hp/dpPartials MTE3 writes before these
        // arrivals. save=none AICs consume bwFlagAiv1Ready_ at the P1 exit;
        // saved modes instead use per-expert dzReady before dW2/P2.
        PipeBarrier<PIPE_ALL>();
        AtomicAdd((__gm__ int32_t*)bwFlagAiv1Ready_, int32_t(1));
        AtomicAdd((__gm__ int32_t*)bwFlagHDone_, int32_t(1));
    }

    // ---- P2: packed dX GEMM + Combine (AIC + AIV0) ----
    // Saved modes wait for each expert's dzReady, compute dW2, and retire
    // all AIC dO readers before overwriting dO with GM dX. AIV0 waits for
    // that expert's dxReady before Combine. save=none uses AIV0 UB output
    // after the full P1 barrier; each AIV enters after its own P1 loop.
    if (useGmDataGrad) {
        if (params_.tilingData->transW) {
            GMM1GradPhase<true, true>();
        } else {
            GMM1GradPhase<false, true>();
        }
    } else {
        if (params_.tilingData->transW) {
            GMM1GradPhase<true, false>();
        } else {
            GMM1GradPhase<false, false>();
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (!useGmDataGrad) {
            // Tail sync: ensure AIV0 consumed every dX tile before P3/P4.

            if (vecSetSyncCom_ == 1) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(0);
                }
            } else if (vecSetSyncCom_ >= 2) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(gmm2PingPongIdx_);
                }
                {
                    MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(1 - gmm2PingPongIdx_);
                }
            }
        }
    }

    // ---- P3 dW1 / save=none P4 dW2 (AIC), overlapped with P5 (AIV) ----
    // Saved-mode dW2 has already run per expert immediately before P2 dX.
    if constexpr (g_coreType == AscendC::AIC) {
        if (params_.tilingData->transW) {
            GMM1DWPhase<true>();
        } else {
            GMM1DWPhase<false>();
        }

        {
            MegaMoeTrainTimerWaitScope timerWait(timer_, MegaMoeTrainTimer::AIC_SYNC_WAIT);
            while (2 * static_cast<int32_t>(blockNum_) != ReadGmByPassDCache(bwFlagHDone_)) {
                const int64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < 100) {
                }
            }
        }

        if (!useGmDataGrad) {
            if (params_.tilingData->transW) {
                GMM2DWPhase<true>();
            } else {
                GMM2DWPhase<false>();
            }
        }
    }

    if constexpr (g_coreType == AscendC::AIV) {
        // AICs write non-empty experts' weight gradients. Use all paired
        // AIVs for the empty-expert zero fill so no single core clears a
        // complete weight plane and the writes overlap the AIC GEMMs.
        ClearEmptyExpertGradWeights();
        // Every AIV0 may still be sending dX tiles into a peer's ptrD while
        // its paired AIV1 has finished its P1 epilogue and empty-expert clear.
        // Do not publish cross-rank completion until all local AIV pipelines
        // are drained and every local AIV has reached this point; otherwise a peer can start
        // dX unpermute from a partially-written ptrD.

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        CrossRankSyncInWorldSize();

        // P5: unpermute dX.
        TPipe unpermutePipe;
        UnpermuteBuffInit(&unpermutePipe);
        UnpermuteBackward();
        PipeBarrier<PIPE_ALL>();
        unpermutePipe.Destroy();
        SyncAll<true>();

        // Finish the dX phase on all ranks before starting the dP exchange.
        CrossRankSyncInWorldSize();

        TPipe topkGradPipe;
        TopkGradBuffInit(&topkGradPipe);

        ReduceAndRouteTopkGrad();
        PipeBarrier<PIPE_ALL>();
        topkGradPipe.Destroy();
        SyncAll<true>();

        // Finish remote dP writes before the source ranks restore TopK order.

        CrossRankSyncInWorldSize();

        UnpermuteTopkGrad();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();

        // The exit barrier protects the next forward from this rank's count-table reset.

        ResetTokenPerExpert();
        SyncAll<true>();
        CrossRankSyncInWorldSize();
    }
}

}  // namespace MegaMoeImpl
#endif
