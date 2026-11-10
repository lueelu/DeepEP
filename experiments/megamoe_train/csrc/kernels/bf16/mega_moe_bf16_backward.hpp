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
 *   P0-A  route gather           raw grad_y -> peermem.ptrA0 (AIV), without p
 *   P0-B  restore count table + Dispatch dO from peers -> ptrDO (AIV0)
 *   P0-C  save=none recompute    Z = Xpacked @ W1.T -> gate/up GM planes (AIC)
 *   P1    GMM2 Grad + epilogue   dHp = dO @ W2; h, Hp, dp partials, dh -> dZ
 *   P2    GMM1 Grad + Combine    packed dX = dZ @ W1 -> peers' ptrD
 *   P3    GMM1 dW                dW1 = dZ.T @ Xpacked -> grad_weight1 (AIC)
 *   P4    save=none GMM2 dW      dW2 = dO.T @ Hp -> grad_weight2 (AIC)
 *   P5    Unpermute              FP32 sum_slot(ptrD) -> BF16 grad_x (AIV)
 *   P6    Reduce/route/unpermute  P1 dp partials -> grad_topk (AIV)
 *
 * P1 saved modes (BF16/MXFP8) write dHp to GM; each AIV waits for all AICs
 * to finish that expert before loading its assigned tiles and replacing
 * dHp with Hp in the same plane. The AIC reads dO using per-producer row
 * progress for eligible multi-M experts, or the complete-expert dispatch
 * flag otherwise. In save=none, P1 waits for complete-expert dispatch and
 * alternates dHp Fixpipe tiles between the paired AIV UB slots, with a
 * per-tile completion handshake. Both paths reduce FP32(dHp)*h over F for dp.
 * Every dispatch band publishes bwFlagDispatch_[expert], including those
 * using row progress. P0 staging occupies a fixed UB window above the
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
#include "op_kernel/mc2_kernel_utils.h"
#include "layout/mega_moe_context.hpp"
#include "routing/mega_moe_combine_send.hpp"
#include "epilogue/block_epilogue_swiglu_backward_bf16.hpp"
#include "catlass/mega_moe_bf16_backward_catlass_impl.hpp"

using namespace AscendC;

namespace MegaMoeImpl {

using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// In save=none, P1 alternates between the paired AIVs: AIC event 0 targets
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
                                MegaMoeTilingData* tilingData, uint32_t splitExperts);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitStageLayout();
    __aicore__ inline AscendC::LocalTensor<QuantOutType> StageUb(uint32_t bytes);
    __aicore__ inline void UnpermuteBuffInit(TPipe* tPipe);
    __aicore__ inline void TopkGradBuffInit(TPipe* tPipe);
    __aicore__ inline void GatherRoutes();
    __aicore__ inline void RestoreTokenPerExpert();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void ClearEmptyExpertGradWeights();
    __aicore__ inline void ClearGradWeightPlane(GM_ADDR base, uint64_t planeElements, uint32_t expertIdx);
    __aicore__ inline void DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx, uint32_t ordinal);
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst, GlobalTensor<QuantOutType> src,
                                                int32_t rows, int32_t hiddenSize, int32_t& pingpongId);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo& gmmAddrInfo, uint32_t phase, uint32_t expertIdx);
    __aicore__ inline void UnpermuteBackward();
    __aicore__ inline void ReduceDpPartialBatch(uint16_t batchRows, uint16_t partialTiles, uint16_t partialStride);
    __aicore__ inline void ReduceAndRouteTopkGrad();
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

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t& coreLen, int32_t& coreOffset,
                                        int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2;
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

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t* sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    // P1/P2 own three work tiles at UB offset zero. P0 reuses the disjoint
    // staging region above them, sequentially on each AIV.
    static constexpr uint32_t BW_TILE_BYTES =
        MegaMoeBF16BackwardCatlassImpl::L1_TILE_M * MegaMoeBF16BackwardCatlassImpl::L1_TILE_N * sizeof(bfloat16_t);
    static constexpr uint32_t BW_STAGE_BASE_BYTES = 3 * BW_TILE_BYTES;
    static constexpr uint32_t BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    static_assert(BW_STAGE_BASE_BYTES + BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES <=
                      MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE,
                  "the static staging window must hold the gradient-zero tile");

    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> TopkGradRouteBuf_;
    TBuf<> TopkGradPartialBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
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
    uint32_t savedGateUpMode_ = MegaMoeBF16Training::kSaveNone;
    uint64_t maxOutputSize_ = 0;
    int32_t dispatchRows_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 0;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    int32_t dispatchCopyElementsPerPing_ = 0;
    uint32_t dispatchPingRows_ = 0;
    __aicore__ inline bool UseDispatchRowReady(uint32_t ordinal) const
    {
        return MegaMoeBF16Training::UseBackwardDispatchRowReady(savedGateUpMode_, ordinal, splitExperts_,
                                                                static_cast<uint64_t>(Get<M_VALUE>(problemShape_)),
                                                                MegaMoeBF16BackwardCatlassImpl::L1_TILE_M);
    }
    __aicore__ inline MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView MakeDispatchRowReady(
        uint32_t expert, uint32_t ordinal) const
    {
        MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView v;
        v.progressBase = reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwDispatchProgress);
        v.cumsum = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM);
        v.sources = worldSize_;
        v.producersPerSource = dispatchBlockNumPerEP_;
        v.experts = expertPerRank_;
        v.expert = expert;
        v.progressSlot = ordinal;
        v.rowLimit = static_cast<uint32_t>(maxOutputSize_);
        v.enabled = UseDispatchRowReady(ordinal);
        return v;
    }

    int32_t vecSetSyncCom_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
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
                                                              MegaMoeTilingData* tilingData, uint32_t splitExperts)
{
#if DEEPEP_BF16_FORCE_EXPERT_READY
    // Keep the device split policy consistent with Host allocation.
    (void)splitExperts;
    splitExperts_ = 0;
#else
    splitExperts_ = splitExperts;
#endif
    m_ = tilingData->m;
    k_ = tilingData->k;
    n_ = tilingData->n;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    rankId_ = tilingData->rankId;
    savedGateUpMode_ =
        MegaMoeBF16Training::ResolveSavedGateUpMode(tilingData->savedGateUpMode, tilingData->swigluLimit);

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    params_.aGmAddr = gradY;
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.probsGmAddr = topkWeights;
    params_.y2GmAddr = gradX;
    savedState_ = BackwardSavedStateInfo(trainingSavedState, dispatchOutput, tilingData);
    params_.workspaceInfo = savedState_;
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, true);
    params_.tilingData = tilingData;
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
        savedGateUpMode_ == MegaMoeBF16Training::kSaveNone ? bwWs_.ptrRecomputeGate : savedState_.ptrSavedGate;
    GM_ADDR upPlane =
        savedGateUpMode_ == MegaMoeBF16Training::kSaveNone ? bwWs_.ptrRecomputeUp : savedState_.ptrSavedUp;
    epilogueOp_.Init({gatePlane, upPlane, savedState_.ptrSavedGateScale, savedState_.ptrSavedUpScale,
                      savedState_.ptrSavedGateHigh, savedState_.ptrSavedUpLow, savedState_.ptrSavedUpHigh,
                      bwWs_.ptrGradZ, bwWs_.ptrHp, bwWs_.ptrDpProducts, savedState_.ptrDispatchedTopkWeights, 0, 0,
                      tilingData->swigluLimit, savedGateUpMode_});
    epilogueOp_.UpdateNextProblem({0, n_, 0, 0});
}

// Fixed-position P0 staging window at the top of UB, above the P1/P2 tile
// region [0, BW_STAGE_BASE_BYTES).  P0-A gather slots, the count-table
// restore tile and the P0-B dispatch ring all bind here; they are
// sequential in time on one lane, so one shared region serves all three.
// Nothing in P0 goes through TPipe, so the staging region never overlaps
// the CATLASS-managed tile area. In save=none, P1 can Fixpipe an early
// expert's dHp tile into either paired AIV while P0 dispatch copies later
// experts. The tile stays in the statically bound work region until that
// lane's epilogue consumes it; CATLASS Resource TBufs that overlap the
// region are allocation-only on the vector side (never written or zeroed).
// The dO read dependency is separate: saved-mode AICs use row progress or
// complete-expert dispatch readiness, while save=none uses the latter.
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
    // Allow for both ping slots and their alignment padding within the staging window.
    const uint32_t stageBytes = MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE - BW_STAGE_BASE_BYTES;
    const uint32_t maxPingRows = stageBytes > 256 ? (stageBytes - 256) / 2 / (k_ * sizeof(bfloat16_t)) : 0;
    dispatchPingRows_ = static_cast<uint32_t>(dispatchRows_) < maxPingRows ? static_cast<uint32_t>(dispatchRows_)
                                                                           : (maxPingRows > 0 ? maxPingRows : 1);
    if (MegaMoeBF16Training::UseGmForBackwardDataGrad(savedGateUpMode_)) {
        dispatchPingRows_ = 1;  // Saved modes use single-row DMA, including splitExperts_=0.
    }
    dispatchCopyElementsPerPing_ = Ops::Base::CeilAlign(dispatchPingRows_ * k_, static_cast<uint32_t>(ALIGN_32));
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
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GatherRoutes()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(static_cast<int32_t>(m_) * static_cast<int32_t>(topK_), coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdx);
    GlobalTensor<QuantOutType> gradY;
    gradY.SetGlobalBuffer((__gm__ QuantOutType*)params_.aGmAddr);
    GlobalTensor<QuantOutType> ptrA0;
    ptrA0.SetGlobalBuffer((__gm__ QuantOutType*)params_.peermemInfo.ptrA0);

    // Both gather paths copy raw dY in route order. The forward scales h
    // before W2, so P1 forms dH=FP32(dHp)*FP32(p), not weighted dO here.
    // Preserve the fixed P0 window above all three P1/P2 work tiles.
    // Do not borrow their space: P1 may already fixpipe while P0-B runs.
    const uint32_t capacity = MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE - BW_STAGE_BASE_BYTES;
    const uint32_t rowBytes = k_ * sizeof(bfloat16_t);

    const uint32_t batchRows = 8U * rowBytes <= capacity ? 4U : (4U * rowBytes <= capacity ? 2U : 1U);
    if (batchRows > 1U && batchRows * rowBytes <= 65504U) {
        const uint32_t slotElements = batchRows * k_;
        auto arena = StageUb(2U * batchRows * rowBytes);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        uint32_t batch = 0;
        for (int32_t first = 0; first < coreLen; first += batchRows, ++batch) {
            const uint32_t rows =
                static_cast<uint32_t>(coreLen - first) < batchRows ? static_cast<uint32_t>(coreLen - first) : batchRows;
            int64_t source[4];
            for (uint32_t i = 0; i < rows; ++i) {
                source[i] =
                    static_cast<int64_t>(expandedRowIdx(coreOffset + first + i) / static_cast<int32_t>(topK_)) * k_;
            }
            const auto event = batch % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
            auto slot = arena[(batch % 2U) * slotElements];
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(slot[i * k_], gradY[source[i]], k_);
            }
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            DataCopy(ptrA0[(static_cast<int64_t>(coreOffset) + first) * k_], slot, rows * k_);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        return;
    }

    const uint32_t gatherBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(2 * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<QuantOutType> dataIn0Bf16 = StageUb(gatherBytes);
    LocalTensor<QuantOutType> dataIn1Bf16 = dataIn0Bf16[k_];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t route = coreOffset; route < coreOffset + coreLen; route++) {
        int32_t token = expandedRowIdx(route) / static_cast<int32_t>(topK_);
        auto event = (route % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (route % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(dataInBf16, gradY[token * static_cast<int64_t>(k_)], k_);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        DataCopy(ptrA0[route * static_cast<int64_t>(k_)], dataInBf16, k_);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::RestoreTokenPerExpert()
{
    const bool prepareMetadata = MegaMoeBF16Training::UseGmForBackwardDataGrad(savedGateUpMode_) && splitExperts_ > 0;
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
        if (prepareMetadata) {
            MegaMoeBF16BackwardCatlassImpl::PreparedExpertCursor cursor;
            for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
                uint32_t rows = 0, ordinal = 0;
                if (!cursor.Next(static_cast<int32_t>(groupList_.GetValue(expert)), groupListType_, maxOutputSize_,
                                 rows, ordinal)) {
                    continue;
                }
                if (!MegaMoeBF16Training::UseBackwardDispatchRowReady(savedGateUpMode_, ordinal, splitExperts_, rows,
                                                                      MegaMoeBF16BackwardCatlassImpl::L1_TILE_M)) {
                    continue;
                }
                MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView view;
                view.progressBase = reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwDispatchProgress);
                view.cumsum = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM);
                view.sources = worldSize_;
                view.producersPerSource = dispatchBlockNumPerEP_;
                view.experts = expertPerRank_;
                view.expert = expert;
                view.progressSlot = ordinal;
                view.rowLimit = static_cast<uint32_t>(maxOutputSize_);
                view.enabled = true;
                if (MegaMoeBF16BackwardCatlassImpl::ProducerPhaseSupported(view, blockNum_)) {
                    MegaMoeBF16BackwardCatlassImpl::PrepareProducerPhaseSource(
                        view, blockNum_, rows, static_cast<uint32_t>(sourceRank), nullptr);
                }
            }
        }
    }
    if (prepareMetadata) {
        // Retire scalar metadata before the AIV barrier; event 7 avoids ring events 0/1.
        SetFlag<AscendC::HardEvent::S_MTE2>(7);
        WaitFlag<AscendC::HardEvent::S_MTE2>(7);
        SetFlag<AscendC::HardEvent::MTE2_S>(7);
        WaitFlag<AscendC::HardEvent::MTE2_S>(7);
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ResetTokenPerExpert()
{
    // Restored counts must be cleared before the next forward's atomic count exchange.
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    const uint64_t countElements =
        static_cast<uint64_t>(worldSize_) * static_cast<uint32_t>(perRankStridesInTokenPerExpert_);
    constexpr uint32_t chunkElements =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes / sizeof(int32_t);
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
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                                                                              GlobalTensor<QuantOutType> src,
                                                                              int32_t rows, int32_t hiddenSize,
                                                                              int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    const uint32_t ringBytes = 2 * dispatchCopyElementsPerPing_ * sizeof(QuantOutType);
    LocalTensor<QuantOutType> tmpBuffer1 = StageUb(ringBytes);
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[dispatchCopyElementsPerPing_];
    auto processCount = Ops::Base::CeilDiv(rows, static_cast<int32_t>(dispatchPingRows_));
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        int32_t rowNum = static_cast<int32_t>(dispatchPingRows_);
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * static_cast<int32_t>(dispatchPingRows_);
        }
        const int64_t elementOffset = static_cast<int64_t>(processIndex) * dispatchPingRows_ * hiddenSize;
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
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx,
                                                                                uint32_t ordinal)
{
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
                GlobalTensor<QuantOutType> gmRemoteA, aGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * widthA;

                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                    GetRankWinAddrWithOffset(dstRankIdx, params_.peermemInfo.payloadOffset)));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstBase));
                if (UseDispatchRowReady(ordinal)) {
                    const auto ready = MakeDispatchRowReady(expertIdx, ordinal);
                    constexpr int32_t chunkRows = 256;
                    for (int32_t done = 0; done < rowsPerCore; done += chunkRows) {
                        const int32_t count = rowsPerCore - done < chunkRows ? rowsPerCore - done : chunkRows;
                        CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA + static_cast<int64_t>(done) * widthA],
                                             gmRemoteA[gmOffsetPeer + static_cast<int64_t>(done) * widthA], count,
                                             widthA, pingpongId);
                        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                        AtomicAdd(ready.Slot(idx), count);
                    }
                } else {
                    CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], gmRemoteA[gmOffsetPeer], rowsPerCore, widthA,
                                         pingpongId);
                }
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
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::UpdateGlobalBuffer(GMMAddrInfo& gmmAddrInfo, uint32_t phase,
                                                                            uint32_t expertIdx)
{
    const int64_t base = cumMRows_;
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
        if (params_.tilingData->transW) {
            gmmAddrInfo.aGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
            gmmAddrInfo.bGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
        } else {
            gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
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
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount =
        (__gm__ int32_t*)(params_.peermemInfo.ptrBase + MegaMoeBF16Training::PeerSyncCountOffset(worldSize_));
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < static_cast<int>(worldSize_); i += static_cast<int>(blockAivNum_)) {
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
template <bool TransW>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM1RecomputePhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, ColumnMajor, RowMajor>;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> gmm1Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    // P0-C uses GEMM {m=Me, n=2F, k=K}: Xpacked @ W1.T for canonical
    // weights, or Xpacked @ W1 for transposed storage. Me is set per expert
    // by UpdateGroupParams; K=k_ and 2F=n_ are fixed here.
    Get<N_VALUE>(problemShape_) = n_;
    Get<K_VALUE>(problemShape_) = k_;
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
        cumMRows_ += Get<M_VALUE>(problemShape_);
    }
}

template <typename TopkWeightT>
template <bool TransW, bool GmOutput>
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::GMM2GradPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    vecSetSyncCom_ = 0;
    uint32_t nextTargetAiv = 0;
    const int32_t dispatchTarget = static_cast<int32_t>(worldSize_ * dispatchBlockNumPerEP_);
    // GmOutput selects GM dHp output for saved modes. Otherwise the dual
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
    uint32_t nonemptyOrdinal = 0;
    bool metadataKnown = false;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        const int64_t expertRowBase = cumMRows_;
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_GMM2GRAD, expertIdx);
        __gm__ int32_t* dispatchFlag = bwFlagDispatch_ + expertIdx * INT_CACHELINE;
        if constexpr (GmOutput) {
            const auto rowReady = MakeDispatchRowReady(expertIdx, nonemptyOrdinal);
            if constexpr (g_coreType == AscendC::AIC) {
                if (!metadataKnown && ProducerPhaseSupported(rowReady, blockNum_)) {
                    // Wait once for restored count-table/producer metadata.
                    // This flag does not signal completion of dO payload DMA.
                    while (ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + 1) != 1) {
                    }
                    metadataKnown = true;
                }
            }
            GMM2GradSwiGLUGradGM(
                gmm2Params, epilogueOp_, problemShape_, gmmAddrInfo,
                bwWs_.ptrHp + expertRowBase * (n_ / 2) * sizeof(bfloat16_t), startBlockIdx, nextTargetAiv, expertIdx,
                expertRowBase, dispatchFlag, dispatchTarget, rowReady,
                reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + expertIdx * INT_CACHELINE);
            if constexpr (g_coreType == AscendC::AIV) {
                // Idle AIVs must also publish completion after dZ/Hp/partial stores retire.
                PipeBarrier<PIPE_ALL>();
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
        GMMAddrInfo gmmAddrInfo{};
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
        if constexpr (g_coreType == AscendC::AIC) {
            // Wait for all AIV dZ/Hp stores, which also retire the corresponding AIC dO reads.

            while (ReadGmByPassDCache(dzReady) < 2 * static_cast<int32_t>(blockNum_)) {
            }

            {
                // Drain each CATLASS handle before another reuses its L1/L0 storage.
                BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> dw2Params;
                auto dwShape = problemShape_;
                Get<M_VALUE>(dwShape) = TransW ? k_ : n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
                Get<N_VALUE>(dwShape) = TransW ? n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF : k_;
                Get<K_VALUE>(dwShape) = rows;
                GMMAddrInfo dwAddr{};
                UpdateGlobalBuffer(dwAddr, BW_PHASE_DW2, expertIdx);
                GMM2DW(dw2Params, dwShape, dwAddr, dw2StartBlockIdx, expertIdx);
                PipeBarrier<PIPE_ALL>();  // retire this core's dO reads and dW2 writes
            }

            // Count idle AICs too; no core may overwrite dO before all dW2 readers retire.
            AtomicAdd(doReleased, int32_t(1));
            while (ReadGmByPassDCache(doReleased) < static_cast<int32_t>(blockNum_)) {
            }
        }
        {
            BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> dxParams;
            GMMAddrInfo dxAddr{};
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
    GMMAddrInfo gmmAddrInfo{};
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
    GMMAddrInfo gmmAddrInfo{};
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
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
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
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, ptrD[rowIdx * static_cast<int64_t>(k_)], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Adds(dataResFp32, dataInFp32, 0.0F, k_);
            } else {
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
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
__aicore__ inline void MegaMoeBF16Backward<TopkWeightT>::ReduceAndRouteTopkGrad()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    const int32_t f = static_cast<int32_t>(n_ / SWIGLU_N_HALF);
    const int32_t partialTiles = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(f), static_cast<uint64_t>(BACKWARD_DP_PARTIAL_TILE_N)));
    AscendC::GlobalTensor<float> partials;
    partials.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bwWs_.ptrDpProducts));
    const uint64_t remoteDOffset = params_.peermemInfo.ptrD - params_.peermemInfo.ptrBase;
    LocalTensor<TopkWeightT> routeBuf = TopkGradRouteBuf_.template Get<TopkWeightT>();
    LocalTensor<float> partialBuf = TopkGradPartialBuf_.Get<float>();
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
                AscendC::GlobalTensor<TopkWeightT> remoteD;
                remoteD.SetGlobalBuffer(
                    reinterpret_cast<__gm__ TopkWeightT*>(GetRankWinAddrWithOffset(dstRank, remoteDOffset)));
                for (int64_t batchBegin = 0; batchBegin < coreRows; batchBegin += batchCapacity) {
                    const int32_t batchRows = coreRows - batchBegin > batchCapacity
                                                  ? batchCapacity
                                                  : static_cast<int32_t>(coreRows - batchBegin);
                    const int64_t packedRow = expertBase + rankRowBegin + coreRowBegin + batchBegin;
                    AscendC::DataCopyExtParams partialCopyParams{static_cast<uint16_t>(batchRows),
                                                                 static_cast<uint32_t>(partialTiles * sizeof(float)),
                                                                 0U, 0U, 0U};
                    AscendC::DataCopyPadExtParams<float> partialPadParams{false, 0U, 0U, 0U};
                    AscendC::DataCopyPad(partialBuf, partials[static_cast<int64_t>(packedRow) * partialTiles],
                                         partialCopyParams, partialPadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    ReduceDpPartialBatch(static_cast<uint16_t>(batchRows), static_cast<uint16_t>(partialTiles),
                                         static_cast<uint16_t>(partialStride));
                    SyncFunc<AscendC::HardEvent::V_MTE2>();
                    SyncFunc<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchRows * sizeof(TopkWeightT)),
                                                          0U, 0U, 0U};
                    AscendC::DataCopyPad(remoteD[dstExpertOffset + coreRowBegin + batchBegin], routeBuf, copyParams);

                    // Retire MTE3 before the next batch refills routeBuf.
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
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
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(static_cast<int32_t>(m_ * topK_), coreLen, coreOffset, 1);
    __gm__ int32_t* routeToPacked = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.expandedRowIdxGather);
    using GradientBits = std::conditional_t<std::is_same_v<TopkWeightT, float>, uint32_t, uint16_t>;
    __gm__ GradientBits* routedDp = reinterpret_cast<__gm__ GradientBits*>(params_.peermemInfo.ptrD);
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

    // ---- P0-A: gather raw grad_y into route order -> ptrA0 (all AIV) ----
    if constexpr (g_coreType == AscendC::AIV) {
        GatherRoutes();

        SyncAll<true>();
        CrossRankSyncInWorldSize();

        RestoreTokenPerExpert();
        SyncAll<true>();
        if (MegaMoeBF16Training::UseGmForBackwardDataGrad(savedGateUpMode_) && splitExperts_ > 0 && aivCoreIdx_ == 0) {
            // Word1 of expert0's dHp flag cacheline signals that every AIV
            // has finished count-table restore and producer metadata setup.
            AtomicAdd(reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + 1, int32_t(1));
        }

        if (subBlockIdx_ == 0) {
            // AIV0s pull peers' raw route dY into packed dO for all experts.
            // Every producer band publishes bwFlagDispatch_[expert]; eligible
            // saved-mode experts also publish row progress. P1 AICs wait for
            // the selected readiness before reading dO, allowing earlier
            // experts' work to overlap later experts' dispatch.
            cumMRows_ = 0;
            preOffset_ = 0;
            uint32_t nonemptyOrdinal = 0;
            for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
                const int64_t expertRowBase = cumMRows_;
                if (!UpdateGroupParams(expertIdx)) {
                    continue;
                }
                DispatchTokensBackward(bwWs_.ptrDO + expertRowBase * k_ * sizeof(bfloat16_t), expertIdx,
                                       nonemptyOrdinal++);
                cumMRows_ += Get<M_VALUE>(problemShape_);
            }
        }
    }

    // ---- P0-C: save=none GMM1 recompute ----
    // BF16/MXFP8 modes consume the matching forward-persistent gate/up in P1.
    // Only save=none recomputes GMM1 into backward GM scratch. The AIC does
    // not hand these results through AIV UB, so it can overlap P0-A/P0-B.
    if (savedGateUpMode_ == MegaMoeBF16Training::kSaveNone) {
        if (params_.tilingData->transW) {
            GMM1RecomputePhase<true>();
        } else {
            GMM1RecomputePhase<false>();
        }
    }

    // ---- P1: GMM2 dHp + SwiGLU backward epilogue ----
    // Saved BF16/MXFP8: AICs read dO only after the selected producer-row
    // or complete-expert readiness, then write dHp to ptrHp in GM. AIVs
    // wait for all AIC dHp arrivals for that expert, load assigned tiles
    // into UB, and replace dHp with Hp while writing dZ and dp partials.
    // save=none: AICs wait for complete-expert dO dispatch and alternately
    // Fixpipe dHp tiles into paired AIV UB slots. Each slot has a per-tile
    // completion handshake before reuse; gate/up comes from P0-C scratch.
    // P0 staging is disjoint from these work tiles, so neither path needs
    // a whole-stage UB ownership handover after dispatch.
    const bool useGmDataGrad = MegaMoeBF16Training::UseGmForBackwardDataGrad(savedGateUpMode_);
    if (useGmDataGrad) {
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
                WaitForVector(BW_P1_EVENT);  // AIV0 wrote its last dZ tile
            }
            if ((p1OutstandingMask & 2) != 0) {
                WaitForVector(BW_P1_EVENT + FLAG_ID_MAX);  // AIV1 wrote the last dZ tile
            }
            // Pair-local events cover only local tiles; P2/P3 require every AIV's dZ stores.
            const int32_t p1ReadyTarget = 2 * static_cast<int32_t>(blockNum_);
            while (ReadGmByPassDCache(bwFlagAiv1Ready_) < p1ReadyTarget) {
                const int64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < 100) {
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
                WaitForVector(0);
            } else if (vecSetSyncCom_ >= 2) {
                WaitForVector(gmm2PingPongIdx_);
                WaitForVector(1 - gmm2PingPongIdx_);
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

        while (2 * static_cast<int32_t>(blockNum_) != ReadGmByPassDCache(bwFlagHDone_)) {
            const int64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 100) {
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

        // All ranks must finish reading dX before ptrD is reused for dp.

        CrossRankSyncInWorldSize();

        TPipe topkGradPipe;
        TopkGradBuffInit(&topkGradPipe);

        ReduceAndRouteTopkGrad();
        PipeBarrier<PIPE_ALL>();
        topkGradPipe.Destroy();
        SyncAll<true>();

        // Finish remote dp writes before any rank reads ptrD.

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
