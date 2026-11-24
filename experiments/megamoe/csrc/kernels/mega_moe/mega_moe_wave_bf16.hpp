// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*!
 * \file mega_moe_wave_bf16.hpp
 * \brief Fused BF16 MoE inference and training forward dispatch, expert computation and combine.
 */

#ifndef MEGA_MOE_WAVE_BF16_HPP
#define MEGA_MOE_WAVE_BF16_HPP

#include "kernel_operator.h"
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_utils.hpp"
#include "stage/mega_moe_combine_send.hpp"
#include "tla/layout.hpp"
#include "epilogue/block_epilogue_swiglu_bf16.hpp"
#include "catlass_gmm/mega_moe_bf16_catlass_impl.hpp"
#include "common/gmm2_aiv_completion.hpp"
#include "csrc/kernels/common/mega_moe_wave_math.hpp"
#include "common/mega_moe_bf16_peer_row_view.hpp"
#include "stage/mega_moe_route_bf16.hpp"
#include "stage/mega_moe_topk_weight_gather.hpp"
#include "stage/mega_moe_unpermute_bf16.hpp"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

constexpr uint32_t kBf16OverflowModeBit = 60;
constexpr uint32_t TOPK_WEIGHT_COPY_BUFFER_BYTES = MegaMoeBF16Impl::ForwardDispatchUbLayout::kTopkWeightBytes;

template <typename TopkWeightT = bfloat16_t>
class MegaMoeBF16Forward {
public:
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    using TopkWeightBitsT = std::conditional_t<sizeof(TopkWeightT) == sizeof(uint16_t), uint16_t, uint32_t>;
    static constexpr uint32_t TOPK_WEIGHT_COPY_ELEMENTS = TOPK_WEIGHT_COPY_BUFFER_BYTES / sizeof(TopkWeightT);
    using QuantScaleOutType = float;
    __aicore__ inline MegaMoeBF16Forward(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR permutedInputOut,
                                GM_ADDR actualRecvTokensOut, GM_ADDR savedMetaOut, GM_ADDR workspaceGM,
                                MegaMoeBf16TilingData* tilingData, GM_ADDR transientWorkspaceGM = nullptr,
                                GM_ADDR dispatchOutputGM = nullptr, bool allowCompactTokenCountExchange = true);
    __aicore__ inline void Process();
    __aicore__ inline void SetTimer(MegaMoeTrainTimerDevice* timer)
    {
        timer_ = timer;
        params_.timer = timer;
    }
    template <uint32_t SaveMode, bool TransW>
    __aicore__ inline void ProcessWithSaveMode();
    __aicore__ inline void InitForwardWorkspace(GM_ADDR transientWorkspaceGM);

private:
    using WaveRange = MegaMoeBF16Impl::Wave::Range;
    using WaveSlice = MegaMoeBF16Impl::Wave::Slice;
    struct WaveCompletionState {
        MegaMoeBF16Impl::Wave::Cursor cursor{};
        uint32_t groups = 0, startBlock = 0;
        WaveSlice slice{};
        int32_t completed = 0;
        bool pending = false;
    };
    struct WaveCounts {
        const uint32_t* values;
        __aicore__ inline uint64_t operator()(uint32_t expert) const
        {
            return values[expert];
        }
    };
    __aicore__ inline WaveRange NextWaveRange(const MegaMoeBF16Impl::Wave::Cursor& begin) const;
    __aicore__ inline void DispatchWave(const WaveRange& range);
    __aicore__ inline void CopyWavePayload(uint32_t source, uint32_t sourceRow, GM_ADDR waveX, uint32_t localRow,
                                           uint64_t packedRow, uint64_t savedRow, uint32_t rows);
    __aicore__ inline void SaveDispatchWeights();
    template <bool TransW>
    __aicore__ inline void Gmm1Wave(const WaveRange& range, MegaMoeBF16ForwardCatlassImpl::WaveMatmul<TransW>& matmul,
                                    uint32_t& startBlock, uint32_t& sequence);
    template <uint32_t SaveMode, class Epilogue>
    __aicore__ inline void SwigluWave(const WaveRange& range, Epilogue& epilogue, uint32_t& startBlock,
                                      uint32_t& sequence);
    template <bool TransW>
    __aicore__ inline void Gmm2Wave(const WaveRange& range, MegaMoeBF16ForwardCatlassImpl::WaveMatmul<TransW>& matmul,
                                    uint32_t& startBlock, int32_t& issued);
    __aicore__ inline bool NextWaveSlice(MegaMoeBF16Impl::Wave::Cursor& cursor, uint32_t& groups, WaveSlice& slice);
    __aicore__ inline bool NextDownCompletion(MegaMoeBF16Impl::Wave::Cursor& cursor, uint32_t& groups,
                                              uint32_t& startBlock, WaveSlice& slice);
    __aicore__ inline bool PollDownCompletion();
    __aicore__ inline void CombineSlice(const WaveSlice& slice, uint64_t scratchRow);
    template <uint32_t SaveMode, bool TransW>
    __aicore__ inline void ProcessWave(TPipe& firstPipe);
    __aicore__ inline void FinishForward(int64_t oriOverflowMode);
    __aicore__ inline void PrepareDispatchInputs(TPipe& pipe);
    __aicore__ inline void FirstBuffInit(TPipe* tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void CrossRankSyncCntPerExpertCompact();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void SaveForwardState();
    __aicore__ inline void GatherRoutedTopkWeights();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void CrossRankSyncInWorldSize();

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

    TBuf<> CrossRankSyncBuf_;
    TBuf<> PrevSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> TopkWeightCopyBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> SaveStateBuf_;

    __gm__ Mc2Aclnn::Bf16MoeContext* mc2Context_{nullptr};
    Bf16Params params_{};
    MegaMoeTrainTimerDevice* timer_ = nullptr;
    TupleShape problemShape_{};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
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
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();

    uint32_t dispatchCopyElementsPerPing_ = 0;
    bool useSplitTrainingState_ = false;
    uint32_t waveGroups_ = 1;
    uint32_t waveCount_ = 0;
    MegaMoeBF16Impl::Wave::Cursor waveFirst_{};
    uint32_t waveExpertRows_[MegaMoeBF16Impl::kMaxLocalExperts]{};
    GM_ADDR waveWeights_{nullptr};
    GM_ADDR waveReadyBase_{nullptr};
    __gm__ int32_t* waveDispatchReady_{nullptr};
    __gm__ int32_t* waveDownReady_{nullptr};
    __gm__ int32_t* waveIssued_{nullptr};
    MegaMoeBF16Impl::Wave::ForwardScratch scratch_{0, 0, 0};
    __gm__ int32_t* waveReleased_{nullptr};
    __gm__ int32_t* stagedInputDone_{nullptr};
    __aicore__ inline bool OutputSlotAvailable(uint32_t wave) const
    {
        using namespace MegaMoeBF16Impl::Wave;
        return !scratch_.slotRows || wave < MegaMoeImpl::kWaveScratchSlots ||
               ReadGmByPassDCache(waveReleased_ + (wave % MegaMoeImpl::kWaveScratchSlots) * INT_CACHELINE) >=
                   static_cast<int32_t>((wave / MegaMoeImpl::kWaveScratchSlots) * blockNum_);
    }
    __aicore__ inline bool InputsConsumed(const WaveRange& range) const
    {
        using namespace MegaMoeBF16Impl::Wave;
        const WaveCounts counts{waveExpertRows_};
        auto cursor = range.begin;
        auto groups = waveGroups_;
        WaveSlice slice{};
        const auto columns = (params_.tilingData->n / 2U + kPairedColumns - 1U) / kPairedColumns;
        auto* flags = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_);
        while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
            const auto first = ReadySlot(slice.packed - slice.row, slice.expert, slice.row);
            for (uint32_t m = 0; m < slice.rows; m += kGroupRows)
                if (ReadGmByPassDCache(flags + first + m / kGroupRows) < static_cast<int32_t>(columns)) return false;
        }
        return true;
    }
    GM_ADDR waveOutputBase_{nullptr};
    GM_ADDR waveOutputInY_{nullptr};
    uint64_t outputRowsInY_{0};
    // GMM1 reads the Wave's X here. Once the complete Hp row is ready, GMM2 can overwrite X.
    static_assert(MegaMoeBF16Impl::Wave::kGmm1Rows == MegaMoeBF16Impl::Wave::kGmm2Rows);
    __aicore__ inline GM_ADDR SliceOutput(uint64_t packed) const
    {
        if (packed < outputRowsInY_) return waveOutputInY_ + packed * k_ * sizeof(bfloat16_t);
        return waveOutputBase_ + (packed - outputRowsInY_) * k_ * sizeof(bfloat16_t);
    }
    WaveCompletionState waveCompletion_{};
    uint32_t waveCombineSequence_ = 0;
    // Dispatch and Combine can visit different experts in the same forward.
    MegaMoeImpl::PeerRowLimitsCache dispatchPeerLimits_{};
    MegaMoeImpl::PeerRowLimitsCache combinePeerLimits_{};
    MegaMoeImpl::PeerRowLimitsCache savePeerLimits_{};
    LocalTensor<int32_t> waveCumsum_;
    LocalTensor<int32_t> waveSourcePrefix_;
};

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR permutedInputOut, GM_ADDR actualRecvTokensOut, GM_ADDR savedMetaOut,
    GM_ADDR workspaceGM, MegaMoeBf16TilingData* tilingData, GM_ADDR transientWorkspaceGM, GM_ADDR dispatchOutputGM,
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
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::Bf16MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        bf16WinRankAddr[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
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
    if (useSplitTrainingState_) {
        params_.workspaceInfo = ForwardWorkspaceInfo(workspaceGM, dispatchOutputGM, transientWorkspaceGM, tilingData);
    } else {
        params_.workspaceInfo = ForwardWorkspaceInfo(workspaceGM, tilingData);
    }
    params_.peermemInfo = Bf16PeermemInfo(bf16WinRankAddr[rankId_], tilingData, allowCompactTokenCountExchange);
    params_.tilingData = tilingData;
    params_.peerRowsGroups = tilingData->mGroupsPerWave;
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    perRankStridesInTokenPerExpert_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ =
        params_.peermemInfo.ptrTokenPerExpert + TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::InitForwardWorkspace(GM_ADDR workspace)
{
    using namespace MegaMoeBF16Impl;
    const auto groups = params_.tilingData->mGroupsPerWave;
    scratch_ = {maxOutputSize_, groups, params_.tilingData->forwardSchedule};
    const ForwardTransientWorkspaceLayout transient(params_.tilingData->n, expertPerRank_, scratch_.hpRows);
    const Wave::ReadyLayout ready(maxOutputSize_, expertPerRank_, scratch_.slotRows != 0);
    const Wave::ForwardWorkspaceLayout layout(transient.totalBytes, maxOutputSize_, expertPerRank_, k_,
                                              params_.tilingData->routeWeightBeforeGmm2 ? sizeof(TopkWeightT) : 0U, m_,
                                              groups, scratch_.rows);
    waveReadyBase_ = workspace + layout.readyOffset;
    waveDispatchReady_ = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_ + ready.dispatchOffset);
    waveDownReady_ = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_ + ready.downReadyOffset);
    waveIssued_ = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_ + ready.issuedOffset);
    stagedInputDone_ = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_ + ready.inputDoneOffset);
    waveReleased_ = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_ + ready.releaseOffset);
    waveOutputBase_ = waveReadyBase_ + ready.outputOffset;
    outputRowsInY_ = layout.rowsInY;
    const auto alignedY = (reinterpret_cast<uint64_t>(params_.y2GmAddr) + 511U) & ~uint64_t(511U);
    waveOutputInY_ = reinterpret_cast<GM_ADDR>(alignedY);
    waveWeights_ = params_.tilingData->routeWeightBeforeGmm2 ? workspace + layout.weightOffset : nullptr;
    params_.forwardRawGate = workspace + layout.rawGateOffset;
    params_.forwardRawUp = workspace + layout.rawUpOffset;
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::PrepareDispatchInputs(TPipe& pipe)
{
    bool needsFallback = false;
    if constexpr (g_coreType == AscendC::AIV) {
        if (params_.tilingData->routeWeightBeforeGmm2) {
            MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_GATHER_TOPK_WEIGHTS);
            needsFallback = !MegaMoeBF16Routing::TryGatherTopkWeights<TopkWeightT>(
                pipe, params_.probsGmAddr, params_.workspaceInfo.expandedRowIdx,
                params_.peermemInfo.ptrRoutedTopkWeights, static_cast<uint64_t>(m_) * topK_, aivCoreIdx_, blockAivNum_,
                true);
        }
    }
    FirstBuffInit(&pipe);
    if (needsFallback) {
        GatherRoutedTopkWeights();
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::FirstBuffInit(TPipe* tPipe)
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_BUFFER_INIT);
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    const MegaMoeBF16Impl::ForwardDispatchUbLayout layout(worldSize_, expertPerRank_, k_, dispatchRows_);
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
    if constexpr (g_coreType == AIV) {
        const MegaMoeBF16Impl::Wave::ReadyLayout ready(maxOutputSize_, expertPerRank_, scratch_.slotRows != 0);
        auto* words = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_);
        for (uint64_t word = aivCoreIdx_; word < ready.outputOffset / sizeof(int32_t); word += blockAivNum_)
            WriteGmByPassDCache(words + word, 0);
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
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_SAVE_STATE);
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint64_t perRankStride =
        Ops::Base::CeilAlign(static_cast<uint64_t>(worldSize_) * expertPerRank_,
                             static_cast<uint64_t>(MegaMoeBF16Impl::kTokenCountStrideAlignment));
    const uint64_t countsBytes = worldSize_ * perRankStride * sizeof(int32_t);
    auto copyBuffer = SaveStateBuf_.Get<uint8_t>();

    if (useSplitTrainingState_) {
        // Snapshot counts before a later microbatch can overwrite the SHMEM table.
        MegaMoeImpl::CopyContiguous(params_.workspaceInfo.ptrSavedTokenPerExpert, params_.peermemInfo.ptrTokenPerExpert,
                                    countsBytes, blockIdx_, blockNum_, copyBuffer);
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

    MegaMoeImpl::CopyContiguous(params_.permutedInputOutGmAddr, params_.workspaceInfo.ptrA,
                                actualRows * k_ * sizeof(bfloat16_t), blockIdx_, blockNum_, copyBuffer);

    const uint64_t cumsumOffset = countsBytes;
    const uint64_t cumsumBytes =
        Ops::Base::CeilAlign(static_cast<uint64_t>(worldSize_) * expertPerRank_ * sizeof(int32_t),
                             static_cast<uint64_t>(MegaMoeBF16Impl::kMetaAlignment));
    const uint64_t rowIdxGatherOffset = cumsumOffset + cumsumBytes;
    const uint64_t rowIdxGatherBytes =
        Ops::Base::CeilAlign(static_cast<uint64_t>(m_), static_cast<uint64_t>(MegaMoeBF16Impl::kRoutingRowAlignment)) *
        topK_ * sizeof(int32_t);
    const uint64_t sumBeforeRankOffset = rowIdxGatherOffset + rowIdxGatherBytes;
    MegaMoeImpl::CopyContiguous(params_.savedMetaOutGmAddr, params_.peermemInfo.ptrTokenPerExpert, countsBytes,
                                blockIdx_, blockNum_, copyBuffer);
    MegaMoeImpl::CopyContiguous(params_.savedMetaOutGmAddr + cumsumOffset, params_.workspaceInfo.ptrcumsumMM,
                                cumsumBytes, blockIdx_, blockNum_, copyBuffer);
    MegaMoeImpl::CopyContiguous(params_.savedMetaOutGmAddr + rowIdxGatherOffset,
                                params_.workspaceInfo.expandedRowIdxGather, rowIdxGatherBytes, blockIdx_, blockNum_,
                                copyBuffer);
    MegaMoeImpl::CopyContiguous(params_.savedMetaOutGmAddr + sumBeforeRankOffset,
                                params_.workspaceInfo.ptrSumBeforeRank, perRankStride * sizeof(int32_t), blockIdx_,
                                blockNum_, copyBuffer);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CrossRankSyncCntPerExpert()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_COUNT_EXCHANGE);
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint32_t compactDestinationsPerCore = Ops::Base::CeilDiv(worldSize_, blockAivNum_);
    const uint32_t compactPacketOffset =
        Ops::Base::CeilAlign(2U * compactDestinationsPerCore * expertPerRank_, static_cast<uint32_t>(8));
    const uint32_t compactWorkElements =
        compactPacketOffset + static_cast<uint32_t>(MegaMoeBF16Impl::kCompactTokenCountPacketBytes / sizeof(int32_t));
    if (params_.peermemInfo.compactTokenCountExchange &&
        blockAivNum_ <= MegaMoeBF16Impl::kCompactTokenCountMaxAivCores &&
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
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(bf16WinRankAddr, dstRankIdx, offsetInDstWinAddr);
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
    using namespace MegaMoeBF16Impl;
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
            bf16WinRankAddr, destination,
            MegaMoeBF16Impl::kPeerDataOffset + static_cast<uint64_t>(rankId_) * kCompactTokenCountPacketBytes)));
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
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_GATHER_TOPK_WEIGHTS);
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
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ResetTokenPerExpert()
{
    MegaMoeTrainTimerFixedScope timerReset(timer_, MegaMoeTrainTimer::AIV_RESET);
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    const uint64_t countElements =
        static_cast<uint64_t>(worldSize_) *
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    constexpr uint32_t chunkElements =
        MegaMoeBF16Impl::ForwardDispatchUbLayout::kCountResetChunkBytes / sizeof(int32_t);
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
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Unpermute()
{
    MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_UNPERMUTE);
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    MegaMoeImpl::TilingByCore(m_, coreLen, coreOffset, 1);
    if (MegaMoeBF16Routing::TryUnpermute<TopkWeightT>(params_.workspaceInfo.expandedRowIdxGather,
                                                      params_.peermemInfo.ptrD, params_.y2GmAddr, k_, topK_, coreOffset,
                                                      coreLen, Catlass::Arch::Ascend950::UB_SIZE, params_.probsGmAddr,
                                                      params_.tilingData->routeWeightBeforeGmm2 == 0U)) {
        return;
    }
    MegaMoeBF16Routing::UnpermuteByHiddenChunk<TopkWeightT>(
        params_.workspaceInfo.expandedRowIdxGather, params_.peermemInfo.ptrD, params_.y2GmAddr, k_, topK_, coreOffset,
        coreLen, params_.probsGmAddr, params_.tilingData->routeWeightBeforeGmm2 == 0U);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CrossRankSyncInWorldSize()
{
    MegaMoeTrainTimerWaitScope timerWait(g_coreType == AscendC::AIV ? timer_ : nullptr,
                                         MegaMoeTrainTimer::AIV_SYNC_WAIT);
    Bf16CrossRankSyncInWorldSize(bf16WinRankAddr, params_.peermemInfo.ptrBase, MegaMoeBF16Impl::kPeerSyncCountOffset,
                                 rankId_, worldSize_, {aivCoreIdx_, blockAivNum_});
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Process()
{
    using namespace MegaMoeBF16Impl;
    const bool transW = params_.tilingData->transW;
    switch (params_.tilingData->savedGateUpMode) {
        case kSaveNone:
            if (transW) {
                ProcessWithSaveMode<kSaveNone, true>();
            } else {
                ProcessWithSaveMode<kSaveNone, false>();
            }
            break;
        case kSaveBf16:
            if (transW) {
                ProcessWithSaveMode<kSaveBf16, true>();
            } else {
                ProcessWithSaveMode<kSaveBf16, false>();
            }
            break;
        default:
            if (transW) {
                ProcessWithSaveMode<kSaveMxFp8, true>();
            } else {
                ProcessWithSaveMode<kSaveMxFp8, false>();
            }
            break;
    }
}

template <typename TopkWeightT>
template <uint32_t SaveMode, bool TransW>
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
    int64_t oriOverflowMode = GetCtrlSpr<kBf16OverflowModeBit, kBf16OverflowModeBit>();
    // Publish one BF16 row per source token and the route-to-token lookup.
    {
        MegaMoeTrainTimerFixedScope timerStage(timer_, MegaMoeTrainTimer::AIV_INIT_ROUTING);
        if constexpr (g_coreType == AscendC::AIV) {
            InitBf16Routes(params_.expertIdxGmAddr, params_.aGmAddr, params_.peermemInfo.sourceTokens,
                           params_.peermemInfo.routeIndices, tokenPerExpertCurRankAddr_,
                           params_.workspaceInfo.expandedRowIdx, params_.workspaceInfo.expandedRowIdxGather, m_ * topK_,
                           topK_, k_, worldSize_ * expertPerRank_, aivCoreIdx_, blockAivNum_);
        }
    }

    TPipe firstPipe;
    PrepareDispatchInputs(firstPipe);
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
        if (subBlockIdx_ == 0 && useSplitTrainingState_ && params_.tilingData->routeWeightBeforeGmm2) {
            TPipe savePipe;
            savePipe.InitBuffer(SaveStateBuf_, MegaMoeImpl::kCopyBufferBytes);
            SaveForwardState();
            PipeBarrier<PIPE_ALL>();
            savePipe.Destroy();
        }
    }
    SyncAll<false>();
    ProcessWave<SaveMode, TransW>(firstPipe);
    FinishForward(oriOverflowMode);
}

template <typename TopkWeightT>
__aicore__ inline typename MegaMoeBF16Forward<TopkWeightT>::WaveRange MegaMoeBF16Forward<TopkWeightT>::NextWaveRange(
    const MegaMoeBF16Impl::Wave::Cursor& begin) const
{
    return MegaMoeBF16Impl::Wave::PlanScheduled(begin, waveGroups_, expertPerRank_, maxOutputSize_, waveCount_,
                                                WaveCounts{waveExpertRows_});
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CopyWavePayload(uint32_t source, uint32_t sourceRow,
                                                                        GM_ADDR waveX, uint32_t localRow,
                                                                        uint64_t packedRow, uint64_t savedRow,
                                                                        uint32_t rows)
{
    using namespace MegaMoeBF16Impl::Wave;
    const uint32_t stride = (k_ + 31U) / 32U * 32U;
    const uint32_t rowsPerPing = kDispatchRowsPerSlot;
    const uint32_t copyElements = rowsPerPing * stride;
    const uint32_t dispatchBase = DispatchUbBase(k_);
    LocalTensor<bfloat16_t> x0{TPosition::VECIN, dispatchBase, copyElements};
    LocalTensor<bfloat16_t> x1{TPosition::VECIN, dispatchBase + copyElements * 2U, copyElements};
    GlobalTensor<bfloat16_t> sourceX, targetX, savedX;
    sourceX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        GetRankWinAddrWithOffset(bf16WinRankAddr, source, params_.peermemInfo.payloadOffset)));
    GlobalTensor<int32_t> routeIds;
    routeIds.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(GetRankWinAddrWithOffset(
            bf16WinRankAddr, source, params_.peermemInfo.routeIndices - params_.peermemInfo.ptrBase)) +
        sourceRow);
    // The weight scratch is free until the last indexed X read has been issued.
    LocalTensor<int32_t> indices{TPosition::VECIN, dispatchBase + 4U * copyElements, Bf16Route::kIndexBatch};
    targetX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(waveX) + uint64_t(localRow) * k_);
    if (params_.workspaceInfo.ptrA) {
        savedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params_.workspaceInfo.ptrA) + savedRow * k_);
    }
    targetX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t ping = 0;
    uint32_t pendingRow = 0, pendingPing = 0;
    bool pending = false;
    Bf16Route::ForEachSourceToken(routeIds, indices, rows, topK_, [&](uint32_t row, uint32_t token) __aicore__ {
        const TEventID event = ping ? EVENT_ID1 : EVENT_ID0;
        auto buffer = ping ? x1 : x0;
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        DataCopy(buffer, sourceX[static_cast<uint64_t>(token) * k_], k_);
        SetFlag<HardEvent::MTE2_MTE3>(event);
        // Issue the next read before consuming the preceding chunk's read-complete event.
        if (pending) {
            const TEventID previousEvent = pendingPing ? EVENT_ID1 : EVENT_ID0;
            auto previousBuffer = pendingPing ? x1 : x0;
            WaitFlag<HardEvent::MTE2_MTE3>(previousEvent);
            DataCopy(targetX[static_cast<uint64_t>(pendingRow) * k_], previousBuffer, k_);
            if (params_.workspaceInfo.ptrA) {
                DataCopy(savedX[static_cast<uint64_t>(pendingRow) * k_], previousBuffer, k_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(previousEvent);
        }
        pendingRow = row;
        pending = true;
        pendingPing = ping;
        ping = 1U - ping;
    });
    if (pending) {
        const TEventID event = pendingPing ? EVENT_ID1 : EVENT_ID0;
        auto buffer = pendingPing ? x1 : x0;
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        DataCopy(targetX[static_cast<uint64_t>(pendingRow) * k_], buffer, k_);
        if (params_.workspaceInfo.ptrA) {
            DataCopy(savedX[static_cast<uint64_t>(pendingRow) * k_], buffer, k_);
        }
        SetFlag<HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    if (waveWeights_ == nullptr) return;
    constexpr uint32_t elements = kWeightUbBytes / sizeof(TopkWeightT);
    LocalTensor<TopkWeightT> weights{TPosition::VECIN, dispatchBase + 4U * copyElements, elements};
    GlobalTensor<TopkWeightT> sourceP, targetP;
    const uint64_t weightOffset = params_.peermemInfo.ptrRoutedTopkWeights - params_.peermemInfo.ptrBase;
    sourceP.SetGlobalBuffer(
        reinterpret_cast<__gm__ TopkWeightT*>(GetRankWinAddrWithOffset(bf16WinRankAddr, source, weightOffset)) +
        sourceRow);
    targetP.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightT*>(waveWeights_) + packedRow);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    for (uint32_t row = 0; row < rows; row += elements) {
        const uint32_t count = rows - row < elements ? rows - row : elements;
        const DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(TopkWeightT)), 0U, 0U, 0U};
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        DataCopyPad(weights, sourceP[row], copy, DataCopyPadExtParams<TopkWeightT>{false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
        DataCopyPad(targetP[row], weights, copy);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::DispatchWave(const WaveRange& range)
{
    if (timer_ != nullptr) timer_->SetWave(range.begin.wave);
    using namespace MegaMoeBF16Impl::Wave;
    const WaveCounts counts{waveExpertRows_};
    // Match inference: partition the complete Wave across all AIV1 workers,
    // then project each owned interval onto expert/source intersections.
    // A Wave is bounded by its group budget; packed addresses span all Waves.
    const uint32_t rows = static_cast<uint32_t>(range.end.packed - range.begin.packed);
    const auto owned = GetRotatedBalancedTokenRange(rows, blockIdx_, blockNum_, range.begin.packed);
    const uint64_t ownBegin = range.begin.packed + owned.start;
    const uint64_t ownEnd = ownBegin + owned.count;
    auto cursor = range.begin;
    uint32_t groups = waveGroups_;
    WaveSlice slice{};
    while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
        const uint64_t expertBase = slice.packed - slice.row;
        const uint64_t begin = ownBegin > slice.packed ? ownBegin : slice.packed;
        const uint64_t end = ownEnd < slice.packed + slice.rows ? ownEnd : slice.packed + slice.rows;
        if (begin < end) {
            MegaMoeTrainTimerPassiveScope dispatch(timer_, MegaMoeTrainTimer::AIV_DISPATCH, slice.expert,
                                                   static_cast<uint32_t>(begin - expertBase), end - begin);
            const PeerRowLayout<UbPeerPrefix> layout{
                {waveCumsum_}, {waveSourcePrefix_}, worldSize_, expertPerRank_, rankId_};
            layout.ForEachSpan(
                slice, begin, end, dispatchPeerLimits_, [&](const PeerTransfer& t, uint64_t packed) __aicore__ {
                    CopyWavePayload(t.sourceRank, layout.SourceRow(slice, t),
                                    SliceOutput(scratch_.Row(range.begin.wave, range.begin.packed, slice.packed)),
                                    packed - slice.packed, packed,
                                    params_.workspaceInfo.ptrA ? layout.SavedRow(slice, t) : 0, t.rowCount);
                });
        }
        // All workers retire each slice, including empty intersections.
        // Publish only after both X and p writes have completed.
        SyncFunc<HardEvent::MTE3_S>();
        AtomicAdd(waveDispatchReady_ + ReadySlot(expertBase, slice.expert, slice.row), int32_t(1));
    }
}

template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Gmm1Wave(
    const WaveRange& range, MegaMoeBF16ForwardCatlassImpl::WaveMatmul<TransW>& matmul, uint32_t& startBlock,
    uint32_t& sequence)
{
    if (timer_ != nullptr) timer_->SetWave(range.begin.wave);
    using namespace MegaMoeBF16Impl::Wave;
    const WaveCounts counts{waveExpertRows_};
    auto cursor = range.begin;
    uint32_t groups = waveGroups_;
    WaveSlice slice{};
    while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
        auto* ready = waveDispatchReady_ + ReadySlot(slice.packed - slice.row, slice.expert, slice.row);
        matmul.GateUp(params_, slice, SliceOutput(scratch_.Row(range.begin.wave, range.begin.packed, slice.packed)),
                      ready, startBlock, sequence);
    }
}

template <typename TopkWeightT>
template <uint32_t SaveMode, class Epilogue>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::SwigluWave(const WaveRange& range, Epilogue& epilogue,
                                                                   uint32_t& startBlock, uint32_t& sequence)
{
    if (timer_ != nullptr) timer_->SetWave(range.begin.wave);
    using namespace MegaMoeBF16Impl::Wave;
    using namespace MegaMoeBF16ForwardCatlassImpl;
    const uint32_t f = params_.tilingData->n / 2U;
    const WaveCounts counts{waveExpertRows_};
    auto cursor = range.begin;
    uint32_t groups = waveGroups_;
    WaveSlice slice{};
    while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
        const auto scratchRow = scratch_.Row(range.begin.wave, range.begin.packed, slice.packed);
        const auto hpRow = scratch_.HpRow(scratchRow, slice.packed);
        epilogue.UpdateGlobalAddr({static_cast<int64_t>(hpRow * f), 0, slice.expert, static_cast<int64_t>(slice.packed),
                                   0,
                                   static_cast<int64_t>(ReadySlot(slice.packed - slice.row, slice.expert, slice.row))});
        ForEachGateUpTile<WaveBlockScheduler>(
            slice.rows, f, k_, blockIdx_, blockNum_, startBlock,
            [&](uint32_t m, uint32_t n, const auto& shape) __aicore__ {
                {
                    MegaMoeTrainTimerPassiveScope wait(timer_, MegaMoeTrainTimer::AIV_SWIGLU_WAIT, slice.expert,
                                                       slice.row + m, shape.m(), n, kGmm1Rows, kPairedColumns, k_, f);
                    GateUpCompletion::Wait(sequence);
                    epilogue.LoadRawTile(params_.forwardRawGate, params_.forwardRawUp,
                                         RawTileOffset(blockIdx_, sequence), shape.m(), shape.n());
                }
                MegaMoeTrainTimerPassiveScope activation(timer_, MegaMoeTrainTimer::AIV_SWIGLU, slice.expert,
                                                         slice.row + m, shape.m(), n, kGmm1Rows, kPairedColumns, k_, f);
                if constexpr (SaveMode != MegaMoeBF16Impl::kSaveNone) {
#ifdef DEEPEP_MEGAMOE_TIMER
                    MegaMoeTrainTimerPassiveScope save(timer_, MegaMoeTrainTimer::AIV_SAVE_GATE_UP, slice.expert,
                                                       slice.row + m, shape.m(), n, kGmm1Rows, kPairedColumns, k_, f);
#endif
                    const PeerRowLayout<UbPeerPrefix> layout{{waveCumsum_}, {}, worldSize_, expertPerRank_, rankId_};
                    const uint64_t first = slice.packed + m;
                    layout.ForEachSpan(slice, first, first + shape.m(), savePeerLimits_,
                                       [&](const PeerTransfer& t, uint64_t packed) __aicore__ {
                                           epilogue.template SaveRows<SaveMode>(packed - first, t.rowCount, shape.n(),
                                                                                layout.SavedRow(slice, t) * f + n);
                                       });
                }
                epilogue.Run({shape.m(), shape.n(), 0, 0}, {static_cast<int64_t>(m) * f + n, 0, 0, 0, 0, 0});
                GateUpCompletion::Release(sequence++);
            });
    }
}

template <typename TopkWeightT>
template <bool TransW>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::Gmm2Wave(
    const WaveRange& range, MegaMoeBF16ForwardCatlassImpl::WaveMatmul<TransW>& matmul, uint32_t& startBlock,
    int32_t& issued)
{
    if (timer_ != nullptr) timer_->SetWave(range.begin.wave);
    using namespace MegaMoeBF16Impl::Wave;
    const WaveCounts counts{waveExpertRows_};
    auto cursor = range.begin;
    uint32_t groups = waveGroups_;
    WaveSlice slice{};
    while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
        auto* hp = reinterpret_cast<__gm__ int32_t*>(waveReadyBase_) +
                   ReadySlot(slice.packed - slice.row, slice.expert, slice.row);
        const auto scratchRow = scratch_.Row(range.begin.wave, range.begin.packed, slice.packed);
        const auto hpRow = scratch_.HpRow(scratchRow, slice.packed);
        const bool owns = matmul.Down(params_, slice, SliceOutput(scratchRow), hpRow, hp, startBlock);
        if (owns) Gmm2AivCompletion::Submit(waveIssued_ + blockIdx_ * INT_CACHELINE, ++issued);
    }
}

template <typename TopkWeightT>
__aicore__ inline bool MegaMoeBF16Forward<TopkWeightT>::NextWaveSlice(MegaMoeBF16Impl::Wave::Cursor& cursor,
                                                                      uint32_t& groups, WaveSlice& slice)
{
    return MegaMoeBF16Impl::Wave::NextScheduledSlice(cursor, groups, waveGroups_, expertPerRank_, maxOutputSize_,
                                                     waveCount_, WaveCounts{waveExpertRows_}, slice);
}

template <typename TopkWeightT>
__aicore__ inline bool MegaMoeBF16Forward<TopkWeightT>::NextDownCompletion(MegaMoeBF16Impl::Wave::Cursor& cursor,
                                                                           uint32_t& groups, uint32_t& startBlock,
                                                                           WaveSlice& slice)
{
    using namespace MegaMoeBF16Impl::Wave;
    while (NextWaveSlice(cursor, groups, slice)) {
        const uint32_t tasks = GetGmm2SliceTiles<kGmm2Rows, kTileColumns>(slice.rows, k_);
        if (AdvanceGmmSlice(blockIdx_, blockNum_, startBlock, tasks)) return true;
    }
    return false;
}

template <typename TopkWeightT>
__aicore__ inline bool MegaMoeBF16Forward<TopkWeightT>::PollDownCompletion()
{
    using namespace MegaMoeBF16Impl::Wave;
    auto& state = waveCompletion_;
    if (!state.pending || !Gmm2AivCompletion::HasPending(waveIssued_ + blockIdx_ * INT_CACHELINE, state.completed))
        return false;
    // issued only queues a notification. FULL establishes the actual FIX completion boundary.
    const auto& slice = state.slice;
    if (timer_ != nullptr) timer_->SetWave(state.cursor.wave);
    MegaMoeTrainTimerPassiveScope completion(timer_, MegaMoeTrainTimer::AIV_COMBINE_WAIT, slice.expert, slice.row,
                                             slice.rows, uint32_t(-1), kGmm2Rows, kTileColumns,
                                             params_.tilingData->n / 2U, k_, 1U);
    auto* ready = waveDownReady_ + ReadySlot(slice.packed - slice.row, slice.expert, slice.row);
    Gmm2AivCompletion::Acknowledge(ready, state.completed);
    state.pending = NextDownCompletion(state.cursor, state.groups, state.startBlock, state.slice);
    return true;
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::CombineSlice(const WaveSlice& slice, uint64_t scratchRow)
{
    SyncFunc<HardEvent::S_MTE2>();
    const auto owned = GetBalancedTokenRange(slice.rows, blockIdx_, blockNum_);
    const uint64_t begin = slice.packed + owned.start;
    const uint64_t end = begin + owned.count;
    MegaMoeTrainTimerPassiveScope combine(timer_, MegaMoeTrainTimer::AIV_COMBINE, slice.expert,
                                          slice.row + static_cast<uint32_t>(begin - slice.packed), end - begin);
    const uint32_t rowElements = Ops::Base::CeilAlign(k_, 32U);
    LocalTensor<bfloat16_t> buffers[2] = {{TPosition::VECIN, 0, rowElements},
                                          {TPosition::VECIN, rowElements * uint32_t(sizeof(bfloat16_t)), rowElements}};
    using namespace MegaMoeBF16Impl::Wave;
    const PeerRowLayout<UbPeerPrefix> layout{{waveCumsum_}, {waveSourcePrefix_}, worldSize_, expertPerRank_, rankId_};
    auto* sliceOutput = reinterpret_cast<__gm__ bfloat16_t*>(SliceOutput(scratchRow));
    layout.ForEachSpan(
        slice, begin, end, combinePeerLimits_, [&](const PeerTransfer& t, uint64_t packedRow) __aicore__ {
            const uint32_t source = t.sourceRank, rows = t.rowCount;
            const uint64_t remoteRow = layout.SourceRow(slice, t);
            GlobalTensor<bfloat16_t> input, output;
            input.SetGlobalBuffer(sliceOutput + (packedRow - slice.packed) * k_);
            input.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            const uint64_t returnOffset = params_.peermemInfo.ptrD - params_.peermemInfo.ptrBase;
            output.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(GetRankWinAddrWithOffset(bf16WinRankAddr, source, returnOffset)) +
                remoteRow * k_);
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t ping = waveCombineSequence_ & 1U;
                const TEventID event = ping ? EVENT_ID7 : EVENT_ID6;
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                DataCopy(buffers[ping], input[static_cast<uint64_t>(row) * k_], k_);
                SetFlag<HardEvent::MTE2_MTE3>(event);
                WaitFlag<HardEvent::MTE2_MTE3>(event);
                DataCopy(output[static_cast<uint64_t>(row) * k_], buffers[ping], k_);
                SetFlag<HardEvent::MTE3_MTE2>(event);
                ++waveCombineSequence_;
                // Keep the completion cursor moving while this slice's token sends are in flight.
                PollDownCompletion();
            }
        });
}

template <typename TopkWeightT>
template <uint32_t SaveMode, bool TransW>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::ProcessWave(TPipe& firstPipe)
{
    int combineDrainIndex = -1;
    using namespace MegaMoeBF16Impl::Wave;
    using Matmul = MegaMoeBF16ForwardCatlassImpl::WaveMatmul<TransW>;
    using Epilogue =
        MegaMoeBF16Impl::BlockEpilogueSwigluForwardBF16<MegaMoeBF16ForwardCatlassImpl::WaveEpilogueTileShape,
                                                        TopkWeightT, SaveMode == MegaMoeBF16Impl::kSaveMxFp8>;
    const uint32_t f = params_.tilingData->n / 2U;
    waveGroups_ = params_.tilingData->mGroupsPerWave;
    for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
        const int32_t rows = ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM) +
                                                (worldSize_ - 1U) * expertPerRank_ + expert);
        waveExpertRows_[expert] = rows > 0 ? static_cast<uint32_t>(rows) : 0;
    }
    const WaveCounts counts{waveExpertRows_};
    waveCount_ = CountWaves(waveGroups_, expertPerRank_, maxOutputSize_, counts);
    waveFirst_ = {};
    // Choose a rank-dependent start among complete physical Waves.
    // Source-major prefixes select the start only; peer quotas determine
    // the row origins inside every slice.
    for (uint32_t expert = 0; expert < expertPerRank_ && waveCount_ != 0U; ++expert) {
        const uint64_t rows = waveExpertRows_[expert] < maxOutputSize_ ? waveExpertRows_[expert] : maxOutputSize_;
        if (rows == 0U) continue;
        uint32_t firstWave = 0;
        for (uint32_t ordinal = 0; ordinal < worldSize_; ++ordinal) {
            const uint32_t source = (rankId_ + ordinal) % worldSize_;
            auto* prefix = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM);
            const int32_t begin =
                source == 0U ? 0 : ReadGmByPassDCache(prefix + (source - 1U) * expertPerRank_ + expert);
            const int32_t end = ReadGmByPassDCache(prefix + source * expertPerRank_ + expert);
            if (begin >= 0 && end > begin && static_cast<uint64_t>(begin) < rows) {
                firstWave = static_cast<uint32_t>(begin) / kGroupRows / waveGroups_;
                break;
            }
        }
        for (uint32_t i = 0; i < firstWave; ++i)
            waveFirst_ = Plan(waveFirst_, waveGroups_, expertPerRank_, maxOutputSize_, counts).end;
        break;
    }
    const WaveRange first = NextWaveRange(waveFirst_);
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            if (aivCoreIdx_ == 1) ExpertTokenNumCopyOut();
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
            const uint32_t elements = (worldSize_ * expertPerRank_ + 7U) / 8U * 8U;
            waveCumsum_ =
                LocalTensor<int32_t>{TPosition::VECCALC, MetadataUbBase(worldSize_, expertPerRank_), elements};
            waveSourcePrefix_ = LocalTensor<int32_t>{
                TPosition::VECCALC, MetadataUbBase(worldSize_, expertPerRank_) + elements * 4U, elements};
            const DataCopyExtParams copy{1U, worldSize_ * expertPerRank_ * 4U, 0U, 0U, 0U};
            DataCopyPad(waveCumsum_, cumsumInWorkSpace_, copy, DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
            DataCopyPad(waveSourcePrefix_, preSumBeforeRankWorkSpace_, copy,
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
            SyncFunc<HardEvent::MTE2_S>();
            Gmm2AivCompletion::InitConsumer();
            waveCompletion_ = {};
            waveCompletion_.cursor = waveFirst_;
            waveCompletion_.groups = waveGroups_;
            waveCompletion_.pending = NextDownCompletion(waveCompletion_.cursor, waveCompletion_.groups,
                                                         waveCompletion_.startBlock, waveCompletion_.slice);
            // Keep row-buffer credits across slices; only slot reuse waits for its previous send.
            waveCombineSequence_ = 0;
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            Cursor combineCursor = waveFirst_;
            uint32_t combineGroups = waveGroups_;
            WaveSlice combine{};
            bool hasCombine = NextWaveSlice(combineCursor, combineGroups, combine);
            uint64_t combineFirstRow = combine.packed;
            WaveRange dispatch = first;
            WaveRange priorInput = first;
            const bool stagedRing = scratch_.slotRows && params_.tilingData->forwardSchedule == 1U;
            for (uint32_t i = 0; i < 2 && dispatch.valid; ++i) {
                DispatchWave(dispatch);
                dispatch = NextWaveRange(dispatch.end);
            }
            bool preferCombine = true;
            while (dispatch.valid || hasCombine || waveCompletion_.pending) {
                bool progressed = PollDownCompletion();
                const bool ready =
                    hasCombine &&
                    ReadGmByPassDCache(waveDownReady_ +
                                       ReadySlot(combine.packed - combine.row, combine.expert, combine.row)) >=
                        GetGmm2ProducerCount(GetGmm2SliceTiles<kGmm2Rows, kTileColumns>(combine.rows, k_), blockNum_);
                const auto dispatchWave = dispatch.begin.wave;
                const bool reuseInput = stagedRing && dispatchWave >= MegaMoeImpl::kWaveScratchSlots;
                const bool canDispatch =
                    dispatch.valid && (reuseInput ? InputsConsumed(priorInput) : OutputSlotAvailable(dispatchWave));
                if (ready && ((params_.tilingData->forwardSchedule == 0U && preferCombine) || !canDispatch)) {
                    const auto sent = combineCursor.wave;
                    if (timer_ != nullptr) timer_->SetWave(sent);
                    CombineSlice(combine, scratch_.Row(sent, combineFirstRow, combine.packed));
                    hasCombine = NextWaveSlice(combineCursor, combineGroups, combine);
                    if (!hasCombine || sent != combineCursor.wave) {
                        combineFirstRow = combine.packed;
                        if (scratch_.slotRows && waveCount_ - sent > MegaMoeImpl::kWaveScratchSlots) {
                            // Each sender retires its last DMA before Dispatch can overwrite this slot.
                            SyncFunc<HardEvent::MTE3_S>();
                            AtomicAdd(waveReleased_ + (sent % MegaMoeImpl::kWaveScratchSlots) * INT_CACHELINE,
                                      int32_t(1));
                        }
                    }
                    preferCombine = false;
                    progressed = true;
                } else if (canDispatch) {
                    if (reuseInput) priorInput = NextWaveRange(priorInput.end);
                    DispatchWave(dispatch);
                    dispatch = NextWaveRange(dispatch.end);
                    preferCombine = true;
                    progressed = true;
                }
                if (!progressed) {
                    const auto begin = GetSystemCycle();
                    while (GetSystemCycle() - begin < 100) {
                    }
                }
            }
            // Retire both row buffers before the final cross-core/rank synchronization.
            if (timer_ != nullptr && timer_->Enabled() && combine.rows != 0U) {
                timer_->SetWave(MegaMoeTrainTimerSchema::NO_WAVE);
                timer_->SetSlice(combine.expert, combine.row, combine.rows);
                timer_->SetTaskShape(kGmm2Rows, k_, 0U, 0U, k_);
                timer_->SetTaskContext(-1, 2U);
                combineDrainIndex = timer_->BeginPassiveInterval(MegaMoeTrainTimer::AIV_COMBINE, uint32_t(-1), k_);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
        } else {
            GateUpCompletion::InitConsumer();
            Epilogue epilogue;
            if constexpr (SaveMode != MegaMoeBF16Impl::kSaveNone) {
                const uint32_t entries = worldSize_ * expertPerRank_;
                waveCumsum_ = epilogue.SavedRowPrefix((entries + 7U) / 8U * 8U);
                const DataCopyExtParams copy{1U, entries * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
                DataCopyPad(waveCumsum_, cumsumInWorkSpace_, copy, DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
                SyncFunc<HardEvent::MTE2_S>();
            }
            epilogue.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale, waveReadyBase_, waveWeights_,
                           nullptr, nullptr, nullptr, kGmm1Rows, kPairedColumns, params_.tilingData->swigluLimit,
                           params_.workspaceInfo.ptrSavedGate, params_.workspaceInfo.ptrSavedUp,
                           params_.workspaceInfo.ptrSavedGateScale, params_.workspaceInfo.ptrSavedUpScale,
                           params_.workspaceInfo.ptrSavedGateHigh, params_.workspaceInfo.ptrSavedUpLow,
                           params_.workspaceInfo.ptrSavedUpHigh, params_.tilingData->routeWeightBeforeGmm2 != 0U});
            epilogue.UpdateNextProblem({0, f, k_, 0});
            auto current = first;
            uint32_t startBlock = 0, sequence = 0;
            while (current.valid) {
                SwigluWave<SaveMode>(current, epilogue, startBlock, sequence);
                current = NextWaveRange(current.end);
            }
            if (scratch_.slotRows && params_.tilingData->forwardSchedule == 1U) {
                // All Hp tiles live in GM. GMM2 can now take ownership of the Dispatch slots.
                SyncFunc<HardEvent::MTE3_S>();
                AtomicAdd(stagedInputDone_, int32_t(1));
            }
        }
    } else if constexpr (g_coreType == AscendC::AIC) {
        Matmul matmul;
        auto current = first;
        WaveRange pending{};
        uint32_t gateStart = 0, downStart = 0, sequence = 0;
        int32_t issued = 0;
        const bool staged = params_.tilingData->forwardSchedule == 1U;
        const bool lag = !staged && UseGmm2Lag(m_);
        while (current.valid) {
            Gmm1Wave<TransW>(current, matmul, gateStart, sequence);
            if (staged) {
                // GMM2 starts after this core has submitted all GMM1 tasks.
            } else if (lag) {
                if (pending.valid) Gmm2Wave<TransW>(pending, matmul, downStart, issued);
                pending = current;
            } else {
                Gmm2Wave<TransW>(current, matmul, downStart, issued);
            }
            current = NextWaveRange(current.end);
        }
        if (staged) {
            if (scratch_.slotRows) {
                while (ReadGmByPassDCache(stagedInputDone_) < static_cast<int32_t>(blockNum_)) {
                }
            }
            current = first;
            while (current.valid) {
                while (!OutputSlotAvailable(current.begin.wave)) {
                }
                Gmm2Wave<TransW>(current, matmul, downStart, issued);
                current = NextWaveRange(current.end);
            }
        }
        if (lag && pending.valid) Gmm2Wave<TransW>(pending, matmul, downStart, issued);
        GateUpCompletion::Drain();
        Gmm2AivCompletion::Drain(GetTPipePtr()->FetchEventID(HardEvent::FIX_S));
    }
    PipeBarrier<PIPE_ALL>();
    if (timer_ != nullptr) timer_->SetWave(MegaMoeTrainTimerSchema::NO_WAVE);
    if (timer_ != nullptr && combineDrainIndex >= 0) {
        timer_->EndPassiveInterval(MegaMoeTrainTimer::AIV_COMBINE, combineDrainIndex);
    }
}
// Each AIV1 owns complete destination cache lines, including p fragments from different Waves.
template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::SaveDispatchWeights()
{
    using namespace MegaMoeBF16Impl::Wave;
    constexpr uint32_t line = 32 / sizeof(TopkWeightT), elements = kWeightUbBytes / sizeof(TopkWeightT);
    const uint64_t chunk = ((maxOutputSize_ + blockNum_ * line - 1) / (blockNum_ * line)) * line;
    const uint64_t lo = chunk * blockIdx_, hi = lo + chunk;
    if (lo >= maxOutputSize_) return;
    MegaMoeTrainTimerFixedScope timerSave(timer_, MegaMoeTrainTimer::AIV_SAVE_DISPATCH_WEIGHTS);
    LocalTensor<TopkWeightT> buffer{TPosition::VECIN, 0, elements};
    GlobalTensor<TopkWeightT> src, dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightT*>(waveWeights_));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightT*>(params_.workspaceInfo.ptrDispatchedTopkWeights));
    const PeerRowLayout<UbPeerPrefix> layout{{waveCumsum_}, {}, worldSize_, expertPerRank_, rankId_};
    const WaveCounts counts{waveExpertRows_};
    Cursor cursor{};
    WaveSlice slice{};
    uint32_t groups = waveGroups_;
    while (Next(cursor, groups, expertPerRank_, maxOutputSize_, counts, slice)) {
        if (!groups) groups = waveGroups_;
        const uint64_t base = slice.packed - slice.row;
        if (base >= hi) break;
        if (base + slice.fullRows <= lo) continue;
        layout.ForEachSpan(slice, slice.packed, slice.packed + slice.rows, savePeerLimits_,
                           [&](const PeerTransfer& t, uint64_t packed) __aicore__ {
                               const uint64_t saved = layout.SavedRow(slice, t);
                               const uint64_t begin = saved < lo ? lo : saved;
                               const uint64_t end = saved + t.rowCount < hi ? saved + t.rowCount : hi;
                               for (uint64_t i = begin; i < end; i += elements) {
                                   const uint32_t count = end - i < elements ? end - i : elements;
                                   const DataCopyExtParams cp{1, count * uint32_t(sizeof(TopkWeightT)), 0, 0, 0};
                                   DataCopyPad(buffer, src[packed + i - saved], cp, {});
                                   SyncFunc<HardEvent::MTE2_MTE3>();
                                   DataCopyPad(dst[i], buffer, cp);
                                   SyncFunc<HardEvent::MTE3_MTE2>();
                               }
                           });
    }
}

template <typename TopkWeightT>
__aicore__ inline void MegaMoeBF16Forward<TopkWeightT>::FinishForward(int64_t oriOverflowMode)
{
    SyncAll<true>();

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1 && params_.workspaceInfo.ptrDispatchedTopkWeights) SaveDispatchWeights();
        ResetTokenPerExpert();
        SyncAll<true>();
        CrossRankSyncInWorldSize();  // 全卡软同步，确认combine send完成
        Unpermute();
        PipeBarrier<PIPE_ALL>();
        // Publish a safe-to-reuse boundary only after the last SHMEM payload
        // reader has drained. A faster rank may enqueue the next invocation as
        // soon as it leaves this barrier without racing a slower peer.
        CrossRankSyncInWorldSize();
    }
    SetCtrlSpr<kBf16OverflowModeBit, kBf16OverflowModeBit>(oriOverflowMode);
}

}  // namespace MegaMoeImpl
#endif
