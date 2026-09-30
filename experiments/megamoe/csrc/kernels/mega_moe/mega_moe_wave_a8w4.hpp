// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_wave_a8w4.h
 * \brief MegaMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "common/mega_moe_utils.hpp"
#include "mega_moe_arch35.hpp"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass                                                                    \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool FuseGateUp
#define TemplateMegaMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, FuseGateUp

template <TemplateMegaMoeA8W4WaveTypeClass>
class MegaMoeA8W4Wave : public MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode,
                                       TopkWeightsPrefetch> {
private:
    using MegaMoeBase =
        MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;
    // 一个逻辑任务对应一对 gate/up；生产、SwiGLU 和 GMM2 ready 必须使用同一粒度。
    static constexpr uint32_t GMM1_TASK_N = FuseGateUp ? GMM1_FUSED_TILE_N : L1_TILE_N;

    using Gmm1Config =
        GmmKernel::Config<true, 0, QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, false,
                          TopkWeightsPrefetch, false, false, false, true, FuseGateUp>;
    GmmKernel::CatlassGmm1<typename Gmm1Config::ProblemConfig> gmm1CatlassContext_;

    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::RunGmm2CombineForExpert;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::timer_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;

    using SharedActivationCursor = typename MegaMoeBase::SharedActivationCursor;
    using MegaMoeBase::sharedExpertNum_;
    using MegaMoeBase::TryRunSharedExpertActivation;

    // 各任务流保留一个待办项，不分配随 BS 增长的任务队列。
    struct AivTaskCursor {
        ExpertTokenPosition position{};
        ExpertTokenRange slice{};
        uint64_t wave = 0U;
        uint32_t mGroups = 0U;
        uint32_t startBlockIdx = 0U;
        uint32_t nextTile = 0U;
        uint32_t tileCount = 0U;
    };
    __aicore__ inline bool NextAivSlice(AivTaskCursor& cursor, bool activation);
    __aicore__ inline bool TryRunReadyActivation(const AivTaskCursor& cursor, int32_t& readySequence);
    __aicore__ inline bool TryRunReadyCombine(const AivTaskCursor& cursor);
    __aicore__ inline void ProcessAiv1PriorityStages(int32_t& readySequence);
    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition& position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < mGroupsPerWave_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState& state, GMMAddrInfo& gmmAddrInfo,
                                                      uint32_t& startBlockIdx, int32_t& gmm1TileReadySequence,
                                                      uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount,
                                                      uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& waveMGroupCount,
                                                   uint64_t waveId);
    __aicore__ inline void RunGmm2CombineForWaveRange(const ExpertTokenPosition& waveBeginPosition,
                                                      const ExpertTokenPosition& waveEndPosition,
                                                      uint32_t waveLastActiveExpertIdx, ExpertLoopState& gmm2State,
                                                      GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx,
                                                      WaveCombineBufferConfig& combineBufferConfig,
                                                      uint32_t& combineRowSequence, int32_t& gmm1TileReadySequence);
    __aicore__ inline void ProcessMoeExpertStages(int32_t& gmm1TileReadySequence);
};

// 更新当前专家切片的输入、输出及同步地址，并执行 GMM1 和 Activation。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState& state, GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& gmm1TileReadySequence,
    uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    gmmAddrInfo.timer = timer_.Enabled() ? &timer_ : nullptr;
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    timer_.SetSlice(state.expertIdx, tokenStartIndexInExpert, sliceTokenCount);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 0U) {
            timer_.StartSpan(AscendTimer::AIV_GMM1_PROLOGUE);
        }
    }
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true, FuseGateUp>(
        epilogueOp_, params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmm1TileReadySequence,
        gmmExecutionConfig_.blockJob, static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert,
        state.expertIdx, &gmm1CatlassContext_);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 0U) {
            timer_.EndSpan(AscendTimer::AIV_GMM1_PROLOGUE);
        }
    }
    timer_.ClearSlice();
}

// 只由 AIV1 调用。复用原有分片、分核和完成边界。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::DispatchNextExpertSlice(
    ExpertTokenPosition& position, uint32_t& waveMGroupCount, uint64_t waveId)
{
    ExpertTokenRange range = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank, mGroupsPerWave_,
        waveMGroupCount, position);
    if (range.end.globalTokenIndex > range.begin.globalTokenIndex) {
        // SwiGLU/Combine 会复用 UB；从 GM 备份恢复本专家 prefix，不依赖上一阶段的 UB 内容。
        ReloadDispatchCumsumRange(commonConfig_, tokenDispatchScratch_, range.begin.expertIdx, range.begin.expertIdx);
        timer_.SetWave(static_cast<int64_t>(waveId));
        timer_.StartSpan(AscendTimer::AIV_DISPATCH);
        // AIC-balance使用独立GMM1/GMM2游标；只映射通信job，不能修改计算游标或物理核号。
        auto dispatchJob = gmmExecutionConfig_.blockJob;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            if (dispatchJob.totalJobs != 0U && dispatchJob.jobIndex < dispatchJob.totalJobs) {
                // 只轮转搬运 owner，不改 peer 配额或 GMM/SwiGLU 共用的物理核编号。
                // 同 wave 的所有 slice 使用同一偏移；取调度器 waveId，不依赖 timer。
                uint32_t shift = static_cast<uint32_t>(waveId % dispatchJob.totalJobs);
                dispatchJob.jobIndex = dispatchJob.jobIndex >= shift
                                           ? dispatchJob.jobIndex - shift
                                           : dispatchJob.jobIndex + dispatchJob.totalJobs - shift;
            }
        }
        // 小 batch 每核仅几行，直接二分源 rank，避免空 peer 的配额统计/扫描；索引预取仍保留。
        // 只改变行排列，token metadata 随行写出；GMM 和 Combine 的 slice/ready 边界不变。
        if (commonConfig_.tokenNum <= W4A8_SMALL_BATCH_MAX_TOKENS) {
            DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchConfig_, commonConfig_, dispatchJob, syncWorkspaceLayout_, params_, g_winRankAddr_,
                tokenDispatchScratch_, range);
        } else {
            DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch, true>(
                tokenDispatchConfig_, commonConfig_, dispatchJob, syncWorkspaceLayout_, params_, g_winRankAddr_,
                tokenDispatchScratch_, range);
        }
        timer_.EndSpan(AscendTimer::AIV_DISPATCH);
    }
    // 数据和 ready flag 发布后才提交进度；空专家也要推进，避免重复或死循环。
    position = range.end;
}

// 惰性枚举原 Wave 的 slice；SwiGLU 只重放 AIC 的独立 GMM1 分核游标，
// 即使本核没有当前 slice 的 tile，也不能漏掉它对后续分核起点的影响。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::NextAivSlice(AivTaskCursor& cursor,
                                                                                      bool activation)
{
    const uint32_t gmm1N =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
    const auto& job = gmmExecutionConfig_.blockJob;
    while (true) {
        if (!IsPositionWithinWave(cursor.position, cursor.mGroups) && cursor.mGroups != 0U) {
            cursor.mGroups = 0U;
            ++cursor.wave;
            UpdateGmmLoopCount(gmmLoopCount_, activation ? LoopCountIndex::GMM1 : LoopCountIndex::GMM2, cursor.wave);
        }
        if (cursor.position.expertIdx >= commonConfig_.moeExpertPerRank) {
            return false;
        }
        cursor.slice = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                   countWorkspace_, commonConfig_.moeExpertPerRank,
                                                                   mGroupsPerWave_, cursor.mGroups, cursor.position);
        cursor.position = cursor.slice.end;
        uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - cursor.slice.begin.globalTokenIndex);
        if (rows == 0U) {
            continue;
        }
        if (!activation) {
            return true;
        }
        cursor.tileCount = GetMGroupCountForRows(rows, GMM1_TILE_M) * gmm1N;
        cursor.nextTile = (job.jobIndex + job.totalJobs - cursor.startBlockIdx) % job.totalJobs;
        cursor.startBlockIdx = (cursor.startBlockIdx + cursor.tileCount) % job.totalJobs;
        if (cursor.nextTile < cursor.tileCount) {
            return true;
        }
    }
}

// 只探测本核下一个 GMM1 tile。AIC 按序发布，因此队首未就绪时，后续本核 tile 也不可能就绪。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunReadyActivation(
    const AivTaskCursor& cursor, int32_t& readySequence)
{
    const auto& begin = cursor.slice.begin;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t tilesPerM =
            Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
        uint64_t tileSlot = static_cast<uint64_t>(begin.expertIdx) * syncWorkspaceLayout_.gmm1TileStatusCountPerExpert +
                            static_cast<uint64_t>(begin.tokenIndexInExpert / GMM1_TILE_M) * tilesPerM + cursor.nextTile;
        auto flag =
            reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.gmm1TileStatusPtr) + tileSlot * INT_CACHELINE;
        if (AscendC::ReadGmByPassDCache(flag) != static_cast<int32_t>(begin.expertIdx + 1U)) {
            return false;
        }
    } else {
        auto flag = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                    static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        if (AscendC::ReadGmByPassDCache(flag) < readySequence + 1) {
            return false;
        }
    }

    uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    ExpertLoopState state = CreateExpertLoopState(commonConfig_);
    state.expertIdx = begin.expertIdx;
    state.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
    Get<M_VALUE>(state.problemShape) = rows;
    GMMAddrInfo addr{};
    uint32_t tilesPerM =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, addr,
        state, begin.tokenIndexInExpert, tilesPerM);
    addr.timer = timer_.Enabled() ? &timer_ : nullptr;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    timer_.SetSlice(begin.expertIdx, begin.tokenIndexInExpert, rows);
    RunReadyGmm1ActivationTileA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                   GMM1_TILE_M, TopkWeightsPrefetch, FuseGateUp>(
        epilogueOp_, state.problemShape, addr, gmmExecutionConfig_.blockJob, cursor.nextTile,
        static_cast<uint32_t>(begin.globalTokenIndex));
    timer_.ClearSlice();
    if constexpr (!TopkWeightsPrefetch) {
        ++readySequence;
    }
    return true;
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunReadyCombine(const AivTaskCursor& cursor)
{
    const auto& begin = cursor.slice.begin;
    uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    // 只轮换当前 AIV1 的通信任务，不修改成员中的物理核编号。
    // 输入是 Combine 自己的逻辑 wave（cursor.wave），不是当前 Dispatch/GMM1 的进度；
    // BS>=4096 时 GMM2 滞后一轮，使用其他阶段的 wave 会把同一批输出分给错误的 owner。
    // GMM2 反向分核、通信 owner 轮换都不改变 slice 的身份和实际生产核数量。
    auto combineJob = waveCombineJob_;
    if (combineJob.totalJobs != 0U && combineJob.jobIndex < combineJob.totalJobs) {
        uint32_t shift = static_cast<uint32_t>(cursor.wave % combineJob.totalJobs);
        // 与 Dispatch 相同：物理核 c 在 wave w 执行逻辑核 (c-w) mod 核数 的连续行任务。
        // 条件减法避免无符号下溢；先取模再转换，保证较大的 wave 编号也能正确轮换。
        combineJob.jobIndex = combineJob.jobIndex >= shift ? combineJob.jobIndex - shift
                                                           : combineJob.jobIndex + combineJob.totalJobs - shift;
    }
    // 本次只重映射完整任务，不新增 peer 分段。结合原有全局行前缀，等价首 owner 为
    // (sliceGlobalRowStart + cursor.wave) mod 核数；同 wave 跨专家的 slice 使用同一偏移。
    // 空任务判断必须使用轮换后的 owner，否则小 batch/尾片可能被错误跳过而漏发。
    WorkRange owned = GetWaveCombineOwnedRange(combineJob, rows, begin.globalTokenIndex);
    if (owned.count == 0U) {
        return true;
    }
    // 只读取当前 expert/slice 的计数；不同 slice 不共用槽，快核后续通知不能代偿慢核。
    // 没就绪就返回，让优先级调度器继续 SwiGLU/Dispatch，不能在探测函数里阻塞。
    auto ready = GetTokenCombineReadyCounter(params_, begin.expertIdx, begin.tokenIndexInExpert);
    if (AscendC::ReadGmByPassDCache(ready) <
        static_cast<int32_t>(GetTokenCombineProducerCount(commonConfig_, waveCombineJob_, rows))) {
        return false;
    }
    ExpertLoopState state = CreateExpertLoopState(commonConfig_);
    state.expertIdx = begin.expertIdx;
    state.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    // 将就绪检查使用的同一份 job 传给实际发送，保证判定和发送的行范围一致。
    // 当前 slice 已检查就绪，模板 true 跳过发送入口内的重复等待。计数只增不减，
    // 到本次算子结束前不会重置，因此检查与发送之间无需再次轮询。
    // timer 仍记录物理核；完整行发送及 UB 排空规则保持原样。
    RunTokenCombineSlice<true>(commonConfig_, combineJob, params_, state, begin.tokenIndexInExpert, rows, timer_,
                               waveCombineScratch_);
    return true;
}

// 非抢占式优先级：每完成一个 tile/slice 就重新检查 SwiGLU，不在未就绪任务内阻塞。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessAiv1PriorityStages(
    int32_t& readySequence)
{
    AivTaskCursor activation{}, combine{};
    bool hasActivation = NextAivSlice(activation, true);
    bool hasCombine = NextAivSlice(combine, false);
    SharedActivationCursor shared{};
    shared.done = sharedExpertNum_ == 0U;
    ExpertTokenPosition dispatch{};
    uint32_t dispatchGroups = 0U;
    uint64_t dispatchWave = 0U;
    while (hasActivation || hasCombine || !shared.done || dispatch.expertIdx < commonConfig_.moeExpertPerRank) {
        if (hasActivation && TryRunReadyActivation(activation, readySequence)) {
            activation.nextTile += gmmExecutionConfig_.blockJob.totalJobs;
            if (activation.nextTile >= activation.tileCount) {
                hasActivation = NextAivSlice(activation, true);
            }
            continue;
        }
        if (dispatch.expertIdx < commonConfig_.moeExpertPerRank) {
            if (dispatchGroups >= mGroupsPerWave_) {
                dispatchGroups = 0U;
                ++dispatchWave;
            }
            DispatchNextExpertSlice(dispatch, dispatchGroups, dispatchWave);
            continue;
        }
        if (!shared.done && TryRunSharedExpertActivation(shared)) {
            continue;
        }
        if (hasCombine && TryRunReadyCombine(combine)) {
            hasCombine = NextAivSlice(combine, false);
            continue;
        }
        // 没有可执行任务才退避；不能进入 Combine 的阻塞等待，否则可能挡住 GMM2 所需的 SwiGLU。
        int64_t start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
    timer_.ClearSlice();
    timer_.SetWave(AscendTimer::NO_WAVE);
}

// 消费一个已完成 Wave 的 GMM2/Combine（调度本体在基类 RunGmm2CombineForExpert）。
// Wave 在专家内结束时需包含该专家；在专家边界结束时，waveEndPosition 已指向下一专家。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForWaveRange(
    const ExpertTokenPosition& waveBeginPosition, const ExpertTokenPosition& waveEndPosition,
    uint32_t waveLastActiveExpertIdx, ExpertLoopState& gmm2State, GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx,
    WaveCombineBufferConfig& combineBufferConfig, uint32_t& combineRowSequence, int32_t& gmm1TileReadySequence)
{
    uint32_t waveGmm2ExpertEndExclusive =
        waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
    for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
        uint32_t sliceTokenStartIndexInExpert =
            expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
        // WAVE 从专家起点开始时，准备该专家的 GMM2 状态；跨 WAVE slice 沿用已有状态。
        if (sliceTokenStartIndexInExpert == 0U) {
            this->template PrepareGmmExpertState<false>(gmm2State, expertIdx);
        }
        uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
        uint32_t sliceTokenEndIndexInExpert =
            expertIdx == waveEndPosition.expertIdx ? waveEndPosition.tokenIndexInExpert : expertTokenCount;
        uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
        if (sliceTokenCount != 0U) {
            bool isFinalCombine = waveEndPosition.expertIdx >= commonConfig_.moeExpertPerRank &&
                                  expertIdx == waveLastActiveExpertIdx &&
                                  sliceTokenEndIndexInExpert >= expertTokenCount;
            // W4 的 GMM2/Combine 调度集中在基类，派生模板只负责提供当前专家 slice。
            this->template RunGmm2CombineForExpert<FuseGateUp>(
                gmm2State, gmm2AddrInfo, startBlockIdx, sliceTokenStartIndexInExpert, sliceTokenCount,
                combineBufferConfig, combineRowSequence, gmm1TileReadySequence, isFinalCombine, true);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        DrainCombineRowBuffers(combineRowSequence, combineBufferConfig.rowBufferCount);
    }
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 按专家 slice 交替执行下一 Wave 的 Dispatch 和当前 Wave 的 Activation，同时 AIC/AIV0 执行当前 Wave 的 GMM1。
 * 整 Wave 与专家 slice 都向 DispatchTokenRange 传入显式 [begin, end) 范围。当前 Wave 完成后立即启动
 * GMM2/Combine，无需等待所有专家的 GMM1。
 * 大 bs 下 GMM2 滞后一拍：wave w 的 GMM2/Combine 延至 GMM1(w+1) 之后，用下一 Wave 的 GMM1
 * 覆盖 AIC 等激活的自旋（对齐 A8W8 的 gmm2LaggedTarget 设计；A8W4 的 AIV0 prologue 与
 * AIV1 combine 在 GMM2 调用内均有实活，故三角色整调用同序滞后）。
 * 非量化 token 路径的 AIV1 使用独立的就绪优先级循环；下面保留 AIC/AIV0 和量化路径的原顺序。
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(
    int32_t& gmm1TileReadySequence)
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    DispatchBuffInit();
    PrepareMoeExpertTokenCountTable<true>(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    if constexpr (g_coreType == AIV && CombineQuantMode == COMBINE_NO_QUANT) {
        if (GetSubBlockIdx() == 1U) {
            ProcessAiv1PriorityStages(gmm1TileReadySequence);
            return;
        }
    }
    WaveCombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    }
    uint32_t combineRowSequence = 0U;

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 两阶段各自跨专家、跨 Wave 轮转。GMM1 从物理核 0 正向分配，
    // GMM2 的逻辑核号反转，从物理核 blockNum-1 反向分配。
    uint32_t gmm1StartBlockIdx = 0U;
    uint32_t gmm2StartBlockIdx = 0U;

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));

    ExpertTokenPosition dispatchPosition{};

    // GMM2 滞后一拍门控：与 A8W8 的 GMM2_LAG_MIN_TOKEN_NUM(4096) 同门槛；
    // 关闭态与现役"当前 Wave 完成后立即消费"逐字等价。
    const bool gmm2LagActive = commonConfig_.tokenNum >= 4096U;
    ExpertTokenPosition prevWaveBeginPosition{};
    ExpertTokenPosition prevWaveEndPosition{};
    uint32_t prevWaveLastActiveExpertIdx = commonConfig_.moeExpertPerRank;
    bool hasPendingGmm2Wave = false;

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ExpertTokenRange firstDispatchRange{dispatchPosition, dispatchPosition};
            ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
            uint32_t firstDispatchWaveMGroupCount = 0U;
            while (IsPositionWithinWave(plannedDispatchPosition, firstDispatchWaveMGroupCount)) {
                ExpertTokenRange nextDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                    params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank,
                    mGroupsPerWave_, firstDispatchWaveMGroupCount, plannedDispatchPosition);
                plannedDispatchPosition = nextDispatchRange.end;
                firstDispatchRange.end = nextDispatchRange.end;
            }
            // count-table 准备刚完成，UB prefix 仍有效；首 WAVE 无需从 GM 备份重复恢复。
            timer_.SetWave(0);
            timer_.StartSpan(AscendTimer::AIV_DISPATCH);
            DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
                g_winRankAddr_, tokenDispatchScratch_, firstDispatchRange);
            timer_.EndSpan(AscendTimer::AIV_DISPATCH);
            // 首 WAVE 的全部数据和 ready flag 发布完成后，再提交 Dispatch 进度。
            dispatchPosition = plannedDispatchPosition;
        }
    }

    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        uint32_t waveLastActiveExpertIdx = commonConfig_.moeExpertPerRank;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 每轮先发送下一 Wave 的一个专家 slice，再处理当前 Wave 的一个专家 slice。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    DispatchNextExpertSlice(dispatchPosition, nextDispatchWaveMGroupCount, gmm1Count + 1U);
                    nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    this->template PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, mGroupsPerWave_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    waveLastActiveExpertIdx = gmm1State.expertIdx;
                    timer_.SetWave(static_cast<int64_t>(gmm1Count));
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, gmm1StartBlockIdx, gmm1TileReadySequence,
                                               sliceTokenStartIndexInExpert, sliceTokenCount, gmm1TilesPerMGroup);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 = IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        if (gmm2LagActive) {
            // 滞后一拍：本轮消费上一 Wave 的 GMM2/Combine；当前 Wave 边界缓存到下一轮。
            if (hasPendingGmm2Wave) {
                timer_.SetWave(static_cast<int64_t>(gmm2Count));
                RunGmm2CombineForWaveRange(prevWaveBeginPosition, prevWaveEndPosition, prevWaveLastActiveExpertIdx,
                                           gmm2State, gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig,
                                           combineRowSequence, gmm1TileReadySequence);
                UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
            }
            prevWaveBeginPosition = waveBeginPosition;
            prevWaveEndPosition = waveEndPosition;
            prevWaveLastActiveExpertIdx = waveLastActiveExpertIdx;
            hasPendingGmm2Wave = true;
        } else {
            // 当前 Wave 完成后立即消费，使其 GMM2/Combine 与下一 Wave 的输入预取重叠。
            timer_.SetWave(static_cast<int64_t>(gmm2Count));
            RunGmm2CombineForWaveRange(waveBeginPosition, waveEndPosition, waveLastActiveExpertIdx, gmm2State,
                                       gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig, combineRowSequence,
                                       gmm1TileReadySequence);
            UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        }
    }
    // 滞后流水收尾：三角色共同补跑最后一个 Wave 的 GMM2/Combine。
    if (hasPendingGmm2Wave) {
        timer_.SetWave(static_cast<int64_t>(gmm2Count));
        RunGmm2CombineForWaveRange(prevWaveBeginPosition, prevWaveEndPosition, prevWaveLastActiveExpertIdx, gmm2State,
                                   gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig, combineRowSequence,
                                   gmm1TileReadySequence);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
    }
    timer_.SetWave(AscendTimer::NO_WAVE);
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

}  // namespace MegaMoeImpl

#undef TemplateMegaMoeA8W4WaveTypeClass
#undef TemplateMegaMoeA8W4WaveTypeFunc

#endif
