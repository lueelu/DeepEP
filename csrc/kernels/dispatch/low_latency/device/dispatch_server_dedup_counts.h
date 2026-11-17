// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_COUNTS_H
#define DISPATCH_SERVER_DEDUP_COUNTS_H

#include "kernel_operator.h"
#include "ll_transport.h"
#include "ll_count_frame_transport.h"
#include "ll_count_frame_receive.h"
#include "ll_count_histogram.h"
#include "../common/ll_count_frame_layout.h"
#include "../common/ll_count_assignment.h"
#include "../common/ll_reference_profile_device.h"
#include "ll_count_c_api.h"
#include "../common/ep_memory_server_dedup_count_layout.h"

namespace DispatchDedup {
using namespace LlCountC;

// 中文审计：这里包含跨rank计数交换，不属于four_metadata测得的本地路由阶段。
// 修改计数协议需联查dispatch_server_dedup_kernel.h中的阶段顺序及Receiver使用的前缀布局。
// Adapted from Memory's CalAndSendCntByRankAndExpert / WaitDispatch / cumsum
// chain. Counts cover the complete original routing tensor, including relay routes.
// 本类没有 Process()：上层依次调用 SendCnt，再由接收核执行
// RecvCnt → CalcCumsum → CompleteCumsum；发送核在 SendCnt 后进入 token 发送。
// 计数覆盖原始专家路由（含后续 relay 路由），不按去重后的跨 server 包数计数。
class ServerDedupCounts {
public:
    // 摘要：绑定路由与计数输出，选择本轮控制区；保存并调整 CTRL[60]，供整数位模式的 FP32 归约使用。
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertIds, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                uint32_t bs, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType,
                                int64_t magic, LlReferenceProfile::DeviceProfile* profile = nullptr)
    {
        profile_ = profile;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        epRankId_ = args_->rank;
        epWorldSize_ = args_->rankSize;
        aivId_ = asc_get_block_idx();
        coreNum_ = asc_get_block_num();
        aivUsedAllToAll_ = aivUsedCumSum_ = coreNum_ / 2U;
        expertIdsCnt_ = bs * topK;
        rscvStatusNum_ = experts;
        moeExpertNumPerRank_ = experts / epWorldSize_;
        expertTokenNumsType_ = expertTokenNumsType;
        useAllCoreCount_ = bs < kEpServerDedupAllCoreCountBsThreshold;
        packedFrames_ = LlTransport::Rdma(args_) && epWorldSize_ > 16U;
        directRanks_ = LlTransport::Rdma(args_) && epWorldSize_ > kLlPhysicalRanksPerHost;
        countWindow_ = DispatchServerDedupWindowOffset(magic);
        retirement_.Init(args_);
        layout_ = DispatchServerDedupCountLayout(experts, coreNum_);
        stateOffset_ = DispatchServerDedupWindowOffset(magic) + layout_.cntOffset;
        GM_ADDR localState = args_->peerMems[epRankId_] + stateOffset_;
        stateGM_ = localState;
        // 计数记录与 cumsum 槽位都由本机 L2 可达的写者更新（远端 RDMA 与同机 MTE3），
        // 读侧保持默认缓存视图才能立即看到新值；绕过 L2 会读到尚未回写的 HBM，
        // 表现为 ready 轮询多等十几微秒。
        recvCntWorkspaceGM_ = localState + layout_.prefixOffset;
        sendCountsOutGM_ = sendCountsOut;
        expertIdsGM_ = expertIds;
        expertTokenNumsGM_ = expertTokenNumsOut;
        // Memory's count reductions add integer bit patterns as FP32 subnormals.
        originalOverflowMode_ = uint64_t(asc_get_ctrl()) & (1ULL << 60U);
        asc_set_ctrl(uint64_t(asc_get_ctrl()) & ~(1ULL << 60U));
    }

    // 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
    __aicore__ inline void SendCnt();
    // 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
    __aicore__ inline void RecvCnt();
    // 摘要：汇合接收核片段总数，补齐跨核前缀偏移并发布完整前缀表的就绪标记。
    __aicore__ inline void CalcCumsum()
    {
        CalRecvAndSetFlag();
    }
    // 摘要：等待所有前缀片段就绪，仅首个接收核加载前缀并输出各专家计数。
    __aicore__ inline void CompleteCumsum();
    // 摘要：返回本核计数发送记录占用的 UB 字节数，供后续阶段计算剩余容量。
    __aicore__ inline uint32_t ReservedBytes() const
    {
        return statusCntAlign_ * kEpServerDedupCountRecordBytes;
    }
    __aicore__ inline LlTransport::CountRetirement* Retirement() { return &retirement_; }
    __aicore__ inline void RetireOwnScratch() { retirement_.Finish(reinterpret_cast<__ubuf__ uint8_t*>(histogram_)); }
    // 摘要：等待本核全部流水收尾，并恢复初始化前的 CTRL[60] 设置。
    __aicore__ inline void Finish()
    {
        asc_sync_pipe(PIPE_ALL);
        asc_set_ctrl((uint64_t(asc_get_ctrl()) & ~(1ULL << 60U)) | originalOverflowMode_);
    }

private:
    static constexpr uint32_t UB_ALIGN = 32U;
    static constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    static constexpr uint32_t SIZE_ALIGN_256 = 256U;
    uint32_t aivUsedAllToAll_ = 0, coreNum_ = 0;
    uint32_t aivUsedCumSum_ = 0;

    __aicore__ inline uint32_t Align(uint32_t bytes, uint32_t alignment = 32U)
    { return (bytes + alignment - 1U) / alignment * alignment; }
    __aicore__ inline __ubuf__ uint32_t* Scratch(uint32_t bytes)
    {
        auto ptr = Ub<uint32_t>(scratchEnd_);
        scratchEnd_ += Align(bytes);
        return ptr;
    }
    // 摘要：将连续区间均分到各核，余数优先分配给前面的核，返回半开区间及长度。
    __aicore__ inline void SplitToCore(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start, uint32_t& end,
                                       uint32_t& count)
    {
        const uint32_t base = total / cores;
        const uint32_t remainder = total % cores;
        count = base + (core < remainder ? 1U : 0U);
        start = core * base + (core < remainder ? core : remainder);
        end = start + count;
    }
    // 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
    __aicore__ inline void SendExpertCntRange(uint32_t start, uint32_t end);
    // 摘要：清零槽位并批量打 ready，再用硬件直方图回填 count；本机专家区间与跨机远端 rank 共用。
    __aicore__ inline void CountSlots(__ubuf__ uint32_t* records, uint32_t first, uint32_t bins,
                                     bool wire = false, bool remoteTail = false);
    __aicore__ inline void PackLocalCountWire(uint32_t records);
    // 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
    __aicore__ inline void WaitDispatchClearStatus();
    // 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
    __aicore__ inline void GatherSumRecvCnt();
    // 摘要：将本核片段总数和就绪标记写入本 rank 的跨核汇总矩阵；空片段也必须发布。
    __aicore__ inline void PublishRecvCntSum(uint32_t sumOfRecvCnt);
    // 摘要：等待所有接收核总数就绪，求本核之前各片段之和，再清理本核消费的汇总标记。
    __aicore__ inline uint32_t GetCumSum(uint32_t newAivId);
    // 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
    __aicore__ inline void WaitDispatch();
    // 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与共享前缀，再发布完成标记。
    __aicore__ inline void CalRecvAndSetFlag();
    // 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
    __aicore__ inline void WaitCumSumFlag();

    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    EpServerDedupCountLayout layout_{};
    uint64_t stateOffset_ = 0;
    uint64_t originalOverflowMode_ = 0;
    uint32_t epRankId_ = 0, epWorldSize_ = 0, aivId_ = 0;
    uint32_t expertIdsCnt_ = 0;
    uint32_t rscvStatusNum_ = 0, moeExpertNumPerRank_ = 0, expertTokenNumsType_ = 0;
    uint32_t statusCntAlign_ = 0;
    uint32_t startStatusIndex_ = 0, endStatusIndex_ = 0, recStatusNumPerCore_ = 0;
    uint32_t activeCumSumCores_ = 0;
    bool useAllCoreCount_ = false;
    bool packedFrames_ = false;
    bool directRanks_ = false;
    uint64_t countWindow_ = 0U;
    LlTransport::CountRetirement retirement_;
    GM_ADDR recvCntWorkspaceGM_ = nullptr;
    GM_ADDR sendCountsOutGM_ = nullptr;
    GM_ADDR expertIdsGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR stateGM_ = nullptr;
    // Retained records occupy [0, ReservedBytes()); all scratch follows them.
    uint32_t scratchEnd_ = 0;
    __ubuf__ uint32_t* records_ = nullptr;
    __ubuf__ int32_t* routes_ = nullptr;
    __ubuf__ uint16_t* histogram_ = nullptr;
    __ubuf__ uint32_t* work_ = nullptr;
    __ubuf__ uint32_t* scalar_ = nullptr;
    __ubuf__ uint32_t* snapshot_ = nullptr;
    __ubuf__ uint32_t* dense_ = nullptr;
    __ubuf__ uint32_t* sumSlot_ = nullptr;
    __ubuf__ uint32_t* sumSnapshot_ = nullptr;
    __ubuf__ uint32_t* sumDense_ = nullptr;
    __ubuf__ uint32_t* ready_ = nullptr;
    __ubuf__ uint32_t* clean_ = nullptr;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    bool countRemote_ = false;
#endif
};

// 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
__aicore__ inline void ServerDedupCounts::SendCnt()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t routeBegin = profile_ ? profile_->CountClock() : 0U;
#endif
    // Same-host work keeps the original expert split. Each physical core owns
    // remote rank tasks core, core+coreNum, ... independent of the BS threshold.
    const uint32_t cores = useAllCoreCount_ ? coreNum_ : aivUsedCumSum_;
    const auto assignment = CountAssignment(epRankId_, epWorldSize_, rscvStatusNum_,
                                            coreNum_, aivId_, useAllCoreCount_, directRanks_);
    const uint32_t start = assignment.first, end = assignment.end, count = end - start;
    const bool remoteOwner = assignment.remoteRank < epWorldSize_;
    // Only local records stay reserved into payload. MTE3_S retires the remote
    // UB frame copy; its isolated GM source survives independently until CQ retirement.
    statusCntAlign_ = packedFrames_ ? CountWireBytes(count) / UB_ALIGN : assignment.alignedRecords;
    records_ = Ub<uint32_t>(0U);
    scratchEnd_ = ReservedBytes();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<1>();
#endif
    if (count == 0U && !remoteOwner) return;

    // 3. 为完整 bs×topK 路由及直方图临时数据分配 UB，等待 GM 路由搬入。
    const uint32_t routeBytes = Align(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256);
    const uint32_t remoteBytes = assignment.remoteRecords * UB_ALIGN;
    const uint32_t workBytes = routeBytes < remoteBytes ? remoteBytes : routeBytes;
    routes_ = reinterpret_cast<__ubuf__ int32_t*>(Scratch(routeBytes));
    histogram_ = reinterpret_cast<__ubuf__ uint16_t*>(Scratch(routeBytes));
    work_ = Scratch(workBytes);
    Load(routes_, expertIdsGM_, expertIdsCnt_ * sizeof(int32_t));
    Sync<PIPE_MTE2, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<2>();
    if (profile_) profile_->CountRoute(profile_->CountClock() - routeBegin);
#endif
    if (remoteOwner) {
        auto remoteFrame = work_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint32_t firstFrameIssued = 0U;
#endif
        for (uint32_t task = aivId_; task < epWorldSize_ - kLlPhysicalRanksPerHost; task += coreNum_) {
            const uint32_t destination = CountRemotePeer(epRankId_, epWorldSize_, task);
            const uint32_t first = destination * moeExpertNumPerRank_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            countRemote_ = true;
#endif
            CountSlots(remoteFrame, first, moeExpertNumPerRank_, false, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->CountBoundary<12>();
            const uint64_t publishBegin = profile_ ? profile_->CountClock() : 0U;
            const uint64_t p0 = profile_ ? profile_->PublishClock() : 0U;
#endif
            Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p1 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<2, true>();
#endif
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_, cores,
                                          destination, epRankId_, 0U, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p2 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<3, true>();
#endif
            LlTransport::DirectCountFrame(args_, destination, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(remoteFrame), moeExpertNumPerRank_,
                reinterpret_cast<__ubuf__ uint8_t*>(histogram_), countWindow_
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p3 = profile_ ? profile_->PublishClock() : 0U;
#endif
            retirement_.Posted(task);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<9, true>();
                profile_->CompletePublishAdd<1>(1U);
                const uint64_t p4 = profile_->PublishClock();
                profile_->PublishAdd<1>(1U); profile_->PublishAdd<2>(p1-p0);
                profile_->PublishAdd<3>(p2-p1); profile_->PublishAdd<9>(p4-p3);
                profile_->CountPublish(true, profile_->CountClock() - publishBegin);
                if (!firstFrameIssued) { profile_->Mark<44>(); firstFrameIssued = 1U; }  // w44: send_count_first_frame_issued
                // Only the last frame of this core's stride gets a clock so the
                // per-frame post cost is not inflated by repeated scalar stores.
                if (task + coreNum_ >= epWorldSize_ - kLlPhysicalRanksPerHost) profile_->Mark<45>();  // w45: send_count_last_frame_issued
            }
#endif
            // Source GM slots are disjoint. Acquire the SDK's retired UB scratch
            // before the next Vector pass; no network completion wait here.
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t postSVBegin = profile_ ? profile_->CountClock() : 0U;
            if (profile_) profile_->CountBoundary<12>();
#endif
            Sync<PIPE_S, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<11>();
                profile_->CountPostSV(profile_->CountClock() - postSVBegin);
            }
#endif
        }
    }

    // 4. 本机专家区间统一走硬件直方图；零计数专家也发布 ready=1 和 count=0，避免对端永久等待。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    countRemote_ = false;
#endif
    if (count) {
        CountSlots(records_, start, count, packedFrames_);
        if (packedFrames_) PackLocalCountWire(count);
    }

    // 5. Scalar 填好的记录通过 MTE3 写到目标专家所属 rank。
    // 此处仅下发，不全量等待：records_ 保持不变，而筛选临时 UB 可以交给后续阶段。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t localPublishBegin = profile_ ? profile_->CountClock() : 0U;
    const uint64_t l0 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t l1 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<12, true>();
#endif
    if (count) SendExpertCntRange(start, end);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (count && profile_) {
        profile_->CompletePublishAdd<10>(1U);
        profile_->PublishAdd<10>(1U); profile_->PublishAdd<12>(l1-l0);
        profile_->CountPublish(false, profile_->CountClock() - localPublishBegin);
    }
#endif
    // Only records_ is still read by MTE3. V_S retired the shared scratch.
    // Keep CTRL[60] cleared through the subsequent FP8 quantization.
}

// 本机专家区间与跨机远端 rank 共用：清零对齐槽、批量打 ready，再用一次硬件直方图回填 count。
// 跨机帧同时初始化尾 ready 记录；本机 wire 初始化所有数据及 Flag 块。
__aicore__ inline void ServerDedupCounts::CountSlots(__ubuf__ uint32_t* records, uint32_t first, uint32_t bins,
                                                    bool wire, bool remoteTail)
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<12>();
        else profile_->CountBoundary<13>();
        profile_->CompleteCountCall(countRemote_, bins);
    }
    if (profile_ && profile_->CountNeedsCalibration()) {
        const uint64_t c0 = profile_->CountClock(), c1 = profile_->CountClock();
        uint64_t emptyWait = 0U;
        if constexpr (LlReferenceProfile::kCountComputeMode == 2U) {
            // MTE2_V submits a Vector dependency. Retire it before measuring an
            // idle V_S so the calibration cannot include the initial route DMA.
            Sync<PIPE_V, PIPE_S>();
            const uint64_t e0 = profile_->Clock();
            Sync<PIPE_V, PIPE_S>();
            emptyWait = profile_->Clock() - e0;
        }
        profile_->CountCalibration(c1 - c0, emptyWait);
    }
    const uint64_t t0 = profile_ ? profile_->CountClock() : 0U;
#endif
    const uint32_t slots = (bins + (remoteTail ? 1U : 0U) + 7U) / 8U * 8U;
    Fill(records, 0U, wire ? CountWireBytes(bins) / sizeof(int32_t) : slots * UB_ALIGN_DATA_COUNT);
    asc_sync_pipe(PIPE_V);
    RecordReady(records, 0x3f800000U, slots);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if constexpr (LlReferenceProfile::kCountComputeMode == 2U) {
        if (profile_ && profile_->Enabled()) Sync<PIPE_V, PIPE_S>();
    }
    const uint64_t t1 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<3>();
        else profile_->CountBoundary<7>();
    }
#endif
    histogram_accumulate_u16(routes_, histogram_,
                             static_cast<int>(expertIdsCnt_), static_cast<int>(first),
                             static_cast<int>(bins));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<4>();
        else profile_->CountBoundary<8>();
    }
#endif
    Sync<PIPE_V, PIPE_S>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t3 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<5>();
        else profile_->CountBoundary<9>();
    }
#endif
    for (uint32_t i = 0; i < bins; ++i)
        records[i * UB_ALIGN_DATA_COUNT + 1U] = uint32_t(histogram_[i + 1U]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<6>();
        else profile_->CountBoundary<10>();
        profile_->CountSections(countRemote_, bins, t0, t1, t2, t3, profile_->CountClock());
    }
#endif
}

__aicore__ inline void ServerDedupCounts::PackLocalCountWire(uint32_t records)
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t wirePackBegin = profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    // Reverse order preserves dense sources while inserting one 32B flag per
    // fifteen records. Every destination reads this same image without edits.
    for (uint32_t i = records; i > kLlCountWireRecords;) {
        --i;
        const uint32_t count = records_[i * UB_ALIGN_DATA_COUNT + 1U];
        const uint32_t target = CountWireRecordOffset(i) / sizeof(int32_t);
        records_[target] = 0x3f800000U;
        records_[target + 1U] = count;
    }
    for (uint32_t block = 0; block < CountWireBytes(records); block += kLlCountWireBlockBytes)
        for (uint32_t word = 0; word < UB_ALIGN_DATA_COUNT; ++word)
            records_[(block + kLlCountWireFlagOffset) / sizeof(int32_t) + word] = 0x3f800000U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        profile_->CountBoundary<16, true>();
        profile_->CompletePublishAdd<17>(1U);
        profile_->CountWirePack(profile_->PublishClock() - wirePackBegin);
    }
#endif
}

// 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
__aicore__ inline void ServerDedupCounts::SendExpertCntRange(uint32_t start, uint32_t end)
{
    for (uint32_t expert = start; expert < end;) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t planBegin = profile_ ? profile_->PublishClock() : 0U;
        if (profile_) profile_->CountBoundary<13>();
#endif
        const uint32_t dst = expert / moeExpertNumPerRank_;
        const uint32_t localExpert = expert % moeExpertNumPerRank_;
        const uint32_t count =
            end - expert < moeExpertNumPerRank_ - localExpert ? end - expert : moeExpertNumPerRank_ - localExpert;
        const uint64_t offset = (uint64_t(localExpert) * epWorldSize_ + epRankId_) * kEpServerDedupCountGmStride;
        if (packedFrames_) {
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, dst, epRankId_, localExpert, directRanks_);
            if (frame.firstExpert != localExpert || frame.records != count) { trap(); return; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t planEnd = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<13, true>();
#endif
            LlTransport::CountFrame(args_, dst, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(records_) + frame.sourceOffset, frame.wireBytes, count
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->PublishAdd<13>(planEnd-planBegin);
#endif
            expert += count;
            continue;
        }
        LlTransport::Records(args_, dst, stateOffset_ + offset,
            reinterpret_cast<__ubuf__ uint8_t*>(records_) + (expert - start) * UB_ALIGN,
            count, epWorldSize_ * kEpServerDedupCountGmStride, true,
            reinterpret_cast<__ubuf__ uint8_t*>(work_));
        expert += count;
    }
}

// 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
__aicore__ inline void ServerDedupCounts::RecvCnt()
{
    // 接收端控制区共有 localExperts×worldSize（等于 experts）条记录，
    // 按“本地专家 → 来源 rank”排列；每个接收核负责一段连续记录。
    // Each count owns a 512B GM slot, so contiguous ownership can split at any
    // record. Keep all nonempty receivers active and their loads within one record.
    activeCumSumCores_ = rscvStatusNum_ < aivUsedCumSum_ ? rscvStatusNum_ : aivUsedCumSum_;
    scratchEnd_ = ReservedBytes();
    scalar_ = Scratch(3U * UB_ALIGN);
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // Inactive receivers still join prefix readiness in CompleteCumsum.
    if (receiver >= activeCumSumCores_) return;
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, receiver, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);
    snapshot_ = Scratch(Align(recStatusNumPerCore_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256));
    dense_ = Scratch(rscvStatusNum_ * sizeof(int32_t));
    // Vector publication/cleanup touches complete groups of eight 32B records.
    // Padding is UB-only; matrix strides and participation use actual core counts.
    const uint32_t sumBytes = Align(aivUsedCumSum_ * UB_ALIGN, SIZE_ALIGN_256);
    sumSlot_ = Scratch(sumBytes);
    sumSnapshot_ = Scratch(sumBytes);
    sumDense_ = Scratch(aivUsedCumSum_ * sizeof(float));
    // 等待本片 ready 到齐后提取 count，并计算局部累计值、发布本片总数。
    // 本阶段不会等待 token payload；即使分到空片段也要发布总数 0，参与后续跨核汇总。
    WaitDispatch();
}

// 摘要：等待所有前缀片段就绪，仅首个接收核加载前缀并输出各专家计数。
__aicore__ inline void ServerDedupCounts::CompleteCumsum()
{
    // Convert the physical AIV index to its receive-group index.
    // All receivers join readiness; only receiver0 needs a count-stage prefix.
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    ready_ = Scratch(Align(activeCumSumCores_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256));
    clean_ = Scratch(activeCumSumCores_ * UB_ALIGN);
    // 各核先写前缀再发完成标记；全部标记到齐后，读取才不会混入未更新片段。
    if (profile_) profile_->Mark<28>();  // w28: prefix_ready_wait_begin
    WaitCumSumFlag();
    if (profile_) profile_->Mark<29>();  // w29: prefix_ready_wait_clear_issue_end
    // 仅接收组的第 0 核写公开专家计数，避免多核重复写同一输出。
    // 每个专家最后一个来源 rank 的前缀即该专家结束位置；type=0 输出累计数，
    // 其他已验证类型输出与上一专家累计数之差，即该专家自己的接收条数。
    if (receiver == 0U) {
        // Receiver::Process resets the count UB and loads its own prefix.
        // Other cores have no consumer for an additional count-stage copy.
auto prefix = Scratch(rscvStatusNum_ * sizeof(int32_t));
        Load(prefix, recvCntWorkspaceGM_, rscvStatusNum_ * sizeof(int32_t));
        Sync<PIPE_MTE2, PIPE_S>();
        auto expertCounts = reinterpret_cast<__ubuf__ int64_t*>(Scratch(moeExpertNumPerRank_ * sizeof(int64_t)));
        int64_t previous = 0;
        for (uint32_t expert = 0; expert < moeExpertNumPerRank_; ++expert) {
            const int64_t cumulative = prefix[(expert + 1U) * epWorldSize_ - 1U];
            expertCounts[expert] = expertTokenNumsType_ == 0U ? cumulative : cumulative - previous;
            previous = cumulative;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        Store(expertTokenNumsGM_, expertCounts, moeExpertNumPerRank_ * sizeof(int64_t));
    }
    // No listening table or payload receive/copy is constructed in this increment.
}

// 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
__aicore__ inline void ServerDedupCounts::WaitDispatchClearStatus()
{
    if (packedFrames_) {
        // Called only after GetCumSum has observed every receiver's snapshot.
        // Exactly the owner of a frame's first logical record clears its wire.
        auto zero = snapshot_;
        Fill(zero, 0U, kLlCountWireBlockBytes / sizeof(float));
        Sync<PIPE_V, PIPE_MTE3>();
        for (uint32_t logical = startStatusIndex_; logical < endStatusIndex_; ++logical) {
            const uint32_t expert = logical / epWorldSize_, source = logical % epWorldSize_;
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_, source, expert, directRanks_);
            // Remote frames retain dense-record cleanup. Local copies include
            // neighbouring source records, so reclaim their entire wire too:
            // a later shape may put a remote commit at any old record boundary.
            if (frame.flagWords == 1U)
                Store(stateGM_ + frame.recordOffset, zero, 32U);
            if (expert == frame.firstExpert) {
                if (frame.flagWords == 1U)
                    Store(stateGM_ + frame.commitOffset, zero, 32U);
                else
                    for (uint32_t block = 0; block < frame.wireBytes; block += kLlCountWireBlockBytes)
                        Store(stateGM_ + frame.frameOffset + block, zero, kLlCountWireBlockBytes);
            }
        }
        return;
    }
    Sync<PIPE_S, PIPE_V>();
    RecordReady(snapshot_, 0U, Align(recStatusNumPerCore_, 8U));
    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride, snapshot_,
          32U, recStatusNumPerCore_, kEpServerDedupCountGmStride - 32U);
    // The snapshot is no longer modified after this issue. Its DMA can overlap
    // cumsum; the caller retires all count-stage writes before receiver Reset.
}

// 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
__aicore__ inline void ServerDedupCounts::GatherSumRecvCnt()
{
    Sync<PIPE_S, PIPE_V>();
    GatherWords(dense_, snapshot_, recStatusNumPerCore_, 1);
    Sync<PIPE_V, PIPE_S>();

    // Compute the relative prefix while outgoing cnt may still be in flight.
    // Gather preserves integer count bits; no floating-point count arithmetic.
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
        cumulative += dense_[i];
        dense_[i] = cumulative;
    }
    // The final prefix is already the subtotal; keep its original wire bits.
    PublishRecvCntSum(cumulative);
}

// 摘要：将本核片段总数和就绪标记写入本 rank 的紧凑记分板；空片段也必须发布。
__aicore__ inline void ServerDedupCounts::PublishRecvCntSum(uint32_t sumOfRecvCnt)
{
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    // Compact scoreboard: one writer-owned 32B slot per receiver core, so
    // consumers read a single contiguous burst instead of a transposed row.
    for (uint32_t i = 0; i < UB_ALIGN_DATA_COUNT; ++i) sumSlot_[i] = 0U;
    sumSlot_[0] = sumOfRecvCnt;
    sumSlot_[1] = 0x3f800000U;
    Sync<PIPE_S, PIPE_MTE3>();
    Store(stateGM_ + layout_.cumsumOffset + newAivId * UB_ALIGN, sumSlot_, UB_ALIGN);
    // sumSlot_ stays immutable until CalRecvAndSetFlag reuses it. Polling
    // incoming sums must not wait here for this core's outgoing cnt.
}

// 摘要：等待所有接收核总数就绪，求本核之前各片段之和；槽位回收在后续屏障后执行。
__aicore__ inline uint32_t ServerDedupCounts::GetCumSum(uint32_t newAivId)
{
    uint32_t offset = 0;
    // Subtotals are contiguous 32B slots, so one burst covers every active core.
    Sync<PIPE_S, PIPE_V>();

    while (true) {
        Load(sumSnapshot_, stateGM_ + layout_.cumsumOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, activeCumSumCores_, 1);
        asc_sync_pipe(PIPE_V);
        Sum<true>(scalar_ + 8U, sumDense_, activeCumSumCores_);
        Sync<PIPE_V, PIPE_S>();
        if (*reinterpret_cast<__ubuf__ float*>(scalar_ + 8U) == float(activeCumSumCores_)) break;
    }
    if (newAivId != 0U) {
        Sync<PIPE_S, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, newAivId, 0);
        asc_sync_pipe(PIPE_V);
        Sum<false>(scalar_ + 8U, sumDense_, newAivId);
        Sync<PIPE_V, PIPE_S>();
        offset = scalar_[8U];
    }

    // The subtotal slots are reclaimed after WaitCumSumFlag, when every receiver
    // has already consumed them; no per-consumer column clear remains here.
    return offset;
}

// 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
__aicore__ inline void ServerDedupCounts::WaitDispatch()
{
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    LlCountReadPlan readPlan;
    if (packedFrames_ && !BuildCountReadPlan(rscvStatusNum_, epWorldSize_,
            useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_,
            startStatusIndex_, recStatusNumPerCore_, readPlan, directRanks_)) { trap(); return; }
    if (packedFrames_) {
        compareTarget = static_cast<float>(readPlan.expectedFlagSum);
    }

    Sync<PIPE_S, PIPE_V>();
    if (profile_) profile_->Mark<24>();  // w24: count_ready_poll_begin
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    uint32_t firstReadySeen = 0U, firstIterDone = 0U;
#endif
    while (sumOfFlag != compareTarget) {
        if (packedFrames_) {
            ReadCountFrames(snapshot_, stateGM_, readPlan, false);
        } else Load(snapshot_, stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride,
                    UB_ALIGN, recStatusNumPerCore_, kEpServerDedupCountGmStride - UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        if (packedFrames_) {
            Sum<true>(scalar_ + 8U, snapshot_, recStatusNumPerCore_ * UB_ALIGN_DATA_COUNT);
        } else {
            GatherWords(dense_, snapshot_, recStatusNumPerCore_, 0);
            asc_sync_pipe(PIPE_V);
            Sum<true>(scalar_ + 8U, dense_, recStatusNumPerCore_);
        }
        Sync<PIPE_V, PIPE_S>();
        sumOfFlag = *reinterpret_cast<__ubuf__ float*>(scalar_ + 8U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) {
            // Partial arrival plus the first full poll iteration separate "count
            // data is late on the fabric" from "one scan is expensive".
            if (!firstReadySeen && sumOfFlag != 0.0f) { profile_->Mark<46>(); firstReadySeen = 1U; }  // w46: recv_count_first_ready
            if (!firstIterDone) { profile_->Mark<47>(); firstIterDone = 1U; }  // w47: recv_count_poll_first_iter_end
        }
#endif
    }
    if (profile_) profile_->Mark<25>();  // w25: count_ready_poll_end_after_V_S

    if (packedFrames_) {
        // Commit acquired: load the actual dense records, preserving the old
        // logical expert/source ordering for prefix sums and Receiver metadata.
        ReadCountFrames(snapshot_, stateGM_, readPlan, true);
        Sync<PIPE_MTE2, PIPE_V>();
    }
    GatherSumRecvCnt();
    if (!packedFrames_) WaitDispatchClearStatus();
}

// 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与共享前缀，再发布完成标记。
__aicore__ inline void ServerDedupCounts::CalRecvAndSetFlag()
{
    if (recStatusNumPerCore_ == 0U) return;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 当前 UB 中是片段内前缀；先等待所有片段总数，再累加编号更小接收核的总数。
    // 这样得到全 rank 接收表中的基址，各核可独立修正自己负责的片段。
    if (profile_) profile_->Mark<26>();  // w26: cumsum_peer_join_begin
    const uint32_t offset = GetCumSum(newAivId);
    if (profile_) profile_->Mark<27>();  // w27: cumsum_peer_join_clear_issue_end
    if (packedFrames_) WaitDispatchClearStatus();

    if (recStatusNumPerCore_ != 0U) {
        for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
            dense_[i] += offset;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        // sendCountsOut 写的是累计前缀，而非单条记录的原始发送数量。
        Store(sendCountsOutGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));

        // Producers own disjoint fragments of one read-only GM prefix. All
        // receivers acquire the existing ready row before reading this copy.
        Store(recvCntWorkspaceGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));
    }

    // 发布前缀完成标记前，先等待旧片段总数发送不再读取 sumSlot_，
    // 再复用它构造标记；前缀写入与完成标记之间保留原 MTE3 顺序约束。
    // This is the first overwrite of the asynchronous local-sum publication.
    Sync<PIPE_MTE3, PIPE_V>();
    Fill(sumSlot_, 1U, Align(aivUsedCumSum_, 8U) * UB_ALIGN_DATA_COUNT);
    Sync<PIPE_V, PIPE_MTE3>();
    asc_sync_pipe(PIPE_MTE3);

    Store(stateGM_ + layout_.cumsumFlagOffset + newAivId * UB_ALIGN,
          sumSlot_, UB_ALIGN, aivUsedCumSum_, (aivUsedCumSum_ - 1U) * UB_ALIGN);
}

// 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
__aicore__ inline void ServerDedupCounts::WaitCumSumFlag()
{
    const uint32_t targetFlag = activeCumSumCores_ * UB_ALIGN_DATA_COUNT;
    const uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    uint32_t cumSumFlagOffset =
        layout_.cumsumFlagOffset + newAivId * aivUsedCumSum_ * UB_ALIGN;

    while (true) {
        Load(ready_, stateGM_ + cumSumFlagOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        Sum<false>(scalar_, ready_, targetFlag);
        Sync<PIPE_V, PIPE_S>();
        if (scalar_[0] == targetFlag) break;
    }
    Fill(clean_, 0U, targetFlag);
    Sync<PIPE_S, PIPE_MTE3>();

    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + cumSumFlagOffset, clean_, activeCumSumCores_ * UB_ALIGN);

    // This barrier opens only after every receiver has completed GetCumSum, so
    // each publisher may reclaim its own subtotal slot. The next use of this
    // bank then starts from zeros without an extra epoch tag.
    if (newAivId < activeCumSumCores_) {
        Store(stateGM_ + layout_.cumsumOffset + newAivId * UB_ALIGN,
              clean_ + newAivId * UB_ALIGN_DATA_COUNT, UB_ALIGN);
    }
}

}  // namespace DispatchDedup
#endif
