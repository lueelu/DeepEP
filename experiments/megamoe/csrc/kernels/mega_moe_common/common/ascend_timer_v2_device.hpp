// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H

#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include "kernels/mega_moe_common/common/ascend_timer_v2.hpp"

class DEEPEP_MEGAMOE_TIMER_DEVICE {
public:
    __aicore__ inline void Init(GM_ADDR timerAddr)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        timerAddr_ = timerAddr;
#else
#ifdef DEEPEP_MEGAMOE_ROCE
        (void)timerAddr;
#endif
        timerAddr_ = nullptr;
        coreId_ = -1;
        return;
#endif
        if (timerAddr_ == nullptr) {
            coreId_ = -1;
            return;
        }
        waveId_ = activeWaveId_ = DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE;
        activeType_ = activeIndex_ = waitType_ = waitIndex_ = -1;
        expert_ = -1;
        tileActive_ = false;
#ifdef DEEPEP_MEGAMOE_ROCE
        startupActive_ = false;
#endif
        int blockId = AscendC::GetBlockIdx();
        int taskRatio = AscendC::GetTaskRation();
        if (taskRatio <= 0) {
            coreId_ = -1;
            return;
        }
        // A MIX_AIC_1_2 launch has one AIC and two AIV lanes per logical
        // group.  GetTaskRation() is engine-local: it is 1 on AIC and 2 on
        // AIV, so deriving the group width from it maps AIC block 1 to slot 2
        // (which the host decodes as group 0 / AIV1).  Keep the published
        // [AIC, AIV0, AIV1] layout independent of the current engine.
        constexpr int coresPerGroup = 3;
        if ASCEND_IS_AIV {
            int groupId = blockId / taskRatio;
            coreId_ = groupId * coresPerGroup + AscendC::GetSubBlockIdx() + 1;
        } else {
            coreId_ = blockId * coresPerGroup;
        }
        for (int typeId = 0; typeId < DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT; ++typeId) {
            dynamicCounts_[typeId] = 0;
#ifdef DEEPEP_MEGAMOE_ROCE
        }
        if (Enabled()) {
            WriteValue(DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_DIAGNOSTIC_OFFSET + coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_DIAGNOSTIC_COUNT,
                       0);  // A later launch may take a non-snapshot count path.
        }
#else
        }
#endif
    }

    __aicore__ inline bool Enabled() const
    {
        return timerAddr_ != nullptr && coreId_ >= 0 && coreId_ < DEEPEP_MEGAMOE_TIMER_NAMESPACE::N_CORE_COUNT;
    }

    // 由调度层指定实际服务的 Wave，不能从本核任务序号反推。
    __aicore__ inline void SetWave(int64_t waveId)
    {
        if (Enabled()) {
            waveId_ = waveId;
        }
    }

    __aicore__ inline void SetSlice(int expert, uint32_t rowStart, uint32_t rows)
    {
        if (!Enabled()) {
            return;
        }
        expert_ = expert;
        sliceRow_ = rowStart;
        rowStart_ = rowStart;
        rows_ = rows;
        nTile_ = -1;
    }

    __aicore__ inline void ClearSlice()
    {
        if (Enabled()) {
            expert_ = -1;
        }
    }

    // 起点复用原有同步；Combine 结束点等待本核流水完成，计入异步搬运耗时。
    __aicore__ inline bool StartTile(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, uint32_t m, uint32_t n,
                                     uint32_t rows, uint32_t tileN = 256U)
    {
        if (!Enabled() || expert_ < 0) {
            return false;
#ifdef DEEPEP_MEGAMOE_ROCE
        }
        if (type == DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIC_GMM1) {
            EndStartup();
#endif
        }
        tileActive_ = true;
        rowStart_ = sliceRow_ + m;
        rows_ = rows;
        // 完整 token 行没有 N tile；沿用 -1 元数据，解析端不把它误标为第 0 列块。
        // Use routed logical N128 or the selected shared N192/N224/N256 width.
        nTile_ = n == static_cast<uint32_t>(-1) ? -1 : static_cast<int>(n / tileN);
        activeType_ = static_cast<int>(type);
        activeWaveId_ = waveId_;
        activeIndex_ = BeginRaw(activeType_, AscendC::GetSystemCycle());
        return true;
    }

    __aicore__ inline void EndTile()
    {
        if (!Enabled() || !tileActive_) {
            return;
        }
        if (activeType_ == static_cast<int>(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE)) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = activeIndex_ = -1;
        tileActive_ = false;
    }

    __aicore__ inline void Start(DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteWave(timingId, DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE);
        WriteStart(timingId, AscendC::GetSystemCycle());
    }

    __aicore__ inline void End(DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteEnd(timingId, AscendC::GetSystemCycle());
    }

#ifdef DEEPEP_MEGAMOE_ROCE
    // Observe existing synchronization without adding a pipeline drain. Fixed
    // slots are used only once per invocation and do not touch tile/span state.
    __aicore__ inline void StartUnfenced(DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type)
    {
        if (Enabled()) {
            WriteWave(static_cast<int>(type), DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE);
            WriteStart(static_cast<int>(type), AscendC::GetSystemCycle());
        }
    }

    __aicore__ inline void EndUnfenced(DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type)
    {
        if (Enabled()) {
            WriteEnd(static_cast<int>(type), AscendC::GetSystemCycle());
        }
    }

    __aicore__ inline void BeginStartup()
    {
        if (Enabled()) {
            startupActive_ = true;
            StartUnfenced(coreId_ % 3 == 0 ? DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIC_MOE_STARTUP : DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_MOE_STARTUP);
        }
    }

    __aicore__ inline void EndStartup()
    {
        if (Enabled() && startupActive_) {
            EndUnfenced(coreId_ % 3 == 0 ? DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIC_MOE_STARTUP : DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_MOE_STARTUP);
            startupActive_ = false;
        }
    }

#endif
    // 每次 slice 独立记录；依赖等待会把当前执行段拆成真实的前后两段。
    __aicore__ inline void StartSpan(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
    {
        if (!Enabled()) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#ifdef DEEPEP_MEGAMOE_ROCE
        if (type == DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_DISPATCH || type == DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_GMM1_PROLOGUE) {
            EndStartup();
        }
#endif
        activeType_ = static_cast<int>(type);
        activeWaveId_ = waveId_;
        activeIndex_ = BeginRaw(activeType_, AscendC::GetSystemCycle());
    }

    __aicore__ inline void EndSpan(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
    {
        if (!Enabled() || activeType_ != static_cast<int>(type)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = -1;
        activeIndex_ = -1;
    }

    __aicore__ inline bool BeginWait(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
    {
        if (!Enabled() || activeType_ < 0 || waitType_ >= 0) {
            return false;
        }
        // GM flag 轮询已通过原有同步确认数据就绪；此处不新增 PIPE_ALL，
        // 避免把其他流水排空时间误计为数据依赖等待。
        int64_t cycle = AscendC::GetSystemCycle();
        waitType_ = static_cast<int>(type);
        if (tileActive_) {
            // 一个 tile 最多一次输入等待；无等待的 tile 在 WAIT 数组中保留空槽。
            waitIndex_ = activeIndex_;
            dynamicCounts_[waitType_] = dynamicCounts_[activeType_];
            if (waitIndex_ >= 0) {
                WriteMetadata(waitType_, waitIndex_);
                WriteWave(DynamicTimingIndex(waitType_, waitIndex_), activeWaveId_);
                WriteStart(DynamicTimingIndex(waitType_, waitIndex_), cycle);
            }
        } else {
            EndRaw(activeType_, activeIndex_, cycle);
            waitIndex_ = BeginRaw(waitType_, cycle);
        }
        return true;
    }

    __aicore__ inline void EndWait()
    {
        if (!Enabled() || waitType_ < 0) {
            return;
        }
        int64_t cycle = AscendC::GetSystemCycle();
        EndRaw(waitType_, waitIndex_, cycle);
        waitType_ = -1;
        waitIndex_ = -1;
        if (tileActive_) {
            // 覆盖等待前的起点，不生成“准备工作冒充 tile”的区间。
            if (activeIndex_ >= 0) {
                WriteStart(DynamicTimingIndex(activeType_, activeIndex_), cycle);
            }
        } else {
            activeIndex_ = BeginRaw(activeType_, cycle);
        }
    }

#ifdef DEEPEP_MEGAMOE_ROCE
    // RDMA may progress inside a compute tile/span. Keep its handle independent
    // of activeType_/waitType_ and never drain a pipeline just for timestamps.
    __aicore__ inline int BeginRdma(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, DEEPEP_MEGAMOE_TIMER_NAMESPACE::RdmaPhase phase, uint32_t peer,
                                    uint32_t bytes, bool isCommit, uint32_t qp = 0U, uint32_t rowStart = 0U,
                                    uint32_t rowCount = 0U)
    {
        return BeginIndependent(
            type, (static_cast<uint64_t>(peer + 1) << 32) | (rowCount << 8) | (static_cast<uint32_t>(phase) + 1),
            (static_cast<uint64_t>(rowStart) << 48) | (static_cast<uint64_t>(qp) << 40) |
                (static_cast<uint64_t>(isCommit) << 32) | bytes);
    }

    __aicore__ inline void EndRdma(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, int index)
    {
        if (Enabled() && index >= 0) {
            EndRaw(static_cast<int>(type), index, AscendC::GetSystemCycle());
        }
    }

    __aicore__ inline int BeginGatewayWait(uint32_t source, uint32_t gateway, uint32_t sequence)
    {
        return BeginIndependent(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_GATEWAY_PAYLOAD_WAIT,
                                (static_cast<uint64_t>(source + 1) << 32) | (gateway + 1), sequence);
    }

    __aicore__ inline void EndGatewayWait(int index)
    {
        if (Enabled() && index >= 0) {
            EndRaw(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_GATEWAY_PAYLOAD_WAIT, index, AscendC::GetSystemCycle());
        }
    }

    // Independent row handles retain the originating wave across expert/wave
    // boundaries. Lifetime ends when software observes retirement, not at a
    // hardware-exact DMA timestamp. No timer-only synchronization is added.
    __aicore__ inline int BeginCombineRow(uint32_t expert, uint32_t row, uint32_t slot,
                                          DEEPEP_MEGAMOE_TIMER_NAMESPACE::CombineRowPhase phase)
    {
        return BeginCombineRowAtWave(expert, row, slot, phase, waveId_);
    }

    __aicore__ inline int64_t CurrentWave() const
    {
        return waveId_;
    }

    __aicore__ inline int BeginCombineRowAtWave(uint32_t expert, uint32_t row, uint32_t slot,
                                                DEEPEP_MEGAMOE_TIMER_NAMESPACE::CombineRowPhase phase, int64_t wave)
    {
        return BeginIndependent(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_ASYNC, (static_cast<uint64_t>(expert + 1U) << 32) | row,
                                (static_cast<uint64_t>(slot + 1U) << 32) | static_cast<uint32_t>(phase), wave);
    }

    __aicore__ inline void EndCombineRow(int index)
    {
        if (Enabled() && index >= 0) {
            EndRaw(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_ASYNC, index, AscendC::GetSystemCycle());
        }
    }

    // Progress can surround RDMA calls while a compute tile/span or wait is
    // active. Capture the scheduler's wave without changing any active handle.
    __aicore__ inline int BeginCombineProgress(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
    {
        if (type != DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_PROGRESS && type != DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_DRAIN) {
            return -1;
        }
        return BeginIndependent(type, 0, 0, waveId_);
    }

    __aicore__ inline void EndCombineProgress(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, int index, uint32_t pendingBefore,
                                              uint32_t pendingAfter, uint32_t posts, uint32_t scannedRows,
                                              uint64_t scanControlCycles)
    {
        if (!Enabled() || index < 0 || index >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER ||
            (type != DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_PROGRESS && type != DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_DRAIN)) {
            return;
        }
        int64_t cycle = AscendC::GetSystemCycle();
        if (type == DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_DRAIN) {
            posts = scannedRows = 0;
            scanControlCycles = 0;
        }
        // Scheduler counters are bounded by 1024. Masks keep the wire fields
        // independent even if a caller supplies an invalid wider value.
        uint64_t counts =
            static_cast<uint64_t>(pendingBefore & 0xffffU) | (static_cast<uint64_t>(pendingAfter & 0xffffU) << 16) |
            (static_cast<uint64_t>(posts & 0xffffU) << 32) | (static_cast<uint64_t>(scannedRows & 0xffffU) << 48);
        int typeId = static_cast<int>(type);
        int offset = DEEPEP_MEGAMOE_TIMER_NAMESPACE::TASK_METADATA_OFFSET +
                     ((coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT + typeId) * DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER + index) * 2;
        WriteValue(offset, static_cast<int64_t>(counts));
        WriteValue(offset + 1, static_cast<int64_t>(scanControlCycles));
        EndRaw(typeId, index, cycle);
    }

    // Deferred writes keep detailed timing out of active tile/wait state and
    // add no synchronization. The caller samples only existing phase edges.
    __aicore__ inline void RecordDetail(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, int64_t start, int64_t end, uint64_t first,
                                        uint64_t second)
    {
        if (!Enabled() || end <= start) {
            return;
        }
        const int typeId = static_cast<int>(type);
        const int index = dynamicCounts_[typeId]++;
        if (index >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER) {
            return;
        }
        const int offset =
            DEEPEP_MEGAMOE_TIMER_NAMESPACE::TASK_METADATA_OFFSET +
            ((coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT + typeId) * DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER + index) * 2;
        WriteValue(offset, static_cast<int64_t>(first));
        WriteValue(offset + 1, static_cast<int64_t>(second));
        const int timing = DynamicTimingIndex(typeId, index);
        WriteWave(timing, waveId_);
        WriteStart(timing, start);
        WriteEnd(timing, end);
    }

    __aicore__ inline GM_ADDR CountSnapshotAddress() const
    {
        return timerAddr_ + sizeof(int64_t) * (DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_SNAPSHOT_OFFSET +
                                               coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_SNAPSHOT_SLOTS_PER_CORE);
    }

    __aicore__ inline void WriteCountDiagnostics(const uint64_t* values)
    {
        if (!Enabled()) {
            return;
        }
        for (int i = 0; i < DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_DIAGNOSTIC_COUNT; ++i) {
            WriteValue(DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_DIAGNOSTIC_OFFSET + coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::COUNT_DIAGNOSTIC_COUNT + i,
                       static_cast<int64_t>(values[i]));
        }
    }

#endif
    __aicore__ inline void FlushDynamicCounts()
    {
        if (!Enabled()) {
            return;
        }
        for (int typeId = 0; typeId < DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT; ++typeId) {
            int32_t index = DEEPEP_MEGAMOE_TIMER_NAMESPACE::N_TIMING_COUNTER +
                            coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_ITER_PER_CORE_ALIGN + typeId;
            WriteValue(index, dynamicCounts_[typeId]);
        }
    }

private:
    GM_ADDR timerAddr_ = nullptr;
    int coreId_ = -1;
    int64_t waveId_ = DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE;
    int64_t activeWaveId_ = DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE;
    int dynamicCounts_[DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT]{};
    int activeType_ = -1;
    int activeIndex_ = -1;
    int waitType_ = -1;
    int waitIndex_ = -1;
    bool tileActive_ = false;
#ifdef DEEPEP_MEGAMOE_ROCE
    bool startupActive_ = false;
#endif
    int expert_ = -1;
    int nTile_ = -1;
    uint32_t sliceRow_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rows_ = 0;
#ifdef DEEPEP_MEGAMOE_ROCE

    __aicore__ inline int BeginIndependent(DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type, uint64_t first, uint64_t second,
                                           int64_t waveId = DEEPEP_MEGAMOE_TIMER_NAMESPACE::NO_WAVE)
    {
        if (!Enabled()) {
            return -1;
        }
        int typeId = static_cast<int>(type);
        int index = dynamicCounts_[typeId]++;
        if (index >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER) {
            return -1;
        }
        int offset = DEEPEP_MEGAMOE_TIMER_NAMESPACE::TASK_METADATA_OFFSET +
                     ((coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT + typeId) * DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER + index) * 2;
        WriteValue(offset, static_cast<int64_t>(first));
        WriteValue(offset + 1, static_cast<int64_t>(second));
        WriteWave(DynamicTimingIndex(typeId, index), waveId);
        WriteStart(DynamicTimingIndex(typeId, index), AscendC::GetSystemCycle());
        return index;
    }
#endif

    __aicore__ inline void WriteMetadata(int typeId, int index)
    {
        if (expert_ < 0) {
            return;
        }
        int offset = DEEPEP_MEGAMOE_TIMER_NAMESPACE::TASK_METADATA_OFFSET +
                     ((coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::DYNAMIC_TYPE_COUNT + typeId) *
                          DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER +
                      index) *
                         2;
        WriteValue(offset, static_cast<int64_t>((static_cast<uint64_t>(expert_ + 1) << 32) | rowStart_));
        WriteValue(offset + 1, static_cast<int64_t>((static_cast<uint64_t>(nTile_ + 1) << 32) | rows_));
    }

    __aicore__ inline int BeginRaw(int typeId, int64_t cycle)
    {
        int index = dynamicCounts_[typeId]++;
        // 即使溢出也保留实际计数，主机必须报错，不能静默输出截断流水。
        if (index >= DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER) {
            return -1;
        }
        WriteMetadata(typeId, index);
        WriteWave(DynamicTimingIndex(typeId, index), activeWaveId_);
        WriteStart(DynamicTimingIndex(typeId, index), cycle);
        return index;
    }

    __aicore__ inline void EndRaw(int typeId, int index, int64_t cycle)
    {
        if (index >= 0) {
            WriteEnd(DynamicTimingIndex(typeId, index), cycle);
        }
    }

    __aicore__ inline int DynamicTimingIndex(int typeId, int taskIndex) const
    {
        return DEEPEP_MEGAMOE_TIMER_NAMESPACE::FIXED_TIMING_COUNT +
               typeId * DEEPEP_MEGAMOE_TIMER_NAMESPACE::MAX_DYNAMIC_ITER + taskIndex;
    }

    __aicore__ inline int32_t BufferIndex(int timingId) const
    {
        return coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::N_TIMING_COUNTER_PER_CORE_ALIGN + timingId * 2;
    }

    __aicore__ inline void WriteValue(int32_t index, int64_t value) const
    {
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(timerAddr_));
        timer(index) = value;
    }

    __aicore__ inline void WriteWave(int timingId, int64_t waveId)
    {
        WriteValue(DEEPEP_MEGAMOE_TIMER_NAMESPACE::WAVE_ID_OFFSET +
                       coreId_ * DEEPEP_MEGAMOE_TIMER_NAMESPACE::WAVE_ID_PER_CORE_ALIGN + timingId,
                   waveId);
    }

    __aicore__ inline void WriteStart(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < DEEPEP_MEGAMOE_TIMER_NAMESPACE::N_TIMING_COUNTER) {
            WriteValue(index, cycle);
        }
    }

    __aicore__ inline void WriteEnd(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < DEEPEP_MEGAMOE_TIMER_NAMESPACE::N_TIMING_COUNTER) {
            WriteValue(index + 1, cycle);
        }
    }
};

#ifdef DEEPEP_MEGAMOE_ROCE
class AscendTimerRelayScope {
public:
    __aicore__ inline AscendTimerRelayScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer, DEEPEP_MEGAMOE_TIMER_NAMESPACE::RelayStep step, uint32_t source,
                                            uint32_t firstExpert, uint32_t expertCount)
        : timer_(timer && timer->Enabled() ? timer : nullptr),
          first_((static_cast<uint64_t>(step) << 32) | (source + 1U)),
          second_((static_cast<uint64_t>(firstExpert) << 32) | expertCount)
    {
        if (timer_) {
            start_ = AscendC::GetSystemCycle();
        }
    }
    __aicore__ inline ~AscendTimerRelayScope()
    {
        if (timer_) {
            const int64_t end = AscendC::GetSystemCycle();
            timer_->RecordDetail(DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_RELAY_STEP, start_, end, first_, second_);
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    uint64_t first_;
    uint64_t second_;
    int64_t start_ = 0;
};

class AscendTimerFixedScope {
public:
    __aicore__ inline AscendTimerFixedScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer, DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type)
        : timer_(timer), type_(type)
    {
        if (timer_ != nullptr) {
            timer_->StartUnfenced(type_);
        }
    }
    __aicore__ inline ~AscendTimerFixedScope()
    {
        if (timer_ != nullptr) {
            timer_->EndUnfenced(type_);
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    DEEPEP_MEGAMOE_TIMER_NAMESPACE::FixedTiming type_;
};

// The first routed Dispatch/prologue/tile closes this envelope. Empty-work
// cores close it at stage exit, so they retain a visible startup interval too.
class AscendTimerStartupScope {
public:
    __aicore__ inline explicit AscendTimerStartupScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer) : timer_(timer)
    {
        if (timer_ != nullptr) {
            timer_->BeginStartup();
        }
    }
    __aicore__ inline ~AscendTimerStartupScope()
    {
        if (timer_ != nullptr) {
            timer_->EndStartup();
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
};

class AscendTimerGatewayWaitScope {
public:
    __aicore__ inline AscendTimerGatewayWaitScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer, uint32_t source, uint32_t gateway,
                                                  uint32_t sequence)
        : timer_(timer), index_(timer ? timer->BeginGatewayWait(source, gateway, sequence) : -1)
    {
    }
    __aicore__ inline ~AscendTimerGatewayWaitScope()
    {
        if (timer_ != nullptr) {
            timer_->EndGatewayWait(index_);
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    int index_;
};

class AscendTimerRdmaScope {
public:
    __aicore__ inline AscendTimerRdmaScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer, DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type,
                                           DEEPEP_MEGAMOE_TIMER_NAMESPACE::RdmaPhase phase, uint32_t peer, uint32_t bytes, bool isCommit,
                                           uint32_t qp = 0U, uint32_t rowStart = 0U, uint32_t rowCount = 0U)
        : timer_(timer),
          type_(type),
          index_(timer ? timer->BeginRdma(type, phase, peer, bytes, isCommit, qp, rowStart, rowCount) : -1)
    {
    }
    __aicore__ inline ~AscendTimerRdmaScope()
    {
        if (timer_ != nullptr) {
            timer_->EndRdma(type_, index_);
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type_;
    int index_;
};

// Bounded progress constructs after its separately timed initialization. Fused
// route-run completion includes route discovery in this scope. Progress covers
// scan/post work; drain surrounds child progress calls and backoff. Scan
// segments exclude RocePutBytes and paused backoff. Each segment includes the
// call-boundary and timestamp overhead between its StartScan and StopScan cycle
// reads; bookkeeping after the StopScan read and work while paused are excluded.
// Timing never determines whether scheduler work executes, and adds no barrier.
class AscendTimerCombineProgressScope {
public:
    __aicore__ inline AscendTimerCombineProgressScope(DEEPEP_MEGAMOE_TIMER_DEVICE* timer, DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
        : timer_(timer), type_(type), index_(timer ? timer->BeginCombineProgress(type) : -1)
    {
    }
    __aicore__ inline ~AscendTimerCombineProgressScope()
    {
        // Early exits still close the envelope. Normal callers supply counters
        // explicitly; no scheduler state is read or changed by this fallback.
        Finish(0, 0, 0, 0);
    }

    __aicore__ inline void StartScan()
    {
        if (index_ >= 0 && timer_->Enabled() && type_ == DEEPEP_MEGAMOE_TIMER_NAMESPACE::AIV_COMBINE_PROGRESS && !scanActive_) {
            scanStart_ = AscendC::GetSystemCycle();
            scanActive_ = true;
        }
    }

    __aicore__ inline void StopScan()
    {
        if (index_ >= 0 && timer_->Enabled() && scanActive_) {
            scanControlCycles_ += static_cast<uint64_t>(AscendC::GetSystemCycle()) - scanStart_;
            scanActive_ = false;
        }
    }

    __aicore__ inline void Finish(uint32_t pendingBefore, uint32_t pendingAfter, uint32_t posts, uint32_t scannedRows)
    {
        if (index_ < 0) {
            return;
        }
        StopScan();
        timer_->EndCombineProgress(type_, index_, pendingBefore, pendingAfter, posts, scannedRows, scanControlCycles_);
        index_ = -1;
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type_;
    int index_;
    bool scanActive_ = false;
    uint64_t scanStart_ = 0;
    uint64_t scanControlCycles_ = 0;
};

#endif
class DEEPEP_MEGAMOE_TIMER_TILE_SCOPE {
public:
    __aicore__ inline DEEPEP_MEGAMOE_TIMER_TILE_SCOPE(DEEPEP_MEGAMOE_TIMER_DEVICE* timer,
                                                      DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type,
                                                      uint32_t m, uint32_t n, uint32_t rows, uint32_t tileN = 256U)
        : timer_(timer != nullptr && timer->StartTile(type, m, n, rows, tileN) ? timer : nullptr)
    {
    }
    __aicore__ inline ~DEEPEP_MEGAMOE_TIMER_TILE_SCOPE()
    {
        if (timer_ != nullptr) {
            timer_->EndTile();
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
};

// Prologue is a per-expert interval; keep the existing span completion semantics.
// Declare before prologue resources so their cleanup completes before EndSpan.
class DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE {
public:
    __aicore__ inline DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE(DEEPEP_MEGAMOE_TIMER_DEVICE* timer,
                                                      DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
        : timer_(timer), type_(type)
    {
        if (timer_ != nullptr) {
            timer_->StartSpan(type_);
        }
    }
    __aicore__ inline ~DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE()
    {
        if (timer_ != nullptr) {
            timer_->EndSpan(type_);
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
    DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type_;
};

// 作用域只包围已有的依赖等待，不改变等待条件和 flag 发布顺序。
class DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE {
public:
    __aicore__ inline DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE(DEEPEP_MEGAMOE_TIMER_DEVICE* timer,
                                                      DEEPEP_MEGAMOE_TIMER_NAMESPACE::DynamicTimingType type)
        : timer_(timer != nullptr && timer->BeginWait(type) ? timer : nullptr)
    {
    }
    __aicore__ inline ~DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE()
    {
        if (timer_ != nullptr) {
            timer_->EndWait();
        }
    }

private:
    DEEPEP_MEGAMOE_TIMER_DEVICE* timer_;
};

#endif  // ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
