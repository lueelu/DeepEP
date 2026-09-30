// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H

#include "ascend_timer_v2.hpp"

class AscendTimerDevice {
public:
    __aicore__ inline void Init(GM_ADDR timerAddr)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        timerAddr_ = timerAddr;
#else
        timerAddr_ = nullptr;
        coreId_ = -1;
        return;
#endif
        if (timerAddr_ == nullptr) {
            coreId_ = -1;
            return;
        }
        waveId_ = activeWaveId_ = AscendTimer::NO_WAVE;
        activeType_ = activeIndex_ = waitType_ = waitIndex_ = -1;
        expert_ = -1;
        tileActive_ = false;
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
        for (int typeId = 0; typeId < AscendTimer::DYNAMIC_TYPE_COUNT; ++typeId) {
            dynamicCounts_[typeId] = 0;
        }
    }

    __aicore__ inline bool Enabled() const
    {
        return timerAddr_ != nullptr && coreId_ >= 0 && coreId_ < AscendTimer::N_CORE_COUNT;
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
    __aicore__ inline bool StartTile(AscendTimer::DynamicTimingType type, uint32_t m, uint32_t n, uint32_t rows,
                                     uint32_t tileN = 256U)
    {
        if (!Enabled() || expert_ < 0) {
            return false;
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
        if (activeType_ == static_cast<int>(AscendTimer::AIV_COMBINE)) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = activeIndex_ = -1;
        tileActive_ = false;
    }

    __aicore__ inline void Start(AscendTimer::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= AscendTimer::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteWave(timingId, AscendTimer::NO_WAVE);
        WriteStart(timingId, AscendC::GetSystemCycle());
    }

    __aicore__ inline void End(AscendTimer::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= AscendTimer::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteEnd(timingId, AscendC::GetSystemCycle());
    }

    // 每次 slice 独立记录；依赖等待会把当前执行段拆成真实的前后两段。
    __aicore__ inline void StartSpan(AscendTimer::DynamicTimingType type)
    {
        if (!Enabled()) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        activeType_ = static_cast<int>(type);
        activeWaveId_ = waveId_;
        activeIndex_ = BeginRaw(activeType_, AscendC::GetSystemCycle());
    }

    __aicore__ inline void EndSpan(AscendTimer::DynamicTimingType type)
    {
        if (!Enabled() || activeType_ != static_cast<int>(type)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = -1;
        activeIndex_ = -1;
    }

    __aicore__ inline bool BeginWait(AscendTimer::DynamicTimingType type)
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

    __aicore__ inline void FlushDynamicCounts()
    {
        if (!Enabled()) {
            return;
        }
        for (int typeId = 0; typeId < AscendTimer::DYNAMIC_TYPE_COUNT; ++typeId) {
            int32_t index = AscendTimer::N_TIMING_COUNTER + coreId_ * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + typeId;
            WriteValue(index, dynamicCounts_[typeId]);
        }
    }

private:
    GM_ADDR timerAddr_ = nullptr;
    int coreId_ = -1;
    int64_t waveId_ = AscendTimer::NO_WAVE;
    int64_t activeWaveId_ = AscendTimer::NO_WAVE;
    int dynamicCounts_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    int activeType_ = -1;
    int activeIndex_ = -1;
    int waitType_ = -1;
    int waitIndex_ = -1;
    bool tileActive_ = false;
    int expert_ = -1;
    int nTile_ = -1;
    uint32_t sliceRow_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rows_ = 0;

    __aicore__ inline void WriteMetadata(int typeId, int index)
    {
        if (expert_ < 0) {
            return;
        }
        int offset = AscendTimer::TASK_METADATA_OFFSET +
                     ((coreId_ * AscendTimer::DYNAMIC_TYPE_COUNT + typeId) * AscendTimer::MAX_DYNAMIC_ITER + index) * 2;
        WriteValue(offset, static_cast<int64_t>((static_cast<uint64_t>(expert_ + 1) << 32) | rowStart_));
        WriteValue(offset + 1, static_cast<int64_t>((static_cast<uint64_t>(nTile_ + 1) << 32) | rows_));
    }

    __aicore__ inline int BeginRaw(int typeId, int64_t cycle)
    {
        int index = dynamicCounts_[typeId]++;
        // 即使溢出也保留实际计数，主机必须报错，不能静默输出截断流水。
        if (index >= AscendTimer::MAX_DYNAMIC_ITER) {
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
        return AscendTimer::FIXED_TIMING_COUNT + typeId * AscendTimer::MAX_DYNAMIC_ITER + taskIndex;
    }

    __aicore__ inline int32_t BufferIndex(int timingId) const
    {
        return coreId_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN + timingId * 2;
    }

    __aicore__ inline void WriteValue(int32_t index, int64_t value) const
    {
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(timerAddr_));
        timer(index) = value;
    }

    __aicore__ inline void WriteWave(int timingId, int64_t waveId)
    {
        WriteValue(AscendTimer::WAVE_ID_OFFSET + coreId_ * AscendTimer::WAVE_ID_PER_CORE_ALIGN + timingId, waveId);
    }

    __aicore__ inline void WriteStart(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < AscendTimer::N_TIMING_COUNTER) {
            WriteValue(index, cycle);
        }
    }

    __aicore__ inline void WriteEnd(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < AscendTimer::N_TIMING_COUNTER) {
            WriteValue(index + 1, cycle);
        }
    }
};

class AscendTimerTileScope {
public:
    __aicore__ inline AscendTimerTileScope(AscendTimerDevice* timer, AscendTimer::DynamicTimingType type, uint32_t m,
                                           uint32_t n, uint32_t rows, uint32_t tileN = 256U)
        : timer_(timer != nullptr && timer->StartTile(type, m, n, rows, tileN) ? timer : nullptr)
    {
    }
    __aicore__ inline ~AscendTimerTileScope()
    {
        if (timer_ != nullptr) {
            timer_->EndTile();
        }
    }

private:
    AscendTimerDevice* timer_;
};

// Prologue is a per-expert interval; keep the existing span completion semantics.
// Declare before prologue resources so their cleanup completes before EndSpan.
class AscendTimerSpanScope {
public:
    __aicore__ inline AscendTimerSpanScope(AscendTimerDevice* timer, AscendTimer::DynamicTimingType type)
        : timer_(timer), type_(type)
    {
        if (timer_ != nullptr) {
            timer_->StartSpan(type_);
        }
    }
    __aicore__ inline ~AscendTimerSpanScope()
    {
        if (timer_ != nullptr) {
            timer_->EndSpan(type_);
        }
    }

private:
    AscendTimerDevice* timer_;
    AscendTimer::DynamicTimingType type_;
};

// 作用域只包围已有的依赖等待，不改变等待条件和 flag 发布顺序。
class AscendTimerWaitScope {
public:
    __aicore__ inline AscendTimerWaitScope(AscendTimerDevice* timer, AscendTimer::DynamicTimingType type)
        : timer_(timer != nullptr && timer->BeginWait(type) ? timer : nullptr)
    {
    }
    __aicore__ inline ~AscendTimerWaitScope()
    {
        if (timer_ != nullptr) {
            timer_->EndWait();
        }
    }

private:
    AscendTimerDevice* timer_;
};

#endif  // ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
