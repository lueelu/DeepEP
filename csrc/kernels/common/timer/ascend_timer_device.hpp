// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef DEEPEP_ASCEND_TIMER_DEVICE_HPP
#define DEEPEP_ASCEND_TIMER_DEVICE_HPP

#include <stdint.h>

template <class Schema>
class BasicAscendTimerDevice {
public:
    using SchemaType = Schema;
    using DynamicTimingType = typename Schema::DynamicTimingType;
    using FixedTiming = typename Schema::FixedTiming;

    __aicore__ inline void Init(GM_ADDR timerAddr, int cores = Schema::N_CORE_COUNT,
                                int capacity = Schema::MAX_DYNAMIC_ITER)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        timerAddr_ = timerAddr;
        coreCount_ = cores;
        capacity_ = capacity;
#else
        timerAddr_ = nullptr;
        coreId_ = -1;
        return;
#endif
        if (timerAddr_ == nullptr) {
            coreId_ = -1;
            return;
        }
        waveId_ = activeWaveId_ = Schema::NO_WAVE;
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
        for (int typeId = 0; typeId < Schema::DYNAMIC_TYPE_COUNT; ++typeId) {
            dynamicCounts_[typeId] = 0;
        }
    }

    __aicore__ inline bool Enabled() const
    {
#ifndef DEEPEP_MEGAMOE_TIMER
        return false;
#else
        return timerAddr_ != nullptr && coreId_ >= 0 && coreId_ < coreCount_;
#endif
    }

    __aicore__ inline void SetTaskShape(uint32_t tileM, uint32_t tileN, uint32_t kStart = 0, uint32_t kCount = 0,
                                        uint32_t nCount = 0)
    {
        if (!Enabled()) return;
        tileM_ = tileM;
        tileN_ = tileN;
        kStart_ = kStart;
        kCount_ = kCount;
        nCount_ = nCount;
    }
    __aicore__ inline void SetTaskContext(int source, uint32_t flags = 0)
    {
        if (!Enabled()) return;
        sourceRank_ = source;
        taskFlags_ = flags;
    }

    __aicore__ inline int BeginInterval(typename Schema::DynamicTimingType type)
    {
        if (!Enabled()) return -1;
        AscendC::PipeBarrier<PIPE_ALL>();
        return BeginRaw(static_cast<int>(type), AscendC::GetSystemCycle());
    }
    __aicore__ inline void EndInterval(typename Schema::DynamicTimingType type, int index)
    {
        if (!Enabled() || index < 0) return;
        AscendC::PipeBarrier<PIPE_ALL>();
        EndRaw(static_cast<int>(type), index, AscendC::GetSystemCycle());
    }

    // Observe native issue/completion boundaries without inserting a pipeline drain.
    __aicore__ inline int BeginPassiveInterval(typename Schema::DynamicTimingType type, uint32_t n, uint32_t tileN)
    {
        if (!Enabled()) return -1;
        nStart_ = n;
        nTile_ = n == static_cast<uint32_t>(-1) ? -1 : static_cast<int>(n / tileN);
        return BeginRaw(static_cast<int>(type), AscendC::GetSystemCycle());
    }
    __aicore__ inline void EndPassiveInterval(typename Schema::DynamicTimingType type, int index)
    {
        if (Enabled() && index >= 0) EndRaw(static_cast<int>(type), index, AscendC::GetSystemCycle());
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
        if constexpr (Schema::kMetadataWords > 2) {
            tileM_ = tileN_ = kStart_ = kCount_ = nStart_ = nCount_ = 0;
            sourceRank_ = -1;
            taskFlags_ = 0;
        }
    }

    __aicore__ inline void ClearSlice()
    {
        if (Enabled()) {
            expert_ = -1;
        }
    }

    // 起点复用原有同步；Combine 结束点等待本核流水完成，计入异步搬运耗时。
    __aicore__ inline bool StartTile(typename Schema::DynamicTimingType type, uint32_t m, uint32_t n, uint32_t rows,
                                     uint32_t tileN = 256U)
    {
        if (!Enabled() || expert_ < 0) {
            return false;
        }
        tileActive_ = true;
        rowStart_ = sliceRow_ + m;
        rows_ = rows;
        nStart_ = n;
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
        if (Schema::DrainTile(activeType_)) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = activeIndex_ = -1;
        tileActive_ = false;
    }

    __aicore__ inline void Start(typename Schema::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= Schema::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteWave(timingId, Schema::NO_WAVE);
        WriteStart(timingId, AscendC::GetSystemCycle());
    }

    __aicore__ inline void End(typename Schema::FixedTiming type)
    {
        int timingId = static_cast<int>(type);
        if (!Enabled() || timingId < 0 || timingId >= Schema::FIXED_TIMING_COUNT) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteEnd(timingId, AscendC::GetSystemCycle());
    }

    // 每次 slice 独立记录；依赖等待会把当前执行段拆成真实的前后两段。
    __aicore__ inline void StartSpan(typename Schema::DynamicTimingType type)
    {
        if (!Enabled()) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        activeType_ = static_cast<int>(type);
        activeWaveId_ = waveId_;
        activeIndex_ = BeginRaw(activeType_, AscendC::GetSystemCycle());
    }

    __aicore__ inline void EndSpan(typename Schema::DynamicTimingType type)
    {
        if (!Enabled() || activeType_ != static_cast<int>(type)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        EndRaw(activeType_, activeIndex_, AscendC::GetSystemCycle());
        activeType_ = -1;
        activeIndex_ = -1;
    }

    __aicore__ inline bool BeginWait(typename Schema::DynamicTimingType type)
    {
        if (!Enabled() || (!Schema::kStandaloneWait && activeType_ < 0) || waitType_ >= 0) {
            return false;
        }
        // GM flag 轮询已通过原有同步确认数据就绪；此处不新增 PIPE_ALL，
        // 避免把其他流水排空时间误计为数据依赖等待。
        int64_t cycle = AscendC::GetSystemCycle();
        waitType_ = static_cast<int>(type);
        if (tileActive_ && Schema::kPairWaitSlots) {
            // 一个 tile 最多一次输入等待；无等待的 tile 在 WAIT 数组中保留空槽。
            waitIndex_ = activeIndex_;
            dynamicCounts_[waitType_] = dynamicCounts_[activeType_];
            if (waitIndex_ >= 0) {
                WriteMetadata(waitType_, waitIndex_);
                WriteWave(DynamicTimingIndex(waitType_, waitIndex_), activeWaveId_);
                WriteStart(DynamicTimingIndex(waitType_, waitIndex_), cycle);
            }
        } else {
            if (!tileActive_ && activeType_ >= 0) {
                EndRaw(activeType_, activeIndex_, cycle);
            }
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
        if (tileActive_ && Schema::kPairWaitSlots) {
            // 覆盖等待前的起点，不生成“准备工作冒充 tile”的区间。
            if (activeIndex_ >= 0) {
                WriteStart(DynamicTimingIndex(activeType_, activeIndex_), cycle);
            }
        } else if (!tileActive_ && activeType_ >= 0) {
            activeIndex_ = BeginRaw(activeType_, cycle);
        }
    }

    __aicore__ inline void FlushDynamicCounts()
    {
        if (!Enabled()) {
            return;
        }
        for (int typeId = 0; typeId < Schema::DYNAMIC_TYPE_COUNT; ++typeId) {
            int32_t index = CountsOffset() + coreId_ * CountStride() + typeId;
            WriteValue(index, dynamicCounts_[typeId]);
        }
    }

private:
    int coreCount_ = Schema::N_CORE_COUNT;
    int capacity_ = Schema::MAX_DYNAMIC_ITER;
    uint32_t tileM_ = 256, tileN_ = 256, kStart_ = 0, kCount_ = 0, nStart_ = 0, nCount_ = 0;
    int sourceRank_ = -1;
    uint32_t taskFlags_ = 0;
    __aicore__ inline int TimingItems() const
    {
        return Schema::FIXED_TIMING_COUNT + Schema::DYNAMIC_TYPE_COUNT * capacity_;
    }
    __aicore__ inline int TimingStride() const
    {
        return (TimingItems() * 2 + 15) / 16 * 16;
    }
    __aicore__ inline int CountStride() const
    {
        return (Schema::DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
    }
    __aicore__ inline int CountsOffset() const
    {
        return Schema::kHeaderWords + TimingStride() * coreCount_;
    }
    __aicore__ inline int MetadataOffset() const
    {
        return CountsOffset() + CountStride() * coreCount_;
    }
    __aicore__ inline int WaveOffset() const
    {
        return MetadataOffset() + coreCount_ * Schema::DYNAMIC_TYPE_COUNT * capacity_ * Schema::kMetadataWords;
    }
    __aicore__ inline int WaveStride() const
    {
        return (TimingItems() + 15) / 16 * 16;
    }
    GM_ADDR timerAddr_ = nullptr;
    int coreId_ = -1;
    int64_t waveId_ = Schema::NO_WAVE;
    int64_t activeWaveId_ = Schema::NO_WAVE;
    int dynamicCounts_[Schema::DYNAMIC_TYPE_COUNT]{};
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

    // Keep repeated recording stores out of the compute loop's inline body.
    __aicore__ __attribute__((noinline)) void WriteMetadata(int typeId, int index)
    {
        if (expert_ < 0) {
            return;
        }
        int offset = MetadataOffset() +
                     ((coreId_ * Schema::DYNAMIC_TYPE_COUNT + typeId) * capacity_ + index) * Schema::kMetadataWords;
        WriteValue(offset, static_cast<int64_t>((static_cast<uint64_t>(expert_ + 1) << 32) | rowStart_));
        WriteValue(offset + 1, static_cast<int64_t>((static_cast<uint64_t>(nTile_ + 1) << 32) | rows_));
        if constexpr (Schema::kMetadataWords > 2) {
            WriteValue(offset + 2, (static_cast<uint64_t>(tileM_) << 32) | tileN_);
            WriteValue(offset + 3, (static_cast<uint64_t>(kStart_) << 32) | kCount_);
            WriteValue(offset + 4, (static_cast<uint64_t>(nStart_) << 32) | nCount_);
            WriteValue(offset + 5, (static_cast<uint64_t>(sourceRank_ + 1) << 32) | taskFlags_);
            WriteValue(offset + 6,
                       typeId == waitType_ && activeType_ >= 0
                           ? (static_cast<uint64_t>(activeType_ + 1) << 32) | static_cast<uint32_t>(activeIndex_ + 1)
                           : 0);
        }
    }

    __aicore__ inline int BeginRaw(int typeId, int64_t cycle)
    {
        int index = dynamicCounts_[typeId]++;
        // 即使溢出也保留实际计数，主机必须报错，不能静默输出截断流水。
        if (index >= capacity_) {
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
        return Schema::FIXED_TIMING_COUNT + typeId * capacity_ + taskIndex;
    }

    __aicore__ inline int32_t BufferIndex(int timingId) const
    {
        return Schema::kHeaderWords + coreId_ * TimingStride() + timingId * 2;
    }

    __aicore__ __attribute__((noinline)) void WriteValue(int32_t index, int64_t value) const
    {
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(timerAddr_));
        timer(index) = value;
    }

    __aicore__ inline void WriteWave(int timingId, int64_t waveId)
    {
        if constexpr (Schema::kWaveIds) {
            WriteValue(WaveOffset() + coreId_ * WaveStride() + timingId, waveId);
        }
    }

    __aicore__ inline void WriteStart(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < CountsOffset()) {
            WriteValue(index, cycle);
        }
    }

    __aicore__ inline void WriteEnd(int timingId, int64_t cycle)
    {
        int32_t index = BufferIndex(timingId);
        if (index >= 0 && index + 1 < CountsOffset()) {
            WriteValue(index + 1, cycle);
        }
    }
};

template <class Timer>
class BasicAscendTimerTileScope {
    using Schema = typename Timer::SchemaType;

public:
    __aicore__ inline BasicAscendTimerTileScope(Timer* timer, typename Schema::DynamicTimingType type, uint32_t m,
                                                uint32_t n, uint32_t rows, uint32_t tileN = 256U)
        : timer_(timer != nullptr && timer->StartTile(type, m, n, rows, tileN) ? timer : nullptr)
    {
    }
    __aicore__ inline ~BasicAscendTimerTileScope()
    {
        if (timer_ != nullptr) {
            timer_->EndTile();
        }
    }

private:
    Timer* timer_;
};

// Prologue is a per-expert interval; keep the existing span completion semantics.
// Declare before prologue resources so their cleanup completes before EndSpan.
template <class Timer>
class BasicAscendTimerSpanScope {
    using Schema = typename Timer::SchemaType;

public:
    __aicore__ inline BasicAscendTimerSpanScope(Timer* timer, typename Schema::DynamicTimingType type)
        : timer_(timer), type_(type)
    {
        if (timer_ != nullptr) {
            if constexpr (Schema::kIndependentSpans) {
                index_ = timer_->BeginInterval(type_);
            } else {
                timer_->StartSpan(type_);
            }
        }
    }
    __aicore__ inline ~BasicAscendTimerSpanScope()
    {
        if (timer_ != nullptr) {
            if constexpr (Schema::kIndependentSpans) {
                timer_->EndInterval(type_, index_);
            } else {
                timer_->EndSpan(type_);
            }
        }
    }

private:
    int index_ = -1;
    Timer* timer_;
    typename Schema::DynamicTimingType type_;
};

// 作用域只包围已有的依赖等待，不改变等待条件和 flag 发布顺序。
template <class Timer>
class BasicAscendTimerWaitScope {
    using Schema = typename Timer::SchemaType;

public:
    __aicore__ inline BasicAscendTimerWaitScope(Timer* timer, typename Schema::DynamicTimingType type)
        : timer_(timer != nullptr && timer->BeginWait(type) ? timer : nullptr)
    {
    }
    __aicore__ inline ~BasicAscendTimerWaitScope()
    {
        if (timer_ != nullptr) {
            timer_->EndWait();
        }
    }

private:
    Timer* timer_;
};

#endif  // DEEPEP_ASCEND_TIMER_DEVICE_HPP
