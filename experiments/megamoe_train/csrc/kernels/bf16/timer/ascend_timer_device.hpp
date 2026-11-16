// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

#ifndef MEGA_MOE_BF16_ASCEND_TIMER_DEVICE_HPP
#define MEGA_MOE_BF16_ASCEND_TIMER_DEVICE_HPP

#include "ascend_timer.hpp"
#include "csrc/kernels/common/timer/ascend_timer_device.hpp"

struct MegaMoeTrainTimerSchema {
    using FixedTiming = MegaMoeTrainTimer::FixedTiming;
    using DynamicTimingType = MegaMoeTrainTimer::DynamicTimingType;
    static constexpr int FIXED_TIMING_COUNT = MegaMoeTrainTimer::FIXED_TIMING_COUNT;
    static constexpr int DYNAMIC_TYPE_COUNT = MegaMoeTrainTimer::DYNAMIC_TYPE_COUNT;
    static constexpr int MAX_DYNAMIC_ITER = 256;
    static constexpr int N_CORE_COUNT = 3 * MegaMoeTrainTimer::kMaxCoreGroups;
    static constexpr int NO_WAVE = -1;
    static constexpr int kHeaderWords = MegaMoeTrainTimer::kHeaderWords;
    static constexpr int kMetadataWords = MegaMoeTrainTimer::kMetadataWords;
    static constexpr bool kWaveIds = false, kPairWaitSlots = false, kStandaloneWait = true;
    static constexpr bool kIndependentSpans = true;
    __aicore__ static inline bool DrainTile(int)
    {
        return false;
    }
};

class MegaMoeTrainTimerDevice : public BasicAscendTimerDevice<MegaMoeTrainTimerSchema> {
    using Base = BasicAscendTimerDevice<MegaMoeTrainTimerSchema>;

public:
    __aicore__ inline void Init(GM_ADDR address)
    {
        int cores = 0, capacity = 0;
#ifdef DEEPEP_MEGAMOE_TIMER
        if (address != nullptr) {
            AscendC::GlobalTensor<int64_t> header;
            header.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(address));
            cores = static_cast<int>(header.GetValue(2));
            capacity = static_cast<int>(header.GetValue(3));
        }
#endif
        BasicAscendTimerDevice::Init(address, cores, capacity);
    }

    __aicore__ inline void SetSlice(int expert, uint32_t rowStart, uint32_t rows)
    {
        Base::SetSlice(expert, rowStart, rows);
        expert_ = expert;
    }

    __aicore__ inline bool BeginStageWait(MegaMoeTrainTimer::DynamicTimingType type)
    {
        if (!Enabled()) return false;
        if (type == MegaMoeTrainTimer::AIC_GMM1_UB_WAIT || type == MegaMoeTrainTimer::AIC_GMM2_UB_WAIT ||
            type == MegaMoeTrainTimer::AIC_BW_DHP_UB_WAIT || type == MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT) {
            return false;
        }
        const bool shared = type == MegaMoeTrainTimer::AIV_SYNC_WAIT || type == MegaMoeTrainTimer::AIC_SYNC_WAIT;
        const uint32_t key = static_cast<uint32_t>(expert_ + 1);
        if (!shared && key != 0 && lastWaitExpert_[type] == key) return false;
        if (!Base::BeginWait(type)) return false;
        if (!shared && key != 0) lastWaitExpert_[type] = key;
        return true;
    }

    __aicore__ inline void EndStageWait(MegaMoeTrainTimer::DynamicTimingType type)
    {
        // CrossCoreWaitFlag queues a V-pipe dependency. Retire that pipe
        // to Scalar before recording completion with GetSystemCycle().
        if constexpr (g_coreType == AscendC::AIV) {
            if (type == MegaMoeTrainTimer::AIV_SWIGLU_WAIT || type == MegaMoeTrainTimer::AIV_COMBINE_WAIT ||
                type == MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD_WAIT || type == MegaMoeTrainTimer::AIV_BW_COMBINE_WAIT) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            }
        }
        Base::EndWait();
    }

private:
    int expert_ = -1;
    uint32_t lastWaitExpert_[MegaMoeTrainTimer::DYNAMIC_TYPE_COUNT]{};
};

class MegaMoeTrainTimerTileScope : public BasicAscendTimerTileScope<MegaMoeTrainTimerDevice> {
    using Base = BasicAscendTimerTileScope<MegaMoeTrainTimerDevice>;
    __aicore__ static inline MegaMoeTrainTimerDevice* Prepare(MegaMoeTrainTimerDevice* timer, uint32_t expert,
                                                              uint32_t rows, uint32_t cols, uint32_t depth,
                                                              uint32_t tileM, uint32_t tileN, uint32_t flags,
                                                              int source)
    {
        if (timer == nullptr || !timer->Enabled()) return nullptr;
        timer->SetSlice(expert, 0, rows);
        timer->SetTaskShape(tileM, tileN, 0, depth, cols);
        timer->SetTaskContext(source, flags);
        return timer;
    }

public:
    __aicore__ inline MegaMoeTrainTimerTileScope(MegaMoeTrainTimerDevice* timer,
                                                 MegaMoeTrainTimer::DynamicTimingType type, uint32_t expert, uint32_t m,
                                                 uint32_t n, uint32_t rows, uint32_t cols, uint32_t depth,
                                                 uint32_t tileM = 128, uint32_t tileN = 256, uint32_t flags = 0,
                                                 int source = -1)
        : Base(Prepare(timer, expert, rows, cols, depth, tileM, tileN, flags, source), type, m, n, rows, tileN),
          timer_(timer),
          type_(type)
    {
    }
    __aicore__ inline ~MegaMoeTrainTimerTileScope()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            if (timer_ != nullptr && timer_->Enabled() &&
                (type_ == MegaMoeTrainTimer::AIV_DISPATCH || type_ == MegaMoeTrainTimer::AIV_COMBINE ||
                 type_ == MegaMoeTrainTimer::AIV_BW_DISPATCH || type_ == MegaMoeTrainTimer::AIV_BW_COMBINE)) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        if constexpr (g_coreType == AscendC::AIC) {
            if (timer_ != nullptr && timer_->Enabled()) {
                // Match W4A8's tile-ready boundary: retire this tile's Fixpipe
                // output before ending its interval. Only a capture does this.
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            }
        }
    }

private:
    MegaMoeTrainTimerDevice* timer_;
    MegaMoeTrainTimer::DynamicTimingType type_;
};
using MegaMoeTrainTimerSpanScope = BasicAscendTimerSpanScope<MegaMoeTrainTimerDevice>;

class MegaMoeTrainTimerWaitScope {
public:
    __aicore__ inline MegaMoeTrainTimerWaitScope(MegaMoeTrainTimerDevice* timer,
                                                 MegaMoeTrainTimer::DynamicTimingType type)
        : timer_(timer != nullptr && timer->BeginStageWait(type) ? timer : nullptr), type_(type)
    {
    }
    __aicore__ inline ~MegaMoeTrainTimerWaitScope()
    {
        if (timer_ != nullptr) timer_->EndStageWait(type_);
    }

private:
    MegaMoeTrainTimerDevice* timer_;
    MegaMoeTrainTimer::DynamicTimingType type_;
};

// Fixed stages do not share the dynamic tile/WAIT state.
class MegaMoeTrainTimerFixedScope {
public:
    __aicore__ inline MegaMoeTrainTimerFixedScope(MegaMoeTrainTimerDevice* timer, MegaMoeTrainTimer::FixedTiming type)
        : timer_(g_coreType == AscendC::AIV ? timer : nullptr), type_(type)
    {
        if (timer_ != nullptr) {
            timer_->Start(type_);
        }
    }
    __aicore__ inline ~MegaMoeTrainTimerFixedScope()
    {
        if (timer_ != nullptr) {
            timer_->End(type_);
        }
    }

private:
    MegaMoeTrainTimerDevice* timer_;
    MegaMoeTrainTimer::FixedTiming type_;
};

#endif  // MEGA_MOE_BF16_ASCEND_TIMER_DEVICE_HPP
