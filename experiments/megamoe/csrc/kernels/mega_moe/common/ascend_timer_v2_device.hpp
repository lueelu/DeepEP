// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
// Modified by xixing 2026

#ifndef ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H

#include "ascend_timer_v2.hpp"
#include "csrc/kernels/common/timer/ascend_timer_device.hpp"

struct MegaMoeTimerSchema {
    using FixedTiming = AscendTimer::FixedTiming;
    using DynamicTimingType = AscendTimer::DynamicTimingType;
    static constexpr int FIXED_TIMING_COUNT = AscendTimer::FIXED_TIMING_COUNT;
    static constexpr int DYNAMIC_TYPE_COUNT = AscendTimer::DYNAMIC_TYPE_COUNT;
    static constexpr int MAX_DYNAMIC_ITER = AscendTimer::MAX_DYNAMIC_ITER;
    static constexpr int N_CORE_COUNT = AscendTimer::N_CORE_COUNT;
    static constexpr int NO_WAVE = AscendTimer::NO_WAVE;
    static constexpr int kHeaderWords = 0, kMetadataWords = 2;
    static constexpr bool kWaveIds = true, kPairWaitSlots = true, kStandaloneWait = false;
    static constexpr bool kIndependentSpans = false;
    __aicore__ static inline bool DrainTile(int type)
    {
        return type == AscendTimer::AIV_COMBINE;
    }
};
using AscendTimerDevice = BasicAscendTimerDevice<MegaMoeTimerSchema>;
using AscendTimerTileScope = BasicAscendTimerTileScope<AscendTimerDevice>;
using AscendTimerSpanScope = BasicAscendTimerSpanScope<AscendTimerDevice>;
using AscendTimerWaitScope = BasicAscendTimerWaitScope<AscendTimerDevice>;

#endif  // ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H
