// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H

#include <stdint.h>

namespace AscendTimer {

enum FixedTiming {
    KERNEL_TIMING = 0,
    AIV_INPUT_BUFFER_INIT,
    AIV_QUANT,
    AIV_ROUTE_MASK,
    AIV_RESET,
    AIV_INPUT_CROSS_RANK_SYNC,
    AIV_EXPORT_SYNC,
    AIV_FINAL_CROSS_RANK_WAIT,
    AIV_UNPERMUTE,
    AIC_WAIT_INPUT_PREP,
    FIXED_TIMING_COUNT
};

enum DynamicTimingType {
    AIV_DISPATCH = 0,
    AIV_SWIGLU_WAIT,
    AIV_SWIGLU,
    AIV_COMBINE_WAIT,
    AIV_COMBINE,
    AIV_GMM1_PROLOGUE,
    AIV_GMM2_PROLOGUE,
    AIC_GMM1_WAIT,
    AIC_GMM1,
    AIC_GMM2_WAIT,
    AIC_GMM2,
    AIC_SHARED_GMM1,
    AIV_SHARED_GMM1_PROLOGUE,
    AIC_SHARED_GMM2,
    AIV_SHARED_GMM2_PROLOGUE,
    AIV_SHARED_SWIGLU,
    DYNAMIC_TYPE_COUNT_ENUM
};

// W4 tile 事件按真实任务编号；WAIT 复用对应计算任务的编号。
// 为大 BS 的 tile 依赖等待预留空间，超限由主机校验明确报错。
static constexpr int MAX_DYNAMIC_ITER = 4096;
static constexpr int N_CORE_COUNT = 96;
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int TASK_METADATA_OFFSET = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE;
// 每个动态槽追加两个 int64：(expert+1,row_start)、(n_tile+1,row_count)，各占 32 bit。
// 全零表示旧路径无任务坐标；n_tile+1 为零表示 slice 级记录。
static constexpr int TASK_METADATA_SIZE = N_CORE_COUNT * DYNAMIC_TYPE_COUNT * MAX_DYNAMIC_ITER * 2;
// 保留任务坐标，在尾部追加真实 Wave ID；-1 表示全局或未标记的路径。
static constexpr int NO_WAVE = -1;
static constexpr int WAVE_ID_OFFSET = TASK_METADATA_OFFSET + TASK_METADATA_SIZE;
static constexpr int WAVE_ID_PER_CORE_ALIGN = (N_TIMING_ITEM_PER_CORE + 15) / 16 * 16;
static constexpr int WAVE_ID_TOTAL_SIZE = WAVE_ID_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int TOTAL_BUFFER_SIZE = WAVE_ID_OFFSET + WAVE_ID_TOTAL_SIZE;

}  // namespace AscendTimer

#endif  // ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H
