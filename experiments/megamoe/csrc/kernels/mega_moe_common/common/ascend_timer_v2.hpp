// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H

#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include <stdint.h>

namespace DEEPEP_MEGAMOE_TIMER_NAMESPACE {

enum FixedTiming {
#ifdef DEEPEP_MEGAMOE_ROCE
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
    AIC_MOE_STARTUP,
    AIV_MOE_STARTUP,
    AIC_DISPATCH_BUFFER_INIT,
    AIV_DISPATCH_BUFFER_INIT,
    AIV_INPUT_PREP_BARRIER_WAIT,
    AIV_COUNT_TABLE_PREPARE,
    AIV_COUNT_METADATA_WAIT,
    AIV_GATEWAY_PAYLOAD_CHECK,
    AIV_COUNT_PREFIX_BUILD,
    AIV_COUNT_TABLE_PUBLISH,
    AIC_COUNT_READY_WAIT,
    AIV_COUNT_READY_WAIT,
    AIV_COMBINE_INIT,
    AIV_HOST_COMBINE_BARRIER_WAIT,
    AIV_GATEWAY_PAYLOAD_BROADCAST,
    FIXED_TIMING_COUNT
};
#else
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
#endif

enum DynamicTimingType {
#ifdef DEEPEP_MEGAMOE_ROCE
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
    AIV_RDMA_POST,
    AIV_RDMA_QUIET_WAIT,
    AIV_RDMA_RECV_WAIT,
    AIV_GATEWAY_PAYLOAD_WAIT,
    AIV_COMBINE_PROGRESS,
    AIV_COMBINE_DRAIN,
    AIV_RELAY_STEP,
    AIV_COMBINE_RUN,
    AIV_COMBINE_REDUCE,
    AIV_COMBINE_ASYNC,
    AIV_COMBINE_READY,
    AIV_COMBINE_SYNC,
    AIV_PAYLOAD_ALL_LANES_WAIT,
    DYNAMIC_TYPE_COUNT_ENUM
};

enum class RdmaPhase : uint32_t { DispatchMeta = 0, DispatchPayload = 1, Combine = 2 };

enum class RelayStep : uint32_t {
    PacketWait = 0,
    TopkLoad,
    RouteBuild,
    RoutePost,
    RouteComplete,
    CountPublish,
    MetaPack,
    Clos,
    Mesh,
    SourceWait,
    LocalRoute,
    TableWait,
    Histogram,
    CompactPlan,
    RouteRead,
    FlagCheck,
    Retry,
    Backoff,
    PayloadCopy,
    Drain,
    TilePublish
};
// Reuses the historical ASYNC family without changing timer buffer dimensions.
enum class CombineRowPhase : uint32_t { Lifetime = 1, Issue = 2, ReuseWait = 3 };

// One summary per count-table acquisition, not one GM trace write per poll.
// Source coordinates describe the last observed FAILED snapshot, never an
// inferred network arrival order. A zero COUNT_LAST_STALE_WORDS means no failed
// poll. The representative expert index is local to the receiving rank.
enum CountDiagnostic {
    COUNT_PROFILE_VALID = 0,
    COUNT_SNAPSHOT_ATTEMPTS,
    COUNT_SNAPSHOT_FAILURES,
    COUNT_DMA_CYCLES,
    COUNT_CHECK_CYCLES,
    COUNT_BACKOFF_CYCLES,
    COUNT_SUCCESS_DMA_CYCLES,
    COUNT_SUCCESS_CHECK_CYCLES,
    COUNT_LAST_STALE_WORDS,
    COUNT_LAST_STALE_SOURCE_MASK,
    COUNT_LAST_STALE_SOURCE_PLUS_ONE,
    COUNT_LAST_STALE_EXPERT_PLUS_ONE,
    COUNT_LAST_OBSERVED_EPOCH,
    COUNT_EXPECTED_EPOCH,
    COUNT_DIAGNOSTIC_CYCLES,
    COUNT_SNAPSHOT_BYTES,
    COUNT_SOURCE_MASK_OVERFLOW,
    COUNT_SNAPSHOT_SOURCES,
    COUNT_SNAPSHOT_EXPERTS,
    COUNT_CAPTURED_SNAPSHOT_WORDS,
    COUNT_DIAGNOSTIC_COUNT
};
#else
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
#endif

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
#ifdef DEEPEP_MEGAMOE_ROCE
static constexpr int COUNT_DIAGNOSTIC_OFFSET = WAVE_ID_OFFSET + WAVE_ID_TOTAL_SIZE;
// Raw int32 snapshots are decoded on the host, avoiding scalar UB scans at startup.
static constexpr int COUNT_SNAPSHOT_MAX_WORDS = 4096;
static constexpr int COUNT_SNAPSHOT_OFFSET =
    (COUNT_DIAGNOSTIC_OFFSET + N_CORE_COUNT * COUNT_DIAGNOSTIC_COUNT + 3) / 4 * 4;
static constexpr int COUNT_SNAPSHOT_SLOTS_PER_CORE = COUNT_SNAPSHOT_MAX_WORDS / 2;  // int64 buffer slots
static constexpr int TOTAL_BUFFER_SIZE = COUNT_SNAPSHOT_OFFSET + N_CORE_COUNT * COUNT_SNAPSHOT_SLOTS_PER_CORE;
#else
static constexpr int TOTAL_BUFFER_SIZE = WAVE_ID_OFFSET + WAVE_ID_TOTAL_SIZE;
#endif

}  // namespace DEEPEP_MEGAMOE_TIMER_NAMESPACE

#endif  // ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_H
