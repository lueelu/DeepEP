# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Host mirror of kernels/mega_moe/common/ascend_timer_v2.h."""

FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INPUT_BUFFER_INIT",
    "AIV_QUANT",
    "AIV_ROUTE_MASK",
    "AIV_RESET",
    "AIV_INPUT_CROSS_RANK_SYNC",
    "AIV_EXPORT_SYNC",
    "AIV_FINAL_CROSS_RANK_WAIT",
    "AIV_UNPERMUTE",
    "AIC_WAIT_INPUT_PREP",
]

DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIV_GMM1_PROLOGUE",
    "AIV_GMM2_PROLOGUE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
    "AIC_SHARED_GMM1",
    "AIV_SHARED_GMM1_PROLOGUE",
    "AIC_SHARED_GMM2",
    "AIV_SHARED_GMM2_PROLOGUE",
    "AIV_SHARED_SWIGLU",
]

# W4 动态槽保存 tile/slice；其他路径保持原有区间协议。
MAX_DYNAMIC_ITER = 4096
N_CORE_COUNT = 96
CYCLE_TO_US = 1000.0

COUNTERS_PER_CORE = ((len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER) * 2 + 15) // 16 * 16
N_TIMING_COUNTER = COUNTERS_PER_CORE * N_CORE_COUNT
DYNAMIC_ITER_PER_CORE = (len(DYNAMIC_NAMES) + 15) // 16 * 16
TASK_METADATA_OFFSET = N_TIMING_COUNTER + DYNAMIC_ITER_PER_CORE * N_CORE_COUNT
TASK_METADATA_SIZE = N_CORE_COUNT * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER * 2
NO_WAVE = -1
WAVE_ID_OFFSET = TASK_METADATA_OFFSET + TASK_METADATA_SIZE
WAVE_ID_PER_CORE = (len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + 15) // 16 * 16
TOTAL_BUFFER_SIZE = WAVE_ID_OFFSET + WAVE_ID_PER_CORE * N_CORE_COUNT


def task_metadata(raw, core_id, logical_index):
    """坐标直接来自设备，不根据 shape 或耗时猜测任务身份。"""
    dynamic_index = logical_index - len(FIXED_NAMES)
    if dynamic_index < 0:
        return {}
    offset = TASK_METADATA_OFFSET + (core_id * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + dynamic_index) * 2
    first, second = raw[offset : offset + 2]
    if first == second == 0:
        return {}
    mask = (1 << 32) - 1
    expert = ((first >> 32) & mask) - 1
    row_start = first & mask
    n_tile = ((second >> 32) & mask) - 1
    row_count = second & mask
    if expert < 0 or row_count == 0:
        raise ValueError(f"invalid task metadata: core={core_id}, slot={dynamic_index}")
    result = dict(
        expert_id=expert,
        row_start=row_start,
        row_count=row_count,
        m_tile=row_start // 256,
        granularity="tile" if n_tile >= 0 else "slice",
    )
    family = DYNAMIC_NAMES[dynamic_index // MAX_DYNAMIC_ITER]
    if "_SHARED_" in family:
        result["expert_kind"] = "shared"
    if n_tile >= 0:
        result.update(n_tile=n_tile, tile_id=dynamic_index % MAX_DYNAMIC_ITER)
    return result
