# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Training Timer v1 protocol; matrix coordinates are stage-specific."""

MAGIC = 0x44455054524E5431
VERSION = 1
HEADER_WORDS = 16
METADATA_WORDS = 7
CYCLE_TO_US = 1000.0
FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INPUT_CROSS_RANK_SYNC",
    "AIV_INIT_ROUTING",
    "AIV_GATHER_TOPK_WEIGHTS",
    "AIV_BUFFER_INIT",
    "AIV_COUNT_EXCHANGE",
    "AIV_SAVE_STATE",
    "AIV_FINAL_CROSS_RANK_WAIT",
    "AIV_UNPERMUTE",
    "AIV_BW_GATHER",
    "AIV_BW_RESTORE_TABLE",
    "AIV_BW_UNPERMUTE",
    "AIV_BW_DP_REDUCE",
    "AIV_BW_TOPK_UNPERMUTE",
    "AIV_RESET",
]
DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIC_GMM1_WAIT",
    "AIC_GMM1_UB_WAIT",
    "AIC_GMM1",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_SAVE_GATE_UP",
    "AIC_GMM2_WAIT",
    "AIC_GMM2_UB_WAIT",
    "AIC_GMM2",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIV_BW_DISPATCH",
    "AIC_BW_RECOMPUTE",
    "AIC_BW_DHP_WAIT",
    "AIC_BW_DHP_UB_WAIT",
    "AIC_BW_DHP",
    "AIV_BW_SWIGLU_GRAD_WAIT",
    "AIV_BW_SWIGLU_GRAD",
    "AIV_BW_DECODE_GATE_UP",
    "AIC_BW_DX_WAIT",
    "AIC_BW_DO_REUSE_WAIT",
    "AIC_BW_DX_UB_WAIT",
    "AIC_BW_DX",
    "AIV_BW_COMBINE_WAIT",
    "AIV_BW_COMBINE",
    "AIC_BW_DW1",
    "AIC_BW_DW2_WAIT",
    "AIC_BW_DW2",
    "AIV_BW_DP_SEND",
    "AIV_SYNC_WAIT",
    "AIC_SYNC_WAIT",
]


def layout(cores, capacity):
    if not 0 < cores <= 192 or cores % 3 or not 0 < capacity <= 16384:
        raise ValueError("invalid training Timer lane count or capacity")
    items = len(FIXED_NAMES) + len(DYNAMIC_NAMES) * capacity
    stride = (items * 2 + 15) // 16 * 16
    count_stride = (len(DYNAMIC_NAMES) + 15) // 16 * 16
    counts = HEADER_WORDS + cores * stride
    metadata = counts + cores * count_stride
    elements = metadata + cores * len(DYNAMIC_NAMES) * capacity * METADATA_WORDS
    return dict(
        cores=cores,
        capacity=capacity,
        stride=stride,
        count_stride=count_stride,
        counts=counts,
        metadata=metadata,
        elements=elements,
    )
