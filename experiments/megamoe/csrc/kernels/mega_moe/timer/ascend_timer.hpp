// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

#ifndef MEGA_MOE_BF16_ASCEND_TIMER_HPP
#define MEGA_MOE_BF16_ASCEND_TIMER_HPP

#include <stdint.h>

namespace MegaMoeTrainTimer {

enum FixedTiming {
    KERNEL_TIMING = 0,
    AIV_INPUT_CROSS_RANK_SYNC,
    AIV_INIT_ROUTING,
    AIV_GATHER_TOPK_WEIGHTS,
    AIV_BUFFER_INIT,
    AIV_COUNT_EXCHANGE,
    AIV_SAVE_STATE,
    AIV_FINAL_CROSS_RANK_WAIT,
    AIV_UNPERMUTE,
    AIV_BW_GATHER,
    AIV_BW_RESTORE_TABLE,
    AIV_BW_UNPERMUTE,
    AIV_BW_DP_REDUCE,
    AIV_BW_TOPK_UNPERMUTE,
    AIV_RESET,
    AIV_SAVE_DISPATCH_WEIGHTS,
    FIXED_TIMING_COUNT
};

enum DynamicTimingType {
    AIV_DISPATCH = 0,
    AIC_GMM1_WAIT,
    AIC_GMM1_UB_WAIT,
    AIC_GMM1,
    AIV_SWIGLU_WAIT,
    AIV_SWIGLU,
    AIV_SAVE_GATE_UP,
    AIC_GMM2_WAIT,
    AIC_GMM2_UB_WAIT,
    AIC_GMM2,
    AIV_COMBINE_WAIT,
    AIV_COMBINE,
    AIV_BW_DISPATCH,
    AIC_BW_RECOMPUTE,
    AIC_BW_DHP_WAIT,
    AIC_BW_DHP_UB_WAIT,
    AIC_BW_DHP,
    AIV_BW_SWIGLU_GRAD_WAIT,
    AIV_BW_SWIGLU_GRAD,
    AIV_BW_DECODE_GATE_UP,
    AIC_BW_DX_WAIT,
    AIC_BW_DO_REUSE_WAIT,
    AIC_BW_DX_UB_WAIT,
    AIC_BW_DX,
    AIV_BW_COMBINE_WAIT,
    AIV_BW_COMBINE,
    AIC_BW_DW1,
    AIC_BW_DW2_WAIT,
    AIC_BW_DW2,
    AIV_BW_DP_SEND,
    AIV_SYNC_WAIT,
    AIC_SYNC_WAIT,
    DYNAMIC_TYPE_COUNT
};

constexpr int64_t kMagic = 0x44455054524e5431LL;  // DEPTRNT1
constexpr int64_t kVersion = 1;
constexpr int64_t kForwardWaveVersion = 2;
constexpr int64_t kBackwardWaveVersion = 3;
constexpr int kHeaderWords = 16;
constexpr int kMetadataWords = 7;
constexpr int kMaxCoreGroups = 64;
constexpr int kMaxCapacity = 16384;

// The header records actual physical lanes and slots per family. Each slot
// carries timestamps plus matrix coordinates; forward/backward own buffers.
struct Layout {
    int cores;
    int capacity;
    constexpr int Items() const
    {
        return FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT * capacity;
    }
    constexpr int TimingStride() const
    {
        return (Items() * 2 + 15) / 16 * 16;
    }
    constexpr int CountStride() const
    {
        return (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
    }
    constexpr int CountsOffset() const
    {
        return kHeaderWords + cores * TimingStride();
    }
    constexpr int MetadataOffset() const
    {
        return CountsOffset() + cores * CountStride();
    }
    constexpr int64_t Elements(bool withWaves) const
    {
        const int64_t metadataEnd =
            MetadataOffset() + static_cast<int64_t>(cores) * DYNAMIC_TYPE_COUNT * capacity * kMetadataWords;
        return metadataEnd + (withWaves ? static_cast<int64_t>(cores) * ((Items() + 15) / 16 * 16) : 0);
    }
};

}  // namespace MegaMoeTrainTimer

#endif  // MEGA_MOE_BF16_ASCEND_TIMER_HPP
