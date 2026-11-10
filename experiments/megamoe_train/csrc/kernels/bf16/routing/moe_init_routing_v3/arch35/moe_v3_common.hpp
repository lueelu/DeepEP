// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*!
 * \file moe_v3_common.hpp
 * \brief Shared constants, types and helpers for MoE routing initialization.
 */
#ifndef MOE_V3_COMMON_HPP
#define MOE_V3_COMMON_HPP

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.hpp"

namespace MoeInitRoutingV3 {
using namespace AscendC;
#define OVERFLOW_MODE_CTRL 60
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t GATHER = 0;
constexpr int64_t SCATTER = 1;

// Arch35 vector registers are 256 bytes. CANN 9's host pass does not expose
// the device-only VECTOR_REG_WIDTH constant at namespace scope.
constexpr uint16_t FLOAT_REG_TENSOR_LENGTH = 256 / sizeof(float);

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}
template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_COMMON_HPP
