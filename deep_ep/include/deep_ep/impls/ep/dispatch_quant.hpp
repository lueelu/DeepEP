// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/ub.hpp>

// Smooth per-token dynamic INT8 quantization shared by the source (same-host
// direct packets) and the destination relay (BF16 entry packets). Both call
// sites must produce identical bytes, so the numeric order lives in one place:
//   BF16 -> FP32, FP32 * smooth[e], abs-max over hidden,
//   alpha = 127 / max, y * alpha, FP32 -> FP16 (RINT), FP16 -> INT8 (RZ).
// A zero (or non-positive) maximum is defined as q=0 and d=0 instead of the
// reference helper's undefined 127/0.
namespace deep_ep {
namespace llquant {

constexpr float kInt8MaxValue = 127.0f;
constexpr float kInt8InvMaxValue = 1.0f / 127.0f;

// Full-width abs-max reduction for one token row. Mirrors ub::SumPacketFlags:
// a 256B vector holds 64 FP32 lanes, so the accumulator stays in lane zero.
__simd_vf__ inline void AbsMaxStore(__ubuf__ float* out, __ubuf__ float* src, uint32_t count)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    uint32_t oneWord = 1U;
    const auto one = asc_update_mask_b32(oneWord);
    vector_float total;
    asc_duplicate_scalar(total, 0.0f, all);
    for (uint32_t offset = 0, remaining = count; offset < count; offset += 64U) {
        vector_float input, absolute, part;
        const auto mask = asc_update_mask_b32(remaining);
        asc_loadalign(input, src + offset);
        asc_abs(absolute, input, mask);
        asc_reduce_max(part, absolute, mask);
        asc_max(total, total, part, one);
    }
    asc_storealign(out, total, one);
}

// Quantize `hidden` BF16 elements from `in` with `smooth` (FP32, same length)
// into `out`. `work` holds FP32 scratch (>= hidden + 64 floats), `half_buf`
// holds FP16 scratch, and `scalar` is a 32B-aligned FP32 result slot.
// Returns the FP32 dequant scale d = max / 127 (0 when max == 0).
__aicore__ inline float QuantizeToken(__ubuf__ uint8_t* out, __ubuf__ uint8_t* in,
                                      __ubuf__ uint8_t* smooth, __ubuf__ uint8_t* work,
                                      __ubuf__ uint8_t* half_buf, __ubuf__ float* scalar,
                                      uint32_t hidden)
{
    auto* in_bf16 = reinterpret_cast<__ubuf__ bfloat16_t*>(in);
    auto* work_f = reinterpret_cast<__ubuf__ float*>(work);
    auto* smooth_f = reinterpret_cast<__ubuf__ float*>(smooth);
    auto* half_p = reinterpret_cast<__ubuf__ half*>(half_buf);
    auto* out_i8 = reinterpret_cast<__ubuf__ int8_t*>(out);

    asc_bfloat162float(work_f, in_bf16, hidden);
    asc_mul(work_f, work_f, smooth_f, hidden);
    AbsMaxStore(scalar, work_f, hidden);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    const float maximum = scalar[0];
    const float alpha = maximum > 0.0f ? kInt8MaxValue / maximum : 0.0f;
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
    asc_mul_scalar(work_f, work_f, alpha, hidden);
    asc_float2half_rn(half_p, work_f, hidden);
    asc_half2int8_rz(out_i8, half_p, hidden);
    return maximum > 0.0f ? maximum * kInt8InvMaxValue : 0.0f;
}

// Store one FP32 dequant scale into the packed payload at `offset`.
__aicore__ inline void StoreScale(__ubuf__ uint8_t* payload, uint32_t offset, float value)
{
    *reinterpret_cast<__ubuf__ float*>(payload + offset) = value;
}

}  // namespace llquant
}  // namespace deep_ep
