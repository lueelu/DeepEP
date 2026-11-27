// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/ub.hpp>

// Smooth per-token dynamic INT8 quantization shared by the source (same-host
// direct packets) and the destination relay (BF16 entry packets). Both call
// sites must produce identical bytes, so the numeric order lives in one place:
//   BF16 -> FP32 (once per token), FP32 * smooth[e], abs-max over hidden,
//   alpha = 127 / max, y * alpha, FP32 -> INT32 (RINT), INT32 -> FP16, FP16 -> INT8.
// The AscendC LocalTensor path is used for the conversions and elementwise
// math because the UB-pointer asc_* overloads have no linked definition on
// dav-3510. A zero (or non-positive) maximum is defined as q=0 and d=0.
namespace deep_ep {
namespace llquant {

constexpr float kInt8MaxValue = 127.0f;

// Full-width abs-max reduction for one token row. Complete 64-lane blocks
// accumulate lane-wise, then a single horizontal reduction runs at the end; a
// partial tail block keeps its own masked reduction. H3584 (56 complete blocks)
// therefore performs one horizontal reduction instead of 56.
__simd_vf__ inline void AbsMaxStore(__ubuf__ float* out, __ubuf__ float* src, uint32_t count)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    uint32_t oneWord = 1U;
    const auto one = asc_update_mask_b32(oneWord);
    const uint32_t fullBlocks = count / 64U;
    const uint32_t tail = count - fullBlocks * 64U;
    vector_float total;
    asc_duplicate_scalar(total, 0.0f, all);
    for (uint32_t block = 0; block < fullBlocks; ++block) {
        vector_float input, absolute;
        asc_loadalign(input, src + block * 64U);
        asc_abs(absolute, input, all);
        asc_max(total, total, absolute, all);
    }
    if (fullBlocks != 0U) {
        vector_float reduced;
        asc_reduce_max(reduced, total, all);
        if (tail != 0U) {
            vector_float tailInput, tailAbsolute, tailReduced;
            uint32_t remaining = tail;
            const auto tailMask = asc_update_mask_b32(remaining);
            asc_loadalign(tailInput, src + fullBlocks * 64U);
            asc_abs(tailAbsolute, tailInput, tailMask);
            asc_reduce_max(tailReduced, tailAbsolute, tailMask);
            asc_max(reduced, reduced, tailReduced, one);
        }
        asc_storealign(out, reduced, one);
    } else if (count != 0U) {
        // Only a tail block: use its masked reduction directly.
        vector_float tailInput, tailAbsolute, tailReduced;
        uint32_t remaining = count;
        const auto tailMask = asc_update_mask_b32(remaining);
        asc_loadalign(tailInput, src);
        asc_abs(tailAbsolute, tailInput, tailMask);
        asc_reduce_max(tailReduced, tailAbsolute, tailMask);
        asc_storealign(out, tailReduced, one);
    } else {
        asc_storealign(out, total, one);
    }
}

// View an existing static UB region as an AscendC tensor. The region is already
// part of the dispatch UB map; no TPipe arena is created here.
template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(__ubuf__ uint8_t* address, uint32_t elements)
{
    return AscendC::LocalTensor<T>(
        AscendC::TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uint64_t>(address)), elements);
}

// Convert the raw BF16 token into the unsmoothed FP32 token once. Every expert
// of the same token/group reuses this buffer, so the Cast is not repeated.
__aicore__ inline void PrepareToken(__ubuf__ uint8_t* fp32, __ubuf__ uint8_t* bf16, uint32_t hidden)
{
    auto dst = View<float>(fp32, hidden);
    auto src = View<bfloat16_t>(bf16, hidden);
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, hidden);
    AscendC::PipeBarrier<PIPE_V>();
}

// Quantize one expert from the unsmoothed FP32 token. `smooth` holds the FP32
// row of this expert, `work` holds FP32/INT32 scratch (>= hidden + 64 elements),
// `half_buf` holds FP16 scratch and `scalar` is a 32B-aligned FP32 result slot.
// Returns d = 1 / (127 / max), or 0 when max <= 0 (q is then zeroed).
__aicore__ inline float QuantizeExpert(__ubuf__ uint8_t* out, __ubuf__ uint8_t* fp32,
                                       __ubuf__ uint8_t* smooth, __ubuf__ uint8_t* work,
                                       __ubuf__ uint8_t* half_buf, __ubuf__ float* scalar,
                                       uint32_t hidden)
{
    auto src_f = View<float>(fp32, hidden);
    auto work_f = View<float>(work, hidden);
    auto work_i32 = View<int32_t>(work, hidden);
    auto half_t = View<half>(half_buf, hidden);
    auto out_i8 = View<int8_t>(out, hidden);
    auto smooth_f = View<float>(smooth, hidden);

    AscendC::Mul(work_f, src_f, smooth_f, static_cast<int32_t>(hidden));
    AscendC::PipeBarrier<PIPE_V>();
    AbsMaxStore(scalar, reinterpret_cast<__ubuf__ float*>(work), hidden);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    const float maximum = scalar[0];
    if (!(maximum > 0.0f)) {
        // Defined edge case: an all-zero (or non-finite-negative) row emits
        // q=0 and d=0 instead of the reference helper's undefined 127/0.
        ub::FillBytes(reinterpret_cast<__ubuf__ uint8_t*>(out), 0U, hidden);
        return 0.0f;
    }
    // Keep the reference operation order: alpha = FP32(127 / max),
    // d = FP32(1 / alpha). Do not fold this into max * (1/127).
    const float alpha = kInt8MaxValue / maximum;
    const float dequant = 1.0f / alpha;
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
    AscendC::Muls(work_f, work_f, alpha, static_cast<int32_t>(hidden));
    AscendC::PipeBarrier<PIPE_V>();
    // Integer RINT in FP32, then an exact INT32 -> FP16 -> INT8 narrowing.
    AscendC::Cast(work_i32, work_f, AscendC::RoundMode::CAST_RINT, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetDeqScale(static_cast<half>(1.0f));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(half_t, work_i32, AscendC::RoundMode::CAST_ROUND, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(out_i8, half_t, AscendC::RoundMode::CAST_TRUNC, hidden);
    return dequant;
}

// Store one FP32 dequant scale into the packed payload at `offset`.
__aicore__ inline void StoreScale(__ubuf__ uint8_t* payload, uint32_t offset, float value)
{
    *reinterpret_cast<__ubuf__ float*>(payload + offset) = value;
}

}  // namespace llquant
}  // namespace deep_ep
