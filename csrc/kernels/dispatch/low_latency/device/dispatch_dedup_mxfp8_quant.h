// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026
#pragma once
#include <type_traits>
#include <c_api/asc_simd.h>

namespace DispatchDedupMxfp8Quant {
// MXFP8 keeps its original 32-element groups, exponent rules and cast rounding.
// Deinterleave pairs reduce 16 lanes per data block to one 32-input exponent.
template <class T>
__simd_vf__ inline void ComputeMaxExp(__ubuf__ T* src, __ubuf__ uint16_t* output, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t exponentMask, invalidHalf;
    asc_duplicate_scalar(exponentMask, uint16_t(0x7f80U), all);
    asc_duplicate_scalar(invalidHalf, uint16_t(0x7c00U), all);
    vector_store_unalign unalign;
    for (uint32_t offset = 0, remaining = count; offset < count; offset += 256U) {
        const auto mask0 = asc_update_mask_b16(remaining);
        const auto mask1 = asc_update_mask_b16(remaining);
        vector_uint16_t input0, input1, exponent0, exponent1, maximum;
        asc_loadalign_deintlv(input0, input1, reinterpret_cast<__ubuf__ uint16_t*>(src + offset));
        if constexpr (std::is_same<T, half>::value) {
            vector_uint16_t selected0, selected1;
            asc_and(selected0, input0, invalidHalf, mask0);
            asc_and(selected1, input1, invalidHalf, mask0);
            const auto valid0 = asc_ne(selected0, invalidHalf, mask0);
            const auto valid1 = asc_ne(selected1, invalidHalf, mask0);
            vector_bfloat16_t bf0, bf1;
            asc_half2bfloat16_rz(bf0, reinterpret_cast<vector_half&>(input0), mask0);
            asc_half2bfloat16_rz(bf1, reinterpret_cast<vector_half&>(input1), mask0);
            asc_and(exponent0, reinterpret_cast<vector_uint16_t&>(bf0), exponentMask, mask0);
            asc_and(exponent1, reinterpret_cast<vector_uint16_t&>(bf1), exponentMask, mask0);
            asc_select(exponent0, exponent0, exponentMask, valid0);
            asc_select(exponent1, exponent1, exponentMask, valid1);
        } else {
            asc_and(exponent0, input0, exponentMask, mask0);
            asc_and(exponent1, input1, exponentMask, mask0);
        }
        (void)mask1;  // The admitted compute count is padded to complete pairs.
        asc_max(maximum, exponent0, exponent1, mask0);
        asc_reduce_max_datablock(maximum, maximum, mask0);
        asc_storeunalign(output + offset / 32U, unalign, maximum, 8U);
    }
    asc_storeunalign_post(output + count / 32U, unalign);
}

template <bool E4M3>
__simd_vf__ inline void ComputeScale(__ubuf__ uint16_t* maxExp, __ubuf__ uint16_t* mxScale,
                                    __ubuf__ uint16_t* reciprocal, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t exponentMask, fp8Exponent, bias, fp8Nan, zero, nan, special;
    asc_duplicate_scalar(exponentMask, uint16_t(0x7f80U), all);
    asc_duplicate_scalar(fp8Exponent, uint16_t(E4M3 ? 0x0400U : 0x0780U), all);
    asc_duplicate_scalar(bias, uint16_t(0x7f00U), all);
    asc_duplicate_scalar(fp8Nan, uint16_t(0x00ffU), all);
    asc_duplicate_scalar(zero, uint16_t(0U), all);
    asc_duplicate_scalar(nan, uint16_t(0x7f81U), all);
    asc_duplicate_scalar(special, uint16_t(0x0040U), all);
    for (uint32_t offset = 0, remaining = count; offset < count; offset += 128U) {
        const auto mask = asc_update_mask_b16(remaining);
        vector_uint16_t maximum, shared, scale, inv;
        asc_loadalign(maximum, maxExp + offset);
        const auto valid = asc_ne(maximum, exponentMask, mask);
        const auto nonzero = asc_ne(maximum, zero, mask);
        const auto bounded = asc_le(maximum, fp8Exponent, mask);
        asc_select(maximum, fp8Exponent, maximum, bounded);
        asc_sub(shared, maximum, fp8Exponent, mask);
        asc_shiftright_scalar(scale, shared, int16_t(7), mask);
        asc_select(scale, scale, fp8Nan, valid);
        asc_select(scale, scale, zero, nonzero);
        // Pack the low byte of each uint16 into the existing E8M0 scale packs.
        asc_storealign_pack(mxScale + offset / 2U, scale, mask);
        const auto isSpecial = asc_eq(shared, bias, mask);
        asc_sub(inv, bias, shared, mask);
        asc_select(inv, inv, nan, valid);
        asc_select(inv, inv, zero, nonzero);
        asc_select(inv, special, inv, isSpecial);
        asc_storealign(reciprocal + offset, inv, mask);
    }
}

template <bool E4M3>
__simd_callee__ inline void StoreFp8(__ubuf__ int8_t* dst, vector_float values, vector_bool mask)
{
    constexpr auto quarter = std::integral_constant<asc_position_quarter_mode,
        asc_position_quarter_mode::DISPERSE_FIRST_QUARTER>{};
    if constexpr (E4M3) {
        vector_fp8_e4m3fn_t fp8;
        asc_float2e4m3_rn_sat(fp8, values, mask, quarter);
        asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ float*>(dst),
            reinterpret_cast<vector_float&>(fp8), mask);
    } else {
        vector_fp8_e5m2_t fp8;
        asc_float2e5m2_rn_sat(fp8, values, mask, quarter);
        asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ float*>(dst),
            reinterpret_cast<vector_float&>(fp8), mask);
    }
}

template <class T, bool E4M3>
__simd_vf__ inline void ComputeFp8Data(__ubuf__ T* src, __ubuf__ uint16_t* reciprocal,
                                      __ubuf__ int8_t* dst, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    constexpr auto even = std::integral_constant<asc_position_mode, asc_position_mode::EVEN>{};
    constexpr auto odd = std::integral_constant<asc_position_mode, asc_position_mode::ODD>{};
    uint32_t doubled = count * 2U, remaining = count;
    for (uint32_t offset = 0; offset < count; offset += 256U) {
        const auto input0Mask = asc_update_mask_b16(remaining);
        const auto input1Mask = asc_update_mask_b16(remaining);
        const auto float0Mask = asc_update_mask_b16(doubled);
        const auto float1Mask = asc_update_mask_b16(doubled);
        vector_uint16_t raw0, raw1, scales;
        vector_float f00, f01, f10, f11;
        asc_loadalign_deintlv(raw0, raw1, reinterpret_cast<__ubuf__ uint16_t*>(src + offset));
        asc_loadalign_brc_elem2datablock(scales, reciprocal + offset / 32U);
        if constexpr (std::is_same<T, half>::value) {
            vector_float fpScale;
            asc_half2float(f00, reinterpret_cast<vector_half&>(raw0), input0Mask, even);
            asc_half2float(f01, reinterpret_cast<vector_half&>(raw0), input0Mask, odd);
            asc_bfloat162float(fpScale, reinterpret_cast<vector_bfloat16_t&>(scales), all, even);
            asc_mul(f00, f00, fpScale, float0Mask);
            asc_mul(f01, f01, fpScale, float1Mask);
            asc_intlv(f00, f01, f00, f01);
            asc_half2float(f10, reinterpret_cast<vector_half&>(raw1), input0Mask, even);
            asc_half2float(f11, reinterpret_cast<vector_half&>(raw1), input0Mask, odd);
            asc_mul(f10, f10, fpScale, float0Mask);
            asc_mul(f11, f11, fpScale, float1Mask);
            asc_intlv(f10, f11, f10, f11);
            asc_intlv(f00, f10, f00, f10);
            asc_intlv(f01, f11, f01, f11);
            StoreFp8<E4M3>(dst + offset, f00, float0Mask);
            StoreFp8<E4M3>(dst + offset + 64U, f10, float0Mask);
            StoreFp8<E4M3>(dst + offset + 128U, f01, float1Mask);
            StoreFp8<E4M3>(dst + offset + 192U, f11, float1Mask);
        } else {
            vector_bfloat16_t input0, input1;
            asc_mul(input0, reinterpret_cast<vector_bfloat16_t&>(raw0),
                    reinterpret_cast<vector_bfloat16_t&>(scales), input0Mask);
            asc_mul(input1, reinterpret_cast<vector_bfloat16_t&>(raw1),
                    reinterpret_cast<vector_bfloat16_t&>(scales), input0Mask);
            asc_intlv(reinterpret_cast<vector_uint16_t&>(input0), reinterpret_cast<vector_uint16_t&>(input1),
                      reinterpret_cast<vector_uint16_t&>(input0), reinterpret_cast<vector_uint16_t&>(input1));
            asc_bfloat162float(f00, input0, input0Mask, even);
            asc_bfloat162float(f01, input0, input0Mask, odd);
            asc_intlv(f00, f01, f00, f01);
            asc_bfloat162float(f10, input1, input0Mask, even);
            asc_bfloat162float(f11, input1, input0Mask, odd);
            asc_intlv(f10, f11, f10, f11);
            StoreFp8<E4M3>(dst + offset, f00, float0Mask);
            StoreFp8<E4M3>(dst + offset + 64U, f01, float0Mask);
            StoreFp8<E4M3>(dst + offset + 128U, f10, float1Mask);
            StoreFp8<E4M3>(dst + offset + 192U, f11, float1Mask);
        }
        (void)input1Mask;
    }
}
}  // namespace DispatchDedupMxfp8Quant
