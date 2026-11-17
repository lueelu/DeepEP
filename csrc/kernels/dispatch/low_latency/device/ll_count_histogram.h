#pragma once
#include <kernel_operator.h>

#include <c_api/asc_simd.h>

namespace DispatchDedup {

/**
 *  for (int i = 0; i < num_i32_elems; ++ i)
 *      if (offset <= data[i] and data[i] < offset + num_bins)
 *          acc[data[i] - offset + 1] ++;
 */
__simd_vf__ void histogram_accumulate_u16(__ubuf__ int32_t* data, __ubuf__ uint16_t* acc,
                                          const int num_i32_elems, const int offset, const int num_bins) {
    static constexpr int kNumIdsPerVector = sizeof(vector_int32_t) / sizeof(int32_t);
    vector_bool mask_b16 = asc_create_mask_b16(PAT_ALL);
    vector_int32_t value_vec;
    vector_uint16_t count_vec;
    asc_duplicate_scalar(count_vec, static_cast<uint16_t>(0), mask_b16);

    uint32_t remaining = static_cast<uint32_t>(num_i32_elems);
    for (int elem_offset = 0; elem_offset < num_i32_elems; elem_offset += kNumIdsPerVector) {
        uint32_t active_bytes = remaining * static_cast<uint32_t>(sizeof(int32_t));
        const vector_bool mask_b32 = asc_update_mask_b32(remaining);
        asc_loadalign(value_vec, data + elem_offset);
        asc_max_scalar(value_vec, value_vec, offset - 1, mask_b32);
        asc_min_scalar(value_vec, value_vec, offset + num_bins, mask_b32);
        asc_add_scalar(value_vec, value_vec, 1 - offset, mask_b32);

        const vector_bool mask_b8 = asc_update_mask_b8(active_bytes);
        asc_frequency_histogram_bin0(count_vec, (vector_uint8_t&)value_vec, mask_b8);
    }
    asc_storealign(acc, count_vec, mask_b16);
}

}  // namespace DispatchDedup
