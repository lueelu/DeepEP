#pragma once
#include <c_api/asc_memory_vector.h>
#include <c_api/asc_simd.h>
#include <c_api/spr/spr.h>
#include <c_api/sys_var/sys_var.h>

// Count operations use physical UB addresses. Offset zero is a valid address.
namespace LlCountC {
template <pipe_t From, pipe_t To>
__aicore__ inline void Sync(event_t id = EVENT_ID0)
{
    asc_sync_notify(From, To, id);
    asc_sync_wait(From, To, id);
}

template <class T>
__aicore__ inline __ubuf__ T* Ub(uint32_t bytes)
{
    return reinterpret_cast<__ubuf__ T*>(uint64_t(bytes));
}

__aicore__ inline void Load(__ubuf__ void* dst, __gm__ void* src,
                            uint32_t bytes, uint32_t bursts = 1U, uint64_t gmGap = 0U)
{
    // align_v2 takes byte pitches between burst starts, including the copied
    // bytes. A zero UB pitch would overwrite the first burst repeatedly.
    const uint32_t ubPitch = (bytes + 31U) / 32U * 32U;
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(dst),
        reinterpret_cast<__gm__ uint8_t*>(src), bursts, bytes, 0U, 0U, false,
        asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, bytes + gmGap, ubPitch);
}

__aicore__ inline void Store(__gm__ void* dst, __ubuf__ void* src,
                             uint32_t bytes, uint32_t bursts = 1U,
                             uint64_t gmGap = 0U, bool uncached = false)
{
    const uint32_t ubPitch = (bytes + 31U) / 32U * 32U;
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst),
        reinterpret_cast<__ubuf__ uint8_t*>(src), bursts, bytes,
        uncached ? asc_store_l2_cache_mode::NOTALLOC_CLEAN :
                   asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, bytes + gmGap, ubPitch);
}

__simd_vf__ inline void Fill(__ubuf__ uint32_t* dst, uint32_t value, uint32_t words)
{
    vector_uint32_t values;
    asc_duplicate_scalar(values, value, asc_create_mask_b32(PAT_ALL));
    for (uint32_t i = 0, remaining = words; i < words; i += 64U)
        asc_storealign(dst + i, values, asc_update_mask_b32(remaining));
}

// Write only word zero of each 32B record. Other words remain untouched.
__simd_vf__ inline void RecordReady(__ubuf__ uint32_t* dst, uint32_t value, uint32_t records)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    vector_int32_t lane = asc_arange(0), sevens;
    asc_duplicate_scalar(sevens, int32_t(7), all);
    asc_and(lane, lane, sevens, all);
    const auto first = asc_eq_scalar(lane, 0, all);
    vector_uint32_t values;
    asc_duplicate_scalar(values, value, all);
    for (uint32_t i = 0, remaining = records * 8U; i < records * 8U; i += 64U)
        asc_storealign(dst + i, values, asc_and(first, asc_update_mask_b32(remaining), all));
}

// Register gather indices are element indices in [0,63]. Only active record
// lanes are stored; a padded 256B source load handles the final eight records.
__simd_vf__ inline void GatherWords(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
                                   uint32_t records, int32_t word)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    vector_int32_t index = asc_arange(0), sevens;
    asc_duplicate_scalar(sevens, int32_t(7), all);
    asc_and(index, index, sevens, all);
    asc_mul_scalar(index, index, 8, all);
    asc_add_scalar(index, index, word, all);
    for (uint32_t i = 0; i < records; i += 8U) {
        vector_uint32_t input, output;
        asc_loadalign(input, src + i * 8U);
        asc_gather(output, input, (vector_uint32_t&)index);
        uint32_t active = records - i < 8U ? records - i : 8U;
        asc_storealign(dst + i, output, asc_update_mask_b32(active));
    }
}

template <bool Floating>
__simd_vf__ inline void Sum(__ubuf__ uint32_t* out, __ubuf__ uint32_t* src, uint32_t words)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    uint32_t oneWord = 1U;
    const auto one = asc_update_mask_b32(oneWord);
    vector_uint32_t total;
    asc_duplicate_scalar(total, 0U, all);
    for (uint32_t i = 0, remaining = words; i < words; i += 64U) {
        vector_uint32_t input, part;
        asc_loadalign(input, src + i);
        const auto mask = asc_update_mask_b32(remaining);
        if constexpr (Floating) {
            asc_reduce_sum((vector_float&)part, (vector_float&)input, mask);
            asc_add((vector_float&)total, (vector_float&)total, (vector_float&)part, one);
        } else {
            asc_reduce_sum(part, input, mask);
            asc_add(total, total, part, one);
        }
    }
    asc_storealign(out, total, one);
}
}
