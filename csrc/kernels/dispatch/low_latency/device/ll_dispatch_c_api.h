#pragma once
#include <cstdint>
#include <c_api/asc_memory_vector.h>
#include <c_api/asc_simd.h>
#include <c_api/spr/spr.h>
#include <c_api/sys_var/sys_var.h>
#include "ll_count_c_api.h"
#include "../common/ll_core_config.h"
#include "../common/ll_sender_input_reuse.h"
#include "../../../common/ep_packet_receive_layout.h"

// Physical UB layouts and explicit pipeline dependencies, as used by DeepEP-Ascend.
// This layer does not construct tensors, queues, buffers, or allocator state.
namespace LlDispatchC {
constexpr uint32_t kUbBytes = DispatchDedup::kLlUbBytes;

__aicore__ inline uint32_t Align(uint32_t bytes, uint32_t alignment = 32U)
{
    return (bytes + alignment - 1U) / alignment * alignment;
}

template <class T>
__aicore__ inline __ubuf__ T* Ub(uint32_t offset)
{
    return reinterpret_cast<__ubuf__ T*>(uint64_t(offset));
}

class UbLayout {
    uint32_t end_;
public:
    __aicore__ explicit UbLayout(uint32_t reserved = 0U) : end_(Align(reserved)) {}
    template <class T = uint8_t>
    __aicore__ inline __ubuf__ T* Allocate(uint32_t bytes)
    {
        const uint32_t size = Align(bytes);
        if (end_ > kUbBytes || size > kUbBytes - end_) { trap(); }
        auto* result = Ub<T>(end_);
        end_ += size;
        return result;
    }
    __aicore__ inline uint32_t UsedBytes() const { return end_; }
};

template <pipe_t From, pipe_t To>
__aicore__ inline void Sync(event_t id = EVENT_ID0)
{
    asc_sync_notify(From, To, id);
    asc_sync_wait(From, To, id);
}

__aicore__ inline void Load(__ubuf__ void* dst, __gm__ void* src, uint32_t bytes,
                           uint32_t bursts = 1U, uint64_t gmGap = 0U,
                           uint32_t ubGap = 0U, bool uncached = false)
{
    if (!bytes || !bursts) return;
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(dst),
        reinterpret_cast<__gm__ uint8_t*>(src), bursts, bytes, 0U, 0U, false,
        uncached ? asc_load_l2_cache_mode::NOTALLOC_CLEAN :
                   asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
        bytes + gmGap, Align(bytes) + ubGap);
}

__aicore__ inline void Store(__gm__ void* dst, __ubuf__ void* src, uint32_t bytes,
                            uint32_t bursts = 1U, uint64_t gmGap = 0U,
                            uint32_t ubGap = 0U, bool uncached = false)
{
    if (!bytes || !bursts) return;
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst),
        reinterpret_cast<__ubuf__ uint8_t*>(src), bursts, bytes,
        uncached ? asc_store_l2_cache_mode::NOTALLOC_CLEAN :
                   asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
        bytes + gmGap, Align(bytes) + ubGap);
}

__simd_vf__ inline void FillBytes(__ubuf__ uint8_t* dst, uint8_t value, uint32_t bytes)
{
    vector_uint8_t values;
    asc_duplicate_scalar(values, value, asc_create_mask_b8(PAT_ALL));
    for (uint32_t offset = 0, remaining = bytes; offset < bytes; offset += 256U)
        asc_storealign(dst + offset, values, asc_update_mask_b8(remaining));
}

__simd_vf__ inline void FillWords(__ubuf__ uint32_t* dst, uint32_t value, uint32_t words)
{
    vector_uint32_t values;
    asc_duplicate_scalar(values, value, asc_create_mask_b32(PAT_ALL));
    for (uint32_t offset = 0, remaining = words; offset < words; offset += 64U)
        asc_storealign(dst + offset, values, asc_update_mask_b32(remaining));
}

__simd_vf__ inline void FillPacketFlags(__ubuf__ uint8_t* packet, uint32_t blocks)
{
    vector_uint32_t ones;
    asc_duplicate_scalar(ones, 0x3f800000U, asc_create_mask_b32(PAT_ALL));
    for (uint32_t block = 0; block < blocks; ++block) {
        uint32_t words = 8U;
        asc_storealign(reinterpret_cast<__ubuf__ uint32_t*>(packet + block * 512U + 480U),
                       ones, asc_update_mask_b32(words));
    }
}

// Preserve the original 480B payload blocks and preinitialized 32B wire flags.
__simd_vf__ inline void PackPayload(__ubuf__ uint8_t* wire, __ubuf__ uint8_t* payload,
                                   uint32_t blocks)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    for (uint32_t block = 0; block < blocks; ++block) {
        vector_uint32_t first, last;
        auto* input = reinterpret_cast<__ubuf__ uint32_t*>(payload + block * 480U);
        auto* output = reinterpret_cast<__ubuf__ uint32_t*>(wire + block * 512U);
        asc_loadalign(first, input);
        asc_loadalign(last, input + 64U);
        asc_storealign(output, first, all);
        uint32_t words = 56U;
        asc_storealign(output + 64U, last, asc_update_mask_b32(words));
    }
}

__simd_vf__ inline void SetPacketMarker(__ubuf__ uint8_t* wire, float value)
{
    vector_float marker;
    const auto all = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(marker, value, all);
    uint32_t one = 1U;
    asc_storealign(reinterpret_cast<__ubuf__ float*>(wire + DispatchDedup::PacketReceiveLayout::kMarkerOffset),
                   marker, asc_update_mask_b32(one));
}

__simd_vf__ inline void FillPrequantPadding(__ubuf__ uint8_t* input,
                                           DispatchDedup::LlPrequantPadding padding)
{
    vector_uint8_t zeros;
    const auto all = asc_create_mask_b8(PAT_ALL);
    asc_duplicate_scalar(zeros, 0U, all);
    for (uint32_t offset = 0, remaining = padding.inputBytes; offset < padding.inputBytes; offset += 256U)
        asc_storealign(input + padding.inputOffset + offset, zeros, asc_update_mask_b8(remaining));
    if (padding.tailBlockMask) {
        vector_uint32_t wordZeros;
        asc_duplicate_scalar(wordZeros, 0U, asc_create_mask_b32(PAT_ALL));
        // The mask clears a suffix of eight words in this one 32B block.
        vector_int32_t indices = asc_arange(0);
        uint32_t first = 0U;
        while (!(padding.tailBlockMask & (1U << first))) ++first;
        uint32_t words = 8U;
        const auto active = asc_update_mask_b32(words);
        const auto suffix = asc_ge_scalar(indices, int32_t(first), active);
        asc_storealign(reinterpret_cast<__ubuf__ uint32_t*>(input + padding.tailBlockOffset), wordZeros, suffix);
    }
    for (uint32_t offset = 0, remaining = padding.tailBytes; offset < padding.tailBytes; offset += 256U)
        asc_storealign(input + padding.tailOffset + offset, zeros, asc_update_mask_b8(remaining));
}

// Compute the same original-route q without FP32 subnormal count arithmetic.
// The caller retains the first-token special case and waits V_S before reading.
__simd_vf__ inline void RoutePrefixCount(__ubuf__ uint32_t* result,
                                        __ubuf__ int32_t* routes,
                                        uint32_t words, int32_t expert)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    uint32_t oneWord = 1U;
    const auto one = asc_update_mask_b32(oneWord);
    vector_uint32_t total;
    asc_duplicate_scalar(total, 0U, all);
    for (uint32_t offset = 0, remaining = words; offset < words; offset += 64U) {
        vector_int32_t ids;
        vector_uint32_t matches, part;
        asc_loadalign(ids, routes + offset);
        const auto active = asc_update_mask_b32(remaining);
        const auto equal = asc_eq_scalar(ids, expert, active);
        asc_duplicate_scalar(matches, 0U, all);
        asc_duplicate_scalar_merge(matches, 1U, equal);
        asc_reduce_sum(part, matches, active);
        asc_add(total, total, part, one);
    }
    asc_storealign(result, total, one);
}

// Each packet has an independent row; never combine flags from different packets.
__simd_vf__ inline void SumPacketFlags(__ubuf__ float* results, __ubuf__ float* flags,
                                      uint32_t rows, uint32_t wordsPerRow)
{
    const auto all = asc_create_mask_b32(PAT_ALL);
    uint32_t oneWord = 1U;
    const auto one = asc_update_mask_b32(oneWord);
    for (uint32_t row = 0; row < rows; ++row) {
        vector_float total;
        asc_duplicate_scalar(total, 0.0f, all);
        for (uint32_t offset = 0, remaining = wordsPerRow; offset < wordsPerRow; offset += 64U) {
            vector_float input, part;
            asc_loadalign(input, flags + row * wordsPerRow + offset);
            asc_reduce_sum(part, input, asc_update_mask_b32(remaining));
            asc_add(total, total, part, one);
        }
        // Every output uses a 32B aligned slot; Scalar reads results[row * 8].
        asc_storealign(results + row * 8U, total, one);
    }
}
}  // namespace LlDispatchC
