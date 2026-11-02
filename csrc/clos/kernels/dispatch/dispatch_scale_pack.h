#pragma once

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep {

constexpr uint32_t kDispatchScalePackThreads = 128;

// Normalize an arbitrary nonnegative-stride [token, scale] view into the compact
// row-major SHMEM staging consumed by both MTE and UDMA dispatch paths.
__simt_vf__ __launch_bounds__(kDispatchScalePackThreads) inline void
PackV1DispatchScalesSimtVf(
    const __gm__ uint32_t* src,
    __gm__ uint32_t* dst,
    uint32_t num_tokens,
    uint32_t num_scales,
    uint64_t scale_token_stride,
    uint64_t scale_hidden_stride,
    uint32_t worker_index,
    uint32_t worker_count) {
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t token = worker_index;
         token < num_tokens;
         token += worker_count) {
        for (uint32_t scale = thread_index;
             scale < num_scales;
             scale += kDispatchScalePackThreads) {
            dst[static_cast<uint64_t>(token) * num_scales + scale] = src[
                static_cast<uint64_t>(token) * scale_token_stride +
                static_cast<uint64_t>(scale) * scale_hidden_stride];
        }
    }
    asc_threadfence();
}

}  // namespace ascend_deepep
