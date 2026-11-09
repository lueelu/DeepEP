#pragma once

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep {

constexpr uint32_t kDispatchScalePackThreads = 128;

// Build each peer's compact send order. Channels own disjoint prefix ranges;
// source views may be strided and all copies preserve the original 32-bit bits.
__simt_vf__ __launch_bounds__(kDispatchScalePackThreads) inline void
PackDispatchPeerScalesVf(const __gm__ uint32_t* src, __gm__ uint32_t* dst,
    const __gm__ int32_t* plan, const __gm__ int32_t* prefixes,
    uint32_t rank, uint32_t world, uint32_t channels, uint32_t plan_stride,
    uint32_t tokens, uint32_t scales, uint64_t token_stride, uint64_t scale_stride,
    uint32_t core, uint32_t cores)
{
    for (uint32_t task = core; task < world * channels; task += cores) {
        const uint32_t peer = task / channels, channel = task % channels;
        if (peer == rank) continue; // Existing local-copy owner handles self.
        const uint32_t begin = channel ? prefixes[task - 1] : 0;
        const uint32_t end = prefixes[task];
        if (end < begin || end > tokens) { __asc_simt_vf::__trap(); return; }
        // One warp per row avoids a dynamic 64-bit divide for every scale word.
        for (uint32_t row = threadIdx.x / 32U; row < end - begin; row += kDispatchScalePackThreads / 32U) {
            const int32_t token = plan[uint64_t(task) * plan_stride + row];
            if (token < 0 || uint32_t(token) >= tokens) { __asc_simt_vf::__trap(); return; }
            for (uint32_t col = threadIdx.x % 32U; col < scales; col += 32U)
                dst[(uint64_t(peer) * tokens + begin + row) * scales + col] =
                    src[uint64_t(token) * token_stride + uint64_t(col) * scale_stride];
        }
    }
    asc_threadfence();
}

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
