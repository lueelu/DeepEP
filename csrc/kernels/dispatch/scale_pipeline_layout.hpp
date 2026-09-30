// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define SCALE_PIPE_INLINE __aicore__ inline
#else
#define SCALE_PIPE_INLINE inline
#endif

namespace ascend_deepep::scale_pipeline {
constexpr uint32_t kScratchBytes = 132096U;
constexpr uint32_t kWorkers = 32U, kMaxPeers = 8U, kMaxBuffers = 16U;
// Keep manual completion tickets on IDs 0..3; IDs 6/7 may be used by AscendC.
// Larger worlds retain all UB buffers and share tickets after a local wait.
constexpr uint32_t kCompletionEvents = 4U;
constexpr uint32_t kErrorBytes = 128U * 4U;
struct Layout {
    uint32_t peers = 0, batch = 0, tile = 0;
    uint32_t aux_routes = 0;  // Reuse scale AIV's idle 16KiB local + 8KiB WQE buffers.
    uint32_t packed_offset = 0, buffer_bytes = 0, slots_in_buffer = 0;
    uint32_t headers_offset = 0, errors_offset = 0, bytes = 0;
};

// No new TPipe allocation. If the full route scratch cannot fit, reuse the
// scale-only AIV's idle local-copy and WQE buffers for ranks/slots/indices.
// Two 224B scale rows + two slots + generation occupy one 512B block.
// Packet and header blocks stay in slot_buffer; every remote destination is 512B aligned.
// Reduce scan tile first, then batch if needed. Batches contain whole two-row
// blocks. Packet and header offsets are also 512B aligned inside the UB.
SCALE_PIPE_INLINE constexpr Layout Make(uint32_t world, uint32_t stride)
{
    Layout p{};
    if (!world || world > 256U || stride != 224U) {
        return p;
    }
    p.peers = (world + kWorkers - 1U) / kWorkers;
    for (uint32_t batch = 64U; batch >= 8U; batch -= 8U) {
        const uint32_t plane = batch / 2U * 512U;
        const uint32_t fixed = 2U * p.peers * plane + 512U * p.peers + kErrorBytes;
        for (uint32_t tile = 4096U; tile >= 512U; tile -= 256U) {
            const bool aux = tile <= 2048U;
            const uint32_t scan = ((aux ? 1U : 4U) * tile * 4U + tile / 8U + 511U) / 512U * 512U;
            if (fixed + scan > kScratchBytes) {
                continue;
            }
            p.batch = batch;
            p.tile = tile;
            p.packed_offset = scan;
            p.aux_routes = aux ? 1U : 0U;
            p.buffer_bytes = plane;
            p.slots_in_buffer = plane;
            p.headers_offset = scan + 2U * p.peers * plane;
            p.errors_offset = p.headers_offset + 512U * p.peers;
            p.bytes = p.errors_offset + kErrorBytes;
            return p;
        }
    }
    return Layout{};
}
static_assert(Make(32U, 224U).batch == 64U);
static_assert(Make(64U, 224U).batch == 64U && Make(64U, 224U).tile == 3840U);
static_assert(Make(128U, 224U).batch == 56U && Make(128U, 224U).bytes <= kScratchBytes);
static_assert(Make(256U, 224U).batch == 24U && Make(256U, 224U).bytes <= kScratchBytes);
}  // namespace ascend_deepep::scale_pipeline
#undef SCALE_PIPE_INLINE
