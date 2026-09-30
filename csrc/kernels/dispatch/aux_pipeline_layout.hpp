// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include "scale_pipeline_layout.hpp"

namespace ascend_deepep::aux_pipeline {
struct Layout : scale_pipeline::Layout {
    uint32_t weights_offset = 0;  // Input route weights; local-copy UB unless aux_routes.
};
// Weight-only: sixty 8B slot+weight records, 24B padding and an 8B
// generation flag fill each independently ordered 512B block.
// Reuse existing scratch/local/WQE TBufs, never add TPipe allocations.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr Layout
    Make(uint32_t world, uint32_t stride)
{
    Layout p{};
    if ((world < 2U || world > 256U || (world & (world - 1U))) || stride != 0U) {
        return p;
    }
    p.peers = (world + 31U) / 32U;
    for (uint32_t batch = 960U; batch >= 60U; batch -= 60U) {
        const uint32_t plane = batch / 60U * 512U;
        const uint32_t fixed = 2U * p.peers * plane + 512U * p.peers + 512U;
        for (uint32_t tile = 4096U; tile >= 512U; tile -= 256U) {
            const bool aux = tile <= 2048U;
            const uint32_t scan = ((aux ? 2U : 4U) * tile * 4U + tile / 8U + 511U) / 512U * 512U;
            if (fixed + scan > scale_pipeline::kScratchBytes) {
                continue;
            }
            p.batch = batch;
            p.tile = tile;
            p.aux_routes = aux ? 1U : 0U;
            p.weights_offset = tile * 4U + tile / 8U;  // after selected + mask in scratch
            p.packed_offset = scan;
            p.buffer_bytes = plane;
            p.slots_in_buffer = batch * stride;
            p.headers_offset = scan + 2U * p.peers * plane;
            p.errors_offset = p.headers_offset + 512U * p.peers;
            p.bytes = p.errors_offset + 512U;
            return p;
        }
    }
    return Layout{};
}
static_assert(Make(128U, 0U).batch == 960U && Make(128U, 0U).bytes <= 132096U);
static_assert(Make(256U, 0U).batch > 0U && Make(256U, 0U).bytes <= 132096U);
}  // namespace ascend_deepep::aux_pipeline
