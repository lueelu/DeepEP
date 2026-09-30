// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>
#include "../notify/tiling.hpp"
#include "peer_protocol.hpp"
#include "aux_pipeline_layout.hpp"

namespace ascend_deepep {
constexpr uint32_t kMaxDispatchRanks = 256U;

inline constexpr uint64_t DispatchPlanPhaseBytes(uint32_t world, uint32_t /*tokens*/)
{
    return uint64_t(world) * 16U;
}
inline constexpr uint64_t DispatchPlanRowsOffset(uint32_t world)
{
    return (64U + uint64_t(world) * 32U + 511U) / 512U * 512U;
}
inline constexpr uint64_t DispatchPlanScratchOffset(uint32_t world, uint32_t tokens, uint32_t topk)
{
    return DispatchPlanRowsOffset(world) + (uint64_t(tokens) * topk * 8U + 511U) / 512U * 512U;
}
inline constexpr uint64_t DispatchPlanMailboxOffset(uint32_t world, uint32_t tokens, uint32_t topk)
{
    return DispatchPlanScratchOffset(world, tokens, topk) +
           (uint64_t(world) * ((tokens + 31U) / 32U) * 20U + 511U) / 512U * 512U;
}
inline constexpr uint64_t DispatchForwardOffset(uint32_t world, uint32_t tokens, uint32_t topk)
{
    return DispatchPlanMailboxOffset(world, tokens, topk) + kClosSplitMailboxBytes;
}
inline constexpr uint64_t DispatchForwardHeaderBytes(uint32_t world)
{
    return (uint64_t(8U * (world + 1U) + 1U) * 4U + 511U) / 512U * 512U;
}
inline constexpr uint64_t DispatchPlanBytes(uint32_t world, uint32_t tokens, uint32_t topk)
{
    return DispatchForwardOffset(world, tokens, topk) + DispatchForwardHeaderBytes(world) +
           (uint64_t(world) * tokens * topk * 8U + 511U) / 512U * 512U;
}
struct DispatchTiling {
    // Framework-owned live GM buffers. Payload addresses are absolute virtual addresses,
    // never symmetric-heap offsets and never passed through aclshmem_ptr.
    uint64_t input_address = 0;
    uint64_t output_address = 0, output_weights_address = 0, output_scales_address = 0;
    uint64_t detail_profile = 0;  // Optional [64,32], see profile.hpp.
    uint32_t rotation_wqes = 4, detail_mode = 0;
    uint64_t weight_profile = 0;  // Optional [32,16] rear-core aggregate counters.
    uint64_t weight_inbox = 0, weight_inbox_stride = 0;
    uint32_t weight_capacity = 0, output_rows = 0;
    uint64_t profile = 0;  // Optional [64,32] uint64 per-AIV diagnostics, outside SHMEM.
    uint64_t peer_table = 0, generation = 0, peer_book = 0, done = 0, plan_phase_bytes = 0;
    uint64_t forward_plan_offset = 0, forward_header_bytes = 0;
    uint64_t plan_rows_offset = 0, plan_scratch_offset = 0, plan_mailbox_offset = 0;
    uint32_t rank, world, tokens, topk, rows, address_stride, fp8;
    uint64_t sync, book, input, output, weights, scales, inbox, inbox_stride, workspace_bytes;
};

// Caller validates bounds before narrowing to this device ABI. All byte
// calculations stay uint64_t, including receive capacity and scale inboxes.
inline DispatchTiling MakeDispatchTiling(uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk,
                                         uint32_t experts, uint32_t rows, bool fp8, bool weights = false)
{
    DispatchTiling t{};
    t.plan_rows_offset = DispatchPlanRowsOffset(world);
    t.plan_scratch_offset = DispatchPlanScratchOffset(world, tokens, topk);
    t.plan_mailbox_offset = DispatchPlanMailboxOffset(world, tokens, topk);
    t.plan_phase_bytes = DispatchPlanPhaseBytes(world, tokens);
    t.forward_plan_offset = DispatchForwardOffset(world, tokens, topk);
    t.forward_header_bytes = DispatchForwardHeaderBytes(world);
    t.rank = rank;
    t.world = world;
    t.tokens = tokens;
    t.topk = topk;
    t.rows = rows;
    t.fp8 = fp8;
    t.address_stride = 1;
    while (t.address_stride < rows) {
        t.address_stride <<= 1;
    }
    uint64_t cursor = MakeNotifyTiling(rank, world, tokens, topk, experts, 256, 7168, 32).workspace_bytes;
    auto reserve = [&cursor](uint64_t bytes) {
        const auto offset = cursor;
        cursor += (bytes + 511) / 512 * 512;
        return offset;
    };
    const uint64_t row_bytes = fp8 ? 7168 : 14336;
    t.sync = reserve(uint64_t(world) * 49152 + 65536);
    // Retain unused legacy payload slices so Python layout/Combine partitioning
    // stay compatible. Dispatch accesses only the per-call framework GM virtual addresses.
    t.book = reserve(uint64_t(world) * 16);
    t.input = reserve(uint64_t(tokens) * row_bytes);
    t.output = reserve(uint64_t(rows) * row_bytes);
    t.weights = reserve(uint64_t(rows) * 4);
    t.scales = reserve(fp8 ? uint64_t(rows) * 224 : 0);
    t.inbox_stride = fp8 ? 512U + ((uint64_t(tokens) * topk + 1U) / 2U) * 512U : 0U;
    t.inbox = reserve(uint64_t(world) * t.inbox_stride);
    t.peer_book = reserve(uint64_t(world) * 512U);
    t.done = reserve(uint64_t(world) * 512U);
    if (weights) {
        const uint64_t routes = uint64_t(tokens) * topk;
        t.weight_capacity = (routes + 59U) / 60U * 60U;
        t.weight_inbox_stride = 512U + uint64_t(t.weight_capacity / 60U) * 512U;
    }
    t.weight_inbox = reserve(uint64_t(world) * t.weight_inbox_stride);
    t.workspace_bytes = cursor;
    return t;
}
}  // namespace ascend_deepep
