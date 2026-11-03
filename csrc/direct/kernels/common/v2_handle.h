#pragma once

#include "common/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Notify's existing local barrier publishes both input arrays before this VF.
__simt_vf__ __launch_bounds__(128) inline void V2HandleCountsVf(
    const __gm__ int32_t* ranks, const __gm__ int32_t* counts, __gm__ int32_t* rank_prefix,
    __gm__ int32_t* expert_prefix, uint32_t rank, uint32_t world, uint32_t experts, uint32_t alignment)
{
    for (uint32_t p = threadIdx.x; p < world; p += 128)
        rank_prefix[p] = ranks[uint64_t(p) * world + rank];
    if (expert_prefix != nullptr && threadIdx.x == 0) {
        int32_t total = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            total += (counts[e] + alignment - 1) / alignment * alignment;
            expert_prefix[e] = total;
        }
    }
    asc_threadfence();
}

// Eight-row ownership keeps adjacent AIV writers on separate 32-byte lines,
// including odd K. No UB storage or peer traffic is introduced.
__simt_vf__ __launch_bounds__(128) inline void V2HandleSourceVf(
    const __gm__ int32_t* ranks, const __gm__ int32_t* source, const __gm__ topk_idx_t* topk,
    __gm__ int32_t* output, uint32_t rank, uint32_t world, uint32_t capacity, uint32_t k,
    uint32_t rows, uint32_t core, uint32_t cores)
{
    const uint32_t groups = (rows + 7) / 8;
    const uint32_t per_core = (groups + cores - 1) / cores;
    const uint32_t begin = core * per_core * 8;
    const uint32_t limit = (core + 1) * per_core * 8;
    const uint32_t end = limit < rows ? limit : rows;
    const int32_t valid = ranks[uint64_t(world - 1) * world + rank];
    uint32_t source_rank = 0;
    for (uint32_t row = begin + threadIdx.x; row < end; row += 128) {
        const uint64_t base = uint64_t(row) * (k + 2);
        for (uint32_t col = 0; col < k + 2; ++col)
            output[base + col] = -1;
        if (row >= uint32_t(valid))
            continue;
        while (row >= uint32_t(ranks[uint64_t(source_rank) * world + rank]))
            ++source_rank;
        uint32_t master = 0;
        while (master < k && topk[uint64_t(row) * k + master] < 0)
            ++master;
        output[base] = int32_t(uint64_t(source_rank) * capacity + source[row]);
        output[base + 1] = int32_t(source_rank * k + master);
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void V2HandleSlotsVf(
    const __gm__ topk_idx_t* topk, const __gm__ int32_t* head, const __gm__ int32_t* channels,
    __gm__ int32_t* output, uint32_t rank, uint32_t world, uint32_t capacity, uint32_t experts_per_rank,
    uint32_t k, uint32_t tokens, uint32_t channel_count, uint32_t core, uint32_t cores)
{
    const uint32_t per_core = (((tokens + 7) / 8) + cores - 1) / cores;
    const uint32_t begin = core * per_core * 8;
    const uint32_t limit = (core + 1) * per_core * 8;
    const uint32_t end = limit < tokens ? limit : tokens;
    const uint32_t channel_tokens = (tokens + channel_count - 1) / channel_count;
    for (uint32_t t = begin + threadIdx.x; t < end; t += 128) {
        uint64_t seen = 0;
        const uint32_t channel = t / channel_tokens;
        for (uint32_t j = 0; j < k; ++j) {
            const uint64_t i = uint64_t(t) * k + j;
            output[i] = -1;
            const auto expert = topk[i];
            if (expert < 0 || uint64_t(expert) >= uint64_t(world) * experts_per_rank)
                continue;
            const uint32_t peer = uint32_t(expert) / experts_per_rank;
            bool duplicate = false;
            if (world <= 64) {
                const uint64_t bit = uint64_t(1) << peer;
                duplicate = (seen & bit) != 0;
                seen |= bit;
            } else {
                for (uint32_t previous = 0; previous < j; ++previous) {
                    const auto e = topk[uint64_t(t) * k + previous];
                    duplicate |= e >= 0 && uint64_t(e) / experts_per_rank == peer;
                }
            }
            if (duplicate)
                continue;
            const int32_t ordinal = head[uint64_t(t) * world + peer];
            const int32_t prefix = channel ? channels[uint64_t(peer) * channel_count + channel - 1] : 0;
            output[i] = int32_t(uint64_t(rank) * capacity + prefix + ordinal);
        }
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void V2HandleExpandMappingVf(
    const __ubuf__ int32_t* mapping, __gm__ int32_t* metadata, uint32_t begin, uint32_t rows, uint32_t k)
{
    for (uint32_t i = threadIdx.x; i < rows * k; i += 128)
        metadata[uint64_t(begin + i / k) * (k + 2) + 2 + i % k] = mapping[i];
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void V2HandleExpandedCountsVf(
    const __ubuf__ int32_t* offsets, const __ubuf__ int32_t* counts,
    __gm__ int32_t* prefix, uint32_t experts)
{
    for (uint32_t e = threadIdx.x; e < experts; e += 128)
        prefix[e] = offsets[e] + counts[e];
    asc_threadfence();
}
#endif
} // namespace ascend_deepep
