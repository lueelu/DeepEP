// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

// Preserve Notify's public six-word table for combine. Build this cached,
// dispatch-only view once: prefix[lane][source] and [primary row, output row].
// Every input record belongs to at most one lane, so world*capacity pairs fit.
template <bool Emit>
__simt_vf__ __launch_bounds__(128) inline void BuildForwardPairsVf(__gm__ int32_t* forward, __gm__ int32_t* counts,
                                                                   __gm__ uint32_t* prefix, __gm__ uint32_t* pairs,
                                                                   uint32_t rank, uint32_t world, uint32_t capacity,
                                                                   uint32_t rows, uint32_t core = 0U,
                                                                   uint32_t cores = 1U)
{
    if constexpr (Emit) {
        if (asc_ldcg(prefix + 8U * (world + 1U))) return;
    }
    for (uint32_t task = core * 128U + threadIdx.x; task < 8U * world; task += cores * 128U) {
        const uint32_t lane = task / world, source = task % world;
        const uint32_t index = lane * (world + 1U) + source;
        const uint32_t peer = rank / 8U * 8U + lane;
        const int32_t count = asc_ldcg(counts + source);
        if (count < 0 || uint32_t(count) > capacity) {
            if constexpr (!Emit) asc_stcg(prefix + index, 0xffffffffU);
            continue;
        }
        uint32_t cursor = 0U;
        if constexpr (Emit) cursor = asc_ldcg(prefix + index);
        bool invalid = false;
        for (uint32_t i = 0; i < uint32_t(count); ++i) {
            auto* record = forward + (uint64_t(source) * capacity + i) * 6U;
            const int32_t target = asc_ldcg(record + 3U);
            if (target < 0) continue;  // Primary-only Notify record.
            const int32_t primary = asc_ldcg(record + 2U), output = asc_ldcg(record + 4U);
            if (uint32_t(target) / 8U != rank / 8U || uint32_t(target) >= world || primary < 0 ||
                uint32_t(primary) >= rows || output < 0 || uint32_t(output) >= rows) {
                invalid = true;
                continue;
            }
            if (uint32_t(target) != peer) continue;
            if constexpr (Emit) {
                asc_stcg(pairs + uint64_t(cursor) * 2U, uint32_t(primary));
                asc_stcg(pairs + uint64_t(cursor) * 2U + 1U, uint32_t(output));
            }
            ++cursor;
        }
        if constexpr (!Emit) asc_stcg(prefix + index, invalid ? 0xffffffffU : cursor);
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void PrefixForwardPairsVf(__gm__ uint32_t* prefix, uint32_t world,
                                                                    uint32_t capacity)
{
    if (threadIdx.x == 0U) {
        uint32_t total = 0U, error = 0U;
        for (uint32_t lane = 0; lane < 8U; ++lane) {
            for (uint32_t source = 0; source < world; ++source) {
                auto* slot = prefix + lane * (world + 1U) + source;
                const uint32_t count = asc_ldcg(slot);
                asc_stcg(slot, total);
                if (count == 0xffffffffU)
                    error = 1U;
                else
                    total += count;
            }
            asc_stcg(prefix + lane * (world + 1U) + world, total);
        }
        asc_stcg(prefix + 8U * (world + 1U), error || uint64_t(total) > uint64_t(world) * capacity);
    }
    asc_threadfence();
}

__aicore__ inline void BuildForwardPlan(GM_ADDR forward, GM_ADDR counts, GM_ADDR plan,
                                        const ascend_deepep::DispatchTiling& t)
{
    using namespace ascend_deepep;
    auto* prefix = reinterpret_cast<__gm__ uint32_t*>(plan + t.forward_plan_offset);
    auto* pairs = reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<GM_ADDR>(prefix) + t.forward_header_bytes);
    asc_vf_call<BuildForwardPairsVf<false>>(dim3(128), reinterpret_cast<__gm__ int32_t*>(forward),
                                            reinterpret_cast<__gm__ int32_t*>(counts), prefix, pairs, t.rank, t.world,
                                            t.tokens * t.topk, t.rows, GetBlockIdx(), 32U);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    SyncAll<true>();
    if (GetBlockIdx() == 0U) asc_vf_call<PrefixForwardPairsVf>(dim3(128), prefix, t.world, t.tokens * t.topk);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    SyncAll<true>();
    asc_vf_call<BuildForwardPairsVf<true>>(dim3(128), reinterpret_cast<__gm__ int32_t*>(forward),
                                           reinterpret_cast<__gm__ int32_t*>(counts), prefix, pairs, t.rank, t.world,
                                           t.tokens * t.topk, t.rows, GetBlockIdx(), 32U);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SyncAll<true>();
}
