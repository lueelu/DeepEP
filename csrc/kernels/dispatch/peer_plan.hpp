// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include "forward_plan.hpp"

// meta: int32 [2, world, 2, 2] = phase/peer/QP/{begin,count}.
// rows: int32 [tokens*topk, 2] = {token, output row}; only directory ranges are valid.
// scratch: int32 [world, ceil(tokens/32), 5] = primary prefix and four bucket prefixes.
// Each peer/tile owns its scratch and final row ranges. Prefix steps have one
// writer per peer, then one writer for the compact directory. No atomics.
__simt_vf__ __launch_bounds__(128) inline void DispatchPeerPlanVf(__gm__ int32_t* destinations, uint32_t stride,
                                                                  __gm__ int32_t* meta, __gm__ int32_t* rows,
                                                                  __gm__ int32_t* scratch, uint32_t rank,
                                                                  uint32_t world, uint32_t tokens, uint32_t topk,
                                                                  uint32_t step, uint32_t core)
{
    const uint32_t worker = core * 128U + threadIdx.x;
    const uint32_t tiles = (tokens + 31U) / 32U;
    if (step == 1U || step == 3U) {
        for (uint32_t peer = worker; peer < world; peer += 4096U) {
            for (uint32_t bucket = 0; bucket < (step == 1U ? 1U : 4U); ++bucket) {
                uint32_t total = 0;
                for (uint32_t tile = 0; tile < tiles; ++tile) {
                    auto* slot = scratch + (peer * tiles + tile) * 5U + (step == 1U ? 0U : 1U + bucket);
                    const uint32_t count = uint32_t(asc_ldcg(slot));
                    asc_stcg(slot, int32_t(total));
                    total += count;
                }
                const uint32_t entry = step == 1U ? peer * 4U : (bucket / 2U * world + peer) * 4U + bucket % 2U * 2U;
                asc_stcg(meta + entry + 1U, int32_t(total));
            }
        }
    } else if (step == 4U) {
        if (worker == 0U) {
            uint32_t begin = 0;
            for (uint32_t entry = 0; entry < world * 8U; entry += 2U) {
                asc_stcg(meta + entry, int32_t(begin));
                begin += uint32_t(asc_ldcg(meta + entry + 1U));
            }
        }
    } else {
        for (uint32_t task = worker; task < world * tiles; task += 4096U) {
            const uint32_t peer = task % world, tile = task / world;
            auto* counts = scratch + (peer * tiles + tile) * 5U;
            uint32_t total = 0, ordinal = 0, buckets[4]{};
            if (step != 0U) {
                ordinal = uint32_t(asc_ldcg(counts));
                if (step == 2U) {
                    total = uint32_t(asc_ldcg(meta + peer * 4U + 1U));
                } else {
                    for (uint32_t bucket = 0; bucket < 4U; ++bucket) {
                        const uint32_t entry = (bucket / 2U * world + peer) * 4U + bucket % 2U * 2U;
                        total += uint32_t(asc_ldcg(meta + entry + 1U));
                        buckets[bucket] = uint32_t(asc_ldcg(meta + entry)) + uint32_t(asc_ldcg(counts + 1U + bucket));
                    }
                }
            }
            const uint32_t full = total / 64U * 64U;
            const uint32_t tail_qp0 = total - full - (total - full) / 4U;
            const uint32_t end = tokens < (tile + 1U) * 32U ? tokens : (tile + 1U) * 32U;
            if (peer / 8U != rank / 8U) {
                for (uint32_t token = tile * 32U; token < end; ++token) {
                    for (uint32_t k = 0; k < topk; ++k) {
                        const int32_t encoded = asc_ldcg(destinations + token * topk + k);
                        if (encoded < 0 || uint32_t(encoded) / stride != peer) {
                            continue;
                        }
                        if (step == 0U) {
                            ++total;
                            continue;
                        }
                        const uint32_t qp =
                            (ordinal < full ? ordinal % 64U >= 48U : ordinal - full >= tail_qp0) ? 1U : 0U;
                        ++ordinal;
                        bool relay = false;
                        for (uint32_t slot = 0; slot < topk; ++slot) {
                            const int32_t other = asc_ldcg(destinations + token * topk + slot);
                            if (other < 0 && other != (-2147483647 - 1) &&
                                uint32_t(~other) / stride / 8U == peer / 8U) {
                                relay = true;
                            }
                        }
                        const uint32_t bucket = (relay ? 0U : 2U) + qp;
                        const uint32_t at = buckets[bucket]++;
                        if (step == 5U) {
                            asc_stcg(rows + at * 2U, int32_t(token));
                            asc_stcg(rows + at * 2U + 1U, int32_t(uint32_t(encoded) % stride));
                        }
                    }
                }
            }
            if (step == 0U) {
                asc_stcg(counts, int32_t(total));
            } else if (step == 2U) {
                for (uint32_t bucket = 0; bucket < 4U; ++bucket) {
                    asc_stcg(counts + 1U + bucket, int32_t(buckets[bucket]));
                }
            }
        }
    }
    asc_threadfence();
}

__aicore__ inline void PreparePeerPlan(GM_ADDR workspace, GM_ADDR dst, GM_ADDR forward, GM_ADDR forward_counts,
                                       GM_ADDR plan, const ascend_deepep::DispatchTiling& t)
{
    using namespace ascend_deepep;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 4096);
    const uint32_t core = GetBlockIdx(), phase_words = t.plan_phase_bytes / 4U;
    auto* meta = reinterpret_cast<__gm__ int32_t*>(plan + 64U);
    auto* rows = reinterpret_cast<__gm__ int32_t*>(plan + t.plan_rows_offset);
    auto* prefix = reinterpret_cast<__gm__ int32_t*>(plan + t.plan_scratch_offset);
    for (uint32_t step = 0; step < 6U; ++step) {
        asc_vf_call<DispatchPeerPlanVf>(dim3(128), reinterpret_cast<__gm__ int32_t*>(dst), t.address_stride, meta, rows,
                                        prefix, t.rank, t.world, t.tokens, t.topk, step, core);
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
        SyncAll<true>();
    }
    BuildForwardPlan(forward, forward_counts, plan, t);
    if (core == 0U) {
        auto headers = scratch.Get<uint32_t>();
        uint64_t counts[16]{};
        uint32_t error = 0;
        for (uint32_t phase = 0; phase < 2U; ++phase) {
            NativeLoad(headers, reinterpret_cast<__gm__ uint32_t*>(plan + 64U) + phase * phase_words, t.world * 4U);
            for (uint32_t peer = 0; peer < t.world; ++peer) {
                const uint32_t owner = DefaultClosPeerTableOwner(peer);
                counts[owner] += headers.GetValue(peer * 4U + 1U);
                counts[owner + 8U] += headers.GetValue(peer * 4U + 3U);
            }
        }
        for (uint32_t q = 0; q < 16U; ++q)
            error |= counts[q] + 2U + ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD >= shm::UDMA_SQ_BASKBLK_CNT;
        // Separate versioned options from the compact phase/peer/QP directory.
        NativeLoad(headers, reinterpret_cast<__gm__ uint32_t*>(plan), 16U);
        headers.SetValue(4U, 3U);
        headers.SetValue(5U, phase_words);
        headers.SetValue(6U, t.plan_mailbox_offset / 4U);
        headers.SetValue(7U, 7U);
        headers.SetValue(kClosPeerTableOptionsWord, DefaultClosPeerTableOptions(t.world));
        NativeStore(reinterpret_cast<__gm__ uint32_t*>(plan), headers, 16U);
        NativeLoad(headers, reinterpret_cast<__gm__ uint32_t*>(plan + t.forward_plan_offset) + 8U * (t.world + 1U), 1U);
        error |= headers.GetValue(0);
        headers.SetValue(0, error);
        NativeStore(reinterpret_cast<__gm__ uint32_t*>(workspace + t.sync), headers, 1U);
    }
    aclshmemx_barrier_all_vec();
    if (core == 0U) {
        uint32_t error = 0;
        auto value = scratch.Get<uint32_t>();
        // All ranks see the same rejection before any rank enters the sender.
        for (uint32_t peer = 0; peer < t.world; ++peer) {
            NativeLoad(value, reinterpret_cast<__gm__ uint32_t*>(aclshmem_ptr(workspace + t.sync, peer)), 1U);
            error |= value.GetValue(0);
        }
        value.SetValue(0, error);
        NativeStore(reinterpret_cast<__gm__ uint32_t*>(workspace + t.sync + 512U), value, 1U);
    }
    aclshmemx_barrier_all_vec();
}
