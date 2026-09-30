// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

#include "peer_address.hpp"

// Every sender drains hidden, forwarded hidden and scales before advertising
// completion. Receivers wait for every sender before returning output storage.
template <bool Profile>
__aicore__ inline void FinishPeerDispatch(GM_ADDR workspace, const DispatchTiling& t,
                                          TBuf<QuePosition::VECCALC>& scratch, uint64_t* detail)
{
    uint64_t stamp = DetailClock<Profile>(t.detail_mode);
    aclshmemx_mte_quiet();
    DetailAdd<Profile>(detail, 12U, DetailClock<Profile>(t.detail_mode) - stamp);
    stamp = DetailClock<Profile>(t.detail_mode);
    SyncAll<true>();
    DetailAdd<Profile>(detail, 13U, DetailClock<Profile>(t.detail_mode) - stamp);
    stamp = DetailClock<Profile>(t.detail_mode);
    {
        auto data = scratch.Get<uint64_t>();
        data.SetValue(0, t.generation);
        for (uint32_t peer = GetBlockIdx(); peer < t.world; peer += 64U) {
            auto* remote =
                reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(workspace + t.done + uint64_t(t.rank) * 512U, peer));
            NativeStore(remote, data, 1U);
        }
        aclshmemx_mte_quiet();
        DetailAdd<Profile>(detail, 14U, DetailClock<Profile>(t.detail_mode) - stamp);
        stamp = DetailClock<Profile>(t.detail_mode);
        for (uint32_t peer = GetBlockIdx(); peer < t.world; peer += 64U)
            aclshmem_uint64_wait_until(reinterpret_cast<__gm__ uint64_t*>(workspace + t.done + uint64_t(peer) * 512U),
                                       ACLSHMEM_CMP_EQ, t.generation);
    }
    DetailAdd<Profile>(detail, 15U, DetailClock<Profile>(t.detail_mode) - stamp);
    stamp = DetailClock<Profile>(t.detail_mode);
    SyncAll<true>();
    DetailAdd<Profile>(detail, 16U, DetailClock<Profile>(t.detail_mode) - stamp);
}

// Peer-double pipeline: SIMT decode, vector compaction,
// asynchronous 224B scale plane + slot plane, per-peer ping-pong and SIMT scatter.
#include "scale_pipeline.hpp"
#include "aux_pipeline.hpp"
template <bool Profile>
__aicore__ inline void PeerScales(GM_ADDR workspace, GM_ADDR input, GM_ADDR destinations, const DispatchTiling& t,
                                  TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging,
                                  TBuf<TPosition::VECOUT>& wqe_scratch)
{
    PeerScalePlan plan{destinations, input, reinterpret_cast<GM_ADDR>(t.output_scales_address), t.address_stride};
    DeepepSecondaryState unused{};
    DispatchScalePipeline::Run<Profile>(workspace, t, plan, staging, metadata, wqe_scratch, unused);
}
