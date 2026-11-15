// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include "peer_protocol.hpp"
#include "device_simt/shmem_device_simt_instance.h"

// Port of the private C++ FP8 peer-ready publisher, also used for BF16.
// put128(count=2) is NOT a portable guarantee of 32B atomic visibility. This
// path deliberately retains the target-platform assumption validated in that
// experiment. The source is immutable until VF completion; no GET or extra QP.
__simt_vf__ __launch_bounds__(128) inline void PublishPeerAddressRecordVf(__gm__ uint8_t* book,
                                                                          __gm__ uint8_t* outgoing, uint32_t rank,
                                                                          uint32_t world, uint32_t instance_id)
{
    simt::aclshmemx_instance_ctx_set(instance_id);
    for (uint32_t peer = threadIdx.x; peer < world; peer += 128U) {
        simt::aclshmem_putmem(book + uint64_t(rank) * ascend_deepep::kPeerAddressRowBytes, outgoing, 32U,
                              static_cast<int32_t>(peer));
    }
    asc_threadfence();
}

__aicore__ inline void PublishPeerAddresses(GM_ADDR workspace, const DispatchTiling& t,
                                            TBuf<QuePosition::VECCALC>& scratch, TBuf<QuePosition::VECCALC>& slots)
{
    // Only Clos cores prefetch. AIV0 can publish concurrently with their loads.
    CacheClosDirectories(t, slots);
    if (GetBlockIdx() == 0U) {
        auto descriptor = scratch.Get<uint64_t>();
        descriptor.SetValue(0U, t.output_address);
        descriptor.SetValue(1U, t.output_weights_address);
        descriptor.SetValue(2U, t.fp8 ? t.output_scales_address : 0U);
        descriptor.SetValue(ascend_deepep::kPeerAddressGenerationWord, t.generation);
        auto* book = workspace + t.peer_book;
        auto* outgoing =
            book + uint64_t(t.rank) * ascend_deepep::kPeerAddressRowBytes + ascend_deepep::kPeerAddressOutgoingBytes;
        NativeStore(reinterpret_cast<__gm__ uint64_t*>(outgoing), descriptor, ascend_deepep::kPeerAddressRecordWords);
        DataSyncBarrier<MemDsbT::DDR>();
        asc_vf_call<PublishPeerAddressRecordVf>(dim3(128), book, outgoing, t.rank, t.world, t.instance_id);
        SetFlag<HardEvent::V_S>(EVENT_ID3);
        WaitFlag<HardEvent::V_S>(EVENT_ID3);
        DataSyncBarrier<MemDsbT::DDR>();
    }
    // Local initialization and outgoing publication are complete. Do not wait
    // for incoming descriptors here: each consumer waits only for its peer.
    SyncAll<true>();
}
