#pragma once

#include "common/dispatch_output.h"
#include "kernels/common/udma_shared_sender.h"
#include "utils/debug/asc_assert.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Executed once by AIV 0 on every dispatch, including cached calls and graph
// replays. The host never stores an epoch or output address in a routing handle.
__simt_vf__ __launch_bounds__(1) inline void AdvanceDispatchEpoch(__gm__ uint64_t* counter, __ubuf__ uint64_t* result)
{
    const uint64_t old = *counter;
    if (old == UINT64_MAX) {
        __asc_simt_vf::__trap();
        return;
    }
    *counter = old + 1U;
    result[0] = old + 1U;
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void PublishDispatchOutput(
    __gm__ uint8_t* control, uint32_t rank, uint32_t world, uint64_t epoch, uint64_t x, uint64_t scales, uint32_t rows,
    uint32_t actual_rows, uint32_t x_bytes, uint32_t scale_bytes)
{
    const uint64_t payload_bytes = uint64_t(rows) * x_bytes;
    const uint64_t scales_bytes = uint64_t(rows) * scale_bytes;
    if (actual_rows > rows || !x_bytes || (rows && !x) || (scales_bytes && !scales) || x > UINT64_MAX - payload_bytes ||
        scales > UINT64_MAX - scales_bytes) {
        __asc_simt_vf::__trap();
        return;
    }
    auto* slot = reinterpret_cast<__gm__ uint64_t*>(control + (uint64_t(rank) + 1U) * kOutputSlotBytes);
    for (uint32_t peer = threadIdx.x; peer < world; peer += 128U) {
        simt::aclshmem_uint64_p(slot + 0, x, peer);
        simt::aclshmem_uint64_p(slot + 1, scale_bytes ? scales : 0U, peer);
        simt::aclshmem_uint64_p(slot + 2, rows, peer);
        simt::aclshmem_uint64_p(slot + 3, x_bytes, peer);
        simt::aclshmem_uint64_p(slot + 4, scale_bytes, peer);
        // Never assume the descriptor is atomically visible as one cache line.
        asc_threadfence();
        simt::aclshmem_uint64_p(slot + kOutputEpochWord, epoch, peer);
        asc_threadfence();
    }
}

// Called before Begin. Its existing all-rank collective is the consume gate;
// the final dispatch collective protects the table against the next publisher.
template <class Resource>
__aicore__ inline void PrepareDispatchOutput(
    Resource& resource, uint64_t control, uint32_t rank, uint32_t world, GM_ADDR x, GM_ADDR scales, uint32_t rows,
    uint32_t actual_rows, uint32_t x_bytes, uint32_t scale_bytes)
{
    auto epoch = resource.template GetBufferByByte<uint64_t>(0);
    asc_vf_call<AdvanceDispatchEpoch>(
        dim3(1), reinterpret_cast<__gm__ uint64_t*>(control), reinterpret_cast<__ubuf__ uint64_t*>(epoch.GetPhyAddr()));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    const uint64_t value = epoch.GetValue(0);
    asc_vf_call<PublishDispatchOutput>(
        dim3(128), reinterpret_cast<__gm__ uint8_t*>(control), rank, world, value, reinterpret_cast<uint64_t>(x),
        reinterpret_cast<uint64_t>(scales), rows, actual_rows, x_bytes, scale_bytes);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
}
#endif
} // namespace ascend_deepep::udma
