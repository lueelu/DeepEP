#pragma once

#include <cstdint>

#include "kernel_operator.h"

namespace ascend_deepep {

// The transport completion barrier must precede this local copy. Each AIV
// owns a disjoint compact byte interval. All earlier staging users/events
// must be drained; this helper owns and drains its two reuse credits.
__aicore__ inline void CopyDispatchRecvToGm(
    GM_ADDR output, GM_ADDR symmetric_buffer, uint64_t shmem_offset,
    uint64_t byte_begin, uint64_t byte_count, AscendC::LocalTensor<uint8_t> staging,
    uint32_t staging_capacity)
{
    using namespace AscendC;
    if (byte_count == 0 || output == symmetric_buffer + shmem_offset) {
        return;
    }
    AscendC::GlobalTensor<uint8_t> source_gm;
    AscendC::GlobalTensor<uint8_t> output_gm;
    source_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + shmem_offset);
    output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    const uint32_t slot_bytes = (staging_capacity / 2U / 32U) * 32U;
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint64_t copied = 0;
    uint32_t slot = 0;
    while (copied < byte_count) {
        const uint64_t remaining = byte_count - copied;
        const uint32_t tile_bytes = remaining < slot_bytes ? static_cast<uint32_t>(remaining) : slot_bytes;
        const auto event_id = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        auto tile_ub = staging[slot * slot_bytes];
        AscendC::DataCopyExtParams params(1, tile_bytes, 0, 0, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(tile_ub, source_gm[byte_begin + copied], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(output_gm[byte_begin + copied], tile_ub, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        copied += tile_bytes;
        slot ^= 1U;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

}  // namespace ascend_deepep
