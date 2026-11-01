#pragma once

#include "ascend_deepep/kernels/udma_shared_sender.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr uint32_t kAggregateReadyStride = 512;

// A full cache line per chunk avoids cache writeback by a different producer.
__aicore__ inline void StoreAggregateReady(
    uint64_t base, uint32_t chunk, uint64_t value, AscendC::LocalTensor<uint64_t> scratch)
{
    AscendC::GlobalTensor<uint64_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(base + uint64_t(chunk) * kAggregateReadyStride));
    scratch.SetValue(0, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::DataCopyExtParams copy{1, uint32_t(sizeof(uint64_t)), 0, 0, 0};
    AscendC::DataCopyPad(out, scratch, copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
}

__aicore__ inline void WaitAggregateReady(const CombineTilingData& td, uint32_t core, uint32_t row)
{
    const uint32_t chunk = row / td.pipeline_chunk_tokens;
    auto* address = reinterpret_cast<__gm__ uint64_t*>(td.pipeline_ready + uint64_t(chunk) * kAggregateReadyStride);
    AscendC::GlobalTensor<uint64_t> ready;
    ready.SetGlobalBuffer(address);
    const uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
        const uint64_t value = ready.GetValue(0);
        if (value == 1U) return;
        const uint64_t elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - start;
        if (elapsed >= kIncomingTimeoutCycles) {
            AscendC::printf("[V1_UDMA_TIMEOUT] op=combine stage=aggregate_ready rank=%u core=%u "
                "row=%u chunk=%u observed=%llu elapsed_us=%llu timeout_us=10000000\n",
                td.rank, core, row, chunk, static_cast<unsigned long long>(value),
                static_cast<unsigned long long>(elapsed / kIncomingCyclesPerUs));
            trap();
        }
    }
}
#endif
} // namespace ascend_deepep::udma
