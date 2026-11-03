#pragma once

#include "kernel_operator.h"
#include "shmem.h"

namespace ascend_deepep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

// Call after Notify's publication collective and after draining earlier UB users.
// Read only this destination's row from each source's published inclusive table;
// write exclusive channel starts directly to owned GM (the public handle[2]).
// input/output are distinct, 32-byte-aligned UB regions with >= channels int32s.
__aicore__ inline void GatherRecvChannelPrefixes(
    GM_ADDR workspace, GM_ADDR recv_prefix, uint64_t published_offset_elements, uint32_t rank, uint32_t world,
    uint32_t channels, uint32_t channel_stride, uint32_t core, uint32_t cores, AscendC::LocalTensor<int32_t> input,
    AscendC::LocalTensor<int32_t> output)
{
    // Adjacent compact rows can share a 32-byte line (e.g. channels=2/6).
    // Give all rows in such a group to one AIV, including the last partial group.
    uint32_t rows_per_group = 1;
    while ((rows_per_group * channels) % 8U != 0)
        rows_per_group *= 2U;
    AscendC::GlobalTensor<int32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_prefix));
    const AscendC::DataCopyExtParams copy{1, channels * uint32_t(sizeof(int32_t)), 0, 0, 0};
    for (uint32_t first = core * rows_per_group; first < world; first += cores * rows_per_group) {
        const uint32_t end = first + rows_per_group < world ? first + rows_per_group : world;
        for (uint32_t source = first; source < end; ++source) {
            auto* remote = reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(workspace, source));
            AscendC::GlobalTensor<int32_t> published;
            published.SetGlobalBuffer(remote + published_offset_elements + uint64_t(rank) * channel_stride);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(input, published, copy, AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            output.SetValue(0, 0);
            for (uint32_t channel = 1; channel < channels; ++channel)
                output.SetValue(channel, input.GetValue(channel - 1));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(result[uint64_t(source) * channels], output, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
}
#endif
} // namespace ascend_deepep
