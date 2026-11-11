#pragma once
#include "kernel_operator.h"
#include "../common/ll_count_read_plan.h"

namespace DispatchDedup {
// Issue only; caller retains its existing MTE2->Vector->Scalar wait chain.
// Source-major frames are strided across source ranks. Each expert row becomes
// one DMA while the UB snapshot remains in the original expert-major order.
__aicore__ inline void ReadCountFrames(AscendC::LocalTensor<float> snapshot,
                                      AscendC::GlobalTensor<float> state,
                                      const LlCountReadPlan& plan, bool body)
{
    for (uint32_t i = 0; i < plan.size; ++i) {
        const auto& run = plan.runs[i];
        const uint64_t offset = body ? run.recordOffset : run.commitOffset;
        const AscendC::DataCopyParams copy{uint16_t(run.records), 1U, uint16_t(plan.sourceStrideBlocks), 0U};
        AscendC::DataCopy(snapshot[run.first * 8U], state[offset / sizeof(float)], copy);
    }
}
}
