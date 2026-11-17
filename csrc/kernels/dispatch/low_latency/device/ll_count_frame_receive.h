#pragma once
#include "ll_count_c_api.h"
#include "../common/ll_count_read_plan.h"

namespace DispatchDedup {
// Issue only; caller retains its existing MTE2->Vector->Scalar wait chain.
// Source-major frames are strided across source ranks. Each expert row becomes
// one DMA while the UB snapshot remains in the original expert-major order.
__aicore__ inline void ReadCountFrames(__ubuf__ uint32_t* snapshot,
                                      GM_ADDR state,
                                      const LlCountReadPlan& plan, bool body)
{
    for (uint32_t i = 0; i < plan.size; ++i) {
        const auto& run = plan.runs[i];
        const uint64_t offset = body ? run.recordOffset : run.commitOffset;
        LlCountC::Load(snapshot + run.first * 8U, state + offset, 32U,
                       run.records, uint64_t(plan.sourceStrideBlocks) * 32U);
    }
}
}
