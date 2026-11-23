// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include "kernel_operator.h"

namespace MegaMoeImpl {
// Input is one FP32 route. Preserve first-route initialization and slot order.
// SourceIsSum avoids a redundant first-route copy in BF16 preweighted Unpermute.
template <bool ApplyWeight, bool SourceIsSum = false>
__aicore__ inline void AccumulateRoute(AscendC::LocalTensor<float> sum, AscendC::LocalTensor<float> value,
                                       uint32_t length, bool first, float weight = 1.0F)
{
    using namespace AscendC;
    if (first) {
        if constexpr (ApplyWeight)
            Muls(sum, value, weight, length);
        else if constexpr (!SourceIsSum)
            DataCopy(sum, value, length);
    } else {
        if constexpr (ApplyWeight) {
            Muls(value, value, weight, length);
            PipeBarrier<PIPE_V>();
        }
        Add(sum, sum, value, length);
    }
}
}  // namespace MegaMoeImpl
