// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include "kernel_operator.h"

namespace MegaMoeImpl::Activation {
static constexpr AscendC::Reg::DivSpecificMode kAccurateSwiGluDiv = {AscendC::Reg::MaskMergeMode::ZEROING, true};

// Keep the input cast, clamp and output conversion with their payload policies.
// Quantized inference retains default division; BF16 retains accurate division.
template <bool Accurate, bool IsStep = false>
__simd_callee__ inline void SwiGluFp32(AscendC::Reg::RegTensor<float>& output, AscendC::Reg::RegTensor<float>& gate,
                                       AscendC::Reg::RegTensor<float>& up, AscendC::Reg::RegTensor<float>& neg,
                                       AscendC::Reg::RegTensor<float>& exponent,
                                       AscendC::Reg::RegTensor<float>& denominator,
                                       AscendC::Reg::RegTensor<float>& silu, AscendC::Reg::MaskReg& mask,
                                       float stepClamp = 0.0F)
{
    AscendC::Reg::Muls(neg, gate, -1.0F, mask);
    AscendC::Reg::Exp(exponent, neg, mask);
    AscendC::Reg::Adds(denominator, exponent, 1.0F, mask);
    if constexpr (Accurate)
        AscendC::Reg::Div<float, &kAccurateSwiGluDiv>(silu, gate, denominator, mask);
    else
        AscendC::Reg::Div(silu, gate, denominator, mask);
    if constexpr (IsStep) AscendC::Reg::Mins(silu, silu, stepClamp, mask);
    AscendC::Reg::Mul(output, silu, up, mask);
}
}  // namespace MegaMoeImpl::Activation
