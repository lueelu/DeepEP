// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man

#ifndef MEGA_MOE_A8W4_K_PLAN_H
#define MEGA_MOE_A8W4_K_PLAN_H
#include <cstdint>

namespace UrmaMegaMoeImpl::GmmKernel {
// Shared by AIC and AIV0: B must use exactly the same K window on both cores.
struct A8W4BDeepResources {
    static constexpr uint32_t AStages = 2U, BStages = 4U, ScaleStages = 2U;
    static constexpr uint32_t ABytes = 64U * 1024U, BBytes = 64U * 1024U, ScaleBytes = 32U * 1024U;
    static constexpr uint32_t ABase = 0U, BBase = AStages * ABytes;
    static constexpr uint32_t ScaleABase = BBase + BStages * BBytes;
    static constexpr uint32_t ScaleBBase = ScaleABase + ScaleStages * ScaleBytes;
    static constexpr uint16_t BFull = 4U, BEmpty = 8U, ContextEnd = 2U;
    static_assert(ScaleBBase + ScaleStages * ScaleBytes == 512U * 1024U);
    static_assert(ContextEnd < BFull && BFull + BStages <= BEmpty && BEmpty + BStages <= 12U);
};
struct A8W4KPlan {
    uint32_t ka;
    uint32_t kb;
};
__aicore__ constexpr inline A8W4KPlan MakeA8W4KPlan(uint32_t m, uint32_t n, bool allowDynamicB)
{
    const uint32_t kb = allowDynamicB && m <= 128U && n <= 128U ? 512U : 256U;
    const uint32_t alignedM = (m + 15U) / 16U * 16U;
    if (alignedM == 0U || m >= n) {
        return {kb, kb};
    }
    const uint32_t balancedDepth = (n + 2U * alignedM - 1U) / (2U * alignedM);
    const uint32_t capacityDepth = A8W4BDeepResources::ABytes / (alignedM * kb);
    const uint32_t depth = balancedDepth < capacityDepth ? balancedDepth : capacityDepth;
    return {depth * kb, kb};
}
}  // namespace UrmaMegaMoeImpl::GmmKernel
#endif
