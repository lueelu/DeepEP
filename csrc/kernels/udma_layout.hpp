// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>

namespace ascend_deepep::udma_layout {
// One Session, one immutable SDK configuration. Operators reuse this bank only
// after the previous operator has drained its queues on the ordered stream.
constexpr uint32_t kDispatchOwners = 8U;
constexpr uint32_t kCombineOwners = 16U;
constexpr uint32_t kQpCount = 2U * kCombineOwners;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__ inline uint32_t CrossQp(uint32_t owner, uint32_t lane)
#else
inline constexpr uint32_t CrossQp(uint32_t owner, uint32_t lane)
#endif
{
    // Matches Combine's 2*owner+lane and the pinned SDK's alternating Clos
    // routes. Lane 0/1 retain their logical 3:1 membership, not new producers.
    // MESH peer QP0 is a different physical SQ; never apply this map to FM.
    return 2U * owner + lane;
}
static_assert(kQpCount == 32U);
static_assert(kDispatchOwners <= kCombineOwners);
}  // namespace ascend_deepep::udma_layout
