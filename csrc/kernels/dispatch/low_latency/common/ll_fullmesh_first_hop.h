#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"

namespace DispatchDedup {
// Each target physical host contains four consecutive
// full-mesh groups of four ranks. Count actual expert deliveries, not distinct
// cards: each relay carries one equal-sized token for its target expert.
// Two scalar words suffice for K<=32; no UB array, route rescan or cross-core work.
class LlFullMeshFirstHop {
    uint32_t counts_ = 0U;
    uint32_t first_ = 0U;
public:
    __aicore__ __forceinline__ void Add(uint32_t destinationRank, uint32_t k) {
        const uint32_t shift = ((destinationRank & 15U) >> 2U) * 8U;
        counts_ += 1U << shift;
        if (((first_ >> shift) & 255U) == 0U) first_ |= (k + 1U) << shift;
    }
    __aicore__ __forceinline__ uint32_t FirstK() const {
        uint32_t bestCount = 0U, bestK = 32U;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t shift = group * 8U;
            const uint32_t count = (counts_ >> shift) & 255U;
            if (!count) continue;
            const uint32_t k = ((first_ >> shift) & 255U) - 1U;
            if (count > bestCount || (count == bestCount && k < bestK)) {
                bestCount = count;
                bestK = k;
            }
        }
        return bestCount ? bestK : 0U;
    }
};

// Classify supplies a mask for exactly one target host. Select independently
// for each mask; a single token-wide firstK is invalid with multiple hosts.
// Masks are disjoint, so each grouped route is loaded at most once here.
__aicore__ __forceinline__ uint32_t LlRelayMemberMask(uint32_t mask, uint32_t firstK) {
    return mask & ~(1U << firstK);
}
}
