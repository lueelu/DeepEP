#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"

namespace DispatchDedup {
// A non-dedup core can own flat routes without owning a token-weight slice.
// If total routes are fewer than sender cores, only the first routeCount cores
// own routes; otherwise every sender owns at least one.
__aicore__ __forceinline__ bool LlSenderHasWork(bool dedup, uint32_t tokenCount,
    uint32_t routeCount, uint32_t core)
{
    return tokenCount != 0U || (!dedup && core < routeCount);
}

template <class Words>
class LlSplitRouteWords {
    Words first_, second_;
    uint32_t half_;
public:
    __aicore__ __forceinline__ LlSplitRouteWords(Words first, Words second, uint32_t half)
        : first_(first), second_(second), half_(half) {}
    __aicore__ __forceinline__ uint32_t GetValue(uint32_t index) const {
        return index < half_ ? first_.GetValue(index) : second_.GetValue(index - half_);
    }
    __aicore__ __forceinline__ void SetValue(uint32_t index, uint32_t value) {
        if (index < half_) first_.SetValue(index, value);
        else second_.SetValue(index - half_, value);
    }
};

// Two old q-reduction scratch arrays provide at least 2*routeCount slots. Use
// a direct histogram when experts fit, otherwise a <=50%-loaded hash table.
// A packed entry holds expert+1 in its low 16 bits and count in its high 16:
// the existing admission bounds E<=1024 and B*K<2^14 make both fields exact.
// Q remains immutable and independent of packet order after this forward pass.
template <class Ids, class Words>
__aicore__ __forceinline__ void LlBuildRouteQ(Ids ids, uint32_t routeCount, uint32_t topK, uint32_t experts,
    Words firstScratch, Words secondScratch, Words q, uint32_t halfSlots)
{
    if (routeCount <= topK) {
        // Every original q call returns zero for the first token. No histogram
        // or hash state is needed when the complete input is that one token.
        for (uint32_t index = 0; index < routeCount; ++index) q.SetValue(index, 0U);
        return;
    }
    LlSplitRouteWords<Words> table(firstScratch, secondScratch, halfSlots);
    const uint32_t slots = halfSlots * 2U;
    const bool dense = experts <= slots;
    const uint32_t initialized = dense ? experts : slots;
    for (uint32_t slot = 0; slot < initialized; ++slot) table.SetValue(slot, 0U);
    for (uint32_t index = 0; index < routeCount; ++index) {
        const uint32_t expert = uint32_t(ids.GetValue(index));
        if (dense) {
            const uint32_t previous = table.GetValue(expert);
            q.SetValue(index, index < topK ? 0U : previous);
            table.SetValue(expert, previous + 1U);
            continue;
        }
        uint32_t hash = expert * 0x9e3779b9U;
        hash ^= hash >> 16U;
        uint32_t slot = hash % slots;
        const uint32_t key = expert + 1U;
        uint32_t entry = table.GetValue(slot);
        while (entry != 0U && (entry & 0xffffU) != key) {
            if (++slot == slots) slot = 0U;
            entry = table.GetValue(slot);
        }
        const uint32_t previous = entry >> 16U;
        // Preserve the original first-token special case even for repeated IDs.
        q.SetValue(index, index < topK ? 0U : previous);
        table.SetValue(slot, key | ((previous + 1U) << 16U));
    }
}
}
