#pragma once
#include <cstdint>

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_SENDER_PLAN_INLINE __aicore__ inline
#else
#define LL_SENDER_PLAN_INLINE inline
#endif

// A non-dedup core can own flat routes without owning a token-weight slice.
// If total routes are fewer than sender cores, only the first routeCount cores
// own routes; otherwise every sender owns at least one.
LL_SENDER_PLAN_INLINE bool LlSenderHasWork(bool dedup, uint32_t tokenCount,
                                          uint32_t routeCount, uint32_t core)
{
    return tokenCount != 0U || (!dedup && core < routeCount);
}

template <class Words>
class LlSplitRouteWords {
    Words first_, second_;
    uint32_t half_;
public:
    LL_SENDER_PLAN_INLINE LlSplitRouteWords(Words first, Words second, uint32_t half)
        : first_(first), second_(second), half_(half) {}
    LL_SENDER_PLAN_INLINE uint32_t GetValue(uint32_t index) const {
        return index < half_ ? first_.GetValue(index) : second_.GetValue(index - half_);
    }
    LL_SENDER_PLAN_INLINE void SetValue(uint32_t index, uint32_t value) {
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
LL_SENDER_PLAN_INLINE void LlBuildRouteQ(Ids ids, uint32_t routeCount, uint32_t topK, uint32_t experts,
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

// Task records retain token/k/expert in words 0..2. Their former cached-q word
// becomes a next link: q now comes from the immutable route table. Reverse
// construction yields the same ascending task order as the old stable scan.
template <class Words>
LL_SENDER_PLAN_INLINE void LlBuildStableTaskGroups(Words tasks, uint32_t count, uint32_t expertsPerGroup,
                                                  uint32_t groups, Words heads)
{
    for (uint32_t group = 0; group < groups; ++group) heads.SetValue(group, UINT32_MAX);
    for (uint32_t next = count; next != 0U; --next) {
        const uint32_t task = next - 1U;
        const uint32_t group = tasks.GetValue(task * 4U + 2U) / expertsPerGroup;
        tasks.SetValue(task * 4U + 3U, heads.GetValue(group));
        heads.SetValue(group, task);
    }
}

// Non-dedup large-domain routing uses original flat indices. Reuse the second
// retired q hash array for links, including cores owning routes but no tokens.
template <class Ids, class Words>
LL_SENDER_PLAN_INLINE void LlBuildStableRouteGroups(Ids ids, uint32_t begin, uint32_t count,
                                                   uint32_t expertsPerGroup, uint32_t groups,
                                                   Words heads, Words links)
{
    for (uint32_t group = 0; group < groups; ++group) heads.SetValue(group, UINT32_MAX);
    for (uint32_t next = begin + count; next != begin; --next) {
        const uint32_t route = next - 1U;
        const uint32_t group = uint32_t(ids.GetValue(route)) / expertsPerGroup;
        links.SetValue(route, heads.GetValue(group));
        heads.SetValue(group, route);
    }
}

#undef LL_SENDER_PLAN_INLINE
}
