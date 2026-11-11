#pragma once
#include <cstdint>
#include "ll_group_policy.h"

namespace DispatchDedup {
// Same E*512B reservation, physical source-major frames; logical prefixes stay
// expert-major. Legacy/local frames intersect a count-core and target rank;
// Direct remote frames cover all experts of that destination rank.
struct LlCountFrame {
    uint64_t recordOffset, commitOffset, frameOffset;
    uint32_t firstExpert, records;
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
LlCountFrame CountFrame(uint32_t experts, uint32_t world, uint32_t countCores,
                        uint32_t destination, uint32_t source, uint32_t localExpert,
                        bool directRanks = false)
{
    const uint32_t local = experts / world, expert = destination * local + localExpert;
    if (directRanks && !LlSameHost(destination, source)) {
        // One source/core owns all experts of this remote rank. The compact
        // body and its trailing 32B commit travel in the same SO WQE.
        const uint64_t frame = uint64_t(source) * local * 512U;
        return {frame + uint64_t(localExpert) * 32U,
                frame + uint64_t(local) * 32U, frame, 0U, local};
    }
    const uint32_t base = experts / countCores, extra = experts % countCores;
    const uint32_t wideEnd = (base + 1U) * extra;
    const uint32_t owner = expert < wideEnd ? expert / (base + 1U) : extra + (expert - wideEnd) / base;
    const uint32_t begin = owner * base + (owner < extra ? owner : extra);
    const uint32_t end = begin + base + (owner < extra ? 1U : 0U);
    const uint32_t first = begin > destination * local ? begin - destination * local : 0U;
    const uint32_t last = end < (destination + 1U) * local ? end - destination * local : local;
    const uint64_t frame = (uint64_t(source) * local + first) * 512U;
    return {frame + uint64_t(localExpert - first) * 32U,
            frame + uint64_t(last - first) * 512U - 32U, frame, first, last - first};
}
}
