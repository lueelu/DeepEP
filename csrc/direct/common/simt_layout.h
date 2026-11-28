#pragma once

#include <cstdint>

namespace ascend_deepep {
struct SimtLayoutUbPlan {
    uint32_t tile = 0;
    uint32_t lanes = 0;
    uint32_t input_bytes = 0;
    uint32_t mask_bytes = 0;
    uint32_t histogram_offset = 0;
    uint32_t output_offset = 0;
    uint32_t row_elems = 0;
    uint32_t bytes = 0;
};

// Shared host/device capacity calculation. No shape-specific policy branches.
#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr SimtLayoutUbPlan MakeSimtLayoutUbPlan(uint32_t world, uint32_t experts, uint32_t topk,
                                                           uint32_t index_bytes, uint32_t ub_bytes)
{
    const uint64_t row = ((uint64_t(world) + 7U) / 8U + (uint64_t(experts) + 7U) / 8U) * 8U;
    if (!world || !experts || !row || (index_bytes != 4 && index_bytes != 8)) return {};
    for (uint32_t tile = 128; tile; tile /= 2) {
        const uint64_t input = (uint64_t(tile) * topk * index_bytes + 31U) / 32U * 32U;
        const uint64_t mask = (uint64_t(tile) * world + 31U) / 32U * 32U;
        const uint64_t histogram = 2U * (input + mask);
        for (uint32_t lanes = 128; lanes; lanes /= 2) {
            const uint64_t output = histogram + lanes * row * 4U;
            const uint64_t bytes = output + row * 4U;
            if (bytes <= ub_bytes) {
                return {tile,
                        lanes,
                        uint32_t(input),
                        uint32_t(mask),
                        uint32_t(histogram),
                        uint32_t(output),
                        uint32_t(row),
                        uint32_t(bytes)};
            }
        }
    }
    return {};
}
}  // namespace ascend_deepep
