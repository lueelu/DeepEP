#pragma once
#include <cstdint>
#include "../../../common/ep_server_order.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_SEND_ORDER_INLINE __aicore__ inline
#else
#define LL_SEND_ORDER_INLINE inline
#endif

// Rotate an existing permutation; never change route ownership or destination.
// For a scheduled entry ordinal<count. count==0 has no entries; handle it
// defensively as well. Only one modulo is needed because ordinal+offset<2*count.
LL_SEND_ORDER_INLINE uint32_t LlSendStaggerOrdinal(uint32_t count, uint32_t phase, uint32_t ordinal)
{
    if (count <= 1U) return 0U;
    const uint32_t index = ordinal + phase % count;
    return index < count ? index : index - count;
}

// Only rotate the current token's first-hop tasks. Expert selection is owned
// by the sender's topology policy; this function never changes that selection.
LL_SEND_ORDER_INLINE uint32_t LlFirstHopOrder(uint32_t count, uint32_t rankInGroup,
                                           uint32_t core, uint32_t token, uint32_t ordinal)
{
    return LlSendStaggerOrdinal(count, rankInGroup + core + token, ordinal);
}

// Each source card/core enters the existing target-group permutation at a
// different point. Preserve the large-domain half-interleaving policy.
LL_SEND_ORDER_INLINE uint32_t LlDirectGroupOrder(uint32_t groups, uint32_t ownGroup,
                                              uint32_t rankInGroup, uint32_t core, uint32_t ordinal)
{
    return EpServerOrder::ServerDedupDispatchDirectOrder(
        groups, ownGroup, LlSendStaggerOrdinal(groups, rankInGroup + core, ordinal));
}
#undef LL_SEND_ORDER_INLINE
}  // namespace DispatchDedup
