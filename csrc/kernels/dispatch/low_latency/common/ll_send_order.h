#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"
#include "../../../common/ep_server_order.h"

namespace DispatchDedup {
// Rotate an existing permutation; never change route ownership or destination.
// For a scheduled entry ordinal<count. count==0 has no entries; handle it
// defensively as well. Only one modulo is needed because ordinal+offset<2*count.
__aicore__ __forceinline__ uint32_t LlSendStaggerOrdinal(uint32_t count, uint32_t phase, uint32_t ordinal)
{
    if (count <= 1U) return 0U;
    const uint32_t index = ordinal + phase % count;
    return index < count ? index : index - count;
}

// Only rotate the current token's first-hop tasks. Expert selection is owned
// by the sender's topology policy; this function never changes that selection.
__aicore__ __forceinline__ uint32_t LlFirstHopOrder(uint32_t count, uint32_t rankInGroup,
    uint32_t core, uint32_t token, uint32_t ordinal)
{
    return LlSendStaggerOrdinal(count, rankInGroup + core + token, ordinal);
}

// Each source card/core enters the existing target-group permutation at a
// different point. Preserve the large-domain half-interleaving policy.
__aicore__ __forceinline__ uint32_t LlDirectGroupOrder(uint32_t groups, uint32_t ownGroup,
    uint32_t rankInGroup, uint32_t core, uint32_t ordinal,
    bool remoteFirst = false)
{
    if (remoteFirst && groups > 1U) {
        // Post RDMA before local copies so a singleton remote DIRECT is not
        // delayed by a long local group. Rotate only the remote permutation.
        // Remove ownGroup from the original order; keep its half-interleaving.
        if (ordinal == groups - 1U) return ownGroup;
        const uint32_t split = groups / 2U;
        const uint32_t ownLength = ownGroup < split ? split : groups - split;
        const uint32_t ownOrdinal = groups < 16U ? 0U :
            (ownLength > split ? groups - 1U : 2U * (ownLength - 1U));
        uint32_t remoteOrdinal = LlSendStaggerOrdinal(groups - 1U, rankInGroup + core, ordinal);
        if (remoteOrdinal >= ownOrdinal) ++remoteOrdinal;
        return EpServerOrder::ServerDedupDispatchDirectOrder(groups, ownGroup, remoteOrdinal);
    }
    return EpServerOrder::ServerDedupDispatchDirectOrder(
        groups, ownGroup, LlSendStaggerOrdinal(groups, rankInGroup + core, ordinal));
}
}  // namespace DispatchDedup
