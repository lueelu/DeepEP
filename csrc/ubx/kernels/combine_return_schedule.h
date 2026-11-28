// Copyright (c) 2026, Lu Lu

#pragma once
#include <ubx/kernels/ep_server_order.h>
#include <ubx/kernels/ll_group_policy.h>

namespace UbxCombine {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define COMBINE_RETURN_INLINE __aicore__ inline
#else
#define COMBINE_RETURN_INLINE inline
#endif
// Preserve the tested half-order traversal and per-core task ownership.
template <uint32_t Group>
COMBINE_RETURN_INLINE uint32_t CombineReturnServer(uint32_t servers, uint32_t own, uint32_t ordinal)
{
    return EpServerOrder::ServerDedupHalfOrder(servers, own, ordinal);
}

template <uint32_t Group, bool RemoteFirst>
COMBINE_RETURN_INLINE uint32_t CombineDirectServer(uint32_t servers, uint32_t own, uint32_t visit)
{
    if constexpr (RemoteFirst) {
        const uint32_t split = servers / 2U;
        const uint32_t ownLength = own < split ? split : servers - split;
        const uint32_t ownStep = (ownLength - 1U) * (servers >= 16U ? 2U : 1U);
        visit = visit + 1U == servers ? ownStep : visit + uint32_t(visit >= ownStep);
    }
    return CombineReturnServer<Group>(servers, own, visit);
}

#undef COMBINE_RETURN_INLINE
}  // namespace UbxCombine
