// Copyright (c) 2026, Lu Lu

#pragma once
#include "ep_server_order.h"
#include "ll_group_policy.h"

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

#undef COMBINE_RETURN_INLINE
}  // namespace UbxCombine
