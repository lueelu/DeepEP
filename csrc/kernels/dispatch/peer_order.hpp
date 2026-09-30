// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>
#include "peer_offset.hpp"

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define SCALE_CLOS_INLINE __aicore__ inline
#else
#define SCALE_CLOS_INLINE inline
#endif

namespace ascend_deepep::scale_clos {
// Eight QP1 workers for every EP; AIV24..31 own local expansion.
SCALE_CLOS_INLINE uint32_t Qp1Workers(uint32_t)
{
    return 8U;
}
SCALE_CLOS_INLINE uint32_t LocalFirst(uint32_t world)
{
    return 16U + Qp1Workers(world);
}
SCALE_CLOS_INLINE uint32_t LocalWorkers(uint32_t world)
{
    return 32U - LocalFirst(world);
}
// The same ordinal is used by the hidden scheduler, scale producers and
// consumer. EP128 owners 0..3 start inside the pod, 4..7 outside it. Windows
// alternate pods; pod-interleave additionally alternates individual visits.
// Start at the next server so skipping the local server cannot turn an
// inside-first owner into an outside-first owner in interleave mode.
SCALE_CLOS_INLINE uint32_t Peer(uint32_t rank, uint32_t world, uint32_t owner, uint32_t index, bool interleave,
                                bool balanced_layout = true, bool group_offset = false, bool pod_staged = false)
{
    // Source local ranks 0..3 submit intra-pod first, 4..7 cross-pod first.
    // Every eight-visit window stays in one pod; owner/SQ identity is unchanged.
    // EP128 visits two pods; EP256 visits four (XOR stage 0,1,2,3).
    if (pod_staged) {
        const uint32_t pod = rank / 64U ^ (rank % 8U) / 4U ^ index / 8U;
        return pod * 64U + ((rank / 8U + 1U + index % 8U) % 8U) * 8U + owner;
    }
    // One/two-server groups have no 32-rank pod to XOR. Enumerate each
    // real server once; the hidden scheduler omits its own server.
    if (world < 32U) {
        const uint32_t servers = (world + 7U) / 8U;
        return ((rank / 8U + 1U + index) % servers) * 8U + owner;
    }
    if (group_offset) {
        return clos_offset64::Peer(rank, owner + 8U * index);
    }
    if (!balanced_layout) {
        const uint32_t ordinal = interleave ? (index / 8U) * 4U + (index % 8U) / 2U + (index % 2U) * 8U : index;
        return ((rank / 32U) ^ (ordinal / 4U)) * 32U + ((rank / 8U + ordinal % 4U) % 4U) * 8U + owner;
    }
    uint32_t ordinal = index;
    if (world == 128U) {
        ordinal = interleave ? (index / 8U) * 4U + (index % 8U) / 2U + (index % 2U) * 8U
                             : ((index / 4U) % 2U) * 8U + (index / 8U) * 4U + index % 4U;
        if (owner >= 4U) {
            ordinal ^= 8U;
        }
    }
    return ((rank / 32U) ^ (ordinal / 4U)) * 32U + ((rank / 8U + 1U + ordinal % 4U) % 4U) * 8U + owner;
}
}  // namespace ascend_deepep::scale_clos
#undef SCALE_CLOS_INLINE
