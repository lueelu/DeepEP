// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define SCALE_CLOS_INLINE __aicore__ inline
#else
#define SCALE_CLOS_INLINE inline
#endif

namespace ascend_deepep::scale_clos {
// EP8: AIV7..31 copy and expand local inputs. AIV32..63 stay dedicated
// to auxiliary transport; hidden expansion never waits for their scale work.
// Larger EP keeps eight QP1 workers and AIV24..31 local expansion.
SCALE_CLOS_INLINE uint32_t Qp1Workers(uint32_t)
{
    return 8U;
}
SCALE_CLOS_INLINE uint32_t LocalFirst(uint32_t world)
{
    return world == 8U ? 7U : 16U + Qp1Workers(world);
}
SCALE_CLOS_INLINE uint32_t LocalWorkers(uint32_t world)
{
    return 32U - LocalFirst(world);
}
SCALE_CLOS_INLINE uint32_t LocalForwardWorkers(uint32_t world)
{
    return LocalWorkers(world);
}
// Rotate remainders by the global CSR offset, keeping total work balanced
// even when the eight source ranges have very different lengths.
SCALE_CLOS_INLINE void SplitLocalForwardRange(uint32_t world, uint32_t core, uint32_t begin, uint32_t end,
                                              uint32_t& first, uint32_t& last)
{
    const uint32_t workers = LocalForwardWorkers(world);
    const uint32_t worker = core - LocalFirst(world);
    const uint32_t count = end - begin, whole = count / workers, remainder = count % workers;
    const uint32_t rotated = (worker + workers - begin % workers) % workers;
    first = begin + rotated * whole + (rotated < remainder ? rotated : remainder);
    last = first + whole + (rotated < remainder ? 1U : 0U);
}
// One fixed order per EP size: balanced four-server windows through EP64,
// and eight-server mesh_clos stages above it. Start at the next server; callers
// skip the local server without changing ownership or the QP0/QP1 split.
SCALE_CLOS_INLINE uint32_t Peer(uint32_t rank, uint32_t world, uint32_t owner, uint32_t index)
{
    // Source local ranks 0..3 submit intra-mesh_clos first, 4..7 cross-mesh_clos first.
    // Every eight-visit window stays in one mesh_clos; owner/SQ identity is unchanged.
    // EP128 visits two pods; EP256 visits four (XOR stage 0,1,2,3).
    if (world >= 128U) {
        const uint32_t mesh_clos = rank / 64U ^ (rank % 8U) / 4U ^ index / 8U;
        return mesh_clos * 64U + ((rank / 8U + 1U + index % 8U) % 8U) * 8U + owner;
    }
    // One/two-server groups have no 32-rank mesh_clos to XOR. Enumerate each
    // real server once; the hidden scheduler omits its own server.
    if (world < 32U) {
        const uint32_t servers = (world + 7U) / 8U;
        return ((rank / 8U + 1U + index) % servers) * 8U + owner;
    }
    return ((rank / 32U) ^ (index / 4U)) * 32U + ((rank / 8U + 1U + index % 4U) % 4U) * 8U + owner;
}
}  // namespace ascend_deepep::scale_clos
#undef SCALE_CLOS_INLINE
