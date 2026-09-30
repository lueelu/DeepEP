// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>

// EP64 destination order spreads traffic; it is NOT receiver admission.
// There is no cross-source lockstep here, so skew can still cause incast.
namespace ascend_deepep::clos_offset64 {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define CLOS_OFFSET_INLINE __aicore__ inline
#else
#define CLOS_OFFSET_INLINE inline constexpr
#endif
CLOS_OFFSET_INLINE uint32_t Peer(uint32_t rank, uint32_t ordinal)
{
    const uint32_t lane = ordinal % 16U, phase = ordinal / 16U;
    const uint32_t server = rank / 8U * 8U, local = rank % 8U;
    const uint32_t remaining = 56U - phase * 16U;
    const uint32_t remote = remaining < 16U ? remaining : 16U;
    if (lane >= remote) {
        return server + lane - remote;
    }
    uint32_t seen = 0U;
    for (uint32_t side = 0U; side < 2U; ++side) {
        const uint32_t start = side == 0U ? (server + 64U - (phase + 1U) * 8U + local) % 64U
                                          : (server + (phase + 1U) * 8U + (phase == 0U ? 0U : local)) % 64U;
        for (uint32_t i = 0U; i < (side == 0U ? 8U : 64U); ++i) {
            const uint32_t peer = (start + i) % 64U;
            if (peer >= server && peer < server + 8U) {
                continue;
            }
            if (seen++ == lane) {
                return peer;
            }
        }
    }
    return 64U;
}
// Used only at setup / packet construction, never in the hidden WQE loop.
CLOS_OFFSET_INLINE uint32_t Owner(uint32_t source, uint32_t peer)
{
    if (source >= 64U || peer >= 64U) {
        return 8U;
    }
    // Closed-form inverse lane, exhaustively checked against all 4096 entries
    // of the reference schedule. Own-server entries are appended unrotated.
    return peer / 8U == source / 8U ? peer % 8U : (peer + 8U - source % 8U) % 8U;
}
#undef CLOS_OFFSET_INLINE
}  // namespace ascend_deepep::clos_offset64
