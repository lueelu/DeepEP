// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_SHARED_POLICY_INLINE __aicore__ inline
#else
#define MEGAMOE_SHARED_POLICY_INLINE inline
#endif

namespace MegaMoeImpl {
// bs is the actual local input size, never receive capacity or a routed slice's M.
constexpr uint32_t SHARED_DECODE_MAX_BS = 256U;
MEGAMOE_SHARED_POLICY_INLINE constexpr bool UseDecodeInputReady(bool isA8W4, uint32_t bs)
{
    return isA8W4 && bs <= SHARED_DECODE_MAX_BS;
}
struct SharedExpertPolicy {
    bool decode;
    bool routeAiv1Only;
    uint32_t gmm1TileN;
    uint32_t gmm2TileN;
};
MEGAMOE_SHARED_POLICY_INLINE constexpr SharedExpertPolicy SelectSharedExpertPolicy(bool isA8W4, uint32_t sharedExperts,
                                                                                   uint32_t bs)
{
    const bool shared = isA8W4 && sharedExperts > 0U;
    const bool decode = shared && bs <= SHARED_DECODE_MAX_BS;
    return {decode, shared && !decode, decode ? 192U : 256U, decode ? 224U : 256U};
}
}  // namespace MegaMoeImpl
#undef MEGAMOE_SHARED_POLICY_INLINE
