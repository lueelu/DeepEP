#pragma once
#include <cstdint>
namespace DispatchDedup {
constexpr uint32_t kLlPhysicalRanksPerHost = 16U;
constexpr uint32_t kLlMaxRanksPerGroup = 16U;
enum class LlTilingKey : uint8_t { Group4 = 4U, Group16 = 16U };
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_GROUP_INLINE __aicore__ inline
#else
#define LL_GROUP_INLINE inline
#endif
LL_GROUP_INLINE constexpr uint32_t LlGroupForWorld(uint32_t world) { return world <= 16U ? 4U : 16U; }
LL_GROUP_INLINE constexpr uint8_t LlTilingKeyForWorld(uint32_t world) { return uint8_t(LlGroupForWorld(world)); }
LL_GROUP_INLINE constexpr bool LlTilingKeyValid(uint32_t world, uint8_t key) { return key == LlTilingKeyForWorld(world); }
LL_GROUP_INLINE constexpr bool LlSameHost(uint32_t rank, uint32_t peer, uint32_t ranksPerHost = kLlPhysicalRanksPerHost) {
    return ranksPerHost != 0U && rank / ranksPerHost == peer / ranksPerHost;
}
#undef LL_GROUP_INLINE
}
