#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"
namespace DispatchDedup {
constexpr uint32_t kLlPhysicalRanksPerHost = 16U;
constexpr uint32_t kLlMaxRanksPerGroup = 16U;
enum class LlTilingKey : uint8_t { Group4 = 4U, Group16 = 16U };
constexpr __aicore__ __forceinline__ uint32_t LlGroupForWorld(uint32_t world) { return world <= 16U ? 4U : 16U; }
constexpr __aicore__ __forceinline__ uint8_t LlTilingKeyForWorld(uint32_t world) { return uint8_t(LlGroupForWorld(world)); }
constexpr __aicore__ __forceinline__ bool LlTilingKeyValid(uint32_t world, uint8_t key) { return key == LlTilingKeyForWorld(world); }
constexpr __aicore__ __forceinline__ bool LlSameHost(uint32_t rank, uint32_t peer, uint32_t ranksPerHost = kLlPhysicalRanksPerHost) {
    return ranksPerHost != 0U && rank / ranksPerHost == peer / ranksPerHost;
}
}
