// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>
namespace UbxDispatch {
constexpr uint32_t kLlMetadataCores = 4;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_META_INLINE __aicore__ inline
#else
#define LL_META_INLINE inline
#endif
LL_META_INLINE constexpr bool MetadataSplitValid(uint32_t cores)
{
    return cores >= kLlMetadataCores + 2U && cores <= 72U && cores % 2U == 0U;
}
LL_META_INLINE constexpr uint32_t PayloadCores(uint32_t cores)
{
    return (cores - kLlMetadataCores) / 2U;
}
LL_META_INLINE constexpr uint32_t ReceiveBegin(uint32_t cores)
{
    return kLlMetadataCores + PayloadCores(cores);
}
LL_META_INLINE constexpr uint32_t ForwardSlots(uint32_t slots, uint32_t core, uint32_t cores)
{
    return slots > core ? (slots - 1U - core) / cores + 1U : 0U;
}
LL_META_INLINE constexpr uint32_t BalancedBegin(uint32_t total, uint32_t core, uint32_t cores)
{
    return uint32_t(uint64_t(total) * core / cores);
}
#undef LL_META_INLINE
}  // namespace UbxDispatch
