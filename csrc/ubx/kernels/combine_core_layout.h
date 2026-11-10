// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>
namespace UbxCombine {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
    constexpr uint32_t
    CombineLogicalCore(uint32_t physical, uint32_t)
{
    return physical;
}
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
    constexpr uint32_t
    CombineReceiveCoreIndex(uint32_t physical, uint32_t cores, uint32_t receive)
{
    return physical - (cores - receive);
}
}  // namespace UbxCombine
