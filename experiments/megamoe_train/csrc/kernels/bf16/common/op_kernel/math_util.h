// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

/*!
 * \file math_util.h
 * \brief
 */
#ifndef OP_COMMON_OP_KERNEL_MATH_UTIL_H
#define OP_COMMON_OP_KERNEL_MATH_UTIL_H

namespace Ops {
namespace Base {
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#define MEGA_MOE_HOST_AICORE __host_aicore__
#else
#define MEGA_MOE_HOST_AICORE
#endif
template <typename T>
MEGA_MOE_HOST_AICORE inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
MEGA_MOE_HOST_AICORE inline T FloorDiv(T a, T b)
{
    return a / b;
}

template <typename T>
MEGA_MOE_HOST_AICORE inline T CeilAlign(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
MEGA_MOE_HOST_AICORE inline T FloorAlign(T a, T b)
{
    return a / b * b;
}

#undef MEGA_MOE_HOST_AICORE
}  // namespace Base
}  // namespace Ops
#endif  // OP_COMMON_OP_KERNEL_MATH_UTIL_H
