/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
