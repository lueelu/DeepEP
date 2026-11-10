// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

/*!
 * \file platform_util.h
 * \brief UB block and vector-register size helpers for device code.
 */

#ifndef OP_COMMON_OP_KERNEL_PLATFORM_UTIL_H
#define OP_COMMON_OP_KERNEL_PLATFORM_UTIL_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Ops {
namespace Base {

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
}  // namespace Base
}  // namespace Ops
#endif  // OP_COMMON_OP_KERNEL_PLATFORM_UTIL_H
