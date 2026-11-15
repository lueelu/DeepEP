#pragma once

#include <cstdint>

#ifndef ASCEND_DEEPEP_ARCH
#error "ASCEND_DEEPEP_ARCH must be provided by the build system"
#endif

namespace ascend_deepep::platform {

#if ASCEND_DEEPEP_ARCH == 3510
inline constexpr std::int64_t kMaxNumAiv = 64;
inline constexpr std::int64_t kLaunchSharedBytes = 224 * 1024;
// CANN reports 8 KiB of static shared memory plus a 32-byte slot for the
// SHMEM SIMT instance context. Reserve 64 bytes beyond the base 8 KiB so
// the dispatch layout's two equal UB buffers both remain 32-byte aligned.
inline constexpr std::int64_t kCompilerSharedReserveBytes = 8 * 1024 + 64;
inline constexpr std::int64_t kKernelUbBytes = kLaunchSharedBytes - kCompilerSharedReserveBytes;
inline constexpr std::int64_t kDispatchStageBytes = 192 * 1024;
static_assert(kKernelUbBytes % 64 == 0);
#elif ASCEND_DEEPEP_ARCH == 2201
inline constexpr std::int64_t kMaxNumAiv = 48;
inline constexpr std::int64_t kKernelUbBytes = 192 * 1024;
inline constexpr std::int64_t kDispatchStageBytes = 176 * 1024;
static_assert(kKernelUbBytes <= 192 * 1024);
#else
#error "Unsupported ASCEND_DEEPEP_ARCH for ascend-deepep"
#endif

static_assert(kMaxNumAiv > 0 && kMaxNumAiv % 2 == 0);
static_assert(kKernelUbBytes > 0 && kKernelUbBytes % 32 == 0);
static_assert(
    kDispatchStageBytes > 0 &&
    kDispatchStageBytes < kKernelUbBytes &&
    kDispatchStageBytes % 32 == 0);

}  // namespace ascend_deepep::platform
