#pragma once
#include "common/platform.h"

namespace ascend_deepep {
inline constexpr uint32_t kFusedTopkOffset = 106496;
inline constexpr uint32_t kFusedTopkBytes = 32768;
inline constexpr uint32_t kFusedHistogramOffset = kFusedTopkOffset + kFusedTopkBytes;
inline constexpr uint32_t kFusedHistogramBytes = platform::kKernelUbBytes - kFusedHistogramOffset;
static_assert(kFusedHistogramOffset < platform::kKernelUbBytes);
}  // namespace ascend_deepep
