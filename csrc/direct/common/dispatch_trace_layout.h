#pragma once
#include <cstdint>

namespace ascend_deepep::trace {
// Each AIV owns two independent cache lines in a registered Host allocation.
// Only the first aligned uint64_t on each line is published by MTE3.
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kCoreBytes = 2 * kLineBytes;
constexpr uint32_t kTickHz = 1000000000;  // Ascend950
}  // namespace ascend_deepep::trace
