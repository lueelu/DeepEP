// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "comm/kernel/alltoall_arguments.hpp"

namespace deepep::hierarchical {
using Layout = comm::kernel::AllToAllLayout<1, 1, 32>;
using comm::Align;
constexpr uint32_t kCoreCount = 32;
constexpr uint8_t kPadding = 0xa5;
inline uint64_t CheckBytes(const Layout& layout) { return uint64_t(layout.Windows()) * sizeof(uint64_t); }

DEEPEP_COMM_HOST_DEVICE uint8_t Pattern(uint32_t source, uint32_t peer, uint64_t index, bool scale)
{
    const uint64_t key = index * 17 + (index / 251) * 13 + source * 29 + peer * 43;
    // Finite FP8 payload encodings; positive finite E8M0 scales, no NaN encoding.
    return scale ? uint8_t(120 + key % 15) : uint8_t(1 + key % 119);
}

DEEPEP_COMM_HOST_DEVICE uint64_t HashByte(uint64_t hash, uint8_t value)
{
    return (hash ^ value) * 1099511628211ULL;
}
constexpr uint64_t kHashSeed = 14695981039346656037ULL;
} // namespace deepep::hierarchical
