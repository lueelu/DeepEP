// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "comm/common.hpp"

namespace deepep::comm::tile {
// Independent of tensor shape/dtype. Every incoming marker has a unique writer
// and cache line; host initializes each read-only send value to window+1.
// Writable slots use separate cache lines. Immutable send values are packed
// uint32 entries shared by all peer AIVs. Nothing is reset before final drains.
struct ReadyLayout {
    uint32_t world, qps, windows;
    DEEPEP_COMM_HOST_DEVICE uint64_t ReadyOffset(uint32_t source, uint32_t qp) const
    {
        return (uint64_t(source) * qps + qp) * kSlotBytes;
    }
    DEEPEP_COMM_HOST_DEVICE uint64_t FrontierOffset() const { return uint64_t(world) * qps * kSlotBytes; }
    DEEPEP_COMM_HOST_DEVICE uint64_t ValueOffset(uint32_t window) const
    {
        return FrontierOffset() + kSlotBytes + uint64_t(window) * sizeof(uint32_t);
    }
    DEEPEP_COMM_HOST_DEVICE uint64_t Bytes() const { return Align(ValueOffset(windows)); }
};
} // namespace deepep::comm::tile
