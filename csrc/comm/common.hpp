// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include <cstdint>

#ifdef __CCE__
#define DEEPEP_COMM_HOST_DEVICE __forceinline__[host, aicore]
#define DEEPEP_COMM_DEVICE __forceinline__ __aicore__
#else
#define DEEPEP_COMM_HOST_DEVICE inline
#define DEEPEP_COMM_DEVICE inline
#endif

namespace deepep::comm {
enum class CopyTransport { Mte, Rdma, Udma };
// Deferred completion is an explicit caller contract, not the generic default.
enum class RdmaCompletion { Wait, Deferred };
constexpr uint32_t kMaxQps = 4;
constexpr uint32_t kSlotBytes = 64;
DEEPEP_COMM_HOST_DEVICE uint64_t Align(uint64_t bytes, uint64_t alignment = kSlotBytes)
{
    return (bytes + alignment - 1) / alignment * alignment;
}

// Current peer tensor shape, or a {row block, column block, peer} coordinate.
struct CommCoord {
    uint32_t rows, columns, peers;
    DEEPEP_COMM_HOST_DEVICE uint32_t row() const { return rows; }
    DEEPEP_COMM_HOST_DEVICE uint32_t column() const { return columns; }
    DEEPEP_COMM_HOST_DEVICE uint32_t rank() const { return peers; }
};
} // namespace deepep::comm
