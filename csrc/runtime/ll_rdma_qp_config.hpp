#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace deepep {
// Native ROCE capacity in the pinned SHMEM master (8e3fc610).
// This is separate from LL's 72-core control storage capacity.
constexpr uint32_t kLlNativeRoceQpLimit = 56;

inline void validate_ll_rdma_qps(uint32_t qps, bool cross_host_rdma)
{
    if (cross_host_rdma && (qps == 0U || qps > kLlNativeRoceQpLimit)) {
        throw std::runtime_error(
            "Cross-host RDMA QP count=" +
            std::to_string(qps) + ", pinned SHMEM native QP limit=" +
            std::to_string(kLlNativeRoceQpLimit) +
            "; unsupported before SHMEM initialization");
    }
}
}  // namespace deepep
