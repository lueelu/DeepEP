// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "comm/tile/ready_layout.hpp"

namespace deepep::comm::kernel {
// Dense per-peer matrices, one scale per ScaleGroup columns. Host validates
// positive dimensions, N divisibility, per-window uint32 counts and allocation overflow.
// All windows occupy distinct immutable regions until every consumer finishes.
template <uint32_t DataElementBytes = 1, uint32_t ScaleElementBytes = 1, uint32_t ScaleGroup = 32>
struct AllToAllLayout {
    static_assert(DataElementBytes > 0 && ScaleElementBytes > 0 && ScaleGroup > 0);
    uint32_t m, n, world, windowM, qps;
    DEEPEP_COMM_HOST_DEVICE uint32_t Windows() const { return m / windowM + (m % windowM != 0); }
    DEEPEP_COMM_HOST_DEVICE uint32_t Rows(uint32_t window) const
    {
        const uint32_t left = m - window * windowM;
        return left < windowM ? left : windowM;
    }
    DEEPEP_COMM_HOST_DEVICE uint32_t ScaleN() const { return n / ScaleGroup; }
    DEEPEP_COMM_HOST_DEVICE uint64_t DataStride() const { return Align(uint64_t(m) * n * DataElementBytes); }
    DEEPEP_COMM_HOST_DEVICE uint64_t ScaleStride() const { return Align(uint64_t(m) * ScaleN() * ScaleElementBytes); }
    DEEPEP_COMM_HOST_DEVICE uint64_t DataBytes() const { return DataStride() * world; }
    DEEPEP_COMM_HOST_DEVICE uint64_t ScaleBytes() const { return ScaleStride() * world; }
    DEEPEP_COMM_HOST_DEVICE uint64_t TensorBytes() const { return 2 * (DataBytes() + ScaleBytes()); }
    DEEPEP_COMM_HOST_DEVICE tile::ReadyLayout Ready() const { return {world, qps, Windows()}; }
};

struct EmptyConsumerArguments {};
template <class Layout, class ConsumerArguments = EmptyConsumerArguments>
struct AllToAllArguments {
    Layout layout;
    uint32_t rank;
    uintptr_t input, output, inputScale, outputScale, control;
    ConsumerArguments consumer;
};
} // namespace deepep::comm::kernel
