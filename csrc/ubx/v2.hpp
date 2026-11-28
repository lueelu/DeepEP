// Copyright (c) 2026, Lu Lu
#pragma once
#include <array>
#include <cstdint>

namespace deepep::ubx {
uint32_t combine_ub_bytes(int64_t available);
uint64_t v2_capacity(int world, int tokens, int hidden, int topk, int experts, uint32_t cores,
                     uint32_t ranks_per_host, uint32_t ub_bytes);
std::array<uint8_t, 16> v2_tiling(int world, int tokens, int hidden, int topk, int experts, uint8_t dtype, int capacity,
                                  uint32_t cores, uint32_t ranks_per_host, uint32_t ub_bytes);
}  // namespace deepep::ubx
