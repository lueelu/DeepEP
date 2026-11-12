// Copyright (c) 2026, Lu Lu
#pragma once
#include <array>
#include <cstdint>
#include "launch.hpp"

namespace deepep::ubx {
// Keep the isolated kernel headers out of the Mesh+Clos translation unit.
uint64_t v2_capacity(int world, int tokens, int hidden, int topk, int experts, uint32_t cores, uint32_t ranks_per_host);
std::array<uint8_t, 16> v2_tiling(int world, int tokens, int hidden, int topk, int experts, uint8_t dtype, int capacity,
                                  uint32_t cores, uint32_t ranks_per_host);
}  // namespace deepep::ubx
