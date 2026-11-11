// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>

namespace deepep::ubx {
struct CombineArgs {
    uint8_t *comm, *expert, *destination, *counts, *weights, *types, *relay, *mask, *output, *tiling;
    uint32_t cores;
};
}  // namespace deepep::ubx
extern "C" void ubx_combine_launch(void* stream, const deepep::ubx::CombineArgs* args);
