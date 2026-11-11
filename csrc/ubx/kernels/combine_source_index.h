// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>

// Dispatch-produced counts, destination/relay tables and masks remain int32.
namespace UbxCombine {
using SourceIndex = int32_t;
constexpr uint32_t kSourceIndexBits = 32;
constexpr uint32_t kSourceIndexBytes = sizeof(SourceIndex);
constexpr bool kSourceIndicesRead = false;
}  // namespace UbxCombine
