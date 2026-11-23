// Copyright (c) 2026, Lu Lu
#pragma once
#include <deep_ep/common/compiled.hpp>

namespace UbxDispatch {
constexpr uint32_t kLlPhysicalRanksPerHost = deep_ep::kNumRanksPerHost;
constexpr uint32_t kLlMaxRanksPerGroup = deep_ep::kNumMaxRanksPerGroup;
using deep_ep::LlGroupForWorld;
using deep_ep::LlTilingKeyForWorld;
using deep_ep::LlTilingKeyValid;
using deep_ep::LlSameHost;
}  // namespace UbxDispatch
