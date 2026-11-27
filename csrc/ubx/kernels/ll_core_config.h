// Copyright (c) 2026, Lu Lu
#pragma once
#include <deep_ep/layout/ep/workspace.hpp>

namespace UbxDispatch {
constexpr uint32_t kLlDefaultPlanCoreNum = deep_ep::kEpServerDedupMaxVecCores;
constexpr uint32_t kLlMaxRecvCoreNum = deep_ep::kEpServerDedupMaxVecCores / 2U;
using deep_ep::LlCoreCountValid;
}  // namespace UbxDispatch
