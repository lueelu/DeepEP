// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {
#ifdef DEEPEP_MEGAMOE_BF16_BS96
inline constexpr uint32_t PROTOCOL = 5U;
inline constexpr uint32_t COMBINE_QPS = 24U;
#else
inline constexpr uint32_t PROTOCOL = 4U;
inline constexpr uint32_t COMBINE_QPS = 16U;
#endif
inline constexpr uint32_t DISPATCH_QPS = 8U;
inline constexpr uint32_t DISPATCH_WORKERS = 14U;
}  // namespace deepep::megamoe_config
