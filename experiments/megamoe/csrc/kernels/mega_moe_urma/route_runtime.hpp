// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/device_route_plan.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {
// Outgoing prefixes are immutable after local preparation. The receive portion
// is published later on separate cache lines and must be acquired before use.
__aicore__ inline const volatile __gm__ DeviceRouteInfo* RouteInfo(const Arguments& a)
{
    return reinterpret_cast<const volatile __gm__ DeviceRouteInfo*>(a.routeInfo);
}
__aicore__ inline uint32_t RuntimeExpertForExpandedRow(const Arguments& a, uint64_t row)
{
    const auto* r = RouteInfo(a);
    uint32_t lo = 0U, hi = a.config.world * a.config.localExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (r->outgoing[mid + 1U] <= row)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}
}  // namespace MegaMoeUrma
