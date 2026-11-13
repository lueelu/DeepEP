// Copyright (c) 2026, Lu Lu

// Host-side LL protocol constants that used to live in the AOT device library.
// They are pure Host values; the device entries are JIT-compiled.
#include <cstdint>

#include "kernels/dispatch/low_latency/launch.hpp"
#include "kernels/dispatch/low_latency/common/ll_core_config.h"
#include "kernels/dispatch/low_latency/common/ll_group_policy.h"
#include "kernels/dispatch/low_latency/common/ll_reference_profile_layout.h"

extern "C" uint32_t deepep_ll_stage_timer_enabled()
{
    return 0U;
}

extern "C" uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}

extern "C" uint32_t deepep_ll_reference_profile_schema()
{
    return LlReferenceProfile::kSchema;
}

extern "C" uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return world <= 16U ? uint32_t(DispatchDedup::LlTilingKey::Group4)
                        : uint32_t(DispatchDedup::LlTilingKey::Group16);
}

extern "C" uint32_t deepep_ll_core_protocol_version()
{
#ifdef DEEPEP_LL_COUNT_HISTOGRAM
    return 35U;
#else
    return 34U;
#endif
}

extern "C" uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}
