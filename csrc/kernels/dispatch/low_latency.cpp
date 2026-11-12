// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// One LL dispatch entry and one device translation unit, matching main. Only the
// input element class and the group size are compile-time; plain, quantized and
// prequantized wire formats are runtime tiling fields. The entry itself lives in
// dispatch_entry.h so the DeepJIT translation unit compiles the same body.
#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"
#include "low_latency/dispatch_entry.h"
#include "low_latency/common/dispatch_dedup_types.h"
#include "low_latency/common/ep_memory_server_dedup_tiling.h"
#include "low_latency/common/ll_core_config.h"
#include "low_latency/common/ll_dispatch_variant.h"
#include "low_latency/common/ll_group_policy.h"
#include "low_latency/common/ll_reference_profile_layout.h"
#include "low_latency/launch.hpp"

#if CATLASS_ARCH == 3510
// Same in-library launch mechanism as every other LL kernel. JIT launches the
// entry through DeepJIT instead.
extern "C" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do(void* stream, const deepep::EpServerDedupKernelArgs* args)
{
    const auto& a = *args;
    dispatch_server_dedup_kernel<<<a.coreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.x, a.expertIds, a.expertScales, a.expandXOut, a.dynamicScalesOut, a.expertTokenNumsOut,
        a.sendCountsOut, a.tokenTypeOut, a.destinationIndexOut, a.relayReadIndexOut, a.sourceMaskOut, a.expertScalesOut,
        a.tilingData, a.reservedMagic, a.inputScales, a.stageTimer, a.timerMode, a.expertTimer);
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_stage_timer_enabled()
{
    return 0U;  // Legacy stage/expert device instrumentation has been removed.
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_schema()
{
    return LlReferenceProfile::kSchema;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return world <= 16U ? uint32_t(DispatchDedup::LlTilingKey::Group4) : uint32_t(DispatchDedup::LlTilingKey::Group16);
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_core_protocol_version()
{
    return DispatchDedup::kLlCoreProtocolVersion;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}
#endif
