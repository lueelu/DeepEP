// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// One LL dispatch entry and one device translation unit, matching main. Only the
// input element class and the group size are compile-time; plain, quantized and
// prequantized wire formats are runtime tiling fields.
#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"
#include "low_latency/dispatch_server_dedup_kernel.h"
#include "low_latency/common/dispatch_dedup_types.h"
#include "low_latency/common/ep_memory_server_dedup_tiling.h"
#include "low_latency/common/ll_core_config.h"
#include "low_latency/common/ll_dispatch_variant.h"
#include "low_latency/common/ll_group_policy.h"
#include "low_latency/common/ll_reference_profile_layout.h"
#include "low_latency/launch.hpp"

#if CATLASS_ARCH == 3510
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales, GM_ADDR stageTimer, uint32_t timerMode, GM_ADDR expertTimer)
{
    (void)timerMode; (void)expertTimer;  // Reserved launch ABI slots; legacy tracing removed.
    const uint64_t referenceEntry = LlReferenceProfile::DeviceProfile::EntryClock(stageTimer);
    util_set_ffts_config(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->fftsVal);
    LlReferenceProfile::DeviceProfile profile;
    profile.Init(stageTimer);
    profile.Value<0>(referenceEntry);  // w0: kernel_entry
    profile.Mark<1>();  // w1: config_ready

    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tilingData);
    const uint32_t rankSize = uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rankSize);
    if (!DispatchDedup::LlTilingKeyValid(rankSize, tiling->tilingKey)) { AscendC::Trap(); return; }
    const uint32_t bs = tiling->bs;
    const uint32_t h = tiling->h;
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank, bs, rankSize);
    const uint32_t topK = tiling->topK;
    const uint32_t moeExpertNum = tiling->moeExpertNum;
    const uint32_t expertTokenNumsType = tiling->expertTokenNumsType;
    const uint8_t quantMode = tiling->quantMode;
    const uint8_t dtype = tiling->dtype;
    const uint8_t expandXOutDtype = tiling->expandXOutDtype;
    const auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    const uint32_t inputClass =
        DispatchDedup::LlDispatchInputClass(dtype, quantMode, expandXOutDtype, uint8_t(inputMode));
    AscendC::TPipe pipe;

    // Host chooses the same class from the same tiling builder; the device
    // rechecks it so an unsupported format can never run a different kernel.
    if (inputClass == DispatchDedup::kLlDispatchInvalidVariant) { AscendC::Trap(); return; }
    const bool e4m3 = expandXOutDtype == DispatchDedup::kLlDispatchE4M3;
    const uint32_t group = tiling->tilingKey;

#define DISPATCH_SERVER_DEDUP_RUN(XType, Group)                                                              \
    {                                                                                                        \
        DispatchDedup::DispatchMemoryServerDedupKernel<XType, Group> kernel;                                 \
        kernel.Init(commArgs, x, expertIds, expertScales, expandXOut, dynamicScalesOut, expertTokenNumsOut,   \
                    sendCountsOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut, sourceMaskOut,       \
                    expertScalesOut, bs, h, topK, moeExpertNum, expertTokenNumsType, inputMode, e4m3, &pipe,  \
                    inputScales, enableDedup, &profile);                                                     \
        kernel.Process();                                                                                    \
    }
    if (inputClass == DispatchDedup::kLlDispatchInputBf16) {
        if (group == uint32_t(DispatchDedup::LlTilingKey::Group4)) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 4U);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 16U);
        }
    } else {
        if (group == uint32_t(DispatchDedup::LlTilingKey::Group4)) {
            DISPATCH_SERVER_DEDUP_RUN(half, 4U);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(half, 16U);
        }
    }
#undef DISPATCH_SERVER_DEDUP_RUN

    profile.Finish(uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rank), rankSize, bs);
}

// Same in-library launch mechanism as every other LL kernel.
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
