// Copyright (c) 2026, Lu Lu

// One LL dispatch entry. The kernel body lives here rather than in the
// translation unit so the same source can be compiled ahead of time or through
// a JIT backend without duplicating any logic. The input element class and the
// group size are compile-time in every build; the JIT backend additionally
// bakes B/H/K/E, per-rank capacity, world/rank, the active AIV count, the input
// mode, E4M3 and the dedup decision into the translation unit
// (DEEPEP_LL_JIT_SPECIALIZED). The Ahead-of-Time build keeps those as runtime
// tiling/communication fields of the two instantiations it already has.
//
// The ordinary global entry, the SuperKernel sub-entry and the bind record live
// in this one source for both routes. The SK sub-entry is always compiled; the
// external caller's SuperKernel schedule decides whether it is used, not a
// compile-time switch or a runtime environment variable.
//
// The body receives its logical execution context instead of reading
// AscendC::GetBlockIdx()/GetBlockNum() directly, because inside a fused
// SuperKernel task those accessors describe the fused grid.
#pragma once

#include <cstddef>
#include <cstdint>

#include <kernel_operator.h>
#include <shmem.h>

#include <kernels/launch.hpp>
#include <kernels/common/core_context.h>
#include <kernels/dispatch/low_latency/common/dispatch_dedup_types.h>
#include <kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>
#include <kernels/dispatch/low_latency/common/ll_core_config.h>
#include <kernels/dispatch/low_latency/common/ll_dispatch_variant.h>
#include <kernels/dispatch/low_latency/common/ll_group_policy.h>
#include <kernels/dispatch/low_latency/common/ll_reference_profile_layout.h>
#include <kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h>
#include <kernels/dispatch/low_latency/launch.hpp>

#if CATLASS_ARCH == 3510
namespace DispatchDedupEntry {
// Mirrors the ordinary global entry's parameter list, one field per argument
// and in the same order. GM_ADDR is pointer sized, int64_t/uint32_t keep their
// natural alignment, so the block matches the Host launch ABI exactly.
struct ArgsBlock {
    GM_ADDR commArgs;
    GM_ADDR x;
    GM_ADDR expertIds;
    GM_ADDR expertScales;
    GM_ADDR expandXOut;
    GM_ADDR dynamicScalesOut;
    GM_ADDR expertTokenNumsOut;
    GM_ADDR sendCountsOut;
    GM_ADDR tokenTypeOut;
    GM_ADDR destinationIndexOut;
    GM_ADDR relayReadIndexOut;
    GM_ADDR sourceMaskOut;
    GM_ADDR expertScalesOut;
    GM_ADDR tilingData;
    int64_t reservedMagic;
    GM_ADDR inputScales;
    GM_ADDR stageTimer;
    uint32_t timerMode;
    GM_ADDR expertTimer;
};
static_assert(sizeof(ArgsBlock) == 152U, "SuperKernel argument view must match the global ABI");
static_assert(offsetof(ArgsBlock, reservedMagic) == 112U, "reserved magic slot offset");
static_assert(offsetof(ArgsBlock, inputScales) == 120U, "input scale slot offset");
static_assert(offsetof(ArgsBlock, timerMode) == 136U, "timer mode slot offset");
static_assert(offsetof(ArgsBlock, expertTimer) == 144U, "expert timer slot offset");

template <class XType, uint32_t Group>
__aicore__ inline void RunGroup(const ArgsBlock& block, deepep::CoreContext core, uint32_t bs, uint32_t h,
                                uint32_t topK, uint32_t experts, uint32_t rankSize, uint32_t rank, uint32_t expertTokenNumsType,
                                DispatchDedup::EpServerDedupInputMode inputMode, bool e4m3, bool enableDedup,
                                bool skMode)
{
    // Reserved launch ABI slots; legacy tracing removed.
    (void)block.timerMode;
    (void)block.expertTimer;
    const uint64_t referenceEntry = LlReferenceProfile::DeviceProfile::EntryClock(block.stageTimer);
    util_set_ffts_config(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs)->fftsVal);
    LlReferenceProfile::DeviceProfile profile;
    profile.Init(block.stageTimer, core);
    profile.Value<0>(referenceEntry);  // w0: kernel_entry
    profile.Mark<1>();  // w1: config_ready

    AscendC::TPipe pipe;
    DispatchDedup::DispatchMemoryServerDedupKernel<XType, Group> kernel;
    kernel.Init(block.commArgs, block.x, block.expertIds, block.expertScales, block.expandXOut, block.dynamicScalesOut,
                block.expertTokenNumsOut, block.sendCountsOut, block.tokenTypeOut, block.destinationIndexOut,
                block.relayReadIndexOut, block.sourceMaskOut, block.expertScalesOut, bs, h, topK, experts,
                expertTokenNumsType, inputMode, e4m3, &pipe, core, block.inputScales, enableDedup, &profile);
    kernel.Process();
    // A SuperKernel sub-task is an ordinary function to the compiler, so the
    // implicit destructor no longer guarantees the pipeline barrier. All
    // business DMA and transfer retirement completed inside Process().
    if (skMode) pipe.DestroyWithoutPipeAll();
    const auto* commArgs = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs);
    profile.Finish(rank, rankSize, bs);
}

template <uint32_t RankSize, uint32_t Rank, uint32_t BS, uint32_t H, uint32_t TopK, uint32_t Experts,
          uint32_t Capacity, uint32_t Cores, uint32_t InputClass, uint32_t Group, uint32_t InputMode,
          bool E4M3, bool Dedup>
__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
    (void)block.tilingData;
    const auto* commArgs = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs);
    static_assert(Capacity > 0U, "JIT dispatch specialization needs the per-rank capacity");
    if (uint32_t(commArgs->rankSize) != RankSize || uint32_t(commArgs->rank) != Rank || core.count != Cores) {
        AscendC::Trap();
        return;
    }
    constexpr uint32_t expertTokenNumsType = DispatchDedup::kLlExpertTokenNumsType;
    constexpr auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(InputMode);
    if constexpr (InputClass == 1) {
        RunGroup<bfloat16_t, Group>(block, core, BS, H, TopK, Experts, RankSize, Rank, expertTokenNumsType, inputMode, E4M3, Dedup,
                                   skMode);
    } else {
        RunGroup<half, Group>(block, core, BS, H, TopK, Experts, RankSize, Rank, expertTokenNumsType, inputMode, E4M3, Dedup, skMode);
    }
}
}  // namespace DispatchDedupEntry
#endif  // CATLASS_ARCH == 3510

