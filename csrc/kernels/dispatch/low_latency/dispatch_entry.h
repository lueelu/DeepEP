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

#include "kernel_operator.h"
#include "shmem.h"

#include "../../launch.hpp"
#include "../../common/core_context.h"
#include "common/dispatch_dedup_types.h"
#include "common/ep_memory_server_dedup_tiling.h"
#include "common/ll_core_config.h"
#include "common/ll_dispatch_variant.h"
#include "common/ll_group_policy.h"
#include "common/ll_jit_spec.h"
#include "common/ll_reference_profile_layout.h"
#include "dispatch_server_dedup_kernel.h"
#include "launch.hpp"

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
                                uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType,
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
    profile.Finish(uint32_t(DEEPEP_LL_LAUNCH_RANK(commArgs)), DEEPEP_LL_LAUNCH_WORLD(commArgs), bs);
}

__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
#ifdef DEEPEP_LL_JIT_SPECIALIZED
    // The JIT unit bakes the entire launch identity. Tiling is neither read nor
    // uploaded for this build; the cache key keeps one binary per shape/world/
    // core combination, and the entry rechecks it against the immutable
    // communication description.
    (void)block.tilingData;
    constexpr uint32_t rankSize = DEEPEP_LL_JIT_SPEC_WORLD;
    constexpr uint32_t rank = DEEPEP_LL_JIT_SPEC_RANK;
    constexpr uint32_t bs = DEEPEP_LL_JIT_SPEC_BS;
    constexpr uint32_t h = DEEPEP_LL_JIT_SPEC_H;
    constexpr uint32_t topK = DEEPEP_LL_JIT_SPEC_TOPK;
    constexpr uint32_t moeExpertNum = DEEPEP_LL_JIT_SPEC_EXPERTS;
    constexpr uint32_t capacity = DEEPEP_LL_JIT_SPEC_CAPACITY;
    static_assert(capacity > 0U, "JIT dispatch specialization needs the per-rank capacity");
    constexpr uint32_t expertTokenNumsType = DispatchDedup::kLlExpertTokenNumsType;
    constexpr auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(DEEPEP_LL_JIT_SPEC_INPUT_MODE);
    constexpr bool e4m3 = bool(DEEPEP_LL_JIT_SPEC_E4M3);
    constexpr bool enableDedup = bool(DEEPEP_LL_JIT_SPEC_DEDUP);
    const auto* commArgs = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs);
    // The immutable communication description must agree with the compiled
    // specialization. Capacity is a fixed part of that identity; it still does
    // not make addresses, routing values or window state constant.
    if (uint32_t(commArgs->rankSize) != rankSize || uint32_t(commArgs->rank) != rank || core.count != DEEPEP_LL_JIT_SPEC_CORES) {
        AscendC::Trap();
        return;
    }
#if DEEPEP_LL_JIT_SPEC_INPUT_CLASS == 1
    RunGroup<bfloat16_t, DEEPEP_LL_JIT_SPEC_GROUP>(block, core, bs, h, topK, moeExpertNum, expertTokenNumsType,
                                                   inputMode, e4m3, enableDedup, skMode);
#else
    RunGroup<half, DEEPEP_LL_JIT_SPEC_GROUP>(block, core, bs, h, topK, moeExpertNum, expertTokenNumsType, inputMode,
                                             e4m3, enableDedup, skMode);
#endif
#else
    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(block.tilingData);
    const uint32_t rankSize = uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs)->rankSize);
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

    // Host chooses the same class from the same tiling builder; the device
    // rechecks it so an unsupported format can never run a different kernel.
    if (inputClass == DispatchDedup::kLlDispatchInvalidVariant) { AscendC::Trap(); return; }
    const bool e4m3 = expandXOutDtype == DispatchDedup::kLlDispatchE4M3;
    const uint32_t group = tiling->tilingKey;

#define DISPATCH_SERVER_DEDUP_RUN(XType, Group)                                                              \
    RunGroup<XType, Group>(block, core, bs, h, topK, moeExpertNum, expertTokenNumsType, inputMode, e4m3,     \
                           enableDedup, skMode)
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
#endif
}
}  // namespace DispatchDedupEntry

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales, GM_ADDR stageTimer, uint32_t timerMode, GM_ADDR expertTimer)
{
    const DispatchDedupEntry::ArgsBlock block{commArgs,     x,           expertIds,   expertScales, expandXOut,
                                             dynamicScalesOut, expertTokenNumsOut, sendCountsOut, tokenTypeOut,
                                             destinationIndexOut, relayReadIndexOut, sourceMaskOut, expertScalesOut,
                                             tilingData,   reservedMagic, inputScales,  stageTimer,    timerMode,
                                             expertTimer};
    DispatchDedupEntry::Run(block, deepep::OrdinaryCoreContext(), false);
}

// Equivalent SuperKernel sub-entries. The template parameter only gives the
// framework four distinct symbols; every instance runs the same body.
//
// The SuperKernel capability is a property of this translation unit, not of
// whether DeepJIT produced it: the Ahead-of-Time device library and the JIT ELF
// both carry the ordinary entry, the SK sub-entries and their bind record. The
// DeepJIT loader identifies ordinary entries from the ordinary kernel metadata;
// the __sk__ functions are not independent starts.
template <uint32_t splitNum>
__sk__ __vector__ void dispatch_server_dedup_kernel_sk(const DispatchDedupEntry::ArgsBlock* block,
                                                       sk::SkSystemArgs* sysArgs)
{
    sysArgs->SkInit();
    deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};
    // Unpack the framework-provided argument block once, like the ordinary
    // entry's register arguments, instead of re-reading each field from GM.
    const DispatchDedupEntry::ArgsBlock local = *block;
    DispatchDedupEntry::Run(local, core, true);
}

// Default capability mask: no early start and no disabled DCCI. The operator
// keeps its publish/consume protocol and does not rely on the compiler's
// end-of-kernel cache flush.
SK_BIND(dispatch_server_dedup_kernel, 0, dispatch_server_dedup_kernel_sk<0>, dispatch_server_dedup_kernel_sk<1>,
        dispatch_server_dedup_kernel_sk<2>, dispatch_server_dedup_kernel_sk<3>);
#endif
