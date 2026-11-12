// Copyright (c) 2026, Lu Lu

// LL Combine entry shared by the Ahead-of-Time device library and the DeepJIT
// translation unit. The ordinary global entry and the SuperKernel sub-entry
// call the same body with an explicit logical core context.
#pragma once

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../../launch.hpp"
#include "../../common/core_context.h"
#include "../../dispatch/low_latency/common/dispatch_dedup_types.h"
#include "../../dispatch/low_latency/device/dispatch_server_dedup_magic.h"
#include "combine_server_dedup_kernel.h"
#include "ep_memory_server_dedup_combine_tiling.h"
#include "launch.hpp"

namespace CombineDedupEntry {
struct ArgsBlock {
    GM_ADDR commArgs;
    GM_ADDR expertOut;
    GM_ADDR destinationIndex;
    GM_ADDR sendCounts;
    GM_ADDR expertScales;
    GM_ADDR tokenType;
    GM_ADDR relayReadIndex;
    GM_ADDR sourceMask;
    GM_ADDR yOut;
    GM_ADDR tilingData;
};
static_assert(sizeof(ArgsBlock) == 10U * sizeof(GM_ADDR), "SuperKernel argument view must match the global ABI");

__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, core.index, &pipe);
        // A SuperKernel sub-task is an ordinary function to the compiler, so the
        // implicit destructor no longer guarantees the pipeline barrier.
        if (skMode) pipe.DestroyWithoutPipeAll();
    }
    if (skMode) AscendC::PipeBarrier<PIPE_ALL>();

    const auto* td =
        reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(block.tilingData);

#ifdef DEEPEP_LL_JIT_SPECIALIZED
    // The JIT unit bakes the launch identity. Tiling is rechecked rather than
    // trusted, and no Host-controllable field selects a different kernel.
    (void)td;
    constexpr uint32_t bs = DEEPEP_LL_JIT_SPEC_BS;
    constexpr uint32_t h = DEEPEP_LL_JIT_SPEC_H;
    constexpr uint32_t topK = DEEPEP_LL_JIT_SPEC_TOPK;
    constexpr uint32_t experts = DEEPEP_LL_JIT_SPEC_EXPERTS;
    constexpr uint32_t world = DEEPEP_LL_JIT_SPEC_WORLD;
    constexpr uint32_t cores = DEEPEP_LL_JIT_SPEC_CORES;
    constexpr uint8_t dtype = DEEPEP_LL_JIT_SPEC_BF16 ? DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16
                                                     : DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    constexpr bool dedup = bool(DEEPEP_LL_JIT_SPEC_DEDUP);
    if (uint32_t(args->rankSize) != world || core.count != cores ||
        td->bs != bs || td->h != h || td->topK != topK || td->moeExpertNum != experts || td->dtype != dtype ||
        DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank) != dedup) {
        AscendC::Trap();
        return;
    }
#else
    const uint32_t bs = td->bs;
    const uint32_t h = td->h;
    const uint32_t topK = td->topK;
    const uint32_t experts = td->moeExpertNum;
    const uint8_t dtype = td->dtype;
    const uint32_t world = uint32_t(args->rankSize);
    const bool dedup = DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank);
#endif

    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(block.commArgs, block.expertOut, block.destinationIndex, block.sendCounts, block.expertScales,
                block.tokenType, block.relayReadIndex, block.sourceMask, block.yOut, bs, h, topK, experts, dtype, 0,
                magic, world, core, dedup);
    kernel.Process();
    // Member TPipe boundary: business DMA and output retirement already
    // completed inside Process().
    if (skMode) kernel.Destroy();
}
}  // namespace CombineDedupEntry

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR destinationIndex, GM_ADDR sendCounts, GM_ADDR expertScales,
    GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask, GM_ADDR yOut, GM_ADDR tilingData)
{
    const CombineDedupEntry::ArgsBlock block{commArgs, expertOut, destinationIndex, sendCounts, expertScales,
                                             tokenType, relayReadIndex, sourceMask, yOut, tilingData};
    CombineDedupEntry::Run(block, deepep::OrdinaryCoreContext(), false);
}

#ifndef DEEPEP_LL_JIT
// Equivalent SuperKernel sub-entries. The JIT translation unit omits them
// because the DeepJIT loader requires exactly one kernel per ELF; fusion uses
// the AOT device library.
template <uint32_t splitNum>
__sk__ __vector__ void combine_server_dedup_kernel_sk(const CombineDedupEntry::ArgsBlock* block,
                                                      sk::SkSystemArgs* sysArgs)
{
    deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};
    const CombineDedupEntry::ArgsBlock local = *block;
    CombineDedupEntry::Run(local, core, true);
}

SK_BIND(combine_server_dedup_kernel, 0, combine_server_dedup_kernel_sk<0>, combine_server_dedup_kernel_sk<1>,
        combine_server_dedup_kernel_sk<2>, combine_server_dedup_kernel_sk<3>);
#endif
