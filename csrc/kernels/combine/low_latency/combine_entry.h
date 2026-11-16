// Copyright (c) 2026, Lu Lu

// LL Combine entry shared by the Ahead-of-Time device library and the DeepJIT
// translation unit. The ordinary global entry and the SuperKernel sub-entry
// call the same body with an explicit logical core context.
#pragma once

#include <cstddef>
#include <cstdint>

#include <kernel_operator.h>
#include <shmem.h>

#include <kernels/launch.hpp>
#include <kernels/common/core_context.h>
#include <kernels/dispatch/low_latency/common/dispatch_dedup_types.h>
#include <kernels/dispatch/low_latency/device/dispatch_server_dedup_magic.h>
#include <kernels/combine/low_latency/combine_server_dedup_kernel.h>
#include <kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>
#include <kernels/combine/low_latency/launch.hpp>

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

template <uint32_t BS, uint32_t H, uint32_t TopK, uint32_t Experts, uint32_t World, uint32_t Rank, uint32_t Cores,
          uint32_t Capacity, bool Bf16, bool Dedup>
__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(block.commArgs);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, core.index, &pipe);
        if (skMode) pipe.DestroyWithoutPipeAll();
    }
    if (skMode) AscendC::PipeBarrier<PIPE_ALL>();

    const auto* td =
        reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(block.tilingData);
    constexpr uint8_t dtype = Bf16 ? DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16
                                   : DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (uint32_t(args->rankSize) != World || uint32_t(args->rank) != Rank || core.count != Cores || td->bs != BS ||
        td->h != H || td->topK != TopK || td->moeExpertNum != Experts || td->dtype != dtype ||
        uint32_t(td->numMaxTokensPerRank) != Capacity ||
        DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank, td->bs, uint32_t(args->rankSize)) != Dedup) {
        AscendC::Trap();
        return;
    }
    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(block.commArgs, block.expertOut, block.destinationIndex, block.sendCounts, block.expertScales,
                block.tokenType, block.relayReadIndex, block.sourceMask, block.yOut, BS, H, TopK, Experts, dtype, 0,
                magic, World, core, Dedup);
    kernel.Process();
    if (skMode) kernel.Destroy();
}
}  // namespace CombineDedupEntry

