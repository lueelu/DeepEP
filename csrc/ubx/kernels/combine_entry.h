// Copyright (c) 2026, Lu Lu

// UBX RDMA Combine JIT entry. The ordinary and SuperKernel entries share this
// body; the Host passes the ten ubx::CombineArgs pointers and the vector-core
// count only as the launch grid.
#pragma once

#include <cstdint>

#include <kernel_operator.h>
#include <shmem.h>

#include <kernels/common/core_context.h>
#include <ubx/launch.hpp>
#include <ubx/kernels/combine_server_dedup_kernel.h>
#include <ubx/kernels/dispatch_server_dedup_magic.h>
#include <ubx/kernels/ep_memory_server_dedup_combine_tiling.h>

namespace UbxCombineEntry {

struct ArgsBlock {
    GM_ADDR comm;
    GM_ADDR expert;
    GM_ADDR destination;
    GM_ADDR counts;
    GM_ADDR weights;
    GM_ADDR types;
    GM_ADDR relay;
    GM_ADDR mask;
    GM_ADDR output;
    GM_ADDR tiling;
    uint32_t cores;
};
static_assert(sizeof(ArgsBlock) == sizeof(deepep::ubx::CombineArgs),
              "UBX Combine argument block must match ubx::CombineArgs");

template <uint32_t Group, uint32_t Capacity, bool Dedup, uint32_t Cores>
__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
    const auto* args = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(block.comm);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = UbxDispatch::RotateDispatchServerDedupMagic(args, core.index, &pipe);
        if (skMode) pipe.DestroyWithoutPipeAll();
    }
    if (skMode) AscendC::PipeBarrier<PIPE_ALL>();

    const auto* td = reinterpret_cast<const volatile __gm__ UbxCombine::EpServerDedupCombineTilingData*>(block.tiling);
    if (!UbxDispatch::LlTilingKeyValid(args->rankSize, td->tilingKey) || td->tilingKey != Group ||
        uint32_t(td->numMaxTokensPerRank) != Capacity) {
        AscendC::Trap();
        return;
    }
    UbxCombine::CombineMemoryServerDedupKernel<Group> kernel;
    kernel.Init(block.comm, block.expert, block.destination, block.counts, block.weights, block.types, block.relay,
                block.mask, block.output, td->bs, td->h, td->topK, td->moeExpertNum, td->dtype, magic, Dedup, core);
    kernel.Process();
}

}  // namespace UbxCombineEntry
