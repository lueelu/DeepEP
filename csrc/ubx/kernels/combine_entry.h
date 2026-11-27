// Copyright (c) 2026, Lu Lu
#pragma once
#include <kernel_operator.h>
#include <shmem.h>
#include <ubx/kernels/combine_server_dedup_kernel.h>
#include <ubx/kernels/dispatch_server_dedup_magic.h>
#include <ubx/kernels/ep_memory_server_dedup_combine_tiling.h>

template <uint32_t Group, uint32_t Hidden, uint32_t TopK, uint32_t Experts>
__aicore__ inline void CombinePut(
    GM_ADDR comm, GM_ADDR expert, GM_ADDR destination, GM_ADDR counts, GM_ADDR weights, GM_ADDR types, GM_ADDR relay,
    GM_ADDR mask, GM_ADDR output, GM_ADDR tiling)
{
    const auto* args = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(comm);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = UbxDispatch::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }
    const auto* td = reinterpret_cast<const volatile __gm__ UbxCombine::EpServerDedupCombineTilingData*>(tiling);
    if (td->tilingKey != Group || !UbxDispatch::LlTilingKeyValid(args->rankSize, td->tilingKey)) {
        AscendC::Trap();
        return;
    }
    const bool dedup = deep_ep::EpServerDedupEnabled(td->numMaxTokensPerRank, args->rankSize);
    UbxCombine::CombineMemoryServerDedupKernel<Group> kernel;
    kernel.Init(comm, expert, destination, counts, weights, types, relay, mask, output,
                td->bs, Hidden, TopK, Experts, td->dtype, magic, dedup);
    kernel.Process();
}
