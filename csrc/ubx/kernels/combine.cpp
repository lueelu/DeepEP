// Copyright (c) 2026, Lu Lu
#include "kernel_operator.h"
#include "shmem.h"
#include "../launch.hpp"
#include "combine_server_dedup_kernel.h"
#include "dispatch_server_dedup_magic.h"
#include "ep_memory_server_dedup_combine_tiling.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ubx_combine_route_plan(
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
    if (!UbxDispatch::LlTilingKeyValid(args->rankSize, td->tilingKey)) {
        AscendC::Trap();
        return;
    }
    const bool dedup = UbxDispatch::EpServerDedupEnabled(td->numMaxTokensPerRank, td->bs, args->rankSize);
#define UBX_COMBINE(Group)                                                                                       \
    UbxCombine::CombineMemoryServerDedupKernel<Group> kernel;                                                    \
    kernel.Init(comm, expert, destination, counts, weights, types, relay, mask, output, td->bs, td->h, td->topK, \
                td->moeExpertNum, td->dtype, magic, dedup);                                                      \
    kernel.Process()
    if (td->tilingKey == 2U) {
        UBX_COMBINE(2U);
    } else if (td->tilingKey == 4U) {
        UBX_COMBINE(4U);
    } else {
        UBX_COMBINE(16U);
    }
#undef UBX_COMBINE
}

extern "C" __attribute__((visibility("default"))) void ubx_combine_launch(void* stream,
                                                                          const deepep::ubx::CombineArgs* args)
{
    const auto& a = *args;
    ubx_combine_route_plan<<<a.cores, 0, stream>>>(a.comm, a.expert, a.destination, a.counts, a.weights, a.types,
                                                   a.relay, a.mask, a.output, a.tiling);
}
