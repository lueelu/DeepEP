// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"

#if CATLASS_ARCH == 3510
#include "low_latency/combine_server_dedup_kernel.h"
#include "low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../dispatch/low_latency/device/dispatch_server_dedup_magic.h"
#include "low_latency/launch.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR destinationIndex, GM_ADDR sendCounts, GM_ADDR expertScales,
    GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask, GM_ADDR yOut, GM_ADDR tilingData)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }
    const auto* td = reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(tilingData);
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank);
    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(commArgs, expertOut, destinationIndex, sendCounts, expertScales, tokenType, relayReadIndex, sourceMask,
                yOut, td->bs, td->h, td->topK, td->moeExpertNum, td->dtype, 0, magic, enableDedup);
    kernel.Process();
}
extern "C" DEEPEP_EXPORT void combine_server_dedup_kernel_do(void* stream,
                                                             const deepep::EpServerDedupCombineKernelArgs* args)
{
    const auto& a = *args;
    combine_server_dedup_kernel<<<64, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.expertOut, a.destinationIndex, a.sendCounts, a.expertScales, a.tokenType, a.relayReadIndex,
        a.sourceMask, a.yOut, a.tilingData);
}
#endif
