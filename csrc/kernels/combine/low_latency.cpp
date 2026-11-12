// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../launch.hpp"

#if CATLASS_ARCH == 3510
#include "low_latency/combine_entry.h"
#include "low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "low_latency/launch.hpp"

// Same in-library launch mechanism as the other LL kernels. JIT launches the
// entry through DeepJIT instead.
extern "C" DEEPEP_EXPORT void combine_server_dedup_kernel_do(void* stream,
                                                             const deepep::EpServerDedupCombineKernelArgs* args)
{
    const auto& a = *args;
    combine_server_dedup_kernel<<<CombineDedup::kEpServerDedupCombineCoreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.expertOut, a.destinationIndex, a.sendCounts, a.expertScales, a.tokenType, a.relayReadIndex,
        a.sourceMask, a.yOut, a.tilingData);
}
#endif
