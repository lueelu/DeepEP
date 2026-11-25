// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

// MegaMoE MTE device entry shared by the runtime DeepJIT source. The ordinary
// kernel keeps its original by-value parameter ABI; the SuperKernel entries
// read the same packed parameter block through MegaMoeArgs so both launches
// share one ABI, exactly like the LL dispatch entry.
#pragma once

#include <cstddef>
#include <cstdint>

#include "basic_api/kernel_basic_intf.h"
#include "kernels/mega_moe_common/mega_moe_tiling.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"

#if defined(__NPU_DEVICE__)
#include "mega_moe_wave_a8w4.hpp"
#include "mega_moe_wave_a8w8.hpp"
#endif

namespace MegaMoeEntry {

// Runtime argument block of the SuperKernel sub-entry. One field per explicit
// kernel parameter and in the same order, so the bind record and the ordinary
// launch share one ABI.
struct MegaMoeArgs {
    __gm__ uint8_t* context;
    __gm__ uint8_t* x;
    __gm__ uint8_t* xScales;
    __gm__ uint8_t* topkIds;
    __gm__ uint8_t* topkWeights;
    __gm__ uint8_t* weight1;
    __gm__ uint8_t* weight2;
    __gm__ uint8_t* weightScales1;
    __gm__ uint8_t* weightScales2;
    __gm__ uint8_t* sharedWeight1;
    __gm__ uint8_t* sharedWeight2;
    __gm__ uint8_t* sharedWeightScales1;
    __gm__ uint8_t* sharedWeightScales2;
    __gm__ uint8_t* output;
    __gm__ uint8_t* expertTokenNums;
    __gm__ uint8_t* workspace;
    __gm__ uint8_t* timer;
    MegaMoeTilingData tiling;
    uint64_t ffts;
};
static_assert(sizeof(MegaMoeArgs) == 368U, "MegaMoE SuperKernel argument view must match the global ABI");
static_assert(offsetof(MegaMoeArgs, tiling) == 136U, "MegaMoE tiling slot offset");
static_assert(offsetof(MegaMoeArgs, ffts) == 360U, "MegaMoE ffts slot offset");

#if defined(__NPU_DEVICE__)
using namespace AscendC;

template <bool IsA8W4, bool FuseGateUp, bool EarlyWeights>
__aicore__ inline void MegaMoeBody(const MegaMoeArgs* args)
{
    InitSocState();
    util_set_ffts_config(args->ffts);
    constexpr int32_t kDispatchE4M3 = 4;
    // The wave keeps its own tiling copy: Init takes a mutable pointer while the
    // SuperKernel argument block is read-only.
    MegaMoeTilingData tiling = args->tiling;
    if constexpr (IsA8W4) {
        MegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, fp4x2_e2m1_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, EarlyWeights, FuseGateUp>
            op;
        op.Init(args->context, args->x, args->topkIds, args->topkWeights, args->weight1, args->weight2, nullptr,
                args->weightScales1, args->weightScales2, args->xScales, args->sharedWeight1, args->sharedWeight2,
                args->sharedWeightScales1, args->sharedWeightScales2, args->output, args->expertTokenNums,
                args->workspace, &tiling, nullptr, args->timer);
        op.Process();
    } else {
        MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, EarlyWeights, false>
            op;
        op.Init(args->context, args->x, args->topkIds, args->topkWeights, args->weight1, args->weight2, nullptr,
                args->weightScales1, args->weightScales2, args->xScales, args->sharedWeight1, args->sharedWeight2,
                args->sharedWeightScales1, args->sharedWeightScales2, args->output, args->expertTokenNums,
                args->workspace, &tiling, nullptr);
        op.Process();
    }
}

template <bool IsA8W4, bool FuseGateUp, bool EarlyWeights, uint32_t kSkSlot>
__aicore__ inline void MegaMoeSk(const MegaMoeArgs* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    MegaMoeBody<IsA8W4, FuseGateUp, EarlyWeights>(args);
}
#endif

}  // namespace MegaMoeEntry
