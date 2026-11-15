#pragma once

// Copyright (c) 2026, Lu Lu
// MegaMoE direct-call entry for the shared DeepEP kernel library.
// Keep the wave implementations and their tiling layout independent of the launcher.
// Modified by SimpleBright_Man 2026
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_launch.hpp"
#include "mega_moe_tiling.hpp"
#include "common/mega_moe_constants.hpp"
#if defined(__NPU_DEVICE__)
#include "mega_moe_wave_a8w8.hpp"
#include "mega_moe_wave_a8w4.hpp"
#endif

using namespace AscendC;

// A single argument block keeps the ordinary entry and the SuperKernel
// sub-entry ABI-identical.
struct MegaMoeArgs {
    GM_ADDR context;
    GM_ADDR x;
    GM_ADDR x_scales;
    GM_ADDR topk_ids;
    GM_ADDR topk_weights;
    GM_ADDR weight1;
    GM_ADDR weight2;
    GM_ADDR weight_scales1;
    GM_ADDR weight_scales2;
    GM_ADDR shared_weight1;
    GM_ADDR shared_weight2;
    GM_ADDR shared_weight_scales1;
    GM_ADDR shared_weight_scales2;
    GM_ADDR y;
    GM_ADDR expert_token_nums;
    GM_ADDR workspace;
    GM_ADDR timer;
    MegaMoeTilingData tiling;
    uint64_t ffts_addr;
};

template <uint32_t num_max_tokens_per_rank, uint32_t hidden, uint32_t num_ranks,
          uint32_t num_topk, uint32_t num_vec_cores, bool is_a8w4, bool fuse_gate_up = false>
__aicore__ inline void MegaMoeRun(const MegaMoeArgs& args, deepep::CoreContext core_context)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(args.ffts_addr);
    MegaMoeTilingData tiling = args.tiling;
    constexpr int32_t kDispatchE4M3 = 4;
    if constexpr (is_a8w4) {
        MegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, fp4x2_e2m1_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, fuse_gate_up>
            op;
        op.SetCoreContext(core_context);
        op.Init(args.context, args.x, args.topk_ids, args.topk_weights, args.weight1, args.weight2, nullptr,
                args.weight_scales1, args.weight_scales2, args.x_scales, args.shared_weight1, args.shared_weight2,
                args.shared_weight_scales1, args.shared_weight_scales2, args.y, args.expert_token_nums, args.workspace,
                &tiling, nullptr, args.timer);
        op.Process();
    } else {
        MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, false>
            op;
        op.SetCoreContext(core_context);
        op.Init(args.context, args.x, args.topk_ids, args.topk_weights, args.weight1, args.weight2, nullptr,
                args.weight_scales1, args.weight_scales2, args.x_scales, args.shared_weight1, args.shared_weight2,
                args.shared_weight_scales1, args.shared_weight_scales2, args.y, args.expert_token_nums, args.workspace,
                &tiling, nullptr);
        op.Process();
    }
#else
    (void)args;
    (void)core_context;
#endif
}

template <uint32_t num_max_tokens_per_rank, uint32_t hidden, uint32_t num_ranks,
          uint32_t num_topk, uint32_t num_vec_cores, bool is_a8w4, bool fuse_gate_up = false>
__schedmode__(1) __global__ __mix__(1, 2) void mega_moe_kernel(
    GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids, GM_ADDR topk_weights, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR weight_scales1, GM_ADDR weight_scales2, GM_ADDR shared_weight1, GM_ADDR shared_weight2,
    GM_ADDR shared_weight_scales1, GM_ADDR shared_weight_scales2, GM_ADDR y, GM_ADDR expert_token_nums,
    GM_ADDR workspace, GM_ADDR timer, MegaMoeTilingData tiling, uint64_t ffts_addr)
{
    const MegaMoeArgs args{context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1,
                           weight_scales2, shared_weight1, shared_weight2, shared_weight_scales1,
                           shared_weight_scales2, y, expert_token_nums, workspace, timer, tiling, ffts_addr};
    MegaMoeRun<num_max_tokens_per_rank, hidden, num_ranks, num_topk, num_vec_cores, is_a8w4, fuse_gate_up>(args, deepep::OrdinaryCoreContext());
}
