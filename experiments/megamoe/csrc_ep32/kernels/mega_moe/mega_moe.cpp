// Copyright (c) 2026, Lu Lu
// MegaMoE direct-call entry for the shared DeepEP kernel library.
// Keep the wave implementations and their tiling layout independent of the launcher.
// Modified by SimpleBright_Man 2026
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "csrc_ep32/kernel_launch.hpp"
#include "kernels/mega_moe_common/mega_moe_tiling.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#if defined(__NPU_DEVICE__)
#include "kernels/mega_moe/mega_moe_wave_a8w8.hpp"
#include "kernels/mega_moe/mega_moe_wave_a8w4.hpp"
#endif

using namespace AscendC;

// Keep helper names distinct from ep32_mega_moe_kernel: profiler samples exclude them.
__global__ __mix__(1, 2) void ep32_benchmark_sleep_kernel(uint64_t cycles)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    const uint64_t start = GetSystemCycle();
    while (static_cast<uint64_t>(GetSystemCycle()) - start < cycles) {
        __asm__ __volatile__("nop");
    }
#endif
}

[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void benchmark_roce_barrier_kernel(uint64_t ffts_addr)
{
#if defined(__NPU_DEVICE__) && defined(DEEPEP_MEGAMOE_ROCE)
    // A single AIV owns the SDK WORLD collective. No default SDK UB or QP1
    // submissions are shared with MegaMoE: stream dependencies retire it first.
    InitSocState();
    util_set_ffts_config(ffts_addr);
    TPipe pipe;
    TBuf<TPosition::VECOUT> scratch;
    pipe.InitBuffer(scratch, 256);
    aclshmemx_roce_barrier_all<uint8_t>(reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr()), 0);
#endif
}

extern "C" __attribute__((visibility("default"))) void ep32_benchmark_sleep_do(uint32_t blocks, void* stream,
                                                                          uint64_t cycles)
{
    ep32_benchmark_sleep_kernel<<<blocks, 0, stream>>>(cycles);
}

extern "C" __attribute__((visibility("default"))) void ep32_benchmark_roce_barrier_do(void* stream, uint64_t ffts_addr)
{
    benchmark_roce_barrier_kernel<<<1, 0, stream>>>(ffts_addr);
}

// Explicit types replace asc_opc dtype/format/tiling-key specialisation.
// A8W4's existing matmul prologue continues to interpret weights as NZ/C0_32.
// Batch scheduling preserves the old ACL schemMode=1 cross-core contract.
template <bool IsA8W4, bool FuseGateUp = false>
__schedmode__(1) __global__
    __mix__(1, 2) void ep32_mega_moe_kernel(GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids,
                                       GM_ADDR topk_weights, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weight_scales1,
                                       GM_ADDR weight_scales2, GM_ADDR shared_weight1, GM_ADDR shared_weight2,
                                       GM_ADDR shared_weight_scales1, GM_ADDR shared_weight_scales2, GM_ADDR y,
                                       GM_ADDR expert_token_nums, GM_ADDR workspace, GM_ADDR timer,
                                       MegaMoeTilingData tiling, uint64_t ffts_addr)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(ffts_addr);
    constexpr int32_t kDispatchE4M3 = 4;
    MegaMoeTilingData invocation = tiling;
    if constexpr (IsA8W4) {
        MegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, fp4x2_e2m1_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, FuseGateUp>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &invocation, nullptr, timer);
        op.Process();
    } else {
        MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, false>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &invocation, nullptr);
        op.Process();
    }
#endif
}

extern "C" __attribute__((visibility("default"))) void ep32_mega_moe_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    static_assert(sizeof(MegaMoeTilingData) == 368, "MegaMoE tiling ABI changed");
    // 按本次实际 BS 编译期特化；不改变 wave 预算、Dispatch/Combine 或共享专家策略。
    if (is_a8w4 && tiling.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        ep32_mega_moe_kernel<true, true><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    } else if (is_a8w4) {
        ep32_mega_moe_kernel<true, false><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    } else {
        ep32_mega_moe_kernel<false><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    }
}
