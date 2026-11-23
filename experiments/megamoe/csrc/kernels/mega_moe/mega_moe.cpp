// Copyright (c) 2026, Lu Lu
// MegaMoE direct-call entries for the shared DeepEP kernel library.
// Keep the wave implementations and their tiling layout independent of the launcher.
// The quantized and BF16 training entries share one compile configuration; the
// training kernels align to the quantized flag regime.
// Modified by SimpleBright_Man 2026
// Modified by xixing 2026
// Modified by guofanfeng 2026

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_launch.hpp"
#include "mega_moe_quantized_tiling.hpp"
#include "common/mega_moe_constants.hpp"
#if defined(__NPU_DEVICE__)
#include "mega_moe_wave_a8w8.hpp"
#include "mega_moe_wave_a8w4.hpp"
#endif

using namespace AscendC;

// Explicit types replace asc_opc dtype/format/tiling-key specialisation.
// A8W4's existing matmul prologue continues to interpret weights as NZ/C0_32.
// Batch scheduling preserves the old ACL schemMode=1 cross-core contract.
template <bool IsA8W4, bool FuseGateUp = false>
__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_kernel(GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids,
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
    if constexpr (IsA8W4) {
        MegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, fp4x2_e2m1_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, FuseGateUp>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &tiling, nullptr, timer);
        op.Process();
    } else {
        MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, kDispatchE4M3,
                                     MegaMoeImpl::COMBINE_NO_QUANT, false, false>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &tiling, nullptr);
        op.Process();
    }
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
    // 按本次实际 BS 编译期特化；不改变 wave 预算、Dispatch/Combine 或共享专家策略。
    if (is_a8w4 && tiling.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        mega_moe_kernel<true, true><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    } else if (is_a8w4) {
        mega_moe_kernel<true, false><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    } else {
        mega_moe_kernel<false><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr);
    }
}

/*
 * BF16 training forward and backward entries.  The implementation uses
 * DeepEP's routing/dispatch/combine pipeline and only references CATLASS
 * primitives for BF16 MMAD; it does not call the CATCCOS DispatchFFNCombine
 * operator.
 */
#include "kernel_operator.h"
#include "mega_moe_bf16_tiling.hpp"
#include "mega_moe_wave_bf16.hpp"
#include "mega_moe_bf16_backward.hpp"
#include "kernel_launch.hpp"

using namespace AscendC;
using namespace MegaMoeImpl;

template <typename TopkWeightT>
__mix__(1, 2) __global__ __aicore__
    void mega_moe_bf16_forward(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                               GM_ADDR weight2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR trainingSavedStateGM,
                               GM_ADDR dispatchOutputGM, GM_ADDR forwardTransientWorkspaceGM, GM_ADDR timerAddr,
                               MegaMoeBf16TilingData tilingData)
{
    InitSocState();
    MegaMoeTrainTimerDevice timer;
    timer.Init(timerAddr);
    timer.Start(MegaMoeTrainTimer::KERNEL_TIMING);
    MegaMoeBF16Forward<TopkWeightT> op;
    op.Init(context, x, topkIds, topkWeights, weight1, weight2, nullptr, nullptr, nullptr, nullptr, yOut,
            expertTokenNumsOut, nullptr, nullptr, nullptr, trainingSavedStateGM, &tilingData,
            forwardTransientWorkspaceGM, dispatchOutputGM);
    op.InitForwardWorkspace(forwardTransientWorkspaceGM);
    op.SetTimer(timer.Enabled() ? &timer : nullptr);
    op.Process();
    if (timer.Enabled()) AscendC::PipeBarrier<PIPE_ALL>();
    timer.End(MegaMoeTrainTimer::KERNEL_TIMING);
    timer.FlushDynamicCounts();
}

extern "C" __attribute__((visibility("default"))) void mega_moe_bf16_forward_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x, uint8_t* topk_ids, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* training_saved_state,
    uint8_t* dispatch_output, uint8_t* forward_transient_workspace, const uint8_t* tiling_host, uint64_t ffts_addr)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeBf16TilingData*>(tiling_host);
    if (tiling.topkWeightType == 1U) {
        mega_moe_bf16_forward<float><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, nullptr, tiling);
    } else {
        mega_moe_bf16_forward<bfloat16_t><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, nullptr, tiling);
    }
}

extern "C" __attribute__((visibility("default"))) void mega_moe_bf16_forward_timer_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x, uint8_t* topk_ids, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* training_saved_state,
    uint8_t* dispatch_output, uint8_t* forward_transient_workspace, const uint8_t* tiling_host, uint64_t ffts_addr,
    uint8_t* timer)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeBf16TilingData*>(tiling_host);
    if (tiling.topkWeightType == 1U) {
        mega_moe_bf16_forward<float><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, timer, tiling);
    } else {
        mega_moe_bf16_forward<bfloat16_t><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, timer, tiling);
    }
}

// Retain the exported capability queries: both schedules stage raw Gate/Up in GM.
extern "C" __attribute__((visibility("default"))) int mega_moe_bf16_swiglu_decoupled(uint64_t hidden)
{
    (void)hidden;
    return 1;
}
extern "C" __attribute__((visibility("default"))) int mega_moe_bf16_swiglu_decoupled_for_saved_mode(uint64_t hidden,
                                                                                                    uint32_t savedMode)
{
    (void)hidden;
    (void)savedMode;
    return 1;
}

template <typename TopkWeightT>
__mix__(1, 2) __global__ __aicore__
    void mega_moe_bf16_backward(GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR gradX, GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
                                GM_ADDR trainingSavedState, GM_ADDR dispatchOutput, GM_ADDR backwardWorkspace,
                                GM_ADDR timerAddr, MegaMoeBf16TilingData tilingData)
{
    InitSocState();
    MegaMoeTrainTimerDevice timer;
    timer.Init(timerAddr);
    timer.Start(MegaMoeTrainTimer::KERNEL_TIMING);
    MegaMoeBF16Backward<TopkWeightT> op;
    op.Init(context, gradY, topkWeights, weight1, weight2, gradX, gradTopkWeights, gradWeight1, gradWeight2,
            trainingSavedState, dispatchOutput, backwardWorkspace, &tilingData, tilingData.backwardSplitExperts);
    op.SetTimer(timer.Enabled() ? &timer : nullptr);
    op.Process();
    if (timer.Enabled()) AscendC::PipeBarrier<PIPE_ALL>();
    timer.End(MegaMoeTrainTimer::KERNEL_TIMING);
    timer.FlushDynamicCounts();
}

extern "C" __attribute__((visibility("default"))) void mega_moe_bf16_backward_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights, uint8_t* weight1,
    uint8_t* weight2, uint8_t* grad_x, uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output, uint8_t* backward_workspace, const uint8_t* tiling_host,
    uint64_t ffts_addr, uint32_t split_experts)
{
    (void)ffts_addr;
    auto tiling = *reinterpret_cast<const MegaMoeBf16TilingData*>(tiling_host);
    tiling.backwardSplitExperts = split_experts;
    if (tiling.topkWeightType == 1U) {
        mega_moe_bf16_backward<float><<<block_dim, nullptr, stream>>>(
            context, grad_y, topk_weights, weight1, weight2, grad_x, grad_topk_weights, grad_weight1, grad_weight2,
            training_saved_state, dispatch_output, backward_workspace, nullptr, tiling);
    } else {
        mega_moe_bf16_backward<bfloat16_t><<<block_dim, nullptr, stream>>>(
            context, grad_y, topk_weights, weight1, weight2, grad_x, grad_topk_weights, grad_weight1, grad_weight2,
            training_saved_state, dispatch_output, backward_workspace, nullptr, tiling);
    }
}

extern "C" __attribute__((visibility("default"))) void mega_moe_bf16_backward_timer_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights, uint8_t* weight1,
    uint8_t* weight2, uint8_t* grad_x, uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output, uint8_t* backward_workspace, const uint8_t* tiling_host,
    uint64_t ffts_addr, uint32_t split_experts, uint8_t* timer)
{
    (void)ffts_addr;
    auto tiling = *reinterpret_cast<const MegaMoeBf16TilingData*>(tiling_host);
    tiling.backwardSplitExperts = split_experts;
    if (tiling.topkWeightType == 1U) {
        mega_moe_bf16_backward<float><<<block_dim, nullptr, stream>>>(
            context, grad_y, topk_weights, weight1, weight2, grad_x, grad_topk_weights, grad_weight1, grad_weight2,
            training_saved_state, dispatch_output, backward_workspace, timer, tiling);
    } else {
        mega_moe_bf16_backward<bfloat16_t><<<block_dim, nullptr, stream>>>(
            context, grad_y, topk_weights, weight1, weight2, grad_x, grad_topk_weights, grad_weight1, grad_weight2,
            training_saved_state, dispatch_output, backward_workspace, timer, tiling);
    }
}
