// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstdint>

#if defined(__GNUC__)
#define ASCEND_MEGAMOE_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_MEGAMOE_EXPORT
#endif

extern "C" {

ASCEND_MEGAMOE_EXPORT int deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(uint64_t hidden,
                                                                                     uint32_t saved_mode);

ASCEND_MEGAMOE_EXPORT void deepep_train_mega_moe_bf16_forward_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x, uint8_t* topk_ids, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* training_saved_state,
    uint8_t* dispatch_output, uint8_t* forward_transient_workspace, const uint8_t* tiling_host, uint64_t ffts_addr);

ASCEND_MEGAMOE_EXPORT void deepep_train_mega_moe_bf16_backward_ready_k_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights, uint8_t* weight1,
    uint8_t* weight2, uint8_t* grad_x, uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output, uint8_t* backward_workspace, const uint8_t* tiling_host,
    uint64_t ffts_addr, uint32_t split_experts);

}  // extern "C"
