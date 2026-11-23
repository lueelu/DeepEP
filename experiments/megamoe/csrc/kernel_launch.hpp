// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once
#include <cstdint>

// Host 和 Bisheng 共享唯一入口声明；FP8 scale 非空时绕过输入量化。
extern "C" void mega_moe_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context,
                                   uint8_t* x, uint8_t* x_scales, uint8_t* topk_ids, uint8_t* topk_weights,
                                   uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1, uint8_t* weight_scales2,
                                   uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
                                   uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums,
                                   uint8_t* workspace, uint8_t* timer, const uint8_t* tiling_host, uint64_t ffts_addr);

extern "C" int mega_moe_bf16_swiglu_decoupled_for_saved_mode(uint64_t hidden, uint32_t saved_mode);

extern "C" void mega_moe_bf16_forward_kernel_do(uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x,
                                                uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1,
                                                uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums,
                                                uint8_t* training_saved_state, uint8_t* dispatch_output,
                                                uint8_t* forward_transient_workspace, const uint8_t* tiling_host,
                                                uint64_t ffts_addr);

extern "C" void mega_moe_bf16_backward_kernel_do(uint32_t block_dim, void* stream, uint8_t* context, uint8_t* grad_y,
                                                 uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2,
                                                 uint8_t* grad_x, uint8_t* grad_topk_weights, uint8_t* grad_weight1,
                                                 uint8_t* grad_weight2, uint8_t* training_saved_state,
                                                 uint8_t* dispatch_output, uint8_t* backward_workspace,
                                                 const uint8_t* tiling_host, uint64_t ffts_addr,
                                                 uint32_t split_experts);

extern "C" void mega_moe_bf16_forward_timer_kernel_do(uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x,
                                                      uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1,
                                                      uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums,
                                                      uint8_t* training_saved_state, uint8_t* dispatch_output,
                                                      uint8_t* forward_transient_workspace, const uint8_t* tiling_host,
                                                      uint64_t ffts_addr, uint8_t* timer);

extern "C" void mega_moe_bf16_backward_timer_kernel_do(uint32_t block_dim, void* stream, uint8_t* context,
                                                       uint8_t* grad_y, uint8_t* topk_weights, uint8_t* weight1,
                                                       uint8_t* weight2, uint8_t* grad_x, uint8_t* grad_topk_weights,
                                                       uint8_t* grad_weight1, uint8_t* grad_weight2,
                                                       uint8_t* training_saved_state, uint8_t* dispatch_output,
                                                       uint8_t* backward_workspace, const uint8_t* tiling_host,
                                                       uint64_t ffts_addr, uint32_t split_experts, uint8_t* timer);
