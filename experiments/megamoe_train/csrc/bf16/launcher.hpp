// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstddef>
#include <cstdint>

#include <torch/extension.h>

struct MegaMoeTilingData;

void LaunchMegaMoeBF16ForwardKernel(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                    torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                    torch::Tensor y, torch::Tensor expert_token_nums,
                                    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
                                    torch::Tensor forward_transient_workspace, const MegaMoeTilingData& tiling,
                                    uint32_t block_dim, void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
                                    int rank, int world_size, uint64_t ffts_addr);

bool MegaMoeBF16SwigluDecoupled(uint64_t hidden, uint32_t saved_mode);

void InitializeMegaMoeBF16Context(torch::Tensor context, void* symmetric_buffer, int rank, int world_size);

void LaunchMegaMoeBF16BackwardKernel(torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
                                     torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
                                     torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
                                     torch::Tensor grad_weight2, torch::Tensor training_saved_state,
                                     torch::Tensor dispatch_output, torch::Tensor backward_workspace,
                                     const MegaMoeTilingData& tiling, uint32_t splitExperts, uint32_t block_dim,
                                     void* symmetric_buffer, std::size_t symmetric_buffer_bytes, int rank,
                                     int world_size, uint64_t ffts_addr);
