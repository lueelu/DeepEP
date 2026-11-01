#pragma once

#include <cstdint>

#include <torch/extension.h>

#include "ascend_deepep/kernels/types.h"

void LaunchDispatchLayoutKernel(torch::Tensor topk_idx, torch::Tensor num_tokens_per_rank,
                                torch::Tensor num_tokens_per_expert, torch::Tensor is_token_in_rank,
                                const ascend_deepep::DispatchLayoutTilingData& tiling, torch::Tensor scratch_buffer,
                                uint64_t ffts_addr);

void LaunchNotifyDispatchKernel(torch::Tensor num_tokens_per_rank, torch::Tensor num_tokens_per_expert,
                                torch::Tensor is_token_in_rank, torch::Tensor rank_prefix_matrix,
                                torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix,
                                torch::Tensor num_recv_tokens, torch::Tensor num_recv_tokens_per_expert,
                                const c10::optional<torch::Tensor>& dispatch_plan, torch::Tensor send_head,
                                const ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer,
                                uint64_t ffts_addr);

void LaunchCachedNotifyDispatchKernel(const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr);

void LaunchCachedDispatchPlanKernel(torch::Tensor routes, torch::Tensor plan, torch::Tensor send_head,
                                    const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr);

void LaunchV1DispatchKernel(
    torch::Tensor x, const c10::optional<torch::Tensor>& x_scales, const c10::optional<torch::Tensor>& topk_idx,
    const c10::optional<torch::Tensor>& topk_weights, const c10::optional<torch::Tensor>& num_tokens_per_rank,
    torch::Tensor is_token_in_rank, const c10::optional<torch::Tensor>& num_tokens_per_expert, torch::Tensor recv_x,
    const c10::optional<torch::Tensor>& recv_x_scales, const c10::optional<torch::Tensor>& recv_topk_idx,
    const c10::optional<torch::Tensor>& recv_topk_weights, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix, torch::Tensor recv_src_idx,
    torch::Tensor send_head, const c10::optional<torch::Tensor>& dispatch_plan,
    const ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr, bool use_udma);

enum class CombinePhase { Fused, Communication, Reduction, EarlySingles, Remaining, Pipeline };

void LaunchV1CombineKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& bias_0, const c10::optional<torch::Tensor>& bias_1,
                           torch::Tensor src_idx, torch::Tensor rank_prefix_matrix, torch::Tensor channel_prefix_matrix,
                           torch::Tensor send_head, torch::Tensor recv_x,
                           const c10::optional<torch::Tensor>& recv_topk_weights,
                           const ascend_deepep::CombineTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr,
                           bool use_udma, CombinePhase phase);

void LaunchExpandKernel(torch::Tensor x, const c10::optional<torch::Tensor>& x_scales,
                        const c10::optional<torch::Tensor>& topk_idx, const c10::optional<torch::Tensor>& topk_weights,
                        torch::Tensor expanded_x, const c10::optional<torch::Tensor>& expanded_x_scales,
                        const c10::optional<torch::Tensor>& expanded_topk_weights, torch::Tensor expert_offsets,
                        torch::Tensor actual_expert_counts, torch::Tensor recv_to_expanded_idx,
                        torch::Tensor scratch_buffer, const ascend_deepep::ExpandTilingData& tiling,
                        uint64_t ffts_addr);

void LaunchAggregateKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& expanded_values, torch::Tensor recv_to_expanded_idx,
                           torch::Tensor aggregated_x, const c10::optional<torch::Tensor>& aggregated_values,
                           const ascend_deepep::AggregateTilingData& tiling, uint64_t ffts_addr);

void LaunchElasticBarrierKernel(int64_t block_dim, uint64_t ffts_addr);

void LaunchDupServerDispatchScaleOutKernel(torch::Tensor x, torch::Tensor send_token_count,
                                           torch::Tensor send_token_ids, torch::Tensor send_gateway_lane,
                                           void* symmetric_buffer,
                                           const ascend_deepep::DupServerScaleOutTilingData& tiling,
                                           uint64_t ffts_addr);

void LaunchDupServerDispatchKernel(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                   torch::Tensor send_gateway_lane, torch::Tensor recv_route_position,
                                   torch::Tensor recv_gateway_lane, torch::Tensor recv_src_rank,
                                   torch::Tensor recv_segment_offset, torch::Tensor recv_x, void* symmetric_buffer,
                                   const ascend_deepep::DupServerScaleOutTilingData& tiling, uint64_t ffts_addr);

void LaunchDupServerNotifyScaleOutKernel(void* symmetric_buffer,
                                         const ascend_deepep::DupServerScaleOutTilingData& tiling, uint64_t ffts_addr);

void LaunchDupServerNotifyPrepareKernel(
    torch::Tensor topk_idx, const c10::optional<torch::Tensor>& topk_weights, torch::Tensor send_token_count,
    torch::Tensor send_token_ids, torch::Tensor send_gateway_lane, torch::Tensor send_route_count,
    torch::Tensor send_expert_prefix, torch::Tensor packed_route_position, torch::Tensor packed_gateway_lane,
    torch::Tensor source_server_mask, const c10::optional<torch::Tensor>& packed_topk_weights,
    torch::Tensor recv_route_position, torch::Tensor recv_gateway_lane, torch::Tensor recv_segment_offset,
    torch::Tensor recv_weight_segment_begin, torch::Tensor recv_chunk_token_end, torch::Tensor gateway_route_count,
    torch::Tensor gateway_slot_mask, torch::Tensor notify_status, torch::Tensor scratch_buffer, void* symmetric_buffer,
    const ascend_deepep::DupServerNotifyPrepareTilingData& tiling, uint64_t ffts_addr);
