#pragma once

#include <cstdint>

#if defined(__GNUC__)
#define ASCEND_DEEPEP_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_DEEPEP_EXPORT
#endif

// Ascend950 accepts the dynamic UB byte count as the second launch argument.
// On dav-c220 (Ascend 910C), that argument is a stream-mode descriptor.
#if CATLASS_ARCH == 3510
#define ASCEND_DEEPEP_KERNEL_LAUNCH_UB(bytes) (bytes)
#elif CATLASS_ARCH == 2201
#define ASCEND_DEEPEP_KERNEL_LAUNCH_UB(bytes) nullptr
#else
#error "Unsupported CATLASS_ARCH for ascend-deepep kernel launch"
#endif

extern "C" {

ASCEND_DEEPEP_EXPORT void dispatch_layout_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* topk_idx,
    uint8_t* scratch_buffer,
    uint8_t* num_tokens_per_rank,
    uint8_t* num_tokens_per_expert,
    uint8_t* is_token_in_rank,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

// A distinct symbol prevents the old Notify ABI from accepting the new output argument.
ASCEND_DEEPEP_EXPORT void notify_dispatch_plan_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* num_tokens_per_rank,
    uint8_t* num_tokens_per_expert,
    uint8_t* is_token_in_rank,
    uint8_t* symmetric_buffer,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* recv_channel_prefix_matrix,
    uint8_t* num_recv_tokens,
    uint8_t* num_recv_tokens_per_expert,
    uint8_t* dispatch_plan,
    uint8_t* send_head,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void cached_dispatch_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* routes, uint8_t* plan,
    uint8_t* head, uint8_t* tiling_host, uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void cached_notify_dispatch_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_dispatch_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* x_scales,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank,
    uint8_t* is_token_in_rank,
    uint8_t* num_tokens_per_expert,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_x_scales,
    uint8_t* recv_topk_idx,
    uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx,
    uint8_t* send_head,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_dispatch_udma_plan_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* x_scales,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank,
    uint8_t* is_token_in_rank,
    uint8_t* num_tokens_per_expert,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_x_scales,
    uint8_t* recv_topk_idx,
    uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx,
    uint8_t* send_head,
    uint8_t* dispatch_plan,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void cached_notify_combine_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_comm_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_reduce_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_comm_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_early_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_pipeline_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_remaining_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void v1_combine_udma_reduce_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* bias_0,
    uint8_t* bias_1,
    uint8_t* src_idx,
    uint8_t* rank_prefix_matrix,
    uint8_t* channel_prefix_matrix,
    uint8_t* send_head,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_weights,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void expand_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* x_scales,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* expanded_x,
    uint8_t* expanded_x_scales,
    uint8_t* expanded_topk_weights,
    uint8_t* expert_offsets,
    uint8_t* actual_expert_counts,
    uint8_t* recv_to_expanded_idx,
    uint8_t* scratch_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void aggregate_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_weights,
    uint8_t* expanded_values,
    uint8_t* recv_to_expanded_idx,
    uint8_t* aggregated_x,
    uint8_t* aggregated_values,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void dup_server_dispatch_scaleout_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* send_token_count,
    uint8_t* send_token_ids,
    uint8_t* send_gateway_lane,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void dup_server_dispatch_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* send_token_count,
    uint8_t* send_token_ids,
    uint8_t* send_gateway_lane,
    uint8_t* recv_route_position,
    uint8_t* recv_gateway_lane,
    uint8_t* recv_src_rank,
    uint8_t* recv_segment_offset,
    uint8_t* recv_x,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void dup_server_notify_scaleout_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void dup_server_notify_prepare_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* send_token_count,
    uint8_t* send_token_ids,
    uint8_t* send_gateway_lane,
    uint8_t* send_route_count,
    uint8_t* send_expert_prefix,
    uint8_t* packed_route_position,
    uint8_t* packed_gateway_lane,
    uint8_t* source_server_mask,
    uint8_t* packed_topk_weights,
    uint8_t* recv_route_position,
    uint8_t* recv_gateway_lane,
    uint8_t* recv_segment_offset,
    uint8_t* recv_weight_segment_begin,
    uint8_t* recv_chunk_token_end,
    uint8_t* gateway_route_count,
    uint8_t* gateway_slot_mask,
    uint8_t* notify_status,
    uint8_t* scratch_buffer,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

}  // extern "C"
