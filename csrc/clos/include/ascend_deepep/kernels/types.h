#pragma once

#include <cstdint>

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

namespace ascend_deepep {

inline constexpr uint32_t kDupServerRoutePositionTileElements = 32U;
inline constexpr uint32_t kDupServerSourceLaneRouteTileElements = 256U;
inline constexpr uint32_t kDupServerReceiverTraceFields = 5U;
inline constexpr uint32_t kDupServerReceiverTraceRecordBytes =
    kDupServerReceiverTraceFields * sizeof(uint64_t);

template <int kNumBits>
struct int_with_bits;

template <>
struct int_with_bits<8> {
    using type = int8_t;
};

template <>
struct int_with_bits<16> {
    using type = int16_t;
};

template <>
struct int_with_bits<32> {
    using type = int32_t;
};

template <>
struct int_with_bits<64> {
    using type = int64_t;
};

using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;

struct alignas(8) DispatchLayoutTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t block_dim;
    uint32_t rank_stride;
    uint32_t expert_stride;
};

struct alignas(8) DispatchTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t expert_alignment;
    uint32_t block_dim;
    uint32_t has_topk_weights;
    uint32_t has_x_scales;
    uint32_t cached_mode;
    uint32_t num_channels;
    uint32_t rank_stride;
    uint32_t expert_stride;
    uint32_t channel_stride;
    uint32_t num_local_experts;
    uint32_t x_element_bytes;
    uint32_t scale_element_bytes;
    uint32_t num_scales;
    uint32_t num_output_tokens;
    uint32_t needs_notify_entry_barrier;
    uint32_t needs_x_scales_pack;
    uint32_t reserved;
    uint64_t scale_token_stride;
    uint64_t scale_hidden_stride;
    uint64_t notify_workspace_offset;
    uint64_t recv_channel_prefix_offset;
    uint64_t x_buffer_offset;
    uint64_t src_idx_buffer_offset;
    uint64_t topk_idx_buffer_offset;
    uint64_t topk_weights_buffer_offset;
    uint64_t x_scales_buffer_offset;
    uint64_t x_scales_input_buffer_offset;
    uint64_t udma_completion_signal_offset;
    uint64_t dispatch_workspace_bytes;
    uint32_t udma_peer_order;
    uint32_t udma_ranks_per_group;
    uint32_t udma_peers_per_worker;
    uint32_t udma_wqes_per_visit;
    uint32_t udma_token_schedule;
    uint32_t udma_workers_per_peer = 1;
    uint32_t udma_qps_per_worker = 1;
    // Fresh V2 handle outputs; zero for V1 and cached dispatch.
    uint64_t v2_rank_prefix;
    uint64_t v2_expert_prefix;
    uint64_t v2_src_metadata;
    uint64_t v2_dst_slots;
    // Buffer-owned symmetric address mailbox; never part of a cached handle.
    uint64_t udma_output_control;
};

struct alignas(8) CombineTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_recv_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t block_dim;
    uint32_t num_channels;
    uint32_t has_topk_weights;
    uint32_t has_bias_0;
    uint32_t has_bias_1;
    uint32_t x_dtype;
    uint32_t x_element_bytes;
    uint32_t token_capacity;
    uint32_t x_row_stride;
    uint32_t topk_weights_row_stride;
    uint64_t x_buffer_offset;
    uint64_t topk_weights_buffer_offset;
    uint64_t udma_completion_signal_offset;
    uint64_t combine_workspace_bytes;
    uint32_t udma_peer_order;
    uint32_t udma_ranks_per_group;
    uint32_t udma_peers_per_worker;
    uint32_t udma_wqes_per_visit;
    uint32_t udma_token_schedule;
    uint32_t num_expanded_tokens;
    uint64_t expanded_x;
    uint64_t single_source_slots;
    uint64_t pipeline_mapping;
    uint64_t pipeline_values;
    uint64_t pipeline_ready;
    uint32_t pipeline_chunk_tokens;
    uint32_t pipeline_topk;
    uint32_t udma_workers_per_peer = 1;
    uint32_t udma_qps_per_worker = 1;
};

struct alignas(8) ExpandTilingData {
    uint32_t num_recv_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t num_local_experts;
    uint32_t num_expanded_tokens;
    uint32_t expert_alignment;
    uint32_t block_dim;
    uint32_t cached_mode;
    uint32_t has_topk_weights;
    uint32_t has_x_scales;
    uint32_t x_element_bytes;
    uint32_t scale_element_bytes;
    uint32_t num_scales;
    uint64_t v2_expert_prefix;
    uint64_t v2_src_metadata;
    uint64_t single_source_slots;
};

struct alignas(8) AggregateTilingData {
    uint32_t num_recv_tokens;
    uint32_t num_expanded_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t block_dim;
    uint32_t has_topk_weights;
    uint32_t has_expanded_values;
    uint32_t x_dtype;
    uint32_t x_element_bytes;
    uint32_t rank;
    uint32_t world_size;
    uint64_t single_source_slots;
    uint64_t rank_prefix_matrix;
};

struct alignas(8) DupServerScaleOutTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t ranks_per_server;
    uint32_t num_servers;
    uint32_t num_tokens;
    uint32_t token_capacity;
    uint32_t hidden;
    uint32_t hidden_bytes;
    uint32_t block_dim;
    uint32_t max_inflight_same_frame_servers;
    uint32_t num_recv_tokens;
    uint32_t num_topk;
    uint32_t num_topk_shift;
    uint32_t num_local_experts;
    uint32_t num_tokens_per_chunk;
    uint32_t num_token_chunks;
    uint32_t chunk_capacity;
    uint32_t copy_tile_bytes;
    uint32_t use_urma_get;
    uint32_t enable_timer;
    uint32_t gateway_policy;
    uint32_t remote_get_servers;
    uint64_t x_buffer_offset;
    uint64_t completion_signal_offset;
    uint64_t producer_signal_offset;
    uint64_t receiver_trace_offset;
    uint64_t workspace_bytes;
    uint32_t remote_get_rx_tx_gate = 0U;
};

struct alignas(8) DupServerNotifyPrepareTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t ranks_per_server;
    uint32_t num_servers;
    uint32_t num_tokens;
    uint32_t token_capacity;
    uint32_t num_topk;
    uint32_t num_topk_shift;
    uint32_t num_experts;
    uint32_t experts_per_server;
    uint32_t experts_per_server_shift;
    uint32_t send_route_capacity;
    uint32_t recv_route_capacity;
    uint32_t block_dim;
    uint32_t expert_stride;
    uint32_t server_stride;
    uint32_t scratch_stride;
    uint32_t max_tokens_per_core;
    uint32_t pack_workers;
    uint32_t inputs_in_ub;
    uint32_t has_topk_weights;
    uint32_t enable_timer;
    uint32_t gateway_policy;
    uint32_t num_tokens_per_chunk;
    uint32_t num_token_chunks;
    uint32_t chunk_capacity;
    uint32_t num_local_experts;
    uint32_t notify_producer_workers;
    uint32_t notify_gateway_workers;
    uint32_t gateway_prefix_stride;
    uint32_t gateway_route_stride;
    uint32_t gateway_route_lane_stride;
    uint32_t rank_prefix_stride;
    uint32_t rank_route_stride;
    uint32_t rank_route_lane_stride;
    uint64_t gateway_prefix_offset;
    uint64_t gateway_route_offset;
    uint64_t gateway_route_lane_offset;
    uint64_t gateway_ready_offset;
    uint64_t rank_prefix_offset;
    uint64_t rank_route_offset;
    uint64_t rank_route_lane_offset;
    uint64_t rank_ready_offset;
    uint64_t workspace_bytes;
};

}  // namespace ascend_deepep
