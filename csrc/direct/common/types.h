#pragma once

#include <cstdint>

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

namespace ascend_deepep {

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
    // Reserved symmetric scale staging, owned until dispatch completion.
    uint64_t packed_peer_scales = 0;
    // Cache-isolated local producer flags in the unused self-peer interval.
    uint64_t packed_scale_ready = 0;
    // Optional pinned-host count+1 slots. Never used by cached/graph dispatch.
    uint64_t notify_host_counts = 0;
};

struct alignas(8) CombineTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_recv_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t block_dim;     // Actual AIVs for this communication/reduction launch.
    uint32_t num_channels;  // Logical channels retained from the dispatch handle.
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
    uint32_t udma_self_copy_first_aiv = 0;
    uint32_t udma_self_copy_after_submit = 0;
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

}  // namespace ascend_deepep
