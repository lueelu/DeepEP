#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/extension.h>

class ShmemInstance;

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor> GetDispatchLayout(
    torch::Tensor topk_idx, int64_t num_experts, int rank, int world_size, int num_device_aiv, uint64_t ffts_addr,
    c10::optional<int64_t> allocation_stream_id = c10::nullopt);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
           std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
V1Dispatch(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
           c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
           torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert,
           int64_t cached_num_recv_tokens, c10::optional<torch::Tensor> cached_rank_prefix_matrix,
           c10::optional<torch::Tensor> cached_channel_prefix_matrix, int64_t expert_alignment,
           int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
           int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy, int rank, int world_size,
           std::size_t symmetric_buffer_bytes, std::size_t& notify_workspace_bytes, bool symmetric_workspace_active,
           bool use_udma, ShmemInstance& shmem, void* symmetric_buffer = nullptr,
           c10::optional<int64_t> allocation_stream_id = c10::nullopt,
           c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt, int64_t udma_peers_per_worker = 4,
           int64_t udma_wqes_per_visit = 8, c10::optional<torch::Tensor> cached_dispatch_plan = c10::nullopt,
           std::vector<torch::Tensor>* public_handle = nullptr, bool defer_public_expert_prefix = false,
           void* dispatch_output_control = nullptr, const std::vector<int64_t>& launch_plan = {});

}  // namespace ascend_deepep::ops
