#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
           torch::Tensor>
Expand(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
       c10::optional<torch::Tensor> topk_weights, std::vector<int64_t> num_tokens_per_expert, int64_t expert_alignment,
       int64_t cached_num_expanded_tokens, c10::optional<torch::Tensor> cached_expert_offsets,
       c10::optional<torch::Tensor> cached_actual_expert_counts,
       c10::optional<torch::Tensor> cached_recv_to_expanded_idx, int64_t num_aiv, uint64_t ffts_addr,
       c10::optional<int64_t> allocation_stream_id = c10::nullopt,
       c10::optional<torch::Tensor> public_expert_prefix = c10::nullopt,
       c10::optional<torch::Tensor> public_src_metadata = c10::nullopt,
       c10::optional<torch::Tensor> single_source_slots = c10::nullopt);

}  // namespace ascend_deepep::ops
