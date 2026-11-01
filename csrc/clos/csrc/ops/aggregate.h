#pragma once

#include <tuple>
#include <functional>

#include <torch/extension.h>

namespace ascend_deepep::ops {

// Preparation validates and retains inputs, but does not launch a kernel.
// Expanded combine uses this to submit independent single rows first.
struct AggregatePlan {
    torch::Tensor x;
    c10::optional<torch::Tensor> values;
    std::function<void()> launch;
};

AggregatePlan PrepareAggregate(torch::Tensor x, c10::optional<torch::Tensor> topk_weights,
                               c10::optional<torch::Tensor> expanded_values, torch::Tensor recv_to_expanded_idx,
                               int64_t num_aiv, uint64_t ffts_addr, c10::optional<int64_t> allocation_stream_id,
                               c10::optional<torch::Tensor> single_source_slots,
                               c10::optional<torch::Tensor> rank_prefix_matrix, int rank, int world_size);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Aggregate(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> expanded_values,
    torch::Tensor recv_to_expanded_idx, int64_t num_aiv, uint64_t ffts_addr,
    c10::optional<int64_t> allocation_stream_id = c10::nullopt,
    c10::optional<torch::Tensor> single_source_slots = c10::nullopt,
    c10::optional<torch::Tensor> rank_prefix_matrix = c10::nullopt, int rank = 0, int world_size = 0);

}  // namespace ascend_deepep::ops
