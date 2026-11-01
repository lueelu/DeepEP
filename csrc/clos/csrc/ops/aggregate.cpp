#include "csrc/ops/aggregate.h"

#include <cstdint>
#include <limits>

#include "ascend_deepep/kernels/types.h"
#include "ascend_deepep/platform.h"
#include "csrc/launcher.h"
#include "csrc/ops/v1_allocator.h"

namespace {

constexpr int64_t kMaxAggregateTopk = 128;
constexpr uint32_t kAggregateDtypeBfloat16 = 0;
constexpr uint32_t kAggregateDtypeFloat16 = 1;

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalNpuContiguous(const c10::optional<torch::Tensor>& tensor, const char* name)
{
    if (tensor.has_value()) {
        CheckNpuContiguous(*tensor, name);
    }
}

int64_t ResolveNumAiv(int64_t num_aiv)
{
    TORCH_CHECK(num_aiv >= 0, "num_aiv must be nonnegative");
    const int64_t resolved = num_aiv == 0 ? 32 : num_aiv;
    TORCH_CHECK(resolved % 2 == 0, "num_aiv must be even");
    TORCH_CHECK(resolved <= ascend_deepep::platform::kMaxNumAiv, "num_aiv must not exceed ",
                ascend_deepep::platform::kMaxNumAiv);
    return resolved;
}

}  // namespace

namespace ascend_deepep::ops {

AggregatePlan PrepareAggregate(torch::Tensor x, c10::optional<torch::Tensor> topk_weights,
                               c10::optional<torch::Tensor> expanded_values, torch::Tensor recv_to_expanded_idx,
                               int64_t num_aiv, uint64_t ffts_addr, c10::optional<int64_t> allocation_stream_id,
                               c10::optional<torch::Tensor> single_source_slots,
                               c10::optional<torch::Tensor> rank_prefix_matrix, int rank, int world_size)
{
    V1Allocator allocator(allocation_stream_id);
    allocator.Record({x, topk_weights, expanded_values, recv_to_expanded_idx, single_source_slots, rank_prefix_matrix});
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(recv_to_expanded_idx, "recv_to_expanded_idx");
    CheckOptionalNpuContiguous(topk_weights, "topk_weights");
    CheckOptionalNpuContiguous(expanded_values, "expanded_values");

    TORCH_CHECK(x.dim() == 2, "x must have shape [num_expanded_tokens, hidden]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16,
                "aggregate x must be bfloat16 or float16");
    TORCH_CHECK(x.size(1) > 0, "x hidden dimension must be positive");
    TORCH_CHECK(recv_to_expanded_idx.dim() == 2 && recv_to_expanded_idx.scalar_type() == torch::kInt32,
                "recv_to_expanded_idx must be a two-dimensional int32 tensor");
    const int64_t num_recv_tokens = recv_to_expanded_idx.size(0);
    const int64_t num_topk = recv_to_expanded_idx.size(1);
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(num_topk <= kMaxAggregateTopk, "num_topk must not exceed ", kMaxAggregateTopk);

    if (topk_weights.has_value()) {
        TORCH_CHECK(topk_weights->dim() == 1 && topk_weights->size(0) == x.size(0),
                    "topk_weights must have shape [num_expanded_tokens]");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
    }
    if (expanded_values.has_value()) {
        TORCH_CHECK(expanded_values->dim() == 1 && expanded_values->size(0) == x.size(0),
                    "expanded_values must have shape [num_expanded_tokens]");
        TORCH_CHECK(expanded_values->scalar_type() == torch::kFloat32, "expanded_values must be float32");
    }

    const int64_t resolved_num_aiv = ResolveNumAiv(num_aiv);
    TORCH_CHECK(
        num_recv_tokens <= std::numeric_limits<uint32_t>::max() && x.size(0) <= std::numeric_limits<uint32_t>::max() &&
            x.size(1) <= std::numeric_limits<uint32_t>::max() && num_topk <= std::numeric_limits<uint32_t>::max(),
        "aggregate dimensions must fit uint32");

    auto aggregated_x = allocator.Empty({num_recv_tokens, x.size(1)}, x.options());
    c10::optional<torch::Tensor> aggregated_values = c10::nullopt;
    if (expanded_values.has_value()) {
        aggregated_values = allocator.Empty({num_recv_tokens, num_topk}, expanded_values->options());
    }

    TORCH_CHECK(single_source_slots.has_value() == rank_prefix_matrix.has_value(),
                "single-source aggregate requires both slots and rank prefix");
    ascend_deepep::AggregateTilingData tiling{};
    if (single_source_slots.has_value()) {
        TORCH_CHECK(!topk_weights.has_value(), "single-source mode cannot multiply route weights");
        CheckNpuContiguous(*single_source_slots, "single_source_slots");
        CheckNpuContiguous(*rank_prefix_matrix, "rank_prefix_matrix");
        TORCH_CHECK(single_source_slots->device() == x.device() && rank_prefix_matrix->device() == x.device(),
                    "single-source metadata must be on x's device");
        TORCH_CHECK(single_source_slots->scalar_type() == torch::kInt32 && single_source_slots->dim() == 1 &&
                        single_source_slots->size(0) == num_recv_tokens,
                    "invalid single-source slots");
        TORCH_CHECK(world_size > 0 && rank >= 0 && rank < world_size &&
                        rank_prefix_matrix->scalar_type() == torch::kInt32 && rank_prefix_matrix->dim() == 2 &&
                        rank_prefix_matrix->size(0) == world_size && rank_prefix_matrix->size(1) == world_size,
                    "invalid single-source rank prefix");
        tiling.single_source_slots = reinterpret_cast<uint64_t>(single_source_slots->data_ptr());
        tiling.rank_prefix_matrix = reinterpret_cast<uint64_t>(rank_prefix_matrix->data_ptr());
        tiling.rank = static_cast<uint32_t>(rank);
        tiling.world_size = static_cast<uint32_t>(world_size);
    }
    tiling.num_recv_tokens = static_cast<uint32_t>(num_recv_tokens);
    tiling.num_expanded_tokens = static_cast<uint32_t>(x.size(0));
    tiling.hidden = static_cast<uint32_t>(x.size(1));
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.block_dim = static_cast<uint32_t>(resolved_num_aiv);
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.has_expanded_values = expanded_values.has_value() ? 1U : 0U;
    tiling.x_dtype = x.scalar_type() == torch::kBFloat16 ? kAggregateDtypeBfloat16 : kAggregateDtypeFloat16;
    tiling.x_element_bytes = static_cast<uint32_t>(x.element_size());

    auto launch = [=] {
        // Retain owners of addresses embedded in the tiling.
        (void)single_source_slots;
        (void)rank_prefix_matrix;
        LaunchAggregateKernel(x, topk_weights, expanded_values, recv_to_expanded_idx, aggregated_x, aggregated_values,
                              tiling, ffts_addr);
    };
    return {aggregated_x, aggregated_values, std::move(launch)};
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Aggregate(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> expanded_values,
    torch::Tensor recv_to_expanded_idx, int64_t num_aiv, uint64_t ffts_addr,
    c10::optional<int64_t> allocation_stream_id, c10::optional<torch::Tensor> single_source_slots,
    c10::optional<torch::Tensor> rank_prefix_matrix, int rank, int world_size)
{
    auto plan = PrepareAggregate(x, topk_weights, expanded_values, recv_to_expanded_idx, num_aiv, ffts_addr,
                                 allocation_stream_id, single_source_slots, rank_prefix_matrix, rank, world_size);
    plan.launch();
    return {plan.x, plan.values};
}

}  // namespace ascend_deepep::ops
