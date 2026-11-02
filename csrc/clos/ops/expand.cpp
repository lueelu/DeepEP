#include "ops/expand.h"

#include <cstdint>
#include <limits>

#include "common/types.h"
#include "common/platform.h"
#include "ops/launcher.h"
#include "ops/stream_allocator.h"

namespace {

constexpr int64_t kMaxExpandTopk = 128;
constexpr int64_t kMaxLocalExperts = 1024;

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

int64_t CheckedAdd(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " values must be nonnegative");
    TORCH_CHECK(lhs <= std::numeric_limits<int64_t>::max() - rhs, name, " overflow");
    return lhs + rhs;
}

}  // namespace

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
           torch::Tensor>
Expand(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
       c10::optional<torch::Tensor> topk_weights, std::vector<int64_t> num_tokens_per_expert, int64_t expert_alignment,
       int64_t cached_num_expanded_tokens, c10::optional<torch::Tensor> cached_expert_offsets,
       c10::optional<torch::Tensor> cached_actual_expert_counts,
       c10::optional<torch::Tensor> cached_recv_to_expanded_idx, int64_t num_aiv, uint64_t ffts_addr,
       c10::optional<int64_t> allocation_stream_id, c10::optional<torch::Tensor> public_expert_prefix,
       c10::optional<torch::Tensor> public_src_metadata, c10::optional<torch::Tensor> single_source_slots)
{
    StreamAllocator allocator(allocation_stream_id);
    allocator.Record({x, x_scales, topk_idx, topk_weights, cached_expert_offsets, cached_actual_expert_counts,
                      cached_recv_to_expanded_idx, public_expert_prefix, public_src_metadata, single_source_slots});
    CheckNpuContiguous(x, "x");
    CheckOptionalNpuContiguous(x_scales, "x_scales");
    CheckOptionalNpuContiguous(topk_idx, "topk_idx");
    CheckOptionalNpuContiguous(topk_weights, "topk_weights");
    CheckOptionalNpuContiguous(cached_expert_offsets, "cached_expert_offsets");
    CheckOptionalNpuContiguous(cached_actual_expert_counts, "cached_actual_expert_counts");
    CheckOptionalNpuContiguous(cached_recv_to_expanded_idx, "cached_recv_to_expanded_idx");

    TORCH_CHECK(x.dim() == 2, "x must have shape [num_recv_tokens, hidden]");
    TORCH_CHECK(x.size(1) > 0, "x hidden dimension must be positive");
    TORCH_CHECK(expert_alignment > 0, "expert_alignment must be positive");
    TORCH_CHECK(expert_alignment <= std::numeric_limits<uint32_t>::max(), "expert_alignment must fit uint32");
    const int64_t resolved_num_aiv = ResolveNumAiv(num_aiv);
    const bool cached_mode = cached_recv_to_expanded_idx.has_value();
    TORCH_CHECK(
        cached_mode == cached_expert_offsets.has_value() && cached_mode == cached_actual_expert_counts.has_value(),
        "cached expand metadata must be provided together");

    int64_t num_topk = 0;
    int64_t num_local_experts = 0;
    int64_t num_expanded_tokens = 0;
    torch::Tensor expert_offsets;
    torch::Tensor actual_expert_counts;
    torch::Tensor recv_to_expanded_idx;

    auto int_options = x.options().dtype(torch::kInt32);
    if (cached_mode) {
        TORCH_CHECK(!topk_idx.has_value(), "cached expand requires topk_idx=None");
        TORCH_CHECK(num_tokens_per_expert.empty(), "cached expand does not accept num_tokens_per_expert");
        expert_offsets = *cached_expert_offsets;
        actual_expert_counts = *cached_actual_expert_counts;
        recv_to_expanded_idx = *cached_recv_to_expanded_idx;
        TORCH_CHECK(expert_offsets.dim() == 1 && actual_expert_counts.dim() == 1,
                    "cached expert metadata must be one-dimensional");
        TORCH_CHECK(
            expert_offsets.scalar_type() == torch::kInt32 && actual_expert_counts.scalar_type() == torch::kInt32,
            "cached expert metadata must be int32");
        num_local_experts = actual_expert_counts.numel();
        TORCH_CHECK(expert_offsets.numel() == num_local_experts + 1,
                    "cached expert_offsets must have num_local_experts + 1 elements");
        TORCH_CHECK(recv_to_expanded_idx.dim() == 2 && recv_to_expanded_idx.scalar_type() == torch::kInt32,
                    "cached recv_to_expanded_idx must be a two-dimensional int32 tensor");
        TORCH_CHECK(recv_to_expanded_idx.size(0) == x.size(0),
                    "cached mapping and x must have the same num_recv_tokens");
        num_topk = recv_to_expanded_idx.size(1);
        num_expanded_tokens = cached_num_expanded_tokens;
        TORCH_CHECK(num_expanded_tokens >= 0, "cached_num_expanded_tokens must be nonnegative");
    } else {
        TORCH_CHECK(topk_idx.has_value(), "topk_idx is required for expand");
        TORCH_CHECK(cached_num_expanded_tokens == 0, "cached_num_expanded_tokens is only valid in cached mode");
        TORCH_CHECK(topk_idx->dim() == 2 && topk_idx->size(0) == x.size(0),
                    "topk_idx must have shape [num_recv_tokens, num_topk]");
        TORCH_CHECK(topk_idx->scalar_type() == c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value,
                    "topk_idx must have dtype deep_ep.topk_idx_t");
        num_topk = topk_idx->size(1);
        num_local_experts = static_cast<int64_t>(num_tokens_per_expert.size());
        TORCH_CHECK(num_local_experts > 0, "num_tokens_per_expert must not be empty");
        TORCH_CHECK(num_local_experts <= kMaxLocalExperts, "num_local_experts must not exceed ", kMaxLocalExperts);
        for (int64_t count : num_tokens_per_expert) {
            TORCH_CHECK(count >= 0, "expert token counts must be nonnegative");
            TORCH_CHECK(count % expert_alignment == 0, "expert token counts must be aligned to expert_alignment");
            num_expanded_tokens = CheckedAdd(num_expanded_tokens, count, "num_expanded_tokens");
        }
        expert_offsets = allocator.Empty({num_local_experts + 1}, int_options);
        actual_expert_counts = allocator.Empty({num_local_experts}, int_options);
        recv_to_expanded_idx = allocator.Empty({x.size(0), num_topk}, int_options);
    }

    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(num_topk <= kMaxExpandTopk, "num_topk must not exceed ", kMaxExpandTopk);
    TORCH_CHECK(num_local_experts <= kMaxLocalExperts, "num_local_experts must not exceed ", kMaxLocalExperts);
    TORCH_CHECK(x.size(0) <= std::numeric_limits<uint32_t>::max() &&
                    x.size(1) <= std::numeric_limits<uint32_t>::max() &&
                    num_topk <= std::numeric_limits<uint32_t>::max() &&
                    num_local_experts <= std::numeric_limits<uint32_t>::max() &&
                    num_expanded_tokens <= std::numeric_limits<uint32_t>::max(),
                "expand dimensions must fit uint32");

    if (topk_weights.has_value()) {
        TORCH_CHECK(topk_weights->dim() == 2 && topk_weights->size(0) == x.size(0) && topk_weights->size(1) == num_topk,
                    "topk_weights must have shape [num_recv_tokens, num_topk]");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
    }

    int64_t num_scales = 0;
    int64_t scale_element_bytes = 0;
    if (x_scales.has_value()) {
        TORCH_CHECK(x_scales->dim() == 2 && x_scales->size(0) == x.size(0),
                    "x_scales must have shape [num_recv_tokens, num_scales]");
        num_scales = x_scales->size(1);
        scale_element_bytes = x_scales->element_size();
        TORCH_CHECK(num_scales <= std::numeric_limits<uint32_t>::max(), "num_scales must fit uint32");
    }

    auto expanded_x = allocator.Empty({num_expanded_tokens, x.size(1)}, x.options());
    c10::optional<torch::Tensor> expanded_x_scales = c10::nullopt;
    if (x_scales.has_value()) {
        expanded_x_scales = allocator.Empty({num_expanded_tokens, num_scales}, x_scales->options());
    }
    c10::optional<torch::Tensor> expanded_topk_weights = c10::nullopt;
    if (topk_weights.has_value()) {
        expanded_topk_weights = allocator.Empty({num_expanded_tokens}, topk_weights->options());
    }

    torch::Tensor scratch_buffer =
        cached_mode ? recv_to_expanded_idx : allocator.Empty({2 * resolved_num_aiv * num_local_experts}, int_options);

    TORCH_CHECK(public_expert_prefix.has_value() == public_src_metadata.has_value(),
                "both V2 expand metadata outputs are required");
    if (public_expert_prefix.has_value()) {
        TORCH_CHECK(!cached_mode, "cached expand reuses public metadata");
        CheckOptionalNpuContiguous(public_expert_prefix, "public_expert_prefix");
        CheckOptionalNpuContiguous(public_src_metadata, "public_src_metadata");
        TORCH_CHECK(public_expert_prefix->scalar_type() == torch::kInt32 && public_expert_prefix->dim() == 1 &&
                        public_expert_prefix->numel() == num_local_experts,
                    "invalid public expert prefix");
        TORCH_CHECK(public_src_metadata->scalar_type() == torch::kInt32 && public_src_metadata->dim() == 2 &&
                        public_src_metadata->size(0) == x.size(0) && public_src_metadata->size(1) == num_topk + 2,
                    "invalid public source metadata");
    }
    if (single_source_slots.has_value()) {
        CheckNpuContiguous(*single_source_slots, "single_source_slots");
        TORCH_CHECK(!cached_mode, "cached expand reuses single-source metadata");
        TORCH_CHECK(single_source_slots->device() == x.device() &&
                        single_source_slots->scalar_type() == torch::kInt32 && single_source_slots->dim() == 1 &&
                        single_source_slots->size(0) == x.size(0),
                    "single_source_slots must be int32[num_recv_tokens] on x's device");
        TORCH_CHECK(num_expanded_tokens <= std::numeric_limits<int32_t>::max(),
                    "single-source expanded row index must fit int32");
    }
    ascend_deepep::ExpandTilingData tiling{};
    if (single_source_slots.has_value())
        tiling.single_source_slots = reinterpret_cast<uint64_t>(single_source_slots->data_ptr());
    if (public_expert_prefix.has_value()) {
        tiling.v2_expert_prefix = reinterpret_cast<uint64_t>(public_expert_prefix->data_ptr());
        tiling.v2_src_metadata = reinterpret_cast<uint64_t>(public_src_metadata->data_ptr());
    }
    tiling.num_recv_tokens = static_cast<uint32_t>(x.size(0));
    tiling.hidden = static_cast<uint32_t>(x.size(1));
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.num_local_experts = static_cast<uint32_t>(num_local_experts);
    tiling.num_expanded_tokens = static_cast<uint32_t>(num_expanded_tokens);
    tiling.expert_alignment = static_cast<uint32_t>(expert_alignment);
    tiling.block_dim = static_cast<uint32_t>(resolved_num_aiv);
    tiling.cached_mode = cached_mode ? 1U : 0U;
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.has_x_scales = x_scales.has_value() ? 1U : 0U;
    tiling.x_element_bytes = static_cast<uint32_t>(x.element_size());
    tiling.scale_element_bytes = static_cast<uint32_t>(scale_element_bytes);
    tiling.num_scales = static_cast<uint32_t>(num_scales);

    LaunchExpandKernel(x, x_scales, topk_idx, topk_weights, expanded_x, expanded_x_scales, expanded_topk_weights,
                       expert_offsets, actual_expert_counts, recv_to_expanded_idx, scratch_buffer, tiling, ffts_addr);

    return {expanded_x,     expanded_x_scales,    expanded_topk_weights,
            expert_offsets, actual_expert_counts, recv_to_expanded_idx};
}

}  // namespace ascend_deepep::ops
