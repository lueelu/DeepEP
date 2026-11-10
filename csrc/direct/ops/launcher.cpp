#include "launcher.h"

#include <acl/acl.h>
#include <cstdlib>
#include <cstring>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "common/launch.h"

namespace {

aclrtStream GetCurrentAclStream()
{
    return c10_npu::getCurrentNPUStream().stream();
}

uint8_t* BytePtr(void* ptr)
{
    return static_cast<uint8_t*>(ptr);
}

uint8_t* OptionalTensorPtr(const c10::optional<torch::Tensor>& tensor)
{
    return tensor.has_value() ? BytePtr(tensor->data_ptr()) : nullptr;
}

}  // namespace

void LaunchDispatchLayoutKernel(torch::Tensor topk_idx, torch::Tensor num_tokens_per_rank,
                                torch::Tensor num_tokens_per_expert, torch::Tensor is_token_in_rank,
                                const ascend_deepep::DispatchLayoutTilingData& tiling, torch::Tensor scratch_buffer,
                                uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    dispatch_layout_kernel_do(tiling.block_dim, stream, BytePtr(topk_idx.data_ptr()),
                              BytePtr(scratch_buffer.data_ptr()), BytePtr(num_tokens_per_rank.data_ptr()),
                              BytePtr(num_tokens_per_expert.data_ptr()), BytePtr(is_token_in_rank.data_ptr()),
                              reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchLayoutTilingData*>(&tiling)),
                              ffts_addr);
}

void LaunchNotifyDispatchKernel(torch::Tensor num_tokens_per_rank, torch::Tensor num_tokens_per_expert,
                                torch::Tensor is_token_in_rank, torch::Tensor rank_prefix_matrix,
                                torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix,
                                torch::Tensor num_recv_tokens, torch::Tensor num_recv_tokens_per_expert,
                                const c10::optional<torch::Tensor>& dispatch_plan, torch::Tensor send_head,
                                const ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer,
                                uint64_t ffts_addr)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    notify_dispatch_plan_kernel_do(
        tiling.block_dim, stream, BytePtr(num_tokens_per_rank.data_ptr()), BytePtr(num_tokens_per_expert.data_ptr()),
        BytePtr(is_token_in_rank.data_ptr()), BytePtr(symmetric_buffer), BytePtr(rank_prefix_matrix.data_ptr()),
        BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
        BytePtr(num_recv_tokens.data_ptr()), BytePtr(num_recv_tokens_per_expert.data_ptr()),
        OptionalTensorPtr(dispatch_plan), BytePtr(send_head.data_ptr()),
        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
}

void LaunchCachedNotifyDispatchKernel(const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    cached_notify_dispatch_kernel_do(tiling.block_dim, stream, ffts_addr);
}

void LaunchCachedDispatchPlanKernel(torch::Tensor routes, torch::Tensor plan, torch::Tensor send_head,
                                    const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr)
{
    cached_dispatch_plan_kernel_do(
        tiling.block_dim, GetCurrentAclStream(), BytePtr(routes.data_ptr()), BytePtr(plan.data_ptr()),
        send_head.numel() ? BytePtr(send_head.data_ptr()) : nullptr,
        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
}

void LaunchV1DispatchKernel(
    torch::Tensor x, const c10::optional<torch::Tensor>& x_scales, const c10::optional<torch::Tensor>& topk_idx,
    const c10::optional<torch::Tensor>& topk_weights, const c10::optional<torch::Tensor>& num_tokens_per_rank,
    torch::Tensor is_token_in_rank, const c10::optional<torch::Tensor>& num_tokens_per_expert, torch::Tensor recv_x,
    const c10::optional<torch::Tensor>& recv_x_scales, const c10::optional<torch::Tensor>& recv_topk_idx,
    const c10::optional<torch::Tensor>& recv_topk_weights, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix, torch::Tensor recv_src_idx,
    torch::Tensor send_head, const c10::optional<torch::Tensor>& dispatch_plan,
    const ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr, bool use_udma)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    if (use_udma) {
        v1_dispatch_udma_plan_kernel_do(
            tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales), OptionalTensorPtr(topk_idx),
            OptionalTensorPtr(topk_weights), OptionalTensorPtr(num_tokens_per_rank),
            BytePtr(is_token_in_rank.data_ptr()), OptionalTensorPtr(num_tokens_per_expert), BytePtr(symmetric_buffer),
            BytePtr(recv_x.data_ptr()), OptionalTensorPtr(recv_x_scales), OptionalTensorPtr(recv_topk_idx),
            OptionalTensorPtr(recv_topk_weights), BytePtr(rank_prefix_matrix.data_ptr()),
            BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
            BytePtr(recv_src_idx.data_ptr()), BytePtr(send_head.data_ptr()), OptionalTensorPtr(dispatch_plan),
            reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
    } else {
        v1_dispatch_kernel_do(
            tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales), OptionalTensorPtr(topk_idx),
            OptionalTensorPtr(topk_weights), OptionalTensorPtr(num_tokens_per_rank),
            BytePtr(is_token_in_rank.data_ptr()), OptionalTensorPtr(num_tokens_per_expert), BytePtr(symmetric_buffer),
            BytePtr(recv_x.data_ptr()), OptionalTensorPtr(recv_x_scales), OptionalTensorPtr(recv_topk_idx),
            OptionalTensorPtr(recv_topk_weights), BytePtr(rank_prefix_matrix.data_ptr()),
            BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
            BytePtr(recv_src_idx.data_ptr()), BytePtr(send_head.data_ptr()),
            reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
    }
}

void LaunchV1CombineKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& bias_0, const c10::optional<torch::Tensor>& bias_1,
                           torch::Tensor src_idx, torch::Tensor rank_prefix_matrix, torch::Tensor channel_prefix_matrix,
                           torch::Tensor send_head, torch::Tensor recv_x,
                           const c10::optional<torch::Tensor>& recv_topk_weights,
                           const ascend_deepep::CombineTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr,
                           bool use_udma, CombinePhase phase)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    // Profiling may separate arrival skew into an all-rank/all-AIV notify.
    // The benchmark enables this process-local switch only during capture;
    // every rank must use the same setting. Ordinary V1/V2 UDMA relies on
    // Begin, which retains its control-header reset and synchronization.
    const char* profile_notify = std::getenv("DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY");
    const bool separate_notify = profile_notify && std::strcmp(profile_notify, "1") == 0;
    const bool has_udma_begin =
        use_udma && (phase == CombinePhase::Fused || phase == CombinePhase::Communication) && !separate_notify;
    if (!has_udma_begin && phase != CombinePhase::Reduction && phase != CombinePhase::Remaining) {
        cached_notify_combine_kernel_do(tiling.block_dim, stream, ffts_addr);
    }
    auto* combine_kernel_do = use_udma ? &v1_combine_udma_kernel_do : &v1_combine_kernel_do;
    if (phase == CombinePhase::Communication) {
        combine_kernel_do = use_udma ? &v1_combine_udma_comm_kernel_do : &v1_combine_comm_kernel_do;
    } else if (phase == CombinePhase::Reduction) {
        combine_kernel_do = use_udma ? &v1_combine_udma_reduce_kernel_do : &v1_combine_reduce_kernel_do;
    } else if (phase == CombinePhase::Pipeline) {
        TORCH_CHECK(use_udma, "aggregate pipeline requires UDMA");
        combine_kernel_do = &v1_combine_udma_pipeline_kernel_do;
    } else if (phase == CombinePhase::EarlySingles) {
        TORCH_CHECK(use_udma, "early single-source combine requires UDMA");
        combine_kernel_do = &v1_combine_udma_early_kernel_do;
    } else if (phase == CombinePhase::Remaining) {
        TORCH_CHECK(use_udma, "early single-source combine requires UDMA");
        combine_kernel_do = &v1_combine_udma_remaining_kernel_do;
    }
    // Communication retains every original completion wait. Reduction reads
    // the same SHMEM slots on the same stream after the optional event wait.
    auto launch = [&](auto* kernel_do) {
        kernel_do(tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(topk_weights),
                  OptionalTensorPtr(bias_0), OptionalTensorPtr(bias_1), BytePtr(src_idx.data_ptr()),
                  BytePtr(rank_prefix_matrix.data_ptr()), BytePtr(channel_prefix_matrix.data_ptr()),
                  BytePtr(send_head.data_ptr()), BytePtr(symmetric_buffer), BytePtr(recv_x.data_ptr()),
                  OptionalTensorPtr(recv_topk_weights),
                  reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::CombineTilingData*>(&tiling)), ffts_addr);
    };
    launch(combine_kernel_do);
}

void LaunchExpandKernel(torch::Tensor x, const c10::optional<torch::Tensor>& x_scales,
                        const c10::optional<torch::Tensor>& topk_idx, const c10::optional<torch::Tensor>& topk_weights,
                        torch::Tensor expanded_x, const c10::optional<torch::Tensor>& expanded_x_scales,
                        const c10::optional<torch::Tensor>& expanded_topk_weights, torch::Tensor expert_offsets,
                        torch::Tensor actual_expert_counts, torch::Tensor recv_to_expanded_idx,
                        torch::Tensor scratch_buffer, const ascend_deepep::ExpandTilingData& tiling, uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    expand_kernel_do(tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales),
                     OptionalTensorPtr(topk_idx), OptionalTensorPtr(topk_weights), BytePtr(expanded_x.data_ptr()),
                     OptionalTensorPtr(expanded_x_scales), OptionalTensorPtr(expanded_topk_weights),
                     BytePtr(expert_offsets.data_ptr()), BytePtr(actual_expert_counts.data_ptr()),
                     BytePtr(recv_to_expanded_idx.data_ptr()), BytePtr(scratch_buffer.data_ptr()),
                     reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::ExpandTilingData*>(&tiling)), ffts_addr);
}

void LaunchAggregateKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& expanded_values, torch::Tensor recv_to_expanded_idx,
                           torch::Tensor aggregated_x, const c10::optional<torch::Tensor>& aggregated_values,
                           const ascend_deepep::AggregateTilingData& tiling, uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    aggregate_kernel_do(tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(topk_weights),
                        OptionalTensorPtr(expanded_values), BytePtr(recv_to_expanded_idx.data_ptr()),
                        BytePtr(aggregated_x.data_ptr()), OptionalTensorPtr(aggregated_values),
                        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::AggregateTilingData*>(&tiling)),
                        ffts_addr);
}

void LaunchElasticBarrierKernel(int64_t block_dim, uint64_t ffts_addr)
{
    TORCH_CHECK(block_dim > 0, "elastic barrier block_dim must be positive");
    cached_notify_combine_kernel_do(static_cast<uint32_t>(block_dim), GetCurrentAclStream(), ffts_addr);
}
