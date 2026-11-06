// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "launcher.hpp"

#include <acl/acl.h>
#include <array>

#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "kernel_launch.hpp"
#include "csrc/kernels/bf16/common/mega_moe_tiling.hpp"

namespace {

aclrtStream GetCurrentAclStream()
{
    return c10_npu::getCurrentNPUStream().stream();
}

uint8_t* BytePtr(void* ptr)
{
    return static_cast<uint8_t*>(ptr);
}

struct MegaMoeContext {
    uint32_t ep_rank_id;
    uint32_t rank_size_per_server;
    uint64_t kfc_context_addr;
    uint64_t ep_buffer[1024];
};
static_assert(sizeof(MegaMoeContext) == 8208, "MegaMoe context ABI changed");

}  // namespace

bool MegaMoeBF16SwigluDecoupled(uint64_t hidden, uint32_t saved_mode)
{
    return deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(hidden, saved_mode) != 0;
}

void InitializeMegaMoeBF16Context(torch::Tensor context, void* symmetric_buffer, int rank, int world_size)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    const std::size_t context_bytes = static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext), "MegaMoe BF16 context tensor is too small");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(peer_ptr != nullptr, "aclshmem_ptr failed for MegaMoe BF16 peer rank ", peer);
        host_context.ep_buffer[peer] = reinterpret_cast<uint64_t>(peer_ptr);
    }
    const auto copy_status =
        aclrtMemcpy(context.data_ptr(), context_bytes, &host_context, sizeof(host_context), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_status == ACL_SUCCESS, "failed to initialize MegaMoe BF16 communication context, acl error ",
                copy_status);
}

void LaunchMegaMoeBF16ForwardKernel(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                    torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                    torch::Tensor y, torch::Tensor expert_token_nums,
                                    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
                                    torch::Tensor forward_transient_workspace, const MegaMoeTilingData& tiling,
                                    uint32_t block_dim, void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
                                    int rank, int world_size, uint64_t ffts_addr)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    const std::size_t context_bytes = static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext), "MegaMoe BF16 context tensor is too small");

    aclrtStream stream = GetCurrentAclStream();

    auto kernel = &deepep_train_mega_moe_bf16_forward_kernel_do;
    kernel(block_dim, stream, BytePtr(context.data_ptr()), BytePtr(x.data_ptr()), BytePtr(topk_ids.data_ptr()),
           BytePtr(topk_weights.data_ptr()), BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
           BytePtr(y.data_ptr()), BytePtr(expert_token_nums.data_ptr()), BytePtr(training_saved_state.data_ptr()),
           dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
           BytePtr(forward_transient_workspace.data_ptr()), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr);

    (void)symmetric_buffer_bytes;
}

void LaunchMegaMoeBF16BackwardKernel(torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
                                     torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
                                     torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
                                     torch::Tensor grad_weight2, torch::Tensor training_saved_state,
                                     torch::Tensor dispatch_output, torch::Tensor backward_workspace,
                                     const MegaMoeTilingData& tiling, uint32_t splitExperts, uint32_t block_dim,
                                     void* symmetric_buffer, std::size_t symmetric_buffer_bytes, int rank,
                                     int world_size, uint64_t ffts_addr)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    const std::size_t context_bytes = static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext), "MegaMoe BF16 context tensor is too small");

    aclrtStream stream = GetCurrentAclStream();
    auto kernel = &deepep_train_mega_moe_bf16_backward_ready_k_kernel_do;
    kernel(block_dim, stream, BytePtr(context.data_ptr()), BytePtr(grad_y.data_ptr()), BytePtr(topk_weights.data_ptr()),
           BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()), BytePtr(grad_x.data_ptr()),
           BytePtr(grad_topk_weights.data_ptr()), BytePtr(grad_weight1.data_ptr()), BytePtr(grad_weight2.data_ptr()),
           BytePtr(training_saved_state.data_ptr()), BytePtr(dispatch_output.data_ptr()),
           BytePtr(backward_workspace.data_ptr()), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr, splitExperts);
    (void)symmetric_buffer_bytes;
}
