// Copyright (c) 2026, Lu Lu

// JIT-backed direct entries whose AOT sources have been removed.
#include "runtime/jit_runtime.hpp"
#include "common/launch.h"
#include "common/platform.h"
#include "common/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::string> flags()
{
    return {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DASCEND_DEEPEP_ARCH=3510",
        "-DEP_NUM_TOPK_IDX_BITS=64",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        "--cce-auto-infer-kernel-type=false",
    };
}

std::string source(const char* file)
{
    return std::string("#include <direct/") + file + ">\n";
}

std::shared_ptr<deepep::JitKernel> compile(const char* name, const char* file)
{
    return deepep::jit_compile(name, source(file), flags());
}

}  // namespace

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_combine_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr)
{
    const auto kernel = compile("cached_notify_combine", "kernels/notify/notify_combine.cpp");
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), 0, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void notify_dispatch_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* num_tokens_per_rank, uint8_t* num_tokens_per_expert,
    uint8_t* is_token_in_rank, uint8_t* symmetric_buffer, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix,
    uint8_t* recv_channel_prefix_matrix, uint8_t* num_recv_tokens, uint8_t* num_recv_tokens_per_expert,
    uint8_t* dispatch_plan, uint8_t* send_head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto kernel = compile("notify_dispatch_plan", "kernels/notify/notify_dispatch_kernel.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, num_tokens_per_rank,
                       num_tokens_per_expert, is_token_in_rank, symmetric_buffer, rank_prefix_matrix,
                       channel_prefix_matrix, recv_channel_prefix_matrix, num_recv_tokens, num_recv_tokens_per_expert,
                       dispatch_plan, send_head, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr)
{
    const auto kernel = compile("cached_notify_dispatch", "kernels/notify/cached_notify_dispatch_kernel.cpp");
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), 0, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_dispatch_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* routes, uint8_t* plan, uint8_t* head, uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    const auto kernel = compile("cached_dispatch_plan", "kernels/notify/cached_dispatch_plan_kernel.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, routes, plan, head, td,
                       ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_layout_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* topk_idx, uint8_t* scratch_buffer, uint8_t* num_tokens_per_rank,
    uint8_t* num_tokens_per_expert, uint8_t* is_token_in_rank, uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto kernel =
        compile("dispatch_layout", "kernels/dispatch/dispatch_layout.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::DispatchLayoutTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, topk_idx, scratch_buffer,
                       num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void expand_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* expanded_x, uint8_t* expanded_x_scales, uint8_t* expanded_topk_weights, uint8_t* expert_offsets,
    uint8_t* actual_expert_counts, uint8_t* recv_to_expanded_idx, uint8_t* scratch_buffer, uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    const auto kernel = compile("expand", "kernels/expand/expand.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::ExpandTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, x, x_scales, topk_idx, topk_weights,
                       expanded_x, expanded_x_scales, expanded_topk_weights, expert_offsets, actual_expert_counts,
                       recv_to_expanded_idx, scratch_buffer, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void aggregate_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* expanded_values,
    uint8_t* recv_to_expanded_idx, uint8_t* aggregated_x, uint8_t* aggregated_values, uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    const auto kernel = compile("aggregate", "kernels/aggregate/aggregate.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::AggregateTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, x, topk_weights,
                       expanded_values, recv_to_expanded_idx, aggregated_x, aggregated_values, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto kernel = compile("v1_dispatch", "kernels/dispatch/dispatch.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, x, x_scales, topk_idx, topk_weights,
                       num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, symmetric_buffer, recv_x,
                       recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix,
                       recv_channel_prefix_matrix, recv_src_idx, send_head, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_udma_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* dispatch_plan, uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto kernel =
        compile("v1_dispatch_udma", "kernels/dispatch/dispatch_udma.cpp");
    const auto td = *reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), ascend_deepep::platform::kKernelUbBytes, x, x_scales, topk_idx, topk_weights,
                       num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, symmetric_buffer, recv_x,
                       recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix,
                       recv_channel_prefix_matrix, recv_src_idx, send_head, dispatch_plan, td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT uint64_t v1_token_schedule_abi_3()
{
    return (uint64_t(sizeof(ascend_deepep::DispatchTilingData)) << 32) |
           uint64_t(sizeof(ascend_deepep::CombineTilingData));
}

#define DEEPEP_DIRECT_COMBINE_WRAPPER(function, jit_name, file)                                              \
    extern "C" ASCEND_DEEPEP_EXPORT void function(                                                                  \
        uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,      \
        uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,           \
        uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host,                \
        uint64_t ffts_addr)                                                                                         \
    {                                                                                                               \
        const auto kernel = compile(jit_name, file);                                                         \
        const auto td = *reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);                          \
        deepep::jit_launch(kernel, stream, static_cast<int>(block_dim),                                             \
                           static_cast<int>(ascend_deepep::platform::kKernelUbBytes), x, topk_weights, bias_0,       \
                           bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, symmetric_buffer,  \
                           recv_x, recv_topk_weights, td, ffts_addr);                                               \
    }

DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_kernel_do, "v1_combine", "kernels/combine/v1_combine_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_comm_kernel_do, "v1_combine_comm", "kernels/combine/v1_combine_comm_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_reduce_kernel_do, "v1_combine_reduce", "kernels/combine/v1_combine_reduce_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_kernel_do, "v1_combine_udma", "kernels/combine/v1_combine_udma_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_comm_kernel_do, "v1_combine_udma_comm", "kernels/combine/v1_combine_udma_comm_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_reduce_kernel_do, "v1_combine_udma_reduce", "kernels/combine/v1_combine_udma_reduce_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_early_kernel_do, "v1_combine_udma_early", "kernels/combine/v1_combine_udma_early_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_remaining_kernel_do, "v1_combine_udma_remaining", "kernels/combine/v1_combine_udma_remaining_kernel.cpp")
DEEPEP_DIRECT_COMBINE_WRAPPER(v1_combine_udma_pipeline_kernel_do, "v1_combine_udma_pipeline", "kernels/combine/v1_combine_udma_pipeline_kernel.cpp")

#undef DEEPEP_DIRECT_COMBINE_WRAPPER
