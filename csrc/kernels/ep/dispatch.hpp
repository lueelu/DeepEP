// Copyright (c) 2026, Lu Lu
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

#include <acl/acl.h>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>

#include "runtime/jit.hpp"

namespace deep_ep {

// Host admission. Mirrors the device DispatchUBLayout constants; capacity
// (not the actual token count) determines the GM and UB footprint.
inline EpMemoryServerDedupLayout get_dispatch_buffer_size(uint32_t num_ranks, uint32_t num_max_tokens_per_rank,
                                                         uint32_t num_hidden_elems, uint32_t num_topk,
                                                         uint32_t num_experts, EpServerDedupInputMode mode,
                                                         uint32_t num_vec_cores)
{
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0U && num_hidden_elems > 0U);
    EP_HOST_ASSERT(num_topk > 0U && num_topk <= kEpServerDedupMaxTopK);
    EP_HOST_ASSERT(num_experts > 0U && num_experts <= kEpServerDedupMaxExperts && num_experts % num_ranks == 0U);
    EP_HOST_ASSERT(LlCoreCountValid(num_vec_cores) && num_vec_cores <= kEpServerDedupMaxVecCores);
    EP_HOST_ASSERT(num_topk <= num_experts);

    EpMemoryServerDedupLayout layout{};
    layout.inputMode = mode;
    layout.globalBs = uint64_t(num_ranks) * num_max_tokens_per_rank;
    layout.localExpertNum = num_experts / num_ranks;
    layout.capacityRows = layout.globalBs * std::min<uint64_t>(num_topk, layout.localExpertNum);
    layout.data = DispatchServerDedupDataLayout(num_max_tokens_per_rank, num_hidden_elems, num_topk, num_experts,
                                                mode);
    const bool prequantized = mode == EpServerDedupInputMode::PrequantizedFp8Packs;
    const bool wire_fp8 = mode != EpServerDedupInputMode::Plain16;
    layout.xBytes = uint64_t(num_max_tokens_per_rank) * num_hidden_elems * (prequantized ? 1U : 2U);
    layout.inputScalesBytes = prequantized ? uint64_t(num_max_tokens_per_rank) * layout.data.scaleBytes : 0U;
    layout.expertIdsBytes = uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(int32_t);
    layout.weightBytes = uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(float);
    layout.expandXBytes = layout.capacityRows * (wire_fp8 ? num_hidden_elems : num_hidden_elems * 2U);
    layout.dynamicScalesBytes = layout.capacityRows * layout.data.scaleBytes;
    layout.expertTokenNumsBytes = layout.localExpertNum * sizeof(int64_t);
    layout.sendCountsBytes = uint64_t(num_experts) * sizeof(int32_t);
    layout.tokenTypeBytes = layout.capacityRows * sizeof(uint32_t);
    layout.destinationIndexBytes = layout.capacityRows * 3U * sizeof(uint32_t);
    layout.sourceMaskBytes = uint64_t(num_max_tokens_per_rank) * sizeof(uint32_t);
    layout.relayReadIndexRowStrideBytes = uint64_t(num_topk) * 8U;
    layout.relayReadIndexBytes = layout.capacityRows * layout.relayReadIndexRowStrideBytes;
    EP_HOST_ASSERT(layout.data.requiredBytes <= kEpServerDedupWindowUsableBytes);
    EP_HOST_ASSERT(layout.data.blockCount > 0U && layout.data.blockCount <= UINT8_MAX);
    return layout;
}

static std::vector<std::string> dispatch_flags()
{
#ifndef DEEPEP_JIT_PI_ON_CHIP
#error "DEEPEP_JIT_PI_ON_CHIP must be forwarded from the SHMEM SDK identity"
#endif
    // The SDK identity decides the HNS_1825 doorbell mode; the JIT device code
    // must match the library it will call, so CMake forwards the recorded value.
    static_assert(DEEPEP_JIT_PI_ON_CHIP == 0 || DEEPEP_JIT_PI_ON_CHIP == 1,
                  "DEEPEP_JIT_PI_ON_CHIP must be the SDK's recorded 0/1 doorbell mode");
    std::vector<std::string> flags = {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        std::format("-DACLSHMEMI_HNS_1825_PI_ON_CHIP={}", DEEPEP_JIT_PI_ON_CHIP),
        "--cce-auto-infer-kernel-type=false",
    };
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    flags.emplace_back("-DDEEPEP_LL_REFERENCE_PROFILE=1");
#endif
    return flags;
}

static void launch_dispatch(
    void* x, void* expert_ids, void* expert_scales,
    void* expand_x_out, void* dynamic_scales_out,
    void* expert_token_nums_out, void* send_counts_out,
    void* token_type_out, void* destination_index_out,
    void* relay_read_index_out, void* source_mask_out,
    void* source_weights_out, void* input_scales,
    void* peer_ptrs, void* rdma_remote_base, void* rdma_stage_base,
    void* profile_out,
    const int& rank_idx, const int& num_tokens,
    const int& physical_ranks_per_host, const int& backend,
    const int& stage_block_num, const int64_t& peer_mem_bytes,
    const int64_t& ffts_val,
    const int& num_ranks, const int& num_max_tokens_per_rank,
    const int& num_hidden_bytes, const int& num_topk,
    const int& num_experts, const int& num_vec_cores, const int& num_ub_bytes,
    const int& num_ranks_per_group, const bool& prequantized,
    const bool& dedup, const bool& rdma,
    void* stream) {
    EP_HOST_ASSERT(num_vec_cores > 0 && num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_hidden_bytes > 0 && num_experts > 0 && num_topk > 0);
    EP_HOST_ASSERT(num_ranks > 0 && num_ranks_per_group > 0 && num_ub_bytes > 0);
    EP_HOST_ASSERT(x != nullptr && peer_ptrs != nullptr);

    // Compile. The template list mirrors the device dispatch_impl axes.
    const std::string shape = std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
        num_ranks, num_max_tokens_per_rank, num_hidden_bytes, num_topk,
        num_experts, num_vec_cores, num_ub_bytes, num_ranks_per_group,
        prequantized, dedup, rdma);
    const std::string call_args =
        "x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out, expert_token_nums_out, "
        "send_counts_out, token_type_out, destination_index_out, relay_read_index_out, source_mask_out, "
        "source_weights_out, input_scales, peer_ptrs, rdma_remote_base, rdma_stage_base, profile_out, "
        "rank_idx, num_tokens, physical_ranks_per_host, backend, stage_block_num, "
        "peer_mem_bytes, ffts_val, OrdinaryCoreContext()";
    std::string source = std::format(R"(
#include <deep_ep/impls/ep/dispatch.hpp>

using namespace deep_ep;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_jit(
    __gm__ uint8_t* x, __gm__ uint8_t* expert_ids, __gm__ uint8_t* expert_scales,
    __gm__ uint8_t* expand_x_out, __gm__ uint8_t* dynamic_scales_out,
    __gm__ uint8_t* expert_token_nums_out, __gm__ uint8_t* send_counts_out,
    __gm__ uint8_t* token_type_out, __gm__ uint8_t* destination_index_out,
    __gm__ uint8_t* relay_read_index_out, __gm__ uint8_t* source_mask_out,
    __gm__ uint8_t* source_weights_out, __gm__ uint8_t* input_scales,
    __gm__ uint8_t* peer_ptrs, __gm__ uint8_t* rdma_remote_base, __gm__ uint8_t* rdma_stage_base,
    __gm__ uint8_t* profile_out,
    int32_t rank_idx, int32_t num_tokens, int32_t physical_ranks_per_host, int32_t backend,
    int32_t stage_block_num, int64_t peer_mem_bytes, int64_t ffts_val)
{{
    dispatch_body<{}>({});
}}
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_jit);
    (void)ptr;
}}
)", shape, call_args);
    for (uint32_t slot = 0; slot < 4U; ++slot) {
        source += std::format(
            "extern \"C\" __sk__ __vector__ void dispatch_jit_sk{}"
            "(const DispatchArgs* args, sk::SkSystemArgs* system_args)\n{{\n"
            "    dispatch_impl_sk<{}, {}>(args, system_args);\n}}\n", slot, shape, slot);
    }
    source += "SK_BIND(dispatch_jit, 0, dispatch_jit_sk0, dispatch_jit_sk1, dispatch_jit_sk2, dispatch_jit_sk3);\n";
    deep_jit::ascend::CompilerOptions options;
    options.extra_bisheng_flags = dispatch_flags();
    const auto kernel = jit->compile("dispatch", source, options);

    // Launch. One explicit parameter per kernel argument.
    deep_jit::ascend::LaunchOptions launch_options;
    launch_options.stream = static_cast<aclrtStream>(stream);
    launch_options.num_blocks = num_vec_cores;
    launch_options.num_ubuf_bytes = num_ub_bytes;
    jit->launch(kernel, launch_options,
                x, expert_ids, expert_scales,
                expand_x_out, dynamic_scales_out,
                expert_token_nums_out, send_counts_out,
                token_type_out, destination_index_out,
                relay_read_index_out, source_mask_out,
                source_weights_out, input_scales,
                peer_ptrs, rdma_remote_base, rdma_stage_base,
                profile_out,
                rank_idx, num_tokens,
                physical_ranks_per_host, backend,
                stage_block_num, peer_mem_bytes,
                ffts_val);
}

}  // namespace deep_ep
