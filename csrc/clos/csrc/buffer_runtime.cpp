/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csrc/buffer_runtime.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include <acl/acl.h>

#include "csrc/common/check.h"
#include "csrc/common/peer_mapping.h"
#include "ascend_deepep/dispatch_output.h"
#include "csrc/launcher.h"
#include "csrc/ops/aggregate.h"
#include "csrc/ops/combine.h"
#include "csrc/ops/dispatch.h"
#include "csrc/ops/dup_server.h"
#include "csrc/ops/expand.h"
#include "shmem.h"
#include "tiling/platform/platform_ascendc.h"

extern "C" uint64_t v1_token_schedule_abi_3();

namespace ascend_deepep::runtime {
namespace {

constexpr int64_t kMinimumEpWorldSize = 2;
constexpr int64_t kDeviceLargePageBytes = 2LL * 1024LL * 1024LL;
constexpr int64_t kCommMetaBytes = 2LL * 1024LL * 1024LL;

int64_t ResolveDataOpEngineType(const std::string& transport)
{
    if (transport == "auto" || transport == "mte") {
        return ACLSHMEM_DATA_OP_MTE;
    }
    if (transport == "roce") {
        return ACLSHMEM_DATA_OP_ROCE;
    }
    if (transport == "sdma") {
        return ACLSHMEM_DATA_OP_SDMA;
    }
    if (transport == "udma") {
        // URMA dispatch/combine use UDMA for token payloads while compact
        // metadata and local copies continue to use the MTE path.
        return ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA;
    }
    throw std::invalid_argument("unsupported transport: " + transport);
}

void ValidateHugePageMode(bool use_1g_huge_pages, int64_t num_bytes, int64_t data_op_engine_type)
{
    if (!use_1g_huge_pages) {
        return;
    }
    if (num_bytes <= 0) {
        throw std::invalid_argument("use_1g_huge_pages requires a non-empty communication workspace");
    }
    constexpr int64_t kSupportedEngines = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA;
    if ((data_op_engine_type & ~kSupportedEngines) != 0) {
        throw std::invalid_argument("use_1g_huge_pages currently supports only MTE and UDMA transports");
    }
}

int64_t ValidateAndGetLocalMemSize(int64_t rank, int64_t num_ranks, int64_t num_bytes, const std::string& transport)
{
    if (rank < 0 || num_ranks < kMinimumEpWorldSize || rank >= num_ranks) {
        throw std::invalid_argument("Buffer requires 0 <= rank < num_ranks and num_ranks >= 2");
    }
    if (num_bytes < 0) {
        throw std::invalid_argument("num_bytes must be greater than or equal to zero");
    }
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(num_ranks <= std::numeric_limits<uint32_t>::max(), "num_ranks must fit uint32");
    const int64_t control_bytes =
        transport == "udma"
            ? static_cast<int64_t>(ascend_deepep::udma::OutputControlBytes(static_cast<uint32_t>(num_ranks)))
            : 0;
    if (num_bytes > kMax - kCommMetaBytes - control_bytes) {
        throw std::overflow_error("requested communication buffer size is too large");
    }
    const int64_t required = num_bytes + kCommMetaBytes + control_bytes;
    if (required > kMax - (kDeviceLargePageBytes - 1)) {
        throw std::overflow_error("aligned communication buffer size is too large");
    }
    return ((required + kDeviceLargePageBytes - 1) / kDeviceLargePageBytes) * kDeviceLargePageBytes;
}

int64_t GetDeviceNumAiv()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get the AscendC platform manager");
    const int64_t num_aiv = static_cast<int64_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(num_aiv > 0, "AscendC reported an invalid AIV core count: ", num_aiv);
    return num_aiv;
}

}  // namespace

BufferRuntime::BufferRuntime(int64_t rank, int64_t num_ranks, int64_t num_bytes, const std::string& transport,
                             bool low_latency_mode, const std::string& ip_port, bool use_1g_huge_pages,
                             const std::vector<int64_t>& unique_id, const std::string& bootstrap_key,
                             bool v1_shared_jetty, bool v1_token_schedule, int64_t udma_dispatch_peer_order,
                             int64_t udma_combine_peer_order, int64_t udma_ranks_per_group,
                             int64_t udma_peers_per_group_per_worker, int64_t udma_workers_per_peer,
                             int64_t udma_qps_per_worker)
    : rank_(rank),
      num_ranks_(num_ranks),
      num_bytes_(num_bytes),
      local_mem_size_(ValidateAndGetLocalMemSize(rank, num_ranks, num_bytes, transport)),
      transport_(transport),
      data_op_engine_type_(ResolveDataOpEngineType(transport)),
      low_latency_mode_(low_latency_mode),
      use_1g_huge_pages_(use_1g_huge_pages),
      shmem_(std::make_shared<ShmemInstance>(
          rank, num_ranks, ip_port, data_op_engine_type_, use_1g_huge_pages, unique_id, bootstrap_key, v1_shared_jetty,
          v1_token_schedule, udma_dispatch_peer_order, udma_combine_peer_order, udma_ranks_per_group,
          udma_peers_per_group_per_worker, udma_workers_per_peer, udma_qps_per_worker))
{
    TORCH_CHECK(v1_token_schedule_abi_3() == ((uint64_t(sizeof(ascend_deepep::DispatchTilingData)) << 32) |
                                              uint64_t(sizeof(ascend_deepep::CombineTilingData))),
                "Kernel/native tiling ABI mismatch; rebuild and install with scripts/build.sh");

    ValidateHugePageMode(use_1g_huge_pages_, num_bytes_, data_op_engine_type_);
    num_device_aiv_ = GetDeviceNumAiv();
    try {
        if (use_1g_huge_pages_) {
            // The fixed prefix owns the workspace; the tail owns comm metadata
            // and, for UDMA, the independently-lived output address mailbox.
            const auto control_bytes =
                transport_ == "udma" ? ascend_deepep::udma::OutputControlBytes(static_cast<uint32_t>(num_ranks_)) : 0;
            shmem_->Init(static_cast<int64_t>(kCommMetaBytes + control_bytes), num_bytes_);
        } else {
            shmem_->Init(local_mem_size_);
        }
    } catch (const std::exception& error) {
        DestroyNoThrow();
        throw std::runtime_error("failed to initialize CANN SHMEM transport '" + transport_ + "': " + error.what());
    }
    initialized_ = true;
    try {
        AllocateCommMeta();
        AllocateWorkspace();
    } catch (...) {
        DestroyNoThrow();
        throw;
    }
}

BufferRuntime::~BufferRuntime() noexcept
{
    DestroyNoThrow();
}

bool BufferRuntime::IsInitialized() const
{
    return initialized_ && shmem_ != nullptr && shmem_->Initialized();
}

void BufferRuntime::Barrier()
{
    if (!IsInitialized()) {
        throw std::runtime_error("Buffer runtime is not initialized");
    }
    shmem_->Barrier();
}

void BufferRuntime::Destroy()
{
    ReleaseResources(true);
}

void BufferRuntime::AbortInitialization()
{
    ReleaseResources(false);
}

std::vector<std::string> BufferRuntime::MteMappingErrors() const
{
    RequireShmem();
    TORCH_CHECK(data_op_engine_type_ == ACLSHMEM_DATA_OP_MTE, "mapping preflight requires MTE-only transport");
    std::vector<std::string> errors;
    for (int peer = 0; peer < num_ranks_; ++peer) {
        const auto check = [&](void* local, std::size_t bytes, const char* region) {
            const auto error = MteMappingError(aclshmem_ptr(local, peer), bytes);
            if (!error.empty()) {
                errors.push_back("peer " + std::to_string(peer) + " " + region + ": " + error);
            }
        };
        check(comm_meta_ptr_, kCommMetaBytes, "metadata");
        if (num_bytes_ > 0) {
            check(workspace_ptr_, static_cast<std::size_t>(num_bytes_), "workspace");
        }
    }
    return errors;
}

void BufferRuntime::ReleaseResources(bool synchronize)
{
    if (!initialized_) {
        return;
    }
    if (synchronize && shmem_ != nullptr) {
        shmem_->Barrier();
    }
    FreeResources();
    initialized_ = false;
    shmem_.reset();
}

int64_t BufferRuntime::CommMetaPtr() const
{
    if (!IsInitialized() || comm_meta_ptr_ == nullptr) {
        throw std::runtime_error("communication metadata is not available");
    }
    return reinterpret_cast<int64_t>(comm_meta_ptr_);
}

int64_t BufferRuntime::WorkspacePtr() const
{
    if (!IsInitialized() || (num_bytes_ > 0 && workspace_ptr_ == nullptr)) {
        throw std::runtime_error("communication workspace is not available");
    }
    return reinterpret_cast<int64_t>(workspace_ptr_);
}

int64_t BufferRuntime::Rank() const
{
    return rank_;
}

int64_t BufferRuntime::NumRanks() const
{
    return num_ranks_;
}

int64_t BufferRuntime::NumBytes() const
{
    return num_bytes_;
}

int64_t BufferRuntime::LocalMemSize() const
{
    return local_mem_size_;
}

int64_t BufferRuntime::NumDeviceAiv() const
{
    return num_device_aiv_;
}

std::string BufferRuntime::Transport() const
{
    return transport_;
}

int64_t BufferRuntime::DataOpEngineType() const
{
    return data_op_engine_type_;
}

bool BufferRuntime::LowLatencyMode() const
{
    return low_latency_mode_;
}

bool BufferRuntime::Use1gHugePages() const
{
    return use_1g_huge_pages_;
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor> BufferRuntime::GetDispatchLayout(
    torch::Tensor topk_idx, int64_t num_experts, c10::optional<int64_t> allocation_stream_id)
{
    return ascend_deepep::ops::GetDispatchLayout(topk_idx, num_experts, static_cast<int>(rank_),
                                                 static_cast<int>(num_ranks_), static_cast<int>(num_device_aiv_),
                                                 RequireShmem().FftsAddr(), allocation_stream_id);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
           std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BufferRuntime::IntranodeDispatch(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
    c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
    torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert, int64_t cached_num_recv_tokens,
    c10::optional<torch::Tensor> cached_rank_prefix_matrix, c10::optional<torch::Tensor> cached_channel_prefix_matrix,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy, c10::optional<int64_t> allocation_stream_id,
    c10::optional<int64_t> num_max_tokens_per_rank, int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit)
{
    auto result = IntranodeDispatchWithPlan(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        cached_num_recv_tokens, cached_rank_prefix_matrix, cached_channel_prefix_matrix, expert_alignment,
        num_worst_tokens, num_aiv, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, zero_copy,
        allocation_stream_id, num_max_tokens_per_rank, udma_peers_per_worker, udma_wqes_per_visit);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result), std::get<4>(result),
            std::get<5>(result), std::get<6>(result), std::get<7>(result), std::get<8>(result), std::get<9>(result)};
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
           std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
BufferRuntime::IntranodeDispatchWithPlan(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
    c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
    torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert, int64_t cached_num_recv_tokens,
    c10::optional<torch::Tensor> cached_rank_prefix_matrix, c10::optional<torch::Tensor> cached_channel_prefix_matrix,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy, c10::optional<int64_t> allocation_stream_id,
    c10::optional<int64_t> num_max_tokens_per_rank, int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit,
    c10::optional<torch::Tensor> cached_dispatch_plan)
{
    TORCH_CHECK(num_aiv <= num_device_aiv_, "dispatch num_aiv must not exceed the device AIV core count ",
                num_device_aiv_, ", got ", num_aiv);
    auto result = ascend_deepep::ops::V1Dispatch(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        cached_num_recv_tokens, cached_rank_prefix_matrix, cached_channel_prefix_matrix, expert_alignment,
        num_worst_tokens, num_aiv, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, zero_copy,
        static_cast<int>(rank_), static_cast<int>(num_ranks_),
        static_cast<std::size_t>(num_bytes_ > 0 ? num_bytes_ : kCommMetaBytes), notify_workspace_bytes_,
        symmetric_workspace_active_, (data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0, RequireShmem(),
        KernelWorkspace(), allocation_stream_id, num_max_tokens_per_rank, udma_peers_per_worker, udma_wqes_per_visit,
        cached_dispatch_plan, nullptr, false, dispatch_output_control_);
    symmetric_workspace_active_ = true;
    return result;
}

std::tuple<std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
                      c10::optional<torch::Tensor>, std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>,
           std::vector<torch::Tensor>>
BufferRuntime::IntranodeDispatchWithHandle(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
    c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
    torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert, int64_t cached_num_recv_tokens,
    c10::optional<torch::Tensor> cached_rank_prefix_matrix, c10::optional<torch::Tensor> cached_channel_prefix_matrix,
    int64_t expert_alignment, int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
    int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy, c10::optional<int64_t> allocation_stream_id,
    c10::optional<int64_t> num_max_tokens_per_rank, int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit,
    c10::optional<torch::Tensor> cached_dispatch_plan, bool defer_public_expert_prefix)
{
    TORCH_CHECK(num_aiv <= num_device_aiv_, "dispatch num_aiv must not exceed the device AIV core count ",
                num_device_aiv_, ", got ", num_aiv);
    std::vector<torch::Tensor> public_handle;
    auto result = ascend_deepep::ops::V1Dispatch(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        cached_num_recv_tokens, cached_rank_prefix_matrix, cached_channel_prefix_matrix, expert_alignment,
        num_worst_tokens, num_aiv, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, zero_copy,
        static_cast<int>(rank_), static_cast<int>(num_ranks_),
        static_cast<std::size_t>(num_bytes_ > 0 ? num_bytes_ : kCommMetaBytes), notify_workspace_bytes_,
        symmetric_workspace_active_, (data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0, RequireShmem(),
        KernelWorkspace(), allocation_stream_id, num_max_tokens_per_rank, udma_peers_per_worker, udma_wqes_per_visit,
        cached_dispatch_plan, &public_handle, defer_public_expert_prefix, dispatch_output_control_);
    symmetric_workspace_active_ = true;
    return {result, public_handle};
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> BufferRuntime::IntranodeCombine(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
    c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
    c10::optional<int64_t> allocation_stream_id, c10::optional<int64_t> num_max_tokens_per_rank,
    int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit)
{
    TORCH_CHECK(num_aiv <= num_device_aiv_, "combine num_aiv must not exceed the device AIV core count ",
                num_device_aiv_, ", got ", num_aiv);
    auto result = ascend_deepep::ops::V1Combine(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, num_aiv,
        num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, static_cast<int>(rank_),
        static_cast<int>(num_ranks_), static_cast<std::size_t>(num_bytes_ > 0 ? num_bytes_ : kCommMetaBytes),
        notify_workspace_bytes_, (data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0, RequireShmem(), KernelWorkspace(),
        allocation_stream_id, num_max_tokens_per_rank, udma_peers_per_worker, udma_wqes_per_visit);
    symmetric_workspace_active_ = true;
    return result;
}

c10::intrusive_ptr<ascend_deepep::ops::CombineEpilogue> BufferRuntime::BeginCombine(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
    c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
    c10::optional<int64_t> allocation_stream_id, c10::optional<int64_t> num_max_tokens_per_rank,
    int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit, c10::optional<torch::Tensor> expanded_x,
    c10::optional<torch::Tensor> single_source_slots, c10::optional<torch::Tensor> early_aggregate_map,
    int64_t pipeline_chunk_tokens)
{
    TORCH_CHECK(num_aiv <= num_device_aiv_, "combine num_aiv must not exceed the device AIV core count ",
                num_device_aiv_, ", got ", num_aiv);
    TORCH_CHECK(pipeline_chunk_tokens == 0 || pipeline_chunk_tokens == 32 || pipeline_chunk_tokens == 64,
                "aggregate pipeline chunk must be 0, 32 or 64");
    TORCH_CHECK(!pipeline_chunk_tokens || (RequireShmem().QpsPerWorker() == 1 && early_aggregate_map.has_value() &&
                                           (!RequireShmem().TokenSchedule() || (num_aiv == 32 && num_ranks_ <= 8)) &&
                                           (num_aiv == 32 || num_aiv == 64)),
                "aggregate pipeline requires expanded peer-schedule UDMA with 32/64 AIV");
    const auto original_values = topk_weights;
    std::function<void()> aggregate;
    if (early_aggregate_map.has_value()) {
        TORCH_CHECK((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0 && expanded_x.has_value() &&
                        single_source_slots.has_value(),
                    "early aggregate requires the expanded single-source UDMA path");
        TORCH_CHECK(x.is_same(*expanded_x), "early aggregate must consume the original expanded tensor");
        auto plan = ascend_deepep::ops::PrepareAggregate(x, c10::nullopt, topk_weights, *early_aggregate_map, num_aiv,
                                                         RequireShmem().FftsAddr(), allocation_stream_id,
                                                         single_source_slots, rank_prefix_matrix,
                                                         static_cast<int>(rank_), static_cast<int>(num_ranks_));
        x = plan.x;
        topk_weights = plan.values;
        if (!pipeline_chunk_tokens) aggregate = std::move(plan.launch);
    }
    c10::intrusive_ptr<ascend_deepep::ops::CombineEpilogue> epilogue;
    ascend_deepep::ops::V1Combine(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, num_aiv,
        num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, static_cast<int>(rank_),
        static_cast<int>(num_ranks_), static_cast<std::size_t>(num_bytes_ > 0 ? num_bytes_ : kCommMetaBytes),
        notify_workspace_bytes_, (data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0, RequireShmem(), KernelWorkspace(),
        allocation_stream_id, num_max_tokens_per_rank, udma_peers_per_worker, udma_wqes_per_visit, &epilogue,
        expanded_x, single_source_slots, aggregate, pipeline_chunk_tokens ? early_aggregate_map : c10::nullopt,
        pipeline_chunk_tokens ? original_values : c10::nullopt, pipeline_chunk_tokens);
    symmetric_workspace_active_ = true;
    epilogue->SetOwner(c10::intrusive_ptr<BufferRuntime>::reclaim_copy(this));
    return epilogue;
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
           torch::Tensor>
BufferRuntime::Expand(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
                      c10::optional<torch::Tensor> topk_weights, std::vector<int64_t> num_tokens_per_expert,
                      int64_t expert_alignment, int64_t cached_num_expanded_tokens,
                      c10::optional<torch::Tensor> cached_expert_offsets,
                      c10::optional<torch::Tensor> cached_actual_expert_counts,
                      c10::optional<torch::Tensor> cached_recv_to_expanded_idx, int64_t num_aiv,
                      c10::optional<int64_t> allocation_stream_id, c10::optional<torch::Tensor> public_expert_prefix,
                      c10::optional<torch::Tensor> public_src_metadata,
                      c10::optional<torch::Tensor> single_source_slots)
{
    TORCH_CHECK(num_aiv == 0 || num_aiv <= num_device_aiv_, "expand num_aiv must not exceed the device AIV core count ",
                num_device_aiv_, ", got ", num_aiv);
    return ascend_deepep::ops::Expand(x, x_scales, topk_idx, topk_weights, std::move(num_tokens_per_expert),
                                      expert_alignment, cached_num_expanded_tokens, cached_expert_offsets,
                                      cached_actual_expert_counts, cached_recv_to_expanded_idx, num_aiv,
                                      RequireShmem().FftsAddr(), allocation_stream_id, public_expert_prefix,
                                      public_src_metadata, single_source_slots);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> BufferRuntime::Aggregate(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> expanded_values,
    torch::Tensor recv_to_expanded_idx, int64_t num_aiv, c10::optional<int64_t> allocation_stream_id,
    c10::optional<torch::Tensor> single_source_slots, c10::optional<torch::Tensor> rank_prefix_matrix)
{
    TORCH_CHECK(num_aiv == 0 || num_aiv <= num_device_aiv_,
                "aggregate num_aiv must not exceed the device AIV core count ", num_device_aiv_, ", got ", num_aiv);
    return ascend_deepep::ops::Aggregate(x, topk_weights, expanded_values, recv_to_expanded_idx, num_aiv,
                                         RequireShmem().FftsAddr(), allocation_stream_id, single_source_slots,
                                         rank_prefix_matrix, static_cast<int>(rank_), static_cast<int>(num_ranks_));
}

void BufferRuntime::ElasticBarrier(int64_t num_aiv)
{
    // The reused V1 team barrier uses SHMEM sync-pool signals, not UDMA queues.
    // It is also valid for the shared-Jetty V1 backend of ElasticBuffer.
    TORCH_CHECK(num_aiv > 0 && num_aiv <= num_device_aiv_, "elastic barrier num_aiv must be in (0, ", num_device_aiv_,
                "]");
    LaunchElasticBarrierKernel(num_aiv, RequireShmem().FftsAddr());
}

ascend_deepep::ops::DupServerNotifyPrepareResult BufferRuntime::DupServerNotifyPrepare(
    torch::Tensor topk_idx, c10::optional<torch::Tensor> topk_weights, int64_t num_experts, int64_t token_capacity,
    int64_t ranks_per_server, int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy, bool enable_timer)
{
    TORCH_CHECK(!RequireShmem().SharedJetty(), "V2 kernels require an ElasticBuffer with private queues");
    TORCH_CHECK((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0,
                "duplicate-server notify requires transport='udma'");
    auto result = ascend_deepep::ops::DupServerNotifyPrepare(
        topk_idx, topk_weights, num_experts, token_capacity, ranks_per_server, num_tokens_per_chunk, num_aiv,
        gateway_policy, static_cast<int>(rank_), static_cast<int>(num_ranks_), static_cast<int>(num_device_aiv_),
        enable_timer, static_cast<std::size_t>(num_bytes_), RequireShmem(), KernelWorkspace());
    symmetric_workspace_active_ = true;
    return result;
}

void BufferRuntime::DupServerNotifyScaleOut(int64_t token_capacity, int64_t hidden, int64_t ranks_per_server,
                                            int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy)
{
    TORCH_CHECK(!RequireShmem().SharedJetty(), "V2 kernels require an ElasticBuffer with private queues");
    TORCH_CHECK((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0,
                "duplicate-server scale-out requires transport='udma'");
    ascend_deepep::ops::DupServerNotifyScaleOut(token_capacity, hidden, ranks_per_server, num_tokens_per_chunk, num_aiv,
                                                gateway_policy, static_cast<int>(rank_), static_cast<int>(num_ranks_),
                                                static_cast<int>(num_device_aiv_), static_cast<std::size_t>(num_bytes_),
                                                RequireShmem(), KernelWorkspace());
    symmetric_workspace_active_ = true;
}

torch::Tensor BufferRuntime::DupServerDispatchScaleOut(torch::Tensor x, torch::Tensor send_token_count,
                                                       torch::Tensor send_token_ids, torch::Tensor send_gateway_lane,
                                                       int64_t ranks_per_server,
                                                       int64_t max_inflight_same_frame_servers,
                                                       int64_t num_tokens_per_chunk, int64_t num_aiv, bool enable_timer,
                                                       int64_t gateway_policy)
{
    TORCH_CHECK(!RequireShmem().SharedJetty(), "V2 kernels require an ElasticBuffer with private queues");
    TORCH_CHECK((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0,
                "duplicate-server scale-out requires transport='udma'");
    auto gateway_x = ascend_deepep::ops::DupServerDispatchScaleOut(
        x, send_token_count, send_token_ids, send_gateway_lane, ranks_per_server, max_inflight_same_frame_servers,
        num_tokens_per_chunk, num_aiv, enable_timer, gateway_policy, static_cast<int>(rank_),
        static_cast<int>(num_ranks_), static_cast<int>(num_device_aiv_), static_cast<std::size_t>(num_bytes_),
        RequireShmem(), KernelWorkspace());
    symmetric_workspace_active_ = true;
    return gateway_x;
}

torch::Tensor BufferRuntime::DupServerDispatch(
    torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids, torch::Tensor send_gateway_lane,
    torch::Tensor recv_route_position, torch::Tensor recv_gateway_lane, torch::Tensor recv_src_rank,
    torch::Tensor recv_segment_offset, int64_t num_topk, int64_t ranks_per_server,
    int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk, int64_t num_aiv, bool use_urma_get,
    bool enable_timer, int64_t gateway_policy, int64_t remote_get_servers, bool remote_get_rx_tx_gate)
{
    TORCH_CHECK(!RequireShmem().SharedJetty(), "V2 kernels require an ElasticBuffer with private queues");
    TORCH_CHECK((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0,
                "duplicate-server dispatch requires transport='udma'");
    auto recv_x = ascend_deepep::ops::DupServerDispatch(
        x, send_token_count, send_token_ids, send_gateway_lane, recv_route_position, recv_gateway_lane, recv_src_rank,
        recv_segment_offset, num_topk, ranks_per_server, max_inflight_same_frame_servers, num_tokens_per_chunk, num_aiv,
        use_urma_get, enable_timer, gateway_policy, remote_get_servers, remote_get_rx_tx_gate, static_cast<int>(rank_),
        static_cast<int>(num_ranks_), static_cast<int>(num_device_aiv_), static_cast<std::size_t>(num_bytes_),
        RequireShmem(), KernelWorkspace());
    symmetric_workspace_active_ = true;
    return recv_x;
}

void BufferRuntime::AllocateCommMeta()
{
    comm_meta_ptr_ = aclshmem_malloc(kCommMetaBytes);
    if (comm_meta_ptr_ == nullptr) {
        throw std::runtime_error("aclshmem_malloc failed for the 2 MiB communication metadata slot");
    }
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(comm_meta_ptr_, kCommMetaBytes, 0, kCommMetaBytes));
    if (transport_ == "udma") {
        const auto bytes = ascend_deepep::udma::OutputControlBytes(static_cast<uint32_t>(num_ranks_));
        dispatch_output_control_ = aclshmem_malloc(bytes);
        TORCH_CHECK(dispatch_output_control_ != nullptr, "aclshmem_malloc failed for dispatch output addresses");
        ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(dispatch_output_control_, bytes, 0, bytes));
    }
}

void BufferRuntime::AllocateWorkspace()
{
    if (num_bytes_ == 0) {
        return;
    }
    if (use_1g_huge_pages_) {
        // This is the translated symmetric address of descriptor 0, not the caller-mapped source VA.
        workspace_ptr_ = shmem_->WorkspacePtr();
        return;
    }
    workspace_ptr_ = aclshmem_malloc(static_cast<uint64_t>(num_bytes_));
    if (workspace_ptr_ == nullptr) {
        throw std::runtime_error("aclshmem_malloc failed for the requested communication workspace");
    }
}

void BufferRuntime::FreeResources() noexcept
{
    if (workspace_ptr_ != nullptr) {
        if (!use_1g_huge_pages_) {
            aclshmem_free(workspace_ptr_);
        }
        workspace_ptr_ = nullptr;
    }
    if (comm_meta_ptr_ != nullptr) {
        if (dispatch_output_control_ != nullptr) {
            aclshmem_free(dispatch_output_control_);
            dispatch_output_control_ = nullptr;
        }
        aclshmem_free(comm_meta_ptr_);
        comm_meta_ptr_ = nullptr;
    }
}

void BufferRuntime::DestroyNoThrow() noexcept
{
    try {
        ReleaseResources(false);
    } catch (...) {
        // Destructors must not throw while Python or torch is finalizing.
    }
}

ShmemInstance& BufferRuntime::RequireShmem() const
{
    if (!IsInitialized() || shmem_ == nullptr) {
        throw std::runtime_error("Buffer runtime is not initialized");
    }
    return *shmem_;
}

void* BufferRuntime::KernelWorkspace() const
{
    if (!IsInitialized()) {
        throw std::runtime_error("Buffer runtime is not initialized");
    }
    if (workspace_ptr_ != nullptr) {
        return workspace_ptr_;
    }
    if (comm_meta_ptr_ != nullptr) {
        return comm_meta_ptr_;
    }
    throw std::runtime_error("communication workspace is not available");
}

}  // namespace ascend_deepep::runtime
