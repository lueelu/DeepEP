/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/custom_class.h>
#include <torch/extension.h>

#include "csrc/shmem_instance.h"
#include "csrc/ops/dup_server.h"
#include "csrc/ops/combine.h"

namespace ascend_deepep::runtime {

class BufferRuntime : public torch::jit::CustomClassHolder {
public:
    BufferRuntime(int64_t rank, int64_t num_ranks, int64_t num_bytes, const std::string& transport,
                  bool low_latency_mode, const std::string& ip_port, bool use_1g_huge_pages,
                  const std::vector<int64_t>& unique_id, const std::string& bootstrap_key, bool v1_shared_jetty,
                  bool v1_token_schedule = false, int64_t udma_dispatch_peer_order = 0,
                  int64_t udma_combine_peer_order = 0, int64_t udma_ranks_per_group = 0,
                  int64_t udma_peers_per_group_per_worker = 0, int64_t udma_workers_per_peer = 1,
                  int64_t udma_qps_per_worker = 1);

    ~BufferRuntime() noexcept;

    BufferRuntime(const BufferRuntime&) = delete;
    BufferRuntime& operator=(const BufferRuntime&) = delete;

    bool IsInitialized() const;
    void Barrier();
    void Destroy();
    void AbortInitialization();
    std::vector<std::string> MteMappingErrors() const;
    int64_t CommMetaPtr() const;
    int64_t WorkspacePtr() const;
    int64_t Rank() const;
    int64_t NumRanks() const;
    int64_t NumBytes() const;
    int64_t LocalMemSize() const;
    int64_t NumDeviceAiv() const;
    std::string Transport() const;
    int64_t DataOpEngineType() const;
    bool LowLatencyMode() const;
    bool Use1gHugePages() const;

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor> GetDispatchLayout(
        torch::Tensor topk_idx, int64_t num_experts, c10::optional<int64_t> allocation_stream_id = c10::nullopt);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
               std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    IntranodeDispatch(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
                      c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
                      torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert,
                      int64_t cached_num_recv_tokens, c10::optional<torch::Tensor> cached_rank_prefix_matrix,
                      c10::optional<torch::Tensor> cached_channel_prefix_matrix, int64_t expert_alignment,
                      int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
                      int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy,
                      c10::optional<int64_t> allocation_stream_id = c10::nullopt,
                      c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt, int64_t udma_peers_per_worker = 4,
                      int64_t udma_wqes_per_visit = 8);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
               std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               c10::optional<torch::Tensor>>
    IntranodeDispatchWithPlan(torch::Tensor x, c10::optional<torch::Tensor> x_scales,
                              c10::optional<torch::Tensor> topk_idx, c10::optional<torch::Tensor> topk_weights,
                              c10::optional<torch::Tensor> num_tokens_per_rank, torch::Tensor is_token_in_rank,
                              c10::optional<torch::Tensor> num_tokens_per_expert, int64_t cached_num_recv_tokens,
                              c10::optional<torch::Tensor> cached_rank_prefix_matrix,
                              c10::optional<torch::Tensor> cached_channel_prefix_matrix, int64_t expert_alignment,
                              int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
                              int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy,
                              c10::optional<int64_t> allocation_stream_id = c10::nullopt,
                              c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt,
                              int64_t udma_peers_per_worker = 4, int64_t udma_wqes_per_visit = 8,
                              c10::optional<torch::Tensor> cached_dispatch_plan = c10::nullopt);

    std::tuple<std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
                          c10::optional<torch::Tensor>, std::vector<int64_t>, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>,
               std::vector<torch::Tensor>>
    IntranodeDispatchWithHandle(torch::Tensor x, c10::optional<torch::Tensor> x_scales,
                                c10::optional<torch::Tensor> topk_idx, c10::optional<torch::Tensor> topk_weights,
                                c10::optional<torch::Tensor> num_tokens_per_rank, torch::Tensor is_token_in_rank,
                                c10::optional<torch::Tensor> num_tokens_per_expert, int64_t cached_num_recv_tokens,
                                c10::optional<torch::Tensor> cached_rank_prefix_matrix,
                                c10::optional<torch::Tensor> cached_channel_prefix_matrix, int64_t expert_alignment,
                                int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
                                int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy,
                                c10::optional<int64_t> allocation_stream_id = c10::nullopt,
                                c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt,
                                int64_t udma_peers_per_worker = 4, int64_t udma_wqes_per_visit = 8,
                                c10::optional<torch::Tensor> cached_dispatch_plan = c10::nullopt,
                                bool defer_public_expert_prefix = false);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>> IntranodeCombine(
        torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
        c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
        torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
        int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
        c10::optional<int64_t> allocation_stream_id = c10::nullopt,
        c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt, int64_t udma_peers_per_worker = 4,
        int64_t udma_wqes_per_visit = 8);

    c10::intrusive_ptr<ascend_deepep::ops::CombineEpilogue> BeginCombine(
        torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
        c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
        torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
        int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens,
        c10::optional<int64_t> allocation_stream_id = c10::nullopt,
        c10::optional<int64_t> num_max_tokens_per_rank = c10::nullopt, int64_t udma_peers_per_worker = 4,
        int64_t udma_wqes_per_visit = 8, c10::optional<torch::Tensor> expanded_x = c10::nullopt,
        c10::optional<torch::Tensor> single_source_slots = c10::nullopt,
        c10::optional<torch::Tensor> early_aggregate_map = c10::nullopt, int64_t pipeline_chunk_tokens = 0);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Expand(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
           c10::optional<torch::Tensor> topk_weights, std::vector<int64_t> num_tokens_per_expert,
           int64_t expert_alignment, int64_t cached_num_expanded_tokens,
           c10::optional<torch::Tensor> cached_expert_offsets, c10::optional<torch::Tensor> cached_actual_expert_counts,
           c10::optional<torch::Tensor> cached_recv_to_expanded_idx, int64_t num_aiv,
           c10::optional<int64_t> allocation_stream_id = c10::nullopt,
           c10::optional<torch::Tensor> public_expert_prefix = c10::nullopt,
           c10::optional<torch::Tensor> public_src_metadata = c10::nullopt,
           c10::optional<torch::Tensor> single_source_slots = c10::nullopt);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Aggregate(
        torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> expanded_values,
        torch::Tensor recv_to_expanded_idx, int64_t num_aiv, c10::optional<int64_t> allocation_stream_id = c10::nullopt,
        c10::optional<torch::Tensor> single_source_slots = c10::nullopt,
        c10::optional<torch::Tensor> rank_prefix_matrix = c10::nullopt);

    void ElasticBarrier(int64_t num_aiv);

    ascend_deepep::ops::DupServerNotifyPrepareResult DupServerNotifyPrepare(
        torch::Tensor topk_idx, c10::optional<torch::Tensor> topk_weights, int64_t num_experts, int64_t token_capacity,
        int64_t ranks_per_server, int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy,
        bool enable_timer);

    void DupServerNotifyScaleOut(int64_t token_capacity, int64_t hidden, int64_t ranks_per_server,
                                 int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy);

    torch::Tensor DupServerDispatchScaleOut(torch::Tensor x, torch::Tensor send_token_count,
                                            torch::Tensor send_token_ids, torch::Tensor send_gateway_lane,
                                            int64_t ranks_per_server, int64_t max_inflight_same_frame_servers,
                                            int64_t num_tokens_per_chunk, int64_t num_aiv, bool enable_timer,
                                            int64_t gateway_policy);

    torch::Tensor DupServerDispatch(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                    torch::Tensor send_gateway_lane, torch::Tensor recv_route_position,
                                    torch::Tensor recv_gateway_lane, torch::Tensor recv_src_rank,
                                    torch::Tensor recv_segment_offset, int64_t num_topk, int64_t ranks_per_server,
                                    int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk,
                                    int64_t num_aiv, bool use_urma_get, bool enable_timer, int64_t gateway_policy,
                                    int64_t remote_get_servers, bool remote_get_rx_tx_gate);

private:
    void AllocateCommMeta();
    void AllocateWorkspace();
    void FreeResources() noexcept;
    void ReleaseResources(bool synchronize);
    void DestroyNoThrow() noexcept;
    ShmemInstance& RequireShmem() const;
    void* KernelWorkspace() const;

    int64_t rank_;
    int64_t num_ranks_;
    int64_t num_bytes_;
    int64_t local_mem_size_;
    std::string transport_;
    int64_t data_op_engine_type_;
    bool low_latency_mode_;
    bool use_1g_huge_pages_;
    bool initialized_{false};
    int64_t num_device_aiv_{0};
    std::size_t notify_workspace_bytes_{0};
    bool symmetric_workspace_active_{false};

    std::shared_ptr<ShmemInstance> shmem_;
    void* comm_meta_ptr_{nullptr};
    void* workspace_ptr_{nullptr};
    void* dispatch_output_control_{nullptr};
    static constexpr uint64_t kCommMetaBytes = 2ULL * 1024ULL * 1024ULL;
};

}  // namespace ascend_deepep::runtime
