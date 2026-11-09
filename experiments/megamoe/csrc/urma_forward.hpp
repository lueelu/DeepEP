// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <torch/extension.h>
#include <map>
#include <string>
#include "forward.hpp"
#include "kernels/ubx/urma/abi.hpp"

namespace deepep::megamoe {
int64_t urma_buffer_size(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                         int64_t cv_rows);

class UrmaPlan {
public:
    UrmaPlan(torch::Tensor example, int64_t world, int64_t intermediate, int64_t topk, int64_t cv_rows,
             int64_t max_recv_rows, int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address,
             int64_t symmetric_bytes, int64_t gmm1_ahead_mgroups, int64_t shared_experts = 0,
             int64_t shared_intermediate = 0, int64_t num_experts = 384);
    std::vector<torch::Tensor> routing_tensors(int64_t bs) const;
    torch::Tensor expert_counts() const
    {
        return expert_counts_;
    }
    std::vector<int64_t> route_stats() const;  // explicit post-forward diagnostic D2H only
    torch::Tensor forward(torch::Tensor x, torch::Tensor x_scales, torch::Tensor topk_idx, torch::Tensor topk_weights,
                          torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1,
                          torch::Tensor weight_scales2, c10::optional<torch::Tensor> timer,
                          c10::optional<torch::Tensor> out, int64_t epoch,
                          c10::optional<torch::Tensor> trace = c10::nullopt, bool coarse_trace = false,
                          c10::optional<torch::Tensor> shared_w1 = c10::nullopt,
                          c10::optional<torch::Tensor> shared_w2 = c10::nullopt,
                          c10::optional<torch::Tensor> shared_s1 = c10::nullopt,
                          c10::optional<torch::Tensor> shared_s2 = c10::nullopt);
    int64_t workspace_bytes() const;
    // Explicit diagnostic only: synchronize before copying owned intermediates.
    std::map<std::string, torch::Tensor> debug_snapshot() const;

private:
    PlatformInfo platform_{};
    MegaMoeUrma::Arguments args_{};
    torch::Tensor expanded_x_, expanded_scale_, gmm1_, activation_, activation_scale_, gmm2_;
    torch::Tensor gathered_input_scale_, input_scale_done_;
    torch::Tensor gmm_issued_, gmm2_contiguous_, peer_reorder_ranges_, peer_reorder_spans_;
    torch::Tensor fragments_, send_chunks_, recv_chunks_, gmm1_done_, activation_done_, gmm2_done_, local_flags_;
    torch::Tensor segments_, compute_batches_, compute_tasks_, chunk_group_indices_;
    torch::Tensor pack_rounds_, pack_ready_;
    torch::Tensor reduce_dependencies_, order_, inverse_, route_counts_, route_offsets_;
    torch::Tensor route_histogram_, route_info_, route_config_, expert_routes_;
    torch::Tensor system_workspace_, expert_counts_;
    torch::Tensor shared_gmm1_, shared_activation_, shared_scale_, shared_output_, shared_routed_output_, shared_done1_,
        shared_done_act_;
    int64_t last_epoch_ = 0;
    int64_t max_tokens_ = 0;
};
}  // namespace deepep::megamoe
