// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include <tuple>
#include <memory>
#include <string>
#include "kernel_launch.hpp"
#include <torch/extension.h>
#include "kernels/mega_moe/mega_moe_tiling.hpp"

namespace deepep::megamoe {

struct PlatformInfo {
    uint32_t aic;
    uint32_t aiv;
    uint32_t ub;
    uint32_t block_dim;
    uint32_t system_workspace;
};

struct UbxWorkspace;

// One Host plan; backend selects private resource layout and the device entry.
class Plan {
public:
    Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world, int64_t max_tokens,
         bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts, uint64_t symmetric_address,
         int64_t symmetric_bytes, const std::string& backend = "pod");
    ~Plan();

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out, int64_t generation = 0);

    int64_t workspace_bytes() const;

private:
    Backend backend_;
    std::unique_ptr<UbxWorkspace> ubx_;
    PlatformInfo platform_{};
    MegaMoeTilingData tiling_;
    torch::Tensor workspace_;
    torch::Tensor context_;
    uint64_t ffts_;
    torch::Tensor example_;
    int64_t intermediate_, topk_, experts_, world_, max_tokens_, shared_experts_;
    bool is_a8w4_;
    uint64_t symmetric_address_;
    int64_t symmetric_bytes_;
    int64_t last_generation_ = 0;
    int64_t shared_intermediate_ = 0;
};

torch::Tensor allocate_timer(torch::Tensor example, const std::string& backend = "pod");

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden,
                    const std::string& backend = "pod");

}  // namespace deepep::megamoe
