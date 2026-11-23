// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <torch/extension.h>
#include "host_utils.hpp"
#include "mega_moe_quantized_tiling.hpp"
#include "mega_moe_bf16_tiling.hpp"

namespace MegaMoeBF16Training {
struct TrainingSavedStateLayout;
}

namespace deepep::megamoe {

uint32_t InferenceMaxRecvTokens(uint32_t max_tokens, uint32_t world, uint32_t topk, uint32_t local_experts);
void FillPeerAddresses(uint64_t* addresses, void* window, int64_t world);

// 一个 BS 对应一个准备好的计划；Python owner 保证同流串行复用及销毁顺序。
class Plan {
public:
    Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world, int64_t max_tokens,
         bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts, uint64_t symmetric_address,
         int64_t symmetric_bytes);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out);

    int64_t workspace_bytes() const
    {
        return workspace_.numel();
    }

private:
    PlatformInfo platform_;
    MegaMoeTilingData tiling_;
    torch::Tensor workspace_;
    torch::Tensor context_;
    uint64_t ffts_;
};

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden);

}  // namespace deepep::megamoe

namespace deepep::megamoe::bf16 {

struct ForwardResult {
    torch::Tensor y;
    torch::Tensor counts;
    torch::Tensor saved_state;
    torch::Tensor dispatch_output;
    torch::Tensor timer;
};

struct BackwardResult {
    torch::Tensor dx;
    torch::Tensor dtopk_weights;
    torch::Tensor dw1;
    torch::Tensor dw2;
    torch::Tensor timer;
};

// Borrows the root Session window; Python controls its collective lifetime.
class BF16Plan {
public:
    BF16Plan(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like, bool weight_before_gmm2);

    void close();
    void prepare(int64_t m, int64_t h, int64_t f, int64_t topk, int64_t local_experts, int64_t max_recv, double limit,
                 const std::string& save_type, bool fp32_p);
    int64_t workspace_bytes() const
    {
        return workspace_.defined() ? workspace_.numel() : 0;
    }

    ForwardResult forward(torch::Tensor out, torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
                          torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens, double swiglu_limit,
                          const std::string& save_gmm1_output, bool capture_timer, bool save_dispatched_x = true);

    BackwardResult backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                            torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                            torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                            const std::string& save_gmm1_output, bool capture_timer);

private:
    struct WeightLayout {
        int64_t n;
        bool transW;
    };
    static uint32_t TopkWeightType(torch::ScalarType dtype);
    static MegaMoeBF16Training::TrainingSavedStateLayout MakeSavedStateLayout(const MegaMoeBf16TilingData& tiling,
                                                                              int64_t k, int64_t n,
                                                                              int64_t max_recv_tokens);
    static WeightLayout ResolveWeightLayout(const torch::Tensor& w1, const torch::Tensor& w2, int64_t k);
    void ClearWindowControl() const;
    torch::Tensor AllocateTimer(const torch::Tensor& like, const torch::Tensor& weights, const torch::Tensor& w1,
                                const torch::Tensor& w2, int64_t max_recv, int phase, const std::string& mode);
    MegaMoeBf16TilingData TilingForCall(int64_t m, int64_t k, int64_t n, int64_t topk, int64_t experts,
                                        int64_t max_recv, double limit, const std::string& mode, uint32_t weight_type,
                                        bool trans_w);
    PlatformInfo platform_;
    MegaMoeBf16TilingData tiling_{};
    torch::Tensor Context(const torch::Tensor& like);

    void* SymmetricBuffer() const;
    std::size_t SymmetricBufferBytes() const;
    uintptr_t address_;
    int64_t bytes_;
    int rank_;
    int world_size_;
    uint64_t ffts_addr_;
    torch::Tensor context_;
    torch::Tensor workspace_;
    bool weight_before_gmm2_;

    int context_rank_ = -1;
    int context_world_size_ = -1;
};

int64_t buffer_size(int64_t max_tokens, int64_t hidden, int64_t intermediate, int64_t topk, int64_t world,
                    int64_t local_experts, int64_t max_recv, bool need_grad);
}  // namespace deepep::megamoe::bf16
