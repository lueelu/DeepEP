// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "plan.hpp"

#include <limits>

#include <ATen/MemoryOverlap.h>
#include "kernel_launch.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"

namespace deepep::megamoe {

std::tuple<torch::Tensor, torch::Tensor> Plan::forward(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
    c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
    c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
    c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out)
{
    CheckTensor(x, 2, "x");
    CheckTensor(topk_ids, 2, "topk_ids");
    CheckTensor(topk_weights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(weight_scales1, 4, "weight_scales1");
    CheckTensor(weight_scales2, 4, "weight_scales2");
    const bool prequantized = x_scales.has_value();
    TORCH_CHECK(x.scalar_type() == (prequantized ? c10::ScalarType::Float8_e4m3fn : torch::kBFloat16),
                "x must be BF16, or FP8 E4M3 with E8M0 scales");
    if (prequantized) {
        CheckTensor(*x_scales, 2, "x_scales");
        TORCH_CHECK(x_scales->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        x_scales->sizes() == torch::IntArrayRef({x.size(0), CeilDiv(x.size(1), int64_t{32})}),
                    "x_scales must be E8M0 [BS, H/32]");
    }
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be float32");
    const bool is_a8w4 = weight1.scalar_type() == torch::kUInt8;
    if (timer.has_value()) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before passing timer");
#endif
        TORCH_CHECK(is_a8w4, "MegaMoE timer is supported only by the A8W4 wave kernel");
        CheckTensor(*timer, 1, "timer");
        TORCH_CHECK(timer->scalar_type() == torch::kInt64, "timer must be int64");
        TORCH_CHECK(timer->numel() == AscendTimer::TOTAL_BUFFER_SIZE, "timer must contain exactly ",
                    AscendTimer::TOTAL_BUFFER_SIZE, " int64 values");
    }
    TORCH_CHECK(weight2.scalar_type() == weight1.scalar_type(), "weight1 and weight2 must use the same dtype");
    TORCH_CHECK(is_a8w4 || weight1.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "weight1/weight2 must be packed E2M1 uint8 (A8W4) or FP8 E4M3 (A8W8)");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                "weight scales must be FP8 E8M0 tensors");
    for (const auto* tensor : {&topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2}) {
        TORCH_CHECK(tensor->device() == x.device(), "MegaMoE tensors must be on the same NPU");
    }
    for (const auto* tensor : {&x_scales, &timer, &out}) {
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == x.device(),
                    "optional MegaMoE tensors must be on the same NPU");
    }
    const int64_t bs = x.size(0), h = x.size(1), topk = topk_ids.size(1);
    const int64_t experts = weight1.size(0), hidden = weight1.size(1);
    TORCH_CHECK(bs > 0 && h >= 1024 && h <= 8192 && h % 32 == 0, "invalid x shape for A5 MegaMoE");
    TORCH_CHECK(topk >= 1 && topk <= 32 && topk_ids.sizes() == topk_weights.sizes() && topk_ids.size(0) == bs,
                "top-k tensors must have matching [bs, topk] shapes");
    TORCH_CHECK(hidden >= 512 && hidden <= 8192 && hidden % 256 == 0, "invalid weight1 hiddenDim");
    const int64_t weight1_last = is_a8w4 ? h / 2 : h;
    const int64_t weight2_last = is_a8w4 ? hidden / 4 : hidden / 2;
    TORCH_CHECK(weight1.size(2) == weight1_last,
                is_a8w4 ? "packed A8W4 weight1 shape must be [local_experts, hiddenDim, h/2]"
                        : "A8W8 weight1 shape must be [local_experts, hiddenDim, h]");
    TORCH_CHECK(weight2.sizes() == torch::IntArrayRef({experts, h, weight2_last}),
                is_a8w4 ? "packed A8W4 weight2 shape must be [local_experts, h, hiddenDim/4]"
                        : "A8W8 weight2 shape must be [local_experts, h, hiddenDim/2]");
    TORCH_CHECK(weight_scales1.sizes() == torch::IntArrayRef({experts, hidden, CeilDiv(h, int64_t{64}), 2}),
                "weight_scales1 shape must be [local_experts, hiddenDim, ceil(h/64), 2]");
    TORCH_CHECK(weight_scales2.sizes() == torch::IntArrayRef({experts, h, CeilDiv(hidden / 2, int64_t{64}), 2}),
                "weight_scales2 shape must be [local_experts, h, ceil(hiddenDim/2/64), 2]");
    TORCH_CHECK(x.device() == workspace_.device() && bs == tiling_.bs && h == tiling_.h &&
                    hidden == tiling_.hiddenDim && topk == tiling_.topK && experts == tiling_.moeExpertPerRank &&
                    is_a8w4 == (tiling_.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4),
                "Input/device/weights differ from the prepared plan");

    const bool has_shared = shared_weight1.has_value();
    TORCH_CHECK(shared_weight2.has_value() == has_shared && shared_weight_scales1.has_value() == has_shared &&
                    shared_weight_scales2.has_value() == has_shared,
                "all four shared weight and scale tensors must be supplied together");
    int64_t shared_experts = 0;
    if (has_shared) {
        TORCH_CHECK(is_a8w4, "shared experts currently require A8W4 weights");
        CheckTensor(*shared_weight1, 3, "shared_weight1");
        CheckTensor(*shared_weight2, 3, "shared_weight2");
        CheckTensor(*shared_weight_scales1, 4, "shared_weight_scales1");
        CheckTensor(*shared_weight_scales2, 4, "shared_weight_scales2");
        shared_experts = shared_weight1->size(0);
        TORCH_CHECK(shared_experts > 0 && shared_experts <= std::numeric_limits<uint32_t>::max(),
                    "shared expert count must be positive and fit uint32");
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            TORCH_CHECK(tensor->device() == x.device(), "shared tensors must be on the same device as x");
        }
        TORCH_CHECK(shared_weight1->scalar_type() == torch::kUInt8 && shared_weight2->scalar_type() == torch::kUInt8,
                    "shared weights must use packed E2M1 uint8 NZ storage");
        TORCH_CHECK(shared_weight_scales1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales2->scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                    "shared weight scales must be FP8 E8M0");
        TORCH_CHECK(shared_weight1->sizes() == torch::IntArrayRef({shared_experts, hidden, h / 2}) &&
                        shared_weight2->sizes() == torch::IntArrayRef({shared_experts, h, hidden / 4}),
                    "shared weights must match routed H and hiddenDim: [S, hiddenDim, H/2], [S, H, hiddenDim/4]");
        TORCH_CHECK(shared_weight_scales1->sizes() ==
                            torch::IntArrayRef({shared_experts, hidden, CeilDiv(h, int64_t{64}), 2}) &&
                        shared_weight_scales2->sizes() ==
                            torch::IntArrayRef({shared_experts, h, CeilDiv(hidden / 2, int64_t{64}), 2}),
                    "shared scales must match [S, N, ceil(K/64), 2] for each shared weight");
    }

    TORCH_CHECK(shared_experts == tiling_.sharedExpertNum, "Shared experts differ from the prepared plan");
    torch::Tensor y = out.has_value() ? *out : torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    CheckTensor(y, 2, "out");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && y.sizes() == x.sizes(), "out must be BF16 [BS, H]");
    torch::Tensor expert_tokens = torch::empty({experts}, x.options().dtype(torch::kInt32));

    const auto stream = c10_npu::getCurrentNPUStream();
    auto ptr = [](const torch::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    auto opt = [&](const c10::optional<torch::Tensor>& tensor) { return tensor ? ptr(*tensor) : nullptr; };
    mega_moe_kernel_do(platform_.block_dim, platform_.ub, stream.stream(), is_a8w4, ptr(context_), ptr(x),
                       opt(x_scales), ptr(topk_ids), ptr(topk_weights), ptr(weight1), ptr(weight2), ptr(weight_scales1),
                       ptr(weight_scales2), opt(shared_weight1), opt(shared_weight2), opt(shared_weight_scales1),
                       opt(shared_weight_scales2), ptr(y), ptr(expert_tokens), ptr(workspace_), opt(timer),
                       reinterpret_cast<const uint8_t*>(&tiling_), ffts_);
    // The direct kernel launch bypasses the PyTorch dispatcher; record that
    // every argument remains live until the current NPU stream reaches it.
    for (const auto* tensor : {&x, &topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2,
                               &workspace_, &context_, &y, &expert_tokens}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(timer->storage().data_ptr(), stream);
    }
    if (has_shared) {
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
        }
    }
    if (x_scales) {
        c10_npu::NPUCachingAllocator::recordStream(x_scales->storage().data_ptr(), stream);
    }
    return {y, expert_tokens};
}

}  // namespace deepep::megamoe

namespace deepep::megamoe::bf16 {
namespace {

torch::Tensor AllocateTrainingSavedState(const MegaMoeBF16Training::TrainingSavedStateLayout& savedStateLayout,
                                         const torch::Tensor& like)
{
    // The compact ordinary-BF16 layout begins at cumsumOffset=0. Embedded
    // layouts still skip their ptrA payload here. Keep only
    // the metadata/control portion zero-initialized; gate/up planes are written
    // by every valid forward tile.
    const uint64_t initOffset = savedStateLayout.cumsumOffset;
    const uint64_t initBytes = savedStateLayout.legacyBytes - initOffset;
    TORCH_CHECK(savedStateLayout.ptrAOffset <= initOffset && initOffset <= savedStateLayout.legacyBytes &&
                    savedStateLayout.legacyBytes <= savedStateLayout.totalBytes &&
                    savedStateLayout.totalBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "invalid MegaMoe BF16 training saved-state initialization range");

    auto savedState =
        torch::empty({static_cast<int64_t>(savedStateLayout.totalBytes)}, like.options().dtype(torch::kUInt8));
    savedState.narrow(0, static_cast<int64_t>(initOffset), static_cast<int64_t>(initBytes)).zero_();
    return savedState;
}

}  // namespace

ForwardResult BF16Plan::forward(torch::Tensor out, torch::Tensor x, torch::Tensor topkIds, torch::Tensor topkWeights,
                                torch::Tensor weight1, torch::Tensor weight2, int64_t maxRecvTokens, double swigluLimit,
                                const std::string& save_gmm1_output, bool captureTimer, bool saveDispatchedX)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(out.defined(), "y must be a defined output tensor");
    const torch::Tensor timer =
        captureTimer ? AllocateTimer(x, topkWeights, weight1, weight2, maxRecvTokens, 0, save_gmm1_output)
                     : torch::Tensor();
    auto context = Context(x);
    CheckTensor(x, 2, "x");
    CheckTensor(topkIds, 2, "topk_ids");
    CheckTensor(topkWeights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topkIds.scalar_type() == torch::kInt32, "topk_ids must be int32");
    const auto topkWeightType = TopkWeightType(topkWeights.scalar_type());
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(), "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0), "x and topk_ids must have the same token count");

    CheckTensor(out, 2, "y");
    TORCH_CHECK(out.scalar_type() == x.scalar_type() && out.device() == x.device() && out.sizes() == x.sizes(),
                "y must have the same BF16 shape and NPU device as x");
    at::assert_no_internal_overlap(out);
    for (const auto* input : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2}) {
        at::assert_no_overlap(out, *input);
    }

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveWeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    const auto tiling =
        TilingForCall(m, k, n, topk, expertPerRank, maxRecvTokens, swigluLimit,
                      weight_before_gmm2_ ? save_gmm1_output : "none", topkWeightType, weightLayout.transW);

    const auto savedStateLayout = MakeSavedStateLayout(tiling, k, n, maxRecvTokens);
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    // Both unpermute implementations overwrite every output token row once.
    // ExpertTokenNumCopyOut likewise writes all local-expert entries, so these
    // returned tensors do not need a device-wide pre-clear.
    auto expertTokenNums = torch::empty({expertPerRank}, topkIds.options().dtype(torch::kInt32));
    // Inference uses only the working route indices and prefixes up to this offset.
    auto routingState = weight_before_gmm2_
                            ? AllocateTrainingSavedState(savedStateLayout, x)
                            : torch::zeros({static_cast<int64_t>(savedStateLayout.savedTokenCountsOffset)},
                                           x.options().dtype(torch::kUInt8));
    torch::Tensor dispatchOutput;
    if (weight_before_gmm2_ && saveDispatchedX) {
        dispatchOutput =
            torch::empty({static_cast<int64_t>(dispatchOutputLayout.totalBytes)}, x.options().dtype(torch::kUInt8));
    }
    ClearWindowControl();

    const auto stream = c10_npu::getCurrentNPUStream();
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    const uint32_t block_dim = platform_.aic;
    if (timer.defined()) {
        mega_moe_bf16_forward_timer_kernel_do(block_dim, stream.stream(), ptr(context), ptr(x), ptr(topkIds),
                                              ptr(topkWeights), ptr(weight1), ptr(weight2), ptr(out),
                                              ptr(expertTokenNums), ptr(routingState),
                                              dispatchOutput.defined() ? ptr(dispatchOutput) : nullptr, ptr(workspace_),
                                              reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_, ptr(timer));
    } else {
        mega_moe_bf16_forward_kernel_do(block_dim, stream.stream(), ptr(context), ptr(x), ptr(topkIds),
                                        ptr(topkWeights), ptr(weight1), ptr(weight2), ptr(out), ptr(expertTokenNums),
                                        ptr(routingState), dispatchOutput.defined() ? ptr(dispatchOutput) : nullptr,
                                        ptr(workspace_), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_);
    }
    for (const auto* tensor : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2, &out, &expertTokenNums,
                               &routingState, &dispatchOutput, &workspace_}) {
        if (tensor->defined()) c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {out, expertTokenNums, weight_before_gmm2_ ? routingState : torch::Tensor(), dispatchOutput, timer};
}

}  // namespace deepep::megamoe::bf16
