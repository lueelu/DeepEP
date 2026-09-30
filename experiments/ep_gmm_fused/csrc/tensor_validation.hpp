// Copyright (c) 2026, Lu Lu
// Modified by zhu-mingzhe71 2026

#ifndef DEEPEP_EP_GMM_FUSED_TENSOR_VALIDATION_HPP
#define DEEPEP_EP_GMM_FUSED_TENSOR_VALIDATION_HPP

#include "tiling_validation.hpp"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

namespace deepep::ep_gmm_fused {

// Call after dtype/device/contiguity checks. ATen compares the occupied byte
// intervals of contiguous views, including views with different storage offsets.
inline void CheckNoOverlap(const at::Tensor& output, const at::Tensor& input, const char* outputName,
                           const char* inputName)
{
    if (output.numel() == 0 || input.numel() == 0) {
        return;
    }
    TORCH_CHECK(at::get_overlap_status(output, input) == at::MemOverlapStatus::No, outputName, " must not overlap ",
                inputName);
}

inline at::Tensor GroupedMatMulAllToAllVUdmaMeta(const at::Tensor& x, const at::Tensor& weight, const at::Tensor&,
                                                 const at::Tensor&, int64_t epSize, int64_t expertNum,
                                                 at::IntArrayRef tiling)
{
    if (!tiling.empty()) {
        TilingValidation::GroupedMatMul parsed{};
        const char* error = TilingValidation::ParseGroupedMatMul(tiling.data(), tiling.size(), parsed);
        TORCH_CHECK(error == nullptr, "invalid GroupedMatMul tiling: ", error);
    }
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(false,
                "gmm_alltoallv has a route-dependent output shape; Meta/FakeTensor and torch.compile "
                "require out= with shape [sum(local_tokens_per_expert), n]");
}

inline at::Tensor GroupedMatMulAllToAllVUdmaOutMeta(const at::Tensor& x, const at::Tensor& weight, const at::Tensor&,
                                                    const at::Tensor&, int64_t epSize, int64_t expertNum,
                                                    const at::Tensor& output, at::IntArrayRef tiling)
{
    if (!tiling.empty()) {
        TilingValidation::GroupedMatMul parsed{};
        const char* error = TilingValidation::ParseGroupedMatMul(tiling.data(), tiling.size(), parsed);
        TORCH_CHECK(error == nullptr, "invalid GroupedMatMul tiling: ", error);
    }
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == weight.size(2), "invalid out shape");
    return output;
}

}  // namespace deepep::ep_gmm_fused
#endif
