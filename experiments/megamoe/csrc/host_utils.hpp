// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

// Host-side shared helpers for MegaMoE quantized and BF16 paths.
// Device/kernel code must not include this header.

#pragma once

#include <cstdint>

#include <torch/extension.h>
#include "tiling/platform/platform_ascendc.h"

namespace deepep::megamoe {

struct PlatformInfo {
    uint32_t aic;
    uint32_t aiv;
    uint32_t ub;
    uint32_t block_dim;
    uint32_t system_workspace;
};

template <typename T, typename U>
constexpr auto CeilDiv(T value, U divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T, typename U>
constexpr auto AlignUp(T value, U alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

inline PlatformInfo GetPlatformInfo()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get AscendC platform information");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic > 0 && aiv > 0 && ub > 0, "invalid A5 platform resources");
    return {aic, aiv, static_cast<uint32_t>(ub), platform->CalcTschBlockDim(aiv, aic, aiv),
            platform->GetLibApiWorkSpaceSize()};
}

inline void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void CheckTensor(const torch::Tensor& t, int64_t dims, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    CheckNpuContiguous(t, name);
    TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "D");
}

}  // namespace deepep::megamoe
