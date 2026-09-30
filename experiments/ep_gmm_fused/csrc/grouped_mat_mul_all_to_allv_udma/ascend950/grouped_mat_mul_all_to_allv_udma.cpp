/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_mat_mul_all_to_allv_udma_device.hpp"
#include "../../all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.hpp"
#include "../../tiling_validation.hpp"
#include "../../tensor_validation.hpp"

#include "catccos/dgemm/device/kernel_adapter.hpp"
#include "platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace deepep::ep_gmm_fused {
namespace {

enum class KernelVariant {
    kM0_128,
    kM0_256,
};

template <class T>
struct TypeTag {
    using type = T;
};

struct TilingProfile {
    KernelVariant kernelVariant;
    uint32_t m0;
    uint32_t n0;
    uint32_t k0;
    uint32_t commInterval;
    uint32_t commTileM;
    uint32_t commBlockM;
    uint32_t commNpuSplit;
    uint32_t commDataSplit;
};

constexpr TilingProfile kDefaultTiling{KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16};
// 2026-09-09 Ascend950DT_9582 search winners, with commTileM = 2 * local copy rows.
// K1024: f1d10672a756a4a9; K512: fdfb758ae9b102f9.
constexpr TilingProfile kK1024N2048Tiling{KernelVariant::kM0_128, 128, 256, 256, 10, 8, 64, 4, 4};
constexpr TilingProfile kK512N2048Tiling{KernelVariant::kM0_128, 128, 256, 256, 16, 32, 16, 1, 20};

const TilingProfile& SelectTiling(uint32_t k, uint32_t n, uint32_t worldSize, uint32_t expertNum)
{
    // The local GMM input rows equal sum(global_tokens) and may differ across
    // ranks under imbalanced routing. UDMA peers must nevertheless use the same
    // M0 and communication schedule, so selection may only depend on topology
    // and K/N, which are rank-invariant for one collective launch.
    const bool isEightRank256ExpertCase = worldSize == 8 && expertNum == 256;
    if (isEightRank256ExpertCase && k == 1024 && n == 2048) {
        return kK1024N2048Tiling;
    }
    if (isEightRank256ExpertCase && k == 512 && n == 2048) {
        return kK512N2048Tiling;
    }
    return kDefaultTiling;
}

TilingProfile ResolveTiling(const TilingProfile& defaultTiling, at::IntArrayRef values)
{
    if (values.empty()) {
        return defaultTiling;
    }
    TilingValidation::GroupedMatMul parsed{};
    const char* error = TilingValidation::ParseGroupedMatMul(values.data(), values.size(), parsed);
    TORCH_CHECK(error == nullptr, "invalid GroupedMatMul tiling: ", error);
    TilingProfile result = defaultTiling;
    result.kernelVariant = parsed.m0 == 128 ? KernelVariant::kM0_128 : KernelVariant::kM0_256;
    result.m0 = parsed.m0;
    result.n0 = parsed.n0;
    result.k0 = parsed.k0;
    result.commInterval = parsed.commInterval;
    result.commTileM = 2 * parsed.localCopyTileRows;
    return result;
}

constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kMaxLocalExperts = 32;
constexpr uint32_t kBf16UbAlignElements = 16;

uint32_t TargetOrderLocalCopyColumns(uint32_t columns)
{
    const uint64_t alignedColumns =
        (static_cast<uint64_t>(columns) + kBf16UbAlignElements - 1) / kBf16UbAlignElements * kBf16UbAlignElements;
    return alignedColumns <= std::numeric_limits<uint32_t>::max() ? static_cast<uint32_t>(alignedColumns) : 0;
}

void CheckNpuTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <class Kernel, class Params>
void LaunchGroupedMatMulAllToAllVUdma(const Params& params, uint32_t blockDim, aclrtStream stream, uint64_t fftsAddress)
{
    if (fftsAddress == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, fftsAddress, nullptr);
    }
}

at::Tensor GroupedMatMulAllToAllVUdmaImpl(const at::Tensor& x, const at::Tensor& weight,
                                          const at::Tensor& localTokensPerExpert,
                                          const at::Tensor& globalTokensPerLocalExpert, int64_t epSize,
                                          int64_t expertNum, const c10::optional<at::Tensor>& outputOptional,
                                          at::IntArrayRef tilingValues)
{
    const c10::OptionalDeviceGuard guard(x.device());
    CheckNpuTensor(x, "x", at::kBFloat16);
    CheckNpuTensor(weight, "weight", at::kBFloat16);
    CheckNpuTensor(localTokensPerExpert, "local_tokens_per_expert", at::kLong);
    CheckNpuTensor(globalTokensPerLocalExpert, "global_tokens_per_local_expert", at::kLong);
    TORCH_CHECK(weight.device() == x.device(), "weight and x must be on the same NPU");
    TORCH_CHECK(localTokensPerExpert.device() == x.device() && globalTokensPerLocalExpert.device() == x.device(),
                "routing tables and x must be on the same NPU");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [local_expert_num, k, n]");

    UdmaRuntime& runtime = UdmaRuntime::Instance();
    runtime.EnsureInitialized();
    runtime.CheckDevice(x.device().index());
    const int64_t worldSize = runtime.WorldSize();
    TORCH_CHECK(epSize == worldSize, "TP=1 is required: ep_size must equal UDMA world_size");
    TORCH_CHECK(expertNum > 0 && expertNum % epSize == 0, "expert_num must be positive and divisible by ep_size");
    const int64_t localExpertNum = expertNum / epSize;
    TORCH_CHECK(runtime.Rank() < epSize, "UDMA rank must be smaller than ep_size");
    TORCH_CHECK(localExpertNum <= kMaxLocalExperts, "local expert count exceeds ", kMaxLocalExperts);

    const int64_t inputRows64 = x.size(0);
    const int64_t k64 = x.size(1);
    const int64_t n64 = weight.size(2);
    TORCH_CHECK(inputRows64 > 0 && k64 > 0 && n64 > 0, "input rows, k and n must be positive");
    TORCH_CHECK(inputRows64 <= std::numeric_limits<uint32_t>::max() && k64 <= std::numeric_limits<uint32_t>::max() &&
                    n64 <= std::numeric_limits<uint32_t>::max(),
                "shape exceeds the kernel's uint32 range");
    TORCH_CHECK(weight.size(0) == localExpertNum && weight.size(1) == k64,
                "weight must have shape [expert_num / ep_size, k, n]");
    TORCH_CHECK(localTokensPerExpert.numel() == expertNum, "local_tokens_per_expert must contain expert_num elements");
    TORCH_CHECK(globalTokensPerLocalExpert.numel() == worldSize * localExpertNum,
                "global_tokens_per_local_expert must contain world_size * local_expert_num elements");

    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(blockDim > 0, "failed to query Ascend AI-core count");
    TORCH_CHECK(static_cast<int64_t>(blockDim) >= worldSize,
                "target-order UDMA staging requires at least one AI core per rank");
    const uint32_t m = static_cast<uint32_t>(inputRows64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const TilingProfile tiling = ResolveTiling(
        SelectTiling(k, n, static_cast<uint32_t>(worldSize), static_cast<uint32_t>(expertNum)), tilingValues);
    uint64_t ubBytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t localCopyColumns = TargetOrderLocalCopyColumns(n);
    // Only the built-in fallback is clamped. Explicit profiles must fit exactly.
    const uint32_t localCopyRows =
        tilingValues.empty() ? std::min(tiling.commTileM / 2, TilingValidation::LocalCopyRowCapacity(n, ubBytes))
                             : tiling.commTileM / 2;
    TORCH_CHECK(localCopyRows > 0 && localCopyColumns > 0, "N is too large for target-order local MTE copy staging");
    const TilingValidation::GroupedMatMul requested{tiling.m0, tiling.n0, tiling.k0, tiling.commInterval,
                                                    localCopyRows};
    const char* error =
        TilingValidation::ValidateGroupedMatMul(requested, n, static_cast<uint32_t>(worldSize), blockDim, ubBytes);
    TORCH_CHECK(error == nullptr, "invalid GroupedMatMul tiling: ", error);
    TORCH_CHECK(inputRows64 <= std::numeric_limits<uint32_t>::max() - tiling.m0 + 1,
                "input rows overflow the target-order tiling scheduler");
    TORCH_CHECK((static_cast<uint64_t>(blockDim) * tiling.commInterval) % worldSize == 0,
                "AI-core count * comm_interval must be divisible by world_size");

    // Target-order staging packs all N tiles of each selected M group into a
    // contiguous [rows, N] buffer, allowing one UDMA request per source rank
    // and communication round.
    const uint64_t blockPerComm = static_cast<uint64_t>(blockDim) * tiling.commInterval;
    const uint64_t tilePerCommInRank = blockPerComm / static_cast<uint64_t>(worldSize);
    const uint64_t nLoops = (static_cast<uint64_t>(n64) + tiling.n0 - 1) / tiling.n0;
    TORCH_CHECK(tilePerCommInRank >= nLoops,
                "target-order UDMA staging requires AI-core count * comm_interval / "
                "world_size >= ceil(n / N0)");
    const uint64_t groupsPerCommInRank = std::max<uint64_t>(1, tilePerCommInRank / nLoops);
    const uint64_t stageRows = groupsPerCommInRank * tiling.m0;
    const uint64_t requiredWorkspaceBytes = static_cast<uint64_t>(kWorkspaceStages) * stageRows *
                                            static_cast<uint64_t>(worldSize) * static_cast<uint64_t>(n64) *
                                            sizeof(uint16_t);
    TORCH_CHECK(requiredWorkspaceBytes <= GroupedMatMulAllToAllVUdmaDefaultConfig::kSymmetricDataBytes,
                "kernel tiling requires ", requiredWorkspaceBytes, " symmetric workspace bytes, exceeding ",
                GroupedMatMulAllToAllVUdmaDefaultConfig::kSymmetricDataBytes);

    // This fused direction consumes sum(global_tokens) rows and returns
    // sum(local_tokens) rows. Use the caller-provided capacity on the reusable
    // out path to avoid a device-to-host reduction in the hot launch path.
    int64_t outputRows64;
    at::Tensor output;
    if (outputOptional.has_value()) {
        output = outputOptional.value();
        TORCH_CHECK(output.dim() == 2 && output.size(1) == n64,
                    "out must have shape [sum(local_tokens_per_expert), n]");
        outputRows64 = output.size(0);
    } else {
        outputRows64 = localTokensPerExpert.sum().item<int64_t>();
        TORCH_CHECK(outputRows64 >= 0, "sum(local_tokens_per_expert) must be non-negative");
        output = at::empty({outputRows64, n64}, x.options());
    }
    TORCH_CHECK(outputRows64 <= std::numeric_limits<uint32_t>::max(),
                "output row count exceeds the kernel's uint32 range");
    CheckNpuTensor(output, "out", at::kBFloat16);
    TORCH_CHECK(output.device() == x.device(), "out and x must be on the same NPU");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == n64, "out must have shape [sum(local_tokens_per_expert), n]");
    CheckNoOverlap(output, x, "out", "x");
    CheckNoOverlap(output, weight, "out", "weight");
    CheckNoOverlap(output, localTokensPerExpert, "out", "local_tokens_per_expert");
    CheckNoOverlap(output, globalTokensPerLocalExpert, "out", "global_tokens_per_local_expert");

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    runtime.RecordStream(stream);
    Catlass::GemmCoord shape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{tiling.commDataSplit, tiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{tiling.commBlockM, static_cast<uint32_t>(n64)};
    Catlass::MatrixCoord commTileShape{localCopyRows, localCopyColumns};
    const uint64_t fftsAddress = runtime.FftsAddress();
    auto launchWithConfig = [&](auto configTag) {
        using SelectedConfig = typename decltype(configTag)::type;
        using SelectedDeviceOp = typename SelectedConfig::Device;
        using SelectedKernel = typename SelectedConfig::Kernel;
        TORCH_CHECK(
            tiling.m0 == SelectedConfig::kM0 && tiling.n0 == SelectedConfig::kN0 && tiling.k0 == SelectedConfig::kK0,
            "tiling profile does not match the selected kernel instance");
        typename SelectedDeviceOp::Arguments arguments{
            shape,
            static_cast<uint32_t>(runtime.Rank()),
            static_cast<uint32_t>(worldSize),
            tiling.commInterval,
            static_cast<uint32_t>(epSize),
            static_cast<uint32_t>(expertNum),
            reinterpret_cast<uint8_t*>(x.data_ptr()),
            reinterpret_cast<uint8_t*>(weight.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            reinterpret_cast<uint8_t*>(localTokensPerExpert.data_ptr()),
            reinterpret_cast<uint8_t*>(globalTokensPerLocalExpert.data_ptr()),
            runtime.Workspace(),
            commCoreSplit,
            commBlockShape,
            commTileShape,
            static_cast<uint32_t>(outputRows64),
        };

        SelectedDeviceOp deviceOp;
        const Catlass::Status initializeStatus = deviceOp.Initialize(arguments);
        TORCH_CHECK(initializeStatus == Catlass::Status::kSuccess, "failed to initialize the CatCCOS device operator");
        const typename SelectedDeviceOp::Params params = deviceOp.params();
        auto launch = [=]() -> int {
            LaunchGroupedMatMulAllToAllVUdma<SelectedKernel>(params, blockDim, stream, fftsAddress);
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("GroupedMatMulAllToAllVUdma", launch);
    };
    if (tiling.kernelVariant == KernelVariant::kM0_256) {
        launchWithConfig(TypeTag<GroupedMatMulAllToAllVUdmaM0_256Config>{});
    } else {
        launchWithConfig(TypeTag<GroupedMatMulAllToAllVUdmaDefaultConfig>{});
    }
    return output;
}

at::Tensor GroupedMatMulAllToAllVUdmaNpu(const at::Tensor& x, const at::Tensor& weight,
                                         const at::Tensor& localTokensPerExpert,
                                         const at::Tensor& globalTokensPerLocalExpert, int64_t epSize,
                                         int64_t expertNum, at::IntArrayRef tiling)
{
    return GroupedMatMulAllToAllVUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize,
                                          expertNum, c10::nullopt, tiling);
}

at::Tensor GroupedMatMulAllToAllVUdmaOutNpu(const at::Tensor& x, const at::Tensor& weight,
                                            const at::Tensor& localTokensPerExpert,
                                            const at::Tensor& globalTokensPerLocalExpert, int64_t epSize,
                                            int64_t expertNum, const at::Tensor& output, at::IntArrayRef tiling)
{
    return GroupedMatMulAllToAllVUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize,
                                          expertNum, output, tiling);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "grouped_mat_mul_all_to_allv_udma(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int[] tiling=[]) -> Tensor");
    m.def(
        "grouped_mat_mul_all_to_allv_udma_out(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, "
        "Tensor(a!) out, int[] tiling=[]) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("grouped_mat_mul_all_to_allv_udma", GroupedMatMulAllToAllVUdmaNpu);
    m.impl("grouped_mat_mul_all_to_allv_udma_out", GroupedMatMulAllToAllVUdmaOutNpu);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("grouped_mat_mul_all_to_allv_udma", GroupedMatMulAllToAllVUdmaMeta);
    m.impl("grouped_mat_mul_all_to_allv_udma_out", GroupedMatMulAllToAllVUdmaOutMeta);
}

}  // namespace deepep::ep_gmm_fused
