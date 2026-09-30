/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "all_to_allv_grouped_mat_mul_udma_device.hpp"
#include "udma_runtime.hpp"
#include "../../tiling_validation.hpp"
#include "../../tensor_validation.hpp"

#include "catccos/dgemm/device/kernel_adapter.hpp"
#include "catccos/dgemm/moe_route_metadata.hpp"
#include "platform/platform_ascendc.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace deepep::ep_gmm_fused {
namespace {

constexpr uint32_t kMaxLocalExperts = 64;
constexpr uint64_t kSymmetricDataBytes = 200UL * 1024 * 1024 * sizeof(uint16_t);
constexpr uint32_t kMaxK0 = 256;
static_assert(sizeof(Catccos::DGemm::MoeRouteSegment) == 24, "unexpected MoeRouteSegment ABI");
static_assert(sizeof(Catccos::DGemm::MoeRouteMetadata) == 12880, "unexpected MoeRouteMetadata ABI");

enum class KernelVariant {
    kM0_128,
    kM0_256,
    kM0_256N0_256,
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
    // These two values are retained from the tuning record. The migrated
    // CatCCOS UDMA kernel ABI does not consume either legacy host field.
    uint32_t commNpuSplit;
    uint32_t commDataSplit;
    // Physical pipeline controls introduced by CatCCOS 54522865. Keeping
    // these explicit decouples communication capacity from legacy tiling
    // fields and lets workspace depth/worker pools evolve independently.
    uint32_t stageRows;
    uint32_t workspaceStages;
    uint32_t localCopyTileRows;
    uint32_t mSplit;
    uint32_t exportCoreCount;
    uint32_t selfCopyCores;
};

constexpr TilingProfile kDefaultTiling{KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16, 1280, 2, 32, 1, 0, 1};
constexpr TilingProfile kM131072K2048N1024NonExportTiling{
    KernelVariant::kM0_128, 128, 256, 256, 12, 128, 128, 4, 4, 1536, 2, 64, 1, 0, 1};
constexpr TilingProfile kM131072K2048N1024ExportTiling{
    KernelVariant::kM0_256N0_256, 256, 256, 256, 5, 32, 32, 1, 16, 1280, 3, 16, 1, 24, 8};
constexpr TilingProfile kM131072K2048N512NonExportTiling{
    KernelVariant::kM0_256, 256, 128, 256, 6, 128, 128, 4, 4, 1536, 2, 64, 1, 0, 1};
constexpr TilingProfile kM131072K2048N512ExportTiling{
    KernelVariant::kM0_256, 256, 128, 256, 5, 32, 64, 1, 16, 1280, 2, 8, 1, 0, 8};

// Export profiles use the 2026-09-09 Ascend950DT_9582 search winners:
//   N1024: 986c5d67aef6c9a2; N512: 55f6ea82e55c35f2.
// The search execution mode is not part of the migrated kernel ABI.
// commNpuSplit/commDataSplit remain recorded for traceability but are not
// consumed by this operator's current host/kernel ABI.

const TilingProfile& SelectTiling(uint32_t m, uint32_t k, uint32_t n, uint32_t worldSize, uint32_t expertNum,
                                  bool outputAllToAllV)
{
    // Every rank participating in one collective launch must select the same
    // communication geometry. In an imbalanced route, received_rows is local
    // to a rank and may differ, so it must not participate in tiling selection.
    // This operator requires the input M/K/N and the export mode to be
    // rank-invariant; those values are therefore safe selection keys.
    const bool isEightRank256ExpertCase = worldSize == 8 && expertNum == 256;
    if (isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 1024) {
        return outputAllToAllV ? kM131072K2048N1024ExportTiling : kM131072K2048N1024NonExportTiling;
    }
    if (isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 512) {
        return outputAllToAllV ? kM131072K2048N512ExportTiling : kM131072K2048N512NonExportTiling;
    }
    return kDefaultTiling;
}

TilingProfile ResolveTiling(const TilingProfile& defaultTiling, at::IntArrayRef values, bool exportOutput)
{
    if (values.empty()) {
        return defaultTiling;
    }
    TilingValidation::AllToAllV parsed{};
    const char* error = TilingValidation::ParseAllToAllV(values.data(), values.size(), exportOutput, parsed);
    TORCH_CHECK(error == nullptr, "invalid AllToAllV tiling: ", error);
    TilingProfile result = defaultTiling;
    result.kernelVariant = parsed.m0 == 128
                               ? KernelVariant::kM0_128
                               : (parsed.n0 == 128 ? KernelVariant::kM0_256 : KernelVariant::kM0_256N0_256);
    result.m0 = parsed.m0;
    result.n0 = parsed.n0;
    result.k0 = parsed.k0;
    result.commBlockM = parsed.commBlockM;
    result.stageRows = parsed.stageRows;
    result.commInterval = parsed.stageRows / parsed.m0;
    result.workspaceStages = parsed.workspaceStages;
    result.localCopyTileRows = parsed.localCopyTileRows;
    result.mSplit = parsed.mSplit;
    result.exportCoreCount = parsed.exportCoreCount;
    result.selfCopyCores = parsed.selfCopyCores;
    return result;
}

std::vector<int64_t> TilingHardware(const at::Tensor& deviceAnchor)
{
    TORCH_CHECK(deviceAnchor.defined() && deviceAnchor.device().type() == c10::DeviceType::PrivateUse1,
                "device_anchor must be an NPU tensor");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t coreCount = platform->GetCoreNumAic();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(coreCount > 0 && ubBytes > TilingValidation::kWqeScratchBytes &&
                    ubBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "failed to query Ascend AI-core count and UB capacity");
    return {static_cast<int64_t>(coreCount), static_cast<int64_t>(ubBytes)};
}

uint32_t RoundUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void CheckNpuTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void AttachUdma(const at::Tensor& deviceAnchor, int64_t rank, int64_t worldSize, int64_t workspaceAddress,
                int64_t workspaceBytes)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8 && rank >= 0 && rank < worldSize, "invalid UDMA topology");
    TORCH_CHECK(workspaceAddress > 0 && workspaceBytes > 0, "invalid public runtime workspace");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    UdmaRuntime::Instance().Attach(static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
                                   static_cast<uintptr_t>(workspaceAddress), static_cast<uint64_t>(workspaceBytes));
}

void DetachUdma(const at::Tensor& deviceAnchor)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    UdmaRuntime::Instance().Detach(stream);
}

template <class Kernel, class Params>
void LaunchAllToAllVGroupedMatMulUdma(const Params& params, uint32_t blockDim, aclrtStream stream, uint64_t fftsAddress)
{
    // Match CatCCOS DeviceDGemm::Run: zero selects the adapter overload that
    // does not configure an explicit FFTS synchronization base address.
    if (fftsAddress == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, fftsAddress, nullptr);
    }
}

template <class Config>
void InitializeAndLaunch(const TilingProfile& tiling, const Catlass::GemmCoord& shape, uint32_t rank,
                         uint32_t worldSize, uint32_t epSize, uint32_t expertNum, uint32_t outputRows, uint8_t* x,
                         uint8_t* weight, uint8_t* output, uint8_t* localTokensPerExpert,
                         uint8_t* globalTokensPerLocalExpert, uint8_t* workspace, uint8_t* allToAllVOutput,
                         uint8_t* routeMetadata, uint32_t blockDim, aclrtStream stream, uint64_t fftsAddress)
{
    using DeviceOp = typename Config::Device;
    using Kernel = typename Config::Kernel;
    TORCH_CHECK(tiling.m0 == Config::kM0 && tiling.n0 == Config::kN0 && tiling.k0 == Config::kK0 &&
                    tiling.workspaceStages == Config::kWorkspaceStages && tiling.mSplit == Config::kMSplit,
                "tiling profile does not match the selected kernel instance");
    Catlass::MatrixCoord commBlockShape{tiling.commBlockM, RoundUp(shape.k(), tiling.k0)};
    Catlass::MatrixCoord commTileShape{tiling.localCopyTileRows, tiling.k0};
    typename DeviceOp::Arguments arguments{
        shape,
        rank,
        worldSize,
        tiling.commInterval,
        epSize,
        expertNum,
        outputRows,
        x,
        weight,
        output,
        localTokensPerExpert,
        globalTokensPerLocalExpert,
        workspace,
        commBlockShape,
        commTileShape,
        allToAllVOutput,
        // Host route inputs supply precomputed prefixes through this pointer.
        // Device-only routes keep it null and use the kernel's fallback,
        // avoiding a synchronous route-table D2H copy.
        routeMetadata,
        tiling.stageRows,
        tiling.exportCoreCount,
        tiling.selfCopyCores,
    };

    DeviceOp deviceOp;
    const Catlass::Status initializeStatus = deviceOp.Initialize(arguments);
    TORCH_CHECK(initializeStatus == Catlass::Status::kSuccess, "failed to initialize the CatCCOS device operator");
    const typename DeviceOp::Params params = deviceOp.params();
    auto launch = [=]() -> int {
        LaunchAllToAllVGroupedMatMulUdma<Kernel>(params, blockDim, stream, fftsAddress);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AllToAllVGroupedMatMulUdma", launch);
}

at::Tensor AllToAllVGroupedMatMulUdmaImpl(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& localTokensPerExpert,
    const at::Tensor& globalTokensPerLocalExpert, int64_t epSize, int64_t expertNum, int64_t receivedRows64,
    const c10::optional<at::Tensor>& outputOptional, const c10::optional<at::Tensor>& allToAllVOutputOptional,
    const c10::optional<at::Tensor>& routeMetadataOptional, at::IntArrayRef tilingValues)
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

    const int64_t m64 = x.size(0);
    const int64_t k64 = x.size(1);
    const int64_t n64 = weight.size(2);
    TORCH_CHECK(m64 > 0 && k64 > 0 && n64 > 0, "m, k and n must be positive");
    TORCH_CHECK(m64 <= std::numeric_limits<uint32_t>::max() &&
                    k64 <= std::numeric_limits<uint32_t>::max() - (kMaxK0 - 1) &&
                    n64 <= std::numeric_limits<uint32_t>::max(),
                "shape exceeds the kernel's uint32 range");
    TORCH_CHECK(weight.size(0) == localExpertNum && weight.size(1) == k64,
                "weight must have shape [expert_num / ep_size, k, n]");
    TORCH_CHECK(localTokensPerExpert.numel() == expertNum, "local_tokens_per_expert must contain expert_num elements");
    TORCH_CHECK(globalTokensPerLocalExpert.numel() == worldSize * localExpertNum,
                "global_tokens_per_local_expert must contain world_size * local_expert_num elements");
    TORCH_CHECK(receivedRows64 >= 0 && receivedRows64 <= worldSize * m64,
                "received_rows must be in [0, world_size * m]");
    TORCH_CHECK(receivedRows64 <= std::numeric_limits<uint32_t>::max(),
                "received_rows exceeds the kernel's uint32 range");

    const uint32_t m = static_cast<uint32_t>(m64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t receivedRows = static_cast<uint32_t>(receivedRows64);
    const TilingProfile tiling =
        ResolveTiling(SelectTiling(m, k, n, static_cast<uint32_t>(worldSize), static_cast<uint32_t>(expertNum),
                                   allToAllVOutputOptional.has_value()),
                      tilingValues, allToAllVOutputOptional.has_value());
    {
        const auto hardware = TilingHardware(x);
        const TilingValidation::AllToAllV requested{tiling.m0,
                                                    tiling.n0,
                                                    tiling.k0,
                                                    tiling.commBlockM,
                                                    tiling.stageRows,
                                                    tiling.workspaceStages,
                                                    tiling.localCopyTileRows,
                                                    tiling.mSplit,
                                                    tiling.exportCoreCount,
                                                    tiling.selfCopyCores};
        const char* error =
            TilingValidation::ValidateAllToAllV(requested, m, k, static_cast<uint32_t>(worldSize),
                                                static_cast<uint32_t>(hardware[0]), static_cast<uint64_t>(hardware[1]));
        TORCH_CHECK(error == nullptr, "invalid AllToAllV tiling: ", error);
        error = TilingValidation::ValidateAllToAllVGrid(requested, n, static_cast<uint32_t>(worldSize),
                                                        static_cast<uint32_t>(hardware[0]));
        TORCH_CHECK(error == nullptr, "invalid AllToAllV tiling: ", error);
    }
    TORCH_CHECK((tiling.workspaceStages == 2 || tiling.workspaceStages == 3) && tiling.stageRows > 0 &&
                    tiling.stageRows % tiling.m0 == 0,
                "invalid CatCCOS physical staging profile");
    const auto workspaceLayout = Catccos::layout::DistRowMajor::MakeAlignedLayout<bfloat16_t>(
        Catlass::MatrixCoord{tiling.stageRows, k}, static_cast<uint32_t>(worldSize));
    const uint64_t requiredWorkspaceBytes =
        static_cast<uint64_t>(tiling.workspaceStages) * Catccos::layout::Capacity(workspaceLayout) * sizeof(uint16_t);
    TORCH_CHECK(requiredWorkspaceBytes <= kSymmetricDataBytes, "kernel tiling requires ", requiredWorkspaceBytes,
                " symmetric workspace bytes, exceeding ", kSymmetricDataBytes);

    at::Tensor output =
        outputOptional.has_value() ? outputOptional.value() : at::empty({receivedRows64, n64}, x.options());
    CheckNpuTensor(output, "out", at::kBFloat16);
    TORCH_CHECK(output.device() == x.device(), "out and x must be on the same NPU");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows64 && output.size(1) == n64,
                "out must have shape [received_rows, n]");
    CheckNoOverlap(output, x, "out", "x");
    CheckNoOverlap(output, weight, "out", "weight");
    CheckNoOverlap(output, localTokensPerExpert, "out", "local_tokens_per_expert");
    CheckNoOverlap(output, globalTokensPerLocalExpert, "out", "global_tokens_per_local_expert");

    at::Tensor allToAllVOutput;
    if (allToAllVOutputOptional.has_value()) {
        allToAllVOutput = allToAllVOutputOptional.value();
        CheckNpuTensor(allToAllVOutput, "all_to_allv_out", at::kBFloat16);
        TORCH_CHECK(allToAllVOutput.device() == x.device(), "all_to_allv_out and x must be on the same NPU");
        TORCH_CHECK(
            allToAllVOutput.dim() == 2 && allToAllVOutput.size(0) == receivedRows64 && allToAllVOutput.size(1) == k64,
            "all_to_allv_out must have shape [received_rows, k]");
        CheckNoOverlap(allToAllVOutput, x, "all_to_allv_out", "x");
        CheckNoOverlap(allToAllVOutput, weight, "all_to_allv_out", "weight");
        CheckNoOverlap(allToAllVOutput, output, "all_to_allv_out", "out");
        CheckNoOverlap(allToAllVOutput, localTokensPerExpert, "all_to_allv_out", "local_tokens_per_expert");
        CheckNoOverlap(allToAllVOutput, globalTokensPerLocalExpert, "all_to_allv_out",
                       "global_tokens_per_local_expert");
    }
    if (routeMetadataOptional.has_value()) {
        const at::Tensor& routeMetadata = routeMetadataOptional.value();
        TORCH_CHECK(allToAllVOutputOptional.has_value(), "route_metadata is only valid when exporting AllToAllV");
        CheckNpuTensor(routeMetadata, "route_metadata", at::kByte);
        TORCH_CHECK(routeMetadata.device() == x.device(), "route_metadata and x must be on the same NPU");
        TORCH_CHECK(routeMetadata.numel() == static_cast<int64_t>(sizeof(Catccos::DGemm::MoeRouteMetadata)),
                    "route_metadata has an invalid packed size");
        CheckNoOverlap(output, routeMetadata, "out", "route_metadata");
        CheckNoOverlap(allToAllVOutput, routeMetadata, "all_to_allv_out", "route_metadata");
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    runtime.RecordStream(stream);

    Catlass::GemmCoord shape{m, n, k};
    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(blockDim > 0, "failed to query Ascend AI-core count");
    const uint32_t exportCoreCount = tiling.exportCoreCount == 0 ? blockDim : tiling.exportCoreCount;
    TORCH_CHECK(exportCoreCount > 0 && exportCoreCount <= blockDim, "export_core_count must be in [1, AI-core count]");
    TORCH_CHECK(tiling.selfCopyCores > 0 && static_cast<uint64_t>(worldSize) + tiling.selfCopyCores - 1 <= blockDim,
                "self_copy_cores exceeds the available AIV worker pool");
    const uint64_t fftsAddress = runtime.FftsAddress();
    auto launchWithConfig = [&](auto configTag) {
        using SelectedConfig = typename decltype(configTag)::type;
        InitializeAndLaunch<SelectedConfig>(
            tiling, shape, static_cast<uint32_t>(runtime.Rank()), static_cast<uint32_t>(worldSize),
            static_cast<uint32_t>(epSize), static_cast<uint32_t>(expertNum), receivedRows,
            // UDMA PUT reads its local source directly from ordinary NPU GM.
            reinterpret_cast<uint8_t*>(x.data_ptr()), reinterpret_cast<uint8_t*>(weight.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()), reinterpret_cast<uint8_t*>(localTokensPerExpert.data_ptr()),
            reinterpret_cast<uint8_t*>(globalTokensPerLocalExpert.data_ptr()), runtime.Workspace(),
            allToAllVOutputOptional.has_value() ? reinterpret_cast<uint8_t*>(allToAllVOutput.data_ptr()) : nullptr,
            routeMetadataOptional.has_value() ? reinterpret_cast<uint8_t*>(routeMetadataOptional.value().data_ptr())
                                              : nullptr,
            blockDim, stream, fftsAddress);
    };
    auto launchByMSplit = [&](auto mSplit1, auto mSplit2, auto mSplit4) {
        if (tiling.mSplit == 2) {
            launchWithConfig(mSplit2);
        } else if (tiling.mSplit == 4) {
            launchWithConfig(mSplit4);
        } else {
            launchWithConfig(mSplit1);
        }
    };
    if (allToAllVOutputOptional.has_value()) {
        if (tiling.kernelVariant == KernelVariant::kM0_256N0_256) {
            if (tiling.workspaceStages == 3) {
                launchByMSplit(TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 1>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 2>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 4>>{});
            } else {
                launchByMSplit(TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 1>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 2>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 4>>{});
            }
        } else if (tiling.kernelVariant == KernelVariant::kM0_256) {
            if (tiling.workspaceStages == 3) {
                launchByMSplit(TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 1>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 2>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 4>>{});
            } else {
                launchByMSplit(TypeTag<AllToAllVGroupedMatMulUdmaM0_256ExportConfig>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 2, 2>>{},
                               TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 2, 4>>{});
            }
        } else if (tiling.workspaceStages == 3) {
            launchByMSplit(TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 1>>{},
                           TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 2>>{},
                           TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 4>>{});
        } else {
            launchByMSplit(TypeTag<AllToAllVGroupedMatMulUdmaDefaultExportConfig>{},
                           TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 2, 2>>{},
                           TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 2, 4>>{});
        }
    } else {
        TORCH_CHECK(tiling.workspaceStages == 2 && tiling.mSplit == 1,
                    "non-export kernel currently supports workspace_stages=2 and m_split=1");
        if (tiling.kernelVariant == KernelVariant::kM0_256N0_256) {
            launchWithConfig(TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256>>{});
        } else if (tiling.kernelVariant == KernelVariant::kM0_256) {
            launchWithConfig(TypeTag<AllToAllVGroupedMatMulUdmaM0_256Config>{});
        } else {
            launchWithConfig(TypeTag<AllToAllVGroupedMatMulUdmaDefaultConfig>{});
        }
    }
    return output;
}

at::Tensor AllToAllVGroupedMatMulUdmaNpu(const at::Tensor& x, const at::Tensor& weight,
                                         const at::Tensor& localTokensPerExpert,
                                         const at::Tensor& globalTokensPerLocalExpert, int64_t epSize,
                                         int64_t expertNum, int64_t receivedRows, at::IntArrayRef tiling)
{
    return AllToAllVGroupedMatMulUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize,
                                          expertNum, receivedRows, c10::nullopt, c10::nullopt, c10::nullopt, tiling);
}

at::Tensor AllToAllVGroupedMatMulUdmaOutNpu(const at::Tensor& x, const at::Tensor& weight,
                                            const at::Tensor& localTokensPerExpert,
                                            const at::Tensor& globalTokensPerLocalExpert, int64_t epSize,
                                            int64_t expertNum, int64_t receivedRows, const at::Tensor& output,
                                            at::IntArrayRef tiling)
{
    return AllToAllVGroupedMatMulUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize,
                                          expertNum, receivedRows, output, c10::nullopt, c10::nullopt, tiling);
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaExportNpu(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& localTokensPerExpert,
    const at::Tensor& globalTokensPerLocalExpert, int64_t epSize, int64_t expertNum, int64_t receivedRows,
    const c10::optional<at::Tensor>& routeMetadata, const at::Tensor& allToAllVOutput, at::IntArrayRef tiling)
{
    at::Tensor output =
        AllToAllVGroupedMatMulUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize, expertNum,
                                       receivedRows, c10::nullopt, allToAllVOutput, routeMetadata, tiling);
    return {output, allToAllVOutput};
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaOutExportNpu(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& localTokensPerExpert,
    const at::Tensor& globalTokensPerLocalExpert, int64_t epSize, int64_t expertNum, int64_t receivedRows,
    const c10::optional<at::Tensor>& routeMetadata, const at::Tensor& output, const at::Tensor& allToAllVOutput,
    at::IntArrayRef tiling)
{
    AllToAllVGroupedMatMulUdmaImpl(x, weight, localTokensPerExpert, globalTokensPerLocalExpert, epSize, expertNum,
                                   receivedRows, output, allToAllVOutput, routeMetadata, tiling);
    return {output, allToAllVOutput};
}

at::Tensor AllToAllVGroupedMatMulUdmaMeta(const at::Tensor& x, const at::Tensor& weight, const at::Tensor&,
                                          const at::Tensor&, int64_t epSize, int64_t expertNum, int64_t receivedRows,
                                          at::IntArrayRef tiling)
{
    (void)ResolveTiling(kDefaultTiling, tiling, false);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    return at::empty({receivedRows, weight.size(2)}, x.options());
}

at::Tensor AllToAllVGroupedMatMulUdmaOutMeta(const at::Tensor& x, const at::Tensor& weight, const at::Tensor&,
                                             const at::Tensor&, int64_t epSize, int64_t expertNum, int64_t receivedRows,
                                             const at::Tensor& output, at::IntArrayRef tiling)
{
    (void)ResolveTiling(kDefaultTiling, tiling, false);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows && output.size(1) == weight.size(2),
                "invalid out shape");
    return output;
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaExportMeta(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor&, const at::Tensor&, int64_t epSize,
    int64_t expertNum, int64_t receivedRows, const c10::optional<at::Tensor>&, const at::Tensor& allToAllVOutput,
    at::IntArrayRef tiling)
{
    (void)ResolveTiling(kDefaultTiling, tiling, true);
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(
        allToAllVOutput.dim() == 2 && allToAllVOutput.size(0) == receivedRows && allToAllVOutput.size(1) == x.size(1),
        "invalid all_to_allv_out shape");
    return {at::empty({receivedRows, weight.size(2)}, x.options()), allToAllVOutput};
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaOutExportMeta(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor&, const at::Tensor&, int64_t epSize,
    int64_t expertNum, int64_t receivedRows, const c10::optional<at::Tensor>&, const at::Tensor& output,
    const at::Tensor& allToAllVOutput, at::IntArrayRef tiling)
{
    (void)ResolveTiling(kDefaultTiling, tiling, true);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows && output.size(1) == weight.size(2),
                "invalid out shape");
    TORCH_CHECK(
        allToAllVOutput.dim() == 2 && allToAllVOutput.size(0) == receivedRows && allToAllVOutput.size(1) == x.size(1),
        "invalid all_to_allv_out shape");
    return {output, allToAllVOutput};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "_attach_udma(Tensor device_anchor, int rank, int world_size, int workspace_address, int workspace_bytes) -> "
        "()");
    m.def("_detach_udma(Tensor device_anchor) -> ()");
    m.def("_tiling_hardware(Tensor device_anchor) -> int[]");
    m.def(
        "all_to_allv_grouped_mat_mul_udma(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
        "int[] tiling=[]) -> Tensor");
    m.def(
        "all_to_allv_grouped_mat_mul_udma_out(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
        "Tensor(a!) out, int[] tiling=[]) -> Tensor(a!)");
    m.def(
        "all_to_allv_grouped_mat_mul_udma_export(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
        "Tensor? route_metadata, Tensor(a!) all_to_allv_out, int[] tiling=[]) -> (Tensor, Tensor(a!))");
    m.def(
        "all_to_allv_grouped_mat_mul_udma_out_export(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
        "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
        "Tensor? route_metadata, Tensor(a!) out, Tensor(b!) all_to_allv_out, "
        "int[] tiling=[]) -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("_attach_udma", AttachUdma);
    m.impl("_detach_udma", DetachUdma);
    m.impl("_tiling_hardware", TilingHardware);
    m.impl("all_to_allv_grouped_mat_mul_udma", AllToAllVGroupedMatMulUdmaNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_out", AllToAllVGroupedMatMulUdmaOutNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_export", AllToAllVGroupedMatMulUdmaExportNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_out_export", AllToAllVGroupedMatMulUdmaOutExportNpu);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("all_to_allv_grouped_mat_mul_udma", AllToAllVGroupedMatMulUdmaMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_out", AllToAllVGroupedMatMulUdmaOutMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_export", AllToAllVGroupedMatMulUdmaExportMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_out_export", AllToAllVGroupedMatMulUdmaOutExportMeta);
}

}  // namespace deepep::ep_gmm_fused
