/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_mat_mul_all_to_allv_udma_device.hpp"
#include "large_ep_pipeline_launch.hpp"
#include <cstdlib>
#include <cstring>
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
    uint32_t commNpuSplit;
    uint32_t commDataSplit;
    uint32_t stageRows;
    uint32_t workspaceStages;
    uint32_t localCopyTileRows;
    uint32_t selfCopyCores;
};

constexpr TilingProfile kDefaultTiling{KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16, 0, 2, 32, 1};
// 2026-09-09 Ascend950DT_9582 search winners, with commTileM = 2 * local copy rows.
// K1024: f1d10672a756a4a9; K512: fdfb758ae9b102f9.
constexpr TilingProfile kK1024N2048Tiling{KernelVariant::kM0_128, 128, 256, 256, 10, 8, 64, 4, 4, 0, 2, 4, 1};
constexpr TilingProfile kK512N2048Tiling{KernelVariant::kM0_128, 128, 256, 256, 16, 32, 16, 1, 20, 0, 2, 16, 1};

// EP4 / 28 experts lean-staging profiles. The 2026-11-25 Event search found
// 256-row staging at M=65536 (3544.5 / 6989.6 us) and msprof confirmed the same
// profile is also faster at M=131072 (6834.4 / 13546.3 us), so EP4 selects it
// for the whole K/N pair. M stays out of the key because the inverse input rows
// are local to a rank under imbalanced routing.
constexpr TilingProfile kEp4K3072N3584Tiling{
    KernelVariant::kM0_256N0_256, 256, 256, 256, 1, 2, 64, 1, 16, 256, 3, 1, 8};
constexpr TilingProfile kEp4K6144N3584Tiling{
    KernelVariant::kM0_256N0_256, 256, 256, 256, 1, 32, 64, 1, 16, 256, 3, 16, 8};

// EP64 / 896 experts, 2026-11-15 Event search: c21e4912f8fac930 / 2ef5abd707b6fd40.
// Performance-selected profiles; full numerical validation remains pending.
constexpr TilingProfile kEp64K3072N3584Tiling{KernelVariant::kM0_128, 128, 256, 256, 1, 2, 64, 1, 16, 256, 3, 1, 1};
constexpr TilingProfile kEp64K6144N3584Tiling{KernelVariant::kM0_128, 128, 256, 256, 1, 32, 64, 1, 16, 256, 3, 16, 1};

TilingProfile SelectTiling(uint32_t k, uint32_t n, uint32_t worldSize, uint32_t expertNum)
{
    // The local GMM input rows equal sum(global_tokens) and may differ across
    // ranks under imbalanced routing. UDMA peers must nevertheless use the same
    // M0 and communication schedule, so selection may only depend on topology
    // and K/N, which are rank-invariant for one collective launch.
    if (worldSize == 64 && expertNum == 896 && n == 3584) {
        if (k == 3072) {
            return kEp64K3072N3584Tiling;
        }
        if (k == 6144) {
            return kEp64K6144N3584Tiling;
        }
    }
    const bool isFourRank28ExpertCase = worldSize == 4 && expertNum == 28;
    if (isFourRank28ExpertCase && n == 3584) {
        if (k == 3072) {
            return kEp4K3072N3584Tiling;
        }
        if (k == 6144) {
            return kEp4K6144N3584Tiling;
        }
    }
    const bool isEightRank256ExpertCase = worldSize == 8 && expertNum == 256;
    if (isEightRank256ExpertCase && k == 1024 && n == 2048) {
        return kK1024N2048Tiling;
    }
    if (isEightRank256ExpertCase && k == 512 && n == 2048) {
        return kK512N2048Tiling;
    }
    TilingProfile result = kDefaultTiling;
    if (worldSize > 8) {
        // One M tile per peer decouples staging from AI-core count / EP.
        // In particular EP64/128 need not divide coreCount * commInterval.
        result.stageRows = result.m0;
    }
    return result;
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
    result.kernelVariant = parsed.m0 == 128
                               ? KernelVariant::kM0_128
                               : (parsed.n0 == 128 ? KernelVariant::kM0_256 : KernelVariant::kM0_256N0_256);
    result.m0 = parsed.m0;
    result.n0 = parsed.n0;
    result.k0 = parsed.k0;
    result.commInterval = parsed.commInterval;
    result.commTileM = 2 * parsed.localCopyTileRows;
    result.localCopyTileRows = parsed.localCopyTileRows;
    // Legacy overrides preserve derived staging, two buffers and one self worker,
    // even when the shape now has a physical built-in profile.
    result.stageRows = parsed.stageRows;
    result.workspaceStages = parsed.workspaceStages;
    result.selfCopyCores = parsed.selfCopyCores;
    return result;
}

constexpr uint32_t kMaxLocalExperts = 64;
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
    TORCH_CHECK(epSize > 0 && epSize <= TilingValidation::kMaxRanks, "ep_size must be in [1, 128]");
    TORCH_CHECK(epSize == worldSize, "TP=1 is required: ep_size must equal UDMA world_size");
    TORCH_CHECK(expertNum > 0 && expertNum % epSize == 0, "expert_num must be positive and divisible by ep_size");
    const int64_t localExpertNum = expertNum / epSize;
    TORCH_CHECK(runtime.Rank() < epSize, "UDMA rank must be smaller than ep_size");
    TORCH_CHECK(localExpertNum <= kMaxLocalExperts, "local expert count exceeds ", kMaxLocalExperts);

    const int64_t inputRows64 = x.size(0);
    const int64_t k64 = x.size(1);
    const int64_t n64 = weight.size(2);
    TORCH_CHECK(k64 > 0 && n64 > 0, "k and n must be positive");
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
    const uint32_t m = static_cast<uint32_t>(inputRows64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t ranksPerNode = runtime.RanksPerNode();
    const bool highEp = (worldSize == 64 || worldSize == 128) && ranksPerNode > 0;
    const bool matchedHighEp = highEp && (localExpertNum == 7 || localExpertNum == 14) &&
                               (k == 3072 || k == 6144) && n == 3584;
    uint32_t pipelineChunks = matchedHighEp ? 2U : 0U;
    if (highEp && runtime.PipelineChunks() >= 0) {
        pipelineChunks = static_cast<uint32_t>(runtime.PipelineChunks());
    } else if (worldSize == 16) {
        const char* setting = std::getenv("DEEPEP_LARGE_EP_PIPELINE_CHUNKS");
        if (setting && *setting) {
            TORCH_CHECK(std::strcmp(setting, "0") == 0 || std::strcmp(setting, "1") == 0 ||
                            std::strcmp(setting, "2") == 0 || std::strcmp(setting, "4") == 0,
                        "DEEPEP_LARGE_EP_PIPELINE_CHUNKS must be 0, 1, 2 or 4");
            pipelineChunks = static_cast<uint32_t>(setting[0] - '0');
        }
    }
    // Choose using collective-wide dimensions. The routed input row count may differ on each rank.
    auto defaultTiling = SelectTiling(k, n, static_cast<uint32_t>(worldSize), static_cast<uint32_t>(expertNum));
    if (highEp && pipelineChunks != 0) {
        const bool fitsM256 = 2ULL * 256 * worldSize * n * sizeof(uint16_t) <= runtime.SymmetricDataBytes();
        defaultTiling = {fitsM256 ? KernelVariant::kM0_256N0_256 : KernelVariant::kM0_128,
                         fitsM256 ? 256U : 128U, 256, 256, 1, 8, 64, 1, 16, fitsM256 ? 256U : 128U, 2, 4, 1};
    }
    const TilingProfile tiling = ResolveTiling(defaultTiling, tilingValues);
    const bool highEpPipeline = highEp && pipelineChunks != 0 && tiling.workspaceStages == 2;
    const bool useLargeEpPipeline = highEpPipeline || (worldSize == 16 && pipelineChunks != 0);
    TORCH_CHECK(inputRows64 > 0 || highEpPipeline, "input rows must be positive for the selected kernel");
    const uint64_t dataBytes = highEpPipeline ? runtime.SymmetricDataBytes() :
                                                GroupedMatMulAllToAllVUdmaDefaultConfig::kSymmetricDataBytes;
    uint64_t ubBytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t localCopyColumns = TargetOrderLocalCopyColumns(n);
    // Only the built-in fallback is clamped. Explicit profiles must fit exactly.
    const uint32_t localCopyRows =
        tilingValues.empty() ? std::min(tiling.localCopyTileRows, TilingValidation::LocalCopyRowCapacity(n, ubBytes))
                             : tiling.localCopyTileRows;
    TORCH_CHECK(localCopyRows > 0 && localCopyColumns > 0, "N is too large for target-order local MTE copy staging");
    const TilingValidation::GroupedMatMul requested{tiling.m0,
                                                    tiling.n0,
                                                    tiling.k0,
                                                    tiling.commInterval,
                                                    localCopyRows,
                                                    tiling.stageRows,
                                                    tiling.workspaceStages,
                                                    tiling.selfCopyCores};
    const char* error =
        TilingValidation::ValidateGroupedMatMul(requested, n, static_cast<uint32_t>(worldSize), blockDim, ubBytes, dataBytes);
    TORCH_CHECK(error == nullptr, "invalid GroupedMatMul tiling: ", error);
    TORCH_CHECK(inputRows64 <= std::numeric_limits<uint32_t>::max() - tiling.m0 + 1,
                "input rows overflow the target-order tiling scheduler");
    // Target-order staging packs all N tiles of each selected M group into a
    // contiguous [rows, N] buffer, allowing one UDMA request per source rank
    // and communication round.
    uint64_t stageRows = tiling.stageRows;
    if (stageRows == 0) {
        TORCH_CHECK((static_cast<uint64_t>(blockDim) * tiling.commInterval) % worldSize == 0,
                    "AI-core count * comm_interval must be divisible by world_size");
        const uint64_t blockPerComm = static_cast<uint64_t>(blockDim) * tiling.commInterval;
        const uint64_t tilePerCommInRank = blockPerComm / static_cast<uint64_t>(worldSize);
        const uint64_t nLoops = (static_cast<uint64_t>(n64) + tiling.n0 - 1) / tiling.n0;
        TORCH_CHECK(tilePerCommInRank >= nLoops,
                    "target-order UDMA staging requires AI-core count * comm_interval / "
                    "world_size >= ceil(n / N0)");
        const uint64_t groupsPerCommInRank = std::max<uint64_t>(1, tilePerCommInRank / nLoops);
        stageRows = groupsPerCommInRank * tiling.m0;
    }
    const uint64_t requiredWorkspaceBytes = static_cast<uint64_t>(tiling.workspaceStages) * stageRows *
                                            static_cast<uint64_t>(worldSize) * static_cast<uint64_t>(n64) *
                                            sizeof(uint16_t);
    TORCH_CHECK(requiredWorkspaceBytes <= dataBytes, "kernel tiling requires ", requiredWorkspaceBytes,
                " symmetric workspace bytes, exceeding ", dataBytes);

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
    TORCH_CHECK(!highEpPipeline || blockDim == 32,
                "the high-EP pipeline requires 32 AICs");
    TORCH_CHECK(worldSize != 16 || !useLargeEpPipeline ||
                    (blockDim == 32 && localExpertNum == 32 && m == 9216 && k == 1024 && n == 384 && tiling.m0 == 256 &&
                     tiling.n0 == 128 && tiling.k0 == 256 && tiling.workspaceStages == 2 &&
                     tiling.selfCopyCores <= 16 && (stageRows == 8192 || stageRows == 2816)),
                "EP16 pipeline requires the measured S1/M3 shape and M256/N128/K256, WS2 tiling");
    if (highEpPipeline) {
        // The rank sends and receives at most one acknowledgement per peer and round.
        const uint64_t groups = static_cast<uint64_t>(std::max(inputRows64, outputRows64)) / tiling.m0 + localExpertNum;
        const uint64_t groupsPerRound = stageRows / tiling.m0;
        const uint64_t rounds = (groups + groupsPerRound - 1) / groupsPerRound;
        TORCH_CHECK(rounds * std::max<uint64_t>(pipelineChunks, worldSize) <= std::numeric_limits<int32_t>::max(),
                    "high-EP pipeline signal sequence exceeds int32 range");
    }
    const bool wholeRound = worldSize == 16 && useLargeEpPipeline && stageRows == 2816 && pipelineChunks == 1;
    auto launchWithConfig = [&](auto configTag) {
        using SelectedConfig = typename decltype(configTag)::type;
        using SelectedDeviceOp = typename SelectedConfig::Device;
        using SelectedKernel = typename SelectedConfig::Kernel;
        TORCH_CHECK(tiling.m0 == SelectedConfig::kM0 && tiling.n0 == SelectedConfig::kN0 &&
                        tiling.k0 == SelectedConfig::kK0 && tiling.workspaceStages == SelectedConfig::kWorkspaceStages,
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
            static_cast<uint32_t>(stageRows),
            tiling.selfCopyCores,
        };

        SelectedDeviceOp deviceOp;
        const Catlass::Status initializeStatus = deviceOp.Initialize(arguments);
        TORCH_CHECK(initializeStatus == Catlass::Status::kSuccess, "failed to initialize the CatCCOS device operator");
        typename SelectedDeviceOp::Params params = deviceOp.params();
        if (highEpPipeline) {
            // The pipeline uses only the first control region, after the actual symmetric data capacity.
            params.syncMmadFinish = runtime.Workspace() + dataBytes;
        }
        at::Tensor schedulerWorkspace;
        if constexpr (SelectedConfig::kUseGmScheduler) {
            if (!useLargeEpPipeline) {
                using Scheduler = typename SelectedConfig::BlockMmadScheduler;
                const int64_t schedulerBytes =
                    static_cast<int64_t>(static_cast<uint64_t>(blockDim) * Scheduler::WORKSPACE_BYTES);
                schedulerWorkspace = at::empty({schedulerBytes}, x.options().dtype(at::kByte));
                TORCH_CHECK(reinterpret_cast<uintptr_t>(schedulerWorkspace.data_ptr()) % 64 == 0,
                            "scheduler workspace must be cache-line aligned");
                c10_npu::NPUCachingAllocator::recordStream(schedulerWorkspace.storage().data_ptr(),
                                                           c10_npu::getCurrentNPUStream());
                params.ptrSchedulerWorkspace = reinterpret_cast<uint8_t*>(schedulerWorkspace.data_ptr());
            }
        }
        auto launch = [=]() -> int {
            // Keep the allocation alive while RunOpApi queues the launch callback.
            (void)schedulerWorkspace;
            if constexpr (std::is_same_v<SelectedConfig, GroupedMatMulAllToAllVUdmaDefaultConfig> ||
                          std::is_same_v<SelectedConfig, LargeEpN128Config> ||
                          std::is_same_v<SelectedConfig, GroupedMatMulAllToAllVUdmaM0_256N0_256Config>) {
                if (highEpPipeline) {
                    LaunchLargeEp(LargeEpParams<SelectedConfig>{params, pipelineChunks, ranksPerNode},
                                  blockDim, stream, fftsAddress);
                    return 0;
                }
            }
            if constexpr (std::is_same_v<SelectedConfig, LargeEpN128Config>) {
                if (useLargeEpPipeline) {
                    LaunchLargeEpN128(LargeEpN128Params{params, pipelineChunks, 0}, wholeRound, blockDim, stream,
                                      fftsAddress);
                    return 0;
                }
            }
            LaunchGroupedMatMulAllToAllVUdma<SelectedKernel>(params, blockDim, stream, fftsAddress);
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("GroupedMatMulAllToAllVUdma", launch);
    };
    auto launchWithScheduler = [&](auto configTag) {
        using GmConfig = typename decltype(configTag)::type;
        // Preserve the complete pre-expansion path for its original EP/expert
        // range. New small-EP configurations with more experts retain GM.
        if (worldSize <= 8 && localExpertNum <= 32) {
            launchWithConfig(TypeTag<typename GmConfig::template WithScheduler<false>>{});
        } else {
            launchWithConfig(configTag);
        }
    };
    if (tiling.kernelVariant == KernelVariant::kM0_256N0_256) {
        if (tiling.workspaceStages == 3) {
            launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaM0_256N0_256WS3Config>{});
        } else {
            launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaM0_256N0_256Config>{});
        }
    } else if (tiling.kernelVariant == KernelVariant::kM0_256) {
        if (tiling.workspaceStages == 3) {
            launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaM0_256WS3Config>{});
        } else {
            launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaM0_256Config>{});
        }
    } else if (tiling.workspaceStages == 3) {
        launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaDefaultWS3Config>{});
    } else {
        launchWithScheduler(TypeTag<GroupedMatMulAllToAllVUdmaDefaultConfig>{});
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
