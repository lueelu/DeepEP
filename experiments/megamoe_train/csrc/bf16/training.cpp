// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "training.hpp"
#include "tiling.hpp"

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>

#include <tiling/platform/platform_ascendc.h>

#include "shmem.h"

#include "check.hpp"
#include "launcher.hpp"
#include "csrc/kernels/bf16/common/mega_moe_tiling.hpp"
#include "csrc/kernels/bf16/layout/mega_moe_layout.hpp"
#include "csrc/kernels/bf16/layout/mega_moe_saved_meta.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace {

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;
constexpr int64_t kMegaMoeContextBytes = static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025);

constexpr std::size_t kBackwardTopkPartialUbBytes = 128ULL * 1024ULL;
constexpr std::size_t kA5Fp32PerVectorRegister = 64ULL;

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor AllocateForwardTransientWorkspace(
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout& dispatchReadyLayout, const torch::Tensor& like,
    const MegaMoeTilingData& tiling)
{
    // The transient payload is overwritten before use. The forward kernel
    // resets every control flag at the start of each invocation, before its
    // existing pre-Dispatch SyncAll publishes those resets to consumers.
    TORCH_CHECK(dispatchReadyLayout.totalBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "MegaMoe BF16 forward transient workspace exceeds int64 capacity");

    const auto savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(tiling.savedGateUpMode, tiling.swigluLimit);
    const uint64_t extraBytes = MegaMoeBF16Training::ForwardSwigluExtraBytes(
        tiling.n, tiling.maxOutputSize, savedMode, MegaMoeBF16SwigluDecoupled(tiling.k, savedMode));
    TORCH_CHECK(
        extraBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - dispatchReadyLayout.totalBytes,
        "MegaMoe BF16 decoupled transient workspace exceeds int64 capacity");

    return torch::empty({static_cast<int64_t>(dispatchReadyLayout.totalBytes + extraBytes)},
                        like.options().dtype(torch::kUInt8));
}

struct Bf16WeightLayoutInfo {
    int64_t n;
    bool transW;
};

Bf16WeightLayoutInfo ResolveBf16WeightLayout(const torch::Tensor& weight1, const torch::Tensor& weight2, int64_t k)
{
    const int64_t expertPerRank = weight1.size(0);
    const bool canonical = weight1.size(2) == k && weight2.size(0) == expertPerRank && weight2.size(1) == k &&
                           weight1.size(1) % 2 == 0 && weight2.size(2) == weight1.size(1) / 2;
    if (canonical) {
        return {weight1.size(1), true};
    }

    const bool legacy = weight1.size(1) == k && weight2.size(0) == expertPerRank && weight2.size(2) == k &&
                        weight1.size(2) % 2 == 0 && weight2.size(1) == weight1.size(2) / 2;
    TORCH_CHECK(legacy,
                "MegaMoe BF16 weights must use either canonical layout "
                "weight1 [E, N, K], weight2 [E, K, N / 2], or legacy Megatron "
                "layout weight1 [E, K, N], weight2 [E, N / 2, K]");
    return {weight1.size(2), false};
}

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

std::size_t MegaMoeWindowControlRegionBytes(int64_t worldSize, int64_t expertPerRank, bool compactTokenCountExchange)
{
    // Clear the optional compact inbox and the unchanged dense token table.
    // Forward/backward payload starts at payloadOffset and is never cleared.
    const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
        static_cast<uint64_t>(worldSize), static_cast<uint64_t>(expertPerRank), compactTokenCountExchange);
    return static_cast<std::size_t>(
        MegaMoeBF16Training::AlignUp(countLayout.payloadOffset, MegaMoeBF16Training::kAlignment512));
}

void ClearMegaMoeWindowControlRegion(void* symmetricBuffer, std::size_t symmetricBufferBytes, int64_t worldSize,
                                     int64_t expertPerRank, bool compactTokenCountExchange)
{
    const std::size_t controlBytes =
        MegaMoeWindowControlRegionBytes(worldSize, expertPerRank, compactTokenCountExchange);
    const std::size_t clearOffset = static_cast<std::size_t>(MegaMoeBF16Training::PeerDataOffset(worldSize));
    TORCH_CHECK(controlBytes <= symmetricBufferBytes,
                "MegaMoe BF16 SHMEM control region exceeds the symmetric buffer: need ", controlBytes, " bytes, got ",
                symmetricBufferBytes);
    TORCH_CHECK(clearOffset <= controlBytes, "invalid MegaMoe BF16 SHMEM control clear range");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ASCEND_MEGAMOE_CHECK_ACL(aclrtMemsetAsync(static_cast<uint8_t*>(symmetricBuffer) + clearOffset,
                                              symmetricBufferBytes - clearOffset, 0, controlBytes - clearOffset,
                                              stream));
}

constexpr std::size_t Bf16Align(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct MegaMoeBF16BackwardWorkspaceLayout {
    std::size_t controlOffset;
    std::size_t totalBytes;

    explicit MegaMoeBF16BackwardWorkspaceLayout(const MegaMoeTilingData& tiling, uint32_t splitExperts)
    {
        const std::size_t e = tiling.expertPerRank;
        const std::size_t r = tiling.maxOutputSize;
        const std::size_t k = tiling.k;
        const std::size_t n = tiling.n;
        const std::size_t f = n / 2;
        const std::size_t dpPartialTiles = (f + BACKWARD_DP_PARTIAL_TILE_N - 1) / BACKWARD_DP_PARTIAL_TILE_N;
        std::size_t offset = 0;
        offset += Bf16Align(2 * r * k, 512);
        const uint32_t savedMode =
            MegaMoeBF16Training::ResolveSavedGateUpMode(tiling.savedGateUpMode, tiling.swigluLimit);
        if (savedMode == MegaMoeBF16Training::kSaveNone) {
            offset += Bf16Align(2 * r * f, 512);
            offset += Bf16Align(2 * r * f, 512);
        }
        offset += Bf16Align(2 * r * n, 512);
        offset += Bf16Align(2 * r * f, 512);
        offset += Bf16Align(4 * r * dpPartialTiles, 512);
        // dHp reuses Hp and saved-mode dX reuses dO; no extra payload.
        controlOffset = offset;
        offset += Bf16Align(4 * e * 16, 128);
        offset += Bf16Align(4, 32);
        offset += Bf16Align(4, 32);
        offset += Bf16Align(4 * e * 16, 128);
        offset += Bf16Align(4 * e * 16, 128);  // dZ ready, one arrival per AIV
        offset += Bf16Align(4 * e * 16, 128);  // dX ready, one arrival per AIC
        if (MegaMoeBF16Training::UseGmForBackwardDataGrad(savedMode)) {
            offset += Bf16Align(4 * e * 16, 128);  // all AICs retired dO reads
        }
        totalBytes = MegaMoeBF16Training::BackwardDispatchProgressLayout::TotalBytes(
            offset, splitExperts, tiling.epWorldSize, tiling.blockNumPerEP, savedMode);
    }
};

torch::Tensor AllocateBackwardWorkspace(const MegaMoeBF16BackwardWorkspaceLayout& layout, const torch::Tensor& like)
{
    // dO, recomputed gate/up, dZ, Hp and dP partials are payload regions.
    // Their valid packed rows are fully overwritten before any consumer runs.
    // Atomic dispatch/ready/done counters and the per-producer progress suffix require zero.
    TORCH_CHECK(layout.controlOffset <= layout.totalBytes &&
                    layout.totalBytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "invalid MegaMoe BF16 backward workspace initialization range");
    auto workspace = torch::empty({static_cast<int64_t>(layout.totalBytes)}, like.options().dtype(torch::kUInt8));
    workspace
        .narrow(0, static_cast<int64_t>(layout.controlOffset),
                static_cast<int64_t>(layout.totalBytes - layout.controlOffset))
        .zero_();
    return workspace;
}

}  // namespace

namespace deepep::megamoe_train::ops {

ForwardResult MegaMoeBF16Forward(torch::Tensor context, torch::Tensor x, torch::Tensor topkIds,
                                 torch::Tensor topkWeights, torch::Tensor weight1, torch::Tensor weight2,
                                 int64_t maxRecvTokens, int rank, int worldSize, std::size_t symmetricBufferBytes,
                                 uint64_t ffts_addr, void* symmetricBuffer, double swigluLimit,
                                 const std::string& save_gmm1_output)
{
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(topkIds, "topk_ids");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topkIds.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16 || topkWeights.scalar_type() == torch::kFloat32,
                "topk_weights must be bfloat16 or float32");
    const uint32_t topkWeightType = topkWeights.scalar_type() == torch::kFloat32 ? MegaMoeBF16Training::kTopkWeightFp32
                                                                                 : MegaMoeBF16Training::kTopkWeightBf16;
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(x.dim() == 2 && topkIds.dim() == 2, "x and topk_ids must be rank-2 tensors");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(), "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0), "x and topk_ids must have the same token count");
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3, "weight1 and weight2 must be rank-3 tensors");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 dimensions must be positive");
    TORCH_CHECK(worldSize >= 2 && worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
                "MegaMoe BF16 requires 2..1024 EP ranks");
    TORCH_CHECK(m <= 16384 && expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
                    expertPerRank * worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
                "MegaMoe BF16 forward shape exceeds supported routing limits "
                "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize, "MegaMoe BF16 topk must not exceed the total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0, "MegaMoe BF16 requires k % 256 == 0 and n % 512 == 0");
    MegaMoeTilingData tiling{};
    const auto hostTiling =
        detail::MakeMegaMoeHostTiling(tiling, m, k, n, topk, worldSize, expertPerRank, maxRecvTokens, rank, true,
                                      swigluLimit, true, save_gmm1_output, topkWeightType);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers = detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes = hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes, "MegaMoe BF16 local_mem_size is too small: need ", peerBytes,
                " bytes, got ", symmetricBufferBytes);

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n,
        MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode),
        tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone,
        MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false, topkWeightBytes);
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    const MegaMoeBF16Training::ForwardTransientWorkspaceLayout transientWorkspaceLayout(n, expertPerRank,
                                                                                        maxRecvTokens);
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout dispatchTileReadyLayout(n, expertPerRank, maxRecvTokens);
    const uint64_t perRankTokenCountStride =
        MegaMoeBF16Training::AlignUp(worldSize * expertPerRank, MegaMoeBF16Training::kAlignment128);
    const uint64_t legacyTrainingTailBytes = MegaMoeBF16Training::AlignUp(
        worldSize * perRankTokenCountStride * MegaMoeBF16Training::kInt32Bytes, MegaMoeBF16Training::kAlignment512);
    TORCH_CHECK(savedStateLayout.legacyBytes + dispatchOutputLayout.totalBytes + transientWorkspaceLayout.totalBytes ==
                    hostTiling.workspace_bytes + earlyWeightBuffers.workspace_bytes + legacyTrainingTailBytes,
                "MegaMoe BF16 split workspace layout does not match the legacy "
                "forward workspace capacity");

    // Both unpermute implementations overwrite every output token row once.
    // ExpertTokenNumCopyOut likewise writes all local-expert entries, so these
    // returned tensors do not need a device-wide pre-clear.
    auto y = torch::empty({m, k}, x.options());
    auto expertTokenNums = torch::empty({expertPerRank}, topkIds.options().dtype(torch::kInt32));
    auto trainingSavedState = AllocateTrainingSavedState(savedStateLayout, x);
    auto dispatchOutput =
        torch::empty({static_cast<int64_t>(dispatchOutputLayout.totalBytes)}, x.options().dtype(torch::kUInt8));
    auto forwardTransientWorkspace = AllocateForwardTransientWorkspace(dispatchTileReadyLayout, x, tiling);
    ClearMegaMoeWindowControlRegion(symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
                                    tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16ForwardKernel(
        context, x, topkIds, topkWeights, weight1, weight2, y, expertTokenNums, trainingSavedState, dispatchOutput,
        forwardTransientWorkspace, tiling,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize, ffts_addr);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2, &y, &expertTokenNums,
                               &trainingSavedState, &dispatchOutput, &forwardTransientWorkspace}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    return {y, expertTokenNums, trainingSavedState, dispatchOutput};
}

BackwardResult MegaMoeBF16Backward(torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
                                   torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
                                   torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
                                   int64_t maxRecvTokens, int rank, int worldSize, std::size_t symmetricBufferBytes,
                                   uint64_t ffts_addr, void* symmetricBuffer, double swigluLimit,
                                   const std::string& save_gmm1_output)
{
    const int64_t m = gradY.size(0);
    const int64_t k = gradY.size(1);
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3, "weight1 and weight2 must be rank-3 tensors");
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(topkWeights.dim() == 2 && topkWeights.size(0) == m, "topk_weights must have shape [m, topk]");
    const int64_t topk = topkWeights.size(1);
    TORCH_CHECK(expertTokenNums.numel() == expertPerRank, "expert_token_nums must have one entry per local expert");
    // Only persistent saved-state is consumed by backward.
    TORCH_CHECK(trainingSavedState.scalar_type() == torch::kUInt8 &&
                    trainingSavedState.device().type() == c10::DeviceType::PrivateUse1,
                "training_saved_state must be an NPU UInt8 tensor");
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 backward dimensions must be positive");
    TORCH_CHECK(worldSize >= 2 && worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
                "MegaMoe BF16 backward requires 2..1024 EP ranks");
    TORCH_CHECK(m <= 16384 && expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
                    expertPerRank * worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
                "MegaMoe BF16 backward shape exceeds supported routing limits "
                "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize, "MegaMoe BF16 backward topk must not exceed total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "MegaMoe BF16 backward max_recv_tokens must be positive");
    TORCH_CHECK(maxRecvTokens <= std::numeric_limits<int32_t>::max(),
                "MegaMoe BF16 backward max_recv_tokens exceeds int32 routing capacity");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0, "MegaMoe BF16 backward requires k % 256 == 0 and n % 512 == 0");
    TORCH_CHECK(gradY.scalar_type() == torch::kBFloat16, "grad_y must be bfloat16");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16 || topkWeights.scalar_type() == torch::kFloat32,
                "topk_weights must be bfloat16 or float32");
    const uint32_t topkWeightType = topkWeights.scalar_type() == torch::kFloat32 ? MegaMoeBF16Training::kTopkWeightFp32
                                                                                 : MegaMoeBF16Training::kTopkWeightBf16;
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(expertTokenNums.scalar_type() == torch::kInt32, "expert_token_nums must be int32");
    CheckNpuContiguous(gradY, "grad_y");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");
    CheckNpuContiguous(trainingSavedState, "training_saved_state");
    CheckNpuContiguous(dispatchOutput, "dispatch_output");
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    TORCH_CHECK(dispatchOutput.scalar_type() == torch::kUInt8 &&
                    static_cast<uint64_t>(dispatchOutput.numel()) >= dispatchOutputLayout.totalBytes,
                "dispatch_output must be an NPU UInt8 tensor with at least ", dispatchOutputLayout.totalBytes,
                " bytes");

    MegaMoeTilingData tiling{};
    const auto hostTiling =
        detail::MakeMegaMoeHostTiling(tiling, m, k, n, topk, worldSize, expertPerRank, maxRecvTokens, rank, true,
                                      swigluLimit, true, save_gmm1_output, topkWeightType);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers = detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes = hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes, "MegaMoe BF16 backward local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes);

    const std::size_t ubTileBytes = 3ULL * 128ULL * 256ULL * 2ULL;
    // MXFP8 backward decodes 16 rows at a time after the three persistent
    // BF16 work tiles: one FP8 data chunk, compact/padded E8M0 scales,
    // expanded BF16/FP32 scales, a temporary FP32 decoded chunk and three
    // optional clipping-mask chunks.
    constexpr std::size_t kMxDecodeRows = 16ULL;
    constexpr std::size_t kMxTileColumns = 256ULL;
    const std::size_t ubMxFp8TileBytes =
        ubTileBytes + kMxDecodeRows * kMxTileColumns + kMxDecodeRows * 32ULL +
        kMxDecodeRows * (kMxTileColumns / 32ULL) * (2ULL * sizeof(uint16_t) + 4ULL * sizeof(float)) +
        kMxDecodeRows * kMxTileColumns * sizeof(float) + 3ULL * kMxDecodeRows * 32ULL;
    const std::size_t ubRestoreBytes = Bf16Align(static_cast<std::size_t>(worldSize) * expertPerRank, 128ULL) * 4ULL;
    const std::size_t dispatchElements =
        Bf16Align(static_cast<std::size_t>(tiling.dispatchRows) * static_cast<std::size_t>(k), 32ULL);
    const std::size_t ubP0ScratchBytes = 4ULL * dispatchElements +
                                         Bf16Align(2ULL * static_cast<std::size_t>(k) * sizeof(uint16_t), 32ULL) +
                                         ubRestoreBytes;
    const std::size_t ubUnpermuteBytes = 3ULL * Bf16Align(6ULL * static_cast<std::size_t>(k), 32ULL);
    const std::size_t dpPartialTiles =
        (static_cast<std::size_t>(n / 2) + BACKWARD_DP_PARTIAL_TILE_N - 1) / BACKWARD_DP_PARTIAL_TILE_N;
    const std::size_t alignedDpPartialTiles = Bf16Align(dpPartialTiles * sizeof(float), 32ULL) / sizeof(float);
    TORCH_CHECK((alignedDpPartialTiles + kA5Fp32PerVectorRegister - 1) * sizeof(float) <= kBackwardTopkPartialUbBytes,
                "MegaMoe BF16 backward n=", n, " exceeds the P6 partial-reduction UB capacity");
    const std::size_t ubTopkBytes = kBackwardTopkPartialUbBytes + Bf16Align(1024ULL * topkWeightBytes, 32ULL);
    const std::size_t ubResetBytes = MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    // These phases own UB at different times. Include the post-P6 count
    // reset and the P6 route buffer, not just the P0 and SwiGLU allocations.
    const uint32_t savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(tiling.savedGateUpMode, tiling.swigluLimit);
    const std::size_t p1UbBytes = savedMode == MegaMoeBF16Training::kSaveMxFp8 ? ubMxFp8TileBytes : ubTileBytes;
    const std::size_t ubFootprint =
        std::max({p1UbBytes, ubP0ScratchBytes, ubUnpermuteBytes, ubTopkBytes, ubResetBytes});
    uint64_t deviceUbBytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                                                            deviceUbBytes);
    const uint64_t ubBudget = std::min<uint64_t>(deviceUbBytes, 248ULL * 1024ULL);
    TORCH_CHECK(ubFootprint <= ubBudget, "MegaMoe BF16 backward k=", k, " exceeds the UB budget (kernel needs ",
                ubFootprint, " bytes, have ", ubBudget, ")");

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n,
        MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode),
        tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone,
        MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false, topkWeightBytes);
    TORCH_CHECK(static_cast<std::size_t>(trainingSavedState.numel()) * trainingSavedState.element_size() >=
                    savedStateLayout.totalBytes,
                "MegaMoe BF16 training_saved_state is too small: need ", savedStateLayout.totalBytes, " bytes");
#if DEEPEP_BF16_FORCE_EXPERT_READY
    // This build policy takes precedence over the diagnostic split override.
    const uint32_t splitExperts = 0;
#else
    uint32_t splitExperts = MegaMoeBF16Training::DefaultBackwardDispatchSplitExperts(
        tiling.savedGateUpMode, static_cast<uint32_t>(expertPerRank));
    // Explicit k is a diagnostic override; none still forces k=0 below.
    const char* splitText = std::getenv("MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS");
    if (splitText != nullptr) {
        if (std::string(splitText) == "all") {
            splitExperts = static_cast<uint32_t>(expertPerRank);
        } else {
            char* end = nullptr;
            const long value = std::strtol(splitText, &end, 10);
            TORCH_CHECK(end != splitText && *end == '\0' && value >= 0 && value <= expertPerRank,
                        "MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS must be 0..localE or all");
            splitExperts = static_cast<uint32_t>(value);
        }
    }
    if (!MegaMoeBF16Training::UseGmForBackwardDataGrad(tiling.savedGateUpMode)) splitExperts = 0;
#endif
    const MegaMoeBF16BackwardWorkspaceLayout backwardLayout(tiling, splitExperts);
    auto backwardWorkspace = AllocateBackwardWorkspace(backwardLayout, gradY);

    // P5 unpermutes and sums packed dX into every gradX token row; P6 writes
    // every TopK route once. The kernel overwrites every non-empty expert's
    // weight-gradient plane and clears only the experts whose effective row
    // count is zero, so none of these outputs needs a full pre-clear here.
    auto gradX = torch::empty({m, k}, gradY.options());
    auto gradTopkWeights = torch::empty_like(topkWeights);
    auto gradWeight1 = torch::empty_like(weight1);
    auto gradWeight2 = torch::empty_like(weight2);
    ClearMegaMoeWindowControlRegion(symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
                                    tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16BackwardKernel(
        context, gradY, topkWeights, weight1, weight2, gradX, gradTopkWeights, gradWeight1, gradWeight2,
        trainingSavedState, dispatchOutput, backwardWorkspace, tiling, splitExperts,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize, ffts_addr);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor :
         {&context, &gradY, &topkWeights, &weight1, &weight2, &expertTokenNums, &trainingSavedState, &dispatchOutput,
          &gradX, &gradTopkWeights, &gradWeight1, &gradWeight2, &backwardWorkspace}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    return {gradX, gradTopkWeights, gradWeight1, gradWeight2};
}

}  // namespace deepep::megamoe_train::ops

namespace deepep::megamoe_train {

BF16Executor::BF16Executor(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like)
    : address_(address), bytes_(bytes), rank_(rank), world_size_(world_size)
{
    TORCH_CHECK(address != 0 && bytes > 0 && bytes % kDeviceLargePageBytes == 0, "invalid borrowed training window");
    TORCH_CHECK(rank >= 0 && rank < world_size && world_size <= 1024, "invalid training rank/world size");
    TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == world_size,
                "training identity disagrees with the root Session");
    ffts_addr_ = shmemx_get_ffts_config();
    (void)Context(like);
}

void BF16Executor::close()
{
    address_ = 0;
    context_ = torch::Tensor();
}

ops::ForwardResult BF16Executor::forward(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
                                         torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
                                         double swiglu_limit, const std::string& save_gmm1_output)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    return ops::MegaMoeBF16Forward(Context(x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens, rank_,
                                   world_size_, SymmetricBufferBytes(), ffts_addr_, SymmetricBuffer(), swiglu_limit,
                                   save_gmm1_output);
}

ops::BackwardResult BF16Executor::backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                           torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                           torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                           const std::string& save_gmm1_output)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    return ops::MegaMoeBF16Backward(Context(grad_y), grad_y, topk_weights, weight1, weight2, counts, saved_state,
                                    dispatch_output, max_recv_tokens, rank_, world_size_, SymmetricBufferBytes(),
                                    ffts_addr_, SymmetricBuffer(), swiglu_limit, save_gmm1_output);
}

torch::Tensor BF16Executor::Context(const torch::Tensor& like)
{
    TORCH_CHECK(like.defined(), "MegaMoe BF16 context reference tensor must be defined");
    TORCH_CHECK(like.device().type() == c10::DeviceType::PrivateUse1,
                "MegaMoe BF16 context reference tensor must be an NPU tensor");
    TORCH_CHECK(like.is_contiguous(), "MegaMoe BF16 context reference tensor must be contiguous");
    TORCH_CHECK(like.scalar_type() == torch::kBFloat16, "MegaMoe BF16 context reference tensor must be bfloat16");
    const int rank = rank_;
    const int world_size = world_size_;
    if (!context_.defined()) {
        auto new_context = torch::empty({kMegaMoeContextBytes}, like.options().dtype(torch::kUInt8));
        InitializeMegaMoeBF16Context(new_context, SymmetricBuffer(), rank, world_size);
        // Publish the persistent state only after initialization succeeds.
        // A failed first call must remain retryable instead of leaving a
        // defined but partially initialized context behind.
        context_ = new_context;
        context_rank_ = rank;
        context_world_size_ = world_size;
    } else {
        TORCH_CHECK(context_.device() == like.device(), "MegaMoe BF16 context device changed across calls");
        TORCH_CHECK(context_rank_ == rank && context_world_size_ == world_size,
                    "MegaMoe BF16 SHMEM rank/world size changed across calls");
    }
    return context_;
}

void* BF16Executor::SymmetricBuffer() const
{
    return reinterpret_cast<void*>(address_);
}

std::size_t BF16Executor::SymmetricBufferBytes() const
{
    return static_cast<std::size_t>(bytes_);
}

}  // namespace deepep::megamoe_train
