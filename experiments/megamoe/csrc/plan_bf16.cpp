// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "kernels/mega_moe/mc2_common/mc2_moe_context.hpp"
#include "plan.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include <acl/acl.h>
#include "check.hpp"
#include "kernel_launch.hpp"
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_wave_plan.hpp"
#include "kernels/mega_moe/timer/ascend_timer.hpp"

namespace {

using deepep::megamoe::AlignUp;
using deepep::megamoe::CeilDiv;

constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t INT_CACHELINE = 16;
constexpr int64_t SWIGLU_N_HALF = 2;
constexpr int64_t SIZE_BF_16 = 2;
constexpr int64_t SIZE_INT_32 = 4;

using namespace AscendC;

namespace optiling {

size_t CalcWorkspaceSize(const MegaMoeBf16TilingData& td)
{
    size_t bytes = 0;
    bytes += AlignUp(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.k, ALIGN_512);
    bytes += AlignUp(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.n / SWIGLU_N_HALF, ALIGN_512);
    bytes += AlignUp(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank * SIZE_INT_32, ALIGN_512);
    bytes += SIZE_INT_32 * AlignUp(static_cast<int64_t>(td.m), ALIGN_256) * td.topK * 2;
    bytes += SIZE_INT_32 * AlignUp(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128);
    bytes += AlignUp(static_cast<int64_t>(SIZE_INT_32) * td.expertPerRank * INT_CACHELINE, ALIGN_128);
    return bytes;
}

inline size_t CalcBf16EarlyTopkWeightWorkspaceSize(const MegaMoeBf16TilingData& td)
{
    return AlignUp(static_cast<int64_t>(td.maxOutputSize) * MegaMoeBF16Impl::TopkWeightBytes(td.topkWeightType),
                   ALIGN_512);
}

}  // namespace optiling

struct BufferSizes {
    uint64_t workspace_bytes;
    uint64_t peer_bytes;
};

uint32_t SaveMode(const std::string& mode)
{
    using namespace MegaMoeBF16Impl;
    if (mode == "none") return kSaveNone;
    if (mode == "mxfp8") return kSaveMxFp8;
    TORCH_CHECK(mode == "bf16", "gmm1_save_type must be none, bf16 or mxfp8");
    return kSaveBf16;
}

float ClampLimit(double limit)
{
    return limit > 0.0 ? static_cast<float>(limit) : kMegaMoeSwigluLimitDisabled;
}

BufferSizes MakeBf16Tiling(MegaMoeBf16TilingData& tiling, int64_t m, int64_t k, int64_t f, int64_t topk,
                           int64_t world_size, int64_t expert_per_rank, int64_t max_recv_tokens, int rank,
                           double swiglu_limit, const std::string& save_gmm1_output, uint32_t topk_weight_type,
                           const deepep::megamoe::PlatformInfo& platform)
{
    using namespace MegaMoeBF16Impl;
    constexpr int64_t maxDim = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(m > 0 && m <= 16384 && max_recv_tokens > 0 && max_recv_tokens <= maxDim, "invalid BF16 token capacity");
    TORCH_CHECK(k > 0 && k <= maxDim && k % 256 == 0 && f > 0 && f <= maxDim / 2 && f % 256 == 0,
                "BF16 hidden and intermediate dimensions must be positive multiples of 256 within tiling limits");
    TORCH_CHECK(world_size >= 2 && world_size <= static_cast<int64_t>(kMaxEpRanks) && expert_per_rank > 0 &&
                    expert_per_rank <= static_cast<int64_t>(kMaxLocalExperts) &&
                    world_size * expert_per_rank <= static_cast<int64_t>(kMaxTotalExperts) && topk > 0 && topk <= 32 &&
                    topk <= world_size * expert_per_rank,
                "invalid BF16 expert configuration");
    const int64_t n = 2 * f;
    const auto aic_num = platform.aic;
    const auto ub_size = platform.ub;
    {
        const MegaMoeBF16Impl::ForwardDispatchUbLayout dispatchUb(world_size, expert_per_rank, k, 2);
        TORCH_CHECK(dispatchUb.totalBytes <= ub_size, "MegaMoe BF16 Dispatch requires ", dispatchUb.totalBytes,
                    " UB bytes, but the device has ", ub_size);
        TORCH_CHECK(MegaMoeBF16Impl::ForwardDispatchUbLayout::kCountResetChunkBytes <= ub_size,
                    "MegaMoe BF16 count reset chunk exceeds device UB capacity");
    }

    const float normalizedSwigluLimit = ClampLimit(swiglu_limit);
    const uint32_t saveMode = SaveMode(save_gmm1_output);
    if (saveMode == kSaveMxFp8) {
        const uint64_t quantUbBytes = SaveGateUpClipMasks(normalizedSwigluLimit, saveMode)
                                          ? SavedGateUpQuantUbLayout::totalBytesWithClipMasks
                                          : SavedGateUpQuantUbLayout::totalBytes;
        const uint64_t ubBudget = std::min<uint64_t>(ub_size, 248ULL * 1024ULL);
        TORCH_CHECK(quantUbBytes <= ubBudget, "MegaMoe BF16 saved MXFP8 epilogue requires ", quantUbBytes,
                    " UB bytes, have ", ubBudget);
        TORCH_CHECK((n / 2) % 32 == 0, "saved MXFP8 requires F aligned to 32");
    }

    const uint32_t blocks_per_ep = std::max<uint32_t>(1, aic_num / world_size);
    tiling.m = m;
    tiling.k = k;
    tiling.n = n;
    tiling.topK = topk;
    tiling.epWorldSize = world_size;
    tiling.expertPerRank = expert_per_rank;
    tiling.blockNumPerEP = blocks_per_ep;
    tiling.maxOutputSize = max_recv_tokens;
    tiling.rankId = rank;
    tiling.dispatchRows = 2;
    tiling.groupListType = 1;
    tiling.transW = tiling.transW2 = true;
    tiling.swigluLimit = normalizedSwigluLimit;
    tiling.savedGateUpMode = saveMode;
    TORCH_CHECK(topk_weight_type <= kTopkWeightFp32, "unsupported TopK weight type");
    tiling.topkWeightType = topk_weight_type;
    tiling.mGroupsPerWave = Wave::GroupsPerWave(aic_num, static_cast<uint32_t>(n / 2), static_cast<uint32_t>(k));
    TORCH_CHECK(aic_num <= Wave::kMaxCores, "BF16 forward supports at most ", Wave::kMaxCores, " AIC groups");
    TORCH_CHECK(Wave::ScratchFits(static_cast<uint32_t>(k), static_cast<uint32_t>(world_size),
                                  static_cast<uint32_t>(expert_per_rank)),
                "BF16 forward Dispatch/Combine buffers and routing metadata exceed the AIV1 UB budget");
    tiling.compactTokenCountExchange = UseCompactTokenCountExchange(world_size, expert_per_rank);
    const auto workspaceBytes = optiling::CalcWorkspaceSize(tiling);
    const CompactTokenCountExchangeLayout counts(world_size, expert_per_rank, tiling.compactTokenCountExchange != 0);
    const PeerPayloadLayout payload(counts.payloadOffset, m, k, topk, TopkWeightBytes(topk_weight_type));
    const auto peerBytes = tiling.routeWeightBeforeGmm2 ? payload.totalBytes : payload.topkGradOffset;
    return {workspaceBytes + optiling::CalcBf16EarlyTopkWeightWorkspaceSize(tiling), peerBytes};
}

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;

void InitializeMegaMoeBF16Context(torch::Tensor context, void* symmetric_buffer, int rank, int world_size)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    const std::size_t context_bytes = static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(Mc2Aclnn::Bf16MoeContext), "MegaMoe BF16 context tensor is too small");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    Mc2Aclnn::Bf16MoeContext host_context{};
    host_context.epRankId = static_cast<uint32_t>(rank);
    host_context.rankSizePerServer = static_cast<uint32_t>(world_size);
    deepep::megamoe::FillPeerAddresses(host_context.epHcclBuffer, symmetric_buffer, world_size);
    const auto copy_status =
        aclrtMemcpy(context.data_ptr(), context_bytes, &host_context, sizeof(host_context), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_status == ACL_SUCCESS, "failed to initialize MegaMoe BF16 communication context, acl error ",
                copy_status);
}

uint64_t ForwardTransientWorkspaceBytes(const MegaMoeBf16TilingData& tiling)
{
    using namespace MegaMoeBF16Impl;
    const Wave::ForwardScratch scratch(tiling.maxOutputSize, tiling.mGroupsPerWave, tiling.forwardSchedule);
    const ForwardTransientWorkspaceLayout transient(tiling.n, tiling.expertPerRank, scratch.hpRows);
    const auto expertWeightBytes = tiling.routeWeightBeforeGmm2 ? TopkWeightBytes(tiling.topkWeightType) : 0U;
    const Wave::ForwardWorkspaceLayout layout(transient.totalBytes, tiling.maxOutputSize, tiling.expertPerRank,
                                              tiling.k, expertWeightBytes, tiling.m, tiling.mGroupsPerWave,
                                              scratch.rows);
    const uint64_t bytes = layout.totalBytes;
    TORCH_CHECK(bytes <= uint64_t(std::numeric_limits<int64_t>::max()), "forward workspace exceeds int64 capacity");
    return bytes;
}

torch::Tensor AllocateForwardTransientWorkspace(const torch::Tensor& like, const MegaMoeBf16TilingData& tiling)
{
    return torch::empty({static_cast<int64_t>(ForwardTransientWorkspaceBytes(tiling))},
                        like.options().dtype(torch::kUInt8));
}

std::size_t MegaMoeWindowControlRegionBytes(int64_t worldSize, int64_t expertPerRank, bool compactTokenCountExchange)
{
    // Clear the optional compact inbox and the unchanged dense token table.
    // Forward/backward payload starts at payloadOffset and is never cleared.
    const MegaMoeBF16Impl::CompactTokenCountExchangeLayout countLayout(
        static_cast<uint64_t>(worldSize), static_cast<uint64_t>(expertPerRank), compactTokenCountExchange);
    return static_cast<std::size_t>(
        MegaMoeBF16Impl::AlignUp(countLayout.payloadOffset, MegaMoeBF16Impl::kAlignment512));
}

}  // namespace

namespace deepep::megamoe::bf16 {

uint32_t BF16Plan::TopkWeightType(torch::ScalarType dtype)
{
    TORCH_CHECK(dtype == torch::kBFloat16 || dtype == torch::kFloat32, "topk_weights must be bfloat16 or float32");
    return dtype == torch::kFloat32 ? MegaMoeBF16Impl::kTopkWeightFp32 : MegaMoeBF16Impl::kTopkWeightBf16;
}

MegaMoeBF16Impl::TrainingSavedStateLayout BF16Plan::MakeSavedStateLayout(const MegaMoeBf16TilingData& tiling, int64_t k,
                                                                         int64_t n, int64_t max_recv_tokens)
{
    using namespace MegaMoeBF16Impl;
    const auto mode = tiling.savedGateUpMode;
    return TrainingSavedStateLayout(tiling.m, k, tiling.topK, tiling.epWorldSize, tiling.expertPerRank, max_recv_tokens,
                                    n, mode == kSaveMxFp8, mode != kSaveNone,
                                    SaveGateUpClipMasks(tiling.swigluLimit, mode), false,
                                    TopkWeightBytes(tiling.topkWeightType));
}

BF16Plan::WeightLayout BF16Plan::ResolveWeightLayout(const torch::Tensor& weight1, const torch::Tensor& weight2,
                                                     int64_t k)
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

void BF16Plan::ClearWindowControl() const
{
    const auto worldSize = tiling_.epWorldSize;
    const auto expertPerRank = tiling_.expertPerRank;
    const bool compactTokenCountExchange = tiling_.compactTokenCountExchange != 0;
    const auto symmetricBufferBytes = SymmetricBufferBytes();
    const std::size_t controlBytes =
        MegaMoeWindowControlRegionBytes(worldSize, expertPerRank, compactTokenCountExchange);
    const std::size_t clearOffset = static_cast<std::size_t>(MegaMoeBF16Impl::kPeerDataOffset);
    TORCH_CHECK(controlBytes <= symmetricBufferBytes,
                "MegaMoe BF16 SHMEM control region exceeds the symmetric buffer: need ", controlBytes, " bytes, got ",
                symmetricBufferBytes);
    TORCH_CHECK(clearOffset <= controlBytes, "invalid MegaMoe BF16 SHMEM control clear range");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemsetAsync(static_cast<uint8_t*>(SymmetricBuffer()) + clearOffset,
                                             symmetricBufferBytes - clearOffset, 0, controlBytes - clearOffset,
                                             stream));
}

torch::Tensor BF16Plan::AllocateTimer(const torch::Tensor& like, const torch::Tensor& weights,
                                      const torch::Tensor& weight1, const torch::Tensor& weight2, int64_t maxRecv,
                                      int phase, const std::string& mode)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before capturing training timers");
#endif
    using namespace MegaMoeTrainTimer;
    const int64_t h = like.size(1), e = weight1.size(0);
    const auto w = ResolveWeightLayout(weight1, weight2, h);
    const int64_t f = w.n / 2;
    const int64_t groups = platform_.aic;
    TORCH_CHECK(groups > 0 && groups <= kMaxCoreGroups, "unsupported Timer AIC count");
    const int64_t rows = phase == 0 ? 192 : 128;
    const int64_t tiles = CeilDiv(maxRecv + e * (rows - 1), rows) * CeilDiv(std::max(h, 2 * f), int64_t{256});
    const int64_t dwTiles = e * std::max(CeilDiv(2 * f, int64_t{128}) * CeilDiv(h, int64_t{256}),
                                         CeilDiv(h, int64_t{128}) * CeilDiv(f, int64_t{256}));
    const int64_t gemmTasks = CeilDiv(std::max(tiles, phase == 0 ? int64_t(0) : dwTiles), groups);
    // Short dP sends from every peer/expert pair can all be owned by the first AIV.
    const int64_t sendTasks = phase == 0 ? 0 : int64_t(world_size_) * e;
    const int64_t bound = std::max(gemmTasks, sendTasks) + 4 * e + 32;
    const int capacity = static_cast<int>(CeilDiv(std::max(bound, int64_t(128)), int64_t{64}) * 64);
    TORCH_CHECK(capacity <= kMaxCapacity, "training Timer task capacity exceeds supported bound");
    const Layout layout{static_cast<int>(groups * 3), capacity};
    const bool backwardWaves = phase == 1 && MegaMoeBF16Impl::UseBackwardStagedWave(SaveMode(mode), f);
    const bool withWaves = phase == 0 || backwardWaves;
    const int64_t elements = layout.Elements(withWaves);
    TORCH_CHECK(elements * 8 <= 256LL * 1024 * 1024, "training Timer exceeds 256 MiB per phase");
    auto timer = torch::zeros({elements}, like.options().dtype(torch::kInt64));
    const int64_t saved = mode == "none" ? 0 : (mode == "bf16" ? 1 : 2);
    constexpr int64_t kDecoupledSwiglu = 1LL << 8;
    const int64_t flags = saved | kDecoupledSwiglu | (w.transW ? 1024 : 0);
    std::vector<int64_t> values{kMagic,
                                phase == 0 ? kForwardWaveVersion : (backwardWaves ? kBackwardWaveVersion : kVersion),
                                groups * 3,
                                capacity,
                                phase,
                                FIXED_TIMING_COUNT,
                                DYNAMIC_TYPE_COUNT,
                                kMetadataWords,
                                maxRecv,
                                like.size(0),
                                h,
                                f,
                                e,
                                weights.size(1),
                                weights.scalar_type() == torch::kFloat32 ? 1 : 0,
                                flags};
    auto header = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
    timer.narrow(0, 0, kHeaderWords).copy_(header);
    return timer;
}

BF16Plan::BF16Plan(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like,
                   bool weight_before_gmm2)
    : address_(address), bytes_(bytes), rank_(rank), world_size_(world_size), weight_before_gmm2_(weight_before_gmm2)
{
    TORCH_CHECK(address != 0 && bytes > 0 && bytes % kDeviceLargePageBytes == 0, "invalid borrowed training window");
    TORCH_CHECK(rank >= 0 && rank < world_size && world_size <= 1024, "invalid training rank/world size");
    TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == world_size,
                "training identity disagrees with the root Session");
    ffts_addr_ = shmemx_get_ffts_config();
    (void)Context(like);
    platform_ = GetPlatformInfo();
}

int64_t buffer_size(int64_t max_tokens, int64_t hidden, int64_t intermediate, int64_t topk, int64_t world,
                    int64_t local_experts, int64_t max_recv, bool need_grad)
{
    MegaMoeBf16TilingData tiling{};
    tiling.routeWeightBeforeGmm2 = need_grad ? 1U : 0U;
    const auto host = MakeBf16Tiling(tiling, max_tokens, hidden, intermediate, topk, world, local_experts, max_recv, 0,
                                     0.0, "bf16", MegaMoeBF16Impl::kTopkWeightFp32, GetPlatformInfo());
    const uint64_t total = host.peer_bytes;
    TORCH_CHECK(total <= uint64_t(std::numeric_limits<int64_t>::max()) - kDeviceLargePageBytes,
                "BF16 symmetric capacity overflow");
    return static_cast<int64_t>(deepep::megamoe::AlignUp(total, uint64_t{kDeviceLargePageBytes}));
}

void BF16Plan::prepare(int64_t m, int64_t h, int64_t f, int64_t topk, int64_t local_experts, int64_t max_recv,
                       double limit, const std::string& save_type, bool fp32_p)
{
    TORCH_CHECK(address_ != 0, "BF16 Plan has been closed");
    MegaMoeBf16TilingData tiling{};
    static_assert(DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE == 0 || DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE == 1,
                  "BF16 training forward schedule must be 0 (interleaved) or 1 (staged)");
    tiling.routeWeightBeforeGmm2 = weight_before_gmm2_ ? 1U : 0U;
    tiling.forwardSchedule = weight_before_gmm2_ ? DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE : 0U;
    const auto host =
        MakeBf16Tiling(tiling, m, h, f, topk, world_size_, local_experts, max_recv, rank_, limit,
                       weight_before_gmm2_ ? save_type : "none",
                       fp32_p ? MegaMoeBF16Impl::kTopkWeightFp32 : MegaMoeBF16Impl::kTopkWeightBf16, platform_);
    TORCH_CHECK(host.peer_bytes <= SymmetricBufferBytes(), "BF16 symmetric window is too small");
    const auto savedStateLayout = MakeSavedStateLayout(tiling, h, tiling.n, max_recv);
    const MegaMoeBF16Impl::DispatchOutputLayout dispatchOutputLayout(
        h, max_recv, MegaMoeBF16Impl::TopkWeightBytes(tiling.topkWeightType));
    const MegaMoeBF16Impl::ForwardTransientWorkspaceLayout transientWorkspaceLayout(tiling.n, local_experts, max_recv);
    const uint64_t perRankTokenCountStride =
        MegaMoeBF16Impl::AlignUp(world_size_ * local_experts, MegaMoeBF16Impl::kAlignment128);
    const uint64_t legacyTrainingTailBytes = MegaMoeBF16Impl::AlignUp(
        world_size_ * perRankTokenCountStride * MegaMoeBF16Impl::kInt32Bytes, MegaMoeBF16Impl::kAlignment512);
    TORCH_CHECK(savedStateLayout.legacyBytes + dispatchOutputLayout.totalBytes + transientWorkspaceLayout.totalBytes ==
                    host.workspace_bytes + legacyTrainingTailBytes,
                "MegaMoe BF16 split workspace layout does not match the legacy "
                "forward workspace capacity");

    auto workspace = AllocateForwardTransientWorkspace(context_, tiling);
    tiling_ = tiling;
    workspace_ = std::move(workspace);
}

MegaMoeBf16TilingData BF16Plan::TilingForCall(int64_t m, int64_t k, int64_t n, int64_t topk, int64_t experts,
                                              int64_t max_recv, double limit, const std::string& mode,
                                              uint32_t weight_type, bool trans_w)
{
    if (!workspace_.defined() || tiling_.m != m || tiling_.k != k || tiling_.n != n || tiling_.topK != topk ||
        tiling_.expertPerRank != experts || tiling_.maxOutputSize != max_recv ||
        tiling_.topkWeightType != weight_type || tiling_.savedGateUpMode != SaveMode(mode) ||
        tiling_.swigluLimit != ClampLimit(limit)) {
        prepare(m, k, n / 2, topk, experts, max_recv, limit, mode, weight_type == MegaMoeBF16Impl::kTopkWeightFp32);
    }
    auto tiling = tiling_;
    tiling.transW = tiling.transW2 = trans_w;
    return tiling;
}

void BF16Plan::close()
{
    address_ = 0;
    context_ = torch::Tensor();
    workspace_ = torch::Tensor();
}

torch::Tensor BF16Plan::Context(const torch::Tensor& like)
{
    TORCH_CHECK(like.defined(), "MegaMoe BF16 context reference tensor must be defined");
    TORCH_CHECK(like.device().type() == c10::DeviceType::PrivateUse1,
                "MegaMoe BF16 context reference tensor must be an NPU tensor");
    TORCH_CHECK(like.is_contiguous(), "MegaMoe BF16 context reference tensor must be contiguous");
    TORCH_CHECK(like.scalar_type() == torch::kBFloat16, "MegaMoe BF16 context reference tensor must be bfloat16");
    const int rank = rank_;
    const int world_size = world_size_;
    if (!context_.defined()) {
        auto new_context =
            torch::empty({static_cast<int64_t>(sizeof(Mc2Aclnn::Bf16MoeContext))}, like.options().dtype(torch::kUInt8));
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

void* BF16Plan::SymmetricBuffer() const
{
    return reinterpret_cast<void*>(address_);
}

std::size_t BF16Plan::SymmetricBufferBytes() const
{
    return static_cast<std::size_t>(bytes_);
}

}  // namespace deepep::megamoe::bf16
