// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "kernel_launch.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"

#include <acl/acl.h>
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "tiling/platform/platform_ascendc.h"

#include "check.hpp"
#include "kernels/mega_moe_common/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#include "kernels/mega_moe_common/common/mega_moe_workspace.hpp"
#include "kernels/mega_moe_common/mc2_common/mc2_moe_context.hpp"

namespace deepep::megamoe {
namespace {

// Raw launches allocate the kernel-visible WorkspaceInfo regions,
// the AscendC system workspace, and a 50 MiB reserve.
constexpr int64_t kReservedWorkspaceBytes = 50LL * 1024LL * 1024LL;

template <typename T>
T CeilDiv(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
T AlignUp(T value, T alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint32_t DispatchCopyBytes(const MegaMoeTilingData& t)
{
    const uint32_t token = AlignUp(t.h, uint32_t{256});
    const uint32_t scale = AlignUp(CeilDiv(t.h, uint32_t{32}), uint32_t{32});
    return token + scale;
}

MegaMoeDispatchBufferConfig DispatchConfig(const MegaMoeTilingData& t, uint32_t ub)
{
    MegaMoeDispatchBufferConfig c{};
    // Compact-route receive stores at most one local-expert index per source
    // token.  The old mask route used numMaxTokens * topK here.
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank);
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.copyBufferBytes = DispatchCopyBytes(t);
    const uint32_t fixed =
        AlignUp(t.epWorldSize * t.moeExpertPerRank * 4U, 32U) + AlignUp(t.moeExpertPerRank * 4U, 32U);
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_RECV_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    const uint32_t route = static_cast<uint32_t>(c.routeItemsPerBatch);
    const uint32_t without_slots = fixed + route * sizeof(int32_t);
    const uint32_t budget = ub > without_slots ? ub - without_slots : 0U;
    c.bufferCount = std::clamp(static_cast<int32_t>(budget / (c.copyBufferBytes + 32U)), MIN_DISPATCH_BUFFER_COUNT,
                               MAX_DISPATCH_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint32_t fixed_with_slots = fixed + static_cast<uint32_t>(c.bufferCount) * (c.copyBufferBytes + 32U);
        const uint32_t route_budget = ub > fixed_with_slots ? ub - fixed_with_slots : 0U;
        uint32_t expanded = route_budget / sizeof(int32_t);
        expanded = expanded / 256U * 256U;
        expanded = static_cast<uint32_t>(std::min<uint64_t>(expanded, aligned));
        if (expanded > static_cast<uint32_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
        }
    }
    return c;
}

MegaMoeSendMaskBufferConfig SendMaskConfig(const MegaMoeTilingData& t, uint32_t fixed, uint32_t owned_experts,
                                           uint32_t ub)
{
    MegaMoeSendMaskBufferConfig c{};
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank) * t.topK;
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_SEND_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    auto route_slot_bytes = [&](uint32_t route_items) {
        // A batch can begin in the middle of a token's top-k segment.  Reserve
        // one matched index per covered token, including both boundary tokens.
        const uint64_t matched = CeilDiv<uint64_t>(static_cast<uint64_t>(route_items) + t.topK - 1U, t.topK);
        const uint64_t valid_index_bytes = AlignUp<uint64_t>(matched * sizeof(int32_t), 32U);
        return static_cast<uint32_t>(route_items / 8U + valid_index_bytes);
    };
    c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(c.routeItemsPerBatch));
    const uint32_t route_bytes = static_cast<uint32_t>(c.routeItemsPerBatch) * 4U;
    const uint32_t without_masks = fixed + 2U * route_bytes;
    const uint32_t budget = ub > without_masks ? ub - without_masks : 0U;
    c.bufferCount = std::min(static_cast<int32_t>(budget / c.bufferBytes), MAX_SEND_MASK_BUFFER_COUNT);
    const uint64_t pushes = static_cast<uint64_t>(c.routeBatchCount) * owned_experts;
    if (pushes > 0 && static_cast<uint64_t>(c.bufferCount) > pushes) {
        c.bufferCount = static_cast<int32_t>(pushes);
    }
    c.bufferCount = std::max(c.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint64_t fixed_with_padding =
            static_cast<uint64_t>(fixed) + static_cast<uint64_t>(c.bufferCount) * (32U + 2U * sizeof(int32_t));
        const uint64_t route_budget = ub > fixed_with_padding ? ub - fixed_with_padding : 0U;
        const uint64_t ring_bits =
            static_cast<uint64_t>(c.bufferCount) +
            CeilDiv<uint64_t>(static_cast<uint64_t>(c.bufferCount) * sizeof(int32_t) * 8U, t.topK);
        uint64_t expanded = route_budget * 8U / (2U * sizeof(int32_t) * 8U + ring_bits);
        expanded = expanded / 256U * 256U;
        expanded = std::min<uint64_t>(expanded, aligned);
        if (expanded > static_cast<uint64_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
            c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(expanded));
        }
    }
    return c;
}

uint64_t HostFlagElements(const MegaMoeTilingData& t)
{
    const uint64_t waves = CeilDiv<uint64_t>(t.maxOutputSize, 256U);
    const uint64_t wave_slots = waves * MegaMoeImpl::INT_CACHELINE;
    uint64_t count = static_cast<uint64_t>(t.moeExpertPerRank) *
                     (wave_slots + wave_slots + static_cast<uint64_t>(MegaMoeImpl::INT_CACHELINE) * t.aicNum);
    const bool is_w4 = t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 ||
                       t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A4W4 ||
                       t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A4W4_NZ;
    if (is_w4 || (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT)) {
        // Pairwise GMM2/combine publishes one sequence per AIC.
        count += static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
    }
    if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && (t.combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT ||
                                                     t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4)) {
        // Match WorkspaceInfo: the atomic A8W4 path owns one counter per M-group,
        // while quantized Combine retains one ready slot per AIC.
        const uint64_t ready_slots_per_expert = t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT
                                                    ? wave_slots
                                                    : static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
        count += static_cast<uint64_t>(t.moeExpertPerRank) * ready_slots_per_expert;
    }
    if (t.sharedExpertNum > 0U) {
        count += CeilDiv<uint64_t>(t.bs, 256U) * t.sharedExpertNum * MegaMoeImpl::INT_CACHELINE;
        if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4) {
            count += 4ULL * t.aicNum * MegaMoeImpl::INT_CACHELINE;
        }
    }
    return count;
}

void SetSendMaskConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t reset_per_core =
        static_cast<uint32_t>(CeilDiv(HostFlagElements(t), static_cast<uint64_t>(t.blockAivNum)));
    const uint32_t reset_count = std::min(reset_per_core, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    const uint32_t reset_bytes = AlignUp(reset_count, uint32_t{MegaMoeImpl::INT32_PER_256B}) * 4U;
    const uint32_t quant_out = AlignUp(t.h, 256U) + AlignUp(CeilDiv(t.h, 32U), 32U);
    const uint32_t quant_in = AlignUp(t.h, 128U) * 2U;
    const auto sharedPolicy = MegaMoeImpl::SelectSharedExpertPolicy(
        t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4, t.sharedExpertNum, t.bs);
    const uint32_t route_cores = sharedPolicy.routeAiv1Only ? t.aicNum : t.blockAivNum;
    const uint32_t max_experts = MegaMoeImpl::CalcSendMaskExpertCapacity(&t, route_cores);
    const uint32_t accumulator = AlignUp(max_experts * 4U, 32U);
    const uint32_t fixed = reset_bytes + 2048U + 2U * quant_out + 2U * quant_in + accumulator;
    const uint32_t experts = t.epWorldSize * t.moeExpertPerRank;
    const uint32_t base = experts / route_cores;
    t.sendMaskCoreCountWithExtraExpert = experts % route_cores;
    const bool rail = MegaMoeImpl::UseDecodeRailRouting(&t);
    t.sendMaskConfigForCoreWithExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base + 1U, ub);
    t.sendMaskConfigForCoreWithoutExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base, ub);
}

MegaMoeUnpermuteBufferConfig UnpermuteConfig(const MegaMoeTilingData& t, uint32_t core_tokens, uint32_t ub)
{
    MegaMoeUnpermuteBufferConfig c{};
    if (core_tokens == 0) return c;
    const uint32_t bf16_bytes = AlignUp(t.h * 2U, 32U);
    const uint32_t fp32_bytes = AlignUp(t.h * 4U, 32U);
    const uint32_t slot_bytes = bf16_bytes + fp32_bytes;
    c.bf16SlotElementCount = bf16_bytes / 2U;
    c.fp32SlotElementCount = fp32_bytes / 4U;
    const uint32_t base_tokens = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / t.topK;
    c.tokensPerBatch = static_cast<int32_t>(std::min(base_tokens, core_tokens));
    auto set_weight_bytes = [&]() {
        c.topKWeightsBufferBytes = AlignUp(static_cast<uint32_t>(c.tokensPerBatch) * t.topK * 4U, 32U);
        c.topKWeightsConversionBufferBytes = 0U;
    };
    set_weight_bytes();
    const uint32_t before_inputs = c.topKWeightsBufferBytes + slot_bytes;
    const uint32_t input_budget = ub > before_inputs ? ub - before_inputs : 0U;
    c.inputBufferCount = std::clamp(static_cast<int32_t>(input_budget / slot_bytes), MIN_UNPERMUTE_INPUT_BUFFER_COUNT,
                                    MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    c.inputBufferCount = std::min(c.inputBufferCount, c.tokensPerBatch * static_cast<int32_t>(t.topK));
    c.inputBufferCount = std::max(c.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);
    if (base_tokens < core_tokens) {
        const uint32_t fixed = (static_cast<uint32_t>(c.inputBufferCount) + 1U) * slot_bytes;
        const uint32_t weight_budget = ub > fixed + UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           ? ub - fixed - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           : 0U;
        const uint32_t expanded = std::min(weight_budget / (t.topK * 4U), core_tokens);
        if (expanded > static_cast<uint32_t>(c.tokensPerBatch)) {
            c.tokensPerBatch = static_cast<int32_t>(expanded);
            set_weight_bytes();
        }
    }
    return c;
}

void SetUnpermuteConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t full = CeilDiv(t.bs, t.blockAivNum);
    const uint32_t active = CeilDiv(t.bs, full);
    const uint32_t tail = t.bs - (active - 1U) * full;
    const bool tail_full = tail == full;
    t.unpermuteFullTokenChunkCoreCount = tail_full ? active : active - 1U;
    t.unpermuteConfigForFullTokenChunk = UnpermuteConfig(t, full, ub);
    t.unpermuteConfigForTailTokenChunk = tail_full ? MegaMoeUnpermuteBufferConfig{} : UnpermuteConfig(t, tail, ub);
}

PlatformInfo GetPlatformInfo()
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

MegaMoeTilingData MakeTiling(int64_t bs, int64_t h, int64_t hidden_dim, int64_t topk, int64_t local_experts,
                             int64_t world, int64_t max_tokens, int64_t max_recv, bool is_a8w4, int64_t shared_experts,
                             const PlatformInfo& p)
{
    MegaMoeTilingData t{};
    t.moeExpertPerRank = static_cast<uint32_t>(local_experts);
    t.bs = static_cast<uint32_t>(bs);
    t.h = static_cast<uint32_t>(h);
    t.hiddenDim = static_cast<uint32_t>(hidden_dim);
    t.epWorldSize = static_cast<uint32_t>(world);
    t.blockNumPerEP = std::max(1U, p.aic / t.epWorldSize);
    t.numMaxTokensPerRank = static_cast<uint32_t>(max_tokens == 0 ? bs : max_tokens);
    t.maxOutputSize = static_cast<uint32_t>(
        max_recv == 0 ? t.numMaxTokensPerRank * t.epWorldSize * std::min<uint32_t>(topk, local_experts) : max_recv);
    t.topK = static_cast<uint32_t>(topk);
    t.aicNum = p.aic;
    t.blockAivNum = p.aiv;
    t.combineQuantMode = MegaMoeImpl::COMBINE_NO_QUANT;
    t.clampLimit = std::numeric_limits<float>::max();
    t.groupedMatmulMode = is_a8w4 ? MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 : MegaMoeImpl::GROUPED_MATMUL_MODE_GENERAL;
    t.topoType = MegaMoeImpl::TOPO_TYPE_MTE;
    t.sharedExpertNum = static_cast<uint32_t>(shared_experts);
    t.combineSyncSlotCountPerExpert = 0;
    t.topkWeightsPrefetch = 0;
    t.maxTilesPerExpert = 0;
    t.actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    t.actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    t.activationAlpha = 1.0F;
    t.activationBeta = 1.0F;
    // 按本次每卡 BS 选择，不用预分配容量；小 batch 缩短 wave，较早释放 GMM2。
    if (is_a8w4 && t.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        const uint64_t gmm1_tasks_per_m_group =
            CeilDiv<uint64_t>(t.hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF, MegaMoeImpl::GMM1_FUSED_TILE_N);
        t.mGroupsPerWave =
            static_cast<uint32_t>(CeilDiv<uint64_t>(static_cast<uint64_t>(p.aic) * 2U, gmm1_tasks_per_m_group));
    } else {
        // 大 batch 和 A8W8 的预算包括 GMM2 的最低任务量约束。
        const uint64_t gmm1_tiles = CeilDiv<uint64_t>(t.hiddenDim, 256U);
        const uint64_t gmm2_tiles = CeilDiv<uint64_t>(t.h, 256U);
        t.mGroupsPerWave = static_cast<uint32_t>(std::max(
            CeilDiv<uint64_t>(static_cast<uint64_t>(p.aic) * 4U, gmm1_tiles), CeilDiv<uint64_t>(p.aic, gmm2_tiles)));
    }
    t.isPerExpertWeightTensor = false;
    t.dispatchBufferConfig = DispatchConfig(t, p.ub);
    SetSendMaskConfigs(t, p.ub);
    SetUnpermuteConfigs(t, p.ub);
    return t;
}

void CheckTensor(const torch::Tensor& t, int64_t dims, const char* name)
{
    TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "D");
}

using MegaMoeUrma::Config;
constexpr uint64_t kAlignment = 512;
uint64_t Align(uint64_t value, uint64_t alignment = kAlignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

Config UbxLayout(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                 int64_t cv_rows)
{
    TORCH_CHECK((world == 4 || world == 8 || world == 16) && max_tokens > 0 &&
                    max_tokens <= MegaMoeUrma::kMaxTokensPerRank && topk >= 1 && topk <= 8 && hidden >= 256 &&
                    hidden <= 8192 && hidden % 256 == 0,
                "MegaMoE URMA v1 requires EP4/EP8/EP16, H256..8192 aligned256, K1..8 and max_tokens in [1,16384]");
    TORCH_CHECK(max_recv_rows == world * max_tokens * topk,
                "Device routing requires worst-case receive capacity world*max_tokens*topk");
    Config config{};
    config.world = static_cast<uint32_t>(world);
    config.hidden = hidden;
    config.topk = topk;
    config.cvRows = cv_rows;
    config.maxPeerChunks = (max_tokens * topk + cv_rows - 1) / cv_rows;
    config.sourceRowCapacity = static_cast<uint64_t>(max_tokens) * topk;
    config.receiveXOffset = 0;
    config.receiveScaleOffset = Align(static_cast<uint64_t>(max_recv_rows) * hidden);
    config.returnYOffset = Align(config.receiveScaleOffset + static_cast<uint64_t>(max_recv_rows) * (hidden / 32));
    config.dispatchReadyOffset = Align(config.returnYOffset + static_cast<uint64_t>(max_tokens) * topk * hidden * 2);
    config.combineReadyOffset =
        Align(config.dispatchReadyOffset +
              static_cast<uint64_t>(world) * ((max_tokens * topk + 127) / 128) * MegaMoeUrma::kFlagBytes);
    config.barrierOffset =
        Align(config.combineReadyOffset +
              static_cast<uint64_t>(world) * ((max_tokens * topk + 127) / 128) * MegaMoeUrma::kFlagBytes);
    config.routeInboxOffset = Align(config.barrierOffset + 4ULL * world * MegaMoeUrma::kFlagBytes);
    config.routeEntryOffset = Align(config.routeInboxOffset + world * MegaMoeUrma::kRouteMailboxBytes);
    config.routeReadyOffset = config.routeEntryOffset + world * MegaMoeUrma::kFlagBytes;
    config.symmetricBytes = Align(config.routeReadyOffset + world * MegaMoeUrma::kFlagBytes, 2ULL << 20);
    return config;
}

uint64_t Address(const torch::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}
uint32_t SelectCv(int64_t world, int64_t bs)
{
    return world == 16 && bs <= 2560 ? 128U : 256U;
}

}  // namespace

struct UbxWorkspace {
    MegaMoeUrma::Arguments args_{};
    torch::Tensor expanded_x_, expanded_scale_, gmm1_, activation_, activation_scale_, gmm2_;
    torch::Tensor gathered_input_scale_, input_scale_done_;
    torch::Tensor gmm_issued_, gmm2_contiguous_, peer_reorder_ranges_, peer_reorder_spans_;
    torch::Tensor fragments_, send_chunks_, recv_chunks_, gmm1_done_, activation_done_, gmm2_done_, local_flags_;
    torch::Tensor segments_, compute_batches_, compute_tasks_, chunk_group_indices_;
    torch::Tensor pack_rounds_, pack_ready_;
    torch::Tensor reduce_dependencies_, order_, inverse_, route_counts_, route_offsets_;
    torch::Tensor route_histogram_, route_info_, route_config_, expert_routes_;
    torch::Tensor system_workspace_;
    torch::Tensor shared_gmm1_, shared_activation_, shared_scale_, shared_output_, shared_routed_output_, shared_done1_,
        shared_done_act_;

    UbxWorkspace(torch::Tensor example, int64_t world, int64_t intermediate, int64_t topk, int64_t cv_rows,
                 int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address, int64_t shared_experts,
                 int64_t shared_intermediate, int64_t num_experts, const PlatformInfo& platform)
    {
        const int64_t max_recv_rows = world * max_tokens * topk;
        args_.config = UbxLayout(world, max_tokens, topk, example.size(1), max_recv_rows, cv_rows);
        auto& config = args_.config;
        config.bs = example.size(0);
        config.intermediate = intermediate;
        config.localExperts = num_experts / world;
        config.rank = rank;
        config.gmm1AheadMGroups = 12U;
        args_.sharedExperts = shared_experts;
        args_.sharedIntermediate = shared_experts ? shared_intermediate : 0;
        args_.symmetric = symmetric_address;
        args_.ffts = ffts;
        const auto options = example.options().dtype(torch::kUInt8);
        const uint64_t expanded_rows = static_cast<uint64_t>(max_tokens) * topk;
        const uint64_t groups = MegaMoeUrma::RouteGroupCapacity(max_recv_rows, config.localExperts);
        const uint64_t chunks = uint64_t(world) * config.maxPeerChunks;
        auto allocate = [&](uint64_t bytes) {
            return torch::empty({static_cast<int64_t>(std::max<uint64_t>(bytes, 512))}, options);
        };
        if (args_.sharedExperts) {
            const uint64_t si = args_.sharedIntermediate, groups = (max_tokens + 255U) / 256U;
            shared_gmm1_ = allocate(max_tokens * si * 4);
            shared_activation_ = allocate(max_tokens * si);
            shared_scale_ = allocate(max_tokens * (si / 32));
            shared_output_ = allocate(uint64_t(max_tokens) * config.hidden * 2);
            shared_routed_output_ = allocate(uint64_t(max_tokens) * config.hidden * 4);
            args_.sharedRoutedOutput = Address(shared_routed_output_);
            shared_done1_ = allocate(groups * (si / 256) * MegaMoeUrma::kFlagBytes);
            shared_done_act_ = allocate(groups * MegaMoeUrma::kFlagBytes);
            args_.sharedGmm1 = Address(shared_gmm1_);
            args_.sharedActivation = Address(shared_activation_);
            args_.sharedActivationScale = Address(shared_scale_);
            args_.sharedOutput = Address(shared_output_);
            args_.sharedGmm1Done = Address(shared_done1_);
            args_.sharedActivationDone = Address(shared_done_act_);
        }
        expanded_x_ = allocate(expanded_rows * config.hidden);
        expanded_scale_ = allocate(expanded_rows * (config.hidden / 32));
        gathered_input_scale_ = allocate(max_recv_rows * (config.hidden / 32));
        gmm1_ = allocate(0);  // routed fused GMM1 writes to paired UB
        activation_ = allocate(max_recv_rows * intermediate);
        activation_scale_ = allocate(max_recv_rows * (intermediate / 32));
        gmm2_ = allocate(max_recv_rows * config.hidden * 2);
        fragments_ = allocate(0);  // lazy descriptor; placeholder for lifetime bookkeeping
        segments_ = allocate(0);   // lazy descriptor; placeholder for lifetime bookkeeping
        compute_batches_ = allocate(groups * sizeof(MegaMoeUrma::ComputeBatch));
        compute_tasks_ = allocate(2 * groups * sizeof(MegaMoeUrma::ComputeBatchTask));
        chunk_group_indices_ = allocate(0);  // lazy descriptor; placeholder for lifetime bookkeeping
        send_chunks_ = allocate(0);          // lazy descriptor; placeholder for lifetime bookkeeping
        recv_chunks_ = allocate(0);          // lazy descriptor; placeholder for lifetime bookkeeping
        pack_rounds_ = allocate(0);          // lazy descriptor; placeholder for lifetime bookkeeping
        pack_ready_ = allocate(chunks * MegaMoeUrma::kFlagBytes);
        gmm1_done_ = allocate(0);  // FIFO hardware completion replaces per-tile GM flags
        input_scale_done_ = allocate(groups * MegaMoeUrma::kFlagBytes);
        activation_done_ = allocate(groups * MegaMoeUrma::kFlagBytes);
        gmm2_done_ = allocate(groups * MegaMoeUrma::kFlagBytes);
        order_ = allocate(0);  // compact order exists only in explicit post-run diagnostics
        inverse_ = torch::empty({int64_t(expanded_rows)}, example.options().dtype(torch::kInt64));
        reduce_dependencies_ = allocate(0);  // Reduce constructs its own UB dependency cache
        expert_routes_ = torch::empty({num_experts, int64_t(expanded_rows)}, example.options().dtype(torch::kInt32));
        route_counts_ = torch::empty({num_experts}, example.options().dtype(torch::kInt64));
        route_offsets_ = torch::empty({num_experts + 1}, example.options().dtype(torch::kInt64));
        route_histogram_ = allocate(num_experts * sizeof(uint32_t));
        route_info_ = allocate(Align(sizeof(MegaMoeUrma::DeviceRouteInfo)));
        route_config_ = allocate(Align(sizeof(Config)));
        args_.order = Address(order_);
        args_.reduceIndex = Address(inverse_);
        args_.expertRoutes = Address(expert_routes_);
        args_.reduceDependencies = Address(reduce_dependencies_);
        args_.routeCounts = Address(route_counts_);
        args_.routeOffsets = Address(route_offsets_);
        args_.routeHistogram = Address(route_histogram_);
        args_.routeInfo = Address(route_info_);
        args_.routeConfig = Address(route_config_);
        gmm2_contiguous_ = allocate(0);      // routed GMM2 writes directly to paired UB
        peer_reorder_ranges_ = allocate(0);  // lazy descriptor; placeholder for lifetime bookkeeping
        peer_reorder_spans_ = allocate(0);   // lazy descriptor; placeholder for lifetime bookkeeping
        local_flags_ = allocate((shared_experts ? MegaMoeUrma::kSharedLocalFlagCount : 16U) * MegaMoeUrma::kFlagBytes);
        gmm_issued_ = allocate(MegaMoeUrma::kAic * MegaMoeUrma::kFlagBytes);
        args_.gmmIssued = Address(gmm_issued_);
        system_workspace_ = allocate(platform.system_workspace + (2ULL << 20));

        args_.expandedX = Address(expanded_x_);
        args_.expandedScale = Address(expanded_scale_);
        args_.gatheredInputScale = Address(gathered_input_scale_);
        args_.gmm1Output = Address(gmm1_);
        args_.activation = Address(activation_);
        args_.activationScale = Address(activation_scale_);
        args_.gmm2Output = Address(gmm2_);
        args_.gmm2Contiguous = Address(gmm2_contiguous_);
        args_.fragments = Address(fragments_);
        args_.segments = Address(segments_);
        args_.computeBatches = Address(compute_batches_);
        args_.computeTasks = Address(compute_tasks_);
        args_.chunkGroupIndices = Address(chunk_group_indices_);
        args_.peerReorderRanges = Address(peer_reorder_ranges_);
        args_.peerReorderSpans = Address(peer_reorder_spans_);
        args_.sendChunks = Address(send_chunks_);
        args_.recvChunks = Address(recv_chunks_);
        args_.packRounds = Address(pack_rounds_);
        args_.packReady = Address(pack_ready_);
        args_.gmm1Done = Address(gmm1_done_);
        args_.inputScaleDone = Address(input_scale_done_);
        args_.activationDone = Address(activation_done_);
        args_.gmm2Done = Address(gmm2_done_);
        args_.gmm2MatmulDone = 0;  // reserved ABI slot; direct same-core completion -> Reorder
        args_.localFlags = Address(local_flags_);
        args_.systemWorkspace = Address(system_workspace_);
    }

    template <class Visitor>
    void for_each_tensor(Visitor&& visit) const
    {
        for (const auto* tensor : {&expanded_x_,
                                   &expanded_scale_,
                                   &gathered_input_scale_,
                                   &gmm1_,
                                   &activation_,
                                   &activation_scale_,
                                   &gmm2_,
                                   &gmm2_contiguous_,
                                   &gmm_issued_,
                                   &fragments_,
                                   &segments_,
                                   &compute_batches_,
                                   &compute_tasks_,
                                   &chunk_group_indices_,
                                   &peer_reorder_ranges_,
                                   &peer_reorder_spans_,
                                   &order_,
                                   &inverse_,
                                   &route_counts_,
                                   &route_offsets_,
                                   &route_histogram_,
                                   &route_info_,
                                   &route_config_,
                                   &expert_routes_,
                                   &reduce_dependencies_,
                                   &send_chunks_,
                                   &recv_chunks_,
                                   &pack_rounds_,
                                   &pack_ready_,
                                   &gmm1_done_,
                                   &input_scale_done_,
                                   &activation_done_,
                                   &gmm2_done_,
                                   &local_flags_,
                                   &system_workspace_,
                                   &shared_gmm1_,
                                   &shared_activation_,
                                   &shared_scale_,
                                   &shared_output_,
                                   &shared_routed_output_,
                                   &shared_done1_,
                                   &shared_done_act_})
            visit(*tensor);
    }
};

Plan::Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world,
           int64_t max_tokens, bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts,
           uint64_t symmetric_address, int64_t symmetric_bytes, const std::string& backend)
    : backend_(backend == "pod" ? Backend::Pod : Backend::Ubx),
      ffts_(ffts),
      example_(example),
      intermediate_(intermediate),
      topk_(topk),
      experts_(experts),
      world_(world),
      max_tokens_(max_tokens),
      shared_experts_(shared_experts),
      is_a8w4_(is_a8w4),
      symmetric_address_(symmetric_address),
      symmetric_bytes_(symmetric_bytes)
{
    TORCH_CHECK(backend == "pod" || backend == "ubx", "MegaMoE backend must be pod or ubx");
    CheckTensor(example, 2, "plan input");
    TORCH_CHECK(example.size(0) > 0 && example.size(0) <= max_tokens, "BS outside allocated capacity");
    TORCH_CHECK(ranks_per_node == 0 || (ranks_per_node > 0 && world % ranks_per_node == 0),
                "ranks_per_node must divide EP");
    const bool ubx = backend_ == Backend::Ubx;
    if (ubx) {
        TORCH_CHECK(is_a8w4 && intermediate >= 256 && intermediate <= 16384 && intermediate % 256 == 0,
                    "UBX supports FP8xFP4 with I256..16384 aligned256");
        TORCH_CHECK(shared_experts == 0 || shared_experts == 1, "UBX supports S0/S1");
    } else {
        TORCH_CHECK(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                    "Plan intermediate_hidden must be divisible by 128, in [256,4096]");
        TORCH_CHECK(shared_experts >= 0 && shared_experts <= 8 && (is_a8w4 || shared_experts == 0),
                    "Plan shared experts require W4A8, with count in [0,8]");
    }
    const auto required = buffer_size(world, experts, max_tokens, topk, example.size(1), backend);
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= required, "MegaMoE symmetric buffer needs ", required,
                " bytes, got ", symmetric_bytes);
    platform_ = GetPlatformInfo();
    if (ubx) {
        TORCH_CHECK(platform_.aic >= MegaMoeUrma::kAic && platform_.aiv >= 2U * MegaMoeUrma::kAic &&
                        platform_.ub >= MegaMoeUrma::kRequiredUbBytes,
                    "UBX requires 28 AIC/56 AIV and 248 KiB usable UB");
        auto* device_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platform_.aic = MegaMoeUrma::kAic;
        platform_.aiv = 2U * MegaMoeUrma::kAic;
        platform_.block_dim = device_platform->CalcTschBlockDim(platform_.aiv, platform_.aic, platform_.aiv);
        TORCH_CHECK(aclshmem_n_pes() == world, "MegaMoE plan does not match its SHMEM runtime");
    }

    // Both backends resolve peer mappings once during plan preparation.
    // Only POD uploads this context; UBX passes the symmetric base in its ABI.
    Mc2Aclnn::Mc2MoeContext context{};
    const int rank = aclshmem_my_pe();
    TORCH_CHECK(rank >= 0 && rank < world && world <= Mc2Aclnn::HCCL_MAX_RANK_SIZE, "Invalid SHMEM rank/world: ", rank,
                "/", world);
    context.epRankId = static_cast<uint32_t>(rank);
    context.rankSizePerServer = static_cast<uint32_t>(world);
    for (int64_t peer = 0; peer < world; ++peer) {
        void* address = aclshmem_ptr(reinterpret_cast<void*>(symmetric_address), static_cast<int32_t>(peer));
        TORCH_CHECK(address != nullptr, "rank=", rank, " prepare: cannot resolve peer=", peer);
        context.epHcclBuffer_[peer] = reinterpret_cast<uint64_t>(address);
    }
    if (!ubx) {
        tiling_ = MakeTiling(example.size(0), example.size(1), intermediate * 2, topk, experts, world, max_tokens, 0,
                             is_a8w4, shared_experts, platform_);
        tiling_.dispatchRanksPerNode = static_cast<uint32_t>(ranks_per_node);
        static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
        const MegaMoeImpl::WorkspaceInfo info(nullptr, &tiling_);
        const int64_t bytes =
            info.workspaceSize + static_cast<int64_t>(platform_.system_workspace) + kReservedWorkspaceBytes;
        TORCH_CHECK(bytes > info.workspaceSize, "MegaMoE workspace size overflow");
        workspace_ = torch::empty({bytes}, example.options().dtype(torch::kUInt8));
        context_ = torch::empty({static_cast<int64_t>(sizeof(context))}, workspace_.options());
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemcpy(context_.data_ptr(), sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    // UBX workspace is prepared in forward once the shared tensor dimensions
    // are available. POD keeps its existing eager allocation in prepare.
}

Plan::~Plan() = default;

int64_t Plan::workspace_bytes() const
{
    if (backend_ == Backend::Pod) return workspace_.defined() ? workspace_.nbytes() : 0;
    int64_t bytes = 0;
    if (ubx_)
        ubx_->for_each_tensor([&](const torch::Tensor& t) {
            if (t.defined()) bytes += t.nbytes();
        });
    return bytes;
}

std::tuple<torch::Tensor, torch::Tensor> Plan::forward(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
    c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
    c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
    c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out, int64_t generation)
{
    const bool ubx = backend_ == Backend::Ubx;
    CheckTensor(x, 2, "x");
    CheckTensor(topk_ids, 2, "topk_ids");
    CheckTensor(topk_weights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(weight_scales1, 4, "weight_scales1");
    CheckTensor(weight_scales2, 4, "weight_scales2");
    const int64_t bs = x.size(0), h = x.size(1), hidden = 2 * intermediate_;
    TORCH_CHECK(x.device() == example_.device() && bs == example_.size(0) && h == example_.size(1),
                "Input shape/device differs from prepared plan");
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.sizes() == torch::IntArrayRef({bs, topk_}),
                "topk_ids must be int32 [BS,TopK]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32 && topk_weights.sizes() == topk_ids.sizes(),
                "topk_weights must be FP32 [BS,TopK]");
    const bool prequantized = x_scales.has_value();
    TORCH_CHECK(!ubx || prequantized, "UBX requires prequantized FP8 input with E8M0 scales");
    TORCH_CHECK(x.scalar_type() == (prequantized ? c10::ScalarType::Float8_e4m3fn : torch::kBFloat16),
                "x must be BF16, or FP8 E4M3 with E8M0 scales");
    if (prequantized) {
        if (ubx && x_scales->dim() == 3) {
            CheckTensor(*x_scales, 3, "x_scales");
            TORCH_CHECK(x_scales->sizes() == torch::IntArrayRef({bs, h / 64, 2}), "Invalid UBX input scale shape");
            x_scales = x_scales->reshape({bs, h / 32});
        }
        CheckTensor(*x_scales, 2, "x_scales");
        TORCH_CHECK(x_scales->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        x_scales->sizes() == torch::IntArrayRef({bs, CeilDiv(h, int64_t{32})}),
                    "x_scales must be E8M0 [BS,H/32]");
    }
    const auto weight_type = is_a8w4_ ? torch::kUInt8 : c10::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(weight1.scalar_type() == weight_type && weight2.scalar_type() == weight_type,
                "Weights differ from prepared matrix mode");
    TORCH_CHECK(weight1.sizes() == torch::IntArrayRef({experts_, hidden, is_a8w4_ ? h / 2 : h}) &&
                    weight2.sizes() == torch::IntArrayRef({experts_, h, is_a8w4_ ? intermediate_ / 2 : intermediate_}),
                "Routed weight shape differs from plan");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales1.sizes() == torch::IntArrayRef({experts_, hidden, CeilDiv(h, int64_t{64}), 2}) &&
                    weight_scales2.sizes() == torch::IntArrayRef({experts_, h, CeilDiv(intermediate_, int64_t{64}), 2}),
                "Invalid routed weight scales");
    for (const auto* tensor : {&topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2})
        TORCH_CHECK(tensor->device() == x.device(), "MegaMoE tensors must be on the same NPU");
    for (const auto* tensor :
         {&x_scales, &timer, &out, &shared_weight1, &shared_weight2, &shared_weight_scales1, &shared_weight_scales2})
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == x.device(),
                    "Optional MegaMoE tensors must be on the same NPU");

    const bool has_shared = shared_weight1.has_value();
    TORCH_CHECK(shared_weight2.has_value() == has_shared && shared_weight_scales1.has_value() == has_shared &&
                    shared_weight_scales2.has_value() == has_shared && has_shared == (shared_experts_ > 0),
                "All shared weight/scale tensors must match the configured shared expert count");
    int64_t shared_i = intermediate_;
    if (has_shared) {
        CheckTensor(*shared_weight1, 3, "shared_weight1");
        CheckTensor(*shared_weight2, 3, "shared_weight2");
        CheckTensor(*shared_weight_scales1, 4, "shared_weight_scales1");
        CheckTensor(*shared_weight_scales2, 4, "shared_weight_scales2");
        TORCH_CHECK(shared_weight1->size(1) % 2 == 0, "Shared GMM1 must contain gate/up pairs");
        shared_i = shared_weight1->size(1) / 2;
        TORCH_CHECK(ubx ? (shared_i >= 256 && shared_i <= 16384 && shared_i % 256 == 0) : shared_i == intermediate_,
                    "Shared intermediate is unsupported for the selected backend");
        const auto shared_type = ubx ? c10::ScalarType::Float8_e4m3fn : torch::kUInt8;
        TORCH_CHECK(
            is_a8w4_ && shared_weight1->scalar_type() == shared_type && shared_weight2->scalar_type() == shared_type &&
                shared_weight1->sizes() == torch::IntArrayRef({shared_experts_, 2 * shared_i, ubx ? h : h / 2}) &&
                shared_weight2->sizes() == torch::IntArrayRef({shared_experts_, h, ubx ? shared_i : shared_i / 2}),
            "Shared weights have invalid dtype or shape for the selected backend");
        TORCH_CHECK(shared_weight_scales1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales2->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales1->sizes() ==
                            torch::IntArrayRef({shared_experts_, 2 * shared_i, CeilDiv(h, int64_t{64}), 2}) &&
                        shared_weight_scales2->sizes() ==
                            torch::IntArrayRef({shared_experts_, h, CeilDiv(shared_i, int64_t{64}), 2}),
                    "Invalid shared weight scales");
    }
    uint64_t trace_capacity = 0;
    int64_t trace_elements = 0;
    if (timer) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before passing timer");
#endif
        TORCH_CHECK(is_a8w4_, "MegaMoE timer requires routed A8W4");
        CheckTensor(*timer, 1, "timer");
        TORCH_CHECK(timer->scalar_type() == torch::kInt64, "timer must be int64");
        if (ubx) {
            constexpr int64_t row_elements = MegaMoeUrma::kTimerCores * MegaMoeUrma::kTraceRecordElements;
            trace_elements = timer->numel() - MegaMoeUrma::kTimerElements;
            TORCH_CHECK(Address(*timer) % 128U == 0 && trace_elements % row_elements == 0 &&
                            trace_elements >= 2 * row_elements &&
                            trace_elements <= (int64_t{MegaMoeUrma::kTraceMaxCapacity} + 1) * row_elements,
                        "Invalid UBX fine trace allocation");
            trace_capacity = trace_elements / row_elements - 1;
        } else
            TORCH_CHECK(timer->numel() == AscendTimer::TOTAL_BUFFER_SIZE, "Invalid POD timer allocation");
    }
    torch::Tensor y = out ? *out : torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    CheckTensor(y, 2, "out");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && y.sizes() == x.sizes(), "out must be BF16 [BS,H]");
    torch::Tensor expert_tokens = torch::empty({experts_}, x.options().dtype(torch::kInt32));
    const auto stream = c10_npu::getCurrentNPUStream();
    // Prepare backend state before assembling either device argument block.
    if (ubx) {
        TORCH_CHECK(generation > last_generation_ && generation < (1LL << 31),
                    "UBX generation must be positive and increase across the Buffer lifetime");
        if (!ubx_ || shared_intermediate_ != shared_i) {
            if (ubx_) ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream.stream()));
            ubx_.reset();
            ubx_ = std::make_unique<UbxWorkspace>(example_, world_, intermediate_, topk_, SelectCv(world_, bs),
                                                  max_tokens_, aclshmem_my_pe(), ffts_, symmetric_address_,
                                                  shared_experts_, shared_i, world_ * experts_, platform_);
            shared_intermediate_ = shared_i;
        }
        ubx_->args_.config.bs = bs;
        // Invocation flags are reset by AIV workers before the first route join.
        if (timer) {
            ASCEND_DEEPEP_CHECK_ACL(
                aclrtMemsetAsync(timer->data_ptr(), timer->nbytes(), 0, timer->nbytes(), stream.stream()));
        }
    }
    // Common dispatch stage; only the device argument representation differs.
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    auto opt = [&](const c10::optional<torch::Tensor>& t) { return t ? ptr(*t) : nullptr; };
    PodLaunchArgs pod{};
    MegaMoeUrma::Arguments* ubx_args = nullptr;
    if (ubx) {
        auto& args = ubx_->args_;
        args.config.epoch = generation;
        args.x = Address(x);
        args.xScale = Address(*x_scales);
        args.topkIds = Address(topk_ids);
        args.topkIdBytes = topk_ids.element_size();
        args.weights = Address(topk_weights);
        args.weight1 = Address(weight1);
        args.weight2 = Address(weight2);
        args.weightScale1 = Address(weight_scales1);
        args.weightScale2 = Address(weight_scales2);
        args.output = Address(y);
        args.expertCountsOutput = Address(expert_tokens);
        // Aggregate counters occupy the trailer of the same public timer buffer.
        args.timer = timer ? Address(*timer) + trace_elements * sizeof(int64_t) : 0;
        args.trace = timer ? Address(*timer) : 0;
        args.traceCapacity = trace_capacity;
        args.coarseTrace = 0;
        if (has_shared) {
            args.sharedWeight1 = Address(*shared_weight1);
            args.sharedWeight2 = Address(*shared_weight2);
            args.sharedScale1 = Address(*shared_weight_scales1);
            args.sharedScale2 = Address(*shared_weight_scales2);
        }
        ubx_args = &args;
    } else {
        pod = {is_a8w4_,
               ptr(context_),
               ptr(x),
               opt(x_scales),
               ptr(topk_ids),
               ptr(topk_weights),
               ptr(weight1),
               ptr(weight2),
               ptr(weight_scales1),
               ptr(weight_scales2),
               opt(shared_weight1),
               opt(shared_weight2),
               opt(shared_weight_scales1),
               opt(shared_weight_scales2),
               ptr(y),
               ptr(expert_tokens),
               ptr(workspace_),
               opt(timer),
               reinterpret_cast<const uint8_t*>(&tiling_),
               ffts_};
    }
    launch_kernel(backend_, platform_.block_dim, platform_.ub, stream.stream(), pod, ubx_args);
    if (ubx) last_generation_ = generation;
    // Shared lifetime handling for both backends. Input scopes and output ABI do
    // not depend on which private device parameter block was dispatched.
    auto record = [&](const torch::Tensor& t) {
        if (t.defined()) c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
    };
    for (const auto& t :
         {x, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2, y, expert_tokens})
        record(t);
    for (const auto* t :
         {&x_scales, &timer, &shared_weight1, &shared_weight2, &shared_weight_scales1, &shared_weight_scales2})
        if (t->has_value()) record(t->value());
    if (ubx)
        ubx_->for_each_tensor(record);
    else {
        record(workspace_);
        record(context_);
    }
    return {y, expert_tokens};
}

torch::Tensor allocate_timer(torch::Tensor example, const std::string& backend)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before allocating timer");
#endif
    CheckTensor(example, 2, "timer example");
    TORCH_CHECK(backend == "pod" || backend == "ubx", "MegaMoE backend must be pod or ubx");
    int64_t elements = AscendTimer::TOTAL_BUFFER_SIZE;
    if (backend == "ubx") {
        elements = int64_t{MegaMoeUrma::kTimerCores} *
                       (MegaMoeUrma::kTraceHeaderElements +
                        int64_t{MegaMoeUrma::kTraceMaxCapacity} * MegaMoeUrma::kTraceRecordElements) +
                   MegaMoeUrma::kTimerElements;
    }
    return torch::zeros({elements}, example.options().dtype(torch::kInt64));
}

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden,
                    const std::string& backend)
{
    TORCH_CHECK(backend == "pod" || backend == "ubx", "MegaMoE backend must be pod or ubx");
    if (backend == "ubx") {
        TORCH_CHECK((world == 4 || world == 8 || world == 16) && tokens >= 1 &&
                        tokens <= MegaMoeUrma::kMaxTokensPerRank && topk >= 1 && topk <= 8,
                    "Invalid UBX EP, capacity or TopK");
        TORCH_CHECK(experts >= 1 && experts <= 128 && topk <= world * experts, "Invalid UBX expert count");
        return UbxLayout(world, tokens, topk, hidden, world * tokens * topk, 128).symmetricBytes;
    }
    TORCH_CHECK(world >= 2 && world <= 128 && experts >= 1 && experts <= 128 && tokens >= 1 && tokens <= 65536 &&
                    topk >= 1 && topk <= 32 && topk <= world * experts && hidden >= 1024 && hidden <= 8192 &&
                    hidden % 64 == 0,
                "Invalid buffer shape");
    // The buffer outlives BS/weight-mode changes. Reserve the decode rail/count
    // layout as well; prefill continues to bind its original device offsets.
    const MegaMoeImpl::PeermemSizeParams params{
        tokens, topk, hidden, experts, world, 2, 1, false, false, MegaMoeImpl::TOPO_TYPE_MTE, true};
    return AlignUp<int64_t>(MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE + MegaMoeImpl::CalcPeermemLeastSize(params),
                            2LL << 20);
}

}  // namespace deepep::megamoe
