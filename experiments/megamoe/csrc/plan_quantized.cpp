// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "plan.hpp"

#include <algorithm>
#include <limits>

#include <acl/acl.h>
#include "check.hpp"
#include "shmem.h"
#include "csrc/kernels/common/mega_moe_wave_math.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"
#include "kernels/mega_moe/common/mega_moe_workspace.hpp"
#include "kernels/mega_moe/mc2_common/mc2_moe_context.hpp"

namespace deepep::megamoe {
namespace {

// Raw launches allocate the kernel-visible WorkspaceInfo regions,
// the AscendC system workspace, and a 50 MiB reserve.
constexpr int64_t kReservedWorkspaceBytes = 50LL * 1024LL * 1024LL;

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
    if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 &&
        t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT && t.sharedExpertNum == 0U) {
        count += static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
    }
    if (t.sharedExpertNum > 0U) {
        count += CeilDiv<uint64_t>(t.bs, 256U) * t.sharedExpertNum * MegaMoeImpl::INT_CACHELINE;
        if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4) {
            count += 4ULL * t.aicNum * MegaMoeImpl::INT_CACHELINE;
        }
    }
    return count + uint64_t(t.scratchWaveSlots) * MegaMoeImpl::INT_CACHELINE;
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

MegaMoeTilingData MakeTiling(int64_t bs, int64_t h, int64_t hidden_dim, int64_t topk, int64_t local_experts,
                             int64_t world, int64_t max_tokens, bool is_a8w4, int64_t shared_experts,
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
    t.maxOutputSize = InferenceMaxRecvTokens(t.numMaxTokensPerRank, t.epWorldSize, topk, local_experts);
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
        t.mGroupsPerWave = MegaMoeImpl::WaveGroups(p.aic, gmm1_tasks_per_m_group, 1U, 2U, false);
    } else {
        // 大 batch 和 A8W8 的预算包括 GMM2 的最低任务量约束。
        const uint64_t gmm1_tiles = CeilDiv<uint64_t>(t.hiddenDim, 256U);
        const uint64_t gmm2_tiles = CeilDiv<uint64_t>(t.h, 256U);
        t.mGroupsPerWave = MegaMoeImpl::WaveGroups(p.aic, gmm1_tiles, gmm2_tiles, 4U);
    }
    // Routed MTE Combine returns the slot after all send lanes finish the Wave.
    if (is_a8w4 && shared_experts == 0 &&
        MegaMoeImpl::WaveScratchRows(t.maxOutputSize, t.mGroupsPerWave * MegaMoeImpl::L1_TILE_M_256) <
            t.maxOutputSize) {
        t.scratchWaveSlots = MegaMoeImpl::kWaveScratchSlots;
    }
    t.isPerExpertWeightTensor = false;
    t.dispatchBufferConfig = DispatchConfig(t, p.ub);
    SetSendMaskConfigs(t, p.ub);
    SetUnpermuteConfigs(t, p.ub);
    return t;
}

}  // namespace

uint32_t InferenceMaxRecvTokens(uint32_t max_tokens, uint32_t world, uint32_t topk, uint32_t local_experts)
{
    // Each source token can select each local expert at most once.
    const uint64_t rows = uint64_t(max_tokens) * world * std::min(topk, local_experts);
    TORCH_CHECK(rows <= UINT32_MAX, "Inference receive capacity exceeds 32-bit tiling");
    return static_cast<uint32_t>(rows);
}

void FillPeerAddresses(uint64_t* addresses, void* window, int64_t world)
{
    for (int64_t peer = 0; peer < world; ++peer) {
        void* address = aclshmem_ptr(window, static_cast<int32_t>(peer));
        TORCH_CHECK(address != nullptr, "MegaMoE cannot resolve SHMEM peer ", peer);
        addresses[peer] = reinterpret_cast<uint64_t>(address);
    }
}

Plan::Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world,
           int64_t max_tokens, bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts,
           uint64_t symmetric_address, int64_t symmetric_bytes)
    : ffts_(ffts)
{
    CheckTensor(example, 2, "plan input");
    TORCH_CHECK(example.size(0) > 0 && example.size(0) <= max_tokens && max_tokens <= 65536,
                "Plan BS must be in [1, max_tokens], max_tokens <= 65536");
    TORCH_CHECK(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                "Plan intermediate_hidden must be divisible by 128, in [256,4096]");
    TORCH_CHECK(shared_experts >= 0 && shared_experts <= 8 && (is_a8w4 || shared_experts == 0),
                "Plan shared experts require W4A8, with count in [0,8]");
    TORCH_CHECK(ranks_per_node == 0 || (ranks_per_node > 0 && ranks_per_node <= world && world % ranks_per_node == 0),
                "Plan ranks_per_node must divide EP");
    const auto required = buffer_size(world, experts, max_tokens, topk, example.size(1));
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= required, "MegaMoE symmetric buffer needs ", required,
                " bytes, got ", symmetric_bytes);
    platform_ = GetPlatformInfo();
    tiling_ = MakeTiling(example.size(0), example.size(1), intermediate * 2, topk, experts, world, max_tokens, is_a8w4,
                         shared_experts, platform_);
    tiling_.dispatchRanksPerNode = static_cast<uint32_t>(ranks_per_node);
    static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
    const MegaMoeImpl::WorkspaceInfo info(nullptr, &tiling_);
    const int64_t bytes =
        info.workspaceSize + static_cast<int64_t>(platform_.system_workspace) + kReservedWorkspaceBytes;
    TORCH_CHECK(bytes > info.workspaceSize, "MegaMoE workspace size overflow");
    workspace_ = torch::empty({bytes}, example.options().dtype(torch::kUInt8));
    context_ = torch::empty({static_cast<int64_t>(sizeof(Mc2Aclnn::Mc2MoeContext))}, workspace_.options());

    // peer 地址和 context 只在准备阶段解析，不在每次 forward 中重新分配、同步拷贝。
    Mc2Aclnn::Mc2MoeContext context{};
    const int rank = aclshmem_my_pe();
    TORCH_CHECK(rank >= 0 && rank < world && world <= Mc2Aclnn::HCCL_MAX_RANK_SIZE, "Invalid SHMEM rank/world: ", rank,
                "/", world);
    context.epRankId = static_cast<uint32_t>(rank);
    context.rankSizePerServer = static_cast<uint32_t>(world);
    FillPeerAddresses(context.epHcclBuffer_, reinterpret_cast<void*>(symmetric_address), world);
    ASCEND_DEEPEP_CHECK_ACL(
        aclrtMemcpy(context_.data_ptr(), sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE));
}

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden)
{
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
