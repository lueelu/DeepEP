// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#include "urma_forward.hpp"
#include "kernels/ubx/urma/compute_contract.hpp"
#include "kernels/ubx/urma/device_route_plan.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <acl/acl.h>
#include "check.hpp"
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

namespace deepep::megamoe {
namespace {
using MegaMoeUrma::Config;
constexpr uint64_t kAlignment = 512;
uint64_t Align(uint64_t value, uint64_t alignment = kAlignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

Config Layout(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows, int64_t cv_rows)
{
    TORCH_CHECK((world == 4 || world == 8 || world == 16) && max_tokens > 0 &&
                    max_tokens <= MegaMoeUrma::kMaxTokensPerRank && topk >= 1 && topk <= 8 && hidden >= 256 &&
                    hidden <= 8192 && hidden % 256 == 0,
                "MegaMoE URMA v1 requires EP4/EP8/EP16, H256..8192 aligned256, K1..8 and max_tokens in [1,16384]");
    TORCH_CHECK(cv_rows >= 128 && cv_rows <= 4096 && cv_rows % 128 == 0,
                "URMA CV rows must be a multiple of 128 in [128,4096]");
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

void CheckTensor(const torch::Tensor& tensor, int64_t dims, const char* name, const c10::Device& device)
{
    TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.is_contiguous() && tensor.dim() == dims, name,
                " must be a contiguous ", dims, "D tensor on ", device);
}

uint64_t Address(const torch::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}

}  // namespace

int64_t urma_buffer_size(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                         int64_t cv_rows)
{
    return static_cast<int64_t>(Layout(world, max_tokens, topk, hidden, max_recv_rows, cv_rows).symmetricBytes);
}

UrmaPlan::UrmaPlan(torch::Tensor example, int64_t world, int64_t intermediate, int64_t topk, int64_t cv_rows,
                   int64_t max_recv_rows, int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address,
                   int64_t symmetric_bytes, int64_t gmm1_ahead_mgroups, int64_t shared_experts,
                   int64_t shared_intermediate, int64_t num_experts)
{
    TORCH_CHECK(example.device().type() == c10::DeviceType::PrivateUse1 && example.dim() == 2 &&
                    example.is_contiguous() && example.size(0) > 0 && example.size(0) <= max_tokens,
                "URMA example must be contiguous NPU [BS,H], with 0<BS<=max_tokens");
    TORCH_CHECK(gmm1_ahead_mgroups >= 0 && gmm1_ahead_mgroups <= 65536,
                "gmm1_ahead_mgroups must be in [0,65536]; 0 retains the original expert schedule");
    max_tokens_ = max_tokens;
    TORCH_CHECK(intermediate >= 256 && intermediate <= 16384 && intermediate % 256 == 0 && rank >= 0 && rank < world,
                "URMA requires I256..16384 aligned256 and rank in [0,EP)");
    args_.config = Layout(world, max_tokens, topk, example.size(1), max_recv_rows, cv_rows);
    auto& config = args_.config;
    TORCH_CHECK(shared_experts == 0 || shared_experts == 1, "Shared experts must be 0 or 1");
    TORCH_CHECK(shared_experts == 0 ||
                    (shared_intermediate >= 256 && shared_intermediate <= 16384 && shared_intermediate % 256 == 0),
                "Shared I must be aligned256 in [256,16384]");
    args_.sharedExperts = shared_experts;
    args_.sharedIntermediate = shared_experts ? shared_intermediate : 0;
    config.bs = example.size(0);
    config.intermediate = intermediate;
    TORCH_CHECK(num_experts >= world && num_experts <= 128 * world && num_experts % world == 0 && topk <= num_experts,
                "URMA experts must divide EP, with 1<=E/EP<=128 and TopK<=E");
    config.localExperts = num_experts / config.world;
    config.rank = rank;
    config.gmm1AheadMGroups = static_cast<uint32_t>(gmm1_ahead_mgroups);
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= static_cast<int64_t>(config.symmetricBytes),
                "URMA symmetric workspace is too small: required ", config.symmetricBytes);
    args_.symmetric = symmetric_address;
    args_.ffts = ffts;

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "Cannot query Ascend platform");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic >= MegaMoeUrma::kAic && aiv >= 2U * MegaMoeUrma::kAic,
                "URMA v1 needs at least 28 AIC / 56 AIV; detected AIC=", aic, ", AIV=", aiv, ", UB=", ub,
                " bytes. It always launches 28 AIC blocks");
    TORCH_CHECK(ub >= MegaMoeUrma::kRequiredUbBytes, "URMA v1 requires ", MegaMoeUrma::kRequiredUbBytes,
                " bytes of UB for the W4 prologue and 128-row SwiGLU slices; detected UB=", ub, " bytes, AIC=", aic,
                ", AIV=", aiv);
    // Launch 28 AIC / 56 AIV on every supported EP size. AIV1 splits into
    // world-1 communication and 28-(world-1) activation/scale workers.
    constexpr uint32_t launch_aic = MegaMoeUrma::kAic;
    constexpr uint32_t launch_aiv = 2U * launch_aic;
    platform_ = {launch_aic, launch_aiv, static_cast<uint32_t>(ub),
                 platform->CalcTschBlockDim(launch_aiv, launch_aic, launch_aiv), platform->GetLibApiWorkSpaceSize()};

    TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == world, "URMA plan does not match its SHMEM runtime");
    for (int peer = 0; peer < world; ++peer) {
        TORCH_CHECK(aclshmem_ptr(reinterpret_cast<void*>(symmetric_address), peer) != nullptr,
                    "URMA completion signals require an MTE-accessible symmetric window for every peer; rank=", rank,
                    " cannot map peer=", peer,
                    ". Multi-host testing requires one mutually accessible UB memory domain.");
    }

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
    system_workspace_ = allocate(platform_.system_workspace + (2ULL << 20));

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

std::vector<torch::Tensor> UrmaPlan::routing_tensors(int64_t bs) const
{
    TORCH_CHECK(bs > 0 && bs <= max_tokens_, "BS outside allocated capacity");
    const int64_t n = bs * args_.config.topk;
    return {torch::Tensor(), inverse_.narrow(0, 0, n).reshape({bs, args_.config.topk}), route_counts_, route_offsets_};
}

std::vector<int64_t> UrmaPlan::route_stats() const
{
    TORCH_CHECK(last_epoch_ > 0, "Route statistics require a completed forward");
    ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeDevice());
    Config c{};
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(&c, sizeof(c), route_config_.data_ptr(), sizeof(c), ACL_MEMCPY_DEVICE_TO_HOST));
    return {int64_t(c.receiveRows), c.fragmentCount, -1, -1};
}

torch::Tensor UrmaPlan::forward(torch::Tensor x, torch::Tensor x_scales, torch::Tensor topk_idx,
                                torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                torch::Tensor weight_scales1, torch::Tensor weight_scales2,
                                c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> out, int64_t epoch,
                                c10::optional<torch::Tensor> trace, bool coarse_trace,
                                c10::optional<torch::Tensor> shared_w1, c10::optional<torch::Tensor> shared_w2,
                                c10::optional<torch::Tensor> shared_s1, c10::optional<torch::Tensor> shared_s2)
{
    TORCH_CHECK(!coarse_trace || (trace && !timer), "coarse trace requires trace storage and no aggregate timer");
    const auto& c = args_.config;
    const auto device = expanded_x_.device();
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= max_tokens_, "BS outside allocated capacity");
    args_.config.bs = x.size(0);
    CheckTensor(x, 2, "x", device);
    CheckTensor(x_scales, 3, "x_scales", device);
    CheckTensor(topk_idx, 2, "topk_idx", device);
    CheckTensor(topk_weights, 2, "topk_weights", device);
    CheckTensor(weight1, 3, "weight1", device);
    CheckTensor(weight2, 3, "weight2", device);
    CheckTensor(weight_scales1, 4, "weight_scales1", device);
    CheckTensor(weight_scales2, 4, "weight_scales2", device);
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Float8_e4m3fn && x.sizes() == torch::IntArrayRef({c.bs, c.hidden}),
                "x must be prepared FP8 E4M3 [BS,H]");
    TORCH_CHECK(x_scales.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    x_scales.sizes() == torch::IntArrayRef({c.bs, c.hidden / 64, 2}),
                "x_scales must be FP8 E8M0 [BS,H/64,2]");
    TORCH_CHECK((topk_idx.scalar_type() == torch::kInt32 || topk_idx.scalar_type() == torch::kInt64) &&
                    topk_idx.sizes() == torch::IntArrayRef({c.bs, c.topk}),
                "topk_idx must be int32/int64 [BS,K]");
    TORCH_CHECK(
        topk_weights.scalar_type() == torch::kFloat32 && topk_weights.sizes() == torch::IntArrayRef({c.bs, c.topk}),
        "weights must be FP32 [BS,K]");
    TORCH_CHECK(weight1.scalar_type() == torch::kUInt8 && weight2.scalar_type() == torch::kUInt8 &&
                    weight1.sizes() == torch::IntArrayRef({c.localExperts, c.intermediate * 2, c.hidden / 2}) &&
                    weight2.sizes() == torch::IntArrayRef({c.localExperts, c.hidden, c.intermediate / 2}),
                "URMA v1 requires main MegaMoE packed FP4 NZ weight layouts");
    TORCH_CHECK(
        weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
            weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
            weight_scales1.sizes() == torch::IntArrayRef({c.localExperts, c.intermediate * 2, c.hidden / 64, 2}) &&
            weight_scales2.sizes() == torch::IntArrayRef({c.localExperts, c.hidden, c.intermediate / 64, 2}),
        "Invalid FP4 weight scales");
    TORCH_CHECK(bool(shared_w1) == bool(args_.sharedExperts) && bool(shared_w2) == bool(args_.sharedExperts) &&
                    bool(shared_s1) == bool(args_.sharedExperts) && bool(shared_s2) == bool(args_.sharedExperts),
                "Shared weight pairs must match S0/S1");
    if (args_.sharedExperts) {
        const int64_t si = args_.sharedIntermediate;
        CheckTensor(*shared_w1, 3, "shared_w1", device);
        CheckTensor(*shared_w2, 3, "shared_w2", device);
        CheckTensor(*shared_s1, 4, "shared_s1", device);
        CheckTensor(*shared_s2, 4, "shared_s2", device);
        TORCH_CHECK(shared_w1->scalar_type() == c10::ScalarType::Float8_e4m3fn &&
                        shared_w2->scalar_type() == c10::ScalarType::Float8_e4m3fn &&
                        shared_w1->sizes() == torch::IntArrayRef({1, 2 * si, c.hidden}) &&
                        shared_w2->sizes() == torch::IntArrayRef({1, c.hidden, si}),
                    "Shared weights must be FP8 E4M3 ND [1,N,K]");
        TORCH_CHECK(shared_s1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_s2->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_s1->sizes() == torch::IntArrayRef({1, 2 * si, c.hidden / 64, 2}) &&
                        shared_s2->sizes() == torch::IntArrayRef({1, c.hidden, si / 64, 2}),
                    "Invalid shared scales");
        args_.sharedWeight1 = Address(*shared_w1);
        args_.sharedWeight2 = Address(*shared_w2);
        args_.sharedScale1 = Address(*shared_s1);
        args_.sharedScale2 = Address(*shared_s2);
    }
    TORCH_CHECK(epoch > last_epoch_ && epoch < (1LL << 60), "URMA epoch must increase across calls");
    if (timer) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON for URMA timers");
#endif
        CheckTensor(*timer, 1, "timer", device);
        TORCH_CHECK(timer->scalar_type() == torch::kInt64 && timer->numel() == MegaMoeUrma::kTimerElements,
                    "Invalid URMA timer allocation");
    }
    uint64_t trace_capacity = 0;
    if (trace) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON for URMA traces");
#endif
        CheckTensor(*trace, 1, "trace", device);
        TORCH_CHECK(Address(*trace) % 128U == 0,
                    "trace storage must be 128-byte aligned; use UrmaBuffer.allocate_trace()");
        static_assert(MegaMoeUrma::kTraceHeaderElements == MegaMoeUrma::kTraceRecordElements);
        constexpr int64_t trace_row_elements =
            static_cast<int64_t>(MegaMoeUrma::kTimerCores) * MegaMoeUrma::kTraceRecordElements;
        TORCH_CHECK(trace->scalar_type() == torch::kInt64 && trace->numel() % trace_row_elements == 0 &&
                        trace->numel() >= 2 * trace_row_elements &&
                        trace->numel() <= (int64_t{MegaMoeUrma::kTraceMaxCapacity} + 1) * trace_row_elements,
                    "trace must contain 84 * (16 + capacity * 16) int64 elements; capacity must be in [1,",
                    MegaMoeUrma::kTraceMaxCapacity, "]");
        trace_capacity = static_cast<uint64_t>(trace->numel() / trace_row_elements - 1);
    }
    auto output = out ? *out : torch::empty({c.bs, c.hidden}, x.options().dtype(torch::kBFloat16));
    CheckTensor(output, 2, "out", device);
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && output.sizes() == x.sizes(), "out must be BF16 [BS,H]");
    auto stream = c10_npu::getCurrentNPUStream();
    if (args_.sharedExperts)
        for (auto* tensor : {&shared_done1_, &shared_done_act_})
            ASCEND_DEEPEP_CHECK_ACL(
                aclrtMemsetAsync(tensor->data_ptr(), tensor->numel(), 0, tensor->numel(), stream.stream()));
    // The previous invocation has joined its reducers and all final QP quiets.
    // New remote writes cannot arrive before PublishStart in the kernel below.
    // Reset per-CV/finished slots so SO add(epoch) remains correct when a CV
    // was absent in older routes or callers skipped epoch values. Start slots
    // themselves stay monotonic and are never cleared here.
    const uint64_t cv_flag_bytes = static_cast<uint64_t>(c.world) * c.maxPeerChunks * MegaMoeUrma::kFlagBytes;
    for (uint64_t offset : {c.dispatchReadyOffset, c.combineReadyOffset}) {
        ASCEND_DEEPEP_CHECK_ACL(aclrtMemsetAsync(reinterpret_cast<void*>(args_.symmetric + offset), cv_flag_bytes, 0,
                                                 cv_flag_bytes, stream.stream()));
    }
    const uint64_t finished_bytes = static_cast<uint64_t>(c.world) * MegaMoeUrma::kFlagBytes;
    ASCEND_DEEPEP_CHECK_ACL(
        aclrtMemsetAsync(reinterpret_cast<void*>(args_.symmetric + c.barrierOffset + finished_bytes), finished_bytes, 0,
                         finished_bytes, stream.stream()));
    for (auto* tensor :
         {&input_scale_done_, &activation_done_, &gmm2_done_, &gmm_issued_, &local_flags_, &pack_ready_}) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(tensor->data_ptr(), tensor->numel(), 0, tensor->numel(), stream.stream()));
    }
    if (timer) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(timer->data_ptr(), timer->nbytes(), 0, timer->nbytes(), stream.stream()));
    }
    if (trace) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(trace->data_ptr(), trace->nbytes(), 0, trace->nbytes(), stream.stream()));
    }
    args_.config.epoch = epoch;
    args_.x = Address(x);
    args_.xScale = Address(x_scales);
    args_.topkIds = Address(topk_idx);
    args_.topkIdBytes = topk_idx.element_size();
    args_.weights = Address(topk_weights);
    args_.weight1 = Address(weight1);
    args_.weight2 = Address(weight2);
    args_.weightScale1 = Address(weight_scales1);
    args_.weightScale2 = Address(weight_scales2);
    expert_counts_ = torch::empty({c.localExperts}, x.options().dtype(torch::kInt32));
    args_.expertCountsOutput = Address(expert_counts_);
    args_.output = Address(output);
    args_.timer = timer ? Address(*timer) : 0;
    args_.trace = trace ? Address(*trace) : 0;
    args_.traceCapacity = trace_capacity;
    args_.coarseTrace = coarse_trace;
    mega_moe_urma_kernel_do(platform_.block_dim, platform_.ub, stream.stream(), &args_);
    last_epoch_ = epoch;
    for (const auto* tensor : {&x,
                               &x_scales,
                               &topk_idx,
                               &topk_weights,
                               &weight1,
                               &weight2,
                               &weight_scales1,
                               &weight_scales2,
                               &output,
                               &expert_counts_,
                               &expanded_x_,
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
                               &system_workspace_}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (args_.sharedExperts) {
        for (const auto* t : {&shared_gmm1_, &shared_activation_, &shared_scale_, &shared_output_,
                              &shared_routed_output_, &shared_done1_, &shared_done_act_})
            c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(), stream);
        for (const auto* t : {&*shared_w1, &*shared_w2, &*shared_s1, &*shared_s2})
            c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(), stream);
    }
    if (timer) c10_npu::NPUCachingAllocator::recordStream(timer->storage().data_ptr(), stream);
    if (trace) c10_npu::NPUCachingAllocator::recordStream(trace->storage().data_ptr(), stream);
    return output;
}

std::map<std::string, torch::Tensor> UrmaPlan::debug_snapshot() const
{
    TORCH_CHECK(last_epoch_ > 0, "URMA debug snapshot requires a completed forward");
    int32_t device = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&device));
    TORCH_CHECK(device == expanded_x_.device().index(), "URMA debug snapshot requires the owning device");
    ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeDevice());
    std::map<std::string, torch::Tensor> result;
    auto copy = [&](const char* name, uint64_t address, uint64_t bytes) {
        auto cpu = torch::empty({static_cast<int64_t>(bytes)},
                                torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
        if (bytes != 0) {
            ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(cpu.data_ptr(), bytes, reinterpret_cast<const void*>(address), bytes,
                                                ACL_MEMCPY_DEVICE_TO_HOST));
        }
        result.emplace(name, std::move(cpu));
    };
    Config c{};
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(&c, sizeof(c), route_config_.data_ptr(), sizeof(c), ACL_MEMCPY_DEVICE_TO_HOST));
    const uint64_t expanded = static_cast<uint64_t>(c.bs) * c.topk;
    copy("expanded_x", Address(expanded_x_), expanded * c.hidden);
    copy("expanded_scale", Address(expanded_scale_), expanded * (c.hidden / 32));
    copy("received_x", args_.symmetric + c.receiveXOffset, c.world * c.sourceRowCapacity * c.hidden);
    copy("received_scale", args_.symmetric + c.receiveScaleOffset, c.world * c.sourceRowCapacity * (c.hidden / 32));
    copy("gathered_input_scale", Address(gathered_input_scale_), c.receiveRows * (c.hidden / 32));
    // Routed GMM1 is UB-only; no uninitialized GM shadow is exported.
    copy("activation", Address(activation_), c.receiveRows * c.intermediate);
    copy("activation_scale", Address(activation_scale_), c.receiveRows * (c.intermediate / 32));
    // No routed GMM2 scratch: output is consumed in paired AIV1 UB.
    copy("gmm2_remote", Address(gmm2_), c.world * c.sourceRowCapacity * c.hidden * 2);
    copy("returned", args_.symmetric + c.returnYOffset, expanded * c.hidden * 2);
    copy("route_info", args_.routeInfo, sizeof(MegaMoeUrma::DeviceRouteInfo));
    copy("compute_tasks", args_.computeTasks, c.computeTaskCount * sizeof(MegaMoeUrma::ComputeBatchTask));
    copy("reduce_index", args_.reduceIndex, expanded * sizeof(int64_t));
    auto compact_order = torch::empty({int64_t(expanded * sizeof(int64_t))},
                                      torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    const auto* inverse = reinterpret_cast<const int64_t*>(result.at("reduce_index").data_ptr());
    auto* order = reinterpret_cast<int64_t*>(compact_order.data_ptr());
    for (uint64_t flat = 0; flat < expanded; ++flat) {
        TORCH_CHECK(inverse[flat] >= 0 && uint64_t(inverse[flat]) < expanded,
                    "Invalid inverse in post-run route snapshot");
        order[inverse[flat]] = static_cast<int64_t>(flat);
    }
    result.emplace("order", compact_order);
    // GMM1 tile readiness is now the paired hardware FULL event.
    copy("gmm2_done", args_.gmm2Done, c.fragmentCount * MegaMoeUrma::kFlagBytes);
    copy("input_scale_done", args_.inputScaleDone, c.fragmentCount * MegaMoeUrma::kFlagBytes);
    copy("activation_done", args_.activationDone, c.fragmentCount * MegaMoeUrma::kFlagBytes);
    copy("local_flags", args_.localFlags, local_flags_.nbytes());
    if (args_.sharedExperts) {
        copy("shared_gmm1", args_.sharedGmm1, shared_gmm1_.nbytes());
        copy("shared_activation", args_.sharedActivation, shared_activation_.nbytes());
        copy("shared_output", args_.sharedOutput, shared_output_.nbytes());
        copy("shared_routed_fp32", args_.sharedRoutedOutput, shared_routed_output_.nbytes());
    }
    return result;
}

int64_t UrmaPlan::workspace_bytes() const
{
    int64_t bytes = 0;
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
                               &system_workspace_})
        if (tensor->defined()) bytes += tensor->nbytes();
    for (const auto* t : {&shared_gmm1_, &shared_activation_, &shared_scale_, &shared_output_, &shared_routed_output_,
                          &shared_done1_, &shared_done_act_})
        if (t->defined()) bytes += t->nbytes();
    return bytes;
}
}  // namespace deepep::megamoe
