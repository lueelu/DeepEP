#include "ops/dispatch.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <torch_npu/csrc/aten/common/from_blob.h>

#include "common/types.h"
#include "common/platform.h"
#include "common/udma_schedule.h"
#include "ops/launcher.h"
#include "ops/stream_allocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "runtime/shmem_instance.h"

namespace {

constexpr int64_t kDispatchLayoutUbBytes = ascend_deepep::platform::kKernelUbBytes;
constexpr int64_t kDispatchLayoutReduceBufferCount = 2;
constexpr int64_t kDispatchLayoutMaxReduceCols =
    kDispatchLayoutUbBytes / kDispatchLayoutReduceBufferCount / sizeof(int32_t);
constexpr int64_t kDispatchLayoutMaxRanks = 1024;
constexpr int64_t kDispatchLayoutAlignElems = 8;
constexpr int64_t kDeepEpLayoutExpertsPerWorker = 4;
constexpr int64_t kDeepEpLayoutRanksPerWorker = 8;
constexpr int64_t kMaxDispatchAiv = ascend_deepep::platform::kMaxNumAiv;
constexpr int64_t kDispatchWorkspaceAlignment = 32;
constexpr int64_t kNotifyWorkspaceAlignment = 512;
// Keep both Notify outputs aligned while copying their shared storage once.
constexpr int64_t kNotifyExpertCountsOffset = kDispatchWorkspaceAlignment / sizeof(int32_t);
constexpr int64_t kDispatchStageBytes = ascend_deepep::platform::kDispatchStageBytes;
constexpr int64_t kMaxDispatchTopk = 128;
constexpr int64_t kDispatchUdmaCompletionSignalBytes = ascend_deepep::udma::kSignalBytes;
constexpr auto kTopkIdxScalarType = c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value;

struct DispatchWorkspaceLayout {
    int64_t recv_channel_prefix_offset;
    int64_t x_buffer_offset;
    int64_t src_idx_buffer_offset;
    int64_t topk_idx_buffer_offset;
    int64_t topk_weights_buffer_offset;
    int64_t x_scales_buffer_offset;
    int64_t x_scales_input_buffer_offset;
    int64_t udma_completion_signal_offset;
    int64_t total_bytes;
    int64_t end_offset;
};

int64_t AlignUpInt64(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

int64_t CheckedMulInt64(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " dimensions must be nonnegative");
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs, name, " size overflows int64");
    return lhs * rhs;
}

int64_t CheckedAddInt64(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " values must be nonnegative");
    TORCH_CHECK(lhs <= std::numeric_limits<int64_t>::max() - rhs, name, " size overflows int64");
    return lhs + rhs;
}

int64_t AppendWorkspaceRegion(int64_t cursor, int64_t bytes, const char* name)
{
    TORCH_CHECK(bytes >= 0, name, " size must be nonnegative");
    TORCH_CHECK(cursor <= std::numeric_limits<int64_t>::max() - bytes, name, " workspace size overflows int64");
    return AlignUpInt64(cursor + bytes, kDispatchWorkspaceAlignment);
}

torch::Tensor MakeShmemTensor(void* symmetric_buffer, int64_t byte_offset, std::vector<int64_t> sizes,
                              at::ScalarType dtype)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(byte_offset >= 0, "SHMEM tensor byte offset must be nonnegative");
    auto* address = static_cast<uint8_t*>(symmetric_buffer) + byte_offset;
    // BufferRuntime owns the symmetric allocation; returned tensors are
    // non-owning views and must not free interior SHMEM addresses.
    return at_npu::native::from_blob(address, sizes, [](void*) {}, at::TensorOptions().dtype(dtype));
}

DispatchWorkspaceLayout MakeDispatchWorkspaceLayout(int64_t base_offset, int64_t world_size, int64_t num_channels,
                                                    int64_t recv_buffer_tokens, int64_t hidden_bytes, int64_t num_topk,
                                                    bool has_topk_idx, bool has_topk_weights, int64_t scale_row_bytes,
                                                    int64_t scale_input_buffer_bytes, bool use_udma)
{
    DispatchWorkspaceLayout layout{};
    int64_t cursor = base_offset;

    // Reserved for layout compatibility; Notify now writes the prefix
    // directly to owned GM. Keep payload offsets and size hints unchanged.
    layout.recv_channel_prefix_offset = cursor;
    const int64_t channel_prefix_row_bytes =
        CheckedMulInt64(num_channels, static_cast<int64_t>(sizeof(int32_t)), "recv_channel_prefix row");
    cursor = AppendWorkspaceRegion(cursor, CheckedMulInt64(world_size, channel_prefix_row_bytes, "recv_channel_prefix"),
                                   "recv_channel_prefix");

    layout.x_buffer_offset = cursor;
    cursor = AppendWorkspaceRegion(cursor, CheckedMulInt64(recv_buffer_tokens, hidden_bytes, "dispatch x buffer"),
                                   "dispatch x buffer");

    layout.src_idx_buffer_offset = cursor;
    // Source indices use the same compact receive slots as x. Global-memory
    // addresses may be unaligned; only the temporary UB slices are aligned.
    cursor = AppendWorkspaceRegion(
        cursor,
        CheckedMulInt64(recv_buffer_tokens, static_cast<int64_t>(sizeof(int32_t)), "dispatch source-index buffer"),
        "dispatch source-index buffer");

    layout.topk_idx_buffer_offset = cursor;
    const int64_t topk_idx_row_bytes =
        CheckedMulInt64(num_topk, static_cast<int64_t>(sizeof(ascend_deepep::topk_idx_t)), "dispatch top-k index row");
    if (has_topk_idx) {
        cursor = AppendWorkspaceRegion(
            cursor, CheckedMulInt64(recv_buffer_tokens, topk_idx_row_bytes, "dispatch top-k index buffer"),
            "dispatch top-k index buffer");
    }

    layout.topk_weights_buffer_offset = cursor;
    if (has_topk_weights) {
        const int64_t topk_weights_row_bytes =
            CheckedMulInt64(num_topk, static_cast<int64_t>(sizeof(float)), "dispatch top-k weight row");
        cursor = AppendWorkspaceRegion(
            cursor, CheckedMulInt64(recv_buffer_tokens, topk_weights_row_bytes, "dispatch top-k weight buffer"),
            "dispatch top-k weight buffer");
    }

    layout.x_scales_buffer_offset = cursor;
    if (scale_row_bytes > 0) {
        cursor =
            AppendWorkspaceRegion(cursor, CheckedMulInt64(recv_buffer_tokens, scale_row_bytes, "dispatch scale buffer"),
                                  "dispatch scale buffer");
    }

    layout.x_scales_input_buffer_offset = cursor;
    if (scale_input_buffer_bytes > 0) {
        cursor = AppendWorkspaceRegion(cursor, scale_input_buffer_bytes, "dispatch input scale staging buffer");
    }

    // UDMA markers live in the fixed header, outside notify and all payloads.
    // Their location must not change with shape, weights, or operator kind.
    layout.udma_completion_signal_offset = use_udma ? 0 : cursor;

    layout.total_bytes = cursor - base_offset;
    layout.end_offset = cursor;
    return layout;
}

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalNpuContiguous(const c10::optional<torch::Tensor>& tensor, const char* name)
{
    if (tensor.has_value()) {
        CheckNpuContiguous(*tensor, name);
    }
}

void CheckOptionalNpu(const c10::optional<torch::Tensor>& tensor, const char* name)
{
    if (tensor.has_value()) {
        TORCH_CHECK(tensor->device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    }
}

int64_t ResolveNumAiv(int64_t num_aiv)
{
    TORCH_CHECK(num_aiv >= 0, "num_aiv must be greater than or equal to zero");
    const int64_t resolved = num_aiv == 0 ? 32 : num_aiv;
    TORCH_CHECK(resolved % 2 == 0, "num_aiv must be even");
    TORCH_CHECK(resolved <= kMaxDispatchAiv, "num_aiv must not exceed ", kMaxDispatchAiv, ", got ", resolved);
    return resolved;
}

}  // namespace

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor> GetDispatchLayout(
    torch::Tensor topk_idx, int64_t num_experts, int rank, int world_size, int num_device_aiv, uint64_t ffts_addr,
    c10::optional<int64_t> allocation_stream_id)
{
    StreamAllocator allocator(allocation_stream_id);
    allocator.Record({topk_idx});
    CheckNpuContiguous(topk_idx, "topk_idx");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [num_tokens, num_topk]");
    TORCH_CHECK(topk_idx.scalar_type() == kTopkIdxScalarType, "topk_idx must have dtype deep_ep.topk_idx_t");
    TORCH_CHECK(num_experts > 0, "num_experts must be greater than zero");
    TORCH_CHECK(world_size > 0, "world_size must be greater than zero");
    TORCH_CHECK(num_experts % world_size == 0, "num_experts must be divisible by world_size");
    TORCH_CHECK(world_size <= kDispatchLayoutMaxRanks, "get_dispatch_layout currently supports at most ",
                kDispatchLayoutMaxRanks, " ranks, got ", world_size);

    const int64_t num_tokens = topk_idx.size(0);
    const int64_t num_topk = topk_idx.size(1);
    TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(),
                "num_tokens must fit int32 for dispatch layout tiling");
    TORCH_CHECK(num_topk <= std::numeric_limits<int32_t>::max(), "num_topk must fit int32 for dispatch layout tiling");
    TORCH_CHECK(num_experts <= std::numeric_limits<int32_t>::max(),
                "num_experts must fit int32 for dispatch layout tiling");
    TORCH_CHECK(world_size <= std::numeric_limits<uint32_t>::max(),
                "world_size must fit uint32 for dispatch layout tiling");

    TORCH_CHECK(num_device_aiv >= 2, "dispatch layout requires at least 2 device AIV cores");
    // Match DeepEP V1's layout launch scale, then cap it because this kernel
    // uses full-core SyncAll and cannot be scheduled in multiple waves.
    const int64_t layout_parallelism =
        (num_experts + kDeepEpLayoutExpertsPerWorker - 1) / kDeepEpLayoutExpertsPerWorker +
        (world_size + kDeepEpLayoutRanksPerWorker - 1) / kDeepEpLayoutRanksPerWorker;
    const uint32_t block_dim = static_cast<uint32_t>(std::min<int64_t>(
        std::min<int64_t>(num_device_aiv, kMaxDispatchAiv), std::max<int64_t>(2, layout_parallelism)));
    const int64_t rank_stride = AlignUpInt64(world_size, kDispatchLayoutAlignElems);
    const int64_t expert_stride = AlignUpInt64(num_experts, kDispatchLayoutAlignElems);
    const int64_t is_token_in_rank_ub_bytes = AlignUpInt64(world_size, 32);
    const int64_t count_ub_bytes = (rank_stride + expert_stride) * sizeof(int32_t) + is_token_in_rank_ub_bytes;
    const int64_t reduce_ub_bytes = kDispatchLayoutReduceBufferCount * kDispatchLayoutMaxReduceCols * sizeof(int32_t);
    TORCH_CHECK(std::max(count_ub_bytes, reduce_ub_bytes) <= kDispatchLayoutUbBytes,
                "dispatch layout UB workspace is too small: need ", std::max(count_ub_bytes, reduce_ub_bytes),
                " bytes, limit is ", kDispatchLayoutUbBytes, " bytes");

    auto int_options = topk_idx.options().dtype(torch::kInt32);
    auto bool_options = topk_idx.options().dtype(torch::kBool);

    auto num_tokens_per_rank_storage = allocator.Empty({rank_stride}, int_options);
    auto num_tokens_per_expert_storage = allocator.Empty({expert_stride}, int_options);
    auto num_tokens_per_rank = num_tokens_per_rank_storage.narrow(0, 0, world_size);
    auto num_tokens_per_expert = num_tokens_per_expert_storage.narrow(0, 0, num_experts);
    // Layout writes every route row, including all-invalid rows.
    auto is_token_in_rank = allocator.Empty({num_tokens, world_size}, bool_options);
    auto scratch_buffer =
        allocator.Empty({static_cast<int64_t>(block_dim) * (rank_stride + expert_stride)}, int_options);

    ascend_deepep::DispatchLayoutTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.num_experts = static_cast<uint32_t>(num_experts);
    tiling.block_dim = block_dim;
    tiling.rank_stride = static_cast<uint32_t>(rank_stride);
    tiling.expert_stride = static_cast<uint32_t>(expert_stride);

    LaunchDispatchLayoutKernel(topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, tiling,
                               scratch_buffer, ffts_addr);

    return {num_tokens_per_rank, c10::nullopt, num_tokens_per_expert, is_token_in_rank};
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
           std::vector<int64_t>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
V1Dispatch(torch::Tensor x, c10::optional<torch::Tensor> x_scales, c10::optional<torch::Tensor> topk_idx,
           c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> num_tokens_per_rank,
           torch::Tensor is_token_in_rank, c10::optional<torch::Tensor> num_tokens_per_expert,
           int64_t cached_num_recv_tokens, c10::optional<torch::Tensor> cached_rank_prefix_matrix,
           c10::optional<torch::Tensor> cached_channel_prefix_matrix, int64_t expert_alignment,
           int64_t num_worst_tokens, int64_t num_aiv, int64_t num_max_nvl_chunked_send_tokens,
           int64_t num_max_nvl_chunked_recv_tokens, bool zero_copy, int rank, int world_size,
           std::size_t symmetric_buffer_bytes, std::size_t& notify_workspace_bytes, bool symmetric_workspace_active,
           bool use_udma, ShmemInstance& shmem, void* symmetric_buffer, c10::optional<int64_t> allocation_stream_id,
           c10::optional<int64_t> num_max_tokens_per_rank, int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit,
           c10::optional<torch::Tensor> cached_dispatch_plan, std::vector<torch::Tensor>* public_handle,
           bool defer_public_expert_prefix, void* dispatch_output_control)
{
    StreamAllocator allocator(allocation_stream_id);
    TORCH_CHECK(!use_udma || dispatch_output_control != nullptr,
                "UDMA dispatch requires its Buffer-owned output address control");
    allocator.Record({x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
                      cached_rank_prefix_matrix, cached_channel_prefix_matrix, cached_dispatch_plan});
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(is_token_in_rank, "is_token_in_rank");
    CheckOptionalNpu(x_scales, "x_scales");
    CheckOptionalNpuContiguous(topk_idx, "topk_idx");
    CheckOptionalNpuContiguous(topk_weights, "topk_weights");
    CheckOptionalNpuContiguous(num_tokens_per_rank, "num_tokens_per_rank");
    CheckOptionalNpuContiguous(num_tokens_per_expert, "num_tokens_per_expert");
    CheckOptionalNpuContiguous(cached_rank_prefix_matrix, "cached_rank_prefix_matrix");
    CheckOptionalNpuContiguous(cached_channel_prefix_matrix, "cached_channel_prefix_matrix");
    CheckOptionalNpuContiguous(cached_dispatch_plan, "cached_dispatch_plan");

    TORCH_CHECK(world_size > 0, "world_size must be greater than zero");
    TORCH_CHECK(world_size <= kDispatchLayoutMaxRanks, "intranode dispatch currently supports at most ",
                kDispatchLayoutMaxRanks, " ranks, got ", world_size);
    TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size)");
    TORCH_CHECK(x.dim() == 2, "x must have shape [num_tokens, hidden]");
    TORCH_CHECK(is_token_in_rank.dim() == 2, "is_token_in_rank must have shape [num_tokens, num_ranks]");
    TORCH_CHECK(is_token_in_rank.scalar_type() == torch::kBool, "is_token_in_rank must be bool");
    TORCH_CHECK(is_token_in_rank.size(0) == x.size(0), "is_token_in_rank and x must have the same num_tokens");
    TORCH_CHECK(is_token_in_rank.size(1) == world_size, "is_token_in_rank second dim must equal world_size");
    TORCH_CHECK(expert_alignment > 0, "expert_alignment must be greater than zero");
    TORCH_CHECK(expert_alignment <= std::numeric_limits<int32_t>::max(), "expert_alignment must fit int32");
    TORCH_CHECK(num_worst_tokens >= 0, "num_worst_tokens must be greater than or equal to zero");
    TORCH_CHECK(num_worst_tokens <= std::numeric_limits<int32_t>::max(), "num_worst_tokens must fit int32");
    TORCH_CHECK(cached_num_recv_tokens >= 0, "cached_num_recv_tokens must be greater than or equal to zero");
    TORCH_CHECK(cached_num_recv_tokens <= std::numeric_limits<int32_t>::max(), "cached_num_recv_tokens must fit int32");
    TORCH_CHECK(num_max_nvl_chunked_send_tokens > 0, "num_max_nvl_chunked_send_tokens must be greater than zero");
    TORCH_CHECK(num_max_nvl_chunked_recv_tokens > 0, "num_max_nvl_chunked_recv_tokens must be greater than zero");
    TORCH_CHECK(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens,
                "num_max_nvl_chunked_send_tokens must be smaller than num_max_nvl_chunked_recv_tokens");

    const bool cached_mode = cached_rank_prefix_matrix.has_value() || cached_channel_prefix_matrix.has_value();
    TORCH_CHECK(cached_rank_prefix_matrix.has_value() == cached_channel_prefix_matrix.has_value(),
                "cached rank/channel prefix matrices must be passed together");
    if (!cached_mode) {
        TORCH_CHECK(num_tokens_per_rank.has_value(), "num_tokens_per_rank is required for non-cached dispatch");
        TORCH_CHECK(num_tokens_per_expert.has_value(), "num_tokens_per_expert is required for non-cached dispatch");
        TORCH_CHECK(num_tokens_per_rank->dim() == 1, "num_tokens_per_rank must be one-dimensional");
        TORCH_CHECK(num_tokens_per_rank->numel() == world_size, "num_tokens_per_rank length must equal world_size");
        TORCH_CHECK(num_tokens_per_rank->scalar_type() == torch::kInt32, "num_tokens_per_rank must be int32");
        TORCH_CHECK(num_tokens_per_expert->dim() == 1, "num_tokens_per_expert must be one-dimensional");
        TORCH_CHECK(num_tokens_per_expert->numel() % world_size == 0,
                    "num_tokens_per_expert length must be divisible by world_size");
        TORCH_CHECK(num_tokens_per_expert->scalar_type() == torch::kInt32, "num_tokens_per_expert must be int32");
    }
    if (topk_idx.has_value()) {
        TORCH_CHECK(topk_idx->dim() == 2, "topk_idx must have shape [num_tokens, num_topk]");
        TORCH_CHECK(topk_idx->size(0) == x.size(0), "topk_idx and x must have the same num_tokens");
        TORCH_CHECK(topk_idx->scalar_type() == kTopkIdxScalarType, "topk_idx must have dtype deep_ep.topk_idx_t");
    }
    if (topk_weights.has_value()) {
        TORCH_CHECK(topk_idx.has_value(), "topk_weights requires topk_idx");
        TORCH_CHECK(topk_weights->sizes() == topk_idx->sizes(), "topk_weights shape must match topk_idx");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
    }

    const int64_t num_tokens = x.size(0);
    const int64_t token_capacity = num_max_tokens_per_rank.value_or(num_tokens);
    TORCH_CHECK(token_capacity > 0 && num_tokens <= token_capacity,
                "dispatch requires a positive common capacity >= local token count");
    // NPU tensors/allocator already established the context. The non-initializing
    // inline wrapper references a private symbol in torch_npu 2.7.1.post10.
    const bool capturing = c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None;
    TORCH_CHECK(!capturing || cached_mode || num_worst_tokens > 0, "NPU graph dispatch requires num_worst_tokens > 0");
    const int64_t hidden = x.size(1);
    const int64_t num_topk = topk_idx.has_value() ? topk_idx->size(1) : 0;
    const int64_t x_element_bytes = static_cast<int64_t>(x.element_size());
    const int64_t hidden_bytes = CheckedMulInt64(hidden, x_element_bytes, "dispatch hidden row");
    TORCH_CHECK(hidden_bytes > 0, "dispatch hidden row must not be empty");
    TORCH_CHECK(hidden_bytes <= kDispatchStageBytes, "one dispatch token row requires ", hidden_bytes,
                " UB bytes, but the dispatch staging area provides ", kDispatchStageBytes, " bytes");
    TORCH_CHECK(num_topk <= kMaxDispatchTopk, "num_topk must not exceed ", kMaxDispatchTopk,
                " for the dispatch UB layout");

    int64_t num_scales = 0;
    int64_t scale_element_bytes = 0;
    int64_t scale_row_bytes = 0;
    int64_t scale_token_stride = 0;
    int64_t scale_hidden_stride = 0;
    bool needs_x_scales_pack = false;
    if (x_scales.has_value()) {
        TORCH_CHECK(x.element_size() == 1, "x must use a one-byte dtype when x_scales is provided");
        TORCH_CHECK(x_scales->scalar_type() == torch::kFloat32 || x_scales->scalar_type() == torch::kInt32,
                    "x_scales must be float32 or int32");
        TORCH_CHECK(x_scales->dim() == 2, "x_scales must have shape [num_tokens, scale_cols]");
        TORCH_CHECK(x_scales->size(0) == num_tokens, "x_scales and x must have the same num_tokens");
        num_scales = x_scales->size(1);
        scale_element_bytes = static_cast<int64_t>(x_scales->element_size());
        scale_token_stride = x_scales->stride(0);
        scale_hidden_stride = x_scales->stride(1);
        TORCH_CHECK(scale_token_stride >= 0 && scale_hidden_stride >= 0, "x_scales strides must be nonnegative");
        needs_x_scales_pack = num_scales > 0 && !x_scales->is_contiguous();
        scale_row_bytes = CheckedMulInt64(num_scales, scale_element_bytes, "dispatch scale row");
        TORCH_CHECK(scale_row_bytes <= kDispatchStageBytes, "one dispatch scale row requires ", scale_row_bytes,
                    " UB bytes, but the dispatch staging area provides ", kDispatchStageBytes, " bytes");
    }

    const int64_t resolved_num_aiv = ResolveNumAiv(num_aiv);
    const int64_t num_channels = resolved_num_aiv;
    if (shmem.DispatchPeerOrder() == ascend_deepep::udma::kGroupRoundRobinOrder)
        udma_peers_per_worker = shmem.PeersPerGroupPerWorker();
    if (use_udma) {
        TORCH_CHECK(shmem.SharedJetty(), "V1 UDMA requires a V1 shared-Jetty Buffer runtime");
        TORCH_CHECK(udma_peers_per_worker == 1 || udma_peers_per_worker == 2 || udma_peers_per_worker == 4 ||
                        udma_peers_per_worker == 8,
                    "udma_peers_per_worker must be 1, 2, 4 or 8");
        TORCH_CHECK(udma_wqes_per_visit == 4 || udma_wqes_per_visit == 8 || udma_wqes_per_visit == 16 ||
                        udma_wqes_per_visit == 32,
                    "udma_wqes_per_visit must be 4, 8, 16 or 32");
        const int64_t workers = shmem.TokenSchedule()
                                    ? ascend_deepep::udma::TokenWorkers(static_cast<uint32_t>(resolved_num_aiv))
                                    : shmem.WorkersPerPeer() * ascend_deepep::udma::Workers(
                                                                   world_size, udma_peers_per_worker,
                                                                   shmem.DispatchPeerOrder(), shmem.RanksPerGroup());
        TORCH_CHECK(workers <= resolved_num_aiv && workers * shmem.QpsPerWorker() <= ascend_deepep::udma::kQpCount,
                    "UDMA peer grouping exceeds the AIV or 32-QP pool budget");
    }
    const int64_t num_experts = num_tokens_per_expert.has_value() ? num_tokens_per_expert->numel() : 0;
    TORCH_CHECK(cached_mode || num_experts > 0, "num_tokens_per_expert must not be empty");
    TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(), "num_tokens must fit int32 dispatch tiling");
    TORCH_CHECK(hidden <= std::numeric_limits<int32_t>::max(), "hidden must fit int32 dispatch tiling");
    TORCH_CHECK(num_topk <= std::numeric_limits<int32_t>::max(), "num_topk must fit int32 dispatch tiling");
    TORCH_CHECK(num_scales <= std::numeric_limits<uint32_t>::max(), "num_scales must fit uint32 dispatch tiling");
    TORCH_CHECK(x_element_bytes <= std::numeric_limits<uint32_t>::max(),
                "x element size must fit uint32 dispatch tiling");
    TORCH_CHECK(scale_element_bytes <= std::numeric_limits<uint32_t>::max(),
                "scale element size must fit uint32 dispatch tiling");
    TORCH_CHECK(num_experts <= std::numeric_limits<int32_t>::max(), "num_experts must fit int32 dispatch tiling");
    TORCH_CHECK(token_capacity <= std::numeric_limits<int32_t>::max() / world_size,
                "token capacity * world_size must fit int32 receive counters");
    const int64_t num_local_experts = num_experts > 0 ? num_experts / world_size : 0;
    const int64_t rank_stride = AlignUpInt64(world_size, kDispatchLayoutAlignElems);
    const int64_t expert_stride = AlignUpInt64(num_experts, kDispatchLayoutAlignElems);
    const int64_t channel_stride = AlignUpInt64(num_channels, kDispatchLayoutAlignElems);
    const int64_t local_expert_stride = AlignUpInt64(num_local_experts, kDispatchLayoutAlignElems);
    const int64_t recv_buffer_tokens = CheckedMulInt64(token_capacity, world_size, "dispatch receive capacity");
    // Fixed-size output skips the host receive-count readback. A smaller
    // allocation can truncate payload/src_idx while the handle still contains
    // full routing counts, causing combine to read beyond those tensors.
    // Until a device overflow protocol is provided, require the proven bound
    // before issuing even the notify collective. Cached dispatch reuses an
    // already validated handle and its original exact or padded receive size.
    TORCH_CHECK(cached_mode || num_worst_tokens == 0 || num_worst_tokens == recv_buffer_tokens,
                "fixed dispatch output requires num_worst_tokens == world_size * "
                "num_max_tokens_per_rank (or local T when capacity is None); "
                "smaller capacities are unsafe without receive-count readback");
    const int64_t persistent_notify_elems = world_size * static_cast<int64_t>(world_size) + world_size * channel_stride;
    const int64_t normal_notify_elems =
        persistent_notify_elems + rank_stride + expert_stride + num_channels * rank_stride;
    const int64_t required_notify_bytes =
        (cached_mode ? persistent_notify_elems : normal_notify_elems) * static_cast<int64_t>(sizeof(int32_t));
    // Keep the fixed UDMA control header separate from notify and payload.
    const int64_t control_bytes =
        use_udma ? CheckedMulInt64(world_size, kDispatchUdmaCompletionSignalBytes, "UDMA control header") : 0;
    // Padding notify to 512 bytes also aligns the following payload base.
    const int64_t required_aligned_notify_bytes = AlignUpInt64(required_notify_bytes, kNotifyWorkspaceAlignment);
    TORCH_CHECK(notify_workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "Buffer notify workspace reservation exceeds int64");
    const int64_t previous_notify_workspace_bytes = static_cast<int64_t>(notify_workspace_bytes);
    const int64_t reserved_notify_workspace_bytes =
        std::max(previous_notify_workspace_bytes, required_aligned_notify_bytes);
    const DispatchWorkspaceLayout dispatch_workspace = MakeDispatchWorkspaceLayout(
        CheckedAddInt64(control_bytes, reserved_notify_workspace_bytes, "dispatch workspace base"), world_size,
        num_channels, recv_buffer_tokens, hidden_bytes, num_topk, topk_idx.has_value(), topk_weights.has_value(),
        scale_row_bytes,
        CheckedMulInt64(
            // Reserve from capacity, including ranks with T=0.
            token_capacity, scale_row_bytes, "dispatch input scale staging buffer"),
        use_udma);
    TORCH_CHECK(dispatch_workspace.recv_channel_prefix_offset % kNotifyWorkspaceAlignment == 0,
                "dispatch workspace must start at a 512-byte aligned offset");
    const int64_t required_workspace_bytes = dispatch_workspace.end_offset;
    TORCH_CHECK(required_workspace_bytes <= static_cast<int64_t>(symmetric_buffer_bytes), "dispatch requires ",
                required_workspace_bytes, " bytes of symmetric workspace, but Buffer provides ", symmetric_buffer_bytes,
                " bytes");
    notify_workspace_bytes = static_cast<std::size_t>(reserved_notify_workspace_bytes);

    auto int_options = is_token_in_rank.options().dtype(torch::kInt32);
    auto rank_prefix_matrix = cached_rank_prefix_matrix.has_value()
                                  ? *cached_rank_prefix_matrix
                                  : allocator.Empty({world_size, world_size}, int_options);
    auto channel_prefix_matrix = cached_channel_prefix_matrix.has_value()
                                     ? *cached_channel_prefix_matrix
                                     : allocator.Empty({world_size, num_channels}, int_options);
    TORCH_CHECK(rank_prefix_matrix.dim() == 2 && rank_prefix_matrix.size(0) == world_size &&
                    rank_prefix_matrix.size(1) == world_size,
                "rank_prefix_matrix must have shape [world_size, world_size]");
    TORCH_CHECK(channel_prefix_matrix.dim() == 2 && channel_prefix_matrix.size(0) == world_size &&
                    channel_prefix_matrix.size(1) == num_channels,
                "channel_prefix_matrix must have shape [world_size, num_aiv]");
    TORCH_CHECK(rank_prefix_matrix.scalar_type() == torch::kInt32, "rank_prefix_matrix must be int32");
    TORCH_CHECK(channel_prefix_matrix.scalar_type() == torch::kInt32, "channel_prefix_matrix must be int32");

    // Fixed-size handle metadata can be allocated before the receive count
    // is known. Normal Notify writes every element; cached dispatch reuses
    // the caller's existing handle and does not gather a new prefix matrix.
    auto recv_channel_prefix_matrix =
        cached_mode ? allocator.Empty({0}, int_options) : allocator.Empty({world_size, num_channels}, int_options);

    // Notify produces send_head and the private stable token lists for UDMA.
    // The low-level cached+topk API needs a temporary head; ordinary Python
    // cached calls have no top-k and reuse the handle-owned plan unchanged.
    auto send_head = cached_mode && !(use_udma && topk_idx.has_value())
                         ? allocator.Empty({0}, int_options)
                         : (use_udma ? allocator.Empty({num_tokens, world_size}, int_options)
                                     : allocator.Full({num_tokens, world_size}, -1, int_options));
    c10::optional<torch::Tensor> dispatch_plan = c10::nullopt;
    bool rebuild_cached_plan = false;
    TORCH_CHECK(!cached_dispatch_plan.has_value() || (use_udma && cached_mode),
                "dispatch plan is only valid for cached UDMA dispatch");
    if (use_udma) {
        const int64_t length = CheckedAddInt64(num_tokens, num_channels - 1, "plan channel length") / num_channels;
        const int64_t stride = AlignUpInt64(length, 8);
        const int64_t elements =
            CheckedMulInt64(CheckedMulInt64(world_size, num_channels, "plan segments"), stride, "plan elements");
        (void)CheckedMulInt64(elements, sizeof(int32_t), "plan bytes");
        if (cached_dispatch_plan.has_value()) {
            const auto& plan = *cached_dispatch_plan;
            TORCH_CHECK(plan.scalar_type() == torch::kInt32 && plan.dim() == 3 && plan.size(0) == world_size &&
                            plan.size(1) == num_channels && plan.size(2) == stride,
                        "cached dispatch plan must be int32 [world_size, num_aiv, aligned channel tokens]");
        }
        if (cached_dispatch_plan.has_value() && !topk_idx.has_value()) {
            dispatch_plan = cached_dispatch_plan;
        } else {
            dispatch_plan = allocator.Empty({world_size, num_channels, stride}, int_options);
            rebuild_cached_plan = cached_mode;
        }
    }

    // Only fresh notify produces counts. Cached calls use the handle counts.
    torch::Tensor notify_counts, notify_num_recv_tokens, notify_num_recv_tokens_per_expert;
    if (!cached_mode) {
        // Notify overwrites both outputs, including expert stride padding.
        // Views retain storage for V2 handles and share its stream record.
        // No consumer interprets the alignment gap.
        notify_counts = allocator.Empty({kNotifyExpertCountsOffset + local_expert_stride}, int_options);
        notify_num_recv_tokens = notify_counts.narrow(0, 0, 1);
        notify_num_recv_tokens_per_expert = notify_counts.narrow(0, kNotifyExpertCountsOffset, local_expert_stride);
    }
    if (public_handle != nullptr) {
        TORCH_CHECK(!cached_mode && topk_idx.has_value(), "public handle is fresh-dispatch only");
        TORCH_CHECK(
            CheckedMulInt64(world_size, token_capacity, "V2 global token index") <= std::numeric_limits<int32_t>::max(),
            "V2 global token index exceeds int32");
        TORCH_CHECK(
            CheckedMulInt64(world_size, num_topk, "V2 source route index") <= std::numeric_limits<int32_t>::max(),
            "V2 source route index exceeds int32");
        const int64_t max_expert_slots = CheckedAddInt64(
            CheckedMulInt64(recv_buffer_tokens, num_topk, "V2 expert slots"),
            CheckedMulInt64(num_local_experts, expert_alignment - 1, "V2 expert padding"), "V2 aligned expert slots");
        TORCH_CHECK(max_expert_slots <= std::numeric_limits<int32_t>::max(), "V2 expert prefix bound exceeds int32");
        // Exact metadata ownership follows the normal stream-aware allocator.
        *public_handle = {allocator.Empty({world_size}, int_options), allocator.Empty({num_local_experts}, int_options),
                          notify_num_recv_tokens_per_expert.narrow(0, 0, num_local_experts), torch::Tensor(),
                          allocator.Empty({num_tokens, num_topk}, int_options)};
    }
    ascend_deepep::DispatchTilingData tiling{};
    if (public_handle != nullptr) {
        tiling.v2_rank_prefix = reinterpret_cast<uint64_t>((*public_handle)[0].data_ptr());
        if (!defer_public_expert_prefix) {
            tiling.v2_expert_prefix = reinterpret_cast<uint64_t>((*public_handle)[1].data_ptr());
        }
        tiling.v2_dst_slots = reinterpret_cast<uint64_t>((*public_handle)[4].data_ptr());
    }
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.hidden = static_cast<uint32_t>(hidden);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.num_experts = static_cast<uint32_t>(num_experts);
    tiling.num_max_tokens_per_rank = static_cast<uint32_t>(recv_buffer_tokens);
    tiling.expert_alignment = static_cast<uint32_t>(expert_alignment);
    tiling.block_dim = static_cast<uint32_t>(resolved_num_aiv);
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.has_x_scales = x_scales.has_value() ? 1U : 0U;
    tiling.cached_mode = cached_mode ? 1U : 0U;
    tiling.num_channels = static_cast<uint32_t>(num_channels);
    tiling.udma_peer_order = shmem.DispatchPeerOrder();
    tiling.udma_ranks_per_group = shmem.RanksPerGroup();
    tiling.udma_workers_per_peer = shmem.WorkersPerPeer();
    tiling.udma_qps_per_worker = shmem.QpsPerWorker();
    tiling.udma_peers_per_worker = static_cast<uint32_t>(udma_peers_per_worker);
    tiling.udma_wqes_per_visit = static_cast<uint32_t>(udma_wqes_per_visit);
    tiling.udma_token_schedule = use_udma && shmem.TokenSchedule();
    tiling.rank_stride = static_cast<uint32_t>(rank_stride);
    tiling.expert_stride = static_cast<uint32_t>(expert_stride);
    tiling.channel_stride = static_cast<uint32_t>(channel_stride);
    tiling.num_local_experts = static_cast<uint32_t>(num_local_experts);
    tiling.x_element_bytes = static_cast<uint32_t>(x_element_bytes);
    tiling.scale_element_bytes = static_cast<uint32_t>(scale_element_bytes);
    tiling.num_scales = static_cast<uint32_t>(num_scales);
    tiling.needs_x_scales_pack = needs_x_scales_pack ? 1U : 0U;
    tiling.scale_token_stride = static_cast<uint64_t>(scale_token_stride);
    tiling.scale_hidden_stride = static_cast<uint64_t>(scale_hidden_stride);
    tiling.needs_notify_entry_barrier = (symmetric_workspace_active || capturing) ? 1U : 0U;
    tiling.notify_workspace_offset = static_cast<uint64_t>(control_bytes);
    tiling.recv_channel_prefix_offset = static_cast<uint64_t>(dispatch_workspace.recv_channel_prefix_offset);
    tiling.x_buffer_offset = static_cast<uint64_t>(dispatch_workspace.x_buffer_offset);
    tiling.src_idx_buffer_offset = static_cast<uint64_t>(dispatch_workspace.src_idx_buffer_offset);
    tiling.topk_idx_buffer_offset = static_cast<uint64_t>(dispatch_workspace.topk_idx_buffer_offset);
    tiling.topk_weights_buffer_offset = static_cast<uint64_t>(dispatch_workspace.topk_weights_buffer_offset);
    tiling.x_scales_buffer_offset = static_cast<uint64_t>(dispatch_workspace.x_scales_buffer_offset);
    tiling.x_scales_input_buffer_offset = static_cast<uint64_t>(dispatch_workspace.x_scales_input_buffer_offset);
    tiling.udma_completion_signal_offset = static_cast<uint64_t>(dispatch_workspace.udma_completion_signal_offset);
    tiling.udma_output_control = use_udma ? reinterpret_cast<uint64_t>(dispatch_output_control) : 0;
    tiling.dispatch_workspace_bytes = static_cast<uint64_t>(dispatch_workspace.total_bytes);

    if (cached_mode) {
        LaunchCachedNotifyDispatchKernel(tiling, shmem.FftsAddr());
        if (rebuild_cached_plan)
            LaunchCachedDispatchPlanKernel(is_token_in_rank, *dispatch_plan, send_head, tiling, shmem.FftsAddr());
    } else {
        LaunchNotifyDispatchKernel(*num_tokens_per_rank, *num_tokens_per_expert, is_token_in_rank, rank_prefix_matrix,
                                   channel_prefix_matrix, recv_channel_prefix_matrix, notify_num_recv_tokens,
                                   notify_num_recv_tokens_per_expert, dispatch_plan, send_head, tiling,
                                   symmetric_buffer, shmem.FftsAddr());
    }
    int64_t num_recv_tokens = 0;
    std::vector<int64_t> num_recv_tokens_per_expert_list;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
    } else if (num_worst_tokens > 0) {
        num_recv_tokens = num_worst_tokens;
    } else {
        // One blocking readback supplies the exact output size and expert counts.
        // Fixed-capacity and cached paths still skip this readback.
        auto recv_counts_cpu = notify_counts.cpu();
        const int32_t* recv_counts = recv_counts_cpu.data_ptr<int32_t>();
        num_recv_tokens = static_cast<int64_t>(recv_counts[0]);
        TORCH_CHECK(num_recv_tokens >= 0, "notify_dispatch returned a negative receive count");

        const int32_t* recv_expert_counts = recv_counts + kNotifyExpertCountsOffset;
        num_recv_tokens_per_expert_list.reserve(static_cast<std::size_t>(num_local_experts));
        // The notify kernel returns raw counts; the public list is aligned here.
        for (int64_t expert = 0; expert < num_local_experts; ++expert) {
            const int64_t raw_count = static_cast<int64_t>(recv_expert_counts[expert]);
            TORCH_CHECK(raw_count >= 0, "notify_dispatch returned a negative expert receive count");
            num_recv_tokens_per_expert_list.push_back(
                expert_alignment == 1 ? raw_count : AlignUpInt64(raw_count, expert_alignment));
        }
    }

    TORCH_CHECK(num_recv_tokens <= std::numeric_limits<uint32_t>::max(),
                "num_recv_tokens must fit uint32 dispatch tiling");
    TORCH_CHECK(num_recv_tokens <= recv_buffer_tokens, "dispatch output requires ", num_recv_tokens,
                " token slots, but the symmetric workspace reserves ", recv_buffer_tokens);
    tiling.num_output_tokens = static_cast<uint32_t>(num_recv_tokens);

    if (public_handle != nullptr) {
        (*public_handle)[3] = allocator.Empty({num_recv_tokens, num_topk + 2}, int_options);
        tiling.v2_src_metadata = reinterpret_cast<uint64_t>((*public_handle)[3].data_ptr());
    }
    auto recv_x = zero_copy ? MakeShmemTensor(symmetric_buffer, dispatch_workspace.x_buffer_offset,
                                              {num_recv_tokens, hidden}, x.scalar_type())
                            : allocator.Empty({num_recv_tokens, hidden}, x.options());
    c10::optional<torch::Tensor> recv_x_scales = c10::nullopt;
    if (x_scales.has_value()) {
        recv_x_scales = zero_copy ? MakeShmemTensor(symmetric_buffer, dispatch_workspace.x_scales_buffer_offset,
                                                    {num_recv_tokens, num_scales}, x_scales->scalar_type())
                                  : allocator.Empty({num_recv_tokens, num_scales}, x_scales->options());
    }

    c10::optional<torch::Tensor> recv_topk_idx = c10::nullopt;
    c10::optional<torch::Tensor> recv_topk_weights = c10::nullopt;
    if (topk_idx.has_value()) {
        recv_topk_idx = zero_copy ? MakeShmemTensor(symmetric_buffer, dispatch_workspace.topk_idx_buffer_offset,
                                                    {num_recv_tokens, num_topk}, topk_idx->scalar_type())
                                  : allocator.Empty({num_recv_tokens, num_topk}, topk_idx->options());
        if (topk_weights.has_value()) {
            recv_topk_weights = zero_copy
                                    ? MakeShmemTensor(symmetric_buffer, dispatch_workspace.topk_weights_buffer_offset,
                                                      {num_recv_tokens, num_topk}, topk_weights->scalar_type())
                                    : allocator.Empty({num_recv_tokens, num_topk}, topk_weights->options());
        }
    }

    // Dispatch writes every valid source index. Fixed-capacity tails are
    // unspecified; consumers select valid rows by rank/channel prefixes.
    auto recv_src_idx = allocator.Empty({cached_mode ? 0 : num_recv_tokens}, int_options);
    LaunchV1DispatchKernel(x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank,
                           num_tokens_per_expert, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
                           rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
                           send_head, dispatch_plan, tiling, symmetric_buffer, shmem.FftsAddr(), use_udma);
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            dispatch_plan};
}

}  // namespace ascend_deepep::ops
