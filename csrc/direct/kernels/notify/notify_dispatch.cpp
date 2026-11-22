#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "kernels/notify/notify_recv_prefix.h"
#include "kernels/notify/notify_dispatch_plan.h"
#include "kernels/notify/notify_fused_layout.h"
#include "kernels/common/v2_handle.h"
#include "common/launch.h"
#include "common/types.h"

using namespace AscendC;

namespace {

constexpr int32_t kNotifyInt32Bytes = 4;
constexpr int32_t kNotifyInt32AlignElems = 8;
constexpr int32_t kNotifyDispatchUbBytes = static_cast<int32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr int32_t kMaxVectorTileElems = 16 * 1024;
constexpr int32_t kRouteRowUbOffset = 2 * kMaxVectorTileElems * kNotifyInt32Bytes;
constexpr int32_t kRouteHalfUbOffset = kRouteRowUbOffset + kMaxVectorTileElems * sizeof(uint8_t);
constexpr int32_t kRouteHalfUbEnd = kRouteHalfUbOffset + kMaxVectorTileElems * sizeof(half);
static_assert(kRouteHalfUbEnd <= kNotifyDispatchUbBytes);

__aicore__ inline int32_t NotifyIntMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

// A source-major snapshot occupies one 64 KiB UB slot. Every row starts on
// a 32-byte boundary; zero means even one aligned column tile cannot fit.
__aicore__ inline uint32_t NotifyCountTileWidth(uint32_t world, uint64_t capacity)
{
    if (world == 0) return 0;
    const uint64_t elems = capacity < kMaxVectorTileElems ? capacity : kMaxVectorTileElems;
    return static_cast<uint32_t>(elems / world / kNotifyInt32AlignElems * kNotifyInt32AlignElems);
}

// Compact EP2/EP4 rows share a store granule. Assign the entire granule to
// one AIV, including for non-power-of-two worlds accepted by the host.
__aicore__ inline uint32_t NotifyCountRowsPerOwner(uint32_t world)
{
    uint32_t rows = 1;
    while ((uint64_t(rows) * world) % kNotifyInt32AlignElems != 0) rows *= 2;
    return rows;
}

__simt_vf__ __launch_bounds__(128) inline void PrefixNotifyCountsVf(
    const __ubuf__ int32_t* counts, __gm__ int32_t* prefixes, __gm__ int32_t* published,
    uint32_t world, uint32_t begin, uint32_t columns, uint32_t snapshot_stride)
{
    for (uint32_t column = threadIdx.x; column < columns; column += 128) {
        int32_t sum = 0;
        for (uint32_t source = 0; source < world; ++source) {
            sum += counts[source * snapshot_stride + column];
            const uint64_t offset = uint64_t(source) * world + begin + column;
            prefixes[offset] = sum;
            published[offset] = sum;
        }
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void ReduceNotifyExpertsVf(
    const __ubuf__ int32_t* counts, __gm__ int32_t* experts,
    uint32_t world, uint32_t begin, uint32_t columns, uint32_t snapshot_stride)
{
    for (uint32_t column = threadIdx.x; column < columns; column += 128) {
        int32_t sum = 0;
        for (uint32_t source = 0; source < world; ++source)
            sum += counts[source * snapshot_stride + column];
        experts[begin + column] = sum;
    }
    asc_threadfence();
}

// Pack count+1 into UB so zero remains the host's not-ready value.
__simt_vf__ __launch_bounds__(128) inline void PackNotifyHostCountsVf(
    const __gm__ int32_t* total, const __gm__ int32_t* experts, __ubuf__ uint32_t* packed,
    uint32_t offset, uint32_t length)
{
    for (uint32_t i = threadIdx.x; i < length; i += 128) {
        const uint32_t slot = offset + i;
        packed[i] = static_cast<uint32_t>(slot == 0 ? *total : experts[slot - 1]) + 1U;
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void notify_dispatch_kernel(
    GM_ADDR num_tokens_per_rank, GM_ADDR num_tokens_per_expert, GM_ADDR is_token_in_rank, GM_ADDR symmetric_buffer,
    GM_ADDR rank_prefix_matrix, GM_ADDR channel_prefix_matrix, GM_ADDR recv_channel_prefix_matrix,
    GM_ADDR num_recv_tokens,
    GM_ADDR num_recv_tokens_per_expert, GM_ADDR dispatch_plan, GM_ADDR send_head,
    ascend_deepep::DispatchTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
    util_set_ffts_config(ffts_addr);

    if ASCEND_IS_AIC {
        return;
    }

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();
    const int32_t rank = static_cast<int32_t>(tiling.rank);
    const int32_t world_size = static_cast<int32_t>(tiling.world_size);
    const int32_t num_tokens = static_cast<int32_t>(tiling.num_tokens);
    const int32_t num_experts = static_cast<int32_t>(tiling.num_experts);
    const int32_t num_channels = static_cast<int32_t>(tiling.num_channels);
    const int32_t rank_stride = static_cast<int32_t>(tiling.rank_stride);
    const int32_t expert_stride = static_cast<int32_t>(tiling.expert_stride);
    const int32_t channel_stride = static_cast<int32_t>(tiling.channel_stride);
    const int32_t num_local_experts = static_cast<int32_t>(tiling.num_local_experts);
    const int32_t local_expert_stride =
        (num_local_experts + kNotifyInt32AlignElems - 1) / kNotifyInt32AlignElems * kNotifyInt32AlignElems;

    // Notify starts after the fixed UDMA header (at byte zero for MTE).
    // Payload follows the host-reserved, 512-byte-aligned notify workspace.
    const int32_t published_rank_prefix_offset = 0;
    const int32_t published_channel_prefix_offset = world_size * world_size;
    const int32_t rank_counts_offset = published_channel_prefix_offset + world_size * channel_stride;
    const int32_t expert_counts_offset = rank_counts_offset + rank_stride;
    const int32_t channel_values_offset = expert_counts_offset + expert_stride;

    __gm__ int32_t* shmem_base = reinterpret_cast<__gm__ int32_t*>(
        symmetric_buffer + tiling.notify_workspace_offset);
    AscendC::GlobalTensor<int32_t> local_shmem_gm;
    AscendC::GlobalTensor<int32_t> channel_values_gm;
    AscendC::GlobalTensor<int32_t> published_rank_prefix_gm;
    AscendC::GlobalTensor<int32_t> published_channel_prefix_gm;
    AscendC::GlobalTensor<int32_t> rank_prefix_gm;
    AscendC::GlobalTensor<int32_t> channel_prefix_gm;
    AscendC::GlobalTensor<int32_t> recv_tokens_gm;
    AscendC::GlobalTensor<int32_t> recv_expert_counts_gm;
    AscendC::GlobalTensor<int32_t> local_rank_counts_gm;
    AscendC::GlobalTensor<int32_t> local_expert_counts_gm;
    AscendC::GlobalTensor<uint8_t> is_token_in_rank_gm;
    local_shmem_gm.SetGlobalBuffer(shmem_base);
    channel_values_gm.SetGlobalBuffer(shmem_base + channel_values_offset);
    published_rank_prefix_gm.SetGlobalBuffer(shmem_base + published_rank_prefix_offset);
    published_channel_prefix_gm.SetGlobalBuffer(shmem_base + published_channel_prefix_offset);
    rank_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix));
    channel_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix_matrix));
    recv_tokens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_recv_tokens));
    recv_expert_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_recv_tokens_per_expert));
    local_rank_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_tokens_per_rank));
    local_expert_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(num_tokens_per_expert));
    is_token_in_rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(is_token_in_rank));

    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<int32_t> accumulator_ub = resource.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> temporary_ub =
        resource.template GetBufferByByte<int32_t>(kMaxVectorTileElems * kNotifyInt32Bytes);
    AscendC::LocalTensor<uint8_t> route_row_ub = resource.template GetBufferByByte<uint8_t>(kRouteRowUbOffset);
    AscendC::LocalTensor<half> route_half_ub = resource.template GetBufferByByte<half>(kRouteHalfUbOffset);
    AscendC::DataCopyPadExtParams<int32_t> int32_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> route_pad_params{
        true, 0, static_cast<uint8_t>(rank_stride - world_size), static_cast<uint8_t>(0)};

    // The following dispatch payload reuses the symmetric region that may
    // still be read by a slower rank's previous combine reduction. Skip this
    // only for the Buffer's first workspace use.
    if (tiling.needs_notify_entry_barrier != 0) {
        aclshmemx_sync_vec_all();
    }

    // One AIV owns one logical channel and accumulates complete rank vectors.
    const int32_t tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
    const int32_t token_begin = NotifyIntMin(core_idx * tokens_per_channel, num_tokens);
    const int32_t token_end = NotifyIntMin(token_begin + tokens_per_channel, num_tokens);
    if (tiling.fused_layout_notify != 0) {
        ascend_deepep::BuildFusedDispatchPlan(resource, is_token_in_rank, dispatch_plan, send_head, tiling, core_idx);
    } else if (dispatch_plan != nullptr) {
        ascend_deepep::BuildDispatchPlan(resource, is_token_in_rank, dispatch_plan, send_head, tiling, core_idx);
    } else {
        AscendC::Duplicate(accumulator_ub, static_cast<int32_t>(0), rank_stride);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t token = token_begin; token < token_end; ++token) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams route_load_params(1, static_cast<uint32_t>(world_size), 0, 0, 0);
            AscendC::DataCopyPad(
                route_row_ub, is_token_in_rank_gm[token * world_size], route_load_params, route_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // A3 uses half as an exact bridge for the boolean values 0 and 1.
            AscendC::Cast(route_half_ub, route_row_ub, AscendC::RoundMode::CAST_NONE, rank_stride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(temporary_ub, route_half_ub, AscendC::RoundMode::CAST_RINT, rank_stride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, rank_stride);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams channel_count_store_params(
        1, static_cast<uint32_t>(rank_stride * kNotifyInt32Bytes), 0, 0, 0);
    AscendC::DataCopyPad(channel_values_gm[core_idx * rank_stride], accumulator_ub, channel_count_store_params);
    AscendC::SyncAll<true>();

    if (tiling.fused_layout_notify != 0) {
        ascend_deepep::ReduceFusedLayoutPartials(resource,
            reinterpret_cast<GM_ADDR>(shmem_base + channel_values_offset), num_tokens_per_rank,
            num_channels, rank_stride, world_size, core_idx, core_num);
        ascend_deepep::ReduceFusedLayoutPartials(resource,
            reinterpret_cast<GM_ADDR>(tiling.notify_expert_partials), num_tokens_per_expert,
            num_channels, expert_stride, num_experts, core_idx, core_num);
        AscendC::SyncAll<true>();
    }

    // Prefix channel counts as aligned rank-vector tiles in place.
    const int32_t rank_block_count = rank_stride / kNotifyInt32AlignElems;
    const int32_t rank_blocks_per_core = rank_block_count / core_num;
    const int32_t rank_extra_blocks = rank_block_count % core_num;
    const int32_t rank_worker_blocks = rank_blocks_per_core + (core_idx < rank_extra_blocks ? 1 : 0);
    const int32_t rank_begin_block = core_idx * rank_blocks_per_core + NotifyIntMin(core_idx, rank_extra_blocks);
    const int32_t rank_begin = rank_begin_block * kNotifyInt32AlignElems;
    const int32_t rank_worker_elems = rank_worker_blocks * kNotifyInt32AlignElems;
    const uint64_t channel_matrix_elems = static_cast<uint64_t>(num_channels) * rank_stride;
    if (channel_matrix_elems <= kMaxVectorTileElems) {
        if (core_idx == 0) {
            // Batch the small channel matrix instead of issuing one aligned
            // rank-vector DMA and round-trip event sequence per channel.
            const uint32_t bytes = static_cast<uint32_t>(channel_matrix_elems) * kNotifyInt32Bytes;
            AscendC::DataCopyExtParams copy(1, bytes, 0, 0, 0);
            AscendC::DataCopyPad(temporary_ub, channel_values_gm, copy, int32_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<ascend_deepep::PrefixAndPublishChannelsVf>(
                dim3(128), reinterpret_cast<__ubuf__ int32_t*>(temporary_ub.GetPhyAddr()),
                reinterpret_cast<__gm__ int32_t*>(channel_prefix_matrix), shmem_base + published_channel_prefix_offset,
                static_cast<uint32_t>(world_size), static_cast<uint32_t>(rank_stride),
                static_cast<uint32_t>(num_channels), static_cast<uint32_t>(channel_stride));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        }
    } else if (rank_worker_elems > 0) {
        AscendC::Duplicate(accumulator_ub, static_cast<int32_t>(0), rank_worker_elems);
        for (int32_t channel = 0; channel < num_channels; ++channel) {
            AscendC::DataCopyExtParams prefix_copy_params(
                1, static_cast<uint32_t>(rank_worker_elems * kNotifyInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(
                temporary_ub, channel_values_gm[channel * rank_stride + rank_begin], prefix_copy_params,
                int32_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, rank_worker_elems);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                channel_values_gm[channel * rank_stride + rank_begin], accumulator_ub, prefix_copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    AscendC::SyncAll<true>();

    // Transpose [channel, rank] into [rank, channel] and publish a padded copy.
    for (int32_t dst_rank = channel_matrix_elems <= kMaxVectorTileElems ? world_size : core_idx; dst_rank < world_size;
         dst_rank += core_num) {
        AscendC::Duplicate(accumulator_ub, static_cast<int32_t>(0), channel_stride);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t channel = 0; channel < num_channels; ++channel) {
            accumulator_ub.SetValue(channel, channel_values_gm.GetValue(channel * rank_stride + dst_rank));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams channel_output_store_params(
            1, static_cast<uint32_t>(num_channels * kNotifyInt32Bytes), 0, 0, 0);
        AscendC::DataCopyPad(channel_prefix_gm[dst_rank * num_channels], accumulator_ub, channel_output_store_params);
        AscendC::DataCopyExtParams published_channel_store_params(
            1, static_cast<uint32_t>(channel_stride * kNotifyInt32Bytes), 0, 0, 0);
        AscendC::DataCopyPad(
            published_channel_prefix_gm[dst_rank * channel_stride], accumulator_ub, published_channel_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    // Publish per-rank counts with an explicitly zeroed padded tail.
    const int32_t publish_rank_blocks_per_core = rank_block_count / core_num;
    const int32_t publish_rank_extra_blocks = rank_block_count % core_num;
    const int32_t publish_rank_worker_blocks =
        publish_rank_blocks_per_core + (core_idx < publish_rank_extra_blocks ? 1 : 0);
    const int32_t publish_rank_begin_block =
        core_idx * publish_rank_blocks_per_core + NotifyIntMin(core_idx, publish_rank_extra_blocks);
    int32_t publish_rank_current = publish_rank_begin_block * kNotifyInt32AlignElems;
    int32_t publish_rank_remaining = publish_rank_worker_blocks * kNotifyInt32AlignElems;
    while (publish_rank_remaining > 0) {
        const int32_t len = NotifyIntMin(publish_rank_remaining, kMaxVectorTileElems);
        const int32_t logical_len =
            publish_rank_current < world_size ? NotifyIntMin(len, world_size - publish_rank_current) : 0;
        AscendC::Duplicate(temporary_ub, static_cast<int32_t>(0), len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if (logical_len > 0) {
            AscendC::DataCopyExtParams rank_publish_load_params(
                1, static_cast<uint32_t>(logical_len * kNotifyInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(
                temporary_ub, local_rank_counts_gm[publish_rank_current], rank_publish_load_params, int32_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopyExtParams rank_publish_store_params(
            1, static_cast<uint32_t>(len * kNotifyInt32Bytes), 0, 0, 0);
        AscendC::DataCopyPad(
            local_shmem_gm[rank_counts_offset + publish_rank_current], temporary_ub, rank_publish_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        publish_rank_current += len;
        publish_rank_remaining -= len;
    }

    // Publish the global-expert histogram in aligned vector tiles.
    const int32_t expert_block_count = expert_stride / kNotifyInt32AlignElems;
    const int32_t expert_blocks_per_core = expert_block_count / core_num;
    const int32_t expert_extra_blocks = expert_block_count % core_num;
    const int32_t expert_worker_blocks = expert_blocks_per_core + (core_idx < expert_extra_blocks ? 1 : 0);
    const int32_t expert_begin_block = core_idx * expert_blocks_per_core + NotifyIntMin(core_idx, expert_extra_blocks);
    int32_t expert_current = expert_begin_block * kNotifyInt32AlignElems;
    int32_t expert_remaining = expert_worker_blocks * kNotifyInt32AlignElems;
    while (expert_remaining > 0) {
        const int32_t len = NotifyIntMin(expert_remaining, kMaxVectorTileElems);
        const int32_t logical_len = expert_current < num_experts ? NotifyIntMin(len, num_experts - expert_current) : 0;
        AscendC::Duplicate(temporary_ub, static_cast<int32_t>(0), len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if (logical_len > 0) {
            AscendC::DataCopyExtParams expert_publish_load_params(
                1, static_cast<uint32_t>(logical_len * kNotifyInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(
                temporary_ub, local_expert_counts_gm[expert_current], expert_publish_load_params, int32_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopyExtParams expert_publish_store_params(
            1, static_cast<uint32_t>(len * kNotifyInt32Bytes), 0, 0, 0);
        AscendC::DataCopyPad(
            local_shmem_gm[expert_counts_offset + expert_current], temporary_ub, expert_publish_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        expert_current += len;
        expert_remaining -= len;
    }
    aclshmemx_sync_vec_all();

    // Source owners gather complete rank rows once. rank_prefix_matrix holds
    // raw counts until each column tile is snapshotted before overwriting it.
    // Channel values are dead after channel publication/V4 above; reuse that
    // arena for aligned expert rows, never the published channel prefixes.
    const uint32_t rank_tile = NotifyCountTileWidth(world_size, kMaxVectorTileElems);
    const uint32_t expert_capacity = NotifyCountTileWidth(
        world_size, uint64_t(num_channels) * rank_stride);
    const uint32_t expert_tile = NotifyIntMin(expert_capacity, local_expert_stride);
    if (rank_tile != 0 && expert_tile != 0 && rank_stride <= kMaxVectorTileElems) {
        // UB ledger: rank snapshot [0, 64 KiB), expert snapshot/remote-load
        // staging [64 KiB, 128 KiB). Neither overlaps the old route scratch.
        static_assert(2 * kMaxVectorTileElems * kNotifyInt32Bytes <= kNotifyDispatchUbBytes);
        const uint32_t rows_per_owner = NotifyCountRowsPerOwner(world_size);
        const AscendC::DataCopyExtParams ranks_copy(1, world_size * kNotifyInt32Bytes, 0, 0, 0);
        const uint32_t first_experts = NotifyIntMin(expert_tile, num_local_experts);
        const AscendC::DataCopyExtParams first_experts_copy(1, first_experts * kNotifyInt32Bytes, 0, 0, 0);
        for (uint32_t first = core_idx * rows_per_owner; first < uint32_t(world_size);
             first += core_num * rows_per_owner) {
            const uint32_t end = NotifyIntMin(first + rows_per_owner, world_size);
            for (uint32_t source = first; source < end; ++source) {
                auto* remote_base = static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source));
                AscendC::GlobalTensor<int32_t> remote;
                remote.SetGlobalBuffer(remote_base);
                AscendC::DataCopyPad(temporary_ub, remote[rank_counts_offset], ranks_copy, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                AscendC::DataCopyPad(rank_prefix_gm[source * world_size], temporary_ub, ranks_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
                AscendC::DataCopyPad(temporary_ub, remote[expert_counts_offset + rank * num_local_experts],
                    first_experts_copy, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                AscendC::DataCopyPad(channel_values_gm[source * expert_tile], temporary_ub, first_experts_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            }
        }
        AscendC::SyncAll<true>();
        if (core_idx == 0) {
            for (uint32_t begin = 0; begin < uint32_t(world_size); begin += rank_tile) {
                const uint32_t columns = NotifyIntMin(rank_tile, world_size - begin);
                const uint32_t stride = (columns + 7U) / 8U * 8U;
                // GM strides are bytes; UB rows are padded to 32 bytes.
                // Keep the whole-matrix case as one contiguous DMA.
                if (columns == uint32_t(world_size) && stride == columns) {
                    const AscendC::DataCopyExtParams copy(1, world_size * columns * kNotifyInt32Bytes, 0, 0, 0);
                    AscendC::DataCopyPad(accumulator_ub, rank_prefix_gm, copy, int32_pad_params);
                } else {
                    const AscendC::DataCopyExtParams copy(
                        static_cast<uint16_t>(world_size), columns * kNotifyInt32Bytes,
                        (world_size - columns) * kNotifyInt32Bytes, 0, 0);
                    AscendC::DataCopyPad(accumulator_ub, rank_prefix_gm[begin], copy, int32_pad_params);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                asc_vf_call<PrefixNotifyCountsVf>(dim3(128),
                    reinterpret_cast<__ubuf__ int32_t*>(accumulator_ub.GetPhyAddr()),
                    reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix),
                    shmem_base + published_rank_prefix_offset, world_size, begin, columns, stride);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
        }
        for (uint32_t begin = 0; begin < uint32_t(num_local_experts); begin += expert_tile) {
            const uint32_t columns = NotifyIntMin(expert_tile, num_local_experts - begin);
            if (begin != 0) {
                // Finish consumers before scratch reuse. Empty-work AIVs
                // participate in both joins, just like source owners.
                AscendC::SyncAll<true>();
                const AscendC::DataCopyExtParams copy(1, columns * kNotifyInt32Bytes, 0, 0, 0);
                for (uint32_t source = core_idx; source < uint32_t(world_size); source += core_num) {
                    auto* remote_base = static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source));
                    AscendC::GlobalTensor<int32_t> remote;
                    remote.SetGlobalBuffer(remote_base);
                    AscendC::DataCopyPad(temporary_ub,
                        remote[expert_counts_offset + rank * num_local_experts + begin], copy, int32_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID4);
                    AscendC::DataCopyPad(channel_values_gm[source * expert_tile], temporary_ub, copy);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
                }
                AscendC::SyncAll<true>();
            }
            if (core_idx == 0) {
                const AscendC::DataCopyExtParams copy(1, world_size * expert_tile * kNotifyInt32Bytes, 0, 0, 0);
                AscendC::DataCopyPad(temporary_ub, channel_values_gm, copy, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                asc_vf_call<ReduceNotifyExpertsVf>(dim3(128),
                    reinterpret_cast<__ubuf__ int32_t*>(temporary_ub.GetPhyAddr()),
                    reinterpret_cast<__gm__ int32_t*>(num_recv_tokens_per_expert),
                    world_size, begin, columns, expert_tile);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
        }
    } else {
        // Build the complete source-rank prefix matrix with vector Add.
        if (rank_worker_elems > 0) {
            AscendC::Duplicate(accumulator_ub, static_cast<int32_t>(0), rank_worker_elems);
            const int32_t logical_len =
                rank_begin < world_size ? NotifyIntMin(rank_worker_elems, world_size - rank_begin) : 0;

            for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
                __gm__ int32_t* remote_shmem_base = static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source_rank));
                AscendC::GlobalTensor<int32_t> remote_shmem_gm;
                remote_shmem_gm.SetGlobalBuffer(remote_shmem_base);
                AscendC::DataCopyExtParams rank_prefix_load_params(
                    1, static_cast<uint32_t>(rank_worker_elems * kNotifyInt32Bytes), 0, 0, 0);
                AscendC::DataCopyPad(
                    temporary_ub, remote_shmem_gm[rank_counts_offset + rank_begin], rank_prefix_load_params,
                    int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, rank_worker_elems);
                if (logical_len > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyExtParams rank_prefix_store_params(
                        1, static_cast<uint32_t>(logical_len * kNotifyInt32Bytes), 0, 0, 0);
                    AscendC::DataCopyPad(
                        rank_prefix_gm[source_rank * world_size + rank_begin], accumulator_ub, rank_prefix_store_params);
                    AscendC::DataCopyPad(
                        published_rank_prefix_gm[source_rank * world_size + rank_begin], accumulator_ub,
                        rank_prefix_store_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                }
            }
        }

        // Reduce all source histograms for this rank's local experts.
        const int32_t local_expert_block_count = local_expert_stride / kNotifyInt32AlignElems;
        const int32_t local_expert_blocks_per_core = local_expert_block_count / core_num;
        const int32_t local_expert_extra_blocks = local_expert_block_count % core_num;
        const int32_t local_expert_worker_blocks =
            local_expert_blocks_per_core + (core_idx < local_expert_extra_blocks ? 1 : 0);
        const int32_t local_expert_begin_block =
            core_idx * local_expert_blocks_per_core + NotifyIntMin(core_idx, local_expert_extra_blocks);
        int32_t local_expert_current = local_expert_begin_block * kNotifyInt32AlignElems;
        int32_t local_expert_remaining = local_expert_worker_blocks * kNotifyInt32AlignElems;
        const int32_t global_expert_begin = rank * num_local_experts;
        while (local_expert_remaining > 0) {
            const int32_t len = NotifyIntMin(local_expert_remaining, kMaxVectorTileElems);
            const int32_t logical_len =
                local_expert_current < num_local_experts ? NotifyIntMin(len, num_local_experts - local_expert_current) : 0;
            AscendC::Duplicate(accumulator_ub, static_cast<int32_t>(0), len);
            for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
                AscendC::Duplicate(temporary_ub, static_cast<int32_t>(0), len);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                if (logical_len > 0) {
                    __gm__ int32_t* remote_shmem_base = static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source_rank));
                    AscendC::GlobalTensor<int32_t> remote_shmem_gm;
                    remote_shmem_gm.SetGlobalBuffer(remote_shmem_base);
                    AscendC::DataCopyExtParams expert_reduce_load_params(
                        1, static_cast<uint32_t>(logical_len * kNotifyInt32Bytes), 0, 0, 0);
                    AscendC::DataCopyPad(
                        temporary_ub, remote_shmem_gm[expert_counts_offset + global_expert_begin + local_expert_current],
                        expert_reduce_load_params, int32_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                }
                AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, len);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams expert_reduce_store_params(
                1, static_cast<uint32_t>(len * kNotifyInt32Bytes), 0, 0, 0);
            AscendC::DataCopyPad(recv_expert_counts_gm[local_expert_current], accumulator_ub, expert_reduce_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            local_expert_current += len;
            local_expert_remaining -= len;
        }
    }
    AscendC::SyncAll<true>();

    if (tiling.notify_host_counts != 0 && core_idx == 0) {
        // The mapped host address is only accessed by MTE3. Core 0 owns all
        // slots and reuses its aligned accumulator after the local barrier.
        AscendC::GlobalTensor<uint32_t> host_counts_gm;
        host_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling.notify_host_counts));
        auto packed_ub = accumulator_ub.ReinterpretCast<uint32_t>();
        const auto* total = reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix) +
                            uint64_t(world_size - 1) * world_size + rank;
        const uint32_t count = 1U + tiling.num_local_experts;
        for (uint32_t offset = 0; offset < count;) {
            const uint32_t remaining = count - offset;
            const uint32_t length = remaining < kMaxVectorTileElems ? remaining : kMaxVectorTileElems;
            asc_vf_call<PackNotifyHostCountsVf>(
                dim3(128), total, reinterpret_cast<__gm__ int32_t*>(num_recv_tokens_per_expert),
                reinterpret_cast<__ubuf__ uint32_t*>(packed_ub.GetPhyAddr()), offset, length);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopyExtParams copy(1, length * kNotifyInt32Bytes, 0, 0, 0);
            AscendC::DataCopyPad(host_counts_gm[offset], packed_ub, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
            offset += length;
        }
    }
    if (tiling.v2_rank_prefix != 0 && core_idx == 0) {
        asc_vf_call<ascend_deepep::V2HandleCountsVf>(dim3(128),
            reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix),
            reinterpret_cast<__gm__ int32_t*>(num_recv_tokens_per_expert),
            reinterpret_cast<__gm__ int32_t*>(tiling.v2_rank_prefix),
            reinterpret_cast<__gm__ int32_t*>(tiling.v2_expert_prefix),
            tiling.rank, tiling.world_size, tiling.num_local_experts, tiling.expert_alignment);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    }
    // Published channel rows were made visible by the earlier cross-rank
    // collective. The local SyncAll above also releases both scratch buffers.
    ascend_deepep::GatherRecvChannelPrefixes(
        symmetric_buffer + tiling.notify_workspace_offset,
        recv_channel_prefix_matrix, published_channel_prefix_offset,
        rank, world_size, num_channels, channel_stride, core_idx, core_num,
        temporary_ub, accumulator_ub);

    if (core_idx == 0) {
        accumulator_ub.SetValue(0, rank_prefix_gm.GetValue((world_size - 1) * world_size + rank));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams recv_tokens_store_params(1, static_cast<uint32_t>(kNotifyInt32Bytes), 0, 0, 0);
        AscendC::DataCopyPad(recv_tokens_gm[0], accumulator_ub, recv_tokens_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void notify_dispatch_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* num_tokens_per_rank, uint8_t* num_tokens_per_expert,
    uint8_t* is_token_in_rank, uint8_t* symmetric_buffer, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix,
    uint8_t* recv_channel_prefix_matrix, uint8_t* num_recv_tokens, uint8_t* num_recv_tokens_per_expert,
    uint8_t* dispatch_plan, uint8_t* send_head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    notify_dispatch_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kNotifyDispatchUbBytes), stream>>>(
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, symmetric_buffer, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, num_recv_tokens, num_recv_tokens_per_expert,
        dispatch_plan, send_head, *td, ffts_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void cached_notify_dispatch_kernel(uint64_t ffts_addr, uint32_t instance_id)
{
    aclshmemx_instance_ctx_set(instance_id);
    util_set_ffts_config(ffts_addr);
    aclshmemx_sync_vec_all();
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint32_t instance_id)
{
    cached_notify_dispatch_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(ffts_addr, instance_id);
}

// Legacy six-tensor handles have no private plan. Rebuild locally on the same
// stream after cached notify's collective, without touching cached SHMEM data.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void cached_dispatch_plan_kernel(
    GM_ADDR routes, GM_ADDR plan, GM_ADDR head, ascend_deepep::DispatchTilingData td, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC { return; }
    ascend_deepep::UbWorkspace resource;
    const uint32_t channel = get_block_idx() * get_subblockdim() + get_subblockid();
    ascend_deepep::BuildDispatchPlan(resource, routes, plan, head, td, channel);
}
extern "C" ASCEND_DEEPEP_EXPORT void cached_dispatch_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* routes, uint8_t* plan, uint8_t* head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    cached_dispatch_plan_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kNotifyDispatchUbBytes), stream>>>(
        routes, plan, head, *td, ffts_addr);
}
