#pragma once

#include "kernels/common/udma_shared_sender.h"
#include "common/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep::udma {
constexpr uint32_t kDispatchMetadataSimtThreads = 128;
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Notify already chose the stable ordinal. Metadata workers only publish it.
__simt_vf__ __launch_bounds__(kDispatchMetadataSimtThreads) inline void DispatchSourceIndicesVf(
    const __gm__ int32_t* plan, const __gm__ int32_t* ranks, const __gm__ int32_t* channels, __gm__ uint8_t* workspace,
    uint64_t src_offset, uint32_t rank, uint32_t world, uint32_t channel_count, uint32_t channel, uint32_t stride)
{
    constexpr uint32_t lanes = 16;
    for (uint32_t peer = threadIdx.x / lanes; peer < world; peer += kDispatchMetadataSimtThreads / lanes) {
        const int32_t previous = channel ? channels[uint64_t(peer) * channel_count + channel - 1] : 0;
        const int32_t count = channels[uint64_t(peer) * channel_count + channel] - previous;
        const int32_t base = (rank ? ranks[uint64_t(rank - 1) * world + peer] : 0) + previous;
        const uint64_t offset = (uint64_t(peer) * channel_count + channel) * stride;
        for (int32_t i = static_cast<int32_t>(threadIdx.x % lanes); i < count; i += lanes)
            simt::aclshmem_int32_p(
                reinterpret_cast<__gm__ int32_t*>(workspace + src_offset) + base + i, plan[offset + i], peer);
    }
    asc_threadfence();
}

// One VF thread per (token, top-k) slot. Notify precomputes the ordinal in
// send_head. Only the destination that owns the expert is written; destination
// rows are initialized to -1/0 so unwritten slots stay invalid.
__simt_vf__ __launch_bounds__(kDispatchMetadataSimtThreads) inline void DispatchTopkScatterSimtVf(
    __gm__ ascend_deepep::topk_idx_t* topk_idx, __gm__ float* topk_weights, __gm__ uint8_t* symmetric_buffer,
    const __gm__ int32_t* send_head, const __gm__ int32_t* rank_prefix, const __gm__ int32_t* channel_prefix,
    int32_t rank, int32_t num_channels, int32_t channel, int32_t token_begin, int32_t tile_tokens, int32_t num_topk,
    int32_t world_size, int32_t experts_per_rank, uint64_t topk_idx_buffer_offset, uint64_t topk_weights_buffer_offset,
    uint32_t topk_idx_row_stride, uint32_t topk_weights_row_stride, uint32_t scatter_topk_weights)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint64_t route_count = static_cast<uint64_t>(tile_tokens) * num_topk;
    for (uint64_t route = thread_index; route < route_count; route += kDispatchMetadataSimtThreads) {
        const int32_t local_token = static_cast<int32_t>(route / num_topk);
        const int32_t topk = static_cast<int32_t>(route % num_topk);
        const int32_t token = token_begin + local_token;
        const ascend_deepep::topk_idx_t expert = topk_idx[static_cast<uint64_t>(token) * num_topk + topk];
        if (expert < static_cast<ascend_deepep::topk_idx_t>(0)) {
            continue;
        }
        const int32_t dst_rank = static_cast<int32_t>(expert) / experts_per_rank;
        if (dst_rank < 0 || dst_rank >= world_size) {
            continue;
        }
        const int32_t ordinal = send_head[static_cast<uint64_t>(token) * world_size + dst_rank];
        if (ordinal < 0) {
            continue;
        }
        const int32_t recv_position =
            ordinal + (rank ? rank_prefix[static_cast<uint64_t>(rank - 1) * world_size + dst_rank] : 0) +
            (channel ? channel_prefix[static_cast<uint64_t>(dst_rank) * num_channels + channel - 1] : 0);
        const ascend_deepep::topk_idx_t local_expert =
            expert - static_cast<ascend_deepep::topk_idx_t>(dst_rank * experts_per_rank);
        __gm__ ascend_deepep::topk_idx_t* remote_topk_idx = reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(
            symmetric_buffer + topk_idx_buffer_offset + static_cast<uint64_t>(recv_position) * topk_idx_row_stride);
        if constexpr (sizeof(ascend_deepep::topk_idx_t) == sizeof(int64_t)) {
            simt::aclshmem_int64_p(
                reinterpret_cast<__gm__ int64_t*>(remote_topk_idx) + topk, static_cast<int64_t>(local_expert),
                dst_rank);
        } else {
            simt::aclshmem_int32_p(
                reinterpret_cast<__gm__ int32_t*>(remote_topk_idx) + topk, static_cast<int32_t>(local_expert),
                dst_rank);
        }
        if (scatter_topk_weights != 0U) {
            simt::aclshmem_float_p(
                reinterpret_cast<__gm__ float*>(
                    symmetric_buffer + topk_weights_buffer_offset +
                    static_cast<uint64_t>(recv_position) * topk_weights_row_stride) +
                    topk,
                topk_weights[static_cast<uint64_t>(token) * num_topk + topk], dst_rank);
        }
    }
    asc_threadfence();
}

// Sparse scatter only writes hit slots. Clear the valid destination rows on each
// invocation (including cached calls with top-k), before Begin's rank collective.
template <class Resource>
__aicore__ inline void InitializeDispatchTopk(
    Resource& resource, GM_ADDR workspace, const DispatchTilingData& td, uint32_t rows, uint32_t core, uint32_t cores)
{
    const uint32_t extra = rows % cores;
    const uint32_t count = rows / cores + uint32_t(core < extra);
    const uint32_t begin = core * (rows / cores) + (core < extra ? core : extra);
    if (count == 0 || td.num_topk == 0)
        return;
    constexpr uint32_t fill_bytes = 32 * 1024;
    auto fill = resource.template GetBufferByByte<uint8_t>(0);
    auto fill_int32 = resource.template GetBufferByByte<int32_t>(0);
    AscendC::GlobalTensor<uint8_t> local;
    local.SetGlobalBuffer(workspace);
    for (uint32_t field = 0; field < (td.has_topk_weights ? 2U : 1U); ++field) {
        const uint32_t element_bytes = field == 0 ? sizeof(topk_idx_t) : sizeof(float);
        const uint64_t row_bytes = uint64_t(td.num_topk) * element_bytes;
        const uint64_t offset =
            (field == 0 ? td.topk_idx_buffer_offset : td.topk_weights_buffer_offset) + uint64_t(begin) * row_bytes;
        AscendC::Duplicate(fill_int32, int32_t(field == 0 ? -1 : 0), fill_bytes / sizeof(int32_t));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        const uint64_t bytes = uint64_t(count) * row_bytes;
        for (uint64_t copied = 0; copied < bytes; copied += fill_bytes) {
            const uint32_t chunk = bytes - copied < fill_bytes ? uint32_t(bytes - copied) : fill_bytes;
            AscendC::DataCopyExtParams copy{1, chunk, 0, 0, 0};
            AscendC::DataCopyPad(local[offset + copied], fill, copy);
        }
        // MTE3 readers must finish before V changes -1 bytes to zero weights,
        // or Begin reuses the same UB for marker reset.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
}

// The payload owner/QP mapping is unchanged. Spare AIVs own whole logical
// channels for metadata and self-copy; when none are spare, every core does its
// metadata channel before entering the payload sender. No collective belongs
// inside this helper: payload cores must be able to run while metadata runs.
template <class Resource>
__aicore__ inline void DispatchMetadata(
    Resource& resource, GM_ADDR x, GM_ADDR scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR plan,
    GM_ADDR rank_prefix, GM_ADDR channel_prefix, GM_ADDR workspace, GM_ADDR send_head, const DispatchTilingData& td,
    uint32_t core, uint32_t cores, uint32_t stage_bytes, GM_ADDR recv_x = nullptr, GM_ADDR recv_scales = nullptr)
{
    const uint32_t first =
        td.udma_token_schedule ? TokenWorkers(cores) : MetadataFirst(td.world_size, td.udma_peers_per_worker, cores, td.udma_peer_order, td.udma_ranks_per_group, td.udma_workers_per_peer);
    if (core < first)
        return;
    const uint32_t worker = core - first;
    const uint32_t workers = cores - first;
    const bool build_handle = td.cached_mode == 0;
    const uint32_t world = td.world_size;
    const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1) / td.num_channels;
    const uint32_t plan_stride = (channel_tokens + 7U) / 8U * 8U;
    auto staging_ub = resource.template GetBufferByByte<uint8_t>(0);
    auto* rank_ptr = reinterpret_cast<__gm__ int32_t*>(rank_prefix);
    auto* channel_ptr = reinterpret_cast<__gm__ int32_t*>(channel_prefix);
    AscendC::GlobalTensor<int32_t> ranks, channels, plan_gm;
    AscendC::GlobalTensor<uint8_t> x_gm, scales_gm, output_x, output_scales;
    ranks.SetGlobalBuffer(rank_ptr);
    channels.SetGlobalBuffer(channel_ptr);
    plan_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
    x_gm.SetGlobalBuffer(x);
    scales_gm.SetGlobalBuffer(scales);
    output_x.SetGlobalBuffer(td.udma_output_control ? recv_x : workspace + td.x_buffer_offset);
    output_scales.SetGlobalBuffer(td.udma_output_control ? recv_scales : workspace + td.x_scales_buffer_offset);
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

    const uint32_t x_bytes = td.hidden * td.x_element_bytes;
    const uint32_t scale_bytes = td.num_scales * td.scale_element_bytes;
    const uint32_t largest = x_bytes > scale_bytes ? x_bytes : scale_bytes;
    const uint32_t stride = (largest + 31U) / 32U * 32U;
    const uint32_t slot_mask = stride * 2U <= stage_bytes ? 1U : 0U;
    const uint32_t experts_per_rank = td.num_experts / world;
    for (uint32_t ch = worker; ch < td.num_channels; ch += workers) {
        const uint64_t channel_begin = uint64_t(ch) * channel_tokens;
        const uint32_t begin = channel_begin < td.num_tokens ? uint32_t(channel_begin) : td.num_tokens;
        const uint32_t end = begin + channel_tokens < td.num_tokens ? begin + channel_tokens : td.num_tokens;
        if (build_handle) {
            asc_vf_call<DispatchSourceIndicesVf>(
                dim3(kDispatchMetadataSimtThreads), reinterpret_cast<__gm__ int32_t*>(plan), rank_ptr, channel_ptr,
                workspace, td.src_idx_buffer_offset, td.rank, world, td.num_channels, ch, plan_stride);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        }
        if (td.num_topk && experts_per_rank) {
            asc_vf_call<DispatchTopkScatterSimtVf>(
                dim3(kDispatchMetadataSimtThreads), reinterpret_cast<__gm__ topk_idx_t*>(topk_idx),
                reinterpret_cast<__gm__ float*>(topk_weights), workspace, reinterpret_cast<__gm__ int32_t*>(send_head),
                rank_ptr, channel_ptr, int32_t(td.rank), int32_t(td.num_channels), int32_t(ch), int32_t(begin),
                int32_t(end - begin), int32_t(td.num_topk), int32_t(world), int32_t(experts_per_rank),
                td.topk_idx_buffer_offset, td.topk_weights_buffer_offset, td.num_topk * uint32_t(sizeof(topk_idx_t)),
                td.num_topk * uint32_t(sizeof(float)), td.has_topk_weights);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        }

        // Metadata stores are issued before self-copy. Keep self rows in list
        // order and drain BOTH ping-pong slots before the next channel.
        uint32_t row = (td.rank ? ranks.GetValue((td.rank - 1) * world + td.rank) : 0) +
                       (ch ? channels.GetValue(td.rank * td.num_channels + ch - 1) : 0);
        uint32_t slot = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        const uint32_t previous = ch ? channels.GetValue(td.rank * td.num_channels + ch - 1) : 0;
        const uint32_t count = channels.GetValue(td.rank * td.num_channels + ch) - previous;
        if (td.udma_output_control && (row > td.num_output_tokens || count > td.num_output_tokens - row))
            trap();
        const uint64_t list_base = (uint64_t(td.rank) * td.num_channels + ch) * plan_stride;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t token = plan_gm.GetValue(list_base + i);
            const auto x_event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            auto staging = staging_ub[slot * stride];
            AscendC::DataCopyExtParams copy{1, x_bytes, 0, 0, 0};
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(x_event);
            AscendC::DataCopyPad(staging, x_gm[uint64_t(token) * x_bytes], copy, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(x_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(x_event);
            AscendC::DataCopyPad(output_x[uint64_t(row) * x_bytes], staging, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(x_event);
            slot = (slot + 1) & slot_mask;
            if (td.has_x_scales && scale_bytes) {
                const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
                staging = staging_ub[slot * stride];
                AscendC::DataCopyExtParams scale_copy{1, scale_bytes, 0, 0, 0};
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                AscendC::DataCopyPad(staging, scales_gm[uint64_t(token) * scale_bytes], scale_copy, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
                AscendC::DataCopyPad(
                    output_scales[uint64_t(row) * scale_bytes], staging, scale_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                slot = (slot + 1) & slot_mask;
            }
            ++row;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        // Drain MTE3 readers before any later V writer reuses staging UB.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    }
}
#endif
} // namespace ascend_deepep::udma
