#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "kernels/dispatch/dispatch_scale_pack.h"
#include "kernels/dispatch/dispatch_recv_copy.h"
#include "kernels/common/v2_handle.h"
#include "common/launch.h"
#include "common/types.h"
#include "kernels/dispatch/udma_dispatch_schedule.h"
#include "kernels/dispatch/udma_token_dispatch_schedule.h"
#include "kernels/dispatch/udma_dispatch_metadata.h"

using namespace AscendC;

namespace {

constexpr uint32_t kDispatchInt32Bytes = 4;
constexpr uint32_t kDispatchTopkIdxBytes = sizeof(ascend_deepep::topk_idx_t);
constexpr uint32_t kDispatchTopkWeightBytes = sizeof(float);
constexpr uint32_t kDispatchUbBytes = static_cast<uint32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr uint32_t kDispatchStageBytes = static_cast<uint32_t>(ascend_deepep::platform::kDispatchStageBytes);
constexpr uint32_t kDispatchPaddingTopkOffset = kDispatchStageBytes;
static_assert(kDispatchPaddingTopkOffset + 128U * kDispatchTopkIdxBytes <= kDispatchUbBytes);

__aicore__ inline int32_t DispatchMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t DispatchAlignUp32(uint32_t value)
{
    return (value + 31U) / 32U * 32U;
}

} // namespace

__aicore__ inline void V1DispatchUdmaKernelImpl(
    GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR num_tokens_per_rank,
    GM_ADDR is_token_in_rank, GM_ADDR num_tokens_per_expert, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_x_scales, GM_ADDR recv_topk_idx, GM_ADDR recv_topk_weights, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR recv_channel_prefix_matrix, GM_ADDR recv_src_idx, GM_ADDR send_head, GM_ADDR dispatch_plan,
    ascend_deepep::DispatchTilingData tiling, uint64_t ffts_addr)
{
    (void)num_tokens_per_rank;
    (void)num_tokens_per_expert;
    // Notify owns handle[2]; keep this argument for the dispatch entry ABI.
    (void)recv_channel_prefix_matrix;
    util_set_ffts_config(ffts_addr);

    if ASCEND_IS_AIC {
        return;
    }

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();
    const int32_t rank = static_cast<int32_t>(tiling.rank);
    const int32_t world_size = static_cast<int32_t>(tiling.world_size);
    const int32_t hidden = static_cast<int32_t>(tiling.hidden);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t num_channels = static_cast<int32_t>(tiling.num_channels);
    const int32_t num_output_tokens = static_cast<int32_t>(tiling.num_output_tokens);
    const bool rebuild_dispatch_handle = tiling.cached_mode == 0;
    const uint32_t hidden_bytes = static_cast<uint32_t>(hidden) * tiling.x_element_bytes;
    const uint32_t topk_idx_row_bytes = static_cast<uint32_t>(num_topk) * kDispatchTopkIdxBytes;
    const uint32_t topk_weights_row_bytes = static_cast<uint32_t>(num_topk) * kDispatchTopkWeightBytes;
    const uint32_t scale_row_bytes = tiling.num_scales * tiling.scale_element_bytes;
    // Preserve the conservative staging stride for src_idx/self copies.
    const uint32_t channel_prefix_row_bytes = static_cast<uint32_t>(num_channels) * kDispatchInt32Bytes;
    uint32_t max_staging_row_bytes = hidden_bytes;
    max_staging_row_bytes = topk_idx_row_bytes > max_staging_row_bytes ? topk_idx_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes =
        topk_weights_row_bytes > max_staging_row_bytes ? topk_weights_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes = scale_row_bytes > max_staging_row_bytes ? scale_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes =
        channel_prefix_row_bytes > max_staging_row_bytes ? channel_prefix_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes = kDispatchInt32Bytes > max_staging_row_bytes ? kDispatchInt32Bytes : max_staging_row_bytes;
    const uint32_t staging_buffer_stride = DispatchAlignUp32(max_staging_row_bytes);

    GM_ADDR dispatch_scales = x_scales;
    AscendC::GlobalTensor<int32_t> rank_prefix_gm;
    AscendC::GlobalTensor<uint8_t> local_shmem_gm;

    rank_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix));
    local_shmem_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer));

    if (tiling.has_x_scales != 0) {
        if (tiling.needs_x_scales_pack != 0) {
            __gm__ uint8_t* packed_x_scales =
                reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_scales_input_buffer_offset;
            asc_vf_call<ascend_deepep::PackV1DispatchScalesSimtVf>(
                dim3(ascend_deepep::kDispatchScalePackThreads), reinterpret_cast<const __gm__ uint32_t*>(x_scales),
                reinterpret_cast<__gm__ uint32_t*>(packed_x_scales), tiling.num_tokens, tiling.num_scales,
                tiling.scale_token_stride, tiling.scale_hidden_stride, static_cast<uint32_t>(core_idx),
                static_cast<uint32_t>(core_num));
            AscendC::SyncAll<true>();
            dispatch_scales = packed_x_scales;
        }
    }

    const uint32_t staging_capacity = kDispatchStageBytes;
    const int32_t staging_buffer_mask = staging_buffer_stride * 2U <= staging_capacity ? 1 : 0;
    ascend_deepep::UbWorkspace resource;
    auto staging_pingpong_ub = resource.template GetBufferByByte<uint8_t>(0);
    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};

    if (num_topk > 0) {
        const uint32_t rows = DispatchMin(
            rank_prefix_gm.GetValue((world_size - 1) * world_size + rank), num_output_tokens);
        ascend_deepep::udma::InitializeDispatchTopk(resource, symmetric_buffer, tiling, rows, core_idx, core_num);
    }
    if (tiling.udma_output_control && core_idx == 0) {
        ascend_deepep::udma::PrepareDispatchOutput(resource, tiling.udma_output_control, rank, world_size,
            recv_x, recv_x_scales, tiling.num_output_tokens,
            static_cast<uint32_t>(rank_prefix_gm.GetValue((world_size - 1) * world_size + rank)),
            hidden_bytes, tiling.has_x_scales ? scale_row_bytes : 0);
    }
    // Clears incoming payload markers and publishes this invocation's output
    // addresses/top-k initialization before remote writes. Every AIV participates.
    ascend_deepep::udma::Begin(symmetric_buffer, tiling.udma_completion_signal_offset, rank, world_size,
        core_idx, core_num, resource.template GetBufferByByte<uint64_t>(0),
        tiling.udma_token_schedule ? ascend_deepep::udma::TokenWorkers(core_num) : tiling.udma_workers_per_peer * tiling.udma_qps_per_worker);

    // With spare cores, payload owners skip this helper and immediately start
    // UDMA while the other cores build metadata and copy self channels.
    ascend_deepep::udma::DispatchMetadata(resource, x, dispatch_scales, topk_idx, topk_weights,
        dispatch_plan, rank_prefix_matrix, channel_prefix_matrix, symmetric_buffer, send_head,
        tiling, core_idx, core_num, kDispatchStageBytes, recv_x, recv_x_scales);
    if (tiling.udma_token_schedule) {
        ascend_deepep::udma::DispatchTokenRemote(resource, x, dispatch_scales,
            dispatch_plan, rank_prefix_matrix, channel_prefix_matrix, symmetric_buffer, tiling, core_idx, core_num);
    } else {
        ascend_deepep::udma::DispatchRemote(resource, x, dispatch_scales,
            dispatch_plan, rank_prefix_matrix, channel_prefix_matrix, symmetric_buffer, tiling, core_idx, core_num);
    }

    const int32_t actual_recv_tokens = rank_prefix_gm.GetValue((world_size - 1) * world_size + rank);
    const int32_t recv_tokens_to_copy = DispatchMin(actual_recv_tokens, num_output_tokens);
    const int32_t tokens_per_worker = recv_tokens_to_copy / core_num;
    const int32_t extra_tokens = recv_tokens_to_copy % core_num;
    const int32_t worker_tokens = tokens_per_worker + (core_idx < extra_tokens ? 1 : 0);
    const int32_t recv_token_begin = core_idx * tokens_per_worker + DispatchMin(core_idx, extra_tokens);

    AscendC::GlobalTensor<uint8_t> recv_src_idx_bytes_gm;
    if (rebuild_dispatch_handle) {
        recv_src_idx_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_src_idx));
    }

    const int32_t src_idx_tile_capacity = static_cast<int32_t>(staging_buffer_stride / kDispatchInt32Bytes);

    // Source indices already use the public compact receive order in SHMEM.
    // Every AIV copies its disjoint output interval; GM endpoints may be
    // unaligned, while both ping-pong UB starts remain 32-byte aligned.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    if (rebuild_dispatch_handle) {
        int32_t copied_src_idx_tokens = 0;
        int32_t src_idx_buffer_idx = 0;
        while (copied_src_idx_tokens < worker_tokens) {
            const int32_t tile_tokens = DispatchMin(src_idx_tile_capacity, worker_tokens - copied_src_idx_tokens);
            const uint32_t tile_bytes = static_cast<uint32_t>(tile_tokens) * kDispatchInt32Bytes;
            const auto src_idx_event_id = src_idx_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<uint8_t> src_idx_staging_ub =
                staging_pingpong_ub[src_idx_buffer_idx * staging_buffer_stride];
            AscendC::DataCopyExtParams src_idx_copy_params(1, tile_bytes, 0, 0, 0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(src_idx_event_id);
            AscendC::DataCopyPad(
                src_idx_staging_ub,
                local_shmem_gm
                    [tiling.src_idx_buffer_offset +
                     static_cast<uint64_t>(recv_token_begin + copied_src_idx_tokens) * kDispatchInt32Bytes],
                src_idx_copy_params, uint8_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(src_idx_event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(src_idx_event_id);
            AscendC::DataCopyPad(
                recv_src_idx_bytes_gm
                    [static_cast<uint64_t>(recv_token_begin + copied_src_idx_tokens) * kDispatchInt32Bytes],
                src_idx_staging_ub, src_idx_copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(src_idx_event_id);
            copied_src_idx_tokens += tile_tokens;
            src_idx_buffer_idx = (src_idx_buffer_idx + 1) & staging_buffer_mask;
        }
    }
    // Drain src_idx slots before the output-copy helper reuses their UB.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // Mailbox-enabled payload/scales are already in their final storage. Keep
    // the internal staging baseline and the SIMT metadata materialization here.
    if (!tiling.udma_output_control) {
        ascend_deepep::CopyDispatchRecvToGm(
            recv_x, symmetric_buffer, tiling.x_buffer_offset,
            static_cast<uint64_t>(recv_token_begin) * hidden_bytes,
            static_cast<uint64_t>(worker_tokens) * hidden_bytes, staging_pingpong_ub, staging_capacity);
        if (tiling.has_x_scales != 0) {
            ascend_deepep::CopyDispatchRecvToGm(
                recv_x_scales, symmetric_buffer, tiling.x_scales_buffer_offset,
                static_cast<uint64_t>(recv_token_begin) * scale_row_bytes,
                static_cast<uint64_t>(worker_tokens) * scale_row_bytes, staging_pingpong_ub, staging_capacity);
        }
    }
    if (num_topk > 0) {
        ascend_deepep::CopyDispatchRecvToGm(
            recv_topk_idx, symmetric_buffer, tiling.topk_idx_buffer_offset,
            static_cast<uint64_t>(recv_token_begin) * topk_idx_row_bytes,
            static_cast<uint64_t>(worker_tokens) * topk_idx_row_bytes, staging_pingpong_ub, staging_capacity);
        if (tiling.has_topk_weights != 0) {
            ascend_deepep::CopyDispatchRecvToGm(
                recv_topk_weights, symmetric_buffer, tiling.topk_weights_buffer_offset,
                static_cast<uint64_t>(recv_token_begin) * topk_weights_row_bytes,
                static_cast<uint64_t>(worker_tokens) * topk_weights_row_bytes, staging_pingpong_ub, staging_capacity);
        }
    }

    if (tiling.v2_rank_prefix != 0) {
        // The existing completion boundary has made all local SHMEM metadata
        // visible. Read it directly: no dependence on another AIV's GM copy.
        asc_vf_call<ascend_deepep::V2HandleSourceVf>(dim3(128),
            reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix),
            reinterpret_cast<__gm__ int32_t*>(symmetric_buffer + tiling.src_idx_buffer_offset),
            reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(symmetric_buffer + tiling.topk_idx_buffer_offset),
            reinterpret_cast<__gm__ int32_t*>(tiling.v2_src_metadata),
            tiling.rank, tiling.world_size, tiling.num_max_tokens_per_rank / tiling.world_size, tiling.num_topk,
            tiling.num_output_tokens, uint32_t(core_idx), uint32_t(core_num));
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<ascend_deepep::V2HandleSlotsVf>(dim3(128),
            reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx),
            reinterpret_cast<__gm__ int32_t*>(send_head),
            reinterpret_cast<__gm__ int32_t*>(channel_prefix_matrix),
            reinterpret_cast<__gm__ int32_t*>(tiling.v2_dst_slots),
            tiling.rank, tiling.world_size, tiling.num_max_tokens_per_rank / tiling.world_size, tiling.num_local_experts,
            tiling.num_topk, tiling.num_tokens, tiling.num_channels, uint32_t(core_idx), uint32_t(core_num));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    }

    // Padding belongs to the returned storage, whether owned GM or a SHMEM view.
    const int32_t padding_tokens = num_output_tokens - recv_tokens_to_copy;
    const int32_t padding_tokens_per_worker = padding_tokens / core_num;
    const int32_t padding_extra_tokens = padding_tokens % core_num;
    const int32_t worker_padding_tokens = padding_tokens_per_worker + (core_idx < padding_extra_tokens ? 1 : 0);
    const int32_t padding_token_begin =
        recv_tokens_to_copy + core_idx * padding_tokens_per_worker + DispatchMin(core_idx, padding_extra_tokens);
    if (worker_padding_tokens > 0) {
        AscendC::GlobalTensor<uint8_t> output_x_gm;
        AscendC::GlobalTensor<uint8_t> output_scales_gm;
        AscendC::GlobalTensor<uint8_t> output_topk_idx_gm;
        AscendC::GlobalTensor<uint8_t> output_weights_gm;
        output_x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_x));
        if (tiling.has_x_scales != 0) {
            output_scales_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_x_scales));
        }
        if (num_topk > 0) {
            output_topk_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_topk_idx));
            if (tiling.has_topk_weights != 0) {
                output_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_topk_weights));
            }
        }
        // The next staging writer is Duplicate (V), not MTE2.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        uint32_t zero_row_bytes = hidden_bytes;
        zero_row_bytes = scale_row_bytes > zero_row_bytes ? scale_row_bytes : zero_row_bytes;
        zero_row_bytes = topk_weights_row_bytes > zero_row_bytes ? topk_weights_row_bytes : zero_row_bytes;
        const uint32_t zero_elements = DispatchAlignUp32(zero_row_bytes) / sizeof(int32_t);
        AscendC::LocalTensor<uint8_t> zero_staging_ub = staging_pingpong_ub;
        AscendC::LocalTensor<int32_t> zero_staging_int32_ub = zero_staging_ub.ReinterpretCast<int32_t>();
        AscendC::Duplicate(zero_staging_int32_ub, static_cast<int32_t>(0), zero_elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // Metadata/payload work has finished. Keep the invalid top-k row
        // outside the zero-filled staging region while both feed MTE3.
        auto topk_idx_ub =
            resource.template GetBufferByByte<ascend_deepep::topk_idx_t>(kDispatchPaddingTopkOffset);
        auto topk_idx_bytes_ub = topk_idx_ub.ReinterpretCast<uint8_t>();
        if (num_topk > 0) {
            for (int32_t topk = 0; topk < num_topk; ++topk) {
                topk_idx_ub.SetValue(topk, static_cast<ascend_deepep::topk_idx_t>(-1));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
        }

        AscendC::DataCopyExtParams zero_x_store_params(1, hidden_bytes, 0, 0, 0);
        AscendC::DataCopyExtParams zero_scale_store_params(1, scale_row_bytes, 0, 0, 0);
        AscendC::DataCopyExtParams invalid_topk_store_params(1, topk_idx_row_bytes, 0, 0, 0);
        AscendC::DataCopyExtParams zero_topk_weights_store_params(1, topk_weights_row_bytes, 0, 0, 0);
        for (int32_t local_padding_token = 0; local_padding_token < worker_padding_tokens; ++local_padding_token) {
            const int32_t padding_token = padding_token_begin + local_padding_token;
            AscendC::DataCopyPad(
                output_x_gm[static_cast<uint64_t>(padding_token) * hidden_bytes],
                zero_staging_ub, zero_x_store_params);
            if (tiling.has_x_scales != 0 && scale_row_bytes > 0) {
                AscendC::DataCopyPad(
                    output_scales_gm[static_cast<uint64_t>(padding_token) * scale_row_bytes],
                    zero_staging_ub, zero_scale_store_params);
            }
            if (num_topk > 0) {
                AscendC::DataCopyPad(
                    output_topk_idx_gm[static_cast<uint64_t>(padding_token) * topk_idx_row_bytes],
                    topk_idx_bytes_ub, invalid_topk_store_params);
                if (tiling.has_topk_weights != 0) {
                    AscendC::DataCopyPad(
                        output_weights_gm[static_cast<uint64_t>(padding_token) * topk_weights_row_bytes],
                        zero_staging_ub, zero_topk_weights_store_params);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_dispatch_udma_kernel(
    GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR num_tokens_per_rank,
    GM_ADDR is_token_in_rank, GM_ADDR num_tokens_per_expert, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_x_scales, GM_ADDR recv_topk_idx, GM_ADDR recv_topk_weights, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR recv_channel_prefix_matrix, GM_ADDR recv_src_idx, GM_ADDR send_head, GM_ADDR dispatch_plan,
    ascend_deepep::DispatchTilingData tiling, uint64_t ffts_addr)
{
    V1DispatchUdmaKernelImpl(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, dispatch_plan, tiling, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_udma_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* dispatch_plan, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    v1_dispatch_udma_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kDispatchUbBytes), stream>>>(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, dispatch_plan, *td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT uint64_t v1_token_schedule_abi_3() {
    return (uint64_t(sizeof(ascend_deepep::DispatchTilingData)) << 32) |
           uint64_t(sizeof(ascend_deepep::CombineTilingData));
}
