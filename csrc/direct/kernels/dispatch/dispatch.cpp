#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "kernels/dispatch/dispatch_scale_pack.h"
#include "kernels/dispatch/dispatch_recv_copy.h"
#include "kernels/common/v2_handle.h"
#include "common/launch.h"
#include "common/types.h"

using namespace AscendC;

namespace {

constexpr uint32_t kDispatchInt32Bytes = 4;
constexpr int32_t kDispatchInt32AlignElems = 32 / kDispatchInt32Bytes;
constexpr uint32_t kDispatchTopkIdxBytes = sizeof(ascend_deepep::topk_idx_t);
constexpr uint32_t kDispatchTopkWeightBytes = sizeof(float);
constexpr uint32_t kDispatchUbBytes = static_cast<uint32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr uint32_t kDispatchStageBytes = static_cast<uint32_t>(ascend_deepep::platform::kDispatchStageBytes);
constexpr uint32_t kDispatchMaxRanks = 1024;
constexpr uint32_t kDispatchMaxTopk = 128;
constexpr uint32_t kSrcIdxCollectBytes = 32U * 1024U;
constexpr uint32_t kSrcIdxRankBaseBytes = kDispatchMaxRanks * kDispatchInt32Bytes;
constexpr uint32_t kSrcIdxValuesUbOffset = kSrcIdxRankBaseBytes;
constexpr int32_t kSrcIdxCollectCapacity =
    static_cast<int32_t>((kSrcIdxCollectBytes - kSrcIdxValuesUbOffset) / kDispatchInt32Bytes);
constexpr uint32_t kRecvPositionUbOffset = kDispatchStageBytes;
constexpr uint32_t kSendOrdinalUbOffset = kRecvPositionUbOffset + kDispatchMaxRanks * kDispatchInt32Bytes;
constexpr uint32_t kSendHeadRowUbOffset = kSendOrdinalUbOffset + kDispatchMaxRanks * kDispatchInt32Bytes;
constexpr uint32_t kTopkIdxUbOffset = kSendHeadRowUbOffset + kDispatchMaxRanks * kDispatchInt32Bytes;
constexpr uint32_t kTopkWeightsUbOffset = kTopkIdxUbOffset + kDispatchMaxTopk * kDispatchTopkIdxBytes;
constexpr uint32_t kScalarUbOffset = kTopkWeightsUbOffset + kDispatchMaxTopk * kDispatchTopkWeightBytes;
constexpr uint32_t kSendHeadScalarUbOffset = kScalarUbOffset + 32;

static_assert(kSrcIdxCollectBytes < kDispatchStageBytes);
static_assert(kSrcIdxCollectCapacity > 0);
static_assert(kScalarUbOffset + 32 <= kDispatchUbBytes);
static_assert(kSendHeadScalarUbOffset + 32 <= kDispatchUbBytes);

__aicore__ inline int32_t DispatchMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t DispatchAlignUp32(uint32_t value)
{
    return (value + 31U) / 32U * 32U;
}

} // namespace

__aicore__ inline void V1DispatchMteKernelImpl(
    GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR num_tokens_per_rank,
    GM_ADDR is_token_in_rank, GM_ADDR num_tokens_per_expert, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_x_scales, GM_ADDR recv_topk_idx, GM_ADDR recv_topk_weights, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR recv_channel_prefix_matrix, GM_ADDR recv_src_idx, GM_ADDR send_head,
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
    const int32_t num_tokens = static_cast<int32_t>(tiling.num_tokens);
    const int32_t hidden = static_cast<int32_t>(tiling.hidden);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t num_experts = static_cast<int32_t>(tiling.num_experts);
    const int32_t num_channels = static_cast<int32_t>(tiling.num_channels);
    const int32_t rank_stride = static_cast<int32_t>(tiling.rank_stride);
    const int32_t num_output_tokens = static_cast<int32_t>(tiling.num_output_tokens);
    const bool rebuild_dispatch_handle = tiling.cached_mode == 0;
    const uint32_t hidden_bytes = static_cast<uint32_t>(hidden) * tiling.x_element_bytes;
    const uint32_t topk_idx_row_bytes = static_cast<uint32_t>(num_topk) * kDispatchTopkIdxBytes;
    const uint32_t topk_weights_row_bytes = static_cast<uint32_t>(num_topk) * kDispatchTopkWeightBytes;
    const uint32_t scale_row_bytes = tiling.num_scales * tiling.scale_element_bytes;
    // Preserve the conservative staging stride for src_idx/self copies.
    const uint32_t channel_prefix_row_bytes = static_cast<uint32_t>(num_channels) * kDispatchInt32Bytes;
    const uint32_t x_row_stride = hidden_bytes;
    const uint32_t topk_idx_row_stride = topk_idx_row_bytes;
    const uint32_t topk_weights_row_stride = topk_weights_row_bytes;
    const uint32_t scale_row_stride = scale_row_bytes;
    uint32_t max_staging_row_bytes = hidden_bytes;
    max_staging_row_bytes = topk_idx_row_bytes > max_staging_row_bytes ? topk_idx_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes =
        topk_weights_row_bytes > max_staging_row_bytes ? topk_weights_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes = scale_row_bytes > max_staging_row_bytes ? scale_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes =
        channel_prefix_row_bytes > max_staging_row_bytes ? channel_prefix_row_bytes : max_staging_row_bytes;
    max_staging_row_bytes = kDispatchInt32Bytes > max_staging_row_bytes ? kDispatchInt32Bytes : max_staging_row_bytes;
    const uint32_t staging_buffer_stride = DispatchAlignUp32(max_staging_row_bytes);

    AscendC::GlobalTensor<uint8_t> x_gm;
    AscendC::GlobalTensor<uint8_t> x_scales_gm;
    AscendC::GlobalTensor<ascend_deepep::topk_idx_t> topk_idx_gm;
    AscendC::GlobalTensor<float> topk_weights_gm;
    AscendC::GlobalTensor<uint8_t> is_token_in_rank_gm;
    AscendC::GlobalTensor<int32_t> rank_prefix_gm;
    AscendC::GlobalTensor<int32_t> channel_prefix_gm;
    AscendC::GlobalTensor<int32_t> send_head_gm;
    AscendC::GlobalTensor<uint8_t> local_shmem_gm;

    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    is_token_in_rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(is_token_in_rank));
    rank_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix));
    channel_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix_matrix));
    if (rebuild_dispatch_handle) {
        send_head_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_head));
    }
    local_shmem_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer));

    if (num_topk > 0) {
        topk_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(topk_idx));
    }
    if (tiling.has_topk_weights != 0) {
        topk_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topk_weights));
    }
    if (tiling.has_x_scales != 0) {
        __gm__ uint8_t* dispatch_x_scales = reinterpret_cast<__gm__ uint8_t*>(x_scales);
        if (tiling.needs_x_scales_pack != 0) {
            __gm__ uint8_t* packed_x_scales =
                reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_scales_input_buffer_offset;
            asc_vf_call<ascend_deepep::PackV1DispatchScalesSimtVf>(
                dim3(ascend_deepep::kDispatchScalePackThreads), reinterpret_cast<const __gm__ uint32_t*>(x_scales),
                reinterpret_cast<__gm__ uint32_t*>(packed_x_scales), tiling.num_tokens, tiling.num_scales,
                tiling.scale_token_stride, tiling.scale_hidden_stride, static_cast<uint32_t>(core_idx),
                static_cast<uint32_t>(core_num));
            AscendC::SyncAll<true>();
            dispatch_x_scales = packed_x_scales;
        }
        x_scales_gm.SetGlobalBuffer(dispatch_x_scales);
    }

    const int32_t tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
    int32_t channel_src_idx_storage_count = 0;
    for (int32_t dst_rank = 0; dst_rank < world_size; ++dst_rank) {
        const int32_t channel_prefix_begin =
            core_idx > 0 ? channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx - 1) : 0;
        const int32_t channel_prefix_end = channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx);
        const int32_t channel_token_count = channel_prefix_end - channel_prefix_begin;
        if (channel_token_count == 0) {
            continue;
        }
        channel_src_idx_storage_count +=
            (channel_token_count + kDispatchInt32AlignElems - 1) / kDispatchInt32AlignElems * kDispatchInt32AlignElems;
    }
    const bool collect_src_idx = rebuild_dispatch_handle && channel_src_idx_storage_count <= kSrcIdxCollectCapacity &&
                                 staging_buffer_stride <= kDispatchStageBytes - kSrcIdxCollectBytes;
    const uint32_t staging_buffer_base = collect_src_idx ? kSrcIdxCollectBytes : 0U;
    const uint32_t staging_capacity = kDispatchStageBytes - staging_buffer_base;
    // Common rows use ping-pong. Large rows use one staging buffer, and rows
    // that need the complete staging region send source indices inline.
    const int32_t staging_buffer_num = staging_buffer_stride * 2U <= staging_capacity ? 2 : 1;
    const int32_t staging_buffer_mask = staging_buffer_num - 1;

    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<uint8_t> staging_pingpong_ub =
        resource.template GetBufferByByte<uint8_t>(staging_buffer_base);
    AscendC::LocalTensor<int32_t> src_idx_rank_base_ub = resource.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> src_idx_values_ub =
        resource.template GetBufferByByte<int32_t>(kSrcIdxValuesUbOffset);
    AscendC::LocalTensor<int32_t> recv_position_ub =
        resource.template GetBufferByByte<int32_t>(kRecvPositionUbOffset);
    AscendC::LocalTensor<int32_t> send_ordinal_ub =
        resource.template GetBufferByByte<int32_t>(kSendOrdinalUbOffset);
    AscendC::LocalTensor<int32_t> send_head_row_ub =
        resource.template GetBufferByByte<int32_t>(kSendHeadRowUbOffset);
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> topk_idx_ub =
        resource.template GetBufferByByte<ascend_deepep::topk_idx_t>(kTopkIdxUbOffset);
    AscendC::LocalTensor<uint8_t> topk_idx_bytes_ub = topk_idx_ub.ReinterpretCast<uint8_t>();
    AscendC::LocalTensor<float> topk_weights_ub = resource.template GetBufferByByte<float>(kTopkWeightsUbOffset);
    AscendC::LocalTensor<uint8_t> topk_weights_bytes_ub = topk_weights_ub.ReinterpretCast<uint8_t>();
    AscendC::LocalTensor<int32_t> scalar_ub = resource.template GetBufferByByte<int32_t>(kScalarUbOffset);
    AscendC::LocalTensor<int32_t> send_head_scalar_ub =
        resource.template GetBufferByByte<int32_t>(kSendHeadScalarUbOffset);
    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};

    {
        // One AIV owns one channel, including all per-destination cursors.
        AscendC::Duplicate(recv_position_ub, static_cast<int32_t>(0), rank_stride);
        AscendC::Duplicate(send_ordinal_ub, static_cast<int32_t>(0), rank_stride);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        int32_t src_idx_rank_base = 0;
        for (int32_t dst_rank = 0; dst_rank < world_size; ++dst_rank) {
            const int32_t rank_base = rank > 0 ? rank_prefix_gm.GetValue((rank - 1) * world_size + dst_rank) : 0;
            const int32_t channel_prefix_begin =
                core_idx > 0 ? channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx - 1) : 0;
            recv_position_ub.SetValue(dst_rank, rank_base + channel_prefix_begin);
            if (collect_src_idx) {
                const int32_t channel_prefix_end = channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx);
                const int32_t channel_token_count = channel_prefix_end - channel_prefix_begin;
                if (channel_token_count > 0) {
                    src_idx_rank_base_ub.SetValue(dst_rank, src_idx_rank_base);
                    src_idx_rank_base += (channel_token_count + kDispatchInt32AlignElems - 1) /
                                         kDispatchInt32AlignElems * kDispatchInt32AlignElems;
                }
            }
        }

        const int32_t token_begin = DispatchMin(core_idx * tokens_per_channel, num_tokens);
        const int32_t token_end = DispatchMin(token_begin + tokens_per_channel, num_tokens);
        const int32_t experts_per_rank = num_experts > 0 ? num_experts / world_size : 0;

        // One MTE3->MTE2 credit per staging half, matching planning's ping-pong
        // protocol. A half is waited only when MTE2 is about to overwrite it.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        if (!collect_src_idx) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        int32_t send_buffer_idx = 0;

        for (int32_t token = token_begin; token < token_end; ++token) {
            if (rebuild_dispatch_handle) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                AscendC::Duplicate(send_head_row_ub, static_cast<int32_t>(-1), rank_stride);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            }

            for (int32_t dst_rank = 0; dst_rank < world_size; ++dst_rank) {
                if (is_token_in_rank_gm.GetValue(token * world_size + dst_rank) == 0) {
                    continue;
                }
                const int32_t recv_position = recv_position_ub.GetValue(dst_rank);
                const int32_t send_ordinal = send_ordinal_ub.GetValue(dst_rank);
                recv_position_ub.SetValue(dst_rank, recv_position + 1);
                send_ordinal_ub.SetValue(dst_rank, send_ordinal + 1);
                if (rebuild_dispatch_handle) {
                    send_head_row_ub.SetValue(dst_rank, send_ordinal);
                    if (collect_src_idx) {
                        const int32_t local_position = src_idx_rank_base_ub.GetValue(dst_rank) + send_ordinal;
                        src_idx_values_ub.SetValue(local_position, token);
                    }
                }

                __gm__ uint8_t* remote_shmem_base = static_cast<__gm__ uint8_t*>(
                    aclshmem_ptr(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer), dst_rank));
                AscendC::GlobalTensor<uint8_t> remote_shmem_gm;
                remote_shmem_gm.SetGlobalBuffer(remote_shmem_base);

                const auto x_event_id = send_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<uint8_t> x_staging_ub =
                    staging_pingpong_ub[send_buffer_idx * staging_buffer_stride];
                AscendC::DataCopyExtParams x_copy_params(1, hidden_bytes, 0, 0, 0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(x_event_id);
                AscendC::DataCopyPad(
                    x_staging_ub, x_gm[static_cast<uint64_t>(token) * hidden_bytes], x_copy_params, uint8_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(x_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(x_event_id);
                AscendC::DataCopyPad(
                    remote_shmem_gm[tiling.x_buffer_offset + static_cast<uint64_t>(recv_position) * x_row_stride],
                    x_staging_ub, x_copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(x_event_id);
                send_buffer_idx = (send_buffer_idx + 1) & staging_buffer_mask;

                if (rebuild_dispatch_handle && !collect_src_idx) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                    scalar_ub.SetValue(0, token);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
                    AscendC::GlobalTensor<int32_t> remote_src_idx_gm;
                    remote_src_idx_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(remote_shmem_base + tiling.src_idx_buffer_offset));
                    AscendC::DataCopyExtParams src_idx_store_params(1, kDispatchInt32Bytes, 0, 0, 0);
                    AscendC::DataCopyPad(remote_src_idx_gm[recv_position], scalar_ub, src_idx_store_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                }

                if (num_topk > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
                    const int32_t local_expert_begin = dst_rank * experts_per_rank;
                    const int32_t local_expert_end = local_expert_begin + experts_per_rank;
                    for (int32_t topk = 0; topk < num_topk; ++topk) {
                        const int32_t expert = static_cast<int32_t>(topk_idx_gm.GetValue(token * num_topk + topk));
                        const bool is_local = expert >= local_expert_begin && expert < local_expert_end;
                        topk_idx_ub.SetValue(topk, is_local ? expert - local_expert_begin : -1);
                        if (tiling.has_topk_weights != 0) {
                            topk_weights_ub.SetValue(
                                topk, is_local ? topk_weights_gm.GetValue(token * num_topk + topk) : 0.0F);
                        }
                    }

                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
                    AscendC::DataCopyExtParams topk_idx_store_params(1, topk_idx_row_bytes, 0, 0, 0);
                    AscendC::DataCopyPad(
                        remote_shmem_gm
                            [tiling.topk_idx_buffer_offset +
                             static_cast<uint64_t>(recv_position) * topk_idx_row_stride],
                        topk_idx_bytes_ub, topk_idx_store_params);

                    if (tiling.has_topk_weights != 0) {
                        AscendC::DataCopyExtParams topk_weights_store_params(1, topk_weights_row_bytes, 0, 0, 0);
                        AscendC::DataCopyPad(
                            remote_shmem_gm
                                [tiling.topk_weights_buffer_offset +
                                 static_cast<uint64_t>(recv_position) * topk_weights_row_stride],
                            topk_weights_bytes_ub, topk_weights_store_params);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
                }

                if (tiling.has_x_scales != 0 && scale_row_bytes > 0) {
                    const auto scale_event_id = send_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> scale_staging_ub =
                        staging_pingpong_ub[send_buffer_idx * staging_buffer_stride];
                    AscendC::DataCopyExtParams scale_copy_params(1, scale_row_bytes, 0, 0, 0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(scale_event_id);
                    AscendC::DataCopyPad(
                        scale_staging_ub, x_scales_gm[static_cast<uint64_t>(token) * scale_row_bytes],
                        scale_copy_params, uint8_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(scale_event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(scale_event_id);
                    AscendC::DataCopyPad(
                        remote_shmem_gm
                            [tiling.x_scales_buffer_offset + static_cast<uint64_t>(recv_position) * scale_row_stride],
                        scale_staging_ub, scale_copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(scale_event_id);
                    send_buffer_idx = (send_buffer_idx + 1) & staging_buffer_mask;
                }
            }

            if (rebuild_dispatch_handle) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopyExtParams send_head_store_params(
                    1, static_cast<uint32_t>(world_size) * kDispatchInt32Bytes, 0, 0, 0);
                AscendC::DataCopyPad(send_head_gm[token * world_size], send_head_row_ub, send_head_store_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        if (!collect_src_idx) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    if (collect_src_idx) {
        // The token loop has grouped this channel's indices by destination.
        // Prefixes assign every source-rank/channel a disjoint compact GM
        // interval; UB segment bases remain 32-byte aligned independently.
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        for (int32_t dst_rank = 0; dst_rank < world_size; ++dst_rank) {
            const int32_t channel_prefix_begin =
                core_idx > 0 ? channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx - 1) : 0;
            const int32_t channel_prefix_end = channel_prefix_gm.GetValue(dst_rank * num_channels + core_idx);
            const int32_t channel_token_count = channel_prefix_end - channel_prefix_begin;
            if (channel_token_count == 0) {
                continue;
            }

            __gm__ uint8_t* remote_shmem_base = static_cast<__gm__ uint8_t*>(
                aclshmem_ptr(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer), dst_rank));
            AscendC::GlobalTensor<int32_t> remote_src_idx_gm;
            remote_src_idx_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(remote_shmem_base + tiling.src_idx_buffer_offset));
            const int32_t local_src_idx_begin = src_idx_rank_base_ub.GetValue(dst_rank);
            const int32_t rank_base = rank > 0 ? rank_prefix_gm.GetValue((rank - 1) * world_size + dst_rank) : 0;
            const int32_t remote_src_idx_begin = rank_base + channel_prefix_begin;
            AscendC::DataCopyExtParams src_idx_store_params(
                1, static_cast<uint32_t>(channel_token_count) * kDispatchInt32Bytes, 0, 0, 0);
            AscendC::DataCopyPad(
                remote_src_idx_gm[remote_src_idx_begin], src_idx_values_ub[local_src_idx_begin], src_idx_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
    }

    aclshmemi_sync_npu_v4<true>(0);

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

    // Materialize each AIV's disjoint receive interval after transport completion.
    // Equal output/SHMEM pointers preserve the zero-copy path without a copy.
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
                output_x_gm[static_cast<uint64_t>(padding_token) * x_row_stride],
                zero_staging_ub, zero_x_store_params);
            if (tiling.has_x_scales != 0 && scale_row_bytes > 0) {
                AscendC::DataCopyPad(
                    output_scales_gm[static_cast<uint64_t>(padding_token) * scale_row_stride],
                    zero_staging_ub, zero_scale_store_params);
            }
            if (num_topk > 0) {
                AscendC::DataCopyPad(
                    output_topk_idx_gm[static_cast<uint64_t>(padding_token) * topk_idx_row_stride],
                    topk_idx_bytes_ub, invalid_topk_store_params);
                if (tiling.has_topk_weights != 0) {
                    AscendC::DataCopyPad(
                        output_weights_gm[static_cast<uint64_t>(padding_token) * topk_weights_row_stride],
                        zero_staging_ub, zero_topk_weights_store_params);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_dispatch_kernel(
    GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR num_tokens_per_rank,
    GM_ADDR is_token_in_rank, GM_ADDR num_tokens_per_expert, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_x_scales, GM_ADDR recv_topk_idx, GM_ADDR recv_topk_weights, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR recv_channel_prefix_matrix, GM_ADDR recv_src_idx, GM_ADDR send_head,
    ascend_deepep::DispatchTilingData tiling, uint64_t ffts_addr)
{
    V1DispatchMteKernelImpl(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, tiling, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DispatchTilingData*>(tiling_host);
    v1_dispatch_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kDispatchUbBytes), stream>>>(
        x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, *td, ffts_addr);
}
