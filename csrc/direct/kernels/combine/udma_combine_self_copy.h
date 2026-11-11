#pragma once

#include "common/platform.h"
#include "common/types.h"
#include "common/udma_schedule.h"
#include "kernel_operator.h"
#include "shmem.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
template <class Resource>
__aicore__ inline void CombineSelfCopy(
    Resource& resource, GM_ADDR x, GM_ADDR topk_weights, GM_ADDR src_idx, GM_ADDR rank_prefix,
    GM_ADDR channel_prefix, GM_ADDR symmetric_buffer, const CombineTilingData& tiling, int32_t core_idx)
{
    constexpr uint32_t kCombineUbBytes = platform::kKernelUbBytes;
    constexpr uint32_t kCombineAlignmentBytes = 32;
    constexpr int32_t kCombineSrcIdxTileElements = 4096;
    constexpr uint32_t communication_ub_offset = kBatchWqes * kWqeBytes;
    static_assert(kCombineSrcIdxTileElements * sizeof(int32_t) % kCombineAlignmentBytes == 0);
    static_assert(communication_ub_offset < kCombineUbBytes);
    const int32_t rank = tiling.rank;
    const int32_t world_size = tiling.world_size;
    const int32_t num_channels = tiling.num_channels;
    const int32_t num_topk = tiling.num_topk;
    const int32_t token_capacity = tiling.token_capacity;
    const uint32_t hidden_bytes = tiling.hidden * tiling.x_element_bytes;
    const uint32_t topk_weights_row_bytes = tiling.num_topk * sizeof(float);
    AscendC::GlobalTensor<uint8_t> x_bytes_gm;
    AscendC::GlobalTensor<float> topk_weights_gm;
    AscendC::GlobalTensor<int32_t> src_idx_gm, rank_prefix_gm, channel_prefix_gm;
    x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    src_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    rank_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    channel_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix));
    if (tiling.has_topk_weights != 0)
        topk_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topk_weights));
    auto staging_ub = resource.template GetBufferByByte<uint8_t>(communication_ub_offset);
    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> int32_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> float_pad_params{false, 0, 0, 0};

    uint32_t staging_row_bytes = hidden_bytes;
    staging_row_bytes = topk_weights_row_bytes > staging_row_bytes ? topk_weights_row_bytes : staging_row_bytes;
    const uint32_t staging_stride = (staging_row_bytes + kCombineAlignmentBytes - 1U) / kCombineAlignmentBytes * kCombineAlignmentBytes;
    // With top-k, hidden uses slot 0 and weights use slot 1. Without top-k,
    // keep hidden on one slot so a full-row remote store completes before the
    // same slot is refilled; peer rotation still distributes traffic.
    const int32_t staging_buffer_num =
        tiling.has_topk_weights != 0 && communication_ub_offset + staging_stride * 2U <= kCombineUbBytes ? 2 : 1;
    const int32_t staging_buffer_mask = staging_buffer_num - 1;
    // The send-only src_idx tile starts after staging. Reduction reuses this
    // UB later, after all sends and the cross-rank barrier have completed.
    const uint32_t staging_bytes = staging_stride * static_cast<uint32_t>(staging_buffer_num);
    const uint32_t src_idx_available_bytes = kCombineUbBytes - communication_ub_offset - staging_bytes;
    const int32_t aligned_index_elements =
        static_cast<int32_t>(src_idx_available_bytes / kCombineAlignmentBytes * (kCombineAlignmentBytes / sizeof(int32_t)));
    const int32_t src_idx_buffer_elements = aligned_index_elements < kCombineSrcIdxTileElements
                                                ? aligned_index_elements : kCombineSrcIdxTileElements;
    const uint32_t src_idx_ub_offset = src_idx_buffer_elements > 0 ? communication_ub_offset + staging_bytes : 0;
    AscendC::LocalTensor<int32_t> src_idx_ub = resource.template GetBufferByByte<int32_t>(src_idx_ub_offset);
    int32_t staging_buffer_idx = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // With spare AIVs, assign every self-copy channel to a non-sending AIV.
    // Slots remain indexed by (returning rank, original token).
    const int32_t first_copy_aiv = tiling.udma_self_copy_first_aiv;
    const int32_t copy_aivs = num_channels - first_copy_aiv;
    const int32_t first_channel = core_idx < first_copy_aiv ? num_channels : core_idx - first_copy_aiv;
    const int32_t channel_stride = first_copy_aiv == 0 ? num_channels : copy_aivs;
    const int32_t source_rank = rank;
    const int32_t source_rank_base =
        source_rank > 0 ? rank_prefix_gm.GetValue((source_rank - 1) * world_size + rank) : 0;
    const int32_t source_rank_end = rank_prefix_gm.GetValue(source_rank * world_size + rank);
    const int32_t source_rank_tokens = source_rank_end - source_rank_base;
    __gm__ uint8_t* remote_shmem_base = static_cast<__gm__ uint8_t*>(
        aclshmem_ptr(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer), source_rank));
    AscendC::GlobalTensor<uint8_t> remote_shmem_gm;
    remote_shmem_gm.SetGlobalBuffer(remote_shmem_base);

    for (int32_t channel = first_channel; channel < num_channels; channel += channel_stride) {
        const int32_t channel_begin = channel_prefix_gm.GetValue(source_rank * num_channels + channel);
        const int32_t channel_end = channel + 1 < num_channels
                                        ? channel_prefix_gm.GetValue(source_rank * num_channels + channel + 1)
                                        : source_rank_tokens;
        for (int32_t source_token_tile_begin = channel_begin; source_token_tile_begin < channel_end;) {
            const int32_t recv_token_tile_begin = source_rank_base + source_token_tile_begin;
            const int32_t aligned_recv_token_tile_begin = recv_token_tile_begin / 8 * 8;
            const int32_t src_idx_leading_elements = recv_token_tile_begin - aligned_recv_token_tile_begin;
            const int32_t remaining = channel_end - source_token_tile_begin;
            const int32_t available = src_idx_buffer_elements - src_idx_leading_elements;
            const int32_t source_token_tile_count =
                src_idx_buffer_elements > 0 ? (available < remaining ? available : remaining) : 1;
            if (src_idx_buffer_elements > 0) {
                const int32_t src_idx_copy_elements = src_idx_leading_elements + source_token_tile_count;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                AscendC::DataCopyExtParams src_idx_copy_params(
                    1, static_cast<uint32_t>(src_idx_copy_elements) * sizeof(int32_t), 0, 0, 0);
                AscendC::DataCopyPad(
                    src_idx_ub, src_idx_gm[aligned_recv_token_tile_begin], src_idx_copy_params, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            }

            for (int32_t source_token_tile_idx = 0; source_token_tile_idx < source_token_tile_count;
                 ++source_token_tile_idx) {
                const int32_t recv_token = recv_token_tile_begin + source_token_tile_idx;
                const int32_t source_token = src_idx_buffer_elements > 0 ?
                                                 src_idx_ub.GetValue(src_idx_leading_elements + source_token_tile_idx) :
                                                 src_idx_gm.GetValue(recv_token);
                const uint64_t remote_slot = static_cast<uint64_t>(rank) * token_capacity + source_token;

                const auto x_event_id = staging_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<uint8_t> x_staging_ub = staging_ub[staging_buffer_idx * staging_stride];
                AscendC::DataCopyExtParams x_copy_params(1, hidden_bytes, 0, 0, 0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(x_event_id);
                AscendC::DataCopyPad(
                    x_staging_ub, x_bytes_gm[static_cast<uint64_t>(recv_token) * hidden_bytes], x_copy_params,
                    uint8_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(x_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(x_event_id);
                AscendC::DataCopyPad(
                    remote_shmem_gm[tiling.x_buffer_offset + remote_slot * tiling.x_row_stride], x_staging_ub,
                    x_copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(x_event_id);
                staging_buffer_idx = (staging_buffer_idx + 1) & staging_buffer_mask;

                if (tiling.has_topk_weights != 0) {
                    const auto weight_event_id = staging_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> weight_staging_ub = staging_ub[staging_buffer_idx * staging_stride];
                    AscendC::LocalTensor<float> weight_staging_float_ub = weight_staging_ub.ReinterpretCast<float>();
                    AscendC::DataCopyExtParams weight_copy_params(1, topk_weights_row_bytes, 0, 0, 0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(weight_event_id);
                    AscendC::DataCopyPad(
                        weight_staging_float_ub, topk_weights_gm[recv_token * num_topk], weight_copy_params,
                        float_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(weight_event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(weight_event_id);
                    AscendC::DataCopyPad(
                        remote_shmem_gm
                            [tiling.topk_weights_buffer_offset + remote_slot * tiling.topk_weights_row_stride],
                        weight_staging_ub, weight_copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(weight_event_id);
                    staging_buffer_idx = (staging_buffer_idx + 1) & staging_buffer_mask;
                }
            }
            source_token_tile_begin += source_token_tile_count;
        }
    }

    // Drain local writes before sending or entering the completion path.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // Also order a subsequent sender's scalar stores to the reused UB.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}
#endif
} // namespace ascend_deepep::udma
