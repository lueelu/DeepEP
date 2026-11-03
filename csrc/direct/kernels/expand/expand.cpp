#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "simt_api/device_sync_functions.h"

#include "common/launch.h"
#include "common/types.h"
#include "kernels/common/v2_handle.h"

using namespace AscendC;

namespace {

constexpr uint32_t kExpandUbBytes = static_cast<uint32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr uint32_t kExpandAlignmentBytes = 32;
constexpr uint32_t kExpandMaxDataCopyBytes = 65504;
constexpr uint32_t kExpandMaxLocalExperts = 1024;
constexpr uint32_t kExpandMaxTopk = 128;
constexpr uint32_t kExpandWeightScatterThreads = 128;

constexpr uint32_t ExpandAlignUp32(uint32_t value)
{
    return (value + kExpandAlignmentBytes - 1U) / kExpandAlignmentBytes * kExpandAlignmentBytes;
}

constexpr uint32_t kExpandCountOffset = 0;
constexpr uint32_t kExpandCountBytes = kExpandMaxLocalExperts * sizeof(int32_t);
constexpr uint32_t kExpandTailOffset = ExpandAlignUp32(kExpandCountOffset + kExpandCountBytes);
constexpr uint32_t kExpandTailBytes = kExpandCountBytes;
constexpr uint32_t kExpandTemporaryOffset = ExpandAlignUp32(kExpandTailOffset + kExpandTailBytes);
constexpr uint32_t kExpandTemporaryBytes = kExpandCountBytes;
constexpr uint32_t kExpandExpertOffsetsOffset = ExpandAlignUp32(kExpandTemporaryOffset + kExpandTemporaryBytes);
constexpr uint32_t kExpandExpertOffsetsBytes = ExpandAlignUp32((kExpandMaxLocalExperts + 1U) * sizeof(int32_t));
constexpr uint32_t kExpandTopkOffset = ExpandAlignUp32(kExpandExpertOffsetsOffset + kExpandExpertOffsetsBytes);
constexpr uint32_t kExpandTopkBytes = ExpandAlignUp32(kExpandMaxTopk * sizeof(ascend_deepep::topk_idx_t));
constexpr uint32_t kExpandMappingOffset = ExpandAlignUp32(kExpandTopkOffset + kExpandTopkBytes);
constexpr uint32_t kExpandMappingBytes = ExpandAlignUp32(kExpandMaxTopk * sizeof(int32_t));
constexpr uint32_t kExpandWeightOffset = ExpandAlignUp32(kExpandMappingOffset + kExpandMappingBytes);
constexpr uint32_t kExpandWeightBytes = ExpandAlignUp32(kExpandMaxTopk * sizeof(float));
constexpr uint32_t kExpandStageOffset = ExpandAlignUp32(kExpandWeightOffset + kExpandWeightBytes);
constexpr uint32_t kExpandStageBufferBytes =
    (kExpandUbBytes - kExpandStageOffset) / 2U / kExpandAlignmentBytes * kExpandAlignmentBytes;
constexpr uint32_t kExpandNormalTopkTileBytes = kExpandCountBytes;
constexpr uint32_t kExpandNormalMappingTileOffset = kExpandTemporaryOffset;
constexpr uint32_t kExpandNormalMappingTileBytes = kExpandStageOffset - kExpandNormalMappingTileOffset;

static_assert(kExpandStageBufferBytes > 0);
static_assert(kExpandStageOffset + 2U * kExpandStageBufferBytes <= kExpandUbBytes);
static_assert(2U * kExpandStageBufferBytes >= kExpandTopkBytes);
static_assert(kExpandMaxDataCopyBytes >= kExpandTopkBytes);
static_assert(kExpandStageOffset >= kExpandMappingBytes);
static_assert(kExpandNormalTopkTileBytes >= kExpandTopkBytes);
static_assert(kExpandNormalMappingTileBytes >= kExpandMappingBytes);

__aicore__ inline int32_t ExpandMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

// Mapping slots are globally unique. Scatter all route weights once per AIV
// instead of issuing one 4-byte MTE3 transfer for every valid route.
__simt_vf__ __launch_bounds__(kExpandWeightScatterThreads) inline void ExpandWeightScatterSimtVf(
    __gm__ float* topk_weights, __gm__ int32_t* recv_to_expanded_idx, __gm__ float* expanded_topk_weights,
    int32_t token_begin, int32_t token_end, int32_t num_topk)
{
    const int64_t route_begin = static_cast<int64_t>(token_begin) * num_topk;
    const int64_t route_end = static_cast<int64_t>(token_end) * num_topk;
    const int64_t thread_index = static_cast<int64_t>(threadIdx.x);
    for (int64_t route = route_begin + thread_index; route < route_end;
         route += static_cast<int64_t>(kExpandWeightScatterThreads)) {
        const int32_t expanded_slot = recv_to_expanded_idx[route];
        if (expanded_slot >= 0) {
            expanded_topk_weights[expanded_slot] = topk_weights[route];
        }
    }
    asc_threadfence();
}

// One thread owns an expert column. Core order is unchanged, so the resulting
// expert-major slots retain the original deterministic token/route order.
__simt_vf__ __launch_bounds__(128) inline void ExpandCountTotalsSimtVf(
    __ubuf__ int32_t* counts, __ubuf__ int32_t* totals, int32_t cores, int32_t expert_begin, int32_t experts,
    int32_t stride)
{
    for (int32_t expert = static_cast<int32_t>(threadIdx.x); expert < experts; expert += 128) {
        int32_t total = 0;
        for (int32_t core = 0; core < cores; ++core) {
            total += counts[core * stride + expert];
        }
        totals[expert_begin + expert] = total;
    }
}

__simt_vf__ __launch_bounds__(128) inline void ExpandCoreOffsetsSimtVf(
    __ubuf__ int32_t* counts, __ubuf__ int32_t* starts, __ubuf__ int32_t* offsets, int32_t cores, int32_t expert_begin,
    int32_t experts, int32_t stride)
{
    for (int32_t expert = static_cast<int32_t>(threadIdx.x); expert < experts; expert += 128) {
        int32_t offset = offsets[expert_begin + expert];
        for (int32_t core = 0; core < cores; ++core) {
            starts[core * stride + expert] = offset;
            offset += counts[core * stride + expert];
        }
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void expand_kernel(
    GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_idx, GM_ADDR topk_weights, GM_ADDR expanded_x, GM_ADDR expanded_x_scales,
    GM_ADDR expanded_topk_weights, GM_ADDR expert_offsets, GM_ADDR actual_expert_counts, GM_ADDR recv_to_expanded_idx,
    GM_ADDR scratch_buffer, ascend_deepep::ExpandTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();
    const int32_t num_recv_tokens = static_cast<int32_t>(tiling.num_recv_tokens);
    const int32_t hidden = static_cast<int32_t>(tiling.hidden);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t num_local_experts = static_cast<int32_t>(tiling.num_local_experts);
    const int32_t token_grain = tiling.v2_src_metadata ? 8 : 1;
    const int32_t tokens_per_core =
        ((num_recv_tokens + token_grain * core_num - 1) / (token_grain * core_num)) * token_grain;
    const int32_t token_begin = ExpandMin(core_idx * tokens_per_core, num_recv_tokens);
    const int32_t token_end = ExpandMin(token_begin + tokens_per_core, num_recv_tokens);

    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<int32_t> count_ub = resource.template GetBufferByByte<int32_t>(kExpandCountOffset);
    AscendC::LocalTensor<int32_t> tail_ub = resource.template GetBufferByByte<int32_t>(kExpandTailOffset);
    AscendC::LocalTensor<int32_t> expert_offsets_ub =
        resource.template GetBufferByByte<int32_t>(kExpandExpertOffsetsOffset);
    AscendC::LocalTensor<uint8_t> normal_topk_tile_bytes_ub = resource.template GetBufferByByte<uint8_t>(0);
    AscendC::LocalTensor<ascend_deepep::topk_idx_t> normal_topk_tile_ub =
        resource.template GetBufferByByte<ascend_deepep::topk_idx_t>(0);
    AscendC::LocalTensor<int32_t> normal_mapping_tile_ub =
        resource.template GetBufferByByte<int32_t>(kExpandNormalMappingTileOffset);
    AscendC::LocalTensor<int32_t> cached_mapping_tile_ub = resource.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<uint8_t> stage_ub = resource.template GetBufferByByte<uint8_t>(kExpandStageOffset);

    AscendC::GlobalTensor<uint8_t> x_bytes_gm;
    AscendC::GlobalTensor<uint8_t> x_scales_bytes_gm;
    AscendC::GlobalTensor<uint8_t> topk_bytes_gm;
    AscendC::GlobalTensor<uint8_t> expanded_x_bytes_gm;
    AscendC::GlobalTensor<uint8_t> expanded_x_scales_bytes_gm;
    AscendC::GlobalTensor<int32_t> expert_offsets_gm;
    AscendC::GlobalTensor<int32_t> actual_expert_counts_gm;
    AscendC::GlobalTensor<int32_t> mapping_gm;
    AscendC::GlobalTensor<int32_t> scratch_gm;
    x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    x_scales_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x_scales));
    topk_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(topk_idx));
    expanded_x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(expanded_x));
    expanded_x_scales_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(expanded_x_scales));
    expert_offsets_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_offsets));
    actual_expert_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actual_expert_counts));
    mapping_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_to_expanded_idx));
    scratch_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_buffer));

    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> int32_pad_params{false, 0, 0, 0};

    const uint32_t topk_row_bytes = static_cast<uint32_t>(num_topk) * sizeof(ascend_deepep::topk_idx_t);
    const uint32_t mapping_row_bytes = static_cast<uint32_t>(num_topk) * sizeof(int32_t);
    const uint32_t expert_bytes = static_cast<uint32_t>(num_local_experts) * sizeof(int32_t);

    if (tiling.cached_mode == 0) {
        // Each AIV counts its contiguous token range independently. The two
        // scratch matrices hold per-core counts and per-core starting slots.
        AscendC::Duplicate(count_ub, static_cast<int32_t>(0), num_local_experts);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        const uint32_t count_tile_capacity_bytes = ExpandMin(
            static_cast<int32_t>(2U * kExpandStageBufferBytes), static_cast<int32_t>(kExpandMaxDataCopyBytes));
        const int32_t count_tile_tokens = static_cast<int32_t>(count_tile_capacity_bytes / topk_row_bytes);
        AscendC::LocalTensor<uint8_t> count_topk_tile_bytes_ub = stage_ub;
        AscendC::LocalTensor<ascend_deepep::topk_idx_t> count_topk_tile_ub =
            stage_ub.ReinterpretCast<ascend_deepep::topk_idx_t>();
        for (int32_t tile_begin = token_begin; tile_begin < token_end; tile_begin += count_tile_tokens) {
            const int32_t tile_end = ExpandMin(tile_begin + count_tile_tokens, token_end);
            const uint32_t tile_bytes = static_cast<uint32_t>(tile_end - tile_begin) * topk_row_bytes;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams topk_copy_params(1, tile_bytes, 0, 0, 0);
            AscendC::DataCopyPad(
                count_topk_tile_bytes_ub, topk_bytes_gm[static_cast<uint64_t>(tile_begin) * topk_row_bytes],
                topk_copy_params, uint8_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            for (int32_t token = tile_begin; token < tile_end; ++token) {
                const int32_t tile_route_begin = (token - tile_begin) * num_topk;
                for (int32_t k = 0; k < num_topk; ++k) {
                    const int32_t local_expert =
                        static_cast<int32_t>(count_topk_tile_ub.GetValue(tile_route_begin + k));
                    if (local_expert < 0 || local_expert >= num_local_experts) {
                        continue;
                    }
                    count_ub.SetValue(local_expert, count_ub.GetValue(local_expert) + 1);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams count_store_params(1, expert_bytes, 0, 0, 0);
        AscendC::DataCopyPad(scratch_gm[core_idx * num_local_experts], count_ub, count_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        AscendC::SyncAll<true>();

        if (core_idx == 0) {
            // Payload staging is not live yet. Reuse its two slots for a
            // bounded count matrix and its per-core exclusive prefixes.
            // Pad each UB row to 32 bytes, including a short expert tile.
            auto prefix_counts_ub = stage_ub.ReinterpretCast<int32_t>();
            auto prefix_starts_ub = stage_ub[kExpandStageBufferBytes].ReinterpretCast<int32_t>();
            const int32_t prefix_bytes =
                ExpandMin(static_cast<int32_t>(kExpandStageBufferBytes), static_cast<int32_t>(kExpandMaxDataCopyBytes));
            const int32_t experts_per_tile = prefix_bytes / (core_num * static_cast<int32_t>(sizeof(int32_t))) / 8 * 8;
            for (int32_t expert_begin = 0; expert_begin < num_local_experts; expert_begin += experts_per_tile) {
                const int32_t experts = ExpandMin(experts_per_tile, num_local_experts - expert_begin);
                const int32_t stride = (experts + 7) / 8 * 8;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::DataCopyExtParams partial_copy_params(
                    core_num, experts * uint32_t(sizeof(int32_t)),
                    (num_local_experts - experts) * uint32_t(sizeof(int32_t)), 0, 0);
                AscendC::DataCopyPad(prefix_counts_ub, scratch_gm[expert_begin], partial_copy_params, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                asc_vf_call<ExpandCountTotalsSimtVf>(
                    dim3(128), reinterpret_cast<__ubuf__ int32_t*>(prefix_counts_ub.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int32_t*>(count_ub.GetPhyAddr()), core_num, expert_begin, experts,
                    stride);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            }

            int32_t expanded_offset = 0;
            for (int32_t expert = 0; expert < num_local_experts; ++expert) {
                const int32_t actual_count = count_ub.GetValue(expert);
                expert_offsets_ub.SetValue(expert, expanded_offset);
                const int32_t aligned_count = (actual_count + static_cast<int32_t>(tiling.expert_alignment) - 1) /
                                              static_cast<int32_t>(tiling.expert_alignment) *
                                              static_cast<int32_t>(tiling.expert_alignment);
                expanded_offset += aligned_count;
            }
            expert_offsets_ub.SetValue(num_local_experts, expanded_offset);

            if (tiling.v2_expert_prefix) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                asc_vf_call<ascend_deepep::V2HandleExpandedCountsVf>(dim3(128),
                    reinterpret_cast<__ubuf__ int32_t*>(expert_offsets_ub.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int32_t*>(count_ub.GetPhyAddr()),
                    reinterpret_cast<__gm__ int32_t*>(tiling.v2_expert_prefix), uint32_t(num_local_experts));
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams actual_count_store_params(1, expert_bytes, 0, 0, 0);
            AscendC::DataCopyPad(actual_expert_counts_gm, count_ub, actual_count_store_params);
            AscendC::DataCopyExtParams expert_offset_store_params(
                1, static_cast<uint32_t>(num_local_experts + 1) * sizeof(int32_t), 0, 0, 0);
            AscendC::DataCopyPad(expert_offsets_gm, expert_offsets_ub, expert_offset_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

            for (int32_t expert_begin = 0; expert_begin < num_local_experts; expert_begin += experts_per_tile) {
                const int32_t experts = ExpandMin(experts_per_tile, num_local_experts - expert_begin);
                const int32_t stride = (experts + 7) / 8 * 8;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::DataCopyExtParams partial_copy_params(
                    core_num, experts * uint32_t(sizeof(int32_t)),
                    (num_local_experts - experts) * uint32_t(sizeof(int32_t)), 0, 0);
                AscendC::DataCopyPad(prefix_counts_ub, scratch_gm[expert_begin], partial_copy_params, int32_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                asc_vf_call<ExpandCoreOffsetsSimtVf>(
                    dim3(128), reinterpret_cast<__ubuf__ int32_t*>(prefix_counts_ub.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int32_t*>(prefix_starts_ub.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int32_t*>(expert_offsets_ub.GetPhyAddr()), core_num, expert_begin,
                    experts, stride);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::DataCopyExtParams prefix_store_params(
                    core_num, experts * uint32_t(sizeof(int32_t)), 0,
                    (num_local_experts - experts) * uint32_t(sizeof(int32_t)), 0);
                AscendC::DataCopyPad(
                    scratch_gm[core_num * num_local_experts + expert_begin], prefix_starts_ub, prefix_store_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
            }
        }

        AscendC::SyncAll<true>();

        AscendC::DataCopyExtParams tail_copy_params(1, expert_bytes, 0, 0, 0);
        AscendC::DataCopyPad(
            tail_ub, scratch_gm[core_num * num_local_experts + core_idx * num_local_experts], tail_copy_params,
            int32_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t stage_buffer_idx = 0;
    const int32_t normal_topk_tile_tokens =
        static_cast<int32_t>(kExpandNormalTopkTileBytes) / static_cast<int32_t>(topk_row_bytes);
    const int32_t normal_mapping_tile_tokens =
        static_cast<int32_t>(kExpandNormalMappingTileBytes) / static_cast<int32_t>(mapping_row_bytes);
    const int32_t cached_mapping_tile_tokens =
        static_cast<int32_t>(kExpandStageOffset) / static_cast<int32_t>(mapping_row_bytes);
    const int32_t mapping_tile_tokens = tiling.cached_mode == 0 ?
                                            ExpandMin(normal_topk_tile_tokens, normal_mapping_tile_tokens) :
                                            cached_mapping_tile_tokens;
    const uint32_t x_row_bytes = static_cast<uint32_t>(hidden) * tiling.x_element_bytes;
    const uint32_t scale_row_bytes = tiling.num_scales * tiling.scale_element_bytes;
    // Batch compact rows only when every row starts at a 32-byte UB address.
    // Oversized or unaligned rows keep the chunked single-row path below.
    const bool batch_x_rows = x_row_bytes <= kExpandStageBufferBytes && x_row_bytes % kExpandAlignmentBytes == 0;
    const int32_t x_rows_per_stage = batch_x_rows ? static_cast<int32_t>(kExpandStageBufferBytes / x_row_bytes) : 1;
    // Normal mode creates the mapping; cached mode reuses it. Payload rows are
    // loaded once and scattered to all local expert slots selected by a token.
    for (int32_t tile_begin = token_begin; tile_begin < token_end; tile_begin += mapping_tile_tokens) {
        const int32_t tile_end = ExpandMin(tile_begin + mapping_tile_tokens, token_end);
        const int32_t tile_token_count = tile_end - tile_begin;
        const int32_t tile_route_count = tile_token_count * num_topk;
        const uint32_t mapping_tile_bytes = static_cast<uint32_t>(tile_route_count) * sizeof(int32_t);
        AscendC::LocalTensor<int32_t> active_mapping_tile_ub = normal_mapping_tile_ub;
        if (tiling.cached_mode == 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            const uint32_t topk_tile_bytes = static_cast<uint32_t>(tile_token_count) * topk_row_bytes;
            AscendC::DataCopyExtParams topk_copy_params(1, topk_tile_bytes, 0, 0, 0);
            AscendC::DataCopyPad(
                normal_topk_tile_bytes_ub, topk_bytes_gm[static_cast<uint64_t>(tile_begin) * topk_row_bytes],
                topk_copy_params, uint8_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
            AscendC::Duplicate(normal_mapping_tile_ub, static_cast<int32_t>(-1), tile_route_count);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);

            for (int32_t token = tile_begin; token < tile_end; ++token) {
                const int32_t tile_route_begin = (token - tile_begin) * num_topk;
                int32_t single_slot = -1;
                int32_t contributions = 0;
                for (int32_t k = 0; k < num_topk; ++k) {
                    const int32_t local_expert =
                        static_cast<int32_t>(normal_topk_tile_ub.GetValue(tile_route_begin + k));
                    if (local_expert < 0 || local_expert >= num_local_experts) {
                        continue;
                    }
                    const int32_t expanded_slot = tail_ub.GetValue(local_expert);
                    tail_ub.SetValue(local_expert, expanded_slot + 1);
                    normal_mapping_tile_ub.SetValue(tile_route_begin + k, expanded_slot);
                    single_slot = expanded_slot;
                    ++contributions;
                }
                if (tiling.single_source_slots) {
                    // This consumed prefix is behind all unread top-k rows.
                    // Both supported top-k types are at least int32 wide.
                    normal_topk_tile_bytes_ub.ReinterpretCast<int32_t>().SetValue(
                        token - tile_begin, contributions == 1 ? single_slot : -1);
                }
            }

            if (tiling.v2_src_metadata) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
                asc_vf_call<ascend_deepep::V2HandleExpandMappingVf>(dim3(128),
                    reinterpret_cast<__ubuf__ int32_t*>(normal_mapping_tile_ub.GetPhyAddr()),
                    reinterpret_cast<__gm__ int32_t*>(tiling.v2_src_metadata),
                    uint32_t(tile_begin), uint32_t(tile_token_count), uint32_t(num_topk));
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            if (tiling.single_source_slots) {
                AscendC::GlobalTensor<int32_t> single_gm;
                single_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling.single_source_slots));
                AscendC::DataCopyPad(single_gm[tile_begin], normal_topk_tile_bytes_ub.ReinterpretCast<int32_t>(),
                    AscendC::DataCopyExtParams(1, uint32_t(tile_token_count) * sizeof(int32_t), 0, 0, 0));
            }
            AscendC::DataCopyExtParams mapping_store_params(1, mapping_tile_bytes, 0, 0, 0);
            AscendC::DataCopyPad(
                mapping_gm[static_cast<uint64_t>(tile_begin) * num_topk], normal_mapping_tile_ub, mapping_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID5);
            AscendC::DataCopyExtParams mapping_tile_copy_params(1, mapping_tile_bytes, 0, 0, 0);
            AscendC::DataCopyPad(
                cached_mapping_tile_ub, mapping_gm[static_cast<uint64_t>(tile_begin) * num_topk],
                mapping_tile_copy_params, int32_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID5);
            active_mapping_tile_ub = cached_mapping_tile_ub;
        }

        for (int32_t payload_begin = tile_begin; payload_begin < tile_end; payload_begin += x_rows_per_stage) {
            const int32_t payload_end = ExpandMin(payload_begin + x_rows_per_stage, tile_end);
            if (batch_x_rows) {
                const uint32_t payload_bytes = static_cast<uint32_t>(payload_end - payload_begin) * x_row_bytes;
                const auto event_id = stage_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<uint8_t> input_stage_ub = stage_ub[stage_buffer_idx * kExpandStageBufferBytes];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                AscendC::DataCopyExtParams payload_copy_params(1, payload_bytes, 0, 0, 0);
                AscendC::DataCopyPad(
                    input_stage_ub, x_bytes_gm[static_cast<uint64_t>(payload_begin) * x_row_bytes], payload_copy_params,
                    uint8_pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

                AscendC::DataCopyExtParams row_copy_params(1, x_row_bytes, 0, 0, 0);
                for (int32_t token = payload_begin; token < payload_end; ++token) {
                    AscendC::LocalTensor<int32_t> payload_mapping_ub =
                        active_mapping_tile_ub[(token - tile_begin) * num_topk];
                    AscendC::LocalTensor<uint8_t> input_row_ub = input_stage_ub[(token - payload_begin) * x_row_bytes];
                    for (int32_t k = 0; k < num_topk; ++k) {
                        const int32_t expanded_slot = payload_mapping_ub.GetValue(k);
                        if (expanded_slot < 0) {
                            continue;
                        }
                        const uint64_t output_byte_offset = static_cast<uint64_t>(expanded_slot) * x_row_bytes;
                        AscendC::DataCopyPad(expanded_x_bytes_gm[output_byte_offset], input_row_ub, row_copy_params);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                stage_buffer_idx = (stage_buffer_idx + 1) & 1;
            } else {
                const int32_t token = payload_begin;
                AscendC::LocalTensor<int32_t> payload_mapping_ub =
                    active_mapping_tile_ub[(token - tile_begin) * num_topk];
                const uint64_t input_x_row_offset = static_cast<uint64_t>(token) * x_row_bytes;
                for (uint32_t byte_begin = 0; byte_begin < x_row_bytes; byte_begin += kExpandStageBufferBytes) {
                    const uint32_t copy_bytes = static_cast<uint32_t>(ExpandMin(
                        static_cast<int32_t>(kExpandStageBufferBytes), static_cast<int32_t>(x_row_bytes - byte_begin)));
                    const auto event_id = stage_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> input_stage_ub = stage_ub[stage_buffer_idx * kExpandStageBufferBytes];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
                    AscendC::DataCopyPad(
                        input_stage_ub, x_bytes_gm[input_x_row_offset + byte_begin], copy_params, uint8_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                    for (int32_t k = 0; k < num_topk; ++k) {
                        const int32_t expanded_slot = payload_mapping_ub.GetValue(k);
                        if (expanded_slot < 0) {
                            continue;
                        }
                        const uint64_t output_byte_offset =
                            static_cast<uint64_t>(expanded_slot) * x_row_bytes + byte_begin;
                        AscendC::DataCopyPad(expanded_x_bytes_gm[output_byte_offset], input_stage_ub, copy_params);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    stage_buffer_idx = (stage_buffer_idx + 1) & 1;
                }
            }

            if (tiling.has_x_scales != 0) {
                for (int32_t token = payload_begin; token < payload_end; ++token) {
                    AscendC::LocalTensor<int32_t> payload_mapping_ub =
                        active_mapping_tile_ub[(token - tile_begin) * num_topk];
                    const uint64_t input_scale_row_offset = static_cast<uint64_t>(token) * scale_row_bytes;
                    for (uint32_t byte_begin = 0; byte_begin < scale_row_bytes; byte_begin += kExpandStageBufferBytes) {
                        const uint32_t copy_bytes = static_cast<uint32_t>(ExpandMin(
                            static_cast<int32_t>(kExpandStageBufferBytes),
                            static_cast<int32_t>(scale_row_bytes - byte_begin)));
                        const auto event_id = stage_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                        AscendC::LocalTensor<uint8_t> input_stage_ub =
                            stage_ub[stage_buffer_idx * kExpandStageBufferBytes];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
                        AscendC::DataCopyPad(
                            input_stage_ub, x_scales_bytes_gm[input_scale_row_offset + byte_begin], copy_params,
                            uint8_pad_params);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        for (int32_t k = 0; k < num_topk; ++k) {
                            const int32_t expanded_slot = payload_mapping_ub.GetValue(k);
                            if (expanded_slot < 0) {
                                continue;
                            }
                            const uint64_t output_byte_offset =
                                static_cast<uint64_t>(expanded_slot) * scale_row_bytes + byte_begin;
                            AscendC::DataCopyPad(
                                expanded_x_scales_bytes_gm[output_byte_offset], input_stage_ub, copy_params);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        stage_buffer_idx = (stage_buffer_idx + 1) & 1;
                    }
                }
            }
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    if (tiling.has_topk_weights != 0) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        asc_vf_call<ExpandWeightScatterSimtVf>(
            dim3(kExpandWeightScatterThreads), reinterpret_cast<__gm__ float*>(topk_weights),
            reinterpret_cast<__gm__ int32_t*>(recv_to_expanded_idx),
            reinterpret_cast<__gm__ float*>(expanded_topk_weights), token_begin, token_end, num_topk);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void expand_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* expanded_x, uint8_t* expanded_x_scales, uint8_t* expanded_topk_weights, uint8_t* expert_offsets,
    uint8_t* actual_expert_counts, uint8_t* recv_to_expanded_idx, uint8_t* scratch_buffer, uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::ExpandTilingData*>(tiling_host);
    expand_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kExpandUbBytes), stream>>>(
        x, x_scales, topk_idx, topk_weights, expanded_x, expanded_x_scales, expanded_topk_weights, expert_offsets,
        actual_expert_counts, recv_to_expanded_idx, scratch_buffer, *td, ffts_addr);
}
