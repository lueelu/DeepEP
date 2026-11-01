#pragma once
#include <cstdint>

#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "simt_api/device_sync_functions.h"

#include "ascend_deepep/kernels/arch.h"
#include "ascend_deepep/kernels/launch.h"
#include "ascend_deepep/kernels/types.h"

using namespace AscendC;

namespace ascend_deepep::aggregate {

constexpr uint32_t kAggregateUbBytes = static_cast<uint32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr uint32_t kAggregateAlignmentBytes = 32;
constexpr uint32_t kAggregateMaxTopk = 128;
constexpr uint32_t kAggregateDtypeBfloat16 = 0;
constexpr uint32_t kAggregateDtypeFloat16 = 1;

constexpr uint32_t AggregateAlignUp32(uint32_t value)
{
    return (value + kAggregateAlignmentBytes - 1U) / kAggregateAlignmentBytes * kAggregateAlignmentBytes;
}

template <uint32_t ReservedUbBytes = 0>
struct AggregateUbLayout {
    static_assert(ReservedUbBytes % 32 == 0 && ReservedUbBytes + 4608 < kAggregateUbBytes);
    static constexpr uint32_t kAggregateMappingOffset = ReservedUbBytes;
    static constexpr uint32_t kAggregateMappingBytes = 4096;
    static constexpr uint32_t kAggregateRouteWeightOffset = AggregateAlignUp32(kAggregateMappingOffset + kAggregateMappingBytes);
    static constexpr uint32_t kAggregateRouteWeightBytes = AggregateAlignUp32(kAggregateMaxTopk * sizeof(float));
    static constexpr uint32_t kAggregateComputeOffset =
        AggregateAlignUp32(kAggregateRouteWeightOffset + kAggregateRouteWeightBytes);
    // Two 16-bit input slots, one FP32 temporary, one FP32 accumulator, and one
    // 16-bit output slot consume 14 bytes per element.
    static constexpr uint32_t kAggregateChunkElements = (kAggregateUbBytes - kAggregateComputeOffset) / (14U * 64U) * 64U;
    static constexpr uint32_t kAggregateInputBufferBytes = kAggregateChunkElements * sizeof(bfloat16_t);
    static constexpr uint32_t kAggregateInputOffset = kAggregateComputeOffset;
    static constexpr uint32_t kAggregateTemporaryOffset = kAggregateInputOffset + 2U * kAggregateInputBufferBytes;
    static constexpr uint32_t kAggregateFloatBufferBytes = kAggregateChunkElements * sizeof(float);
    static constexpr uint32_t kAggregateAccumulatorOffset = kAggregateTemporaryOffset + kAggregateFloatBufferBytes;
    static constexpr uint32_t kAggregateOutputOffset = kAggregateAccumulatorOffset + kAggregateFloatBufferBytes;

    static_assert(kAggregateChunkElements > 0);
    static_assert(kAggregateChunkElements % 64U == 0);
    static_assert(kAggregateOutputOffset + kAggregateInputBufferBytes <= kAggregateUbBytes);
    static_assert(kAggregateMappingBytes >= kAggregateMaxTopk * sizeof(int32_t));

};

// Each (token, route) output belongs to one thread. These values are returned
// independently of x; they are not the optional weights used in x reduction.
__simt_vf__ __launch_bounds__(128) inline void AggregateValuesSimtVf(
    __gm__ int32_t* mapping, __gm__ float* values, __gm__ float* output, int32_t token_begin, int32_t token_end,
    int32_t num_topk)
{
    const int64_t begin = static_cast<int64_t>(token_begin) * num_topk;
    const int64_t end = static_cast<int64_t>(token_end) * num_topk;
    for (int64_t route = begin + static_cast<int64_t>(threadIdx.x); route < end; route += 128) {
        const int32_t slot = mapping[route];
        output[route] = slot < 0 ? 0.0F : values[slot];
    }
    asc_threadfence();
}

__aicore__ inline int32_t AggregateMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline int32_t AggregateAlignUpElements(int32_t value, int32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T, uint32_t ReservedUbBytes = 0, class Resource>
__aicore__ inline void AggregateRange(
    Resource& resource, GM_ADDR x, GM_ADDR topk_weights, GM_ADDR expanded_values, GM_ADDR recv_to_expanded_idx, GM_ADDR aggregated_x,
    GM_ADDR aggregated_values, const ascend_deepep::AggregateTilingData& tiling, int32_t token_begin, int32_t token_end)
{
    using Ub = AggregateUbLayout<ReservedUbBytes>;
    const int32_t num_recv_tokens = static_cast<int32_t>(tiling.num_recv_tokens);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t hidden = static_cast<int32_t>(tiling.hidden);

    AscendC::LocalTensor<int32_t> mapping_tile_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(Ub::kAggregateMappingOffset);
    AscendC::LocalTensor<float> route_weight_ub =
        resource.ubBuf.template GetBufferByByte<float>(Ub::kAggregateRouteWeightOffset);
    AscendC::LocalTensor<uint8_t> input_bytes_ub =
        resource.ubBuf.template GetBufferByByte<uint8_t>(Ub::kAggregateInputOffset);
    AscendC::LocalTensor<float> temporary_ub =
        resource.ubBuf.template GetBufferByByte<float>(Ub::kAggregateTemporaryOffset);
    AscendC::LocalTensor<float> accumulator_ub =
        resource.ubBuf.template GetBufferByByte<float>(Ub::kAggregateAccumulatorOffset);
    AscendC::LocalTensor<uint8_t> output_bytes_ub =
        resource.ubBuf.template GetBufferByByte<uint8_t>(Ub::kAggregateOutputOffset);
    AscendC::LocalTensor<T> output_ub = output_bytes_ub.ReinterpretCast<T>();

    AscendC::GlobalTensor<uint8_t> x_bytes_gm;
    AscendC::GlobalTensor<float> topk_weights_gm;
    AscendC::GlobalTensor<int32_t> mapping_gm;
    AscendC::GlobalTensor<uint8_t> aggregated_x_bytes_gm;
    x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    topk_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topk_weights));
    mapping_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_to_expanded_idx));
    aggregated_x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(aggregated_x));

    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> int32_pad_params{false, 0, 0, 0};
    const uint32_t mapping_row_bytes = static_cast<uint32_t>(num_topk) * sizeof(int32_t);
    const int32_t mapping_tile_tokens = static_cast<int32_t>(Ub::kAggregateMappingBytes / mapping_row_bytes);
    const uint32_t x_row_bytes = static_cast<uint32_t>(hidden) * tiling.x_element_bytes;

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);

    AscendC::GlobalTensor<int32_t> single_slots_gm;
    int32_t self_begin = 0, self_end = 0;
    if (tiling.single_source_slots) {
        single_slots_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling.single_source_slots));
        AscendC::GlobalTensor<int32_t> prefix_gm;
        prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling.rank_prefix_matrix));
        self_begin = tiling.rank ? prefix_gm.GetValue((tiling.rank - 1) * tiling.world_size + tiling.rank) : 0;
        self_end = prefix_gm.GetValue(tiling.rank * tiling.world_size + tiling.rank);
    }
    int32_t input_buffer_idx = 0;
    // A token row belongs to exactly one AIV, so all of its local expert
    // contributors can be reduced without atomics or inter-core barriers.
    for (int32_t tile_begin = token_begin; tile_begin < token_end; tile_begin += mapping_tile_tokens) {
        const int32_t tile_end = AggregateMin(tile_begin + mapping_tile_tokens, token_end);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::DataCopyExtParams mapping_copy_params(
            1, static_cast<uint32_t>(tile_end - tile_begin) * mapping_row_bytes, 0, 0, 0);
        AscendC::DataCopyPad(
            mapping_tile_ub, mapping_gm[static_cast<uint64_t>(tile_begin) * num_topk], mapping_copy_params,
            int32_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);

        for (int32_t token = tile_begin; token < tile_end; ++token) {
            if (tiling.single_source_slots && (token < self_begin || token >= self_end)) {
                const int32_t slot = single_slots_gm.GetValue(token);
                if (slot >= 0) {
                    if (static_cast<uint32_t>(slot) >= tiling.num_expanded_tokens) trap();
                    // Only private UDMA combine may leave this row unwritten.
                    // Its sender reads the original expert output instead.
                    continue;
                }
            }
            auto mapping_ub = mapping_tile_ub[(token - tile_begin) * num_topk];
            int32_t valid_routes = 0;
            int32_t single_slot = -1;
            for (int32_t k = 0; k < num_topk; ++k) {
                const int32_t slot = mapping_ub.GetValue(k);
                if (slot >= 0) {
                    ++valid_routes;
                    single_slot = slot;
                }
            }
            const bool single_unweighted = valid_routes == 1 && tiling.has_topk_weights == 0;

            if (tiling.has_topk_weights != 0) {
                for (int32_t k = 0; k < num_topk; ++k) {
                    const int32_t expanded_slot = mapping_ub.GetValue(k);
                    route_weight_ub.SetValue(k, expanded_slot < 0 ? 0.0F : topk_weights_gm.GetValue(expanded_slot));
                }
            }

            for (int32_t hidden_begin = 0; hidden_begin < hidden;
                 hidden_begin += static_cast<int32_t>(Ub::kAggregateChunkElements)) {
                const int32_t hidden_count =
                    AggregateMin(static_cast<int32_t>(Ub::kAggregateChunkElements), hidden - hidden_begin);
                const int32_t aligned_hidden_count = AggregateAlignUpElements(hidden_count, 16);
                const uint32_t hidden_chunk_bytes = static_cast<uint32_t>(hidden_count) * tiling.x_element_bytes;

                if (!single_unweighted) {
                    AscendC::Duplicate(accumulator_ub, 0.0F, aligned_hidden_count);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                for (int32_t k = 0; k < (single_unweighted ? 1 : num_topk); ++k) {
                    const int32_t expanded_slot = single_unweighted ? single_slot : mapping_ub.GetValue(k);
                    if (expanded_slot < 0) {
                        continue;
                    }

                    const auto input_event_id = input_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> current_input_bytes_ub =
                        input_bytes_ub[input_buffer_idx * Ub::kAggregateInputBufferBytes];
                    AscendC::LocalTensor<T> current_input_ub = current_input_bytes_ub.ReinterpretCast<T>();
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_event_id);
                    AscendC::DataCopyExtParams input_copy_params(1, hidden_chunk_bytes, 0, 0, 0);
                    AscendC::DataCopyPad(
                        current_input_bytes_ub,
                        x_bytes_gm
                            [static_cast<uint64_t>(expanded_slot) * x_row_bytes +
                             static_cast<uint64_t>(hidden_begin) * tiling.x_element_bytes],
                        input_copy_params, uint8_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_event_id);
                    auto converted_ub = single_unweighted ? accumulator_ub : temporary_ub;
                    AscendC::Cast(converted_ub, current_input_ub, AscendC::RoundMode::CAST_NONE, aligned_hidden_count);
                    AscendC::PipeBarrier<PIPE_V>();
                    if (tiling.has_topk_weights != 0) {
                        AscendC::Muls(temporary_ub, temporary_ub, route_weight_ub.GetValue(k), aligned_hidden_count);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    if (single_unweighted) {
                        // Keep the original FP32 +0 and final cast semantics,
                        // including signed zero and NaNs, instead of a raw copy.
                        AscendC::Adds(accumulator_ub, accumulator_ub, 0.0F, aligned_hidden_count);
                    } else {
                        AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, aligned_hidden_count);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_event_id);
                    input_buffer_idx = (input_buffer_idx + 1) & 1;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
                AscendC::Cast(output_ub, accumulator_ub, AscendC::RoundMode::CAST_RINT, aligned_hidden_count);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopyExtParams output_copy_params(1, hidden_chunk_bytes, 0, 0, 0);
                AscendC::DataCopyPad(
                    aggregated_x_bytes_gm
                        [static_cast<uint64_t>(token) * x_row_bytes +
                         static_cast<uint64_t>(hidden_begin) * tiling.x_element_bytes],
                    output_bytes_ub, output_copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            }
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    if (tiling.has_expanded_values != 0) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        asc_vf_call<AggregateValuesSimtVf>(
            dim3(128), reinterpret_cast<__gm__ int32_t*>(recv_to_expanded_idx),
            reinterpret_cast<__gm__ float*>(expanded_values), reinterpret_cast<__gm__ float*>(aggregated_values),
            token_begin, token_end, num_topk);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
}

} // namespace ascend_deepep::aggregate
