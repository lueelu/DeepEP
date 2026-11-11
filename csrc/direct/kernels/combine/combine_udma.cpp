#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "common/launch.h"
#include "common/types.h"
#include "kernels/combine/udma_combine_schedule.h"
#include "kernels/combine/udma_token_combine_schedule.h"
#include "kernels/combine/combine_pipeline_impl.h"

using namespace AscendC;

namespace {

constexpr uint32_t kCombineUbBytes = static_cast<uint32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr uint32_t kCombineAlignmentBytes = 32;
constexpr uint32_t kCombineFloatBytes = sizeof(float);
constexpr uint32_t kCombineMaxTopk = 128;
// One tile uses 14 bytes per element: two BF16/FP16 input slots, two FP32
// workspaces, and one BF16/FP16 output. Keep the tile on a 64-element vector
// repeat boundary and consume the available UB on each architecture.
constexpr uint32_t kCombineChunkElements = kCombineUbBytes / (14U * 64U) * 64U;
constexpr uint32_t kCombineInputBufferNum = 2;
constexpr uint32_t kCombineInputOffset = 0;
constexpr uint32_t kCombineInputBufferBytes = kCombineChunkElements * sizeof(bfloat16_t);
constexpr uint32_t kCombineTemporaryOffset = kCombineInputOffset + kCombineInputBufferNum * kCombineInputBufferBytes;
constexpr uint32_t kCombineFloatChunkBytes = kCombineChunkElements * sizeof(float);
constexpr uint32_t kCombineAccumulatorOffset = kCombineTemporaryOffset + kCombineFloatChunkBytes;
constexpr uint32_t kCombineOutputOffset = kCombineAccumulatorOffset + kCombineFloatChunkBytes;
constexpr uint32_t kCombineOutputBytes = kCombineInputBufferBytes;
constexpr uint32_t kCombineTopkInputOffset = 0;
constexpr uint32_t kCombineTopkInputBufferBytes = kCombineMaxTopk * sizeof(float);
constexpr uint32_t kCombineTopkAccumulatorOffset = kCombineTopkInputOffset + kCombineTopkInputBufferBytes;
constexpr uint32_t kCombineDtypeBfloat16 = 0;
constexpr uint32_t kCombineDtypeFloat16 = 1;



static_assert(kCombineOutputOffset + kCombineOutputBytes <= kCombineUbBytes);
static_assert(kCombineTopkAccumulatorOffset + kCombineTopkInputBufferBytes <= kCombineUbBytes);

__aicore__ inline int32_t CombineMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline int32_t CombineAlignUpElements(int32_t value, int32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T, bool DoCommunication = true, bool DoReduction = true, uint32_t Phase = 0>
__aicore__ inline void V1CombineUdmaImpl(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, const ascend_deepep::CombineTilingData& tiling)
{
    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t rank = static_cast<int32_t>(tiling.rank);
    const int32_t world_size = static_cast<int32_t>(tiling.world_size);
    const int32_t num_tokens = static_cast<int32_t>(tiling.num_tokens);
    const int32_t hidden = static_cast<int32_t>(tiling.hidden);
    const int32_t num_topk = static_cast<int32_t>(tiling.num_topk);
    const int32_t num_channels = static_cast<int32_t>(tiling.num_channels);
    const int32_t token_capacity = static_cast<int32_t>(tiling.token_capacity);
    const uint32_t hidden_bytes = static_cast<uint32_t>(hidden) * tiling.x_element_bytes;
    const uint32_t topk_weights_row_bytes = static_cast<uint32_t>(num_topk) * kCombineFloatBytes;

    AscendC::GlobalTensor<int32_t> send_head_gm;
    AscendC::GlobalTensor<uint8_t> local_shmem_gm;
    AscendC::GlobalTensor<uint8_t> recv_x_bytes_gm;
    AscendC::GlobalTensor<float> recv_topk_weights_gm;

    send_head_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_head));
    local_shmem_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer));
    recv_x_bytes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recv_x));
    if (tiling.has_topk_weights != 0) {
        recv_topk_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(recv_topk_weights));
    }

    ascend_deepep::UbWorkspace resource;
    AscendC::DataCopyPadExtParams<uint8_t> uint8_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> float_pad_params{false, 0, 0, 0};

    if constexpr (DoCommunication && Phase == 1) {
        ascend_deepep::udma::Begin(symmetric_buffer, tiling.udma_completion_signal_offset, rank, world_size,
            core_idx, num_channels, resource.template GetBufferByByte<uint64_t>(0),
            tiling.udma_token_schedule ? ascend_deepep::udma::TokenWorkers(num_channels) : tiling.udma_workers_per_peer * tiling.udma_qps_per_worker);
        if (tiling.udma_token_schedule) {
            ascend_deepep::udma::CombineTokenRemote<1>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        } else {
            ascend_deepep::udma::CombineRemote<1>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        }
    }
    if constexpr (DoCommunication && Phase != 1) {
        if constexpr (Phase == 0) {
            ascend_deepep::udma::Begin(symmetric_buffer, tiling.udma_completion_signal_offset, rank, world_size,
                core_idx, num_channels, resource.template GetBufferByByte<uint64_t>(0),
                tiling.udma_token_schedule ? ascend_deepep::udma::TokenWorkers(num_channels) : tiling.udma_workers_per_peer * tiling.udma_qps_per_worker);
        }

        if (!tiling.udma_self_copy_after_submit) {
            ascend_deepep::udma::CombineSelfCopy(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        }
        if (tiling.udma_token_schedule) {
            ascend_deepep::udma::CombineTokenRemote<Phase>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        } else {
            ascend_deepep::udma::CombineRemote<Phase>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        }

    }

    if constexpr (DoReduction) {
        // Communication is complete, so the reduction phase reuses UB from zero.
        AscendC::LocalTensor<uint8_t> input_pingpong_bytes_ub =
            resource.template GetBufferByByte<uint8_t>(kCombineInputOffset);
        AscendC::LocalTensor<float> temporary_ub = resource.template GetBufferByByte<float>(kCombineTemporaryOffset);
        AscendC::LocalTensor<float> accumulator_ub =
            resource.template GetBufferByByte<float>(kCombineAccumulatorOffset);
        AscendC::LocalTensor<uint8_t> output_bytes_ub =
            resource.template GetBufferByByte<uint8_t>(kCombineOutputOffset);
        AscendC::LocalTensor<T> output_ub = output_bytes_ub.ReinterpretCast<T>();

        const int32_t tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
        const int32_t token_begin = CombineMin(core_idx * tokens_per_channel, num_tokens);
        const int32_t token_end = CombineMin(token_begin + tokens_per_channel, num_tokens);

        int32_t input_buffer_idx = 0;
        // EVENT_ID0/1 protect the two GM-to-UB input slots. EVENT_ID2 protects
        // the independent output slot while the next token is accumulated.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        for (int32_t token = token_begin; token < token_end; ++token) {
            for (int32_t hidden_begin = 0; hidden_begin < hidden; hidden_begin += kCombineChunkElements) {
                const int32_t hidden_count = CombineMin(static_cast<int32_t>(kCombineChunkElements), hidden - hidden_begin);
                const int32_t aligned_hidden_count = CombineAlignUpElements(hidden_count, 16);
                const uint32_t hidden_chunk_bytes = static_cast<uint32_t>(hidden_count) * tiling.x_element_bytes;

                AscendC::Duplicate(accumulator_ub, 0.0F, aligned_hidden_count);
                AscendC::PipeBarrier<PIPE_V>();

                for (int32_t contributor = -2; contributor < world_size; ++contributor) {
                    __gm__ uint8_t* contributor_ptr = nullptr;
                    if (contributor == -2) {
                        if (tiling.has_bias_0 == 0) {
                            continue;
                        }
                        contributor_ptr =
                            reinterpret_cast<__gm__ uint8_t*>(bias_0) + static_cast<uint64_t>(token) * hidden_bytes;
                    } else if (contributor == -1) {
                        if (tiling.has_bias_1 == 0) {
                            continue;
                        }
                        contributor_ptr =
                            reinterpret_cast<__gm__ uint8_t*>(bias_1) + static_cast<uint64_t>(token) * hidden_bytes;
                    } else {
                        if (send_head_gm.GetValue(token * world_size + contributor) < 0) {
                            continue;
                        }
                        const uint64_t local_slot = static_cast<uint64_t>(contributor) * token_capacity + token;
                        contributor_ptr = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_buffer_offset +
                                          local_slot * tiling.x_row_stride;
                    }

                    AscendC::GlobalTensor<uint8_t> contributor_gm;
                    contributor_gm.SetGlobalBuffer(contributor_ptr);
                    AscendC::DataCopyExtParams contributor_copy_params(1, hidden_chunk_bytes, 0, 0, 0);
                    const auto input_event_id = input_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> input_bytes_ub =
                        input_pingpong_bytes_ub[input_buffer_idx * kCombineInputBufferBytes];
                    AscendC::LocalTensor<T> input_ub = input_bytes_ub.ReinterpretCast<T>();
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(input_event_id);
                    AscendC::DataCopyPad(
                        input_bytes_ub, contributor_gm[static_cast<uint64_t>(hidden_begin) * tiling.x_element_bytes],
                        contributor_copy_params, uint8_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(input_event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(input_event_id);
                    AscendC::Cast(temporary_ub, input_ub, AscendC::RoundMode::CAST_NONE, aligned_hidden_count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(input_event_id);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator_ub, accumulator_ub, temporary_ub, aligned_hidden_count);
                    input_buffer_idx = (input_buffer_idx + 1) & 1;
                }

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
                AscendC::Cast(output_ub, accumulator_ub, AscendC::RoundMode::CAST_RINT, aligned_hidden_count);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopyExtParams output_copy_params(1, hidden_chunk_bytes, 0, 0, 0);
                AscendC::DataCopyPad(
                    recv_x_bytes_gm
                        [static_cast<uint64_t>(token) * hidden_bytes +
                         static_cast<uint64_t>(hidden_begin) * tiling.x_element_bytes],
                    output_bytes_ub, output_copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);

        if (tiling.has_topk_weights != 0) {
            // X reduction is drained, so top-k reduction also starts at UB offset 0.
            AscendC::LocalTensor<uint8_t> weight_input_bytes_ub =
                resource.template GetBufferByByte<uint8_t>(kCombineTopkInputOffset);
            AscendC::LocalTensor<float> weight_input_ub = weight_input_bytes_ub.ReinterpretCast<float>();
            AscendC::LocalTensor<float> weight_accumulator_ub =
                resource.template GetBufferByByte<float>(kCombineTopkAccumulatorOffset);
            const int32_t aligned_num_topk = CombineAlignUpElements(num_topk, 8);
            // EVENT_ID3 protects the single input buffer; EVENT_ID5 protects output.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);

            for (int32_t token = token_begin; token < token_end; ++token) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
                AscendC::Duplicate(weight_accumulator_ub, 0.0F, aligned_num_topk);
                AscendC::PipeBarrier<PIPE_V>();

                for (int32_t contributor = 0; contributor < world_size; ++contributor) {
                    if (send_head_gm.GetValue(token * world_size + contributor) < 0) {
                        continue;
                    }
                    const uint64_t local_slot = static_cast<uint64_t>(contributor) * token_capacity + token;
                    AscendC::DataCopyExtParams weight_copy_params(1, topk_weights_row_bytes, 0, 0, 0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
                    AscendC::DataCopyPad(
                        weight_input_bytes_ub,
                        local_shmem_gm[tiling.topk_weights_buffer_offset + local_slot * tiling.topk_weights_row_stride],
                        weight_copy_params, uint8_pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                    AscendC::Add(weight_accumulator_ub, weight_accumulator_ub, weight_input_ub, aligned_num_topk);
                    // The next contributor reads the accumulator on PIPE_V.
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
                AscendC::DataCopyExtParams weight_output_params(1, topk_weights_row_bytes, 0, 0, 0);
                AscendC::DataCopyPad(recv_topk_weights_gm[token * num_topk], weight_accumulator_ub, weight_output_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        }
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_comm_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t, true, false>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half, true, false>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_comm_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_comm_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_reduce_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t, false, true>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half, false, true>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_reduce_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_reduce_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_early_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t, true, false, 1>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half, true, false, 1>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_early_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_early_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}


extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_remaining_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t, true, false, 2>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half, true, false, 2>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_remaining_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_remaining_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}


extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_pipeline_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        ascend_deepep::udma::CombinePipelineImpl<bfloat16_t>(
            x, topk_weights, src_idx, rank_prefix_matrix, channel_prefix_matrix, symmetric_buffer, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        ascend_deepep::udma::CombinePipelineImpl<half>(
            x, topk_weights, src_idx, rank_prefix_matrix, channel_prefix_matrix, symmetric_buffer, tiling);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_combine_udma_pipeline_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* topk_weights, uint8_t* bias_0, uint8_t* bias_1,
    uint8_t* src_idx, uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* send_head,
    uint8_t* symmetric_buffer, uint8_t* recv_x, uint8_t* recv_topk_weights, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::CombineTilingData*>(tiling_host);
    v1_combine_udma_pipeline_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
        symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
}
