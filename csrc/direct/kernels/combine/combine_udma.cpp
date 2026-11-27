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
constexpr uint32_t kCombineWeightMetaOffset = kCombineTopkAccumulatorOffset + kCombineTopkInputBufferBytes;
constexpr uint32_t kCombineWeightHeadTokens = 16;
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

__aicore__ inline uint64_t ContributorSlot(const ascend_deepep::CombineTilingData& td,
    AscendC::GlobalTensor<int32_t>& channel_prefix, uint32_t contributor, uint32_t token, uint32_t ordinal)
{
    uint32_t row = token;
    if (td.compact_self_rows && contributor == td.rank) {
        const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1) / td.num_channels;
        const uint32_t ch = token / channel_tokens;
        row = channel_prefix.GetValue(td.rank * td.num_channels + ch) + ordinal;
    }
    return uint64_t(contributor) * td.token_capacity + row;
}

__aicore__ inline uint64_t CompactWeightSlot(const ascend_deepep::CombineTilingData& td,
    AscendC::LocalTensor<int32_t>& send_prefix, uint32_t contributor, uint32_t token, uint32_t ordinal)
{
    const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1U) / td.num_channels;
    const uint32_t channel = token / channel_tokens;
    // Shared dense payload/weight mapping. Original send prefixes are
    // inclusive ends, unlike recv-channel starts.
    const uint32_t begin = channel ? send_prefix.GetValue(contributor * td.num_channels + channel - 1U) : 0U;
    return uint64_t(contributor) * td.token_capacity + begin + ordinal;
}

__aicore__ inline uint64_t CombineLocalInputRow(const ascend_deepep::CombineTilingData& td,
    AscendC::GlobalTensor<int32_t>& channel_starts, uint64_t self_input_base, uint32_t token, uint32_t ordinal)
{
    // These are the original dispatch channels, independent of reduce cores.
    // The recv-channel table contains starts relative to the source-rank block.
    const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1U) / td.num_channels;
    const uint32_t channel = token / channel_tokens;
    return self_input_base + channel_starts.GetValue(td.rank * td.num_channels + channel) + ordinal;
}

template <typename T, bool DoCommunication = true, bool DoReduction = true, uint32_t Phase = 0,
          class Trace = ascend_deepep::udma::DisabledDispatchDeviceTrace>
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
    const int32_t core_num = get_block_num() * get_subblockdim();
    Trace trace(tiling.combine_device_trace);
    trace.Set(Trace::Version, 2);
    trace.Set(Trace::Operation, 1);
    trace.Set(Trace::Rank, tiling.rank);
    trace.Set(Trace::Core, core_idx);
    trace.Set(Trace::Entry, trace.Now());
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

    trace.Mark(Trace::InitDone);
    if constexpr (DoCommunication && Phase == 1) {
        ascend_deepep::udma::Begin(symmetric_buffer, tiling.udma_completion_signal_offset, rank, world_size,
            core_idx, core_num, resource.template GetBufferByByte<uint64_t>(0),
            tiling.udma_token_schedule ? ascend_deepep::udma::TokenWorkers(core_num) : tiling.udma_workers_per_peer * tiling.udma_qps_per_worker);
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
                core_idx, core_num, resource.template GetBufferByByte<uint64_t>(0),
                tiling.udma_token_schedule ? ascend_deepep::udma::TokenWorkers(core_num) : tiling.udma_workers_per_peer * tiling.udma_qps_per_worker);
        }

        trace.Mark(Trace::BeginDone);
        // Self copies belong to the sender's post-submit phase, never its entry.
        trace.Mark(Trace::MetadataDone);
        if (tiling.udma_token_schedule) {
            ascend_deepep::udma::CombineTokenRemote<Phase>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx);
        } else {
            ascend_deepep::udma::CombineRemote<Phase>(resource, x, topk_weights, src_idx, rank_prefix_matrix,
                channel_prefix_matrix, symmetric_buffer, tiling, core_idx, trace);
        }

    }

    if constexpr (Trace::enabled) {
        trace.Mark(Trace::OutputDone);
        trace.Mark(Trace::HandleDone);
        trace.Save(resource, core_idx);
    }
    if constexpr (DoReduction) {
        AscendC::GlobalTensor<int32_t> channel_starts;
        channel_starts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix_matrix));
        // Communication is complete, so the reduction phase reuses UB from zero.
        const bool compact_payload = tiling.combine_payload_max_rows > 1U;
        // Reserve an aligned prefix table before the five payload buffers.
        // 14 bytes/element = 2*2 input + 4 temporary + 4 accumulator + 2 output.
        // Reduce cores are independent of the original dispatch channel count.
        const uint32_t payload_prefix_elements = compact_payload ? tiling.world_size * tiling.num_channels : 0U;
        const uint32_t payload_prefix_bytes = (payload_prefix_elements * sizeof(int32_t) + 31U) / 32U * 32U;
        const uint32_t payload_chunk_elements = (kCombineUbBytes - payload_prefix_bytes) / (14U * 64U) * 64U;
        const uint32_t payload_input_bytes = payload_chunk_elements * sizeof(T);
        auto payload_prefix_ub = resource.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<uint8_t> input_pingpong_bytes_ub =
            resource.template GetBufferByByte<uint8_t>(payload_prefix_bytes);
        AscendC::LocalTensor<float> temporary_ub =
            resource.template GetBufferByByte<float>(payload_prefix_bytes + 2U * payload_input_bytes);
        AscendC::LocalTensor<float> accumulator_ub =
            resource.template GetBufferByByte<float>(payload_prefix_bytes + 4U * payload_input_bytes);
        AscendC::LocalTensor<uint8_t> output_bytes_ub =
            resource.template GetBufferByByte<uint8_t>(payload_prefix_bytes + 6U * payload_input_bytes);
        AscendC::LocalTensor<T> output_ub = output_bytes_ub.ReinterpretCast<T>();

        const int32_t tokens_per_channel = (num_tokens + core_num - 1) / core_num;
        const int32_t token_begin = CombineMin(core_idx * tokens_per_channel, num_tokens);
        const int32_t token_end = CombineMin(token_begin + tokens_per_channel, num_tokens);
        if (compact_payload && token_begin < token_end) {
            AscendC::GlobalTensor<int32_t> prefixes;
            prefixes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling.combine_send_channel_prefix));
            AscendC::DataCopyExtParams copy{1, payload_prefix_elements * uint32_t(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(payload_prefix_ub, prefixes, copy,
                                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
        }
        uint64_t self_input_base = 0;
        if (tiling.combine_direct_self && token_begin < token_end && tiling.rank != 0) {
            AscendC::GlobalTensor<int32_t> ranks;
            ranks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix_matrix));
            self_input_base = ranks.GetValue((tiling.rank - 1U) * tiling.world_size + tiling.rank);
        }

        int32_t input_buffer_idx = 0;
        // EVENT_ID0/1 protect the two GM-to-UB input slots. EVENT_ID2 protects
        // the independent output slot while the next token is accumulated.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        for (int32_t token = token_begin; token < token_end; ++token) {
            uint64_t self_input_row = 0;
            const int32_t self_ordinal = tiling.combine_direct_self
                ? send_head_gm.GetValue(token * world_size + rank) : -1;
            if (self_ordinal >= 0)
                self_input_row = CombineLocalInputRow(tiling, channel_starts, self_input_base, token, self_ordinal);
            for (int32_t hidden_begin = 0; hidden_begin < hidden; hidden_begin += payload_chunk_elements) {
                const int32_t hidden_count = CombineMin(static_cast<int32_t>(payload_chunk_elements), hidden - hidden_begin);
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
                        const int32_t ordinal = tiling.combine_direct_self && contributor == rank
                            ? self_ordinal : send_head_gm.GetValue(token * world_size + contributor);
                        if (ordinal < 0) {
                            continue;
                        }
                        if (tiling.combine_direct_self && contributor == rank) {
                            // Input tensors are tightly packed even when SHMEM rows are padded.
                            contributor_ptr = reinterpret_cast<__gm__ uint8_t*>(x) + self_input_row * hidden_bytes;
                        } else {
                            const uint64_t local_slot = compact_payload
                                ? CompactWeightSlot(tiling, payload_prefix_ub, contributor, token, ordinal)
                                : ContributorSlot(tiling, channel_starts, contributor, token, ordinal);
                            contributor_ptr = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_buffer_offset +
                                              local_slot * tiling.x_row_stride;
                        }
                    }

                    AscendC::GlobalTensor<uint8_t> contributor_gm;
                    contributor_gm.SetGlobalBuffer(contributor_ptr);
                    AscendC::DataCopyExtParams contributor_copy_params(1, hidden_chunk_bytes, 0, 0, 0);
                    const auto input_event_id = input_buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<uint8_t> input_bytes_ub =
                        input_pingpong_bytes_ub[input_buffer_idx * payload_input_bytes];
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
            // Scalar prefix readers are finished before weight DMA reuses UB.
            if (compact_payload) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
            }
            // X reduction is drained, so top-k reduction also starts at UB offset 0.
            AscendC::LocalTensor<uint8_t> weight_input_bytes_ub =
                resource.template GetBufferByByte<uint8_t>(kCombineTopkInputOffset);
            AscendC::LocalTensor<float> weight_input_ub = weight_input_bytes_ub.ReinterpretCast<float>();
            AscendC::LocalTensor<float> weight_accumulator_ub =
                resource.template GetBufferByByte<float>(kCombineTopkAccumulatorOffset);
            const int32_t aligned_num_topk = CombineAlignUpElements(num_topk, 8);
            // Payload UB is no longer live. Keep both FP32 vectors in [0,1024),
            // then the aligned original send-prefix table and a head tile.
            const uint32_t prefix_elements = tiling.world_size * tiling.num_channels;
            const uint32_t head_offset = kCombineWeightMetaOffset +
                (prefix_elements * sizeof(int32_t) + 31U) / 32U * 32U;
            auto prefix_ub = resource.template GetBufferByByte<int32_t>(kCombineWeightMetaOffset);
            auto heads_ub = resource.template GetBufferByByte<int32_t>(head_offset);
            if (tiling.combine_compact_weights && token_begin < token_end) {
                AscendC::GlobalTensor<int32_t> prefixes;
                prefixes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling.combine_send_channel_prefix));
                AscendC::DataCopyExtParams copy{1, prefix_elements * uint32_t(sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPad(prefix_ub, prefixes, copy,
                                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
            }
            int32_t head_tile_begin = token_begin;
            // EVENT_ID3 protects the single input buffer; EVENT_ID5 protects output.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);

            for (int32_t token = token_begin; token < token_end; ++token) {
                if (tiling.combine_compact_weights && (token - token_begin) % kCombineWeightHeadTokens == 0) {
                    head_tile_begin = token;
                    const uint32_t count = CombineMin(kCombineWeightHeadTokens, token_end - token);
                    // Scalar readers of the previous tile finish before reuse.
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::DataCopyExtParams copy{1, count * tiling.world_size * uint32_t(sizeof(int32_t)), 0, 0, 0};
                    AscendC::DataCopyPad(heads_ub, send_head_gm[uint64_t(token) * world_size], copy,
                                        AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
                AscendC::Duplicate(weight_accumulator_ub, 0.0F, aligned_num_topk);
                AscendC::PipeBarrier<PIPE_V>();

                for (int32_t contributor = 0; contributor < world_size; ++contributor) {
                    const int32_t ordinal = tiling.combine_compact_weights
                        ? heads_ub.GetValue((token - head_tile_begin) * world_size + contributor)
                        : send_head_gm.GetValue(token * world_size + contributor);
                    if (ordinal < 0) {
                        continue;
                    }
                    const bool direct_self = tiling.combine_direct_self && contributor == rank;
                    const uint64_t local_slot = direct_self
                        ? CombineLocalInputRow(tiling, channel_starts, self_input_base, token, ordinal)
                        : (tiling.combine_compact_weights
                            ? CompactWeightSlot(tiling, prefix_ub, contributor, token, ordinal)
                            : ContributorSlot(tiling, channel_starts, contributor, token, ordinal));
                    AscendC::DataCopyExtParams weight_copy_params(1, topk_weights_row_bytes, 0, 0, 0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
                    if (direct_self || tiling.combine_compact_weights) {
                        AscendC::GlobalTensor<float> weight_source;
                        weight_source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(direct_self
                            ? topk_weights + local_slot * topk_weights_row_bytes
                            : symmetric_buffer + tiling.topk_weights_buffer_offset + local_slot * tiling.topk_weights_row_stride));
                        AscendC::DataCopyPad(weight_input_ub, weight_source, weight_copy_params,
                            AscendC::DataCopyPadExtParams<float>{true, 0,
                                static_cast<uint8_t>(aligned_num_topk - num_topk), 0.0F});
                    } else {
                        AscendC::DataCopyPad(weight_input_bytes_ub,
                            local_shmem_gm[tiling.topk_weights_buffer_offset + local_slot * tiling.topk_weights_row_stride],
                            weight_copy_params, uint8_pad_params);
                    }
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

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_comm_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_comm_trace_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t, true, false, 0, ascend_deepep::udma::DispatchDeviceTrace>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half, true, false, 0, ascend_deepep::udma::DispatchDeviceTrace>(
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
    if (td->combine_device_trace) {
        v1_combine_udma_comm_trace_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
    } else {
        v1_combine_udma_comm_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kCombineUbBytes), stream>>>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, *td, ffts_addr);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_reduce_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_early_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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


extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_remaining_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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


extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void v1_combine_udma_pipeline_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    aclshmemx_instance_ctx_set(tiling.instance_id);
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
