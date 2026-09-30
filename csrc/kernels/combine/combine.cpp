// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/device_functions.h"
#include "tiling.hpp"
#include "launch.hpp"
using namespace AscendC;
namespace {
template <typename T>
__aicore__ inline void NativeLoad(LocalTensor<T> ub, __gm__ T* source, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(source);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(ub, gm, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0), DataCopyPadExtParams<T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}
template <typename T>
__aicore__ inline void NativeStore(__gm__ T* destination, LocalTensor<T> ub, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(destination);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
    DataCopyPad(gm, ub, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}
}  // namespace
#include "../common.hpp"
#include "tiling.hpp"

namespace {
using CombineTiling = ascend_deepep::PutCombineTiling;
namespace combine_profile = ascend_deepep::combine_profile;
namespace combine_control = ascend_deepep::combine_control;
namespace combine_config = ascend_deepep::combine_config;
#if (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)) && CATLASS_ARCH == 3510
__aicore__ inline uint32_t CombineGroup(uint32_t rank, uint32_t world, uint32_t peer, uint32_t width)
{
    return (first_hit_schedule::Group(rank, world, peer) * first_hit_schedule::Width(world) +
            first_hit_schedule::Index(peer, world)) /
           width;
}
__aicore__ inline uint32_t CombinePeer(const CombineTiling& tiling, uint32_t group, uint32_t index)
{
    const uint32_t ordinal = group * tiling.group_size + index;
    const uint32_t width = first_hit_schedule::Width(tiling.rank_num);
    return first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, ordinal / width, ordinal % width);
}
__aicore__ inline uint32_t TraceGroupCapacity(const CombineTiling& tiling)
{
    const uint32_t groups = tiling.rank_num / tiling.group_size;
    return groups > 8U ? groups : 8U;
}
__aicore__ inline uint32_t TraceBaseRecords(const CombineTiling& tiling)
{
    return 27U + 2U * TraceGroupCapacity(tiling);
}
constexpr uint32_t kTraceUbBytes = 10U * combine_profile::kWords * sizeof(uint64_t);
// Trace slots are reused only after their records are saved to GM.
template <bool EnableTrace>
struct Trace {
    TBuf<QuePosition::VECCALC> trace_ub_buffer;
    LocalTensor<uint64_t> trace_words_ub;
    __gm__ uint64_t* trace_records_gm = nullptr;
    uint32_t rank_id = 0, core_id = 0, rank_num = 0;
    uint64_t record_num = 0;
    uint32_t base_records = 0;
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR trace_records_address, const CombineTiling& tiling,
                                uint32_t current_core_id)
    {
        if constexpr (EnableTrace) {
            pipe.InitBuffer(trace_ub_buffer, kTraceUbBytes);
            trace_words_ub = trace_ub_buffer.template Get<uint64_t>();
            trace_records_gm = reinterpret_cast<__gm__ uint64_t*>(trace_records_address);
            rank_id = tiling.rank_id;
            core_id = current_core_id;
            rank_num = tiling.rank_num;
            base_records = TraceBaseRecords(tiling);
            record_num = base_records + uint64_t(tiling.total_chunk_num) * combine_profile::kChunkRecords +
                         uint64_t(tiling.rank_num) * tiling.chunks_per_group;
            Reset(combine_profile::SummarySlot, combine_profile::Summary);
            Set(combine_profile::SummarySlot, combine_profile::World, tiling.rank_num);
            Set(combine_profile::SummarySlot, combine_profile::Hidden, tiling.hidden_size);
        }
    }
    __aicore__ inline void Set(uint32_t trace_slot, uint32_t field_index, uint64_t field_value)
    {
        if constexpr (EnableTrace) {
            trace_words_ub.SetValue(trace_slot * combine_profile::kWords + field_index, field_value);
        }
    }
    __aicore__ inline void Add(uint32_t trace_slot, uint32_t field_index, uint64_t field_value)
    {
        if constexpr (EnableTrace) {
            const uint32_t trace_word_index = trace_slot * combine_profile::kWords + field_index;
            trace_words_ub.SetValue(trace_word_index, trace_words_ub.GetValue(trace_word_index) + field_value);
        }
    }
    __aicore__ inline void Mark(uint32_t trace_slot, uint32_t field_index)
    {
        if constexpr (EnableTrace) {
            Set(trace_slot, field_index, GetSystemCycle());
        }
    }
    __aicore__ inline void First(uint32_t trace_slot, uint32_t field_index)
    {
        if constexpr (EnableTrace) {
            if (!trace_words_ub.GetValue(trace_slot * combine_profile::kWords + field_index)) {
                Mark(trace_slot, field_index);
            }
        }
    }
    __aicore__ inline void Reset(uint32_t trace_slot, uint32_t record_kind, uint32_t group_id = 0)
    {
        if constexpr (EnableTrace) {
            for (uint32_t trace_word_index = 0; trace_word_index < combine_profile::kWords; ++trace_word_index) {
                Set(trace_slot, trace_word_index, 0);
            }
            Set(trace_slot, combine_profile::Version, combine_profile::kVersion);
            Set(trace_slot, combine_profile::Rank, rank_id);
            Set(trace_slot, combine_profile::Core, core_id);
            Set(trace_slot, combine_profile::Type, record_kind);
            Set(trace_slot, combine_profile::GroupId, group_id);
            Set(trace_slot, combine_profile::Peer, rank_num);
            Set(trace_slot, combine_profile::Incoming, rank_num);
        }
    }
    __aicore__ inline uint64_t ChunkRecordIndex(uint32_t global_chunk, uint32_t offset) const
    {
        return base_records + uint64_t(global_chunk) * combine_profile::kChunkRecords + offset;
    }
    __aicore__ inline uint64_t Get(uint32_t trace_slot, uint32_t field_index)
    {
        if constexpr (EnableTrace) {
            return trace_words_ub.GetValue(trace_slot * combine_profile::kWords + field_index);
        }
        return 0;
    }
    __aicore__ inline void Save(uint32_t trace_slot, uint64_t record_index)
    {
        if constexpr (EnableTrace) {
            Set(trace_slot, combine_profile::Valid, 1);
            NativeStore(trace_records_gm + (uint64_t(core_id) * record_num + record_index) * combine_profile::kWords,
                        trace_words_ub[trace_slot * combine_profile::kWords], combine_profile::kWords);
        }
    }
    __aicore__ inline void Load(uint32_t trace_slot, uint64_t record_index)
    {
        if constexpr (EnableTrace) {
            NativeLoad(trace_words_ub[trace_slot * combine_profile::kWords],
                       trace_records_gm + (uint64_t(core_id) * record_num + record_index) * combine_profile::kWords,
                       combine_profile::kWords);
        }
    }
};
template <>
struct Trace<false> {
    __aicore__ inline void Init(TPipe&, GM_ADDR, const CombineTiling&, uint32_t) {}
    __aicore__ inline void Set(uint32_t, uint32_t, uint64_t) {}
    __aicore__ inline void Add(uint32_t, uint32_t, uint64_t) {}
    __aicore__ inline void Mark(uint32_t, uint32_t) {}
    __aicore__ inline void First(uint32_t, uint32_t) {}
    __aicore__ inline void Reset(uint32_t, uint32_t, uint32_t = 0) {}
    __aicore__ inline void Save(uint32_t, uint64_t) {}
    __aicore__ inline void Load(uint32_t, uint64_t) {}
    __aicore__ inline uint64_t ChunkRecordIndex(uint32_t, uint32_t) const
    {
        return 0;
    }
    __aicore__ inline uint64_t Get(uint32_t, uint32_t) const
    {
        return 0;
    }
};
constexpr uint32_t kReduceTileElements = 8192;
constexpr uint32_t kReduceOutputSlots = 2;
constexpr uint32_t kLocalCopyTileBytes = 16384;
constexpr uint32_t kLocalCopySlots = 2;
constexpr uint32_t kLocalCopyBufferBytes = kLocalCopySlots * kLocalCopyTileBytes;
constexpr uint32_t kWqeBytes = 64;
constexpr uint32_t kWqeBatchSize = 128;
constexpr uint32_t kWqeBatchScratchBytes = kWqeBatchSize * kWqeBytes;
// Each cross owner keeps one independent batch per uDie/QP.
constexpr uint32_t kWqeScratchBytes = 2U * kWqeBatchScratchBytes;
constexpr uint32_t kGatherBatchRows = 128;
constexpr uint32_t kGatherTaskFields = 3;
constexpr uint32_t kSignalUbBytes = 512;
constexpr uint32_t kCombineUbBudgetBytes = 192U * 1024U;
constexpr uint32_t kReservedUbBytes = 8U * 1024U;
// 输入 BF16、转换临时 FP32、累加器 FP32，加上两个互不重叠的 BF16 输出槽。
constexpr uint32_t kReduceUbBytes =
    kReduceTileElements * ((1U + kReduceOutputSlots) * sizeof(bfloat16_t) + 2U * sizeof(float));
constexpr uint32_t kMainUbBytes =
    kReduceUbBytes + combine_config::kGatewayUbBytes + kWqeScratchBytes + kSignalUbBytes + kTraceUbBytes;
static_assert(2U * kReduceTileElements * sizeof(bfloat16_t) <= kLocalCopyBufferBytes,
              "source dual-input bank must fit copy UB");
static_assert(kReduceTileElements % 128U == 0U, "two 64-element VF chains need padded tiles");
static_assert(kReduceTileElements % 16U == 0U, "BF16 and FP32 reduction tiles must be 32-byte aligned");
static_assert(kMainUbBytes + kReservedUbBytes <= kCombineUbBudgetBytes, "Combine UB budget exceeded");
static_assert(kCombineUbBudgetBytes <= combine_config::kLaunchUbBytes, "TPipe exceeds launch UB");
static_assert(combine_config::kLaunchUbBytes + 8U * 1024U + 32U * 1024U <= 256U * 1024U,
              "Dynamic UB, system reserve and SIMT cache exceed physical UB");
static_assert(combine_control::kGatherDoneLanes * combine_control::kSlotBytes <= kLocalCopyTileBytes,
              "gather flags must fit byte-copy UB");
static_assert(kGatherBatchRows * kGatherTaskFields * sizeof(int64_t) <= kWqeScratchBytes,
              "gather metadata batch must fit reused WQE scratch UB");
static_assert(kGatherBatchRows * kWqeBytes <= kWqeBatchScratchBytes, "gather WQE batch must fit scratch UB");
static_assert(kGatherBatchRows * kGatherTaskFields * sizeof(int64_t) <= kLocalCopyBufferBytes,
              "URMA gather metadata must fit copy UB");
constexpr aclshmemx_udma_op_config_t kPayloadRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kPayloadRoConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kCompletionSoConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
enum class PartialClass : uint32_t { All, DirectOnly, ReducedOnly };
// Control slots are 512B apart; each reducer publishes its own completion epoch.
__aicore__ inline __gm__ uint64_t* Flag(GM_ADDR workspace, uint32_t flag_slot)
{
    return reinterpret_cast<__gm__ uint64_t*>(workspace + uint64_t(flag_slot) * combine_control::kSlotBytes);
}
// Two 64-lane chains cover adjacent hidden positions, preserving server addition order.
namespace source_reg = AscendC::MicroAPI;
constexpr AscendC::MicroAPI::CastTrait kReduceBf16ToFp32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait kReduceFp32ToBf16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

template <bool HasAccumulator, bool HasSecond, bool Final>
__simd_vf__ void SourceReducePairVf(__ubuf__ bfloat16_t* first, __ubuf__ bfloat16_t* second,
                                    __ubuf__ float* accumulator, __ubuf__ bfloat16_t* output, uint32_t element_num)
{
    uint32_t offset = 0;
    for (; offset + 128U <= element_num; offset += 128U) {
        uint32_t valid = 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value0, bf16_value1;
        AscendC::MicroAPI::RegTensor<float> sum0, sum1, value0, value1;
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::LoadAlign(sum0, accumulator + offset);
            AscendC::MicroAPI::LoadAlign(sum1, accumulator + offset + 64U);
        }
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                               first + offset);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value1,
                                                                                               first + offset + 64U);
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
            AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
            AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        } else {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum1, bf16_value1, mask);
        }
        if constexpr (HasSecond) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                                   second + offset);
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                bf16_value1, second + offset + 64U);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
            AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
            AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        }
        if constexpr (Final) {
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value0, sum0, mask);
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value1, sum1, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                                   bf16_value0, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                output + offset + 64U, bf16_value1, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(accumulator + offset, sum0, mask);
            AscendC::MicroAPI::StoreAlign(accumulator + offset + 64U, sum1, mask);
        }
    }
    // At most two vectors remain; do not load a second vector beyond the tail.
    for (; offset < element_num; offset += 64U) {
        uint32_t valid = element_num - offset < 64U ? element_num - offset : 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value;
        AscendC::MicroAPI::RegTensor<float> sum, value;
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::LoadAlign(sum, accumulator + offset);
        }
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                               first + offset);
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
            AscendC::MicroAPI::Add(sum, sum, value, mask);
        } else {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum, bf16_value, mask);
        }
        if constexpr (HasSecond) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                                   second + offset);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
            AscendC::MicroAPI::Add(sum, sum, value, mask);
        }
        if constexpr (Final) {
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value, sum, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                                   bf16_value, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(accumulator + offset, sum, mask);
        }
    }
}

__aicore__ inline void RunSourceReducePair(__ubuf__ bfloat16_t* first, __ubuf__ bfloat16_t* second,
                                           __ubuf__ float* accumulator, __ubuf__ bfloat16_t* output, uint32_t count,
                                           bool initialize, bool has_second, bool final_pair)
{
    // An unpaired input is necessarily the final active server.
    if (initialize) {
        if (!has_second) {
            asc_vf_call<SourceReducePairVf<false, false, true>>(first, second, accumulator, output, count);
        } else if (final_pair) {
            asc_vf_call<SourceReducePairVf<false, true, true>>(first, second, accumulator, output, count);
        } else {
            asc_vf_call<SourceReducePairVf<false, true, false>>(first, second, accumulator, output, count);
        }
    } else if (!has_second) {
        asc_vf_call<SourceReducePairVf<true, false, true>>(first, second, accumulator, output, count);
    } else if (final_pair) {
        asc_vf_call<SourceReducePairVf<true, true, true>>(first, second, accumulator, output, count);
    } else {
        asc_vf_call<SourceReducePairVf<true, true, false>>(first, second, accumulator, output, count);
    }
}
__aicore__ inline void PrefetchSourcePair(GM_ADDR workspace, const CombineTiling& tiling, uint32_t token_id,
                                          uint32_t server_num, uint32_t first_server, uint32_t second_server,
                                          uint32_t hidden_offset, uint32_t element_num, LocalTensor<bfloat16_t> bank,
                                          uint32_t bank_index, bool& read_pending)
{
    const auto input_event = bank_index == 0U ? EVENT_ID0 : EVENT_ID4;
    if (read_pending) {
        WaitFlag<HardEvent::V_MTE2>(input_event);
        read_pending = false;
    }
    for (uint32_t input = 0; input < 2U; ++input) {
        const uint32_t server = input == 0U ? first_server : second_server;
        if (server >= server_num) {
            continue;  // Odd group: no DMA and no VF read of the absent input.
        }
        GlobalTensor<bfloat16_t> partial_gm;
        partial_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.returned_partial_offset_bytes) +
            (uint64_t(token_id) * server_num + server) * tiling.hidden_size + hidden_offset);
        // hidden_size is a multiple of 16 BF16 elements; tile offsets and UB
        // banks are also 32B aligned, including the last tile's byte count.
        DataCopy(bank[input * kReduceTileElements], partial_gm, element_num);
    }
    SetFlag<HardEvent::MTE2_V>(input_event);
}
__aicore__ inline bool Ready(__gm__ uint64_t* flag_address, uint64_t expected_value)
{
    GlobalTensor<uint64_t> flag_gm;
    flag_gm.SetGlobalBuffer(flag_address);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag_address));
    return flag_gm.GetValue(0) >= expected_value;
}
__aicore__ inline void StoreFlag(__gm__ uint64_t* flag_address, LocalTensor<uint64_t> flag_ub, uint64_t flag_value)
{
    flag_ub.SetValue(0, flag_value);
    NativeStore(flag_address, flag_ub, 1U);
}
__aicore__ inline void Prepare(GM_ADDR workspace, const CombineTiling& tiling)
{
    const uint32_t core_id = GetBlockIdx();
    const uint32_t rank_num = tiling.rank_num, control_slot_num = tiling.control_slot_num;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> signal_buffer;
    pipe.InitBuffer(signal_buffer, kSignalUbBytes);
    auto signal_ub = signal_buffer.Get<uint64_t>();

    // 上次 Combine 已排空本核 QP，并检查所有入站 chunk 的 SO。
    // 较慢的 rank 此后只会归约自己的回传数据，不会再写本卡控制槽。
    // 这里仅重置控制槽，不修改数据区，因此无需在重置前再增加一次集合同步。
    for (uint64_t flag_slot = core_id; flag_slot < control_slot_num; flag_slot += combine_config::kAivCores) {
        // 每个 512 字节槽位只有一个写入核，仅首个 uint64 用作信号。
        // 所有 QP 完成前，SO 使用的信号源值始终保持为 1。
        const bool completion_source = flag_slot >= 3U * rank_num && flag_slot < 3U * rank_num + 64U;
        StoreFlag(Flag(workspace, uint32_t(flag_slot)), signal_ub, completion_source ? 1U : 0U);
    }
    // Entry/exit barriers protect reuse; reset weight flags even for empty ranks.
    if (tiling.with_weights) {
        StoreFlag(reinterpret_cast<__gm__ uint64_t*>(workspace + tiling.weight_local_done_offset_bytes +
                                                     uint64_t(core_id) * combine_control::kSlotBytes),
                  signal_ub, 0U);
        for (uint32_t peer = core_id; peer < rank_num; peer += combine_config::kAivCores) {
            StoreFlag(reinterpret_cast<__gm__ uint64_t*>(workspace + tiling.weight_done_offset_bytes +
                                                         uint64_t(peer) * combine_control::kSlotBytes),
                      signal_ub, 0U);
        }
        auto zero = signal_buffer.Get<float>();
        constexpr uint32_t elements = kSignalUbBytes / sizeof(float);
        Duplicate(zero, 0.0f, elements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        const uint32_t count = tiling.token_num * tiling.weight_topk;
        for (uint32_t begin = core_id * elements; begin < count; begin += combine_config::kAivCores * elements) {
            const uint32_t size = count - begin < elements ? count - begin : elements;
            NativeStore(reinterpret_cast<__gm__ float*>(workspace + tiling.weight_recv_offset_bytes) + begin, zero,
                        size);
        }
    }
    aclshmemx_barrier_all_vec();
}
__aicore__ inline void WaitGatherDone(GM_ADDR workspace, uint32_t rank_num, uint32_t local_rank_num,
                                      uint32_t global_chunk, uint32_t contributor_mask,
                                      LocalTensor<uint64_t> gather_flags_ub)
{
    // 没有远端贡献时无需读取 flag；本卡贡献已由同一 stream 的输入拷贝准备好。
    if (!contributor_mask) {
        return;
    }
    constexpr uint32_t words_per_slot = combine_control::kSlotBytes / sizeof(uint64_t);
    GlobalTensor<uint64_t> gather_flags_gm;
    gather_flags_gm.SetGlobalBuffer(Flag(workspace, 3U * rank_num + combine_control::kGatherDoneBase +
                                                        global_chunk * combine_control::kGatherDoneLanes));
    gather_flags_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    while (true) {
        // 每次轮询进行一次对齐的 GM->UB 搬运（8 个 rank 共 4 KiB）。
        // 只检查每个独立槽位的首个信号值，不读取其填充区作为标志。
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        DataCopy(gather_flags_ub, gather_flags_gm, local_rank_num * words_per_slot);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        bool all_ready = true;
        for (uint32_t local_rank_id = 0; local_rank_id < local_rank_num; ++local_rank_id) {
            if (contributor_mask & (1U << local_rank_id)) {
                all_ready = (gather_flags_ub.GetValue(local_rank_id * words_per_slot) != 0U) && all_ready;
            }
        }
        if (all_ready) {
            return;
        }
    }
}
__aicore__ inline uint32_t ReduceDoneSlot(const CombineTiling& tiling, uint32_t reducer)
{
    return 3U * tiling.rank_num + combine_control::kGatherDoneBase +
           tiling.total_chunk_num * combine_control::kGatherDoneLanes + reducer;
}
// One writer per 512B GM slot; no atomic updates or shared completion counter.
// Each reducer traverses all chunks in order, including chunks with no work.
__aicore__ inline uint32_t ReduceDoneEpoch(const CombineTiling& tiling, uint32_t global_chunk)
{
    // GM masks/trace retain [group, chunk] addressing; publication is chunk-major.
    return (global_chunk % tiling.chunks_per_group) * (tiling.rank_num / tiling.group_size) +
           global_chunk / tiling.chunks_per_group + 1U;
}
__aicore__ inline void PublishReduceDone(GM_ADDR workspace, const CombineTiling& tiling, uint32_t global_chunk,
                                         uint32_t reducer, LocalTensor<int32_t> signal_ub)
{
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    signal_ub.SetValue(0, static_cast<int32_t>(ReduceDoneEpoch(tiling, global_chunk)));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
    GlobalTensor<int32_t> flags_gm;
    flags_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(Flag(workspace, ReduceDoneSlot(tiling, reducer))));
    DataCopyPad(flags_gm, signal_ub, DataCopyExtParams(1, sizeof(int32_t), 0, 0, 0));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}
// Each strided 4B flag occupies one padded 32B UB block. Reuse copy scratch,
// not the 512B signal buffer: up to 56 reducers need 1792B of UB.
static_assert(combine_control::kReduceDoneSlots * 32U <= combine_config::kGatewayScratchBytes,
              "reducer flag snapshot must fit transient copy scratch");
__aicore__ inline bool ReduceDoneReady(GM_ADDR workspace, const CombineTiling& tiling, uint32_t global_chunk,
                                       uint32_t reduce_core_num, LocalTensor<int32_t> reduce_done_ub)
{
    GlobalTensor<int32_t> reduce_done_gm;
    reduce_done_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(Flag(workspace, ReduceDoneSlot(tiling, 0U))));
    reduce_done_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(reduce_done_ub, reduce_done_gm,
                DataCopyExtParams(static_cast<uint16_t>(reduce_core_num), sizeof(int32_t),
                                  combine_control::kSlotBytes - sizeof(int32_t), 0, 0),
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    const int32_t epoch = static_cast<int32_t>(ReduceDoneEpoch(tiling, global_chunk));
    for (uint32_t reducer = 0; reducer < reduce_core_num; ++reducer) {
        if (reduce_done_ub.GetValue(reducer * (32U / sizeof(int32_t))) < epoch) {
            return false;
        }
    }
    return true;
}
__aicore__ inline void WaitReduceDone(GM_ADDR workspace, const CombineTiling& tiling, uint32_t global_chunk,
                                      uint32_t reduce_core_num, LocalTensor<int32_t> reduce_done_ub)
{
    while (!ReduceDoneReady(workspace, tiling, global_chunk, reduce_core_num, reduce_done_ub)) {
    }
}
__aicore__ inline void AddBf16(__gm__ bfloat16_t* source_address, uint32_t element_num,
                               LocalTensor<bfloat16_t> bf16_input_ub, LocalTensor<float> fp32_input_ub,
                               LocalTensor<float> fp32_accumulator_ub)
{
    GlobalTensor<bfloat16_t> input_gm;
    input_gm.SetGlobalBuffer(source_address);
    DataCopy(bf16_input_ub, input_gm, element_num);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fp32_input_ub, bf16_input_ub, RoundMode::CAST_NONE, element_num);
    PipeBarrier<PIPE_V>();
    Add(fp32_accumulator_ub, fp32_accumulator_ub, fp32_input_ub, element_num);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}
// ranges/masks remain resident until gateway reduction finishes. The first
// 4 KiB of copy UB is transient: gather flag polling OR two forward task banks.
__aicore__ inline uint32_t GatewayMaskOffsetWords(const CombineTiling& tiling)
{
    return (tiling.rank_num * tiling.chunks_per_group * 2U + 7U) / 8U * 8U;
}
__aicore__ inline void LoadGatewayControls(LocalTensor<int32_t> controls_ub, GM_ADDR chunk_ranges, GM_ADDR chunk_masks,
                                           const CombineTiling& tiling)
{
    NativeLoad(controls_ub, reinterpret_cast<__gm__ int32_t*>(chunk_ranges),
               tiling.rank_num * tiling.chunks_per_group * 2U);
    NativeLoad(controls_ub[GatewayMaskOffsetWords(tiling)], reinterpret_cast<__gm__ int32_t*>(chunk_masks),
               tiling.total_chunk_num);
}
// Forward pages are relative to THIS core's task slice, not globally aligned
// 64-row pages. Only a predecessor row and token-tail lookahead may lie outside
// the owned slice. Lookahead loads one row rather than a neighbour core's page.
constexpr uint32_t kGatewayMetadataSlots = 2;
constexpr uint32_t kGatewayMetadataRows = 64;
constexpr uint32_t kGatewayMetadataWords = kGatewayMetadataRows * 6U;
static_assert(kGatewayMetadataWords * sizeof(int32_t) % 32U == 0U, "forward banks must be DMA aligned");
static_assert(kGatewayMetadataSlots * kGatewayMetadataWords * sizeof(int32_t) <= combine_config::kGatewayScratchBytes,
              "forward metadata must not overwrite resident controls");
static_assert(combine_control::kGatherDoneLanes * combine_control::kSlotBytes <= combine_config::kGatewayScratchBytes,
              "gather flag polling must not overwrite resident controls");
struct GatewayTaskCache {
    GlobalTensor<int32_t>& tasks_gm;
    LocalTensor<int32_t> tasks_ub;
    uint64_t begin_word, end_word, chunk_end_word;
    uint64_t page_begin[kGatewayMetadataSlots]{};
    uint32_t page_words[kGatewayMetadataSlots]{};
    bool load_pending[kGatewayMetadataSlots]{};
    uint32_t current_slot = 0;

    __aicore__ inline GatewayTaskCache(GlobalTensor<int32_t>& gm, LocalTensor<int32_t> ub, uint64_t begin, uint64_t end,
                                       uint64_t chunk_end)
        : tasks_gm(gm), tasks_ub(ub), begin_word(begin), end_word(end), chunk_end_word(chunk_end)
    {
    }

    __aicore__ inline bool Contains(uint32_t slot, uint64_t word) const
    {
        return word >= page_begin[slot] && word < page_begin[slot] + page_words[slot];
    }
    __aicore__ inline void Wait(uint32_t slot)
    {
        if (load_pending[slot]) {
            const auto event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_S>(event);
            load_pending[slot] = false;
        }
    }
    __aicore__ inline void Load(uint32_t slot, uint64_t word)
    {
        // Consume even an unused prefetch before reusing its event and bank.
        Wait(slot);
        if (word >= end_word) {
            page_begin[slot] = word / 6U * 6U;
            page_words[slot] =
                chunk_end_word - page_begin[slot] < 6U ? static_cast<uint32_t>(chunk_end_word - page_begin[slot]) : 6U;
        } else {
            page_begin[slot] = begin_word + (word - begin_word) / kGatewayMetadataWords * kGatewayMetadataWords;
            page_words[slot] = end_word - page_begin[slot] < kGatewayMetadataWords
                                   ? static_cast<uint32_t>(end_word - page_begin[slot])
                                   : kGatewayMetadataWords;
        }
        const auto event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
        // Scalar reads of the old page must finish before MTE2 overwrites it.
        SetFlag<HardEvent::S_MTE2>(event);
        WaitFlag<HardEvent::S_MTE2>(event);
        DataCopyPad(tasks_ub[slot * kGatewayMetadataWords], tasks_gm[page_begin[slot]],
                    DataCopyExtParams(1, page_words[slot] * sizeof(int32_t), 0, 0, 0),
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(event);
        load_pending[slot] = true;
    }
    __aicore__ inline uint32_t Select(uint64_t word)
    {
        if (!Contains(current_slot, word)) {
            current_slot ^= 1U;
            if (!Contains(current_slot, word)) {
                Load(current_slot, word);
            }
        }
        return current_slot;
    }
    __aicore__ inline void PrefetchNext(uint64_t word)
    {
        const uint32_t slot = Select(word);
        const uint64_t next_word = page_begin[slot] + page_words[slot];
        // Prefetch only owned rows. Tail lookahead remains a demand-only row
        // load, so small slices never fetch a neighbour core's entire page.
        if (next_word < end_word && !Contains(slot ^ 1U, next_word)) {
            Load(slot ^ 1U, next_word);
        }
    }
    __aicore__ inline int32_t GetValue(uint64_t word)
    {
        const uint32_t slot = Select(word);
        Wait(slot);
        return tasks_ub.GetValue(slot * kGatewayMetadataWords + word - page_begin[slot]);
    }
    __aicore__ inline void Finish()
    {
        for (uint32_t slot = 0; slot < kGatewayMetadataSlots; ++slot) {
            Wait(slot);
        }
    }
};
template <typename TaskReader>
__aicore__ inline bool TokenNeedsGatewayReduce(TaskReader& forward_list_gm, uint64_t source_task_base,
                                               uint32_t task_begin)
{
    // The validated planner emits either one primary-only placeholder (-1),
    // or secondary rows whose first contributor is nonnegative, even if local.
    return forward_list_gm.GetValue((source_task_base + task_begin) * 6U + 3U) >= 0;
}
template <typename TaskReader>
__aicore__ inline uint32_t AlignTaskToTokenBoundary(TaskReader& tasks, uint64_t source_task_base,
                                                    uint32_t chunk_task_begin, uint32_t task_index)
{
    if (task_index == chunk_task_begin) {
        return task_index;
    }
    // At a boundary the predecessor ends here; inside a run it ends after here.
    return tasks.GetValue((source_task_base + task_index - 1U) * 6U);
}
template <bool EnableTrace>
__aicore__ inline void ReduceGateway(GM_ADDR workspace, GlobalTensor<int32_t>& forward_list_gm,
                                     LocalTensor<int32_t> chunk_ranges_ub, const CombineTiling& tiling,
                                     uint32_t group_index, uint32_t chunk_index, uint32_t reduce_core_index,
                                     uint32_t reduce_core_num, LocalTensor<bfloat16_t> bf16_input_ub,
                                     LocalTensor<float> fp32_input_ub, LocalTensor<float> fp32_accumulator_ub,
                                     LocalTensor<bfloat16_t> bf16_output_ub, LocalTensor<int32_t> metadata_ub,
                                     Trace<EnableTrace>& trace_recorder)
{
    // 槽 A/B 分别使用 EVENT_ID1/2；仅在实际写回后产生 MTE3_V 事件。
    uint32_t output_slot = 0;
    bool output_store_pending[kReduceOutputSlots]{};
    // 仅归约本组源卡；此处必须与 CrossSend 的出向 Peer 映射一致。
    for (uint32_t source_index = 0; source_index < tiling.group_size; ++source_index) {
        const uint32_t source_rank_id = CombinePeer(tiling, group_index, source_index);
        const uint64_t range_index = (uint64_t(source_rank_id) * tiling.chunks_per_group + chunk_index) * 2U;
        const uint32_t chunk_task_begin = chunk_ranges_ub.GetValue(range_index);
        const uint32_t chunk_task_end = chunk_ranges_ub.GetValue(range_index + 1U);
        const uint32_t task_num = chunk_task_end - chunk_task_begin;
        const uint64_t source_task_base = uint64_t(source_rank_id) * tiling.forward_row_capacity;
        // Rotate the statically sliced partition across source ranks. This
        // keeps the existing disjoint ranges and token-boundary handling, but
        // spreads the q/q+1 slices caused by integer division over all reduce
        // cores instead of assigning the same remainder pattern to each one.
        const uint32_t partition_index = (reduce_core_index + source_index) % reduce_core_num;
        uint32_t task_index = chunk_task_begin + uint64_t(task_num) * partition_index / reduce_core_num;
        const uint32_t worker_task_end =
            chunk_task_begin + uint64_t(task_num) * (partition_index + 1U) / reduce_core_num;
        if (task_index == worker_task_end) {
            continue;
        }
        const uint32_t metadata_begin = task_index > chunk_task_begin ? task_index - 1U : task_index;
        GatewayTaskCache tasks{forward_list_gm, metadata_ub, (source_task_base + metadata_begin) * 6U,
                               (source_task_base + worker_task_end) * 6U, (source_task_base + chunk_task_end) * 6U};
        // 仅当 token 的首条任务落在本核原始分工区间内时，才由本核负责该 token。
        task_index = AlignTaskToTokenBoundary(tasks, source_task_base, chunk_task_begin, task_index);
        while (task_index < worker_task_end) {
            const uint32_t token_task_end = tasks.GetValue((source_task_base + task_index) * 6U);
            const int32_t first_contributor_rank = tasks.GetValue((source_task_base + task_index) * 6U + 3U);
            if (first_contributor_rank < 0) {
                // Direct-only tokens have no vector work; overlap the next
                // page load with subsequent scalar token processing instead.
                tasks.PrefetchNext((source_task_base + task_index) * 6U);
                task_index = token_task_end;
                continue;
            }
            const uint32_t primary_row = tasks.GetValue((source_task_base + task_index) * 6U + 2U);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayTokens, 1);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayRows, 1);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayReadBytes,
                               uint64_t(tiling.hidden_size) * 2U);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayWriteBytes,
                               uint64_t(tiling.hidden_size) * 2U);
            for (uint32_t hidden_offset = 0; hidden_offset < tiling.hidden_size; hidden_offset += kReduceTileElements) {
                const uint32_t element_num = tiling.hidden_size - hidden_offset < kReduceTileElements
                                                 ? tiling.hidden_size - hidden_offset
                                                 : kReduceTileElements;
                // 首份贡献直接初始化累加器，省去清零和首次加法。
                GlobalTensor<bfloat16_t> primary_input_gm;
                primary_input_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.expert_input_offset_bytes) +
                    uint64_t(primary_row) * tiling.hidden_size + hidden_offset);
                DataCopy(bf16_input_ub, primary_input_gm, element_num);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(fp32_accumulator_ub, bf16_input_ub, RoundMode::CAST_NONE, element_num);
                // Enqueue metadata AFTER this token's first payload DMA, so
                // the prefetch can overlap its Cast rather than delaying the
                // payload at the head of the shared MTE2 queue. Keep both
                // banks through contributor scans and hidden-tile replays.
                if (!hidden_offset) {
                    tasks.PrefetchNext((source_task_base + task_index) * 6U);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                for (uint32_t contributor_task_index = task_index; contributor_task_index < token_task_end;
                     ++contributor_task_index) {
                    // Reuse classification across hidden tiles; reduced runs contain only secondary rows.
                    const int32_t contributor_rank =
                        contributor_task_index == task_index
                            ? first_contributor_rank
                            : tasks.GetValue((source_task_base + contributor_task_index) * 6U + 3U);
                    if constexpr (EnableTrace) {
                        if (!hidden_offset) {
                            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayRows, 1);
                            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::GatewayReadBytes,
                                               uint64_t(tiling.hidden_size) * 2U);
                        }
                    }
                    const bool is_local_contribution = uint32_t(contributor_rank) == tiling.rank_id;
                    const uint32_t contributor_row = tasks.GetValue((source_task_base + contributor_task_index) * 6U +
                                                                    (is_local_contribution ? 4U : 5U));
                    AddBf16(reinterpret_cast<__gm__ bfloat16_t*>(workspace + (is_local_contribution
                                                                                  ? tiling.expert_input_offset_bytes
                                                                                  : tiling.gather_input_offset_bytes)) +
                                uint64_t(contributor_row) * tiling.hidden_size + hidden_offset,
                            element_num, bf16_input_ub, fp32_input_ub, fp32_accumulator_ub);
                }
                // server 内先用 FP32 完成累加，再将 partial 舍入为 BF16 后写回并发送。
                // 只等待即将复用的槽，另一个槽可继续写回；按实际 token/tile 写出次数轮转。
                const auto output_event_id = output_slot == 0U ? EVENT_ID1 : EVENT_ID2;
                auto bf16_output_slot_ub = bf16_output_ub[output_slot * kReduceTileElements];
                if (output_store_pending[output_slot]) {
                    WaitFlag<HardEvent::MTE3_V>(output_event_id);
                    output_store_pending[output_slot] = false;
                }
                Cast(bf16_output_slot_ub, fp32_accumulator_ub, RoundMode::CAST_RINT, element_num);
                // 当前 Cast 读完累加器后，下一条 token/tile 才能重新初始化它。
                PipeBarrier<PIPE_V>();
                GlobalTensor<bfloat16_t> partial_output_gm;
                partial_output_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.server_partial_offset_bytes) +
                    uint64_t(primary_row) * tiling.hidden_size + hidden_offset);
                SetFlag<HardEvent::V_MTE3>(output_event_id);
                WaitFlag<HardEvent::V_MTE3>(output_event_id);
                DataCopy(partial_output_gm, bf16_output_slot_ub, element_num);
                SetFlag<HardEvent::MTE3_V>(output_event_id);
                output_store_pending[output_slot] = true;
                output_slot ^= 1U;
            }
            task_index = token_task_end;
        }
        // A skipped/direct-only token may leave a prefetch unread. Drain it
        // before the next source or chunk reuses metadata UB/event IDs 4/5.
        tasks.Finish();
    }
    // 返回后会发布本 chunk 的 reduce_done；两个输出槽都必须排空。
    // 空任务核、只写出一条的核，不等待未产生的事件。
    for (uint32_t slot = 0; slot < kReduceOutputSlots; ++slot) {
        if (output_store_pending[slot]) {
            const auto output_event_id = slot == 0U ? EVENT_ID1 : EVENT_ID2;
            WaitFlag<HardEvent::MTE3_V>(output_event_id);
        }
    }
}
__aicore__ inline bool UseLocalMte(uint32_t rank_num)
{
    return rank_num >= 64U;
}
// Local/MESH copies use separate payload banks, never the gather metadata UB.
__aicore__ inline GM_ADDR LocalMteDestination(GM_ADDR address, uint32_t peer, uint32_t self)
{
    if (peer == self) {
        return address;
    }
    if (!(aclshmemi_get_state()->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) {
        aclshmemi_kernel_abort("PUT combine local MTE requires peer reachability: peer=%u\n", peer);
    }
    return reinterpret_cast<GM_ADDR>(aclshmem_ptr(address, peer));
}
struct LocalMteCopy {
    uint32_t slot = 0U;
    __aicore__ inline void Init()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
    }
    __aicore__ inline void Add(GM_ADDR destination, GM_ADDR source, uint32_t bytes, LocalTensor<uint8_t> copy_ub)
    {
        GlobalTensor<uint8_t> src, dst;
        src.SetGlobalBuffer(source);
        dst.SetGlobalBuffer(destination);
        for (uint32_t offset = 0; offset < bytes; offset += kLocalCopyTileBytes) {
            const uint32_t count = bytes - offset < kLocalCopyTileBytes ? bytes - offset : kLocalCopyTileBytes;
            const auto event = slot == 0U ? EVENT_ID2 : EVENT_ID4;
            auto bank = copy_ub[slot * kLocalCopyTileBytes];
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopyPad(bank, src[offset], DataCopyExtParams(1, count, 0, 0, 0),
                        DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            DataCopyPad(dst[offset], bank, DataCopyExtParams(1, count, 0, 0, 0));
            SetFlag<HardEvent::MTE3_MTE2>(event);
            slot ^= 1U;
        }
    }
    __aicore__ inline void Finish()
    {
        // Consume both events, even for an empty stream, before reusing banks.
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        // Join the waits themselves before reinitializing these event IDs.
        // MTE2 cannot reach this point until both banks' MTE3 stores finish.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }
};
__aicore__ inline void PublishLocalMteFlag(__gm__ uint64_t* flag, uint32_t peer, uint32_t self,
                                           LocalTensor<uint64_t> signal_ub)
{
    // A flag is not an ordered UDMA WQE: explicitly complete all payload MTE
    // stores first. NativeStore then waits for this 8-byte flag itself.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    auto* remote = LocalMteDestination(reinterpret_cast<GM_ADDR>(flag), peer, self);
    StoreFlag(reinterpret_cast<__gm__ uint64_t*>(remote), signal_ub, 1U);
}
template <bool EnableTrace>
__aicore__ inline void SendPartial(GM_ADDR workspace, GlobalTensor<int32_t>& forward_list_gm,
                                   const CombineTiling& tiling, uint32_t destination_rank, uint32_t task_begin,
                                   uint32_t task_end, LocalTensor<uint8_t> copy_ub, __ubuf__ uint8_t* wqe_scratch_ub,
                                   Trace<EnableTrace>& trace_recorder, uint32_t trace_slot, uint32_t chunk_slot = ~0U,
                                   PartialClass partial_class = PartialClass::All, uint32_t jetty_id = 0U,
                                   bool cross_put = false)
{
    const uint32_t server_num = (tiling.rank_num + 7U) / 8U;
    const uint64_t source_task_base = uint64_t(destination_rank) * tiling.forward_row_capacity;
    const bool remote_destination = destination_rank != tiling.rank_id;
    const bool use_mte = UseLocalMte(tiling.rank_num) || !remote_destination;
    uint32_t remaining_rows = 0, pending_wqes = 0;
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer(submit_state);
    aclshmemx_submit_t submit(submit_state);
    if (!use_mte) {
        // Close every filtered stream's tail, including a partial QP0 batch.
        for (uint32_t scan = task_begin; scan < task_end;) {
            const bool reduced = TokenNeedsGatewayReduce(forward_list_gm, source_task_base, scan);
            if ((partial_class != PartialClass::DirectOnly || !reduced) &&
                (partial_class != PartialClass::ReducedOnly || reduced)) {
                ++remaining_rows;
            }
            scan = forward_list_gm.GetValue((source_task_base + scan) * 6U);
        }
    }
    uint32_t rows = 0;
    LocalMteCopy copy;
    if (use_mte) {
        copy.Init();
    }
    for (uint32_t task_index = task_begin; task_index < task_end;) {
        const uint32_t token_end = forward_list_gm.GetValue((source_task_base + task_index) * 6U);
        const bool needs_reduce = TokenNeedsGatewayReduce(forward_list_gm, source_task_base, task_index);
        if ((partial_class == PartialClass::DirectOnly && needs_reduce) ||
            (partial_class == PartialClass::ReducedOnly && !needs_reduce)) {
            task_index = token_end;
            continue;
        }
        const uint32_t token_id = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 1U);
        const uint32_t primary_row = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 2U);
        auto* source_address = workspace +
                               (needs_reduce ? tiling.server_partial_offset_bytes : tiling.expert_input_offset_bytes) +
                               uint64_t(primary_row) * tiling.hidden_size * 2U;
        auto* destination_address = workspace + tiling.returned_partial_offset_bytes +
                                    (uint64_t(token_id) * server_num + tiling.rank_id / 8U) * tiling.hidden_size * 2U;
        if (use_mte) {
            copy.Add(LocalMteDestination(destination_address, destination_rank, tiling.rank_id), source_address,
                     tiling.hidden_size * 2U, copy_ub);
        } else {
            const bool close_batch = --remaining_rows == 0U || pending_wqes + 1U == kWqeBatchSize;
            if (close_batch) {
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoConfig>(
                    destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, destination_rank, 0U,
                    EVENT_ID0, submit);
                pending_wqes = 0;
            } else {
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoNoCqeConfig>(
                    destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, destination_rank, 0U,
                    EVENT_ID0, defer);
                ++pending_wqes;
            }
        }
        if constexpr (EnableTrace) {
            ++rows;
        }
        task_index = token_end;
    }
    if (use_mte) {
        copy.Finish();
    }
    if constexpr (EnableTrace) {
        for (uint32_t index = 0; index < 2U; ++index) {
            const uint32_t slot = index == 0U ? trace_slot : chunk_slot;
            if (slot == ~0U) {
                continue;
            }
            trace_recorder.Add(slot, combine_profile::Rows, rows);
            trace_recorder.Add(slot, remote_destination ? combine_profile::Bytes0 : combine_profile::CopyBytes,
                               uint64_t(rows) * tiling.hidden_size * 2U);
            if (!use_mte) {
                trace_recorder.Add(slot, combine_profile::Wqes0, rows);
            }
        }
    }
}
// Completion slots are outside legacy local-return/gather/reduction control.
// Order peers by (producer group, rank), remove the sender's MESH ranks,
// then round-robin the compacted list. Only call for cross peers.
// Carry the RR offset across groups. EP64/16 alternates the final eight peers
// between owner halves per chunk instead of assigning them permanently.
__aicore__ inline uint32_t CrossPeerOwner(uint32_t sender_rank, uint32_t peer, uint32_t world, uint32_t owners,
                                          uint32_t chunk = 0U)
{
    const uint32_t width = first_hit_schedule::Width(world);
    const uint32_t peer_order =
        first_hit_schedule::Group(sender_rank, world, peer) * width + first_hit_schedule::Index(peer, world);
    uint32_t peer_index = peer_order;
    const uint32_t mesh_begin = sender_rank / 8U * 8U;
    for (uint32_t local = mesh_begin; local < mesh_begin + 8U; ++local) {
        const uint32_t local_order =
            first_hit_schedule::Group(sender_rank, world, local) * width + first_hit_schedule::Index(local, world);
        if (local_order < peer_order) {
            --peer_index;
        }
    }
    // Rotate within each remote MESH, preserving the owner-half load split.
    peer_index = peer_index / 8U * 8U + (peer_index % 8U + 8U - sender_rank % 8U) % 8U;
    // EP64/16: first 48 peers stay fixed (three/core); the final eight use
    // owners 0..7 for even chunks and 8..15 for odd chunks, in every group mode.
    if (world == 64U && owners == 16U && peer_index >= 48U) {
        return peer_index % 8U + (chunk % 2U) * 8U;
    }
    return (peer_index + (owners == 16U ? 8U : 0U)) % owners;
}
__aicore__ inline bool CrossPeerAssigned(const CombineTiling& tiling, uint32_t peer, uint32_t owner)
{
    return CrossPeerOwner(tiling.rank_id, peer, tiling.rank_num, tiling.num_cross_put_cores, 0U) == owner ||
           (tiling.chunks_per_group > tiling.first_chunk_splits &&
            CrossPeerOwner(tiling.rank_id, peer, tiling.rank_num, tiling.num_cross_put_cores, 1U) == owner);
}
__aicore__ inline uint32_t CrossCompletionSlot(const CombineTiling& tiling, uint32_t gateway, uint32_t chunk,
                                               uint32_t lane)
{
    return 3U * tiling.rank_num + combine_control::kGatherDoneBase +
           tiling.total_chunk_num * combine_control::kGatherDoneLanes + combine_control::kReduceDoneSlots +
           (gateway * tiling.chunks_per_group + chunk) * 2U + lane;
}
// All startup subchunks retain logical chunk 0's owner. Later logical chunk
// parity and SO phase/token boundaries are unchanged by the extra pipeline step.
__aicore__ inline uint32_t LogicalChunk(const CombineTiling& tiling, uint32_t chunk)
{
    return chunk < tiling.first_chunk_splits ? 0U : chunk - tiling.first_chunk_splits + 1U;
}
__aicore__ inline uint64_t ChunkTokenEnd(const CombineTiling& tiling, uint32_t chunk)
{
    if (chunk < tiling.first_chunk_splits) {
        const uint32_t first = tiling.token_num < tiling.chunk_token_num ? tiling.token_num : tiling.chunk_token_num;
        return (uint64_t(chunk + 1U) * first + tiling.first_chunk_splits - 1U) / tiling.first_chunk_splits;
    }
    return uint64_t(LogicalChunk(tiling, chunk) + 1U) * tiling.chunk_token_num;
}
__aicore__ inline uint32_t CrossChunkOwner(const CombineTiling& tiling, uint32_t sender, uint32_t peer, uint32_t chunk)
{
    return CrossPeerOwner(sender, peer, tiling.rank_num, tiling.num_cross_put_cores, LogicalChunk(tiling, chunk));
}
// Split the ORIGINAL logical chunks into min(num_so, C) nonempty windows.
// Only map the resulting boundary back to the flattened physical chunk index.
__aicore__ inline uint32_t SoWindowEnd(const CombineTiling& tiling, uint32_t chunk)
{
    const uint32_t split = tiling.first_chunk_splits - 1U;
    const uint32_t chunks = tiling.chunks_per_group - split;
    const uint32_t count = tiling.num_so < chunks ? tiling.num_so : chunks;
    // keep the original first half;split only the second half into two windows.
    // All send-side SO fences and source-side waits use this shared boundary.
    if (count == 3U && chunks >= 4U) {
        const uint32_t first_end = (chunks + 1U) / 2U;
        const uint32_t second_end = first_end + (chunks - first_end + 1U) / 2U;
        const uint32_t logical_chunk = LogicalChunk(tiling, chunk);
        const uint32_t end = logical_chunk < first_end ? first_end : (logical_chunk < second_end ? second_end : chunks);
        return end - 1U + split;
    }
    const uint32_t window = uint64_t(LogicalChunk(tiling, chunk)) * count / chunks;
    return (uint64_t(window + 1U) * chunks + count - 1U) / count - 1U + split;
}
__aicore__ inline bool CrossSoRequired(const CombineTiling& tiling, uint32_t sender, uint32_t peer, uint32_t chunk)
{
    const uint32_t end = SoWindowEnd(tiling, chunk);
    if (chunk == end) {
        return true;
    }
    // EP64/16 tail peers alternate QP owners. Each owner must fence its own
    // stream at its last chunk in the window; one owner's SO cannot fence both.
    return chunk + 1U == end &&
           CrossChunkOwner(tiling, sender, peer, chunk) != CrossChunkOwner(tiling, sender, peer, end);
}
__aicore__ inline uint32_t SelfCopyDoneSlot(const CombineTiling& tiling, uint32_t core)
{
    return CrossCompletionSlot(tiling, tiling.rank_num, 0U, 0U) + core;
}
// Poll both lanes in the same snapshot; neither lane blocks observation of the
// other. Use dedicated signal UB, never the final-reduce input or mask banks.
__aicore__ inline void WaitCompletionFlags(GM_ADDR workspace, uint32_t slot, uint32_t lanes,
                                           LocalTensor<uint64_t> signal_ub, uint64_t expected = 1U)
{
    GlobalTensor<uint64_t> flags_gm;
    flags_gm.SetGlobalBuffer(Flag(workspace, slot));
    flags_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    uint32_t pending = (1U << lanes) - 1U;
    while (pending) {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
        DataCopyPad(signal_ub, flags_gm,
                    DataCopyExtParams(lanes, sizeof(uint64_t), combine_control::kSlotBytes - sizeof(uint64_t), 0, 0),
                    DataCopyPadExtParams<uint64_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            if (signal_ub.GetValue(lane * (32U / sizeof(uint64_t))) >= expected) {
                pending &= ~(1U << lane);
            }
        }
    }
}
// Only notification slots actually published for this window may be polled.
__aicore__ inline void WaitGatewayWindow(GM_ADDR workspace, const CombineTiling& tiling, uint32_t gateway,
                                         uint32_t chunk, LocalTensor<uint64_t> signal_ub)
{
    const uint32_t end = SoWindowEnd(tiling, chunk);
    const bool remote = gateway / 8U != tiling.rank_id / 8U;
    if (remote && end > 0U && SoWindowEnd(tiling, end - 1U) == end &&
        CrossSoRequired(tiling, gateway, tiling.rank_id, end - 1U)) {
        WaitCompletionFlags(workspace, CrossCompletionSlot(tiling, gateway, end - 1U, 0U), 2U, signal_ub);
    }
    WaitCompletionFlags(workspace, CrossCompletionSlot(tiling, gateway, end, 0U), remote ? 2U : 1U, signal_ub);
}
// All cores call once per window, including cores without flags or tokens.
// Each gateway is checked by exactly one core, then SyncAll hands readiness
// to every final-reduce consumer on this rank.
template <bool EnableTrace>
__aicore__ inline void WaitSourceChunk(GM_ADDR workspace, const CombineTiling& tiling, uint32_t chunk, uint32_t core_id,
                                       LocalTensor<uint64_t> signal_ub, Trace<EnableTrace>& trace_recorder)
{
    const uint32_t window_end = SoWindowEnd(tiling, chunk);
    trace_recorder.Reset(combine_profile::ChunkSlot, combine_profile::SourceChunk);
    trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::ChunkIndex, window_end);
    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::Begin);
    for (uint32_t gateway = core_id; gateway < tiling.rank_num; gateway += combine_config::kAivCores) {
        if (gateway == tiling.rank_id) {
            WaitCompletionFlags(workspace, SelfCopyDoneSlot(tiling, tiling.rank_id % 8U), 1U, signal_ub,
                                window_end + 1U);
        } else {
            WaitGatewayWindow(workspace, tiling, gateway, chunk, signal_ub);
        }
    }
    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::SourceFlagsReady);
    AscendC::SyncAll<true>();
    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::End);
    trace_recorder.Save(combine_profile::ChunkSlot, trace_recorder.ChunkRecordIndex(window_end, 2U));
}
__aicore__ inline void DrainIncomingCompletions(GM_ADDR workspace, const CombineTiling& tiling, uint32_t poller_index,
                                                uint32_t poller_num, LocalTensor<uint64_t> signal_ub)
{
    for (uint32_t gateway = poller_index; gateway < tiling.rank_num; gateway += poller_num) {
        if (gateway == tiling.rank_id) {
            continue;
        }
        // Include windows not visited by this core's source token loop, and
        // both parity owners where a window spans their independent SQs.
        for (uint32_t chunk = 0; chunk < tiling.chunks_per_group; chunk = SoWindowEnd(tiling, chunk) + 1U) {
            WaitGatewayWindow(workspace, tiling, gateway, chunk, signal_ub);
        }
    }
}
__aicore__ inline uint32_t CrossLogicalOwner(const CombineTiling& tiling, uint32_t core_id)
{
    return core_id - 8U;
}
__aicore__ inline bool IsCrossPhysicalCore(const CombineTiling& tiling, uint32_t core_id)
{
    return tiling.rank_num > 8U && core_id >= 8U && core_id < 8U + tiling.num_cross_put_cores;
}
// Gateway reducers follow the contiguous cross pool.
__aicore__ inline uint32_t ReduceLogicalIndex(const CombineTiling& tiling, uint32_t core_id)
{
    const uint32_t cross_count = tiling.rank_num > 8U ? tiling.num_cross_put_cores : 0U;
    return core_id - 8U - cross_count;
}

struct CrossQuietPlan {
    uint32_t count = 0;
    uint32_t peers[128];
};

__aicore__ inline uint32_t CrossQuietPeerCount(const CombineTiling& tiling, const CrossQuietPlan& plan)
{
    // CrossSend verifies that every peer aliases the same physical SQ per lane.
    // Each lane is independent; quiet on its representative drains the whole SQ.
    return tiling.shared_jetty && plan.count != 0U ? 1U : plan.count;
}

__aicore__ inline void QuietCrossServerQps(const CombineTiling& tiling, uint32_t core_id, const CrossQuietPlan& plan)
{
    const uint32_t owner = CrossLogicalOwner(tiling, core_id);
    const uint32_t count = CrossQuietPeerCount(tiling, plan);
    for (uint32_t index = 0; index < count; ++index) {
        aclshmemx_udma_qp_quiet(plan.peers[index], 2U * owner);
        aclshmemx_udma_qp_quiet(plan.peers[index], 2U * owner + 1U);
    }
}
// Combine-only mixed-endpoint batching. The public SHMEM aggregate API has a
// same-PE contract; these internal primitives encode the remote segment/EID in
// each WQE. CrossSend validates a common physical SQ before using them across PEs.
// Payload WQEs are unsignaled. Only notification-window SO fences request CQEs.
// Each physical QP has its own batch; rank/QP rotation does not flush it.
struct CrossPutBatch {
    aclshmemx_submit_state_t state[2]{};
    uint32_t peer[2]{};
    uint32_t so_reserve = 1;

    __aicore__ inline void FlushLane(uint32_t lane, uint32_t owner, __ubuf__ uint8_t* scratch)
    {
        const uint32_t count = state[lane].pending_count;
        if (!count) {
            return;
        }
        auto* info = aclshmemi_udma_qp_info_fetch();
        auto* ctx = aclshmemi_udma_get_qp_ctx(info, peer[lane], 2U * owner + lane);
        auto* queue = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
        const uint32_t depth = shm::UDMA_SQ_BASKBLK_CNT;
        const uint32_t head = queue->sq_head;
        if (ctx->depth != depth) {
            aclshmemi_kernel_abort("PUT combine cross SQ depth mismatch: owner=%u lane=%u depth=%u expected=%u\n",
                                   owner, lane, ctx->depth, depth);
        }
        // Keep the staged WQEs in UB while reclaiming earlier window SOs.
        // Credit-sufficient batches do not poll CQ or wait for completion.
        const uint64_t required = uint64_t(count) + so_reserve + ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD;
        if (uint64_t(uint32_t(head - queue->sq_tail)) + required >= depth) {
            if (queue->cqe_cnt != queue->cq_tail) {
                aclshmemx_udma_qp_quiet(peer[lane], 2U * owner + lane);
            }
            if (uint64_t(uint32_t(head - queue->sq_tail)) + required >= depth) {
                aclshmemi_kernel_abort(
                    "PUT combine cross SQ has no reclaimable credit: owner=%u lane=%u head=%u tail=%u pending=%u "
                    "so_reserve=%u cqe=%u cq_tail=%u\n",
                    owner, lane, head, queue->sq_tail, count, so_reserve, queue->cqe_cnt, queue->cq_tail);
            }
        }
        const uint32_t slot = head % depth;
        const uint32_t first = count < depth - slot ? count : depth - slot;
        LocalTensor<uint8_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
        ub.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch + lane * kWqeBatchScratchBytes);
        aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(slot) * kWqeBytes),
                                        ub, first * kWqeBytes, EVENT_ID0);
        if (first != count) {
            ub.address_.bufferAddr += first * kWqeBytes;
            aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), ub,
                                            (count - first) * kWqeBytes, EVENT_ID0);
        }
        // Both wrap segments are visible before the one SQ doorbell update.
        aclshmemi_udma_post_send_update_info(head + count, ctx, queue);
        // These are private raw-stage states, not public aggregate submit actions.
        state[lane] = {};
        // Deliberately do not increment queue->cqe_cnt.
    }

    __aicore__ inline void Flush(uint32_t owner, uint32_t row_bytes, __ubuf__ uint8_t* scratch)
    {
        FlushLane(0U, owner, scratch);
        FlushLane(1U, owner, scratch);
    }

    __aicore__ inline void Add(GM_ADDR dst, GM_ADDR src, uint32_t target, uint32_t lane, uint32_t owner,
                               uint32_t row_bytes, __ubuf__ uint8_t* scratch)
    {
        auto* remote = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(dst, target));
        aclshmemi_udma_stage_send_wqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kPayloadRoNoCqeConfig>(
            remote, src, target, 2U * owner + lane, row_bytes, scratch + lane * kWqeBatchScratchBytes, state[lane]);
        peer[lane] = target;
        if (state[lane].pending_count == 32U) {
            FlushLane(lane, owner, scratch);
        }
    }
};

// Rank-local cross-only barrier. Each owner publishes a monotonic group epoch
// in its own 512B slot; a faster owner may publish a later epoch before a slow
// observer reads it. Prepare resets all slots before the next invocation.
__aicore__ inline void CrossGroupBarrier(GM_ADDR workspace, const CombineTiling& tiling, uint32_t owner, uint32_t epoch,
                                         LocalTensor<uint64_t> signal_ub)
{
    const uint32_t base = 3U * tiling.rank_num + combine_control::kCrossGroupDoneBase;
    StoreFlag(Flag(workspace, base + owner), signal_ub, epoch);
    for (uint32_t other = 0; other < tiling.num_cross_put_cores; ++other) {
        if (other == owner) {
            continue;
        }
        GlobalTensor<uint64_t> done_gm;
        done_gm.SetGlobalBuffer(Flag(workspace, base + other), 1U);
        done_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        do {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(signal_ub, done_gm, DataCopyExtParams(1, sizeof(uint64_t), 0, 0, 0),
                        DataCopyPadExtParams<uint64_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        } while (signal_ub.GetValue(0) < epoch);
    }
}

template <bool EnableTrace>
__aicore__ inline void CrossSend(GM_ADDR workspace, GlobalTensor<int32_t>& forward_list_gm,
                                 GlobalTensor<int32_t>& chunk_ranges_gm, const CombineTiling& tiling, uint32_t core_id,
                                 uint32_t reduce_core_num, LocalTensor<uint8_t> copy_ub,
                                 LocalTensor<uint64_t> signal_ub, LocalTensor<int32_t> reduce_done_ub,
                                 __ubuf__ uint8_t* wqe_scratch_ub, Trace<EnableTrace>& trace_recorder,
                                 CrossQuietPlan& quiet_plan)
{
    const uint32_t owner = CrossLogicalOwner(tiling, core_id);
    const uint32_t server_num = (tiling.rank_num + 7U) / 8U;
    const uint32_t row_bytes = tiling.hidden_size * 2U;
    auto* info = aclshmemi_udma_qp_info_fetch();
    uint32_t peers[128], peer_count = 0;
    uint32_t task_cursor[2][128], task_end[128], token_ordinal[2][128];
    // A matching state alone is insufficient: all mixed-PE WQEs must land on
    // the same SQ with the same doorbell and layout, separately for each uDie.
    for (uint32_t peer = 0; peer < tiling.rank_num; ++peer) {
        if (peer / 8U == tiling.rank_id / 8U || !CrossPeerAssigned(tiling, peer, owner)) {
            continue;
        }
        for (uint32_t lane = 0; lane < 2U; ++lane) {
            auto* ctx = aclshmemi_udma_get_qp_ctx(info, peer, 2U * owner + lane);
            auto* first = aclshmemi_udma_get_qp_ctx(info, peer_count ? peers[0] : peer, 2U * owner + lane);
            if (!ctx->state_addr ||
                (tiling.shared_jetty &&
                 (ctx->state_addr != first->state_addr || ctx->buf_addr != first->buf_addr ||
                  ctx->db_addr != first->db_addr || ctx->depth != first->depth || ctx->wqe_size != first->wqe_size))) {
                aclshmemi_kernel_abort(
                    "PUT combine cross batch requires one physical SQ per owner/lane: owner=%u lane=%u peer=%u\n",
                    owner, lane, peer);
            }
        }
        auto* lane0 = aclshmemi_udma_get_qp_ctx(info, peer, 2U * owner);
        auto* lane1 = aclshmemi_udma_get_qp_ctx(info, peer, 2U * owner + 1U);
        if (lane0->state_addr == lane1->state_addr) {
            aclshmemi_kernel_abort("PUT combine cross lanes share queue state: owner=%u peer=%u\n", owner, peer);
        }
        peers[peer_count++] = peer;
    }
    quiet_plan.count = peer_count;
    const uint32_t all_peer_count = quiet_plan.count;
    auto& all_peers = quiet_plan.peers;
    for (uint32_t index = 0; index < peer_count; ++index) {
        all_peers[index] = peers[index];
    }
    CrossPutBatch batch;
    // Reserve every SO that can share this SQ, including alternating owners'
    // tail slots. FlushLane reclaims completed windows only under SQ pressure.
    batch.so_reserve = 0U;
    for (uint32_t index = 0; index < all_peer_count; ++index) {
        uint32_t peer_so_count = 0U;
        for (uint32_t chunk = 0; chunk < tiling.chunks_per_group; ++chunk) {
            if (CrossChunkOwner(tiling, tiling.rank_id, all_peers[index], chunk) == owner &&
                CrossSoRequired(tiling, tiling.rank_id, all_peers[index], chunk)) {
                ++peer_so_count;
            }
        }
        if (tiling.shared_jetty) {
            batch.so_reserve += peer_so_count;
        } else if (peer_so_count > batch.so_reserve) {
            batch.so_reserve = peer_so_count;
        }
    }
    const uint32_t group_count = tiling.rank_num / tiling.group_size;
    for (uint32_t chunk = 0; chunk < tiling.chunks_per_group; ++chunk) {
        for (uint32_t send_group = 0; send_group < group_count; ++send_group) {
            peer_count = 0;
            for (uint32_t index = 0; index < all_peer_count; ++index) {
                const uint32_t peer = all_peers[index];
                if (CombineGroup(tiling.rank_id, tiling.rank_num, peer, tiling.group_size) == send_group) {
                    peers[peer_count++] = peer;
                }
            }
            if (peer_count) {
                // Conservative group/chunk admission, including direct and empty peers.
                // No cross core can admit its next chunk until all its current ranks finish.
                bool group_ready = false;
                for (uint32_t index = 0; index < peer_count; ++index) {
                    const uint32_t peer = peers[index];
                    const uint32_t group = CombineGroup(tiling.rank_id, tiling.rank_num, peer, tiling.group_size);
                    const bool assigned = CrossChunkOwner(tiling, tiling.rank_id, peer, chunk) == owner;
                    if (tiling.sync_reduce_cross && assigned && !group_ready) {
                        WaitReduceDone(workspace, tiling, group * tiling.chunks_per_group + chunk, reduce_core_num,
                                       reduce_done_ub);
                        group_ready = true;
                    }
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
                    DataCopyPad(reduce_done_ub,
                                chunk_ranges_gm[(uint64_t(peer) * tiling.chunks_per_group + chunk) * 2U],
                                DataCopyExtParams(1, 2U * sizeof(int32_t), 0, 0, 0),
                                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
                    task_cursor[0][index] = task_cursor[1][index] = reduce_done_ub.GetValue(0);
                    task_end[index] = reduce_done_ub.GetValue(1);
                    // Inactive parity retains its idle record and group barrier timing,
                    // but publishes no payload or SO for this chunk.
                    if (!assigned) {
                        task_cursor[0][index] = task_cursor[1][index] = task_end[index];
                    }
                    token_ordinal[0][index] = token_ordinal[1][index] = 0;
                    trace_recorder.Reset(combine_profile::PeerSlotBase, combine_profile::CrossReturn, group);
                    trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Peer, peer);
                    trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::SequenceIndex, owner);
                    trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::ChunkIndex, chunk);
                    trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::CrossChunkAssigned,
                                       assigned ? 1U : 0U);
                    trace_recorder.Save(combine_profile::PeerSlotBase,
                                        TraceBaseRecords(tiling) +
                                            uint64_t(tiling.total_chunk_num) * combine_profile::kChunkRecords +
                                            uint64_t(peer) * tiling.chunks_per_group + chunk);
                }
                uint32_t remaining = peer_count * 2U;
                bool first_round = true;
                while (remaining) {
                    // QP-major round robin: (QP0, every rank), then (QP1, every rank).
                    // Each pair owns its cursor and counts only its own actual tokens.
                    for (uint32_t stream = 0; stream < peer_count * 2U; ++stream) {
                        const uint32_t lane = stream / peer_count;
                        const uint32_t index = stream % peer_count;
                        if (!first_round && task_cursor[lane][index] == task_end[index]) {
                            continue;
                        }
                        const uint32_t peer = peers[index];
                        const uint64_t record = TraceBaseRecords(tiling) +
                                                uint64_t(tiling.total_chunk_num) * combine_profile::kChunkRecords +
                                                uint64_t(peer) * tiling.chunks_per_group + chunk;
                        trace_recorder.Load(combine_profile::PeerSlotBase, record);
                        if (tiling.sync_reduce_cross && first_round && lane == 0U) {
                            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::CrossReduceReady);
                        }
                        trace_recorder.First(combine_profile::PeerSlotBase, combine_profile::Begin);
                        uint64_t slice_begin = 0;
                        if constexpr (EnableTrace) {
                            slice_begin = GetSystemCycle();
                        }
                        const bool empty_chunk =
                            first_round && lane == 0U && task_cursor[lane][index] == task_end[index];
                        uint32_t sent = 0, direct_rows = 0, reduced_rows = 0;
                        const uint64_t base = uint64_t(peer) * tiling.forward_row_capacity;
                        while (sent < tiling.cross_tokens_per_turn && task_cursor[lane][index] < task_end[index]) {
                            const uint32_t begin = task_cursor[lane][index];
                            const uint32_t end = forward_list_gm.GetValue((base + begin) * 6U);
                            // Use the actual distinct-token ordinal, NOT sparse token IDs.
                            // Both cursors scan complete tokens; skipped tokens do not
                            // consume this QP/rank's quantum or generate a WQE.
                            const uint32_t token_lane = token_ordinal[lane][index]++ % 4U == 3U ? 1U : 0U;
                            task_cursor[lane][index] = end;
                            if (token_lane != lane) {
                                continue;
                            }
                            const uint32_t token = forward_list_gm.GetValue((base + begin) * 6U + 1U);
                            const uint32_t primary = forward_list_gm.GetValue((base + begin) * 6U + 2U);
                            const bool reduced = TokenNeedsGatewayReduce(forward_list_gm, base, begin);
                            auto* src =
                                workspace +
                                (reduced ? tiling.server_partial_offset_bytes : tiling.expert_input_offset_bytes) +
                                uint64_t(primary) * row_bytes;
                            auto* dst = workspace + tiling.returned_partial_offset_bytes +
                                        (uint64_t(token) * server_num + tiling.rank_id / 8U) * row_bytes;
                            if (reduced) {
                                if (!reduced_rows++) {
                                    trace_recorder.First(combine_profile::PeerSlotBase,
                                                         combine_profile::CrossReducedFirstIssue);
                                }
                            } else {
                                if (!direct_rows++) {
                                    trace_recorder.First(combine_profile::PeerSlotBase,
                                                         combine_profile::CrossDirectFirstIssue);
                                }
                            }
                            batch.Add(dst, src, peer, lane, owner, row_bytes, wqe_scratch_ub);
                            ++sent;
                        }
                        // Nonshared mode has distinct physical SQs and cannot mix endpoints.
                        if (!tiling.shared_jetty) {
                            batch.Flush(owner, row_bytes, wqe_scratch_ub);
                        }
                        if (sent || empty_chunk) {
                            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::End);
                            if constexpr (EnableTrace) {
                                trace_recorder.Add(
                                    combine_profile::PeerSlotBase, combine_profile::CrossIssueCycles,
                                    trace_recorder.Get(combine_profile::PeerSlotBase, combine_profile::End) -
                                        slice_begin);
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::CrossSliceCount, 1U);
                                if (sent) {
                                    trace_recorder.Add(combine_profile::PeerSlotBase,
                                                       combine_profile::CrossQp0SliceCount + lane, 1U);
                                }
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::CrossDirectRows,
                                                   direct_rows);
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::CrossReducedRows,
                                                   reduced_rows);
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::Rows, sent);
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::Wqes0 + lane, sent);
                                trace_recorder.Add(combine_profile::PeerSlotBase, combine_profile::Bytes0 + lane,
                                                   uint64_t(sent) * row_bytes);
                            }
                        }
                        trace_recorder.Save(combine_profile::PeerSlotBase, record);
                        if (task_cursor[lane][index] == task_end[index]) {
                            --remaining;
                        }
                    }
                    first_round = false;
                }
                // Flush BOTH lane tails before admitting another chunk or emitting SOs.
                batch.Flush(owner, row_bytes, wqe_scratch_ub);
                uint64_t chunk_submit = 0;
                if constexpr (EnableTrace) {
                    chunk_submit = GetSystemCycle();
                }
                for (uint32_t index = 0; index < peer_count; ++index) {
                    const uint32_t peer = peers[index];
                    const uint64_t record = TraceBaseRecords(tiling) +
                                            uint64_t(tiling.total_chunk_num) * combine_profile::kChunkRecords +
                                            uint64_t(peer) * tiling.chunks_per_group + chunk;
                    trace_recorder.Load(combine_profile::PeerSlotBase, record);
                    trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::CrossChunkSubmit, chunk_submit);
                    // Per-peer dual-QP SO at this owner's window tail, even if empty.
                    if (CrossChunkOwner(tiling, tiling.rank_id, peer, chunk) == owner &&
                        CrossSoRequired(tiling, tiling.rank_id, peer, chunk)) {
                        for (uint32_t lane = 0; lane < 2U; ++lane) {
                            aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                                Flag(workspace, CrossCompletionSlot(tiling, tiling.rank_id, chunk, lane)),
                                Flag(workspace, 3U * tiling.rank_num + core_id),
                                reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, peer, 2U * owner + lane,
                                EVENT_ID0);
                            trace_recorder.Mark(combine_profile::PeerSlotBase,
                                                lane == 0U ? combine_profile::CrossSo0 : combine_profile::CrossSo1);
                        }
                        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::CrossFlagOrder, 1U);
                    }
                    trace_recorder.Save(combine_profile::PeerSlotBase, record);
                }
            }
            // Empty owners participate as well. No local/gateway-reduce cores join.
            // Submission barrier only; no unconditional CQ drain here.
            if (send_group + 1U < group_count) {
                uint64_t barrier_begin = 0;
                if constexpr (EnableTrace) {
                    barrier_begin = GetSystemCycle();
                }
                CrossGroupBarrier(workspace, tiling, owner, chunk * group_count + send_group + 1U, signal_ub);
                if constexpr (EnableTrace) {
                    const uint64_t barrier_end = GetSystemCycle();
                    for (uint32_t index = 0; index < peer_count; ++index) {
                        const uint64_t record = TraceBaseRecords(tiling) +
                                                uint64_t(tiling.total_chunk_num) * combine_profile::kChunkRecords +
                                                uint64_t(peers[index]) * tiling.chunks_per_group + chunk;
                        trace_recorder.Load(combine_profile::PeerSlotBase, record);
                        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::CrossGroupBarrierBegin,
                                           barrier_begin);
                        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::CrossGroupBarrierEnd,
                                           barrier_end);
                        trace_recorder.Save(combine_profile::PeerSlotBase, record);
                    }
                }
            }
        }
    }
}

constexpr uint32_t kWeightThreads = 128;
__simt_vf__ __launch_bounds__(kWeightThreads) inline void ScatterWeights(__gm__ float* weights,
                                                                         __gm__ int32_t* metadata,
                                                                         __gm__ float* receive, uint32_t begin,
                                                                         uint32_t end, uint32_t values_per_rank)
{
    for (uint32_t row = begin + threadIdx.x; row < end; row += kWeightThreads) {
        const uint32_t id = uint32_t(metadata[row]);
        ::simt::aclshmem_float_p(receive + id % values_per_rank, weights[row], int32_t(id / values_per_rank));
    }
    asc_threadfence();
}
__simt_vf__ __launch_bounds__(64) inline void NotifyWeightRanks(__gm__ uint64_t* done, uint32_t world)
{
    for (uint32_t peer = threadIdx.x; peer < world; peer += 64U) {
        ::simt::aclshmem_uint64_p(done, uint64_t(1), int32_t(peer));
    }
    asc_threadfence();
}
__aicore__ inline __gm__ uint64_t* WeightFlag(GM_ADDR workspace, uint64_t offset, uint32_t slot)
{
    return reinterpret_cast<__gm__ uint64_t*>(workspace + offset + uint64_t(slot) * combine_control::kSlotBytes);
}
__aicore__ inline void SendWeights(GM_ADDR workspace, GM_ADDR weights, GM_ADDR metadata, const CombineTiling& tiling,
                                   uint32_t reducer, uint32_t reducers, LocalTensor<uint64_t> signal)
{
    const uint32_t begin = uint64_t(tiling.weight_rows) * reducer / reducers;
    const uint32_t end = uint64_t(tiling.weight_rows) * (reducer + 1U) / reducers;
    if (begin != end) {
        asc_vf_call<ScatterWeights>(dim3(kWeightThreads), reinterpret_cast<__gm__ float*>(weights),
                                    reinterpret_cast<__gm__ int32_t*>(metadata),
                                    reinterpret_cast<__gm__ float*>(workspace + tiling.weight_recv_offset_bytes), begin,
                                    end, tiling.token_num * tiling.weight_topk);
        // SIMT remote-store visibility before MTE publication needs validation on the target CANN.
        PipeBarrier<PIPE_ALL>();
    }
    StoreFlag(WeightFlag(workspace, tiling.weight_local_done_offset_bytes, reducer), signal, 1U);
}
__aicore__ inline void PublishWeightDone(GM_ADDR workspace, const CombineTiling& tiling, uint32_t reducers)
{
    for (uint32_t reducer = 0; reducer < reducers; ++reducer) {
        while (!Ready(WeightFlag(workspace, tiling.weight_local_done_offset_bytes, reducer), 1U)) {
        }
    }
    asc_vf_call<NotifyWeightRanks>(dim3(64), WeightFlag(workspace, tiling.weight_done_offset_bytes, tiling.rank_id),
                                   tiling.rank_num);
    PipeBarrier<PIPE_ALL>();
}

template <bool EnableTrace>
__aicore__ inline void Combine(GM_ADDR workspace, GM_ADDR forward_list, GM_ADDR forward_counts, GM_ADDR backward_list,
                               GM_ADDR backward_counts, GM_ADDR token_server_mask, GM_ADDR chunk_ranges,
                               GM_ADDR chunk_masks, GM_ADDR trace_buffer, GM_ADDR weights, GM_ADDR weight_meta,
                               const CombineTiling& tiling)
{
    uint64_t entry_cycle = 0;
    if constexpr (EnableTrace) {
        entry_cycle = GetSystemCycle();
    }
    const uint32_t core_id = GetBlockIdx(), local_rank_num = tiling.rank_num < 8U ? tiling.rank_num : 8U;
    const uint32_t server_num = (tiling.rank_num + 7U) / 8U;
    const uint32_t group_num = tiling.rank_num / tiling.group_size;
    // Each cross owner has an exclusive QP pair across all remote peers.
    const uint32_t cross_sender_num = tiling.rank_num <= 8U ? 0U : tiling.num_cross_put_cores;
    const uint32_t reduce_core_begin = 8U + cross_sender_num;
    const uint32_t reduce_core_num = combine_config::kAivCores - reduce_core_begin;
    const bool is_cross_sender = IsCrossPhysicalCore(tiling, core_id);
    CrossQuietPlan quiet_plan;
    const bool is_reduce_core = core_id >= 8U && core_id < combine_config::kAivCores && !is_cross_sender;
    const uint32_t reduce_core_index = is_reduce_core ? ReduceLogicalIndex(tiling, core_id) : 0U;
    TPipe pipe;
    Trace<EnableTrace> trace_recorder;
    trace_recorder.Init(pipe, trace_buffer, tiling, core_id);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::Begin, entry_cycle);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::CoreCount, combine_config::kAivCores);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::ReduceCoreBegin, reduce_core_begin);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::ReduceCoreNum, reduce_core_num);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::ChunksPerGroup, tiling.chunks_per_group);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::ChunkTokens, tiling.chunk_token_num);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::NumSo, tiling.num_so);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::SharedJetty, tiling.shared_jetty);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::FirstChunkSplits, tiling.first_chunk_splits);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::GroupSize, tiling.group_size);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::SyncMask,
                       tiling.sync_local_reduce | (tiling.sync_reduce_local << 1U) | (tiling.sync_reduce_cross << 2U));
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::CrossTokensPerTurn, tiling.cross_tokens_per_turn);
    trace_recorder.Set(combine_profile::SummarySlot, combine_profile::IndependentChunkStride,
                       combine_profile::kChunkRecords);
    TBuf<QuePosition::VECCALC> bf16_input_buffer, fp32_input_buffer, fp32_accumulator_buffer, bf16_output_buffer,
        copy_buffer, signal_buffer;
    TBuf<TPosition::VECOUT> wqe_buffer;
    pipe.InitBuffer(bf16_input_buffer, kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(fp32_input_buffer, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(fp32_accumulator_buffer, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(bf16_output_buffer, kReduceOutputSlots * kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(copy_buffer, is_reduce_core ? combine_config::kGatewayUbBytes : kLocalCopyBufferBytes);
    pipe.InitBuffer(signal_buffer, kSignalUbBytes);
    pipe.InitBuffer(wqe_buffer, kWqeScratchBytes);
    auto bf16_input_ub = bf16_input_buffer.Get<bfloat16_t>();
    auto fp32_input_ub = fp32_input_buffer.Get<float>();
    auto fp32_accumulator_ub = fp32_accumulator_buffer.Get<float>();
    auto bf16_output_ub = bf16_output_buffer.Get<bfloat16_t>();
    auto copy_ub = copy_buffer.Get<uint8_t>();
    auto signal_ub = signal_buffer.Get<uint64_t>();
    auto* wqe_scratch_ub = reinterpret_cast<__ubuf__ uint8_t*>(wqe_buffer.Get<uint8_t>().GetPhyAddr());
    GlobalTensor<int32_t> forward_list_gm, forward_counts_gm, backward_counts_gm, chunk_ranges_gm;
    forward_list_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(forward_list));
    forward_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(forward_counts));
    backward_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(backward_counts));
    chunk_ranges_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunk_ranges));
    auto gateway_controls_ub = copy_buffer.Get<int32_t>()[combine_config::kGatewayScratchBytes / sizeof(int32_t)];
    if (is_reduce_core) {
        LoadGatewayControls(gateway_controls_ub, chunk_ranges, chunk_masks, tiling);
    }
    // Prepare resets and publishes control slots on the same stream.
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::InitDone);
    if (tiling.with_weights && is_reduce_core) {
        SendWeights(workspace, weights, weight_meta, tiling, reduce_core_index, reduce_core_num, signal_ub);
    }
    const bool local_mte = UseLocalMte(tiling.rank_num);
    // Each local core owns one MESH peer's QP0 or MTE stream, including its final drain.
    if (core_id < local_rank_num) {
        trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::GatherBegin);
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        const uint32_t task_num = backward_counts_gm.GetValue(core_id);
        trace_recorder.Reset(combine_profile::PeerSlotBase, combine_profile::Gather);
        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Peer, peer_rank);
        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Flags,
                           peer_rank != tiling.rank_id ? 1U : 0U);
        trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::Begin);
        // MTE payload banks and URMA WQE scratch must never alias metadata.
        auto gather_tasks_ub = local_mte ? wqe_buffer.Get<int64_t>() : copy_buffer.Get<int64_t>();
        const uint64_t peer_task_word_offset = uint64_t(core_id) * tiling.backward_row_capacity * kGatherTaskFields;
        LocalMteCopy gather_copy;
        if (local_mte) {
            gather_copy.Init();
        }
        aclshmemx_submit_state_t gather_submit_state{};
        aclshmemx_defer_t gather_defer(gather_submit_state);
        aclshmemx_submit_t gather_submit(gather_submit_state);
        uint32_t batch_start = 0;
        bool chunk_open = false;
        uint32_t chunk_task_num = 0;
        bool gather_group_seen[EnableTrace ? 128 : 1]{};
        while (batch_start < task_num) {
            const uint32_t loaded_rows =
                task_num - batch_start < kGatherBatchRows ? task_num - batch_start : kGatherBatchRows;
            // 从紧凑 GM 表加载到对齐的 UB 起点；尾批只读有效行，不读取下一 peer 的任务。
            NativeLoad(gather_tasks_ub,
                       reinterpret_cast<__gm__ int64_t*>(backward_list) + peer_task_word_offset +
                           uint64_t(batch_start) * kGatherTaskFields,
                       loaded_rows * kGatherTaskFields);
            // The packed list is sorted by (raw token chunk, group, source).
            const uint32_t group_index = CombineGroup(
                peer_rank, tiling.rank_num, static_cast<uint32_t>(gather_tasks_ub.GetValue(0)), tiling.group_size);
            if (!chunk_open) {
                if constexpr (EnableTrace) {
                    if (!gather_group_seen[group_index]) {
                        trace_recorder.Reset(combine_profile::GroupSlot, combine_profile::GatherGroup, group_index);
                        trace_recorder.Set(combine_profile::GroupSlot, combine_profile::Peer, peer_rank);
                        trace_recorder.Set(combine_profile::GroupSlot, combine_profile::Flags, 1U);
                        trace_recorder.Mark(combine_profile::GroupSlot, combine_profile::GroupBegin);
                        gather_group_seen[group_index] = true;
                    } else {
                        trace_recorder.Load(combine_profile::GroupSlot,
                                            combine_profile::kGatherGroupBase + group_index);
                    }
                }
                trace_recorder.Reset(combine_profile::ChunkSlot, combine_profile::GatherChunk, group_index);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::Peer, peer_rank);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::Flags, 1U);
                trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::GatherChunkPutBegin);
                chunk_task_num = 0;
                chunk_open = true;
            }
            uint32_t batch_rows = 0;
            uint32_t chunk_end_id = 0;
            for (uint32_t task_index = 0; task_index < loaded_rows; ++task_index) {
                const uint32_t task_word_offset = task_index * kGatherTaskFields;
                const uint64_t packed_offset = static_cast<uint64_t>(gather_tasks_ub.GetValue(task_word_offset + 2U));
                chunk_end_id = static_cast<uint32_t>(packed_offset >> 32U);
                // low32 是纯行号；high32 仅用于 chunk 标记，不能参与 payload 地址计算。
                const uint32_t reduce_row = static_cast<uint32_t>(packed_offset & 0xffffffffULL);
                auto* destination_address =
                    workspace + tiling.gather_input_offset_bytes + uint64_t(reduce_row) * tiling.hidden_size * 2U;
                auto* source_address =
                    workspace + tiling.expert_input_offset_bytes +
                    uint64_t(gather_tasks_ub.GetValue(task_word_offset + 1U)) * tiling.hidden_size * 2U;
                if (local_mte) {
                    gather_copy.Add(LocalMteDestination(destination_address, peer_rank, tiling.rank_id), source_address,
                                    tiling.hidden_size * 2U, copy_ub);
                } else if (chunk_end_id != 0U || task_index + 1U == loaded_rows) {
                    // Submit every chunk or metadata-batch tail with one payload CQE.
                    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoConfig>(
                        destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, peer_rank, 0U,
                        EVENT_ID0, gather_submit);
                } else {
                    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoNoCqeConfig>(
                        destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, peer_rank, 0U,
                        EVENT_ID0, gather_defer);
                }
                ++batch_rows;
                if (chunk_end_id) {
                    // Publish completion only after this chunk's payload has been issued.
                    break;
                }
            }
            // Metadata batch tails alone do not publish a completion flag.
            batch_start += batch_rows;
            chunk_task_num += batch_rows;
            if (chunk_end_id) {
                const uint32_t global_chunk = chunk_end_id - 1U;
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::ChunkIndex,
                                   global_chunk % tiling.chunks_per_group);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::Bytes0,
                                   uint64_t(chunk_task_num) * tiling.hidden_size * 2U);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::Rows, chunk_task_num);
                if (!local_mte) {
                    trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::Wqes0, chunk_task_num);
                }
                trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::GatherChunkPutEnd);
                if (tiling.sync_local_reduce) {
                    const uint32_t done_slot = 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                                               global_chunk * combine_control::kGatherDoneLanes + tiling.rank_id % 8U;
                    if (local_mte) {
                        PublishLocalMteFlag(Flag(workspace, done_slot), peer_rank, tiling.rank_id, signal_ub);
                    } else {
                        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                            Flag(workspace, done_slot), Flag(workspace, 3U * tiling.rank_num + core_id),
                            reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, peer_rank, 0U, EVENT_ID0);
                    }
                    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::SoSubmit);
                    trace_recorder.Mark(combine_profile::GroupSlot, combine_profile::SoSubmit);
                    trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::SoSubmit);
                }
                trace_recorder.Save(combine_profile::ChunkSlot, trace_recorder.ChunkRecordIndex(global_chunk, 0U));
                trace_recorder.Add(combine_profile::GroupSlot, combine_profile::Bytes0,
                                   uint64_t(chunk_task_num) * tiling.hidden_size * 2U);
                trace_recorder.Add(combine_profile::GroupSlot, combine_profile::Rows, chunk_task_num);
                if (!local_mte) {
                    trace_recorder.Add(combine_profile::GroupSlot, combine_profile::Wqes0, chunk_task_num);
                }
                trace_recorder.Mark(combine_profile::GroupSlot, combine_profile::GroupEnd);
                trace_recorder.Save(combine_profile::GroupSlot, combine_profile::kGatherGroupBase + group_index);
                chunk_open = false;
            }
        }
        trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::End);
        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Bytes0,
                           uint64_t(task_num) * tiling.hidden_size * 2U);
        trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Rows, task_num);
        if (!local_mte) {
            trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Wqes0, task_num);
        }
        if (local_mte || task_num) {
            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::QuietBegin);
            if (local_mte) {
                gather_copy.Finish();
            } else {
                aclshmemx_udma_qp_quiet(peer_rank, 0U);
            }
            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::QuietEnd);
        }
        trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::GatherEnd);
        trace_recorder.Save(combine_profile::PeerSlotBase, 1);
    }
    auto reduce_done_ub = signal_buffer.Get<int32_t>();
    if (is_reduce_core) {
        for (uint32_t chunk_index = 0; chunk_index < tiling.chunks_per_group; ++chunk_index) {
            for (uint32_t group_index = 0; group_index < group_num; ++group_index) {
                if (chunk_index == 0U) {
                    trace_recorder.Reset(combine_profile::GroupSlot, combine_profile::ReduceGroup, group_index);
                    trace_recorder.Mark(combine_profile::GroupSlot, combine_profile::GroupBegin);
                } else {
                    trace_recorder.Load(combine_profile::GroupSlot, 27U + TraceGroupCapacity(tiling) + group_index);
                }
                const uint32_t global_chunk = group_index * tiling.chunks_per_group + chunk_index;
                const uint32_t contributor_mask =
                    gateway_controls_ub.GetValue(GatewayMaskOffsetWords(tiling) + global_chunk);
                trace_recorder.Reset(combine_profile::ChunkSlot, combine_profile::ReduceChunk, group_index);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::ChunkIndex, chunk_index);
                trace_recorder.Set(combine_profile::ChunkSlot, combine_profile::ContributorMask, contributor_mask);
                // Gateway waits for this chunk's contributing experts' completion flags.
                if (tiling.sync_local_reduce) {
                    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::ChunkWaitBegin);
                    WaitGatherDone(workspace, tiling.rank_num, local_rank_num, global_chunk, contributor_mask,
                                   copy_buffer.Get<uint64_t>());
                    trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::ChunkReady);
                }
                trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::GatewayChunkReduceBegin);
                ReduceGateway(workspace, forward_list_gm, gateway_controls_ub, tiling, group_index, chunk_index,
                              reduce_core_index, reduce_core_num, bf16_input_ub, fp32_input_ub, fp32_accumulator_ub,
                              bf16_output_ub, copy_buffer.Get<int32_t>(), trace_recorder);
                // Empty reducers publish too; only this reducer writes its slot.
                if (tiling.sync_reduce_local || tiling.sync_reduce_cross) {
                    PublishReduceDone(workspace, tiling, global_chunk, reduce_core_index, reduce_done_ub);
                }
                trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::GatewayChunkReduceEnd);
                trace_recorder.Save(combine_profile::ChunkSlot, trace_recorder.ChunkRecordIndex(global_chunk, 1U));
                trace_recorder.Mark(combine_profile::GroupSlot, combine_profile::GroupEnd);
                trace_recorder.Save(combine_profile::GroupSlot, 27U + TraceGroupCapacity(tiling) + group_index);
            }
        }
    }
    if (tiling.with_weights && is_reduce_core && reduce_core_index == 0U) {
        PublishWeightDone(workspace, tiling, reduce_core_num);
    }
    if (core_id < local_rank_num) {
        // All gather submissions finish first to avoid blocking a later group's
        // inputs while waiting for this local return's reduce_done.
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        const uint32_t local_group = CombineGroup(tiling.rank_id, tiling.rank_num, peer_rank, tiling.group_size);
        trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::LocalReturnBegin);
        for (uint32_t chunk_index = 0; chunk_index < tiling.chunks_per_group; ++chunk_index) {
            trace_recorder.Reset(combine_profile::PeerSlotBase, combine_profile::LocalReturn);
            trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Peer, peer_rank);
            trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::Flags,
                               peer_rank != tiling.rank_id ? 1U : 0U);
            trace_recorder.Set(combine_profile::PeerSlotBase, combine_profile::ChunkIndex, chunk_index);
            if (tiling.sync_reduce_local) {
                trace_recorder.First(combine_profile::SummarySlot, combine_profile::LocalReturnReduceWaitBegin);
                trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::CreditObserved);
                WaitReduceDone(workspace, tiling, local_group * tiling.chunks_per_group + chunk_index, reduce_core_num,
                               copy_buffer.Get<int32_t>());
                trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::IncomingObserved);
                trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::LocalReturnReduceWaitEnd);
            }
            auto range_ub = copy_buffer.Get<int32_t>();
            NativeLoad(range_ub,
                       reinterpret_cast<__gm__ int32_t*>(chunk_ranges) +
                           (uint64_t(peer_rank) * tiling.chunks_per_group + chunk_index) * 2U,
                       2U);
            const uint32_t task_begin = range_ub.GetValue(0), task_end = range_ub.GetValue(1);
            if (peer_rank == tiling.rank_id) {
                // The original local core owns the entire self-copy chunk.
                trace_recorder.First(combine_profile::SummarySlot, combine_profile::SelfCopyBegin);
            }
            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::Begin);
            SendPartial(workspace, forward_list_gm, tiling, peer_rank, task_begin, task_end, copy_ub, wqe_scratch_ub,
                        trace_recorder, combine_profile::PeerSlotBase);
            trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::End);
            if (peer_rank != tiling.rank_id) {
                if (chunk_index == SoWindowEnd(tiling, chunk_index)) {
                    // Distinct immutable-one destinations at window tails, even if empty.
                    if (local_mte) {
                        PublishLocalMteFlag(
                            Flag(workspace, CrossCompletionSlot(tiling, tiling.rank_id, chunk_index, 0U)), peer_rank,
                            tiling.rank_id, signal_ub);
                    } else {
                        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                            Flag(workspace, CrossCompletionSlot(tiling, tiling.rank_id, chunk_index, 0U)),
                            Flag(workspace, 3U * tiling.rank_num + core_id),
                            reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, peer_rank, 0U, EVENT_ID0);
                    }
                    trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::SoSubmit);
                }
            } else if (chunk_index == SoWindowEnd(tiling, chunk_index)) {
                // One self-copy epoch per SO window, including empty windows.
                // Complete all prior writes before authorizing the whole window.
                SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
                StoreFlag(Flag(workspace, SelfCopyDoneSlot(tiling, core_id)), signal_ub, chunk_index + 1U);
                trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::End);
                trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::SelfCopyEnd);
            }
            trace_recorder.Save(combine_profile::PeerSlotBase, trace_recorder.ChunkRecordIndex(chunk_index, 3U));
        }
        trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::LocalReturnEnd);
    } else if (is_cross_sender) {
        trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::CrossReturnTotalBegin);
        CrossSend(workspace, forward_list_gm, chunk_ranges_gm, tiling, core_id, reduce_core_num, copy_ub, signal_ub,
                  copy_buffer.Get<int32_t>(), wqe_scratch_ub, trace_recorder, quiet_plan);
    }
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::SourceReduceBegin);
    // gateway 阶段已排空所有输出事件，此处从两个空闲槽开始复用同一块 UB。
    uint32_t source_output_slot = 0;
    bool source_input_read_pending[2]{};
    bool source_output_store_pending[kReduceOutputSlots]{};
    // This core's producer phase is finished; reuse its UB independently.
    auto source_bank0 = fp32_input_buffer.Get<bfloat16_t>();
    auto source_bank1 = copy_buffer.Get<bfloat16_t>();
    auto source_mask_ub = bf16_input_buffer.Get<uint64_t>();
    // A GM->UB padded block occupies 32 bytes for each 8-byte server mask.
    constexpr uint32_t mask_words_per_slot = 32U / sizeof(uint64_t);
    constexpr uint32_t mask_batch_capacity = kReduceTileElements * sizeof(bfloat16_t) / 32U;
    uint32_t mask_batch_count = 0;
    uint32_t mask_batch_index = 0;
    uint32_t token_id = core_id;
    for (uint32_t window_begin = 0; window_begin < tiling.chunks_per_group;
         window_begin = SoWindowEnd(tiling, window_begin) + 1U) {
        const uint32_t window_end = SoWindowEnd(tiling, window_begin);
        const uint64_t end_token = ChunkTokenEnd(tiling, window_end);
        const uint32_t phase_token_end = end_token < tiling.token_num ? end_token : tiling.token_num;
        WaitSourceChunk(workspace, tiling, window_begin, core_id, signal_ub, trace_recorder);
        for (; token_id < phase_token_end; token_id += combine_config::kAivCores) {
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::FinalTokens, 1);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::FinalWriteBytes,
                               uint64_t(tiling.hidden_size) * 2U);
            if (mask_batch_index == mask_batch_count) {
                const uint32_t remaining_tokens = (tiling.token_num - 1U - token_id) / combine_config::kAivCores + 1U;
                mask_batch_count = remaining_tokens < mask_batch_capacity ? remaining_tokens : mask_batch_capacity;
                mask_batch_index = 0;
                GlobalTensor<uint64_t> masks_gm;
                masks_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(token_server_mask) + token_id);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
                // Source stride is in bytes; destination blocks are padded to 32B.
                DataCopyPad(source_mask_ub, masks_gm,
                            DataCopyExtParams(mask_batch_count, sizeof(uint64_t),
                                              (combine_config::kAivCores - 1U) * sizeof(uint64_t), 0, 0),
                            DataCopyPadExtParams<uint64_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            }
            const uint64_t token_server_bits = source_mask_ub.GetValue(mask_batch_index * mask_words_per_slot);
            ++mask_batch_index;
            uint32_t servers[16];
            uint32_t partial_num = 0;
            for (uint32_t server = 0; server < server_num; ++server) {
                if (token_server_bits & (1ULL << server)) {
                    servers[partial_num++] = server;
                }
            }
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::FinalRows, partial_num);
            trace_recorder.Add(combine_profile::SummarySlot, combine_profile::FinalReadBytes,
                               uint64_t(partial_num) * tiling.hidden_size * 2U);
            for (uint32_t hidden_offset = 0; hidden_offset < tiling.hidden_size; hidden_offset += kReduceTileElements) {
                const uint32_t element_num = tiling.hidden_size - hidden_offset < kReduceTileElements
                                                 ? tiling.hidden_size - hidden_offset
                                                 : kReduceTileElements;
                const auto output_event_id = source_output_slot == 0U ? EVENT_ID1 : EVENT_ID2;
                auto bf16_output_slot_ub = bf16_output_ub[source_output_slot * kReduceTileElements];
                if (source_output_store_pending[source_output_slot]) {
                    WaitFlag<HardEvent::MTE3_V>(output_event_id);
                    source_output_store_pending[source_output_slot] = false;
                }
                if (partial_num == 0U) {
                    Duplicate(bf16_output_slot_ub, static_cast<bfloat16_t>(0.0F), element_num);
                } else {
                    const uint32_t pair_num = (partial_num + 1U) / 2U;
                    PrefetchSourcePair(workspace, tiling, token_id, server_num, servers[0],
                                       partial_num > 1U ? servers[1] : server_num, hidden_offset, element_num,
                                       source_bank0, 0U, source_input_read_pending[0]);
                    for (uint32_t pair = 0; pair < pair_num; ++pair) {
                        const uint32_t bank_index = pair % 2U;
                        const auto input_event = bank_index == 0U ? EVENT_ID0 : EVENT_ID4;
                        auto bank = bank_index == 0U ? source_bank0 : source_bank1;
                        // Enqueue the next pair before waiting on this pair, so
                        // scalar readiness waits do not delay its MTE2 submission.
                        if (pair + 1U < pair_num) {
                            const uint32_t next = (pair + 1U) * 2U;
                            const uint32_t next_bank = bank_index ^ 1U;
                            PrefetchSourcePair(workspace, tiling, token_id, server_num, servers[next],
                                               next + 1U < partial_num ? servers[next + 1U] : server_num, hidden_offset,
                                               element_num, next_bank == 0U ? source_bank0 : source_bank1, next_bank,
                                               source_input_read_pending[next_bank]);
                        }
                        WaitFlag<HardEvent::MTE2_V>(input_event);
                        RunSourceReducePair(
                            reinterpret_cast<__ubuf__ bfloat16_t*>(bank.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(bank.GetPhyAddr()) + kReduceTileElements,
                            reinterpret_cast<__ubuf__ float*>(fp32_accumulator_ub.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(bf16_output_slot_ub.GetPhyAddr()), element_num,
                            pair == 0U, pair * 2U + 1U < partial_num, pair + 1U == pair_num);
                        // Order accumulator stores before the following VF loads;
                        // the next bank's MTE2 transfer can run during this VF.
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(input_event);
                        source_input_read_pending[bank_index] = true;
                    }
                    // Keep bank release events across tiles/tokens; consume them
                    // only on that bank's next overwrite or at final drain below.
                }
                SetFlag<HardEvent::V_MTE3>(output_event_id);
                WaitFlag<HardEvent::V_MTE3>(output_event_id);
                GlobalTensor<bfloat16_t> output_gm;
                output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.output_offset_bytes) +
                                          uint64_t(token_id) * tiling.hidden_size + hidden_offset);
                DataCopy(output_gm, bf16_output_slot_ub, element_num);
                SetFlag<HardEvent::MTE3_V>(output_event_id);
                source_output_store_pending[source_output_slot] = true;
                source_output_slot ^= 1U;
            }
        }
        // Drain this phase before any next-phase SyncAll; idle cores still join.
        for (uint32_t slot = 0; slot < kReduceOutputSlots; ++slot) {
            if (source_output_store_pending[slot]) {
                const auto output_event_id = slot == 0U ? EVENT_ID1 : EVENT_ID2;
                WaitFlag<HardEvent::MTE3_V>(output_event_id);
                source_output_store_pending[slot] = false;
            }
        }
        for (uint32_t bank = 0; bank < 2U; ++bank) {
            if (source_input_read_pending[bank]) {
                WaitFlag<HardEvent::V_MTE2>(bank == 0U ? EVENT_ID0 : EVENT_ID4);
                source_input_read_pending[bank] = false;
            }
        }
        if constexpr (EnableTrace) {
            // Align the scalar phase-end timestamp with the completed output stores.
            SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        }
        trace_recorder.Mark(combine_profile::ChunkSlot, combine_profile::SourcePhaseReduceEnd);
        trace_recorder.Save(combine_profile::ChunkSlot, trace_recorder.ChunkRecordIndex(window_end, 2U));
    }
    // Drain output stores, outgoing CQs and every incoming chunk before the
    // next Prepare can reset control slots. Phase barriers have joined all reads.
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::SourceReduceEnd);
    if (!local_mte && core_id < local_rank_num && tiling.rank_id / 8U * 8U + core_id != tiling.rank_id) {
        // Local return SOs protect incoming data; drain outgoing QP0 before the
        // next invocation reuses its CQ, even for an empty payload window.
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::QuietBegin);
        aclshmemx_udma_qp_quiet(peer_rank, 0U);
        trace_recorder.Mark(combine_profile::PeerSlotBase, combine_profile::QuietEnd);
    }
    if (is_cross_sender) {
        // Incoming SO flags order the received partials. Finish draining outgoing
        // CQs after final reduce, before the next invocation reuses state.
        if constexpr (EnableTrace) {
            trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::FinalCrossQuietBegin);
        }
        QuietCrossServerQps(tiling, core_id, quiet_plan);
        if constexpr (EnableTrace) {
            trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::FinalCrossQuietEnd);
            trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::CrossReturnTotalEnd);
        }
    }
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::ReturnsPollBegin);
    DrainIncomingCompletions(workspace, tiling, core_id, combine_config::kAivCores, signal_ub);
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::ReturnsPollEnd);
    if (tiling.with_weights) {
        for (uint32_t sender = core_id; sender < tiling.rank_num; sender += combine_config::kAivCores) {
            while (!Ready(WeightFlag(workspace, tiling.weight_done_offset_bytes, sender), 1U)) {
            }
        }
    }
    if (core_id < local_rank_num) {
        trace_recorder.Save(combine_profile::PeerSlotBase, 2);
    }
    trace_recorder.Mark(combine_profile::SummarySlot, combine_profile::End);
    trace_recorder.Save(combine_profile::SummarySlot, 0);
}
#else
__aicore__ inline void Prepare(GM_ADDR, const CombineTiling&) {}
template <bool EnableTrace>
__aicore__ inline void Combine(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                               GM_ADDR, const CombineTiling&)
{
}
#endif
}  // namespace
// Session-level entry/exit barriers are separate from the arithmetic kernel.
// All 64 AIVs participate, including empty experts and zero-token workers.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_prepare_kernel(
    GM_ADDR workspace, ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Prepare(workspace, tiling);
#endif
}
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_kernel(
    GM_ADDR workspace, GM_ADDR forward, GM_ADDR counts, GM_ADDR backward, GM_ADDR backward_counts, GM_ADDR mask,
    GM_ADDR ranges, GM_ADDR masks, GM_ADDR weights, GM_ADDR weight_meta, ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Combine<false>(workspace, forward, counts, backward, backward_counts, mask, ranges, masks, nullptr, weights,
                   weight_meta, tiling);
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT void combine_prepare_kernel_do(void* stream, uint8_t* workspace, uint8_t* tiling)
{
    combine_prepare_kernel<<<64, 0, stream>>>(workspace, *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling));
}
extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(void* stream, uint8_t* workspace, uint8_t* forward,
                                                       uint8_t* counts, uint8_t* backward, uint8_t* backward_counts,
                                                       uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights,
                                                       uint8_t* weight_meta, uint8_t* tiling)
{
    combine_kernel<<<64, ascend_deepep::combine_config::kLaunchUbBytes, stream>>>(
        workspace, forward, counts, backward, backward_counts, mask, ranges, masks, weights, weight_meta,
        *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling));
}
