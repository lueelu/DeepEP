// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/device_functions.h"
#include "tiling.hpp"
#include "launch.hpp"
#include "../udma_layout.hpp"
#include "peer_order.hpp"
#include "scale_pipeline_layout.hpp"
#include "profile.hpp"
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

template <typename T>
__aicore__ inline void NativeLoadPacket(LocalTensor<T> ub, __gm__ T* source, uint32_t count)
{
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(source);
    gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(ub, gm, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0), DataCopyPadExtParams<T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}
__aicore__ inline void NativeLoadPacketTails(LocalTensor<int32_t> ub, GM_ADDR blocks, uint32_t count)
{
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blocks + 480U));
    gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(ub, gm, DataCopyExtParams(count, 32U, 480U, 0U, 0U), DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}
}  // namespace

namespace {
// Payload lives in framework GM; only controls and aggregate inboxes use SHMEM.
// Keep peer ownership, dual-QP SO completion and local row expansion unchanged.
#if (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)) && CATLASS_ARCH == 3510
// Preserve cross-server group/credit and dual-QP 3:1. Secondary is in dispatch.cpp.
// Adaptation: stable Notify rows and per-call live output virtual addresses, no expert migration.
namespace DispatchDeepep {
__aicore__ inline __gm__ uint8_t* PeerHiddenAddress(__gm__ uint64_t* book, int32_t peer,
                                                    uint32_t book_words = ascend_deepep::kPeerAddressRowWords)
{
    // EnsureDeepepPeerAddress acquires the current-generation record before use.
    // This ordinary GM VA is a URMA destination, NOT a remote MTE mapping.
    // Never feed it to remote MTE or apply symmetric-heap translation to it.
    GlobalTensor<uint64_t> addresses;
    addresses.SetGlobalBuffer(book);
    return reinterpret_cast<__gm__ uint8_t*>(addresses.GetValue(uint64_t(peer) * book_words));
}

constexpr uint32_t kUdmaWqeBatchSize = 128;
constexpr uint32_t kUdmaWqeBytes = 64;
// Payload uses QP0/1; control values occupy separate 512B workspace slots.
constexpr uint32_t kMaxCreatedUdmaQpCount = 6;
constexpr uint32_t kMaxActiveUdmaQpCount = 2;
constexpr uint32_t kActiveUdmaQpIndices[kMaxActiveUdmaQpCount] = {0, 1};
constexpr uint32_t kPeerCompletionValueSlot = 3;
constexpr uint32_t kPeerCreditFlagSlot = 5;
static_assert(kPeerCompletionValueSlot != kActiveUdmaQpIndices[0] &&
              kPeerCompletionValueSlot != kActiveUdmaQpIndices[1]);
static_assert(kPeerCompletionValueSlot != kPeerCreditFlagSlot);
static_assert(kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1] < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot > kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1]);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaSyncSoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

constexpr uint32_t kUdmaWqeScratchBytes = kUdmaWqeBatchSize * kUdmaWqeBytes;
constexpr uint32_t kWeightGmConflictGroupBytes = 512;
constexpr uint32_t kWeightsPerGmConflictGroup = kWeightGmConflictGroupBytes / sizeof(float);
constexpr uint32_t kWeightSimtThreadCount = kWeightsPerGmConflictGroup;
constexpr uint32_t kMteUbBytes = 16 * 1024;
constexpr uint32_t kLocalCopyTileBytes = 16 * 1024;
constexpr uint32_t slotChunkCount = 8192;
constexpr uint32_t kSlotTensorBytes = slotChunkCount * sizeof(int32_t);
constexpr uint32_t kSlotCompareMaskBytes = ((slotChunkCount / 8 + 31) / 32) * 32;
constexpr uint32_t kSlotSelectUbBytes = 4 * kSlotTensorBytes + kSlotCompareMaskBytes;
constexpr uint32_t kDuplicateHiddenStageBytes = 65504;
constexpr uint32_t kDuplicateMaxPrimaryRows = 4;
constexpr uint32_t kDuplicateHiddenStagingBytes = 2 * kDuplicateHiddenStageBytes;
constexpr uint32_t kAivUbCapacityBytes = 192 * 1024;
constexpr uint32_t kMaxDispatchGroupWidth = 16;
// 128P production topology: two 64P frames, each laid out as 8 direct-link
// rows x 8 columns.  A red-box dispatch group is one 4-row x 4-column tile,
// so each frame contains two row blocks x two left/right sides = four groups.
constexpr uint32_t kDispatchFrameCount = 2;
constexpr uint32_t kDispatchRanksPerFrame = 64;
constexpr uint32_t kDispatchRanksPerDirectRow = 8;
constexpr uint32_t kDispatchRowsPerFrame = 8;
constexpr uint32_t kDispatchGroupRows = 4;
constexpr uint32_t kDispatchGroupColumns = 4;
constexpr uint32_t kDispatchGroupRowBlocks = kDispatchRowsPerFrame / kDispatchGroupRows;
constexpr uint32_t kDispatchGroupSides = kDispatchRanksPerDirectRow / kDispatchGroupColumns;
constexpr uint32_t kDispatchGroupsPerFrame = kDispatchGroupRowBlocks * kDispatchGroupSides;
constexpr uint32_t kTwoFrameDispatchWorldSize = kDispatchFrameCount * kDispatchRanksPerFrame;

// 128P uses the frozen staged, two-frame schedule.
constexpr int64_t kPeerSyncFlagBytes = 512;
constexpr int64_t kPeerSyncSlotBytes = kMaxCreatedUdmaQpCount * kPeerSyncFlagBytes;
constexpr int64_t kRankSyncSlotBytes = kMaxDispatchGroupWidth * kPeerSyncSlotBytes;
constexpr uint32_t kMaxDispatchAivCount = 32;
constexpr int64_t kGroupSyncFlagBytes = 512;
constexpr uint32_t kGroupSyncRoundCount = 4;
constexpr uint32_t kGroupSyncPhasesPerGroup = 2;
constexpr uint32_t kGroupSyncCompletionPhase = 0;
constexpr uint32_t kGroupSyncDonePhase = 1;
constexpr int32_t kStrictGroupSyncMaxWorldSize = 64;
static_assert(kPeerSyncFlagBytes >= sizeof(uint64_t));
static_assert(kPeerSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kGroupSyncFlagBytes >= sizeof(uint64_t));
static_assert(kGroupSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kMaxDispatchGroupWidth % 2 == 0);
static_assert(kMaxDispatchAivCount > kMaxDispatchGroupWidth);
static_assert((1U << kGroupSyncRoundCount) >= kMaxDispatchGroupWidth);
static_assert(kDispatchGroupRows * kDispatchGroupColumns == kMaxDispatchGroupWidth);
static_assert(kDispatchRowsPerFrame * kDispatchRanksPerDirectRow == kDispatchRanksPerFrame);
static_assert(kDispatchGroupsPerFrame * kMaxDispatchGroupWidth == kDispatchRanksPerFrame);
static_assert(kDispatchGroupRowBlocks == 2);
static_assert(kDispatchGroupSides == 2);
static_assert(kDispatchGroupsPerFrame == 4);
static_assert(kTwoFrameDispatchWorldSize == 128);
static_assert(kWeightGmConflictGroupBytes % sizeof(float) == 0);
static_assert(kWeightSimtThreadCount == 128);
static_assert(kDuplicateHiddenStagingBytes <= kSlotSelectUbBytes);
static_assert(kUdmaWqeScratchBytes + kMteUbBytes + kLocalCopyTileBytes + kSlotSelectUbBytes <=
              kAivUbCapacityBytes - 8 * 1024);

constexpr uint32_t kVectorSlotSelectCountAlign = 64;
__aicore__ inline uint32_t GetChunkQpSlot(int64_t chunk_index, uint32_t active_qp_count)
{
    if (active_qp_count == 2) {
        return chunk_index % 4 == 3 ? 1U : 0U;
    }
    return static_cast<uint32_t>(chunk_index % active_qp_count);
}

__aicore__ inline __gm__ uint64_t* GetPeerCompletionFlag(int32_t src_rank, uint32_t peer_rank_index, uint32_t qp_idx,
                                                         __gm__ uint8_t* sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t*>(sync_buffer + static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
                                              static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
                                              qp_idx * kPeerSyncFlagBytes);
}

__aicore__ inline __gm__ uint64_t* GetPeerCreditFlag(int32_t src_rank, uint32_t peer_rank_index,
                                                     __gm__ uint8_t* sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t*>(sync_buffer + static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
                                              static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
                                              kPeerCreditFlagSlot * kPeerSyncFlagBytes);
}

__aicore__ inline uint64_t GetPeerCreditToken(uint64_t magic, uint32_t group, uint32_t group_count)
{
    return magic * (static_cast<uint64_t>(group_count) + 1U) + group + 1U;
}

__aicore__ inline __gm__ uint64_t* GetGroupSyncFlag(uint32_t aiv_index, uint32_t round,
                                                    __gm__ uint8_t* group_sync_buffer)
{
    const int64_t flag_index = static_cast<int64_t>(aiv_index) * kGroupSyncRoundCount + round;
    return reinterpret_cast<__gm__ uint64_t*>(group_sync_buffer + flag_index * kGroupSyncFlagBytes);
}

__aicore__ inline uint64_t GetGroupSyncInitToken(uint64_t magic, uint32_t group_count)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    return magic * token_count + 1U;
}

__aicore__ inline uint64_t GetGroupSyncPhaseToken(uint64_t magic, uint32_t group, uint32_t group_count, uint32_t phase)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    const uint64_t group_token_offset = static_cast<uint64_t>(group) * kGroupSyncPhasesPerGroup + phase + 2U;
    return magic * token_count + group_token_offset;
}

// Local dissemination is used ONLY by AIV8..15; FM/local-copy never join it.
__aicore__ inline void SyncActiveAivs(uint32_t aiv_index, uint32_t active_aiv_count, uint64_t sync_token,
                                      __gm__ uint8_t* group_sync_buffer)
{
    for (uint32_t distance = 1U, round = 0U; distance < active_aiv_count; distance <<= 1U, ++round) {
        const uint32_t next_aiv = (aiv_index + distance) % active_aiv_count;
        auto* next = GetGroupSyncFlag(next_aiv, round, group_sync_buffer);
        *next = sync_token;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(next));
        aclshmem_uint64_wait_until(GetGroupSyncFlag(aiv_index, round, group_sync_buffer), ACLSHMEM_CMP_GE, sync_token);
    }
}

__aicore__ inline uint32_t GetDispatchGroupWidth(int32_t world_size)
{
    return world_size < static_cast<int32_t>(kMaxDispatchGroupWidth) ? static_cast<uint32_t>(world_size)
                                                                     : kMaxDispatchGroupWidth;
}

__aicore__ inline bool UseTwoFrameDispatchSchedule(int32_t world_size)
{
    return world_size == static_cast<int32_t>(kTwoFrameDispatchWorldSize);
}

// Four rounds per frame, with the original per-edge completion/credit dependency.
__aicore__ inline uint32_t GetDispatchFrameToggle(uint32_t group)
{
    return group / kDispatchGroupsPerFrame;
}

__aicore__ inline uint32_t GetDispatchFrameSubround(uint32_t group)
{
    return group % kDispatchGroupsPerFrame;
}

__aicore__ inline int32_t BuildTwoFrameDispatchRank(uint32_t frame, uint32_t group_row, uint32_t group_side,
                                                    uint32_t group_rank_index)
{
    const uint32_t lane_row = group_rank_index / kDispatchGroupColumns;
    const uint32_t lane_column = group_rank_index % kDispatchGroupColumns;
    const uint32_t row = group_row * kDispatchGroupRows + lane_row;
    const uint32_t column = group_side * kDispatchGroupColumns + lane_column;
    return static_cast<int32_t>(frame * kDispatchRanksPerFrame + row * kDispatchRanksPerDirectRow + column);
}

__aicore__ inline uint32_t GetGroupRankIndex(int32_t rank, int32_t world_size)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t column = local_rank % kDispatchRanksPerDirectRow;
        return (row % kDispatchGroupRows) * kDispatchGroupColumns + column % kDispatchGroupColumns;
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t half_group_width = GetDispatchGroupWidth(world_size) / 2;
    return static_cast<uint32_t>(rank / half_world_size) * half_group_width +
           static_cast<uint32_t>(rank % static_cast<int32_t>(half_group_width));
}

// 128P staged policy: left/right groups start in opposite frames, then switch
// frames after four rounds. Preserve independent two-sequence credit progress.
// For every round the transform is bijective, so every destination red-box
// group receives from exactly one source red-box group.
__aicore__ inline int32_t GetGroupDstRank(int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t source_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t source_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t source_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t source_group_row = source_row / kDispatchGroupRows;
        const uint32_t source_group_side = source_column / kDispatchGroupColumns;

        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);

        //
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;

        //
        //
        const uint32_t destination_frame = source_frame ^ source_group_side ^ frame_toggle;
        const uint32_t destination_group_row = (source_group_row + group_row_delta) % kDispatchGroupRowBlocks;
        const uint32_t destination_group_side = source_group_side ^ group_side_delta;

        return BuildTwoFrameDispatchRank(destination_frame, destination_group_row, destination_group_side,
                                         group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t peer_group = (rank_group + group) % group_count;
    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
           static_cast<int32_t>(peer_group * half_group_width + group_rank_index % half_group_width);
}

__aicore__ inline int32_t GetGroupIncomingPeer(int32_t rank, int32_t world_size, uint32_t group,
                                               uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t destination_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t destination_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t destination_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t destination_group_row = destination_row / kDispatchGroupRows;
        const uint32_t destination_group_side = destination_column / kDispatchGroupColumns;

        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;
        const uint32_t source_group_side = destination_group_side ^ group_side_delta;
        const uint32_t source_group_row =
            (destination_group_row + kDispatchGroupRowBlocks - group_row_delta) % kDispatchGroupRowBlocks;
        const uint32_t source_frame = destination_frame ^ source_group_side ^ frame_toggle;

        return BuildTwoFrameDispatchRank(source_frame, source_group_row, source_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;

    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t incoming_group = (rank_group + group_count - group % group_count) % group_count;

    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
           static_cast<int32_t>(incoming_group * half_group_width + group_rank_index % half_group_width);
}

__aicore__ inline int32_t GetNextGroupCreditPeer(int32_t rank, int32_t world_size, uint32_t group,
                                                 uint32_t group_rank_index)
{
    return GetGroupIncomingPeer(rank, world_size, group + 1U, group_rank_index);
}

__aicore__ inline void DispatchSplitRange(int64_t total, int64_t part_count, int64_t part_index, int64_t& range_begin,
                                          int64_t& range_end)
{
    if (total <= 0 || part_count <= 0 || part_index < 0 || part_index >= part_count) {
        range_begin = 0;
        range_end = 0;
        return;
    }
    const int64_t base_count = total / part_count;
    const int64_t remainder = total - base_count * part_count;
    range_begin = part_index * base_count + (part_index < remainder ? part_index : remainder);
    range_end = range_begin + base_count + (part_index < remainder ? 1 : 0);
}

__aicore__ inline void IssueGmBytesToUb(AscendC::LocalTensor<uint8_t> destination_ub, GM_ADDR source_gm, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> source_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source_gm), bytes);
    AscendC::DataCopyPadExtParams<uint8_t> pad_parameters{false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_parameters{1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_ub, source_tensor, copy_parameters, pad_parameters);
}

__aicore__ inline void IssueUbBytesToGm(GM_ADDR destination_gm, AscendC::LocalTensor<uint8_t> source_ub, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination_gm), bytes);
    AscendC::DataCopyExtParams copy_parameters{1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_tensor, source_ub, copy_parameters);
}

// Only refresh outgoing values. Received generations persist across calls;
// clearing them here could erase a fast peer's current-generation publication.
__aicore__ inline void InitializeTokenScatterPeerSignals(int32_t current_rank, uint32_t group_rank_index,
                                                         uint64_t magic, uint32_t active_qp_count,
                                                         __gm__ uint8_t* sync_buffer)
{
    (void)active_qp_count;
    __gm__ uint64_t* completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    *completion_value = magic;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(completion_value));
}

__aicore__ inline bool IsSourceReadyForForwarding(int32_t incoming_peer, uint32_t current_rank_index, uint64_t magic,
                                                  uint32_t active_qp_count, __gm__ uint8_t* sync_buffer)
{
    AscendC::GlobalTensor<uint64_t> flag_tensor;
    const uint32_t used_qps = incoming_peer / 8 == aclshmem_my_pe() / 8 ? 1U : active_qp_count;
    for (uint32_t i = 0; i < used_qps; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t* flag = GetPeerCompletionFlag(incoming_peer, current_rank_index, qp_idx, sync_buffer);
        flag_tensor.SetGlobalBuffer(flag);
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag));
        if (flag_tensor.GetValue(0) != magic) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void PublishPeerCompletionSignal(int32_t dst_rank, int32_t current_rank, uint32_t group_rank_index,
                                                   uint32_t active_qp_count, __gm__ uint8_t* sync_buffer,
                                                   __ubuf__ uint8_t* udma_wqe_buffer)
{
    __gm__ uint64_t* completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    if (dst_rank == current_rank) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t* signal_flag = GetPeerCompletionFlag(current_rank, group_rank_index, qp_idx, sync_buffer);

        if (dst_rank == current_rank) {
            *signal_flag = *completion_value;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(signal_flag));
            continue;
        }
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
            signal_flag, completion_value, reinterpret_cast<__ubuf__ uint64_t*>(udma_wqe_buffer), 1, dst_rank, qp_idx,
            EVENT_ID0);
    }
}

__aicore__ inline void PublishNextPeerCredit(uint32_t group, int32_t current_rank, int32_t world_size,
                                             uint32_t group_rank_index, uint64_t magic, uint32_t group_count,
                                             __gm__ uint8_t* sync_buffer,
                                             AscendC::TBuf<AscendC::QuePosition::VECCALC>& local_copy_buffer)
{
    const int32_t next_peer = GetNextGroupCreditPeer(current_rank, world_size, group, group_rank_index);
    if (next_peer == current_rank) {
        return;
    }

    constexpr uint32_t credit_word_count = kPeerSyncFlagBytes / sizeof(uint64_t);
    const uint64_t next_credit_token = GetPeerCreditToken(magic, group + 1U, group_count);
    AscendC::LocalTensor<uint64_t> credit_local = local_copy_buffer.Get<uint64_t>();
    for (uint32_t i = 0; i < credit_word_count; ++i) {
        credit_local.SetValue(i, i == 0U ? next_credit_token : 0U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    __gm__ uint64_t* local_credit = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    __gm__ uint64_t* remote_credit = reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(local_credit, next_peer));
    AscendC::GlobalTensor<uint64_t> remote_credit_tensor;
    remote_credit_tensor.SetGlobalBuffer(remote_credit, credit_word_count);
    AscendC::DataCopy(remote_credit_tensor, credit_local, credit_word_count);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// One permanent owner per physical peer, including BOTH primary QPs.
// An FM owner never joins a Clos group. Self-copy has no QP.
struct DeepepSecondaryState {
    uint64_t profile = 0;
    uint64_t* detail = nullptr;
    uint32_t detail_mode = 0, rotation_wqes = 4, detail_phase = 0;
    // GM_ADDR is a pointer macro, so each member needs its own declaration.
    GM_ADDR hidden;
    GM_ADDR workspace;
    __gm__ uint64_t* book;
    __gm__ uint8_t* sync;
    uint32_t rank, world, core, lanes, row_bytes;
    uint64_t magic;
    bool peer = false, clos_shared_jetty = true, clos_split_qp = true, address_peer_ready = true;
    uint32_t book_words = 64U, clos_table_owner = 0U, scale_bytes = 0U;
    uint64_t clos_relay_mailbox = 0U, scale_send_offset = 0U, address_seen[4]{};
};

__aicore__ inline void CopyDeepepRows(GM_ADDR source, GM_ADDR destination, uint32_t row_bytes,
                                      AscendC::LocalTensor<int32_t> rows, uint32_t count,
                                      AscendC::TBuf<AscendC::QuePosition::VECCALC>& slot_buffer)
{
    const uint32_t stride = ((row_bytes + 31U) / 32U) * 32U;
    const uint32_t fit = stride <= kDuplicateHiddenStageBytes ? kDuplicateHiddenStageBytes / stride : 1U;
    const uint32_t rows_per_tile = fit < kDuplicateMaxPrimaryRows ? fit : kDuplicateMaxPrimaryRows;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t stage = 0;
    for (uint32_t row = 0; row < count; row += rows_per_tile) {
        const uint32_t n = count - row < rows_per_tile ? count - row : rows_per_tile;
        for (uint32_t offset = 0; offset < row_bytes; offset += kDuplicateHiddenStageBytes) {
            const uint32_t bytes =
                row_bytes - offset < kDuplicateHiddenStageBytes ? row_bytes - offset : kDuplicateHiddenStageBytes;
            const AscendC::TEventID event = stage == 0U ? EVENT_ID0 : EVENT_ID1;
            auto hidden =
                slot_buffer.GetWithOffset<uint8_t>(kDuplicateHiddenStageBytes, stage * kDuplicateHiddenStageBytes);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            for (uint32_t i = 0; i < n; ++i) {
                IssueGmBytesToUb(hidden[i * stride],
                                 source + uint64_t(rows.GetValue((row + i) * 2U)) * row_bytes + offset, bytes);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            for (uint32_t i = 0; i < n; ++i) {
                IssueUbBytesToGm(destination + uint64_t(rows.GetValue((row + i) * 2U + 1U)) * row_bytes + offset,
                                 hidden[i * stride], bytes);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            stage ^= 1U;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

struct DeepepPrimaryState {
    GM_ADDR input;
    GM_ADDR destinations;
    DeepepSecondaryState* secondary;
    uint32_t tokens, topk, address_stride;
    uint64_t clos_peer_table = 0, clos_peer_rows = 0, bytes = 0;
};

// One complete selection tile per visit. All deferred WQEs are submitted before
// switching peers. QP selection remains based on the ORIGINAL route-tile index.
__aicore__ inline void IssueDeepepPrimaryTile(DeepepPrimaryState& p, int32_t dst_rank, int64_t chunk_start,
                                              __ubuf__ uint8_t* udma_wqe_buffer,
                                              AscendC::TBuf<AscendC::QuePosition::VECCALC>& slot_select_buffer)
{
    auto& s = *p.secondary;
    auto* hidden = p.input;
    auto* encoded_destinations = reinterpret_cast<__gm__ int32_t*>(p.destinations);
    auto* book = s.book;
    const uint32_t current_rank = s.rank, active_qp_count = 2U;
    const int64_t num_topk = p.topk, route_count = int64_t(p.tokens) * p.topk;
    const int64_t hidden_row_bytes = s.row_bytes, address_stride = p.address_stride;

    const uint32_t tensor_bytes = slotChunkCount * sizeof(int32_t);
    const uint32_t compare_mask_bytes = ((slotChunkCount / 8 + 31) / 32) * 32;
    AscendC::LocalTensor<int32_t> dst_local = slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, 0);
    AscendC::LocalTensor<int32_t> dst_rank_tensor =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes);
    AscendC::LocalTensor<int32_t> route_indices =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 2);
    AscendC::LocalTensor<int32_t> selected_route_indices =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 3);
    AscendC::LocalTensor<uint8_t> compare_mask =
        slot_select_buffer.GetWithOffset<uint8_t>(compare_mask_bytes, tensor_bytes * 4);

    AscendC::GlobalTensor<int32_t> dst_global;
    dst_global.SetGlobalBuffer(encoded_destinations, route_count);
    uint32_t slot_shift = 0;
    // Address encoding follows receive capacity, NOT the source route count.
    for (int64_t value = address_stride; value > 1; value >>= 1) {
        ++slot_shift;
    }
    const uint32_t route_tile = dst_rank / 8 == current_rank / 8 ? slotChunkCount : 1024U;
    const int64_t chunk_index = chunk_start / route_tile;
    const uint32_t qp_slot = GetChunkQpSlot(chunk_index, dst_rank / 8 == current_rank / 8 ? 1U : active_qp_count);
    const uint32_t qp_idx = kActiveUdmaQpIndices[qp_slot];

    const int64_t remaining_count = route_count - chunk_start;
    const uint32_t chunk_count =
        static_cast<uint32_t>(remaining_count > static_cast<int64_t>(route_tile) ? route_tile : remaining_count);
    const uint32_t padded_count =
        ((chunk_count + kVectorSlotSelectCountAlign - 1) / kVectorSlotSelectCountAlign) * kVectorSlotSelectCountAlign;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    AscendC::DataCopyExtParams dst_copy_parameters{1, static_cast<uint32_t>(chunk_count * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> dst_pad_parameters{false, 0, 0, 0};
    AscendC::DataCopyPad(dst_local, dst_global[chunk_start], dst_copy_parameters, dst_pad_parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

    AscendC::ShiftRight(dst_rank_tensor, dst_local, static_cast<int32_t>(slot_shift), padded_count);
    AscendC::CreateVecIndex(route_indices, static_cast<int32_t>(chunk_start), padded_count);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

    AscendC::Compares(compare_mask, dst_rank_tensor, dst_rank, AscendC::CMPMODE::EQ, padded_count);
    AscendC::PipeBarrier<PIPE_V>();
    uint64_t selected_route_count = 0;
    AscendC::GatherMask(selected_route_indices, route_indices, compare_mask.ReinterpretCast<uint32_t>(), true,
                        chunk_count, {1U, 1U, 0U, 0U}, selected_route_count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

    const uint64_t local_selected_count = selected_route_count;

    aclshmemx_submit_state_t udma_submit_state{};

    constexpr uint32_t batch_rows = kUdmaWqeBatchSize;
    auto* target = PeerHiddenAddress(book, dst_rank, s.book_words);

    for (uint64_t batch_start = 0; batch_start < local_selected_count; batch_start += batch_rows) {
        const uint64_t batch_end =
            local_selected_count - batch_start > batch_rows ? batch_start + batch_rows : local_selected_count;

        for (uint64_t selected = batch_start; selected + 1 < batch_end; ++selected) {
            const int64_t route_index = selected_route_indices.GetValue(selected);
            const int64_t token_index = route_index / num_topk;
            const int64_t output_offset = dst_local.GetValue(route_index - chunk_start) % address_stride;
            __gm__ uint8_t* hidden_destination = target + output_offset * hidden_row_bytes;
            __gm__ uint8_t* hidden_source = hidden + token_index * hidden_row_bytes;
            // Same SDK aggregate primitive as public put, without its heap
            // translation. All intermediate RO remain CQE-free.
            aclshmemi_udma_stage_send_wqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoNoCqeConfig>(
                hidden_destination, hidden_source, dst_rank, qp_idx, hidden_row_bytes, udma_wqe_buffer,
                udma_submit_state);
        }

        const uint64_t selected = batch_end - 1;
        const int64_t route_index = selected_route_indices.GetValue(selected);
        const int64_t token_index = route_index / num_topk;
        const int64_t output_offset = dst_local.GetValue(route_index - chunk_start) % address_stride;
        __gm__ uint8_t* hidden_destination = target + output_offset * hidden_row_bytes;
        __gm__ uint8_t* hidden_source = hidden + token_index * hidden_row_bytes;
        aclshmemi_udma_submit_send_wqes<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoCqeConfig>(
            hidden_destination, hidden_source, dst_rank, qp_idx, hidden_row_bytes, udma_wqe_buffer, EVENT_ID0,
            udma_submit_state);
    }
}

__aicore__ inline void DispatchDeepepLocalPeer(DeepepPrimaryState& p, __ubuf__ uint8_t* wqe,
                                               AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    if (s.core + 1U >= s.lanes) {
        return;
    }
    const uint32_t lane = (s.rank % 8U + s.core + 1U) % s.lanes;
    const int32_t peer = s.rank / 8U * 8U + lane;
    for (uint64_t route = 0; route < uint64_t(p.tokens) * p.topk; route += slotChunkCount) {
        IssueDeepepPrimaryTile(p, peer, route, wqe, slots);
    }
    PublishPeerCompletionSignal(peer, s.rank, GetGroupRankIndex(peer, s.world), 1U, s.sync, wqe);
    aclshmemx_udma_qp_quiet(peer, kActiveUdmaQpIndices[0]);
}

// <=64P: 17 local writers use the unused upper group region.
// >=128P: 16 Clos owners need the full group region; nine local flags follow FM signals.
__aicore__ inline __gm__ uint64_t* DeepepLocalPrimaryDone(DeepepSecondaryState& s, uint32_t worker)
{
    return reinterpret_cast<__gm__ uint64_t*>(s.sync + uint64_t(s.world) * kRankSyncSlotBytes +
                                              (s.world >= 128U ? 40960U : 16384U) + uint64_t(worker) * 512U);
}

__aicore__ inline void DispatchDeepepLocalCopy(DeepepPrimaryState& p, __ubuf__ uint8_t* wqe,
                                               AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy,
                                               AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    const uint32_t worker = s.core - ascend_deepep::scale_clos::LocalFirst(s.world);
    int64_t begin = 0, end = 0;
    DispatchSplitRange(int64_t(p.tokens) * p.topk, ascend_deepep::scale_clos::LocalWorkers(s.world), worker, begin,
                       end);
    auto routes = copy.Get<int32_t>();
    auto rows = routes[kUdmaWqeBatchSize];

    for (int64_t first = begin; first < end; first += kUdmaWqeBatchSize) {
        const uint32_t n = end - first < kUdmaWqeBatchSize ? end - first : kUdmaWqeBatchSize;
        NativeLoad(routes, reinterpret_cast<__gm__ int32_t*>(p.destinations) + first, n);
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t encoded = routes.GetValue(i);
            if (encoded < 0 || uint32_t(encoded) / p.address_stride != s.rank) {
                continue;
            }
            rows.SetValue(count * 2U, (first + i) / p.topk);
            rows.SetValue(count * 2U + 1U, uint32_t(encoded) % p.address_stride);
            ++count;
        }
        if (!count) {
            continue;
        }
        CopyDeepepRows(p.input, s.hidden, s.row_bytes, rows, count, slots);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    auto* done = DeepepLocalPrimaryDone(s, worker);
    *done = s.magic;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(done));

    // Other writers need not wait for worker0; each has drained its own copy.
    // Self expansion starts only after every active writer has published done.
    if (worker == 0U) {
        for (uint32_t i = 0; i < ascend_deepep::scale_clos::LocalWorkers(s.world); ++i) {
            aclshmem_uint64_wait_until(DeepepLocalPrimaryDone(s, i), ACLSHMEM_CMP_EQ, s.magic);
        }
        PublishPeerCompletionSignal(s.rank, s.rank, GetGroupRankIndex(s.rank, s.world), 1U, s.sync, wqe);
    }
}

}  // namespace DispatchDeepep
#endif
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
using namespace DispatchDeepep;
using ascend_deepep::DispatchTiling;

#include "peer_transport.hpp"
#include "peer_plan.hpp"
#include "peer_scale.hpp"

// Notify assigns exactly one final row to each valid expert occurrence. Each
// secondary record has one owner: floor(count*w/workers)..floor(count*(w+1)/workers).
// The record's primary row is read-only once the original source's SO arrives.
// Each FM core owns one private MESH QP0 for BOTH primary and secondary,
// sequentially. Clos shared SQs and scale/weight inbox transfers are separate.
#include "fm_simt.hpp"

#include "fm_forward.hpp"

template <bool Profile>
__aicore__ inline void Dispatch(GM_ADDR workspace, GM_ADDR destinations, GM_ADDR forward, GM_ADDR counts,
                                GM_ADDR weights, GM_ADDR scales, const DispatchTiling& t)
{
    const uint32_t core = GetBlockIdx(), cores = 64U;
    const uint64_t profile = t.profile + uint64_t(core) * 256U;
    DispatchDetail<Profile> detail;
    DispatchProfileMark<Profile>(profile, 0U);
    TPipe pipe;
    TBuf<TPosition::VECOUT> wqe_buffer;
    TBuf<QuePosition::VECCALC> copy, slots;
    // UB ledger: 8192 WQE + 16384 metadata + 132096 select/ping-pong =
    // 156672 bytes. The remaining launch budget includes SIMT's 8KiB reserve.
    pipe.InitBuffer(wqe_buffer, kUdmaWqeScratchBytes);
    pipe.InitBuffer(copy, kLocalCopyTileBytes);
    pipe.InitBuffer(slots, kSlotSelectUbBytes);
    auto* wqe = reinterpret_cast<__ubuf__ uint8_t*>(wqe_buffer.Get<uint8_t>().GetPhyAddr());
    auto* sync = workspace + t.sync;
    auto* book = reinterpret_cast<__gm__ uint64_t*>(workspace + t.peer_book);
    // Host initializes these controls once per layout/workspace lifetime.
    // Never erase incoming current-generation flags in a late-starting kernel.
    DispatchProfileMark<Profile>(profile, 1U);
    // Peer generation/address handshakes gate consumers; no entry collective.
    DispatchProfileMark<Profile>(profile, 2U);
    if (core >= 16U && core - 16U < GetDispatchGroupWidth(t.world)) {
        InitializeTokenScatterPeerSignals(t.rank, core - 16U, t.generation, 2U, sync);
    }
    PublishPeerAddresses(workspace, t, copy, slots);
    DispatchProfileMark<Profile>(profile, 3U);
    if (core >= 32U) {
        if (weights) {
            PeerAuxPlan plan{destinations, weights,
                             nullptr,      reinterpret_cast<GM_ADDR>(t.output_weights_address),
                             nullptr,      t.address_stride};
            DeepepSecondaryState unused{};
            DispatchAuxPipeline::Run<Profile, false>(workspace, t, plan, slots, copy, wqe_buffer, unused);
        }
        // Independent inboxes, metadata, ready publication and local expansion.
        // Weight completion drains its UB/events before scale reuses the scratch.
        DispatchProfileMark<Profile>(profile, 31U);
        if (t.fp8) {
            PeerScales<Profile>(workspace, scales, destinations, t, copy, slots, wqe_buffer);
        }
        DispatchProfileMark<Profile>(profile, 4U);
        DispatchProfileMark<Profile>(profile, 5U);
    } else {
        DeepepSecondaryState s{};
        if constexpr (Profile) {
            s.profile = profile;
            s.detail = detail.data();
            s.detail_mode = t.detail_mode;
        }
        s.rotation_wqes = t.rotation_wqes;
        s.hidden = reinterpret_cast<GM_ADDR>(t.output_address);
        s.workspace = workspace;
        s.book = book;
        s.sync = sync;
        s.rank = t.rank;
        s.world = t.world;
        s.core = core;
        s.lanes = t.world < 8U ? t.world : 8U;
        s.row_bytes = t.fp8 ? 7168U : 14336U;
        s.magic = t.generation;
        s.peer = true;
        if (s.peer) {
            s.book = reinterpret_cast<__gm__ uint64_t*>(workspace + t.peer_book);
            s.clos_relay_mailbox = t.peer_table + t.plan_mailbox_offset;
        }
        DeepepPrimaryState p{};
        p.input = reinterpret_cast<GM_ADDR>(t.input_address);
        p.destinations = destinations;
        p.secondary = &s;
        p.tokens = t.tokens;
        p.topk = t.topk;
        p.address_stride = t.address_stride;
        p.clos_peer_table = t.peer_table;
        p.clos_peer_rows = t.peer_table + t.plan_rows_offset;
        const uint32_t local_first = ascend_deepep::scale_clos::LocalFirst(t.world);
        if (core == 7U) {
            if (t.world > 8U) RunClosRelayAssist<Profile>(s, copy);
        } else if (core >= 8U && core < local_first) {
            if (t.world > 8U) DispatchClosSplitQp<Profile, true>(p, wqe, copy, slots);
        } else if (core < 7U) {
            if (core + 1U < s.lanes) {
                const uint32_t peer = t.rank / 8U * 8U + (t.rank % 8U + core + 1U) % s.lanes;
                EnsureDeepepPeerAddress(s, peer);
            }
            DispatchDeepepLocalPeer(p, wqe, slots);

        } else {
            DispatchDeepepLocalCopy(p, wqe, copy, slots);
        }
        DispatchProfileMark<Profile>(profile, 4U);
        if (core < 7U || core >= local_first) {
            if (s.peer) {
                s.book = reinterpret_cast<__gm__ uint64_t*>(workspace + t.peer_book);
                if (core < 7U)
                    ForwardPeer<Profile, false>(s, t, copy, slots, wqe);
                else
                    ForwardPeer<Profile, true>(s, t, copy, slots, wqe);
            }
        }
        DispatchProfileMark<Profile>(profile, 5U);
    }
    // Fixed participation: all 64 AIVs join, including empty work owners.
    // Join ALL inbound writes before returning framework-owned live outputs.
    DispatchProfileMark<Profile>(profile, 6U);
    FinishPeerDispatch<Profile>(workspace, t, copy, detail.data());
    DispatchProfileMark<Profile>(profile, 7U);
    if constexpr (Profile) {
        if (t.detail_profile) {
            auto* d = detail.data();
            d[17] = t.rotation_wqes;
            d[18] = t.detail_mode;
            d[19] = t.generation;
            d[20] = 0x44504454U;
            d[21] = t.rank;
            d[22] = core;
            d[23] = 2U;
            d[28] = t.world;
            d[29] = ascend_deepep::scale_clos::Qp1Workers(t.world);
            auto* output = reinterpret_cast<__gm__ uint64_t*>(t.detail_profile) + uint64_t(core) * 32U;
            for (uint32_t i = 0; i < 32U; ++i) output[i] = d[i];
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(output), 256U);
        }
        auto* record = reinterpret_cast<__gm__ uint64_t*>(profile);
        record[26] = 0x44504550U;
        record[27] = t.rank;
        record[28] = core;
        record[29] = t.generation;
        record[30] = 7U;
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record), 256U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
}
#endif
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_kernel(GM_ADDR workspace,
                                                                                    GM_ADDR destinations,
                                                                                    GM_ADDR forward, GM_ADDR counts,
                                                                                    GM_ADDR weights, GM_ADDR scales,
                                                                                    ascend_deepep::DispatchTiling t)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Dispatch<false>(workspace, destinations, forward, counts, weights, scales, t);
#endif
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_profile_kernel(
    GM_ADDR workspace, GM_ADDR destinations, GM_ADDR forward, GM_ADDR counts, GM_ADDR weights, GM_ADDR scales,
    ascend_deepep::DispatchTiling t)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Dispatch<true>(workspace, destinations, forward, counts, weights, scales, t);
#endif
}

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(void* stream, uint8_t* workspace, uint8_t* destinations,
                                                        uint8_t* forward, uint8_t* counts, uint8_t* weights,
                                                        uint8_t* scales, uint8_t* tiling)
{
    const auto t = *reinterpret_cast<ascend_deepep::DispatchTiling*>(tiling);
    if (t.profile) {
        dispatch_profile_kernel<<<64, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(192 * 1024), stream>>>(
            workspace, destinations, forward, counts, weights, scales, t);
        return;
    }
    dispatch_kernel<<<64, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(192 * 1024), stream>>>(workspace, destinations, forward,
                                                                                counts, weights, scales, t);
}
