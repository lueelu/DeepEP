#include <cstdint>

#include "ascend_deepep/kernels/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_deepep/kernels/launch.h"
#include "ascend_deepep/kernels/types.h"

using namespace AscendC;

namespace {

// source_lane stores [source_server][token_id][hidden] at the same-lane gateway. One producer owns
// each destination-server task. Compact chunks split 3:1 over two QPs and publish
// one completion cache line on each QP. Completion ownership follows the payload owner;
// producer_completion remains indexed by producer core and chunk. Cache line 2 is a server-ready
// publication only on Direct-GET edges, while cache lines 3..7 aggregate source-server ranks.
constexpr uint32_t kUdmaQpCount = 2;
constexpr uint32_t kUdmaWqeBatchSize = 128;
constexpr uint32_t kUdmaWqeBytes = 64;
constexpr uint32_t kUdmaWqeBufferBytes = kUdmaWqeBatchSize * kUdmaWqeBytes;
constexpr uint32_t kUdmaMaxTransferBytes = 32U * 1024U;
constexpr uint32_t kCompletionSignalBytes = 512;
constexpr uint32_t kSignalCachelineBytes = 64;
constexpr uint32_t kServerReadyCacheline = 2U;
constexpr uint32_t kPeerPutDoneCacheline = 2U;
constexpr uint32_t kMaxSourceRankReadyCacheline = 7U;
constexpr uint32_t kTokenIdTileElements = 64U;
constexpr uint32_t kServersPerFrame = 8;
constexpr uint32_t kSignalUbOffset = kUdmaWqeBufferBytes;
constexpr uint32_t kSignalUbBytes = 32;
constexpr uint32_t kCopyBufferCount = 2;
constexpr uint32_t kCopyUbOffset = 0;
constexpr uint32_t kMaxServers = 64;
constexpr uint32_t kUbAlignment = 32;
constexpr uint32_t kInvalidPowerOfTwoShift = 32U;
constexpr uint32_t kGatewayPolicySourceLane = 0U;
constexpr uint32_t kRemoteGetProducerCount = 4U;
constexpr uint32_t kRemoteGetPublisherCount = 4U;
constexpr uint32_t kRemoteGetReceiverCount = 8U;
constexpr uint32_t kRemoteGetReceiverBegin = kRemoteGetProducerCount + kRemoteGetPublisherCount;
constexpr uint32_t kRemoteGetBlockDim = kRemoteGetReceiverBegin + kRemoteGetReceiverCount;
constexpr uint32_t kMaxRemoteGetServers = 7U;
constexpr uint32_t kQp1WqeOffset = kUdmaWqeBufferBytes;
constexpr uint32_t kRemoteGetRoutePositionUbOffset = 2U * kUdmaWqeBufferBytes;
constexpr uint32_t kDirectGetRouteRows = 256U;
constexpr uint32_t kDirectGetExpertBatch = 32U;
constexpr uint32_t kGatewaySegmentBatch = 32U;
constexpr uint32_t kGatewayRouteRows = ascend_deepep::kDupServerSourceLaneRouteTileElements;
constexpr uint32_t kDirectGetSegmentUbOffset = kRemoteGetRoutePositionUbOffset + kDirectGetRouteRows * sizeof(int32_t);
// Each strided 8-byte GM segment pair occupies one aligned 32-byte UB row.
constexpr uint32_t kDirectGetSegmentUbBytes = kDirectGetExpertBatch * kUbAlignment;
constexpr uint32_t kGateRxArrival = 0U;
constexpr uint32_t kGateTxArrival = 8U;
constexpr uint32_t kGateRxRelease = 16U;
constexpr uint32_t kGateTxRelease = 17U;
constexpr uint32_t kGateLinesPerOwner = 6U;
constexpr uint32_t kTokenIdUbOffset = kRemoteGetRoutePositionUbOffset;
constexpr uint32_t kRoutePositionTileBytes = ascend_deepep::kDupServerRoutePositionTileElements * sizeof(int32_t);
constexpr uint32_t kReceiverTraceFields = ascend_deepep::kDupServerReceiverTraceFields;
constexpr uint32_t kReceiverTraceRecordBytes = ascend_deepep::kDupServerReceiverTraceRecordBytes;
constexpr uint64_t kWaitCyclesPerMicrosecond = 1000U;
constexpr uint64_t kWaitTimeoutMicroseconds = 3U * 1000U * 1000U;
constexpr uint64_t kWaitTimeoutCycles = kWaitCyclesPerMicrosecond * kWaitTimeoutMicroseconds;

// A source-lane producer uses [0, 8192) and [8192, 16384) as independent QP WQE scratch regions.
// Before QP1 WQE construction, [8192, 8224) is temporarily used to stage the immutable completion
// value into GM; no asynchronous operation retains a UB source. Token IDs use [16384, 16640).
// After PUT quiet, the same producer AIV uses [16384, 17408) for Direct-GET route IDs and
// [17408, 18432) for 32 padded expert segment pairs. Both are independent of the two WQE regions.
// A scale-up AIV uses the UB head either for remote GET WQEs or for two DataCopyPad payload tiles.
// Per-source chunk cursors, route cursors, cached segment ends, and one route-position scan tile follow
// the two reserved payload tiles. Timer mode writes compact receiver trace records to local GM.
static_assert(kSignalUbOffset + kSignalUbBytes <= ascend_deepep::platform::kKernelUbBytes);
static_assert(kQp1WqeOffset + kUdmaWqeBufferBytes <= ascend_deepep::platform::kKernelUbBytes);
static_assert(kRemoteGetRoutePositionUbOffset + kRoutePositionTileBytes <= ascend_deepep::platform::kKernelUbBytes);
static_assert(kTokenIdUbOffset + kTokenIdTileElements * sizeof(int32_t) <= ascend_deepep::platform::kKernelUbBytes);
static_assert(kRemoteGetBlockDim == 16U);
static_assert(kDirectGetSegmentUbOffset + kDirectGetSegmentUbBytes <= ascend_deepep::platform::kKernelUbBytes);
static_assert(kDirectGetSegmentUbOffset % kUbAlignment == 0U);
static_assert(kRemoteGetRoutePositionUbOffset % kUbAlignment == 0U);
static_assert(kGateTxRelease + 4U <= 4U * kGateLinesPerOwner);
static_assert(kMaxRemoteGetServers < kRemoteGetReceiverBegin);
static_assert(kUdmaQpCount == 2U);
static_assert(kUdmaQpCount <= kServerReadyCacheline);
static_assert(kServerReadyCacheline * kSignalCachelineBytes < kCompletionSignalBytes);
static_assert(kPeerPutDoneCacheline * kSignalCachelineBytes < kCompletionSignalBytes);
static_assert(kMaxSourceRankReadyCacheline * kSignalCachelineBytes < kCompletionSignalBytes);
static_assert(kRoutePositionTileBytes % kUbAlignment == 0U);
static_assert(kGatewaySegmentBatch * kUbAlignment <= kGatewayRouteRows * sizeof(int32_t));
static_assert(kReceiverTraceRecordBytes == 40U);

enum ReceiverTraceField : uint32_t
{
    kTraceSourceAndChunk = 0,
    kTraceChunkEnd = 1,
    kTraceWaitBeginCycle = 2,
    kTraceReadyCycle = 3,
    kTraceIssueDoneCycle = 4,
};

struct ReceiverStageCycles {
    uint64_t route_scan_cycles;
    uint64_t remote_get_submit_cycles;
    uint64_t local_mte_copy_cycles;
};

struct ReceiverTimerData {
    uint64_t entry_cycle;
    uint64_t receive_start_cycle;
    uint64_t receive_end_cycle;
    uint64_t init_cycles;
    uint64_t wait_cycles;
    uint64_t route_issue_cycles;
    uint64_t route_scan_cycles;
    uint64_t remote_get_submit_cycles;
    uint64_t local_mte_copy_cycles;
    uint64_t bookkeeping_cycles;
    uint64_t final_drain_cycles;
    uint64_t final_drain_begin_cycle;
    uint64_t final_submit_done_cycle;
    uint64_t final_submit_cycles;
    uint64_t final_wait_cycles;
    uint64_t trace_record_cycles;
    uint64_t total_cycles;
    uint32_t trace_count;
};

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kDataNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kDataCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kCompletionCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
inline constexpr aclshmemx_udma_op_config_t kGetCompletionCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

__aicore__ inline __gm__ uint64_t* CompletionSignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_owner,
    uint32_t chunk_id)
{
    const uint64_t slot = static_cast<uint64_t>(source_owner) * tiling.chunk_capacity + chunk_id;
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.completion_signal_offset +
        slot * kCompletionSignalBytes);
}

// QP0 uses the legacy completion address. QP1 uses a separate cache line in the same 512-byte
// owner/chunk slot, leaving the workspace layout and host ABI unchanged.
__aicore__ inline __gm__ uint64_t* PayloadCompletionSignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_owner,
    uint32_t chunk_id, uint32_t qp_slot)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(CompletionSignal(symmetric_buffer, tiling, source_owner, chunk_id)) +
        static_cast<uint64_t>(qp_slot) * kSignalCachelineBytes);
}

// A source-server leader publishes this cache line only to destination servers which do not receive
// normal PUTs from that source. It is disjoint from both payload-completion cache lines.
__aicore__ inline __gm__ uint64_t* SourceServerReadySignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(CompletionSignal(symmetric_buffer, tiling, source_server, 0U)) +
        static_cast<uint64_t>(kServerReadyCacheline) * kSignalCachelineBytes);
}

__aicore__ inline __gm__ uint64_t* ProducerCompletionSignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t producer_core,
    uint32_t chunk_id)
{
    const uint64_t slot = static_cast<uint64_t>(producer_core) * tiling.chunk_capacity + chunk_id;
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.producer_signal_offset +
        slot * kCompletionSignalBytes);
}

// Server lanes 1..7 write these leader-owned aggregation slots. Lanes 1..5 use completion-owner 0
// cache lines 3..7; lanes 6..7 use owner 1 cache lines 3..4. The payload-completion and server-ready
// cache lines remain disjoint. At least four owners exist whenever remote-GET is valid.
__aicore__ inline __gm__ uint64_t* SourceRankReadySignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_lane)
{
    const uint32_t owner = (source_lane - 1U) / 5U;
    const uint32_t cacheline = 3U + (source_lane - 1U) % 5U;
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(CompletionSignal(symmetric_buffer, tiling, owner, 0U)) +
        static_cast<uint64_t>(cacheline) * kSignalCachelineBytes);
}

// A normal PUT producer is the unique writer for each destination server. It stores this local flag
// only after both QPs have drained, allowing that owning AIV to transition the same peer QPs to GET.
__aicore__ inline __gm__ uint64_t* PeerPutDoneSignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t destination_server)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(ProducerCompletionSignal(symmetric_buffer, tiling, destination_server, 0U)) +
        static_cast<uint64_t>(kPeerPutDoneCacheline) * kSignalCachelineBytes);
}

// Gate slots use producer-owner chunk 0, cache lines 1 and 3..7 only. Line 0 remains the
// immutable PUT completion source; line 2 remains PeerPutDone. RX/TX arrivals are lane-owned,
// RX release is local-leader-owned, and each TX release is owned by that source-server leader.
__aicore__ inline __gm__ uint64_t* RemoteGetGateSignal(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t index)
{
    const uint32_t owner = index / kGateLinesPerOwner;
    const uint32_t line_index = index % kGateLinesPerOwner;
    const uint32_t cacheline = line_index == 0U ? 1U : line_index + 2U;
    return reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(ProducerCompletionSignal(symmetric_buffer, tiling, owner, 0U)) +
        static_cast<uint64_t>(cacheline) * kSignalCachelineBytes);
}

__aicore__ inline void StoreSignal(
    __gm__ uint64_t* destination, AscendC::LocalTensor<uint64_t> signal_ub, uint64_t value)
{
    signal_ub.SetValue(0, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::GlobalTensor<uint64_t> signal_gm;
    signal_gm.SetGlobalBuffer(destination);
    AscendC::DataCopyExtParams copy_params(1, sizeof(uint64_t), 0, 0, 0);
    AscendC::DataCopyPad(signal_gm, signal_ub, copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
}

__aicore__ inline void StorePayloadCompletion(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_owner,
    uint32_t chunk_id, AscendC::LocalTensor<uint64_t> signal_ub, uint64_t value)
{
    for (uint32_t qp_slot = 0U; qp_slot < kUdmaQpCount; ++qp_slot) {
        StoreSignal(
            PayloadCompletionSignal(symmetric_buffer, tiling, source_owner, chunk_id, qp_slot), signal_ub, value);
    }
}

__aicore__ inline uint32_t RoutePositionTokenId(
    uint32_t route_position, const ascend_deepep::DupServerScaleOutTilingData& tiling)
{
    return tiling.num_topk_shift < kInvalidPowerOfTwoShift ? route_position >> tiling.num_topk_shift :
                                                             route_position / tiling.num_topk;
}

__aicore__ inline uint32_t CopyBytes(
    __gm__ uint8_t* destination, __gm__ uint8_t* source, uint32_t num_bytes,
    AscendC::LocalTensor<uint8_t> staging_pingpong_ub, uint32_t staging_buffer_idx, uint32_t copy_tile_bytes)
{
    AscendC::GlobalTensor<uint8_t> source_gm;
    AscendC::GlobalTensor<uint8_t> destination_gm;
    source_gm.SetGlobalBuffer(source);
    destination_gm.SetGlobalBuffer(destination);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};

    for (uint32_t offset = 0; offset < num_bytes; offset += copy_tile_bytes) {
        const uint32_t remaining = num_bytes - offset;
        const uint32_t copy_bytes = remaining < copy_tile_bytes ? remaining : copy_tile_bytes;
        const auto event_id = staging_buffer_idx == 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<uint8_t> staging_ub = staging_pingpong_ub[staging_buffer_idx * copy_tile_bytes];
        AscendC::DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
        // Wait only when this slot is reused. With two slots, token N's store can overlap token N+1's load,
        // and is not waited on until token N+2 needs the same UB slot.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(staging_ub, source_gm[offset], copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(destination_gm[offset], staging_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        staging_buffer_idx = (staging_buffer_idx + 1U) & 1U;
    }
    return staging_buffer_idx;
}

__aicore__ inline uint32_t CopyCachedRouteRows(
    AscendC::LocalTensor<int32_t> route_position_ub, __gm__ uint8_t* destination_base,
    __gm__ uint8_t* gateway_payload_base, const ascend_deepep::DupServerScaleOutTilingData& tiling,
    uint32_t source_server, uint32_t output_begin, uint32_t row_count,
    AscendC::LocalTensor<uint8_t> staging_pingpong_ub, uint32_t staging_buffer_idx, uint32_t copy_tile_bytes)
{
    if (tiling.hidden_bytes > copy_tile_bytes) {
        for (uint32_t row = 0; row < row_count; ++row) {
            const uint32_t output_index = output_begin + row;
            const uint32_t route_position = static_cast<uint32_t>(route_position_ub.GetValue(row));
            const uint32_t token_id = RoutePositionTokenId(route_position, tiling);
            const uint64_t source_slot = static_cast<uint64_t>(source_server) * tiling.token_capacity + token_id;
            staging_buffer_idx = CopyBytes(
                destination_base + static_cast<uint64_t>(output_index) * tiling.hidden_bytes,
                gateway_payload_base + source_slot * tiling.hidden_bytes, tiling.hidden_bytes, staging_pingpong_ub,
                staging_buffer_idx, copy_tile_bytes);
        }
        return staging_buffer_idx;
    }

    AscendC::GlobalTensor<uint8_t> destination_gm;
    AscendC::GlobalTensor<uint8_t> gateway_payload_gm;
    destination_gm.SetGlobalBuffer(destination_base);
    gateway_payload_gm.SetGlobalBuffer(gateway_payload_base);
    AscendC::DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyExtParams token_load_params(1, tiling.hidden_bytes, 0, 0, 0);
    const uint32_t rows_per_batch = copy_tile_bytes / tiling.hidden_bytes;

    for (uint32_t batch_offset = 0; batch_offset < row_count; batch_offset += rows_per_batch) {
        const uint32_t batch_begin = output_begin + batch_offset;
        const uint32_t remaining_rows = row_count - batch_offset;
        const uint32_t batch_rows = remaining_rows < rows_per_batch ? remaining_rows : rows_per_batch;
        const uint32_t batch_bytes = batch_rows * tiling.hidden_bytes;
        const auto event_id = staging_buffer_idx == 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<uint8_t> staging_ub = staging_pingpong_ub[staging_buffer_idx * copy_tile_bytes];

        // Reuse a slot only after its previous contiguous store finishes. The other slot remains available
        // for remote loads while that store is in flight.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        for (uint32_t row = 0; row < batch_rows; ++row) {
            const uint32_t output_index = batch_begin + row;
            const uint32_t route_position = static_cast<uint32_t>(route_position_ub.GetValue(batch_offset + row));
            const uint32_t token_id = RoutePositionTokenId(route_position, tiling);
            const uint64_t source_slot = static_cast<uint64_t>(source_server) * tiling.token_capacity + token_id;
            AscendC::DataCopyPad(
                staging_ub[static_cast<uint64_t>(row) * tiling.hidden_bytes],
                gateway_payload_gm[source_slot * tiling.hidden_bytes], token_load_params, pad_params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyExtParams batch_store_params(1, batch_bytes, 0, 0, 0);
        AscendC::DataCopyPad(
            destination_gm[static_cast<uint64_t>(batch_begin) * tiling.hidden_bytes], staging_ub, batch_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        staging_buffer_idx = (staging_buffer_idx + 1U) & 1U;
    }
    return staging_buffer_idx;
}

__aicore__ inline uint32_t SameFramePeerCount(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t frame_servers, uint32_t wave)
{
    const uint32_t first_offset = wave * tiling.max_inflight_same_frame_servers + 1U;
    if (first_offset >= frame_servers) {
        return 0U;
    }
    const uint32_t remaining = frame_servers - first_offset;
    return remaining < tiling.max_inflight_same_frame_servers ? remaining : tiling.max_inflight_same_frame_servers;
}

__aicore__ inline uint64_t AllServerMask(uint32_t num_servers)
{
    return num_servers == kMaxServers ? ~uint64_t{0} : (uint64_t{1} << num_servers) - 1U;
}

__aicore__ inline uint32_t RemoteGetSourceOffset(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t remote_index)
{
    // EP128 keeps the first three sources in this half, then uses four sources from the opposite pod.
    // At the maximum split this is the inverse of PUT offsets 1..4 and 9..12 from every source server.
    if (tiling.num_servers == 16U && remote_index >= 3U) {
        return remote_index + 5U;
    }
    return remote_index + 1U;
}

__aicore__ inline uint32_t RemoteGetSourceServer(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t destination_server, uint32_t remote_index)
{
    return (destination_server + RemoteGetSourceOffset(tiling, remote_index)) % tiling.num_servers;
}

__aicore__ inline bool ServerUsesRemoteGet(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server, uint32_t destination_server)
{
    if (tiling.remote_get_servers == 0U || source_server == destination_server) {
        return false;
    }
    for (uint32_t remote_index = 0U; remote_index < tiling.remote_get_servers; ++remote_index) {
        if (RemoteGetSourceServer(tiling, destination_server, remote_index) == source_server) {
            return true;
        }
    }
    return false;
}

// A peer/QP has no core dimension in the UDMA context. If this rank also PUTs to a Direct-GET
// source server, keep the GET on the producer which already owns that peer. A source omitted from
// the local PUT set has no prior owner, so assign it deterministically among the four producers.
__aicore__ inline uint32_t DirectGetOwnerCore(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t local_server, uint32_t remote_index)
{
    const uint32_t source_server = RemoteGetSourceServer(tiling, local_server, remote_index);
    uint32_t put_index = 0U;
    for (uint32_t destination_offset = 1U; destination_offset < tiling.num_servers; ++destination_offset) {
        const uint32_t destination_server = (local_server + destination_offset) % tiling.num_servers;
        if (ServerUsesRemoteGet(tiling, local_server, destination_server)) {
            continue;
        }
        if (destination_server == source_server) {
            return put_index % kRemoteGetProducerCount;
        }
        ++put_index;
    }
    return remote_index % kRemoteGetProducerCount;
}

__aicore__ inline uint64_t GatewayServerMask(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t destination_server)
{
    uint64_t mask = AllServerMask(tiling.num_servers);
    for (uint32_t source_server = 0; source_server < tiling.num_servers; ++source_server) {
        if (ServerUsesRemoteGet(tiling, source_server, destination_server)) {
            mask &= ~(uint64_t{1} << source_server);
        }
    }
    return mask;
}

__aicore__ inline uint32_t WavePeerCount(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t frame_servers, uint32_t wave)
{
    const uint32_t cross_frame_peers = tiling.num_servers - frame_servers;
    return SameFramePeerCount(tiling, frame_servers, wave) + (wave == 0U ? cross_frame_peers : 0U);
}

__aicore__ inline uint32_t CrossFrameServer(uint32_t logical_peer, uint32_t frame_server_begin, uint32_t frame_servers)
{
    return logical_peer < frame_server_begin ? logical_peer : logical_peer + frame_servers;
}

__aicore__ inline uint32_t OutgoingServer(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server, uint32_t frame_server_begin,
    uint32_t frame_servers, uint32_t logical_peer, uint32_t wave)
{
    const uint32_t cross_frame_peers = tiling.num_servers - frame_servers;
    if (wave == 0U && logical_peer < cross_frame_peers) {
        return CrossFrameServer(logical_peer, frame_server_begin, frame_servers);
    }
    const uint32_t same_frame_peer = logical_peer - (wave == 0U ? cross_frame_peers : 0U);
    const uint32_t offset = wave * tiling.max_inflight_same_frame_servers + same_frame_peer + 1U;
    const uint32_t source_in_frame = source_server - frame_server_begin;
    const uint32_t rotated_server = source_in_frame + offset;
    // Keep the common full-frame path equivalent to the original fixed 8-server
    // schedule.  Bisheng can lower the constant modulo to a mask, while only a
    // partial tail frame pays for the general modulo.
    if (frame_servers == kServersPerFrame) {
        return frame_server_begin + rotated_server % kServersPerFrame;
    }
    return frame_server_begin + rotated_server % frame_servers;
}

__aicore__ inline void SendTokenRange(
    GM_ADDR x, GM_ADDR send_token_ids, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server, uint32_t destination_server,
    uint32_t destination_rank, uint32_t begin, uint32_t end, uint32_t qp_slot, __ubuf__ uint8_t* udma_wqe_buffer,
    AscendC::LocalTensor<int32_t> token_ids_ub)
{
    if (begin >= end) {
        return;
    }

    AscendC::GlobalTensor<int32_t> token_ids_gm;
    token_ids_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_token_ids));
    const uint64_t token_list_base = static_cast<uint64_t>(destination_server) * tiling.token_capacity;
    uint32_t pending_wqes = 0U;
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);

    for (uint32_t position = begin; position < end;) {
        const uint32_t in_tile = (position - begin) % kTokenIdTileElements;
        uint32_t tile_rows = 0U;
        if (in_tile == 0U) {
            tile_rows = end - position < kTokenIdTileElements ? end - position : kTokenIdTileElements;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::DataCopyExtParams copy_params(1, tile_rows * sizeof(int32_t), 0, 0, 0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(token_ids_ub, token_ids_gm[token_list_base + position], copy_params, pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
        } else {
            tile_rows = kTokenIdTileElements;
        }

        const uint32_t token_id = static_cast<uint32_t>(token_ids_ub.GetValue(in_tile));
        const uint32_t available = end - position < tile_rows - in_tile ? end - position : tile_rows - in_tile;
        uint32_t run = 1U;
        if (tiling.hidden_bytes != 0U && tiling.hidden_bytes <= kUdmaMaxTransferBytes / 2U) {
            const uint32_t max_rows = kUdmaMaxTransferBytes / tiling.hidden_bytes;
            const uint32_t limit = available < max_rows ? available : max_rows;
            while (run < limit && static_cast<uint32_t>(token_ids_ub.GetValue(in_tile + run)) == token_id + run) {
                ++run;
            }
        }
        const uint64_t destination_slot = static_cast<uint64_t>(source_server) * tiling.token_capacity + token_id;
        const uint32_t run_bytes = tiling.hidden_bytes * run;
        for (uint32_t byte_offset = 0U; byte_offset < run_bytes; byte_offset += kUdmaMaxTransferBytes) {
            const uint32_t remaining_bytes = run_bytes - byte_offset;
            const uint32_t transfer_bytes =
                remaining_bytes < kUdmaMaxTransferBytes ? remaining_bytes : kUdmaMaxTransferBytes;
            ++pending_wqes;
            const bool close_batch = pending_wqes == kUdmaWqeBatchSize ||
                                     (position + run == end && byte_offset + transfer_bytes == run_bytes);
            __gm__ uint8_t* destination = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_buffer_offset +
                                          destination_slot * tiling.hidden_bytes + byte_offset;
            __gm__ uint8_t* source = reinterpret_cast<__gm__ uint8_t*>(x) +
                                     static_cast<uint64_t>(token_id) * tiling.hidden_bytes + byte_offset;
            if (close_batch) {
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kDataCqeConfig>(
                    destination, source, udma_wqe_buffer, transfer_bytes, static_cast<int32_t>(destination_rank),
                    qp_slot, EVENT_ID5, submit_action);
                pending_wqes = 0U;
            } else {
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kDataNoCqeConfig>(
                    destination, source, udma_wqe_buffer, transfer_bytes, static_cast<int32_t>(destination_rank),
                    qp_slot, EVENT_ID5, defer_action);
            }
        }
        position += run;
    }
}

__aicore__ inline void PublishCompletion(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_owner,
    uint32_t destination_rank, uint32_t producer_core, uint32_t chunk_id, uint32_t qp_slot,
    __ubuf__ uint8_t* udma_wqe_buffer)
{
    __gm__ uint64_t* destination_signal =
        PayloadCompletionSignal(symmetric_buffer, tiling, source_owner, chunk_id, qp_slot);
    // Both QPs read the same immutable, 512-byte-aligned source value. It is written before payload
    // submission and is not modified until both QPs have been drained.
    __gm__ uint64_t* source_signal = ProducerCompletionSignal(symmetric_buffer, tiling, producer_core, chunk_id);
    aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionCqeConfig>(
        destination_signal, source_signal, reinterpret_cast<__ubuf__ uint64_t*>(udma_wqe_buffer), 1U,
        static_cast<int32_t>(destination_rank), qp_slot, EVENT_ID5);
}

__aicore__ inline uint32_t CompactChunkTokenEnd(
    AscendC::GlobalTensor<int32_t> token_ids_gm, uint64_t token_list_base,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t chunk_id, uint32_t compact_begin,
    uint32_t compact_end)
{
    if (compact_begin >= compact_end || chunk_id + 1U == tiling.num_token_chunks) {
        return tiling.num_tokens;
    }
    return static_cast<uint32_t>(token_ids_gm.GetValue(token_list_base + compact_end - 1U)) + 1U;
}

__aicore__ inline uint32_t PollCompletionTokenEnd(__gm__ uint64_t* signal)
{
    AscendC::GlobalTensor<uint8_t> signal_cacheline_gm;
    AscendC::GlobalTensor<uint64_t> signal_gm;
    signal_cacheline_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(signal));
    signal_gm.SetGlobalBuffer(signal);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        signal_cacheline_gm);
    __asm__ __volatile__("");
    return static_cast<uint32_t>(signal_gm.GetValue(0));
}

__aicore__ inline uint32_t PollPayloadCompletionTokenEnd(
    __gm__ uint8_t* gateway_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_owner,
    uint32_t chunk_id)
{
    const uint32_t qp0_token_end =
        PollCompletionTokenEnd(PayloadCompletionSignal(gateway_buffer, tiling, source_owner, chunk_id, 0U));
    const uint32_t qp1_token_end =
        PollCompletionTokenEnd(PayloadCompletionSignal(gateway_buffer, tiling, source_owner, chunk_id, 1U));
    return qp0_token_end != 0U && qp0_token_end == qp1_token_end ? qp0_token_end : 0U;
}

__aicore__ inline uint64_t StoreReceiverTrace(
    AscendC::GlobalTensor<uint64_t> trace_gm, uint32_t trace_index, uint32_t source_id, uint32_t chunk_begin,
    uint32_t chunk_end, uint64_t wait_begin_cycle, uint64_t ready_cycle, uint64_t issue_done_cycle)
{
    const uint32_t trace_base = trace_index * kReceiverTraceFields;
    trace_gm.SetValue(trace_base + kTraceSourceAndChunk, (static_cast<uint64_t>(chunk_begin) << 32U) | source_id);
    trace_gm.SetValue(trace_base + kTraceChunkEnd, chunk_end);
    trace_gm.SetValue(trace_base + kTraceWaitBeginCycle, wait_begin_cycle);
    trace_gm.SetValue(trace_base + kTraceReadyCycle, ready_cycle);
    trace_gm.SetValue(trace_base + kTraceIssueDoneCycle, issue_done_cycle);
    const uint64_t trace_done_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    return trace_done_cycle;
}

__aicore__ inline void SendServerPartition(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server, uint32_t destination_server,
    uint32_t producer_core, __ubuf__ uint8_t* udma_wqe_buffer, AscendC::LocalTensor<uint64_t> signal_ub,
    AscendC::LocalTensor<int32_t> token_ids_ub)
{
    AscendC::GlobalTensor<int32_t> token_count_gm;
    token_count_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_token_count));
    const uint32_t count = static_cast<uint32_t>(token_count_gm.GetValue(destination_server));
    const uint32_t local_rank = tiling.rank % tiling.ranks_per_server;
    const uint32_t destination_rank = destination_server * tiling.ranks_per_server + local_rank;
    AscendC::GlobalTensor<int32_t> token_ids_gm;
    token_ids_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_token_ids));
    const uint64_t token_list_base = static_cast<uint64_t>(destination_server) * tiling.token_capacity;
    __ubuf__ uint8_t* qp0_wqe_buffer = udma_wqe_buffer;
    __ubuf__ uint8_t* qp1_wqe_buffer = udma_wqe_buffer + kQp1WqeOffset;

    // Match the elastic protocol: split each compact chunk 3:1 and publish one completion on each
    // QP. Each SO orders only the payload previously submitted to its own QP. Keep the two QP WQE
    // staging regions independent, matching the already used dual-QP Direct-GET path.
    for (uint32_t chunk_id = 0U; chunk_id < tiling.num_token_chunks; ++chunk_id) {
        const uint32_t chunk_begin = chunk_id * tiling.num_tokens_per_chunk;
        // An exhausted compact list still needs a terminal completion on both QPs. Clamp its
        // empty range before the unsigned 3:1 split so end - begin cannot underflow.
        const uint32_t begin = chunk_begin < count ? chunk_begin : count;
        uint32_t end = begin + tiling.num_tokens_per_chunk;
        end = end < count ? end : count;
        const uint32_t split = begin + ((end - begin) * 3U + 3U) / 4U;
        const uint32_t ready_token_end =
            CompactChunkTokenEnd(token_ids_gm, token_list_base, tiling, chunk_id, begin, end);
        // Store once before either payload stream is submitted. Both completion WQEs may read this
        // GM address asynchronously, so it must remain unchanged through both final quiets.
        StoreSignal(
            ProducerCompletionSignal(symmetric_buffer, tiling, producer_core, chunk_id), signal_ub, ready_token_end);
        SendTokenRange(
            x, send_token_ids, symmetric_buffer, tiling, source_server, destination_server, destination_rank, begin,
            split, 0U, qp0_wqe_buffer, token_ids_ub);
        SendTokenRange(
            x, send_token_ids, symmetric_buffer, tiling, source_server, destination_server, destination_rank, split,
            end, 1U, qp1_wqe_buffer, token_ids_ub);
        for (uint32_t qp_slot = 0U; qp_slot < kUdmaQpCount; ++qp_slot) {
            PublishCompletion(
                symmetric_buffer, tiling, source_server, destination_rank, producer_core, chunk_id, qp_slot,
                qp_slot == 0U ? qp0_wqe_buffer : qp1_wqe_buffer);
        }
        if (ready_token_end == tiling.num_tokens) {
            break;
        }
    }
    for (uint32_t qp_slot = 0U; qp_slot < kUdmaQpCount; ++qp_slot) {
        aclshmemx_udma_qp_quiet(static_cast<int32_t>(destination_rank), qp_slot);
    }
}

__aicore__ inline void CopyLocalServerPartition(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, AscendC::LocalTensor<uint8_t> staging_pingpong_ub,
    AscendC::LocalTensor<uint64_t> signal_ub)
{
    const uint32_t local_server = tiling.rank / tiling.ranks_per_server;
    AscendC::GlobalTensor<int32_t> token_count_gm;
    AscendC::GlobalTensor<int32_t> token_ids_gm;
    token_count_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_token_count));
    token_ids_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(send_token_ids));
    const uint32_t count = static_cast<uint32_t>(token_count_gm.GetValue(local_server));
    const uint64_t token_list_base = static_cast<uint64_t>(local_server) * tiling.token_capacity;
    uint32_t staging_buffer_idx = 0U;

    for (uint32_t chunk_id = 0; chunk_id < tiling.num_token_chunks; ++chunk_id) {
        const uint32_t begin = chunk_id * tiling.num_tokens_per_chunk;
        uint32_t end = begin + tiling.num_tokens_per_chunk;
        end = end < count ? end : count;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t position = begin; position < end; ++position) {
            const uint32_t token_id = static_cast<uint32_t>(token_ids_gm.GetValue(token_list_base + position));
            const uint64_t destination_slot = static_cast<uint64_t>(local_server) * tiling.token_capacity + token_id;
            staging_buffer_idx = CopyBytes(
                reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_buffer_offset +
                    destination_slot * tiling.hidden_bytes,
                reinterpret_cast<__gm__ uint8_t*>(x) + static_cast<uint64_t>(token_id) * tiling.hidden_bytes,
                tiling.hidden_bytes, staging_pingpong_ub, staging_buffer_idx, tiling.copy_tile_bytes);
        }
        // The ready boundary can only be published after both ping-pong stores for this compact chunk drain.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        const uint32_t ready_token_end =
            CompactChunkTokenEnd(token_ids_gm, token_list_base, tiling, chunk_id, begin, end);
        StorePayloadCompletion(symmetric_buffer, tiling, local_server, chunk_id, signal_ub, ready_token_end);
        // Match the remote producer protocol and stop after the first terminal frontier.
        if (ready_token_end == tiling.num_tokens) {
            break;
        }
    }
}

__aicore__ inline void PublishAllLocalTokens(
    GM_ADDR x, GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling,
    uint32_t publisher_idx, AscendC::LocalTensor<uint8_t> staging_pingpong_ub, AscendC::LocalTensor<uint64_t> signal_ub)
{
    const uint32_t source_server = tiling.rank / tiling.ranks_per_server;
    __gm__ uint8_t* source = reinterpret_cast<__gm__ uint8_t*>(x);
    __gm__ uint8_t* destination = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.x_buffer_offset +
                                  static_cast<uint64_t>(source_server) * tiling.token_capacity * tiling.hidden_bytes;

    for (uint32_t chunk_id = publisher_idx; chunk_id < tiling.num_token_chunks; chunk_id += kRemoteGetPublisherCount) {
        const uint32_t token_begin = chunk_id * tiling.num_tokens_per_chunk;
        const uint32_t token_end = token_begin + tiling.num_tokens_per_chunk < tiling.num_tokens ?
                                       token_begin + tiling.num_tokens_per_chunk :
                                       tiling.num_tokens;
        const uint32_t copy_bytes = (token_end - token_begin) * tiling.hidden_bytes;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        CopyBytes(
            destination + static_cast<uint64_t>(token_begin) * tiling.hidden_bytes,
            source + static_cast<uint64_t>(token_begin) * tiling.hidden_bytes, copy_bytes, staging_pingpong_ub, 0U,
            tiling.copy_tile_bytes);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        StorePayloadCompletion(symmetric_buffer, tiling, source_server, chunk_id, signal_ub, token_end);
    }
}

__aicore__ inline uint32_t PublisherLastChunk(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t publisher_idx)
{
    return publisher_idx +
           (tiling.num_token_chunks - 1U - publisher_idx) / kRemoteGetPublisherCount * kRemoteGetPublisherCount;
}

__aicore__ inline uint32_t PublisherReadyTokenEnd(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t publisher_idx)
{
    const uint32_t chunk_token_end = (PublisherLastChunk(tiling, publisher_idx) + 1U) * tiling.num_tokens_per_chunk;
    return chunk_token_end < tiling.num_tokens ? chunk_token_end : tiling.num_tokens;
}

__aicore__ inline bool WaitLocalPublisherStripes(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server,
    uint32_t core_idx)
{
    const uint32_t publisher_count =
        tiling.num_token_chunks < kRemoteGetPublisherCount ? tiling.num_token_chunks : kRemoteGetPublisherCount;
    for (uint32_t publisher_idx = 0; publisher_idx < publisher_count; ++publisher_idx) {
        const uint32_t last_chunk = PublisherLastChunk(tiling, publisher_idx);
        const uint32_t expected_token_end = PublisherReadyTokenEnd(tiling, publisher_idx);
        const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
        while (true) {
            const uint32_t ready_token_end =
                PollPayloadCompletionTokenEnd(symmetric_buffer, tiling, source_server, last_chunk);
            if (ready_token_end == expected_token_end) {
                break;
            }
            const uint64_t waited_cycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin;
            if (waited_cycles >= kWaitTimeoutCycles) {
                AscendC::printf(
                    "[ascend_deepep][dup_local_publish_ready_timeout] rank=%d core=%d "
                    "publisher=%d chunk=%d ready=%d expected=%d waited_cycles=%llu\n",
                    tiling.rank, core_idx, publisher_idx, last_chunk, ready_token_end, expected_token_end,
                    static_cast<unsigned long long>(waited_cycles));
                return false;
            }
        }
    }
    return true;
}

__simt_vf__ __launch_bounds__(128) inline void PublishReadyRangeVf(
    __gm__ uint64_t* destination, uint64_t value, uint32_t first_rank, uint32_t rank_count)
{
    if (threadIdx.x < rank_count) {
        simt::aclshmem_uint64_p(destination, value, static_cast<int32_t>(first_rank + threadIdx.x));
    }
    asc_threadfence();
}

__aicore__ inline void PublishReadyRange(
    __gm__ uint64_t* destination, uint64_t value, uint32_t first_rank, uint32_t rank_count)
{
    // Control publication stays off the two payload QPs. This is required for EP128, where an
    // opposite-pod ready destination can also be a Direct-GET source of the publishing rank.
    asc_vf_call<PublishReadyRangeVf>(dim3(128), destination, value, first_rank, rank_count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// Receiver-0 replaces the old 8-byte GET polling chain with a two-level active publication:
// every source rank joins its four local publisher stripes, lanes 1..7 notify the server leader,
// and the leader publishes one ready cache line to every destination rank which Direct-GETs it.
__aicore__ inline bool PublishSourceServerReady(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server,
    uint32_t core_idx)
{
    if (tiling.num_token_chunks == 0U ||
        !WaitLocalPublisherStripes(symmetric_buffer, tiling, source_server, core_idx)) {
        return tiling.num_token_chunks == 0U;
    }

    const uint32_t source_lane = tiling.rank % tiling.ranks_per_server;
    const uint32_t leader_rank = source_server * tiling.ranks_per_server;
    const uint32_t expected_token_end = PublisherReadyTokenEnd(tiling, 0U);

    if (source_lane != 0U) {
        PublishReadyRange(
            SourceRankReadySignal(symmetric_buffer, tiling, source_lane), expected_token_end, leader_rank, 1U);
        return true;
    }

    for (uint32_t lane = 1U; lane < tiling.ranks_per_server; ++lane) {
        const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
        while (true) {
            const uint32_t ready_token_end =
                PollCompletionTokenEnd(SourceRankReadySignal(symmetric_buffer, tiling, lane));
            if (ready_token_end == expected_token_end) {
                break;
            }
            const uint64_t waited_cycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin;
            if (waited_cycles >= kWaitTimeoutCycles) {
                AscendC::printf(
                    "[ascend_deepep][dup_server_publish_ready_timeout] rank=%d core=%d "
                    "source_lane=%d ready=%d expected=%d waited_cycles=%llu\n",
                    tiling.rank, core_idx, lane, ready_token_end, expected_token_end,
                    static_cast<unsigned long long>(waited_cycles));
                return false;
            }
        }
    }

    for (uint32_t destination_server = 0U; destination_server < tiling.num_servers; ++destination_server) {
        if (!ServerUsesRemoteGet(tiling, source_server, destination_server)) {
            continue;
        }
        PublishReadyRange(
            SourceServerReadySignal(symmetric_buffer, tiling, source_server), expected_token_end,
            destination_server * tiling.ranks_per_server, tiling.ranks_per_server);
    }
    return true;
}

// Gated mode keeps the control AIV live until copy, RX and TX publication have all progressed.
// None of these joins waits for GET completion or output copies, so GET cannot block its own release.
__aicore__ inline bool PublishRemoteGetGates(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server)
{
    if (tiling.num_token_chunks == 0U) {
        return true;
    }
    const uint32_t lane = tiling.rank % tiling.ranks_per_server;
    const uint32_t leader = source_server * tiling.ranks_per_server;
    const uint32_t expected_copy = PublisherReadyTokenEnd(tiling, 0U);
    const uint32_t publisher_count =
        tiling.num_token_chunks < kRemoteGetPublisherCount ? tiling.num_token_chunks : kRemoteGetPublisherCount;
    uint32_t rx_cursor[kMaxServers] = {};
    uint64_t rx_done_mask = 0U;
    uint32_t rx_lane_mask = 0U, tx_lane_mask = 0U;
    const uint32_t all_lanes = (1U << tiling.ranks_per_server) - 1U;
    bool copy_arrived = false, rx_arrived = false, tx_arrived = false;
    bool copy_released = false, rx_released = false, tx_released = false;
    const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        if (!copy_arrived) {
            bool ready = true;
            for (uint32_t publisher = 0U; publisher < publisher_count; ++publisher) {
                ready &= PollPayloadCompletionTokenEnd(
                             symmetric_buffer, tiling, source_server, PublisherLastChunk(tiling, publisher)) ==
                         PublisherReadyTokenEnd(tiling, publisher);
            }
            if (ready) {
                if (lane != 0U) {
                    PublishReadyRange(SourceRankReadySignal(symmetric_buffer, tiling, lane), expected_copy, leader, 1U);
                }
                copy_arrived = true;
            }
        }
        if (!tx_arrived) {
            bool ready = true;
            for (uint32_t destination = 0U; destination < tiling.num_servers; ++destination) {
                if (destination != source_server && !ServerUsesRemoteGet(tiling, source_server, destination)) {
                    ready &= PollCompletionTokenEnd(PeerPutDoneSignal(symmetric_buffer, tiling, destination)) != 0U;
                }
            }
            if (ready) {
                PublishReadyRange(RemoteGetGateSignal(symmetric_buffer, tiling, kGateTxArrival + lane), 1U, leader, 1U);
                tx_arrived = true;
            }
        }
        if (!rx_arrived) {
            bool ready = true;
            for (uint32_t incoming = 0U; incoming < tiling.num_servers; ++incoming) {
                if (incoming == source_server || ServerUsesRemoteGet(tiling, incoming, source_server)) {
                    continue;
                }
                const uint64_t bit = uint64_t{1} << incoming;
                if ((rx_done_mask & bit) == 0U) {
                    // A compact sender may terminate before the last allocated chunk. Walk only
                    // the contiguous dual-QP frontier, recognizing num_tokens as the terminal value.
                    const uint32_t chunk = rx_cursor[incoming];
                    if (chunk < tiling.num_token_chunks) {
                        const uint32_t frontier =
                            PollPayloadCompletionTokenEnd(symmetric_buffer, tiling, incoming, chunk);
                        if (frontier == tiling.num_tokens) {
                            rx_done_mask |= bit;
                        } else if (frontier != 0U) {
                            ++rx_cursor[incoming];
                        }
                    }
                }
                ready &= (rx_done_mask & bit) != 0U;
            }
            if (ready) {
                PublishReadyRange(RemoteGetGateSignal(symmetric_buffer, tiling, kGateRxArrival + lane), 1U, leader, 1U);
                rx_arrived = true;
            }
        }
        if (lane == 0U) {
            if (!copy_released && copy_arrived) {
                bool ready = true;
                for (uint32_t peer_lane = 1U; peer_lane < tiling.ranks_per_server; ++peer_lane) {
                    ready &= PollCompletionTokenEnd(SourceRankReadySignal(symmetric_buffer, tiling, peer_lane)) ==
                             expected_copy;
                }
                if (ready) {
                    for (uint32_t destination = 0U; destination < tiling.num_servers; ++destination) {
                        if (ServerUsesRemoteGet(tiling, source_server, destination)) {
                            PublishReadyRange(
                                SourceServerReadySignal(symmetric_buffer, tiling, source_server), expected_copy,
                                destination * tiling.ranks_per_server, tiling.ranks_per_server);
                        }
                    }
                    copy_released = true;
                }
            }
            for (uint32_t peer_lane = 0U; peer_lane < tiling.ranks_per_server; ++peer_lane) {
                const uint32_t bit = 1U << peer_lane;
                if ((rx_lane_mask & bit) == 0U && PollCompletionTokenEnd(RemoteGetGateSignal(
                                                      symmetric_buffer, tiling, kGateRxArrival + peer_lane)) != 0U) {
                    rx_lane_mask |= bit;
                }
                if ((tx_lane_mask & bit) == 0U && PollCompletionTokenEnd(RemoteGetGateSignal(
                                                      symmetric_buffer, tiling, kGateTxArrival + peer_lane)) != 0U) {
                    tx_lane_mask |= bit;
                }
            }
            if (!rx_released && rx_lane_mask == all_lanes) {
                PublishReadyRange(
                    RemoteGetGateSignal(symmetric_buffer, tiling, kGateRxRelease), 1U, leader, tiling.ranks_per_server);
                rx_released = true;
            }
            if (!tx_released && tx_lane_mask == all_lanes) {
                for (uint32_t destination = 0U; destination < tiling.num_servers; ++destination) {
                    if (ServerUsesRemoteGet(tiling, source_server, destination)) {
                        PublishReadyRange(
                            RemoteGetGateSignal(symmetric_buffer, tiling, kGateTxRelease + source_server), 1U,
                            destination * tiling.ranks_per_server, tiling.ranks_per_server);
                    }
                }
                tx_released = true;
            }
            if (copy_released && rx_released && tx_released) {
                return true;
            }
        } else if (copy_arrived && rx_arrived && tx_arrived) {
            return true;
        }
        const uint64_t waited = static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin;
        if (waited >= kWaitTimeoutCycles) {
            AscendC::printf(
                "[ascend_deepep][dup_gate_publish_timeout] rank=%d copy=%d rx=%d tx=%d rx_mask=%u tx_mask=%u\n",
                tiling.rank, copy_arrived, rx_arrived, tx_arrived, rx_lane_mask, tx_lane_mask);
            return false;
        }
    }
}

__aicore__ inline bool WaitRemoteGetGates(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server)
{
    if (tiling.remote_get_rx_tx_gate == 0U || tiling.num_token_chunks == 0U) {
        return true;
    }
    const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        const uint32_t rx = PollCompletionTokenEnd(RemoteGetGateSignal(symmetric_buffer, tiling, kGateRxRelease));
        const uint32_t tx =
            PollCompletionTokenEnd(RemoteGetGateSignal(symmetric_buffer, tiling, kGateTxRelease + source_server));
        if (rx != 0U && tx != 0U) {
            return true;
        }
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin >= kWaitTimeoutCycles) {
            AscendC::printf(
                "[ascend_deepep][dup_get_gate_timeout] rank=%d source=%d rx=%u tx=%u\n", tiling.rank, source_server, rx,
                tx);
            return false;
        }
    }
}

__aicore__ inline bool WaitSourceServerReady(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server,
    uint32_t direct_core)
{
    if (tiling.num_token_chunks == 0U) {
        return true;
    }
    const uint32_t expected_token_end = PublisherReadyTokenEnd(tiling, 0U);
    const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        const uint32_t ready_token_end =
            PollCompletionTokenEnd(SourceServerReadySignal(symmetric_buffer, tiling, source_server));
        if (ready_token_end == expected_token_end) {
            return true;
        }
        const uint64_t waited_cycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin;
        if (waited_cycles >= kWaitTimeoutCycles) {
            AscendC::printf(
                "[ascend_deepep][dup_remote_get_server_ready_timeout] rank=%d core=%d "
                "source_server=%d ready=%d expected=%d waited_cycles=%llu\n",
                tiling.rank, direct_core, source_server, ready_token_end, expected_token_end,
                static_cast<unsigned long long>(waited_cycles));
            return false;
        }
    }
}

__aicore__ inline bool WaitPeerPutDone(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t peer_server,
    uint32_t direct_core)
{
    const uint32_t local_server = tiling.rank / tiling.ranks_per_server;
    if (ServerUsesRemoteGet(tiling, local_server, peer_server)) {
        return true;
    }
    const uint64_t wait_begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        const uint32_t put_done = PollCompletionTokenEnd(PeerPutDoneSignal(symmetric_buffer, tiling, peer_server));
        if (put_done != 0U) {
            return true;
        }
        const uint64_t waited_cycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) - wait_begin;
        if (waited_cycles >= kWaitTimeoutCycles) {
            AscendC::printf(
                "[ascend_deepep][dup_remote_get_peer_put_timeout] rank=%d core=%d "
                "peer_server=%d waited_cycles=%llu\n",
                tiling.rank, direct_core, peer_server, static_cast<unsigned long long>(waited_cycles));
            return false;
        }
    }
}

__aicore__ inline uint32_t DirectGetSourceServer(
    GM_ADDR recv_route_position, GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t source_server, uint32_t direct_core,
    __ubuf__ uint8_t* qp0_wqe_buffer, __ubuf__ uint8_t* qp1_wqe_buffer, AscendC::LocalTensor<int32_t> route_position_ub,
    AscendC::LocalTensor<int32_t> segment_pairs_ub)
{
    AscendC::GlobalTensor<int32_t> route_position_gm;
    AscendC::GlobalTensor<int32_t> segment_offset_gm;
    route_position_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_route_position));
    segment_offset_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_segment_offset));
    __gm__ uint8_t* symmetric_payload_base = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer);
    __gm__ uint8_t* destination_base = reinterpret_cast<__gm__ uint8_t*>(recv_x);

    uint32_t server_total_rows = 0U;
    if (!WaitPeerPutDone(symmetric_buffer, tiling, source_server, direct_core) ||
        !WaitRemoteGetGates(symmetric_buffer, tiling, source_server) ||
        !WaitSourceServerReady(symmetric_buffer, tiling, source_server, direct_core)) {
        return server_total_rows;
    }
    uint32_t used_qps[kRemoteGetReceiverCount] = {};
    const uint32_t destination_lane = tiling.rank % tiling.ranks_per_server;
    for (uint32_t issued = 0U; issued < tiling.ranks_per_server; ++issued) {
        const uint32_t source_lane = (destination_lane + issued) % tiling.ranks_per_server;
        const uint32_t source_rank = source_server * tiling.ranks_per_server + source_lane;
        uint32_t route_index = 0U;
        for (uint32_t first_expert = 0U; first_expert < tiling.num_local_experts;
             first_expert += kDirectGetExpertBatch) {
            const uint32_t expert_count = tiling.num_local_experts - first_expert < kDirectGetExpertBatch ?
                                              tiling.num_local_experts - first_expert :
                                              kDirectGetExpertBatch;
            const uint64_t first_segment = static_cast<uint64_t>(first_expert) * tiling.world_size + source_rank;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            // Adjacent expert pairs are world_size int32 elements apart in this metadata layout.
            AscendC::DataCopyExtParams pair_params(
                expert_count, 2U * sizeof(int32_t), (tiling.world_size - 2U) * sizeof(int32_t), 0, 0);
            AscendC::DataCopyPad(
                segment_pairs_ub, segment_offset_gm[first_segment], pair_params,
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            uint32_t total_rows = 0U;
            for (uint32_t expert = 0U; expert < expert_count; ++expert) {
                total_rows += static_cast<uint32_t>(
                    segment_pairs_ub.GetValue(expert * 8U + 1U) - segment_pairs_ub.GetValue(expert * 8U));
            }
            server_total_rows += total_rows;
            // Preserve the source-card-wide 0,0,0,1 sequence across expert batches.
            const uint32_t total_qp1_rows = (route_index + total_rows) / 4U - route_index / 4U;
            const uint32_t total_qp0_rows = total_rows - total_qp1_rows;
            uint32_t issued_qp0_rows = 0U;
            uint32_t issued_qp1_rows = 0U;
            uint32_t pending_qp0_wqes = 0U;
            uint32_t pending_qp1_wqes = 0U;
            aclshmemx_submit_state_t qp0_submit_state{};
            aclshmemx_submit_state_t qp1_submit_state{};
            aclshmemx_defer_t qp0_defer(qp0_submit_state);
            aclshmemx_defer_t qp1_defer(qp1_submit_state);
            aclshmemx_submit_t qp0_submit(qp0_submit_state);
            aclshmemx_submit_t qp1_submit(qp1_submit_state);

            for (uint32_t expert = 0U; expert < expert_count; ++expert) {
                const uint32_t segment_begin = static_cast<uint32_t>(segment_pairs_ub.GetValue(expert * 8U));
                const uint32_t segment_end = static_cast<uint32_t>(segment_pairs_ub.GetValue(expert * 8U + 1U));
                for (uint32_t tile_begin = segment_begin; tile_begin < segment_end; tile_begin += kDirectGetRouteRows) {
                    const uint32_t remaining_rows = segment_end - tile_begin;
                    const uint32_t tile_rows =
                        remaining_rows < kDirectGetRouteRows ? remaining_rows : kDirectGetRouteRows;
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                    AscendC::DataCopyExtParams copy_params(1, tile_rows * sizeof(int32_t), 0, 0, 0);
                    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                    AscendC::DataCopyPad(route_position_ub, route_position_gm[tile_begin], copy_params, pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);

                    for (uint32_t row = 0; row < tile_rows; ++row, ++route_index) {
                        const uint32_t output_index = tile_begin + row;
                        const uint32_t route_position = static_cast<uint32_t>(route_position_ub.GetValue(row));
                        const uint32_t token_id = RoutePositionTokenId(route_position, tiling);
                        const uint64_t source_slot =
                            static_cast<uint64_t>(source_server) * tiling.token_capacity + token_id;
                        __gm__ uint8_t* destination =
                            destination_base + static_cast<uint64_t>(output_index) * tiling.hidden_bytes;
                        __gm__ uint8_t* source =
                            symmetric_payload_base + tiling.x_buffer_offset + source_slot * tiling.hidden_bytes;
                        if ((route_index & 3U) != 3U) {
                            ++issued_qp0_rows;
                            ++pending_qp0_wqes;
                            const bool submit =
                                issued_qp0_rows == total_qp0_rows || pending_qp0_wqes == kUdmaWqeBatchSize;
                            if (submit) {
                                aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataCqeConfig>(
                                    destination, source, qp0_wqe_buffer, tiling.hidden_bytes,
                                    static_cast<int32_t>(source_rank), 0U, EVENT_ID5, qp0_submit);
                                pending_qp0_wqes = 0U;
                            } else {
                                aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataNoCqeConfig>(
                                    destination, source, qp0_wqe_buffer, tiling.hidden_bytes,
                                    static_cast<int32_t>(source_rank), 0U, EVENT_ID5, qp0_defer);
                            }
                        } else {
                            ++issued_qp1_rows;
                            ++pending_qp1_wqes;
                            const bool submit =
                                issued_qp1_rows == total_qp1_rows || pending_qp1_wqes == kUdmaWqeBatchSize;
                            if (submit) {
                                aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataCqeConfig>(
                                    destination, source, qp1_wqe_buffer, tiling.hidden_bytes,
                                    static_cast<int32_t>(source_rank), 1U, EVENT_ID5, qp1_submit);
                                pending_qp1_wqes = 0U;
                            } else {
                                aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataNoCqeConfig>(
                                    destination, source, qp1_wqe_buffer, tiling.hidden_bytes,
                                    static_cast<int32_t>(source_rank), 1U, EVENT_ID5, qp1_defer);
                            }
                        }
                    }
                }
            }
            used_qps[issued] |= (total_qp0_rows != 0U ? 1U : 0U) | (total_qp1_rows != 0U ? 2U : 0U);
        }
    }
    // Post every source-card request before the first blocking quiet. Rotating by destination lane
    // prevents all ranks from synchronously concentrating on source lane zero.
    for (uint32_t drained = 0U; drained < tiling.ranks_per_server; ++drained) {
        const uint32_t source_lane = (destination_lane + drained) % tiling.ranks_per_server;
        const uint32_t source_rank = source_server * tiling.ranks_per_server + source_lane;
        for (uint32_t qp_slot = 0U; qp_slot < kUdmaQpCount; ++qp_slot) {
            if ((used_qps[drained] & (1U << qp_slot)) != 0U) {
                aclshmemx_udma_qp_quiet(static_cast<int32_t>(source_rank), qp_slot);
            }
        }
    }
    return server_total_rows;
}

__aicore__ inline uint64_t FindReadyCompletion(
    __gm__ uint8_t* gateway_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling,
    AscendC::LocalTensor<uint32_t> next_chunk_ub, uint64_t pending_server_mask, uint32_t scan_begin,
    uint64_t& signal_reads)
{
    const uint32_t num_servers = tiling.num_servers;
    for (uint32_t offset = 0; offset < num_servers; ++offset) {
        const uint32_t source_server = (scan_begin + offset) % num_servers;
        const uint64_t source_mask = uint64_t{1} << source_server;
        if ((pending_server_mask & source_mask) == 0U) {
            continue;
        }
        const uint32_t chunk_id = next_chunk_ub.GetValue(source_server);
        if (tiling.enable_timer != 0U) {
            signal_reads += kUdmaQpCount;
        }
        const uint32_t ready_token_end = PollPayloadCompletionTokenEnd(gateway_buffer, tiling, source_server, chunk_id);
        if (ready_token_end != 0U) {
            return (static_cast<uint64_t>(ready_token_end) << 32U) | source_server;
        }
    }
    return 0U;
}

// This state belongs to one receiver AIV and its fixed intranode (gateway_rank, QP0).
// Payload slots are not recycled during dispatch, so a batch may span ready sources/chunks.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
struct GatewayGetBatch {
    aclshmemx_submit_state_t submit_state{};
    uint32_t pending_wqes = 0U;
    __gm__ uint8_t* last_source = nullptr;
    __gm__ uint8_t* last_destination = nullptr;
};
#else
// shmem.h exposes aggregate UDMA types only in the device pass. Host parsing needs the name
// for __aicore__ function signatures, but must not instantiate the device-only state.
struct GatewayGetBatch;
#endif

__aicore__ inline void FlushGatewayGetBatch(
    uint32_t gateway_rank, __ubuf__ uint8_t* udma_wqe_buffer, GatewayGetBatch& batch)
{
    if (batch.pending_wqes == 0U) {
        return;
    }
    // At most 127 payload WQEs remain. Re-read 8 bytes of the last valid row to submit the tail;
    // this is RO, not a quiet or SO fence. Final SO + quiet still protect output/source lifetime.
    aclshmemx_submit_t submit_action(batch.submit_state);
    aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataCqeConfig>(
        batch.last_destination, batch.last_source, udma_wqe_buffer, 8U, static_cast<int32_t>(gateway_rank), 0U,
        EVENT_ID5, submit_action);
    batch.pending_wqes = 0U;
}

__aicore__ inline uint32_t GatewayRouteTileRows(
    uint32_t remaining_rows, uint32_t token_begin, uint32_t token_end, uint32_t num_tokens)
{
    if (token_begin >= num_tokens) {
        return kGatewayRouteRows;
    }
    const uint32_t remaining_tokens = num_tokens - token_begin;
    const uint32_t bounded_end = token_end < num_tokens ? token_end : num_tokens;
    const uint32_t span = bounded_end > token_begin ? bounded_end - token_begin : 0U;
    const uint64_t expected = (static_cast<uint64_t>(remaining_rows) * span + remaining_tokens - 1U) / remaining_tokens;
    if (expected >= kGatewayRouteRows) {
        return kGatewayRouteRows;
    }
    const uint32_t aligned = (static_cast<uint32_t>(expected) + 31U) / 32U * 32U;
    return aligned < 32U ? 32U : aligned;
}

__aicore__ inline uint64_t ReceiveGatewaySourceRoutes(
    AscendC::GlobalTensor<int32_t> route_position_gm, __gm__ uint8_t* recv_x, __gm__ uint8_t* gateway_buffer,
    __gm__ uint8_t* symmetric_payload_base, const ascend_deepep::DupServerScaleOutTilingData& tiling,
    uint32_t source_server, uint32_t ready_token_begin, uint32_t ready_token_end, uint32_t local_expert_begin,
    uint32_t local_expert_end, AscendC::LocalTensor<uint8_t> staging_pingpong_ub,
    AscendC::LocalTensor<uint32_t> route_cursor_ub, AscendC::LocalTensor<uint32_t> segment_end_ub,
    AscendC::LocalTensor<int32_t> route_position_ub, uint32_t staging_buffer_idx, bool use_remote_get,
    uint32_t gateway_rank, __ubuf__ uint8_t* udma_wqe_buffer, GatewayGetBatch& get_batch, uint32_t& route_rows,
    ReceiverStageCycles& stage_cycles)
{
    const uint32_t route_end_value = ready_token_end * tiling.num_topk;
    const bool profile = tiling.enable_timer != 0U;
    aclshmemx_defer_t defer_action(get_batch.submit_state);
    aclshmemx_submit_t submit_action(get_batch.submit_state);
    uint32_t last_get_output_end = 0U;

    for (uint32_t local_expert = local_expert_begin; local_expert < local_expert_end; ++local_expert) {
        const uint32_t cursor_index = source_server * tiling.num_local_experts + local_expert;
        const uint32_t segment_end = segment_end_ub.GetValue(cursor_index);
        const uint32_t chunk_begin = route_cursor_ub.GetValue(cursor_index);
        uint32_t chunk_end = chunk_begin;
        uint32_t tile_limit =
            use_remote_get ?
                GatewayRouteTileRows(segment_end - chunk_begin, ready_token_begin, ready_token_end, tiling.num_tokens) :
                kGatewayRouteRows;
        while (chunk_end < segment_end) {
            uint64_t route_scan_start = 0U;
            if (profile) {
                route_scan_start = static_cast<uint64_t>(AscendC::GetSystemCycle());
            }
            const uint32_t remaining_rows = segment_end - chunk_end;
            const uint32_t tile_rows = remaining_rows < tile_limit ? remaining_rows : tile_limit;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::DataCopyExtParams route_copy_params(1, tile_rows * sizeof(int32_t), 0, 0, 0);
            AscendC::DataCopyPadExtParams<int32_t> route_pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(route_position_ub, route_position_gm[chunk_end], route_copy_params, route_pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);

            uint32_t ready_rows = 0U;
            for (; ready_rows < tile_rows; ++ready_rows) {
                const uint32_t route_position = static_cast<uint32_t>(route_position_ub.GetValue(ready_rows));
                if (route_position >= route_end_value) {
                    break;
                }
            }
            if (profile) {
                stage_cycles.route_scan_cycles += static_cast<uint64_t>(AscendC::GetSystemCycle()) - route_scan_start;
            }

            if (ready_rows != 0U) {
                uint64_t transfer_start = 0U;
                if (profile) {
                    transfer_start = static_cast<uint64_t>(AscendC::GetSystemCycle());
                }
                if (use_remote_get) {
                    for (uint32_t row = 0; row < ready_rows; ++row) {
                        const uint32_t output_index = chunk_end + row;
                        const uint32_t route_position = static_cast<uint32_t>(route_position_ub.GetValue(row));
                        const uint32_t token_id = RoutePositionTokenId(route_position, tiling);
                        const uint64_t source_slot =
                            static_cast<uint64_t>(source_server) * tiling.token_capacity + token_id;
                        __gm__ uint8_t* destination =
                            recv_x + static_cast<uint64_t>(output_index) * tiling.hidden_bytes;
                        __gm__ uint8_t* source =
                            symmetric_payload_base + tiling.x_buffer_offset + source_slot * tiling.hidden_bytes;
                        ++get_batch.pending_wqes;
                        if (get_batch.pending_wqes == kUdmaWqeBatchSize) {
                            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataCqeConfig>(
                                destination, source, udma_wqe_buffer, tiling.hidden_bytes,
                                static_cast<int32_t>(gateway_rank), 0U, EVENT_ID5, submit_action);
                            get_batch.pending_wqes = 0U;
                        } else {
                            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_MTE3, kDataNoCqeConfig>(
                                destination, source, udma_wqe_buffer, tiling.hidden_bytes,
                                static_cast<int32_t>(gateway_rank), 0U, EVENT_ID5, defer_action);
                        }
                        get_batch.last_source = source;
                        get_batch.last_destination = destination;
                        last_get_output_end = output_index + 1U;
                    }
                    if (profile) {
                        stage_cycles.remote_get_submit_cycles +=
                            static_cast<uint64_t>(AscendC::GetSystemCycle()) - transfer_start;
                    }
                } else {
                    staging_buffer_idx = CopyCachedRouteRows(
                        route_position_ub, recv_x, gateway_buffer + tiling.x_buffer_offset, tiling, source_server,
                        chunk_end, ready_rows, staging_pingpong_ub, staging_buffer_idx, tiling.copy_tile_bytes);
                    if (profile) {
                        stage_cycles.local_mte_copy_cycles +=
                            static_cast<uint64_t>(AscendC::GetSystemCycle()) - transfer_start;
                    }
                }
            }
            chunk_end += ready_rows;
            if (ready_rows < tile_rows) {
                break;
            }
            // Grow after a dense tile, but still test every route against the actual frontier.
            if (tile_limit < kGatewayRouteRows) {
                tile_limit = tile_limit < kGatewayRouteRows / 2U ? tile_limit * 2U : kGatewayRouteRows;
            }
        }
        if (profile) {
            route_rows += chunk_end - chunk_begin;
        }
        route_cursor_ub.SetValue(cursor_index, chunk_end);
    }

    return (static_cast<uint64_t>(staging_buffer_idx) << 32U) | last_get_output_end;
}

__aicore__ inline void ReceiveGatewayRoutes(
    GM_ADDR recv_route_position, GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t core_idx, uint32_t worker_idx,
    uint32_t gateway_lane, uint32_t local_expert_begin, uint32_t local_expert_end, uint64_t initial_server_mask,
    AscendC::LocalTensor<uint8_t> staging_pingpong_ub, AscendC::LocalTensor<uint32_t> next_chunk_ub,
    AscendC::LocalTensor<uint32_t> route_cursor_ub, AscendC::LocalTensor<uint32_t> segment_end_ub,
    AscendC::LocalTensor<int32_t> route_position_ub, __ubuf__ uint8_t* udma_wqe_buffer,
    AscendC::GlobalTensor<uint64_t> receiver_trace_gm, ReceiverTimerData& timer, bool record_trace)
{
    if (local_expert_begin >= local_expert_end) {
        return;
    }
    if (tiling.enable_timer != 0U) {
        timer.entry_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    AscendC::GlobalTensor<int32_t> route_position_gm;
    AscendC::GlobalTensor<int32_t> segment_offset_gm;
    route_position_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_route_position));
    segment_offset_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recv_segment_offset));

    // Remote gateway workers own one complete gateway lane. In the remote-GET variant, four
    // publisher workers instead split the local gateway by expert, so every recv_x row still has
    // exactly one writer.
    const uint32_t destination_server = tiling.rank / tiling.ranks_per_server;
    const uint32_t gateway_rank = destination_server * tiling.ranks_per_server + gateway_lane;
    const bool is_local_gateway = gateway_rank == tiling.rank;
    const bool use_remote_get = !is_local_gateway && tiling.use_urma_get != 0U;
    __gm__ uint8_t* symmetric_payload_base = reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer);
    __gm__ uint8_t* gateway_buffer = symmetric_payload_base;
    if (!is_local_gateway) {
        gateway_buffer = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(gateway_buffer, gateway_rank));
    }

    uint32_t staging_buffer_idx = 0U;
    if (!use_remote_get) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    uint64_t pending_server_mask = initial_server_mask;
    for (uint32_t source_server = 0; source_server < tiling.num_servers; ++source_server) {
        next_chunk_ub.SetValue(source_server, 0U);
        if ((initial_server_mask & (uint64_t{1} << source_server)) == 0U) {
            continue;
        }
        const uint32_t source_rank = source_server * tiling.ranks_per_server + gateway_lane;
        for (uint32_t first = local_expert_begin; first < local_expert_end; first += kGatewaySegmentBatch) {
            const uint32_t count =
                local_expert_end - first < kGatewaySegmentBatch ? local_expert_end - first : kGatewaySegmentBatch;
            const uint64_t segment = static_cast<uint64_t>(first) * tiling.world_size + source_rank;
            // Reuse the 1024-byte route scratch only during setup: 32 pairs, padded to 32 bytes each.
            // Unlike elastic's [expert][world+1] layout, this table has a world_size stride.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::DataCopyExtParams pair_params(
                count, 2U * sizeof(int32_t), (tiling.world_size - 2U) * sizeof(int32_t), 0, 0);
            AscendC::DataCopyPad(
                route_position_ub, segment_offset_gm[segment], pair_params,
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            for (uint32_t e = 0U; e < count; ++e) {
                const uint32_t cursor_index = source_server * tiling.num_local_experts + first + e;
                route_cursor_ub.SetValue(cursor_index, static_cast<uint32_t>(route_position_ub.GetValue(e * 8U)));
                segment_end_ub.SetValue(cursor_index, static_cast<uint32_t>(route_position_ub.GetValue(e * 8U + 1U)));
            }
        }
    }

    // Each worker consumes contiguous ready chunk ranges in order and keeps independent per-source route cursors.
    GatewayGetBatch get_batch;
    uint32_t source_token_begin[kMaxServers] = {};
    uint32_t last_get_source_server = 0U;
    uint32_t last_get_output_end = 0U;
    uint32_t scan_begin = (destination_server + worker_idx) % tiling.num_servers;
    uint64_t profile_wait_start = 0U;
    if (tiling.enable_timer != 0U) {
        timer.receive_start_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        timer.init_cycles = timer.receive_start_cycle - timer.entry_cycle;
        profile_wait_start = timer.receive_start_cycle;
    }
    uint64_t last_progress_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    uint64_t wait_poll_calls = 0U;
    uint64_t wait_signal_reads = 0U;
    while (pending_server_mask != 0U) {
        if (tiling.enable_timer != 0U) {
            ++wait_poll_calls;
        }
        const uint64_t ready_completion = FindReadyCompletion(
            gateway_buffer, tiling, next_chunk_ub, pending_server_mask, scan_begin, wait_signal_reads);
        if (ready_completion == 0U) {
            // FindReadyCompletion checked every pending source. Submit a short batch before
            // idle polling so already-ready payloads are not held behind a slow producer.
            if (use_remote_get) {
                FlushGatewayGetBatch(gateway_rank, udma_wqe_buffer, get_batch);
            }
            const uint64_t idle_cycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) - last_progress_cycle;
            if (idle_cycles >= kWaitTimeoutCycles) {
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timeout] rank=%d core=%d gateway_rank=%d "
                    "pending_servers=0x%llx idle_cycles=%llu poll_calls=%llu signal_reads=%llu\n",
                    tiling.rank, core_idx, gateway_rank, static_cast<unsigned long long>(pending_server_mask),
                    static_cast<unsigned long long>(idle_cycles), static_cast<unsigned long long>(wait_poll_calls),
                    static_cast<unsigned long long>(wait_signal_reads));
                if (tiling.enable_timer != 0U) {
                    const uint64_t profile_end = static_cast<uint64_t>(AscendC::GetSystemCycle());
                    timer.wait_cycles += profile_end - profile_wait_start;
                    timer.receive_end_cycle = profile_end;
                    timer.final_drain_begin_cycle = profile_end;
                    timer.final_submit_done_cycle = profile_end;
                    timer.total_cycles = profile_end - timer.receive_start_cycle;
                }
                return;
            }
            continue;
        }

        const uint32_t source_server = static_cast<uint32_t>(ready_completion);
        uint32_t ready_token_end = static_cast<uint32_t>(ready_completion >> 32U);
        const uint32_t chunk_begin = next_chunk_ub.GetValue(source_server);
        uint32_t chunk_end = chunk_begin + 1U;
        // Collapse only a contiguous prefix after both QPs report the same ready frontier.
        while (ready_token_end != tiling.num_tokens && chunk_end < tiling.num_token_chunks) {
            if (tiling.enable_timer != 0U) {
                wait_signal_reads += kUdmaQpCount;
            }
            const uint32_t next_ready_token_end =
                PollPayloadCompletionTokenEnd(gateway_buffer, tiling, source_server, chunk_end);
            if (next_ready_token_end == 0U) {
                break;
            }
            ready_token_end = next_ready_token_end;
            ++chunk_end;
        }

        uint64_t ready_cycle = 0U;
        if (tiling.enable_timer != 0U) {
            ready_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
            timer.wait_cycles += ready_cycle - profile_wait_start;
        }
        uint32_t route_rows = 0U;
        ReceiverStageCycles stage_cycles{};
        const uint64_t receive_result = ReceiveGatewaySourceRoutes(
            route_position_gm, reinterpret_cast<__gm__ uint8_t*>(recv_x), gateway_buffer, symmetric_payload_base,
            tiling, source_server, source_token_begin[source_server], ready_token_end, local_expert_begin,
            local_expert_end, staging_pingpong_ub, route_cursor_ub, segment_end_ub, route_position_ub,
            staging_buffer_idx, use_remote_get, gateway_rank, udma_wqe_buffer, get_batch, route_rows, stage_cycles);
        source_token_begin[source_server] = ready_token_end;
        uint64_t issue_done_cycle = 0U;
        if (tiling.enable_timer != 0U) {
            issue_done_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
            timer.route_issue_cycles += issue_done_cycle - ready_cycle;
            timer.route_scan_cycles += stage_cycles.route_scan_cycles;
            timer.remote_get_submit_cycles += stage_cycles.remote_get_submit_cycles;
            timer.local_mte_copy_cycles += stage_cycles.local_mte_copy_cycles;
        }
        staging_buffer_idx = static_cast<uint32_t>(receive_result >> 32U);
        const uint32_t get_output_end = static_cast<uint32_t>(receive_result);
        if (get_output_end != 0U) {
            last_get_source_server = source_server;
            last_get_output_end = get_output_end;
        }
        next_chunk_ub.SetValue(source_server, chunk_end);
        if (ready_token_end == tiling.num_tokens || chunk_end == tiling.num_token_chunks) {
            pending_server_mask &= ~(uint64_t{1} << source_server);
        }
        scan_begin = (source_server + 1U) % tiling.num_servers;
        const uint64_t bookkeeping_done_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        last_progress_cycle = bookkeeping_done_cycle;
        if (tiling.enable_timer != 0U) {
            timer.bookkeeping_cycles += bookkeeping_done_cycle - issue_done_cycle;
            uint64_t next_wait_cycle = bookkeeping_done_cycle;
            if (record_trace && tiling.rank == 0U) {
                next_wait_cycle = StoreReceiverTrace(
                    receiver_trace_gm, timer.trace_count, source_server, chunk_begin, chunk_end, profile_wait_start,
                    ready_cycle, issue_done_cycle);
                timer.trace_record_cycles += next_wait_cycle - bookkeeping_done_cycle;
                ++timer.trace_count;
            }
            wait_poll_calls = 0U;
            wait_signal_reads = 0U;
            if (pending_server_mask != 0U) {
                profile_wait_start = next_wait_cycle;
            }
        }
    }

    uint64_t drain_begin_cycle = 0U;
    if (tiling.enable_timer != 0U) {
        drain_begin_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        timer.final_drain_begin_cycle = drain_begin_cycle;
    }
    if (!use_remote_get) {
        if (tiling.enable_timer != 0U) {
            timer.final_submit_done_cycle = drain_begin_cycle;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    } else if (last_get_output_end != 0U) {
        FlushGatewayGetBatch(gateway_rank, udma_wqe_buffer, get_batch);
        const uint32_t output_index = last_get_output_end - 1U;
        const uint32_t route_position = static_cast<uint32_t>(route_position_gm.GetValue(output_index));
        const uint32_t token_id = RoutePositionTokenId(route_position, tiling);
        const uint64_t source_slot = static_cast<uint64_t>(last_get_source_server) * tiling.token_capacity + token_id;
        __gm__ uint8_t* destination =
            reinterpret_cast<__gm__ uint8_t*>(recv_x) + static_cast<uint64_t>(output_index) * tiling.hidden_bytes;
        __gm__ uint8_t* source = symmetric_payload_base + tiling.x_buffer_offset + source_slot * tiling.hidden_bytes;

        // A final immediate SO GET orders all earlier aggregate RO GETs before quiet returns.
        aclshmemx_udma_qp_get_nbi<uint64_t, PIPE_MTE3, kGetCompletionCqeConfig>(
            reinterpret_cast<__gm__ uint64_t*>(destination), reinterpret_cast<__gm__ uint64_t*>(source),
            reinterpret_cast<__ubuf__ uint64_t*>(udma_wqe_buffer), 1U, static_cast<int32_t>(gateway_rank), 0U,
            EVENT_ID5);
        if (tiling.enable_timer != 0U) {
            timer.final_submit_done_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        }
        aclshmemx_udma_qp_quiet(static_cast<int32_t>(gateway_rank), 0U);
    } else if (tiling.enable_timer != 0U) {
        timer.final_submit_done_cycle = drain_begin_cycle;
    }
    if (tiling.enable_timer != 0U) {
        timer.receive_end_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        timer.final_drain_cycles = timer.receive_end_cycle - drain_begin_cycle;
        timer.final_submit_cycles = timer.final_submit_done_cycle - drain_begin_cycle;
        timer.final_wait_cycles = timer.receive_end_cycle - timer.final_submit_done_cycle;
        timer.total_cycles = timer.receive_end_cycle - timer.receive_start_cycle;
    }
}

__aicore__ inline void DupServerNotifyScaleOut(
    GM_ADDR symmetric_buffer, const ascend_deepep::DupServerScaleOutTilingData& tiling)
{
    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t core_num = AscendC::GetBlockNum();
    ascend_deepep::UbWorkspace resource;
    AscendC::LocalTensor<uint64_t> signal_ub = resource.template GetBufferByByte<uint64_t>(0);

    // Do not clear receiver-owned signals while a slower rank may still consume the previous dispatch.
    aclshmemi_sync_npu_v4<true>(0);

    const uint32_t completion_owner_count = tiling.num_servers;
    const uint64_t signal_count = static_cast<uint64_t>(completion_owner_count) * tiling.chunk_capacity;
    for (uint64_t signal_index = core_idx; signal_index < signal_count; signal_index += core_num) {
        const uint32_t publisher = static_cast<uint32_t>(signal_index / tiling.chunk_capacity);
        const uint32_t chunk_id = static_cast<uint32_t>(signal_index % tiling.chunk_capacity);
        StoreSignal(CompletionSignal(symmetric_buffer, tiling, publisher, chunk_id), signal_ub, 0U);
        if (tiling.gateway_policy == kGatewayPolicySourceLane) {
            StoreSignal(PayloadCompletionSignal(symmetric_buffer, tiling, publisher, chunk_id, 1U), signal_ub, 0U);
        }
    }
    // Source-lane remote-GET reuses spare cache lines in existing 512-byte signal slots. Clear them
    // before the dispatch-wide visibility barrier; every address still has exactly one reset writer.
    if (tiling.gateway_policy == kGatewayPolicySourceLane && tiling.num_servers >= 4U) {
        // Reset independently of the dispatch toggle, which may change between cached calls.
        if (tiling.chunk_capacity != 0U) {
            for (uint32_t index = core_idx; index < kGateTxRelease + tiling.num_servers; index += core_num) {
                StoreSignal(RemoteGetGateSignal(symmetric_buffer, tiling, index), signal_ub, 0U);
            }
        }
        for (uint32_t source_server = core_idx; source_server < tiling.num_servers; source_server += core_num) {
            StoreSignal(SourceServerReadySignal(symmetric_buffer, tiling, source_server), signal_ub, 0U);
            StoreSignal(PeerPutDoneSignal(symmetric_buffer, tiling, source_server), signal_ub, 0U);
        }
        if (tiling.rank % tiling.ranks_per_server == 0U) {
            for (uint32_t source_lane = core_idx + 1U; source_lane < tiling.ranks_per_server; source_lane += core_num) {
                StoreSignal(SourceRankReadySignal(symmetric_buffer, tiling, source_lane), signal_ub, 0U);
            }
        }
    }
    AscendC::SyncAll<true>();
    aclshmemi_sync_npu_v4<true>(0);
}

__aicore__ inline uint32_t AlignUbBytes(uint32_t value)
{
    return (value + kUbAlignment - 1U) / kUbAlignment * kUbAlignment;
}

__aicore__ inline void PrintReceiverTrace(
    AscendC::GlobalTensor<uint64_t> receiver_trace_gm, uint32_t core_idx, uint32_t gateway_lane,
    const ReceiverTimerData& timer)
{
    for (uint32_t sequence = 0; sequence < timer.trace_count; ++sequence) {
        const uint32_t trace_base = sequence * kReceiverTraceFields;
        const uint64_t source_and_chunk = receiver_trace_gm.GetValue(trace_base + kTraceSourceAndChunk);
        const uint32_t chunk_end = static_cast<uint32_t>(receiver_trace_gm.GetValue(trace_base + kTraceChunkEnd));
        const uint64_t wait_begin_cycle = receiver_trace_gm.GetValue(trace_base + kTraceWaitBeginCycle);
        const uint64_t ready_cycle = receiver_trace_gm.GetValue(trace_base + kTraceReadyCycle);
        const uint64_t issue_done_cycle = receiver_trace_gm.GetValue(trace_base + kTraceIssueDoneCycle);
        const uint32_t source_id = static_cast<uint32_t>(source_and_chunk);
        const uint32_t chunk_begin = static_cast<uint32_t>(source_and_chunk >> 32U);
        AscendC::printf(
            "[ascend_deepep][dup_dispatch_trace] core=%d gateway_lane=%d sequence=%d "
            "stage=compact source_server=%d chunk_begin=%d chunk_end=%d merged_chunks=%d "
            "wait_begin_cycle=%llu ready_cycle=%llu issue_done_cycle=%llu\n",
            core_idx, gateway_lane, sequence, source_id, chunk_begin, chunk_end, chunk_end - chunk_begin,
            static_cast<unsigned long long>(wait_begin_cycle), static_cast<unsigned long long>(ready_cycle),
            static_cast<unsigned long long>(issue_done_cycle));
    }
}

__aicore__ inline void FinishDispatchTimer(
    const ascend_deepep::DupServerScaleOutTilingData& tiling, uint32_t core_idx, uint32_t remote_producer_count,
    uint32_t producer_count, bool copy_to_experts, uint64_t role_start_cycle, uint64_t role_end_cycle,
    uint64_t cross_server_cycles, AscendC::GlobalTensor<uint64_t> receiver_trace_gm, const ReceiverTimerData& timer)
{
    if (tiling.enable_timer == 0U) {
        return;
    }
    // Capture all role-local timestamps before any rank-0 printf can perturb another role.
    AscendC::SyncAll<true>();
    if (aclshmem_my_pe() != 0) {
        return;
    }
    const uint32_t core_num = AscendC::GetBlockNum();
    for (uint32_t print_core = 0; print_core < core_num; ++print_core) {
        if (core_idx == print_core) {
            if (core_idx == 0U) {
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] use_urma_get=%d cycles_per_us=1000 "
                    "trace_scope=rank0 trace_clock=absolute_system_cycle trace_schema=3\n",
                    static_cast<int32_t>(tiling.use_urma_get));
            }
            if (core_idx < remote_producer_count) {
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=producer core=%d "
                    "stage=cross_server_transfer start_cycle=%llu end_cycle=%llu cycles=%llu\n",
                    core_idx, static_cast<unsigned long long>(role_start_cycle),
                    static_cast<unsigned long long>(role_end_cycle),
                    static_cast<unsigned long long>(cross_server_cycles));
            } else if (copy_to_experts && core_idx == remote_producer_count) {
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=local_copy core=%d "
                    "stage=local_partition_copy start_cycle=%llu end_cycle=%llu cycles=%llu\n",
                    core_idx, static_cast<unsigned long long>(role_start_cycle),
                    static_cast<unsigned long long>(role_end_cycle),
                    static_cast<unsigned long long>(cross_server_cycles));
            } else if (copy_to_experts && core_idx >= producer_count) {
                const uint32_t scaleup_core_idx = core_idx - producer_count;
                const uint32_t destination_lane = tiling.rank % tiling.ranks_per_server;
                const uint32_t gateway_lane = (scaleup_core_idx + destination_lane) % tiling.ranks_per_server;
                const uint32_t use_remote_get = gateway_lane != destination_lane && tiling.use_urma_get != 0U;
                const uint64_t adjusted_total_cycles = timer.total_cycles >= timer.trace_record_cycles ?
                                                           timer.total_cycles - timer.trace_record_cycles :
                                                           0U;
                const uint64_t receive_transfer_cycles =
                    adjusted_total_cycles >= timer.wait_cycles ? adjusted_total_cycles - timer.wait_cycles : 0U;
                const uint64_t attributed_transfer_cycles =
                    timer.route_issue_cycles + timer.bookkeeping_cycles + timer.final_drain_cycles;
                const uint64_t attributed_route_issue_cycles =
                    timer.route_scan_cycles + timer.remote_get_submit_cycles + timer.local_mte_copy_cycles;
                const uint64_t route_issue_other_cycles = timer.route_issue_cycles >= attributed_route_issue_cycles ?
                                                              timer.route_issue_cycles - attributed_route_issue_cycles :
                                                              0U;
                const uint64_t receive_other_cycles = receive_transfer_cycles >= attributed_transfer_cycles ?
                                                          receive_transfer_cycles - attributed_transfer_cycles :
                                                          0U;
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receiver_window use_remote_get=%d entry_cycle=%llu receive_start_cycle=%llu "
                    "receive_end_cycle=%llu\n",
                    core_idx, gateway_lane, use_remote_get, static_cast<unsigned long long>(timer.entry_cycle),
                    static_cast<unsigned long long>(timer.receive_start_cycle),
                    static_cast<unsigned long long>(timer.receive_end_cycle));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receive_init cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.init_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receive_wait cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.wait_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=route_issue cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.route_issue_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=route_scan cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.route_scan_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=remote_get_submit cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.remote_get_submit_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=local_mte_copy cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.local_mte_copy_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=route_issue_other cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(route_issue_other_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=final_drain cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.final_drain_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=final_drain_timeline drain_begin_cycle=%llu final_submit_done_cycle=%llu "
                    "receive_end_cycle=%llu final_submit_cycles=%llu final_wait_cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.final_drain_begin_cycle),
                    static_cast<unsigned long long>(timer.final_submit_done_cycle),
                    static_cast<unsigned long long>(timer.receive_end_cycle),
                    static_cast<unsigned long long>(timer.final_submit_cycles),
                    static_cast<unsigned long long>(timer.final_wait_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=final_submit cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.final_submit_cycles));
                if (use_remote_get != 0U) {
                    AscendC::printf(
                        "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                        "stage=cqe_wait cycles=%llu\n",
                        core_idx, gateway_lane, static_cast<unsigned long long>(timer.final_wait_cycles));
                } else {
                    AscendC::printf(
                        "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                        "stage=local_mte_drain cycles=%llu\n",
                        core_idx, gateway_lane, static_cast<unsigned long long>(timer.final_wait_cycles));
                }
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receiver_bookkeeping cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.bookkeeping_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receive_other cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(receive_other_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=trace_record_overhead cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.trace_record_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receive_transfer cycles=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(receive_transfer_cycles));
                AscendC::printf(
                    "[ascend_deepep][dup_dispatch_timer] role=receiver core=%d gateway_lane=%d "
                    "stage=receive_total_observed cycles=%llu adjusted_cycles=%llu "
                    "trace_events=%d max_events=%llu\n",
                    core_idx, gateway_lane, static_cast<unsigned long long>(timer.total_cycles),
                    static_cast<unsigned long long>(adjusted_total_cycles), timer.trace_count,
                    static_cast<unsigned long long>(
                        static_cast<uint64_t>(tiling.num_servers) * tiling.num_token_chunks));
                PrintReceiverTrace(receiver_trace_gm, core_idx, gateway_lane, timer);
            }
        }
        AscendC::SyncAll<true>();
    }
}

__aicore__ inline void DupServerDispatchSourceLane(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR recv_route_position,
    GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling, bool copy_to_experts)
{
    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t core_num = AscendC::GetBlockNum();
    const uint32_t source_server = tiling.rank / tiling.ranks_per_server;
    const uint32_t frame_server_begin = source_server / kServersPerFrame * kServersPerFrame;
    const uint32_t remaining_frame_servers = tiling.num_servers - frame_server_begin;
    const uint32_t frame_servers =
        remaining_frame_servers < kServersPerFrame ? remaining_frame_servers : kServersPerFrame;
    const uint32_t max_frame_servers = tiling.num_servers < kServersPerFrame ? tiling.num_servers : kServersPerFrame;
    const uint32_t wave_count =
        (max_frame_servers - 1U + tiling.max_inflight_same_frame_servers - 1U) / tiling.max_inflight_same_frame_servers;
    const uint32_t remote_producer_count =
        copy_to_experts ? core_num - tiling.ranks_per_server - 1U : (tiling.num_servers > 1U ? core_num : 0U);
    const uint32_t producer_count = remote_producer_count + (copy_to_experts ? 1U : 0U);

    ascend_deepep::UbWorkspace resource;
    __ubuf__ uint8_t* udma_wqe_buffer =
        reinterpret_cast<__ubuf__ uint8_t*>(resource.template GetBufferByByte<uint8_t>(0).GetPhyAddr());
    AscendC::LocalTensor<uint64_t> signal_ub = resource.template GetBufferByByte<uint64_t>(kSignalUbOffset);
    AscendC::LocalTensor<int32_t> token_ids_ub = resource.template GetBufferByByte<int32_t>(kTokenIdUbOffset);
    uint64_t role_start_cycle = 0U;
    uint64_t role_end_cycle = 0U;
    uint64_t cross_server_cycles = 0U;
    ReceiverTimerData receiver_timer{};
    const uint32_t next_chunk_ub_bytes = AlignUbBytes(tiling.num_servers * sizeof(uint32_t));
    const uint32_t route_cursor_ub_bytes =
        AlignUbBytes(tiling.num_servers * tiling.num_local_experts * sizeof(uint32_t));
    const uint32_t segment_end_ub_bytes = route_cursor_ub_bytes;
    const uint32_t route_position_ub_bytes = AlignUbBytes(kRoutePositionTileBytes);
    const uint32_t trace_core_idx = core_idx >= producer_count ? core_idx - producer_count : 0U;
    const uint64_t trace_events_per_core = static_cast<uint64_t>(tiling.num_servers) * tiling.chunk_capacity;
    AscendC::GlobalTensor<uint64_t> receiver_trace_gm;
    receiver_trace_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t*>(
            reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.receiver_trace_offset) +
        static_cast<uint64_t>(trace_core_idx) * trace_events_per_core * kReceiverTraceFields);

    if (copy_to_experts) {
        if (core_idx == remote_producer_count) {
            AscendC::LocalTensor<uint8_t> staging_pingpong_ub =
                resource.template GetBufferByByte<uint8_t>(kCopyUbOffset);
            if (tiling.enable_timer != 0U) {
                role_start_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
            }
            CopyLocalServerPartition(
                x, send_token_count, send_token_ids, symmetric_buffer, tiling, staging_pingpong_ub, signal_ub);
            if (tiling.enable_timer != 0U) {
                role_end_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
                cross_server_cycles = role_end_cycle - role_start_cycle;
            }
            FinishDispatchTimer(
                tiling, core_idx, remote_producer_count, producer_count, copy_to_experts, role_start_cycle,
                role_end_cycle, cross_server_cycles, receiver_trace_gm, receiver_timer);
            return;
        }
        if (core_idx >= producer_count) {
            const uint32_t copy_ub_bytes = kCopyBufferCount * tiling.copy_tile_bytes;
            const uint32_t next_chunk_ub_offset = copy_ub_bytes;
            const uint32_t route_cursor_ub_offset = next_chunk_ub_offset + next_chunk_ub_bytes;
            const uint32_t segment_end_ub_offset = route_cursor_ub_offset + route_cursor_ub_bytes;
            const uint32_t route_position_ub_offset = segment_end_ub_offset + segment_end_ub_bytes;

            AscendC::LocalTensor<uint8_t> staging_pingpong_ub =
                resource.template GetBufferByByte<uint8_t>(kCopyUbOffset);
            AscendC::LocalTensor<uint32_t> next_chunk_ub =
                resource.template GetBufferByByte<uint32_t>(next_chunk_ub_offset);
            AscendC::LocalTensor<uint32_t> route_cursor_ub =
                resource.template GetBufferByByte<uint32_t>(route_cursor_ub_offset);
            AscendC::LocalTensor<uint32_t> segment_end_ub =
                resource.template GetBufferByByte<uint32_t>(segment_end_ub_offset);
            AscendC::LocalTensor<int32_t> route_position_ub =
                resource.template GetBufferByByte<int32_t>(route_position_ub_offset);
            const uint32_t worker_idx = core_idx - producer_count;
            const uint32_t gateway_lane =
                (worker_idx + tiling.rank % tiling.ranks_per_server) % tiling.ranks_per_server;
            ReceiveGatewayRoutes(
                recv_route_position, recv_segment_offset, recv_x, symmetric_buffer, tiling, core_idx, worker_idx,
                gateway_lane, 0U, tiling.num_local_experts, AllServerMask(tiling.num_servers), staging_pingpong_ub,
                next_chunk_ub, route_cursor_ub, segment_end_ub, route_position_ub, udma_wqe_buffer, receiver_trace_gm,
                receiver_timer, true);
            FinishDispatchTimer(
                tiling, core_idx, remote_producer_count, producer_count, copy_to_experts, role_start_cycle,
                role_end_cycle, cross_server_cycles, receiver_trace_gm, receiver_timer);
            return;
        }
    }

    if (tiling.enable_timer != 0U) {
        role_start_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    if (core_idx < remote_producer_count) {
        for (uint32_t wave = 0; wave < wave_count; ++wave) {
            const uint32_t peer_count = WavePeerCount(tiling, frame_servers, wave);
            for (uint32_t logical_peer = core_idx; logical_peer < peer_count; logical_peer += remote_producer_count) {
                const uint32_t destination_server =
                    OutgoingServer(tiling, source_server, frame_server_begin, frame_servers, logical_peer, wave);
                SendServerPartition(
                    x, send_token_count, send_token_ids, symmetric_buffer, tiling, source_server, destination_server,
                    core_idx, udma_wqe_buffer, signal_ub, token_ids_ub);
            }
        }
    }
    if (tiling.enable_timer != 0U) {
        role_end_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        cross_server_cycles = role_end_cycle - role_start_cycle;
    }
    FinishDispatchTimer(
        tiling, core_idx, remote_producer_count, producer_count, copy_to_experts, role_start_cycle, role_end_cycle,
        cross_server_cycles, receiver_trace_gm, receiver_timer);
}

__aicore__ inline void DupServerDispatchSourceLaneRemoteGet(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR recv_route_position,
    GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    const ascend_deepep::DupServerScaleOutTilingData& tiling)
{
    const uint32_t core_idx = AscendC::GetBlockIdx();
    const uint32_t source_server = tiling.rank / tiling.ranks_per_server;
    ascend_deepep::UbWorkspace resource;
    __ubuf__ uint8_t* ub_base =
        reinterpret_cast<__ubuf__ uint8_t*>(resource.template GetBufferByByte<uint8_t>(0).GetPhyAddr());
    AscendC::LocalTensor<uint64_t> signal_ub = resource.template GetBufferByByte<uint64_t>(kSignalUbOffset);
    AscendC::LocalTensor<int32_t> token_ids_ub = resource.template GetBufferByByte<int32_t>(kTokenIdUbOffset);
    const uint64_t role_start = tiling.enable_timer != 0U ? static_cast<uint64_t>(AscendC::GetSystemCycle()) : 0U;
    uint32_t direct_get_rows = 0U;

    // Four producer AIVs own disjoint destination servers. Four publisher AIVs publish disjoint
    // source-token chunks and then split the local-gateway output by expert. Receiver-0 aggregates
    // source-server readiness; receivers 1..7 consume the seven nonlocal gateway lanes.
    if (core_idx < kRemoteGetProducerCount) {
        uint32_t put_index = 0U;
        for (uint32_t destination_offset = 1U; destination_offset < tiling.num_servers; ++destination_offset) {
            const uint32_t destination_server = (source_server + destination_offset) % tiling.num_servers;
            if (ServerUsesRemoteGet(tiling, source_server, destination_server)) {
                continue;
            }
            if (put_index % kRemoteGetProducerCount == core_idx) {
                SendServerPartition(
                    x, send_token_count, send_token_ids, symmetric_buffer, tiling, source_server, destination_server,
                    core_idx, ub_base, signal_ub, token_ids_ub);
                StoreSignal(PeerPutDoneSignal(symmetric_buffer, tiling, destination_server), signal_ub, 1U);
            }
            ++put_index;
        }
    } else if (core_idx < kRemoteGetReceiverBegin) {
        const uint32_t publisher_idx = core_idx - kRemoteGetProducerCount;
        AscendC::LocalTensor<uint8_t> staging_pingpong_ub =
            resource.template GetBufferByByte<uint8_t>(kCopyUbOffset);
        PublishAllLocalTokens(x, symmetric_buffer, tiling, publisher_idx, staging_pingpong_ub, signal_ub);
        const uint32_t local_expert_begin =
            static_cast<uint64_t>(tiling.num_local_experts) * publisher_idx / kRemoteGetPublisherCount;
        const uint32_t local_expert_end =
            static_cast<uint64_t>(tiling.num_local_experts) * (publisher_idx + 1U) / kRemoteGetPublisherCount;
        if (local_expert_begin < local_expert_end) {
            const uint32_t next_chunk_ub_bytes = AlignUbBytes(tiling.num_servers * sizeof(uint32_t));
            const uint32_t route_cursor_ub_bytes =
                AlignUbBytes(tiling.num_servers * tiling.num_local_experts * sizeof(uint32_t));
            const uint32_t copy_ub_bytes = kCopyBufferCount * tiling.copy_tile_bytes;
            const uint32_t next_chunk_ub_offset = copy_ub_bytes;
            const uint32_t route_cursor_ub_offset = next_chunk_ub_offset + next_chunk_ub_bytes;
            const uint32_t segment_end_ub_offset = route_cursor_ub_offset + route_cursor_ub_bytes;
            const uint32_t route_position_ub_offset = segment_end_ub_offset + route_cursor_ub_bytes;
            AscendC::LocalTensor<uint32_t> next_chunk_ub =
                resource.template GetBufferByByte<uint32_t>(next_chunk_ub_offset);
            AscendC::LocalTensor<uint32_t> route_cursor_ub =
                resource.template GetBufferByByte<uint32_t>(route_cursor_ub_offset);
            AscendC::LocalTensor<uint32_t> segment_end_ub =
                resource.template GetBufferByByte<uint32_t>(segment_end_ub_offset);
            AscendC::LocalTensor<int32_t> route_position_ub =
                resource.template GetBufferByByte<int32_t>(route_position_ub_offset);
            AscendC::GlobalTensor<uint64_t> unused_trace_gm;
            unused_trace_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.receiver_trace_offset));
            ReceiverTimerData local_receiver_timer{};
            ReceiveGatewayRoutes(
                recv_route_position, recv_segment_offset, recv_x, symmetric_buffer, tiling, core_idx, publisher_idx,
                tiling.rank % tiling.ranks_per_server, local_expert_begin, local_expert_end,
                GatewayServerMask(tiling, source_server), staging_pingpong_ub, next_chunk_ub, route_cursor_ub,
                segment_end_ub, route_position_ub, ub_base, unused_trace_gm, local_receiver_timer, false);
        }
    } else if (core_idx >= kRemoteGetReceiverBegin && core_idx < kRemoteGetBlockDim) {
        const uint32_t receiver_idx = core_idx - kRemoteGetReceiverBegin;
        if (receiver_idx == 0U) {
            if (tiling.remote_get_rx_tx_gate != 0U) {
                PublishRemoteGetGates(symmetric_buffer, tiling, source_server);
            } else {
                PublishSourceServerReady(symmetric_buffer, tiling, source_server, core_idx);
            }
        } else {
            const uint32_t next_chunk_ub_bytes = AlignUbBytes(tiling.num_servers * sizeof(uint32_t));
            const uint32_t route_cursor_ub_bytes =
                AlignUbBytes(tiling.num_servers * tiling.num_local_experts * sizeof(uint32_t));
            const uint32_t copy_ub_bytes = kCopyBufferCount * tiling.copy_tile_bytes;
            const uint32_t next_chunk_ub_offset = copy_ub_bytes;
            const uint32_t route_cursor_ub_offset = next_chunk_ub_offset + next_chunk_ub_bytes;
            const uint32_t segment_end_ub_offset = route_cursor_ub_offset + route_cursor_ub_bytes;
            const uint32_t route_position_ub_offset = segment_end_ub_offset + route_cursor_ub_bytes;
            AscendC::LocalTensor<uint8_t> staging_pingpong_ub =
                resource.template GetBufferByByte<uint8_t>(kCopyUbOffset);
            AscendC::LocalTensor<uint32_t> next_chunk_ub =
                resource.template GetBufferByByte<uint32_t>(next_chunk_ub_offset);
            AscendC::LocalTensor<uint32_t> route_cursor_ub =
                resource.template GetBufferByByte<uint32_t>(route_cursor_ub_offset);
            AscendC::LocalTensor<uint32_t> segment_end_ub =
                resource.template GetBufferByByte<uint32_t>(segment_end_ub_offset);
            AscendC::LocalTensor<int32_t> route_position_ub =
                resource.template GetBufferByByte<int32_t>(route_position_ub_offset);
            const uint64_t trace_events_per_core = static_cast<uint64_t>(tiling.num_servers) * tiling.chunk_capacity;
            AscendC::GlobalTensor<uint64_t> receiver_trace_gm;
            receiver_trace_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(symmetric_buffer) + tiling.receiver_trace_offset) +
                static_cast<uint64_t>(receiver_idx) * trace_events_per_core * kReceiverTraceFields);
            ReceiverTimerData receiver_timer{};
            const uint32_t gateway_lane =
                (receiver_idx + tiling.rank % tiling.ranks_per_server) % tiling.ranks_per_server;
            ReceiveGatewayRoutes(
                recv_route_position, recv_segment_offset, recv_x, symmetric_buffer, tiling, core_idx, receiver_idx,
                gateway_lane, 0U, tiling.num_local_experts, GatewayServerMask(tiling, source_server),
                staging_pingpong_ub, next_chunk_ub, route_cursor_ub, segment_end_ub, route_position_ub, ub_base,
                receiver_trace_gm, receiver_timer, true);
        }
    }

    const uint64_t primary_end = tiling.enable_timer != 0U ? static_cast<uint64_t>(AscendC::GetSystemCycle()) : 0U;
    // Direct GET stays on producer AIVs. For a peer which also received a PUT, DirectGetOwnerCore
    // returns that PUT's owner, so one peer/QP is never controlled by two different AIVs.
    if (core_idx < kRemoteGetProducerCount) {
        AscendC::LocalTensor<int32_t> route_position_ub =
            resource.template GetBufferByByte<int32_t>(kRemoteGetRoutePositionUbOffset);
        AscendC::LocalTensor<int32_t> segment_pairs_ub =
            resource.template GetBufferByByte<int32_t>(kDirectGetSegmentUbOffset);
        for (uint32_t remote_index = 0U; remote_index < tiling.remote_get_servers; ++remote_index) {
            if (DirectGetOwnerCore(tiling, source_server, remote_index) != core_idx) {
                continue;
            }
            const uint32_t remote_source_server = RemoteGetSourceServer(tiling, source_server, remote_index);
            direct_get_rows += DirectGetSourceServer(
                recv_route_position, recv_segment_offset, recv_x, symmetric_buffer, tiling, remote_source_server,
                core_idx, ub_base, ub_base + kQp1WqeOffset, route_position_ub, segment_pairs_ub);
        }
    }

    if (tiling.enable_timer != 0U && tiling.rank == 0U) {
        const uint64_t role_end = static_cast<uint64_t>(AscendC::GetSystemCycle());
        const uint32_t role = core_idx < kRemoteGetProducerCount ?
                                  0U :
                                  (core_idx < kRemoteGetReceiverBegin ? 1U : (core_idx < kRemoteGetBlockDim ? 2U : 3U));
        AscendC::printf(
            "[ascend_deepep][dup_remote_get_timer] core=%d role=%d remote_get_servers=%d "
            "primary_cycles=%llu direct_get_cycles=%llu direct_get_rows=%d\n",
            core_idx, role, tiling.remote_get_servers, static_cast<unsigned long long>(primary_end - role_start),
            static_cast<unsigned long long>(role_end - primary_end), direct_get_rows);
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_dispatch_scaleout_kernel(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR symmetric_buffer,
    ascend_deepep::DupServerScaleOutTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    DupServerDispatchSourceLane(
        x, send_token_count, send_token_ids, nullptr, nullptr, nullptr, symmetric_buffer, tiling, false);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_dispatch_kernel(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR recv_route_position, GM_ADDR recv_src_rank,
    GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    ascend_deepep::DupServerScaleOutTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    (void)recv_src_rank;
    DupServerDispatchSourceLane(
        x, send_token_count, send_token_ids, recv_route_position, recv_segment_offset, recv_x, symmetric_buffer, tiling,
        true);

    aclshmemi_sync_npu_v4<true>(0);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_dispatch_remote_get_kernel(
    GM_ADDR x, GM_ADDR send_token_count, GM_ADDR send_token_ids, GM_ADDR recv_route_position, GM_ADDR recv_src_rank,
    GM_ADDR recv_segment_offset, GM_ADDR recv_x, GM_ADDR symmetric_buffer,
    ascend_deepep::DupServerScaleOutTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    (void)recv_src_rank;
    DupServerDispatchSourceLaneRemoteGet(
        x, send_token_count, send_token_ids, recv_route_position, recv_segment_offset, recv_x, symmetric_buffer,
        tiling);

    // Protect the source slabs until every direct GET and gateway-local copy has drained.
    aclshmemi_sync_npu_v4<true>(0);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dup_server_notify_scaleout_kernel(
    GM_ADDR symmetric_buffer, ascend_deepep::DupServerScaleOutTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    DupServerNotifyScaleOut(symmetric_buffer, tiling);
}

extern "C" ASCEND_DEEPEP_EXPORT void dup_server_dispatch_scaleout_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* send_token_count, uint8_t* send_token_ids,
    uint8_t* send_gateway_lane, uint8_t* symmetric_buffer, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DupServerScaleOutTilingData*>(tiling_host);
    (void)send_gateway_lane;
    dup_server_dispatch_scaleout_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        x, send_token_count, send_token_ids, symmetric_buffer, *td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void dup_server_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* send_token_count, uint8_t* send_token_ids,
    uint8_t* send_gateway_lane, uint8_t* recv_route_position, uint8_t* recv_gateway_lane, uint8_t* recv_src_rank,
    uint8_t* recv_segment_offset, uint8_t* recv_x, uint8_t* symmetric_buffer, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DupServerScaleOutTilingData*>(tiling_host);
    (void)send_gateway_lane;
    (void)recv_gateway_lane;
    if (td->remote_get_servers != 0U) {
        dup_server_dispatch_remote_get_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
            x, send_token_count, send_token_ids, recv_route_position, recv_src_rank, recv_segment_offset, recv_x,
            symmetric_buffer, *td, ffts_addr);
        return;
    }
    dup_server_dispatch_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        x, send_token_count, send_token_ids, recv_route_position, recv_src_rank, recv_segment_offset, recv_x,
        symmetric_buffer, *td, ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void dup_server_notify_scaleout_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* symmetric_buffer, uint8_t* tiling_host, uint64_t ffts_addr)
{
    auto* td = reinterpret_cast<ascend_deepep::DupServerScaleOutTilingData*>(tiling_host);
    dup_server_notify_scaleout_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        symmetric_buffer, *td, ffts_addr);
}
