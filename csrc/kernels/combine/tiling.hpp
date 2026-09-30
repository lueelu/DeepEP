// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#pragma once
#include <cstdint>

namespace ascend_deepep {
namespace combine_config {
constexpr uint32_t kAivCores = 64;
constexpr uint32_t kLaunchUbBytes = 192U * 1024U;
// Reducer copy/meta UB: transient flags/two forward banks followed by resident controls.
constexpr uint32_t kGatewayScratchBytes = 4U * 1024U;
constexpr uint32_t kGatewayUbBytes = 48U * 1024U;
constexpr uint32_t kGatewayControlUbBytes = kGatewayUbBytes - kGatewayScratchBytes;
}  // namespace combine_config
namespace combine_control {
constexpr uint32_t kSlotBytes = 512;
// Per-chunk gather flags followed by per-reducer monotonic completion epochs.
constexpr uint32_t kMaxGroups = 8;
// Rank-local cross-only barrier epochs [owner], disjoint from SO source slots 0..63.
constexpr uint32_t kCrossGroupDoneBase = 64;
constexpr uint32_t kGatherDoneBase = 96;
constexpr uint32_t kGatherDoneLanes = 8;
constexpr uint32_t kReduceDoneSlots = combine_config::kAivCores;
// Only the rank's local core publishes self-copy completion.
constexpr uint32_t kSelfCopyDoneSlots = combine_config::kAivCores;
}  // namespace combine_control
namespace combine_profile {
constexpr uint32_t kVersion = 46, kCores = combine_config::kAivCores, kBaseRecords = 43, kWords = 64;
constexpr uint32_t kChunkRecords = 4;
constexpr uint32_t kGatherGroupBase = 27, kReduceGroupBase = 35;
constexpr uint32_t kQpFieldStride = 5;
static_assert(kCores == combine_config::kAivCores, "trace and launch core counts must match");
enum TraceSlot : uint32_t { SummarySlot = 0, GroupSlot = 1, PeerSlotBase = 2, ChunkSlot = 6 };
enum Kind : uint32_t {
    Summary = 1,
    Gather = 2,
    LocalReturn = 3,
    CrossReturn = 4,
    Group = 5,
    GatherGroup = 6,
    ReduceGroup = 7,
    GatherChunk = 8,
    ReduceChunk = 9,
    CrossChunk = 10,
    ReturnedSo = 11,
    SourceChunk = 12  // Window polling and reduce, indexed by its last chunk.
};
enum Field : uint32_t {
    Valid = 0,
    Version = 1,
    Rank = 2,
    Core = 3,
    Type = 4,
    GroupId = 5,
    Peer = 6,
    Incoming = 7,
    Begin = 8,
    End = 9,
    ChunkIndex = 29,
    ContributorMask = 30,
    SourceFlagsReady = 10,      // SourceChunk: polling ends / SyncAll begins; End is SyncAll end.
    SourcePhaseReduceEnd = 11,  // SourceChunk: this phase's output writes have completed.
    // CrossReturn records only; summary fields at these offsets are unchanged.
    CrossFlagOrder = 47,  // 1=SO at owner/window tails.
    CrossPollCqEnd0 = 48,
    CrossPollCqEnd1 = 49,
    CrossPollCqBegin0 = 50,
    CrossPollCqBegin1 = 51,
    CrossMteBatchComplete = 52,
    CrossGroupBarrierBegin = 53,
    CrossGroupBarrierEnd = 54
};
// QP1 fields use the QP0 index plus kQpFieldStride.
// Local links use MTE for EP >= 64: Quiet drains MTE and SoSubmit marks the flag store.
// Smaller worlds use URMA QP0; cross links always use URMA.
enum Link : uint32_t {
    CreditObserved = 10,
    IncomingObserved = 11,
    FirstIssue = 12,
    LastSubmit = 13,
    QuietBegin = 14,
    QuietEnd = 15,
    SoSubmit = 16,
    Bytes0 = 22,
    Bytes1 = 23,
    Wqes0 = 24,
    Wqes1 = 25,
    Rows = 26,
    CopyBytes = 27,
    Flags = 28
};  // flags: outgoing remote=1, incoming remote=2, credit required=4, credit bypassed=8
// CrossReturnTotal includes waits, submission, synchronization and quiet.
enum SummaryField : uint32_t {
    InitDone = 10,
    ClearBegin = 11,
    ClearEnd = 12,
    SyncMask = 13,          // local->reduce=1, reduce->local/self-copy=2, reduce->cross=4.
    NumSo = 14,             // Return SO windows, clamped to the chunk count.
    SharedJetty = 15,       // Shared SQ mode.
    FirstChunkTokens = 16,  // Reserved legacy alias.
    FirstChunkSplits = 16,  // Startup subchunk count (>=1).
    BarrierBase = 16,
    StageBase = 28,
    LocalReturnReduceWaitBegin = 22,
    LocalReturnReduceWaitEnd = 23,
    ReturnsReadyWaitBegin = 24,
    ReturnsReadyWaitEnd = 25,
    GatherBegin = 28,
    GatherEnd = 29,
    LocalReturnBegin = 32,
    LocalReturnEnd = 33,
    CrossReturnTotalBegin = 34,
    CrossReturnTotalEnd = 35,
    SourceReduceBegin = 36,
    SourceReduceEnd = 37,
    GatewayTokens = 40,
    GatewayRows = 41,
    GatewayReadBytes = 42,
    GatewayWriteBytes = 43,
    FinalTokens = 44,
    FinalRows = 45,
    FinalReadBytes = 46,
    FinalWriteBytes = 47,
    World = 48,
    Hidden = 49,
    CoreCount = 50,
    ReduceCoreBegin = 51,  // Physical boundary: 8 + cross_count.
    ReduceCoreNum = 52,
    ChunksPerGroup = 53,
    ChunkTokens = 54,
    IndependentChunkStride = 55,
    CreditWindowGroups = 56,  // Reserved legacy alias.
    CrossTokensPerTurn = 56,  // Cross rank rotation quantum.
    GroupSize = 57,
    CrossCoreOffset = 31,  // Reserved.
    FinalCrossQuietBegin = 58,
    FinalCrossQuietEnd = 59,
    ReturnsPollBegin = 60,
    ReturnsPollEnd = 61,
    SelfCopyBegin = 62,
    SelfCopyEnd = 63
};
// Reserved group-window fields.
enum GroupField : uint32_t {
    GroupBegin = 8,
    GroupEnd = 9,
    // 首个 chunk 就绪后至本组入向 SO 全部到达，包含后续 chunk 等归约及 credit 等待，不含 quiet。
    GroupIssueAndIncomingWaitBegin = 10,
    GroupIssueAndIncomingWaitEnd = 11,
    GroupIncomingJoinBegin = 12,
    GroupIncomingJoinEnd = 13,
    GroupNextCreditBegin = 14,
    GroupNextCreditEnd = 15,
    GroupOutgoingQuietBegin = 16,
    GroupOutgoingQuietEnd = 17,
    GroupOutgoingJoinBegin = 18,
    GroupOutgoingJoinEnd = 19,
    GroupNextIncomingPeerBase = 20,
    GroupPeerCount = 22,
    GroupFirstChunkReduceWaitBegin = 23,
    GroupFirstChunkReduceWaitEnd = 24
};
enum ChunkField : uint32_t {
    ChunkWaitBegin = 8,
    ChunkReady = 10,
    // gateway 等汇聚 flag 之后开始归约，完成 BF16 写回并发布 reduce_done 后结束。
    GatewayChunkReduceBegin = 11,
    GatewayChunkReduceEnd = 9,
    // 跨 server 发送核等 credit 并提交当前 chunk 的 PUT，尾 chunk 还提交组 SO；不含 quiet。
    CrossChunkPutBegin = 11,
    CrossChunkPutEnd = 9,
    // 专家卡向 gateway 提交当前 chunk 的 PUT，结束点在 chunk SO 提交之前。
    GatherChunkPutBegin = 8,
    GatherChunkPutEnd = 9
};
enum IndependentField : uint32_t {
    SequenceIndex = 31,
    PeerEnvelopeBegin = 32,
    PeerEnvelopeEnd = 33,
    NextCreditBegin = 34,
    NextCreditEnd = 35,
    NextIncomingPeer = 36,
    IncomingWaitBegin = 37,
    PendingChunkBegin = 38,
    ChunkCreditReady = 32
};
// Per-lane SO submission timestamps, not remote arrival.
enum SharedCrossField : uint32_t {
    CrossReduceReady = 10,
    CrossSo0 = 16,
    CrossSo1 = 21,
    CrossSliceCount = 38,
    CrossIssueCycles = 39,
    CrossDirectRows = 40,
    CrossReducedRows = 41,
    CrossReducedFirstIssue = 42,
    CrossDirectFirstIssue = 43,
    CrossChunkSubmit = 44,
    CrossQp0SliceCount = 45,
    CrossQp1SliceCount = 46,
    ReturnSo0Observed = 10,
    ReturnSo1Observed = 11,
    CrossChunkAssigned = 55,  // CrossReturn: payload assigned to this owner.
    ReturnSecondOwner = 53,   // ReturnedSo: owner + 1, zero if absent.
    ReturnSecondSo0Observed = 54,
    ReturnSecondSo1Observed = 55
};
}  // namespace combine_profile
// Host and device must be rebuilt together when this ABI changes.
struct PutCombineTiling {
    uint32_t rank_id, rank_num, token_num, hidden_size, expert_row_capacity, gather_row_capacity;
    uint32_t forward_row_capacity, backward_row_capacity;
    uint32_t chunk_token_num, chunks_per_group, total_chunk_num, control_slot_num;
    uint64_t expert_input_offset_bytes, gather_input_offset_bytes, server_partial_offset_bytes,
        returned_partial_offset_bytes, output_offset_bytes;
    uint32_t wait_credit;
    // One exclusive UDMA QP pair per cross PUT core (1..16).
    uint32_t num_cross_put_cores;
    uint32_t shared_jetty;
    uint32_t group_size;
    uint32_t cross_tokens_per_turn;
    // All enabled by default by the host; disabling is for profiling only.
    uint32_t sync_local_reduce, sync_reduce_local, sync_reduce_cross;
    uint32_t num_so;              // Return notification windows; host default 3, clamped to chunk count.
    uint32_t first_chunk_splits;  // 1 disables startup splitting.
    uint32_t weight_topk, weight_rows, with_weights;
    uint64_t weight_recv_offset_bytes, weight_local_done_offset_bytes, weight_done_offset_bytes;
};

// Caller validates public bounds before narrowing. All region arithmetic is uint64.
inline PutCombineTiling MakeCombineTiling(uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk, uint32_t rows,
                                          uint32_t gather_rows, uint32_t chunk_tokens = 256U)
{
    PutCombineTiling t{};
    t.rank_id = rank;
    t.rank_num = world;
    t.token_num = tokens;
    t.hidden_size = 7168U;
    t.expert_row_capacity = rows;
    t.gather_row_capacity = gather_rows;
    t.forward_row_capacity = tokens * topk;
    t.backward_row_capacity = world * t.forward_row_capacity;
    t.chunk_token_num = chunk_tokens;
    t.first_chunk_splits = tokens < chunk_tokens ? tokens : chunk_tokens;
    if (t.first_chunk_splits > 4U) {
        t.first_chunk_splits = 4U;
    }
    t.chunks_per_group = (tokens + chunk_tokens - 1U) / chunk_tokens + t.first_chunk_splits - 1U;
    t.group_size = world < 64U ? world : 64U;
    t.total_chunk_num = world / t.group_size * t.chunks_per_group;
    t.control_slot_num = 3U * world + combine_control::kGatherDoneBase +
                         t.total_chunk_num * combine_control::kGatherDoneLanes + combine_control::kReduceDoneSlots +
                         world * t.chunks_per_group * 2U + combine_control::kSelfCopyDoneSlots;
    t.wait_credit = 1U;
    t.num_cross_put_cores = 16U;
    t.shared_jetty = 1U;
    t.cross_tokens_per_turn = 8U;
    t.sync_local_reduce = t.sync_reduce_local = t.sync_reduce_cross = 1U;
    t.num_so = 3U;
    uint64_t cursor = uint64_t(t.control_slot_num) * combine_control::kSlotBytes;
    auto reserve = [&](uint64_t bytes) {
        const uint64_t offset = cursor;
        cursor += (bytes + 511U) / 512U * 512U;
        return offset;
    };
    t.expert_input_offset_bytes = reserve(uint64_t(rows) * 7168U * 2U);
    t.gather_input_offset_bytes = reserve(uint64_t(gather_rows) * 7168U * 2U);
    t.server_partial_offset_bytes = reserve(uint64_t(rows) * 7168U * 2U);
    t.returned_partial_offset_bytes = reserve(uint64_t(tokens) * ((world + 7U) / 8U) * 7168U * 2U);
    t.output_offset_bytes = reserve(uint64_t(tokens) * 7168U * 2U);
    t.weight_topk = topk;
    t.weight_recv_offset_bytes = reserve(uint64_t(tokens) * topk * sizeof(float));
    t.weight_local_done_offset_bytes = reserve(uint64_t(combine_config::kAivCores) * combine_control::kSlotBytes);
    t.weight_done_offset_bytes = reserve(uint64_t(world) * combine_control::kSlotBytes);
    return t;
}

inline uint64_t CombineWorkspaceBytes(const PutCombineTiling& t)
{
    return t.weight_done_offset_bytes + uint64_t(t.rank_num) * combine_control::kSlotBytes;
}
}  // namespace ascend_deepep
