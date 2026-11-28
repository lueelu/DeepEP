#pragma once

#include <cstdint>
#include "common/dispatch_output.h"

#ifdef ASCEND_DEEPEP_DEVICE_SCHEDULE
#define ASCEND_DEEPEP_SCHEDULE_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_SCHEDULE_FN inline constexpr
#endif

namespace ascend_deepep::udma {
inline constexpr uint32_t kQpCount = 32;
inline constexpr uint32_t kMaxPeers = 8;
// Preserve existing P8/P16 windows; P32 uses a separate instantiation.
inline constexpr uint32_t kMidRankPeers = 16;
inline constexpr uint32_t kMaxRankPeers = 32;
inline constexpr uint32_t kTileTokens = 256;
inline constexpr uint32_t kBatchWqes = 32;
inline constexpr uint32_t kWqeBytes = 64;
inline constexpr uint32_t kDispatchSavedIndexBytes = kBatchWqes * sizeof(int32_t);
inline constexpr uint32_t kSignalBytes = 512;
static_assert(kSignalBytes % 256 == 0, "completion slots must be 256-byte aligned");
inline constexpr uint32_t kRankOrder = 0;

// Live dispatch WQE/descriptor/header/output-record region plus index windows.
// Shared output-record size keeps Host gating and device layout identical.
#ifdef ASCEND_DEEPEP_DEVICE_SCHEDULE
__aicore__
#endif
inline constexpr uint32_t DispatchLiveSendUbBytes(uint32_t capacity, uint32_t qps)
{
    return kBatchWqes * (kWqeBytes + 4U * sizeof(uint64_t)) +
        capacity * qps * (2U * kWqeBytes + kOutputRecordBytes) + 32U + kDispatchSavedIndexBytes +
        capacity * kTileTokens * sizeof(int32_t);
}

ASCEND_DEEPEP_SCHEDULE_FN bool ValidPeers(uint32_t peers)
{
    return peers == 1 || peers == 2 || peers == 4 || peers == 8 || peers == 16 || peers == 32;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t RankPeerCapacity(uint32_t peers)
{
    return peers > kMidRankPeers ? kMaxRankPeers : peers > kMaxPeers ? kMidRankPeers : kMaxPeers;
}
ASCEND_DEEPEP_SCHEDULE_FN bool ValidVisit(uint32_t wqes)
{
    // A peer visit can span several fixed-size WQE submissions. It does not
    // resize the sender's kBatchWqes scratch or change SQ admission granularity.
    return wqes == 4 || wqes == 8 || wqes == 16 || wqes == 32 || wqes == 64 || wqes == 128;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Workers(uint32_t world, uint32_t peers,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    return (world - 1 + peers - 1) / peers;
}
// Token mode partitions channels/return rows across half the AIVs. The other
// half retains SIMT metadata/self-copy parallelism. QP ownership never depends
// on whether a worker's assigned token range is empty.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t TokenWorkers(uint32_t cores)
{
    return cores / 2U < kQpCount ? cores / 2U : kQpCount;
}
ASCEND_DEEPEP_SCHEDULE_FN uint64_t TokenPeerMask(uint32_t rank, uint32_t peer)
{
    return rank == peer ? 0 : (uint64_t{1} << kQpCount) - 1U;
}
// Spare AIVs own all metadata channels; if every AIV owns a QP, each core
// handles its metadata itself, without changing QP ownership. The peer sender
// can defer this work until after payload submission when private UB fits.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t MetadataFirst(uint32_t world, uint32_t peers, uint32_t cores,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0, uint32_t workers_per_peer = 1)
{
    const uint32_t payload = workers_per_peer * Workers(world, peers, order, ranks_per_group);
    return payload < cores ? payload : 0U;
}
// Fixed local-position ownership. Removing self never shifts another peer.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t WorkerCount(
    uint32_t rank, uint32_t world, uint32_t owner, uint32_t peers,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    const uint32_t workers = Workers(world, peers);
    if (owner >= workers) return 0;
    const uint32_t remaining = world - 1 - owner * peers;
    return remaining < peers ? remaining : peers;
}
// Rank traversal starts after self. Topology fields remain in the launch ABI only.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Peer(
    uint32_t rank, uint32_t world, uint32_t index, uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    return (rank + 1 + index) % world;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Owner(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t peers,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    return ((peer + world - rank - 1) % world) / peers;
}
// Each owner has a contiguous slice of the rank traversal; visits do not change ownership.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t WorkerPeer(
    uint32_t rank, uint32_t world, uint32_t owner, uint32_t peers, uint32_t index,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    return Peer(rank, world, owner * peers + index);
}
// Both endpoints must create the same channel/QP set. Include every supported
// rank mapping, so switching visit after a drained call needs no reinit.
ASCEND_DEEPEP_SCHEDULE_FN uint64_t PeerMask(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t dispatch_order = kRankOrder,
    uint32_t combine_order = kRankOrder, uint32_t ranks_per_group = 0, uint32_t peers_per_group = 0, uint32_t workers_per_peer = 1)
{
    if (rank == peer) return 0;
    uint64_t mask = 1; // SDK control QP, symmetric endpoint ownership.
    for (uint32_t peers = 1; peers <= kMaxRankPeers; peers *= 2) {
        if (Workers(world, peers) * workers_per_peer > kQpCount) continue;
        for (uint32_t lane = 0; lane < workers_per_peer; ++lane) {
            mask |= uint64_t{1} << (Owner(rank, peer, world, peers) * workers_per_peer + lane);
            mask |= uint64_t{1} << (Owner(peer, rank, world, peers) * workers_per_peer + lane);
        }
    }
    return mask;
}
// Union shared by automatic resources and benchmark sessions. Reserve both
// endpoint mappings once; launches retain exclusive SQ ownership. Group size
// is immutable for the lifetime of these resources.
ASCEND_DEEPEP_SCHEDULE_FN uint64_t TuningPeerMask(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t ranks_per_group)
{
    return PeerMask(rank, peer, world, kRankOrder, kRankOrder, ranks_per_group, 0, 1) |
        PeerMask(rank, peer, world, kRankOrder, kRankOrder, ranks_per_group, 0, 2);
}
} // namespace ascend_deepep::udma

#undef ASCEND_DEEPEP_SCHEDULE_FN
