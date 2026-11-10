#pragma once

#include <cstdint>

#ifdef ASCEND_DEEPEP_DEVICE_SCHEDULE
#define ASCEND_DEEPEP_SCHEDULE_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_SCHEDULE_FN inline constexpr
#endif

namespace ascend_deepep::udma {
inline constexpr uint32_t kQpCount = 32;
inline constexpr uint32_t kMaxPeers = 8;
inline constexpr uint32_t kTileTokens = 256;
inline constexpr uint32_t kBatchWqes = 32;
inline constexpr uint32_t kWqeBytes = 64;
inline constexpr uint32_t kSignalBytes = 512;
static_assert(kSignalBytes % 256 == 0, "completion slots must be 256-byte aligned");
inline constexpr uint32_t kRankOrder = 0;
inline constexpr uint32_t kGroupOrder = 1;
inline constexpr uint32_t kGroupStaggeredOrder = 2;
inline constexpr uint32_t kGroupRoundRobinOrder = 3;
inline constexpr uint32_t kGroupPeerCapacity = 64;

ASCEND_DEEPEP_SCHEDULE_FN bool ValidPeers(uint32_t peers)
{
    return peers == 1 || peers == 2 || peers == 4 || peers == 8;
}
ASCEND_DEEPEP_SCHEDULE_FN bool ValidVisit(uint32_t wqes)
{
    return wqes == 4 || wqes == 8 || wqes == 16 || wqes == 32;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Workers(uint32_t world, uint32_t peers,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    if (order == kGroupRoundRobinOrder)
        return (ranks_per_group + peers - 1) / peers;
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
// handles its metadata before sending payload, without changing QP ownership.
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
    const uint32_t workers = Workers(world, peers, order, ranks_per_group);
    if (owner >= workers) return 0;
    if (order == kGroupRoundRobinOrder) {
        const uint32_t locals = (ranks_per_group - 1 - owner) / workers + 1;
        return locals * (world / ranks_per_group) - uint32_t(rank % ranks_per_group % workers == owner);
    }
    const uint32_t remaining = world - 1 - owner * peers;
    return remaining < peers ? remaining : peers;
}
// Group modes require a validated nonzero divisor of world. Rank mode does not
// use ranks_per_group. The omitted self entry is always ordinal groups - 1.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Peer(
    uint32_t rank, uint32_t world, uint32_t index, uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    if (order == kRankOrder)
        return (rank + 1 + index) % world;
    const uint32_t groups = world / ranks_per_group;
    const uint32_t ordinal = index + uint32_t(index >= groups - 1);
    const uint32_t local = (rank % ranks_per_group + ordinal / groups) % ranks_per_group;
    const uint32_t group = (rank / ranks_per_group + 1 + ordinal % groups) % groups;
    return group * ranks_per_group + local;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Owner(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t peers,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    if (order == kGroupRoundRobinOrder)
        return (peer % ranks_per_group) % Workers(world, peers, order, ranks_per_group);
    if (order == kRankOrder)
        return ((peer + world - rank - 1) % world) / peers;
    const uint32_t groups = world / ranks_per_group;
    const uint32_t local = (peer % ranks_per_group + ranks_per_group - rank % ranks_per_group) % ranks_per_group;
    const uint32_t group = (peer / ranks_per_group + groups - rank / ranks_per_group - 1) % groups;
    const uint32_t ordinal = local * groups + group;
    return (ordinal - uint32_t(ordinal > groups - 1)) / peers;
}
// Rotate only within the owner's actual (possibly short) list. QP ownership
// cannot change with visit, token/chunk index, or staggering.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t WorkerPeer(
    uint32_t rank, uint32_t world, uint32_t owner, uint32_t peers, uint32_t index,
    uint32_t order = kRankOrder, uint32_t ranks_per_group = 0)
{
    if (order == kGroupRoundRobinOrder) {
        const uint32_t workers = Workers(world, peers, order, ranks_per_group);
        const uint32_t locals = (ranks_per_group - 1 - owner) / workers + 1;
        const uint32_t groups = world / ranks_per_group;
        const uint32_t self = (groups - 1) * locals + (rank % ranks_per_group) / workers;
        const uint32_t ordinal = index + uint32_t(rank % ranks_per_group % workers == owner && index >= self);
        const uint32_t target = (rank / ranks_per_group + 1 + ordinal / locals) % groups;
        return target * ranks_per_group + owner + ordinal % locals * workers;
    }
    const uint32_t first = owner * peers;
    const uint32_t remaining = world - 1 - first;
    const uint32_t count = remaining < peers ? remaining : peers;
    if (order == kGroupStaggeredOrder)
        index = (index + rank % ranks_per_group + owner) % count;
    return Peer(rank, world, first + index, order, ranks_per_group);
}
// Both endpoints must create the same channel/QP set. Include every supported
// schedule, so switching visit/group size after a drained call needs no reinit.
ASCEND_DEEPEP_SCHEDULE_FN uint64_t PeerMask(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t dispatch_order = kRankOrder,
    uint32_t combine_order = kRankOrder, uint32_t ranks_per_group = 0, uint32_t peers_per_group = 0, uint32_t workers_per_peer = 1)
{
    if (rank == peer)
        return 0;
    uint64_t mask = 1; // SDK control QP.
    for (uint32_t peers = 1; peers <= kMaxPeers; peers *= 2) {
        for (uint32_t op = 0; op < 2; ++op) {
            const uint32_t order = op ? combine_order : dispatch_order;
            const uint32_t count = order == kGroupRoundRobinOrder ? peers_per_group : peers;
            if (Workers(world, count, order, ranks_per_group) * workers_per_peer <= kQpCount) {
                for (uint32_t lane = 0; lane < workers_per_peer; ++lane) {
                    mask |= uint64_t{1} << (Owner(rank, peer, world, count, order, ranks_per_group) * workers_per_peer + lane);
                    mask |= uint64_t{1} << (Owner(peer, rank, world, count, order, ranks_per_group) * workers_per_peer + lane);
                }
            }
        }
    }
    return mask;
}
// Benchmark-only union for the existing peer schedules. Reserve both endpoint
// mappings before SHMEM initialization; launches retain exclusive SQ ownership.
ASCEND_DEEPEP_SCHEDULE_FN uint64_t TuningPeerMask(
    uint32_t rank, uint32_t peer, uint32_t world, uint32_t ranks_per_group)
{
    uint64_t mask = 0;
    for (uint32_t order = kRankOrder; order <= kGroupRoundRobinOrder; ++order)
        for (uint32_t lanes = 1; lanes <= 2; ++lanes)
            for (uint32_t count = 1; count <= 4; count *= 2)
                mask |= PeerMask(rank, peer, world, order, order, ranks_per_group,
                                 order == kGroupRoundRobinOrder ? count : 0, lanes);
    return mask;
}
} // namespace ascend_deepep::udma

#undef ASCEND_DEEPEP_SCHEDULE_FN
