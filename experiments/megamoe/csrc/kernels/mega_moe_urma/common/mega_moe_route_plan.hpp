// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include <cstddef>
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"

// Host/device route metadata and scheduling arithmetic.

// No STL, tensor inspection, allocation, or input-value validation. This
// descriptor compiler runs on Device; a CPU adapter exercises the same
// arithmetic in tests. The caller guarantees legal expert IDs.
#if defined(__NPU_DEVICE__)
#define URMA_ROUTE_INLINE __aicore__ inline
#define URMA_ROUTE_UB __ubuf__
#else
#define URMA_ROUTE_INLINE inline
#define URMA_ROUTE_UB
#endif

namespace MegaMoeUrma {
constexpr uint32_t kRouteMailboxBytes = 1024U;
constexpr uint32_t kRouteHistogramBytes = kLocalWorkers * kExperts * sizeof(uint32_t);
struct DeviceRouteInfo {
    uint32_t outgoing[kExperts + 1U];  // local expert-exclusive prefix, final entry BS*K
    // Keep the early Pack-only prefix on separate cache lines from the late
    // receive summary, which is published while Pack/Dispatch are running.
    uint32_t outgoingPadding[15U];
    uint32_t incoming[kWorld][kMaxLocalExperts + 1U];  // source-local expert prefix
    uint32_t sources[kMaxLocalExperts][kWorld + 1U];   // expert-local source prefix
    uint32_t groupBase[kMaxLocalExperts + 1U];
    uint32_t compactBase[kMaxLocalExperts + 1U];
    uint64_t returnBase[kWorld];
    uint64_t gmm1Base[kMaxLocalExperts + 1U];
    uint32_t sendCvBase[kWorld + 1U], recvCvBase[kWorld + 1U];
};
static_assert(sizeof(DeviceRouteInfo) < 32U * 1024U);
static_assert(offsetof(DeviceRouteInfo, incoming) % 64U == 0U);

URMA_ROUTE_INLINE uint32_t RouteMin(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
URMA_ROUTE_INLINE uint32_t RouteMax(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}
URMA_ROUTE_INLINE uint64_t RouteGroupCapacity(uint64_t rows, uint32_t experts)
{
    return (rows + 255U) / 256U + experts;
}
URMA_ROUTE_INLINE uint64_t RouteSegmentCapacity(uint64_t rows, uint32_t world, uint32_t cv)
{
    // Overlay boundaries of expert M256 groups and each source/expert/CV.
    return RouteGroupCapacity(rows, kExperts / world) + kExperts + (rows + cv - 1U) / cv + world;
}

URMA_ROUTE_INLINE uint64_t RouteGroupPrefixOffset()
{
    return (sizeof(DeviceRouteInfo) + 511U) / 512U * 512U;
}
URMA_ROUTE_INLINE uint64_t RouteLinkPrefixOffset(uint64_t groups)
{
    return (RouteGroupPrefixOffset() + (groups + 1U) * sizeof(uint32_t) + 511U) / 512U * 512U;
}
URMA_ROUTE_INLINE uint64_t RouteInfoAllocationBytes(uint64_t groups, uint64_t chunks)
{
    return (RouteLinkPrefixOffset(groups) + (chunks + 1U) * sizeof(uint32_t) + 511U) / 512U * 512U;
}

template <class Output>
URMA_ROUTE_INLINE void CompileDeviceRoutes(Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Output& out, bool wave12,
                                           bool reorder)
{
    const uint32_t le = c.localExperts;
    r.groupBase[0] = r.compactBase[0] = 0U;
    c.fragmentCount = c.sendChunkCount = c.recvChunkCount = c.packRoundCount = 0U;
    c.gmm1Tiles = c.gmm2Tiles = c.receiveRows = c.computeTaskCount = 0U;
    uint32_t segmentCount = 0U;
    for (uint32_t e = 0; e < le; ++e) {
        r.sources[e][0] = 0U;
        for (uint32_t s = 0; s < c.world; ++s)
            r.sources[e][s + 1U] = r.sources[e][s] + r.incoming[s][e + 1U] - r.incoming[s][e];
        const uint32_t rows = r.sources[e][c.world];
        r.groupBase[e + 1U] = r.groupBase[e] + (rows + 255U) / 256U;
        r.compactBase[e + 1U] = r.compactBase[e] + rows;
        for (uint32_t first = 0U; first < rows; first += 256U) {
            const uint32_t last = RouteMin(first + 256U, rows);
            Fragment f{e, segmentCount, uint64_t(r.compactBase[e]) + first, last - first, 0U, c.gmm1Tiles, c.gmm2Tiles};
            for (uint32_t s = 0; s < c.world; ++s) {
                uint32_t lo = RouteMax(first, r.sources[e][s]);
                const uint32_t hi = RouteMin(last, r.sources[e][s + 1U]);
                while (lo < hi) {
                    const uint32_t peerRow = r.incoming[s][e] + lo - r.sources[e][s];
                    const uint32_t cv = peerRow / c.cvRows;
                    const uint32_t n = RouteMin(hi - lo, c.cvRows - peerRow % c.cvRows);
                    out.Segment(segmentCount++,
                                RowSegment{s, cv, uint64_t(s) * c.sourceRowCapacity + peerRow,
                                           r.returnBase[s] + peerRow, lo - first, n, s * c.maxPeerChunks + cv, 0U});
                    lo += n;
                }
            }
            f.segmentCount = segmentCount - f.segmentBegin;
            out.FragmentAt(c.fragmentCount++, f);
            c.gmm1Tiles += out.Gmm1Tiles(f.rows, c);
            c.gmm2Tiles += c.hidden / 256U;
        }
    }
    c.receiveRows = r.compactBase[le];
    c.selfSendRow = r.outgoing[c.rank * le];
    c.selfRecvRow = uint64_t(c.rank) * c.sourceRowCapacity;
    c.selfRows = r.incoming[c.rank][le];
    uint32_t links = 0U;
    for (uint32_t peer = 0; peer < c.world; ++peer) {
        const uint32_t sendBase = r.outgoing[peer * le];
        const uint32_t sendRows = r.outgoing[(peer + 1U) * le] - sendBase;
        c.packRoundCount = RouteMax(c.packRoundCount, (sendRows + c.cvRows - 1U) / c.cvRows);
        for (uint32_t first = 0U, cv = 0U; first < sendRows; first += c.cvRows, ++cv)
            out.Send(c.sendChunkCount++,
                     Chunk{peer, cv, uint64_t(sendBase) + first, uint64_t(c.rank) * c.sourceRowCapacity + first,
                           RouteMin(c.cvRows, sendRows - first), 0U, 0U, 0U});
        const uint32_t recvRows = r.incoming[peer][le];
        uint32_t e = 0U;
        for (uint32_t first = 0U, cv = 0U; first < recvRows; first += c.cvRows, ++cv) {
            const uint32_t end = RouteMin(first + c.cvRows, recvRows);
            Chunk chunk{
                peer, cv, uint64_t(peer) * c.sourceRowCapacity + first, r.returnBase[peer] + first, end - first, links,
                0U,   0U};
            uint32_t row = first;
            while (row < end) {
                while (r.incoming[peer][e + 1U] <= row) ++e;
                const uint32_t expertRow = r.sources[e][peer] + row - r.incoming[peer][e];
                const uint32_t group = r.groupBase[e] + expertRow / 256U;
                const uint32_t n =
                    RouteMin(RouteMin(end - row, r.incoming[peer][e + 1U] - row), 256U - expertRow % 256U);
                out.Link(links, group);
                // Sorted by destination row, including a partial first/last M-group.
                if (reorder)
                    out.ReorderSpan(
                        links, uint64_t(r.compactBase[e]) + expertRow,
                        peer == c.rank ? r.returnBase[peer] + row : uint64_t(peer) * c.sourceRowCapacity + row, group,
                        n);
                ++links;
                row += n;
            }
            chunk.fragmentCount = links - chunk.fragmentBegin;
            if (reorder) out.ReorderRange(c.recvChunkCount, chunk.fragmentBegin, chunk.fragmentCount);
            out.Receive(c.recvChunkCount++, chunk);
        }
    }
    // Pack is a fixed round/peer table, including empty peers, rather than a
    // data-dependent task queue. Existing static Pack ownership is unchanged.
    for (uint32_t cv = 0; cv < c.packRoundCount; ++cv) {
        uint64_t rows = 0U;
        for (uint32_t peer = 0; peer < kWorld; ++peer) {
            Chunk chunk{};
            if (peer < c.world) {
                const uint32_t base = r.outgoing[peer * le];
                const uint32_t count = r.outgoing[(peer + 1U) * le] - base;
                const uint32_t first = cv * c.cvRows;
                if (first < count)
                    chunk = Chunk{peer,
                                  cv,
                                  uint64_t(base) + first,
                                  uint64_t(c.rank) * c.sourceRowCapacity + first,
                                  RouteMin(c.cvRows, count - first),
                                  0U,
                                  0U,
                                  0U};
            }
            out.PackChunk(cv, peer, chunk);
            rows += chunk.rows;
        }
        out.PackRows(cv, rows);
    }
    if (wave12) {
        for (uint32_t first = 0U, wave = 0U; first < c.fragmentCount; first += 12U, ++wave) {
            out.Batch(wave, ComputeBatch{first, RouteMin(12U, c.fragmentCount - first)});
            out.Task(c.computeTaskCount++, ComputeBatchTask{ComputeBatchStage::Gmm1, wave});
            out.Task(c.computeTaskCount++, ComputeBatchTask{ComputeBatchStage::Gmm2, wave});
        }
    } else {
        uint32_t g1 = 0U, pending = 0U;
        for (uint32_t e = 0U; e < le; ++e)
            out.Batch(e, ComputeBatch{r.groupBase[e], r.groupBase[e + 1U] - r.groupBase[e]});
        for (uint32_t g2 = 0U; g2 < le; ++g2) {
            while (g1 < le && (g1 < RouteMin(le, g2 + 2U) || pending < c.gmm1AheadMGroups)) {
                out.Task(c.computeTaskCount++, ComputeBatchTask{ComputeBatchStage::Gmm1, g1});
                pending += r.groupBase[g1 + 1U] - r.groupBase[g1];
                ++g1;
            }
            out.Task(c.computeTaskCount++, ComputeBatchTask{ComputeBatchStage::Gmm2, g2});
            pending -= r.groupBase[g2 + 1U] - r.groupBase[g2];
        }
    }
    c.routeSegmentCount = segmentCount;
    c.routeLinkCount = links;
}
}  // namespace MegaMoeUrma

// Parallel route descriptor construction. No task stealing or
// atomic append: each worker owns contiguous ranges of every output table.
namespace MegaMoeUrma {
struct RouteWorkRange {
    uint32_t first, last;
};
URMA_ROUTE_INLINE RouteWorkRange RouteWork(uint32_t count, uint32_t worker, uint32_t workers = kLocalWorkers)
{
    return {uint32_t(uint64_t(count) * worker / workers), uint32_t(uint64_t(count) * (worker + 1U) / workers)};
}
URMA_ROUTE_INLINE uint32_t PlanGmm1Tiles(uint32_t rows, const Config& c)
{
    return Gmm1ComputeTileCount(rows, c);
}
URMA_ROUTE_INLINE uint32_t PlanExpertForGroup(uint32_t group, const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r)
{
    uint32_t lo = 0U, hi = c.localExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (r.groupBase[mid + 1U] <= group)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}
URMA_ROUTE_INLINE uint32_t PlanPeerForCv(uint32_t cv, const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, bool send)
{
    uint32_t peer = 0U;
    while (peer + 1U < c.world && (send ? r.sendCvBase[peer + 1U] : r.recvCvBase[peer + 1U]) <= cv) ++peer;
    return peer;
}
URMA_ROUTE_INLINE uint32_t PlanExpertForPeerRow(uint32_t peer, uint32_t row, const Config& c,
                                                URMA_ROUTE_UB DeviceRouteInfo& r)
{
    uint32_t lo = 0U, hi = c.localExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (r.incoming[peer][mid + 1U] <= row)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}

URMA_ROUTE_INLINE void BuildRouteSummary(Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, bool wave12)
{
    const uint32_t le = c.localExperts;
    r.groupBase[0] = r.compactBase[0] = 0U;
    r.gmm1Base[0] = 0U;
    for (uint32_t e = 0; e < le; ++e) {
        r.sources[e][0] = 0U;
        for (uint32_t s = 0; s < c.world; ++s)
            r.sources[e][s + 1U] = r.sources[e][s] + r.incoming[s][e + 1U] - r.incoming[s][e];
        const uint32_t rows = r.sources[e][c.world];
        r.groupBase[e + 1U] = r.groupBase[e] + (rows + 255U) / 256U;
        r.compactBase[e + 1U] = r.compactBase[e] + rows;
        r.gmm1Base[e + 1U] = r.gmm1Base[e] + uint64_t(rows / 256U) * PlanGmm1Tiles(256U, c) +
                             (rows % 256U ? PlanGmm1Tiles(rows % 256U, c) : 0U);
    }
    c.fragmentCount = r.groupBase[le];
    c.receiveRows = r.compactBase[le];
    c.gmm1Tiles = r.gmm1Base[le];
    c.gmm2Tiles = uint64_t(c.fragmentCount) * (c.hidden / 256U);
    c.selfSendRow = r.outgoing[c.rank * le];
    c.selfRecvRow = uint64_t(c.rank) * c.sourceRowCapacity;
    c.selfRows = r.incoming[c.rank][le];
    c.packRoundCount = 0U;
    r.sendCvBase[0] = r.recvCvBase[0] = 0U;
    for (uint32_t p = 0; p < c.world; ++p) {
        const uint32_t outgoing = r.outgoing[(p + 1U) * le] - r.outgoing[p * le];
        const uint32_t sends = (outgoing + c.cvRows - 1U) / c.cvRows;
        const uint32_t receives = (r.incoming[p][le] + c.cvRows - 1U) / c.cvRows;
        r.sendCvBase[p + 1U] = r.sendCvBase[p] + sends;
        r.recvCvBase[p + 1U] = r.recvCvBase[p] + receives;
        c.packRoundCount = RouteMax(c.packRoundCount, sends);
    }
    c.sendChunkCount = r.sendCvBase[c.world];
    c.recvChunkCount = r.recvCvBase[c.world];
    c.computeTaskCount = 2U * (wave12 ? (c.fragmentCount + 11U) / 12U : le);
    c.routeSegmentCount = c.routeLinkCount = 0U;
}

URMA_ROUTE_INLINE uint32_t CountGroupSegments(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, uint32_t group)
{
    const uint32_t e = PlanExpertForGroup(group, c, r), first = (group - r.groupBase[e]) * 256U;
    const uint32_t last = RouteMin(first + 256U, r.sources[e][c.world]);
    uint32_t count = 0U;
    for (uint32_t s = 0; s < c.world; ++s) {
        const uint32_t lo = RouteMax(first, r.sources[e][s]), hi = RouteMin(last, r.sources[e][s + 1U]);
        if (lo < hi) {
            const uint32_t row = r.incoming[s][e] + lo - r.sources[e][s];
            count += (row % c.cvRows + hi - lo + c.cvRows - 1U) / c.cvRows;
        }
    }
    return count;
}
URMA_ROUTE_INLINE uint32_t CountCvLinks(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, uint32_t index)
{
    const uint32_t peer = PlanPeerForCv(index, c, r, false), cv = index - r.recvCvBase[peer];
    uint32_t row = cv * c.cvRows, count = 0U, e = PlanExpertForPeerRow(peer, row, c, r);
    const uint32_t last = RouteMin(row + c.cvRows, r.incoming[peer][c.localExperts]);
    while (row < last) {
        while (r.incoming[peer][e + 1U] <= row) ++e;
        const uint32_t expertRow = r.sources[e][peer] + row - r.incoming[peer][e];
        row += RouteMin(RouteMin(last - row, r.incoming[peer][e + 1U] - row), 256U - expertRow % 256U);
        ++count;
    }
    return count;
}

template <class Prefix, class Output>
URMA_ROUTE_INLINE void EmitRouteGroups(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Prefix& prefix, Output& out,
                                       RouteWorkRange range)
{
    for (uint32_t g = range.first; g < range.last; ++g) {
        const uint32_t e = PlanExpertForGroup(g, c, r), first = (g - r.groupBase[e]) * 256U;
        const uint32_t last = RouteMin(first + 256U, r.sources[e][c.world]);
        uint32_t segment = prefix.Group(g);
        Fragment f{e,
                   segment,
                   uint64_t(r.compactBase[e]) + first,
                   last - first,
                   prefix.Group(g + 1U) - segment,
                   r.gmm1Base[e] + uint64_t(g - r.groupBase[e]) * PlanGmm1Tiles(256U, c),
                   uint64_t(g) * (c.hidden / 256U)};
        for (uint32_t s = 0; s < c.world; ++s) {
            uint32_t lo = RouteMax(first, r.sources[e][s]);
            const uint32_t hi = RouteMin(last, r.sources[e][s + 1U]);
            while (lo < hi) {
                const uint32_t peerRow = r.incoming[s][e] + lo - r.sources[e][s], cv = peerRow / c.cvRows;
                const uint32_t n = RouteMin(hi - lo, c.cvRows - peerRow % c.cvRows);
                out.Segment(segment++,
                            RowSegment{s, cv, uint64_t(s) * c.sourceRowCapacity + peerRow, r.returnBase[s] + peerRow,
                                       lo - first, n, s * c.maxPeerChunks + cv, 0U});
                lo += n;
            }
        }
        out.FragmentAt(g, f);
    }
}
template <class Prefix, class Output>
URMA_ROUTE_INLINE void EmitRouteReceives(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Prefix& prefix, Output& out,
                                         RouteWorkRange range, bool reorder)
{
    for (uint32_t index = range.first; index < range.last; ++index) {
        const uint32_t peer = PlanPeerForCv(index, c, r, false), cv = index - r.recvCvBase[peer];
        const uint32_t first = cv * c.cvRows, last = RouteMin(first + c.cvRows, r.incoming[peer][c.localExperts]);
        uint32_t row = first, e = PlanExpertForPeerRow(peer, row, c, r), link = prefix.Link(index);
        Chunk chunk{peer,
                    cv,
                    uint64_t(peer) * c.sourceRowCapacity + first,
                    r.returnBase[peer] + first,
                    last - first,
                    link,
                    prefix.Link(index + 1U) - link,
                    0U};
        while (row < last) {
            while (r.incoming[peer][e + 1U] <= row) ++e;
            const uint32_t er = r.sources[e][peer] + row - r.incoming[peer][e];
            const uint32_t g = r.groupBase[e] + er / 256U;
            const uint32_t n = RouteMin(RouteMin(last - row, r.incoming[peer][e + 1U] - row), 256U - er % 256U);
            out.Link(link, g);
            if (reorder)
                out.ReorderSpan(link, uint64_t(r.compactBase[e]) + er,
                                peer == c.rank ? r.returnBase[peer] + row : uint64_t(peer) * c.sourceRowCapacity + row,
                                g, n);
            ++link;
            row += n;
        }
        if (reorder) out.ReorderRange(index, chunk.fragmentBegin, chunk.fragmentCount);
        out.Receive(index, chunk);
    }
}
template <class Output>
URMA_ROUTE_INLINE void EmitRouteSends(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Output& out,
                                      RouteWorkRange range)
{
    for (uint32_t index = range.first; index < range.last; ++index) {
        const uint32_t peer = PlanPeerForCv(index, c, r, true), cv = index - r.sendCvBase[peer];
        const uint32_t base = r.outgoing[peer * c.localExperts], first = cv * c.cvRows;
        const uint32_t rows = r.outgoing[(peer + 1U) * c.localExperts] - base;
        out.Send(index, Chunk{peer, cv, uint64_t(base) + first, uint64_t(c.rank) * c.sourceRowCapacity + first,
                              RouteMin(c.cvRows, rows - first), 0U, 0U, 0U});
    }
}
template <class Output>
URMA_ROUTE_INLINE void EmitRoutePack(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Output& out,
                                     RouteWorkRange range)
{
    for (uint32_t cv = range.first; cv < range.last; ++cv) {
        uint64_t total = 0U;
        for (uint32_t peer = 0; peer < kWorld; ++peer) {
            Chunk chunk{};
            if (peer < c.world) {
                const uint32_t base = r.outgoing[peer * c.localExperts],
                               rows = r.outgoing[(peer + 1U) * c.localExperts] - base;
                const uint32_t first = cv * c.cvRows;
                if (first < rows)
                    chunk = Chunk{peer,
                                  cv,
                                  uint64_t(base) + first,
                                  uint64_t(c.rank) * c.sourceRowCapacity + first,
                                  RouteMin(c.cvRows, rows - first),
                                  0U,
                                  0U,
                                  0U};
            }
            out.PackChunk(cv, peer, chunk);
            total += chunk.rows;
        }
        out.PackRows(cv, total);
    }
}
template <class Output>
URMA_ROUTE_INLINE void EmitRouteTasks(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, Output& out, bool wave12)
{
    uint32_t task = 0U;
    if (wave12) {
        for (uint32_t first = 0U, wave = 0U; first < c.fragmentCount; first += 12U, ++wave) {
            out.Batch(wave, ComputeBatch{first, RouteMin(12U, c.fragmentCount - first)});
            out.Task(task++, ComputeBatchTask{ComputeBatchStage::Gmm1, wave});
            out.Task(task++, ComputeBatchTask{ComputeBatchStage::Gmm2, wave});
        }
    } else {
        for (uint32_t e = 0; e < c.localExperts; ++e)
            out.Batch(e, ComputeBatch{r.groupBase[e], r.groupBase[e + 1U] - r.groupBase[e]});
        uint32_t g1 = 0U, pending = 0U;
        for (uint32_t g2 = 0U; g2 < c.localExperts; ++g2) {
            while (g1 < c.localExperts && (g1 < RouteMin(c.localExperts, g2 + 2U) || pending < c.gmm1AheadMGroups)) {
                out.Task(task++, ComputeBatchTask{ComputeBatchStage::Gmm1, g1});
                pending += r.groupBase[g1 + 1U] - r.groupBase[g1];
                ++g1;
            }
            out.Task(task++, ComputeBatchTask{ComputeBatchStage::Gmm2, g2});
            pending -= r.groupBase[g2 + 1U] - r.groupBase[g2];
        }
    }
}
}  // namespace MegaMoeUrma

namespace MegaMoeUrma {
// Temporary per-expert totals occupy the same arrays that the leader later
// converts to prefixes. No consumer sees these arrays until final publication.
URMA_ROUTE_INLINE void SummarizeRouteExperts(const Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, uint32_t begin,
                                             uint32_t end)
{
    for (uint32_t e = begin; e < end; ++e) {
        for (uint32_t s = 0; s <= kWorld; ++s) r.sources[e][s] = 0U;
        for (uint32_t s = 0; s < c.world; ++s)
            r.sources[e][s + 1U] = r.sources[e][s] + r.incoming[s][e + 1U] - r.incoming[s][e];
        const uint32_t rows = r.sources[e][c.world];
        r.groupBase[e + 1U] = (rows + 255U) / 256U;
        r.compactBase[e + 1U] = rows;
        r.gmm1Base[e + 1U] =
            uint64_t(rows / 256U) * PlanGmm1Tiles(256U, c) + (rows % 256U ? PlanGmm1Tiles(rows % 256U, c) : 0U);
    }
}

URMA_ROUTE_INLINE void FinishRouteSummaryPrefix(Config& c, URMA_ROUTE_UB DeviceRouteInfo& r, bool wave12)
{
    const uint32_t le = c.localExperts;
    r.groupBase[0] = r.compactBase[0] = 0U;
    r.gmm1Base[0] = 0U;
    for (uint32_t e = 0; e < le; ++e) {
        r.groupBase[e + 1U] += r.groupBase[e];
        r.compactBase[e + 1U] += r.compactBase[e];
        r.gmm1Base[e + 1U] += r.gmm1Base[e];
    }
    c.fragmentCount = r.groupBase[le];
    c.receiveRows = r.compactBase[le];
    c.gmm1Tiles = r.gmm1Base[le];
    c.gmm2Tiles = uint64_t(c.fragmentCount) * (c.hidden / 256U);
    c.selfSendRow = r.outgoing[c.rank * le];
    c.selfRecvRow = uint64_t(c.rank) * c.sourceRowCapacity;
    c.selfRows = r.incoming[c.rank][le];
    c.packRoundCount = 0U;
    r.sendCvBase[0] = r.recvCvBase[0] = 0U;
    for (uint32_t p = 0; p < c.world; ++p) {
        const uint32_t rows = r.outgoing[(p + 1U) * le] - r.outgoing[p * le];
        const uint32_t sends = (rows + c.cvRows - 1U) / c.cvRows;
        r.sendCvBase[p + 1U] = r.sendCvBase[p] + sends;
        r.recvCvBase[p + 1U] = r.recvCvBase[p] + (r.incoming[p][le] + c.cvRows - 1U) / c.cvRows;
        c.packRoundCount = RouteMax(c.packRoundCount, sends);
    }
    c.sendChunkCount = r.sendCvBase[c.world];
    c.recvChunkCount = r.recvCvBase[c.world];
    c.computeTaskCount = 2U * (wave12 ? (c.fragmentCount + 11U) / 12U : le);
    c.routeSegmentCount = c.routeLinkCount = 0U;
}
}  // namespace MegaMoeUrma
