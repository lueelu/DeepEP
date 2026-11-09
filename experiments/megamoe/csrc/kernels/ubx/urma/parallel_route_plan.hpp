// Experimental commfirst parallel descriptor compiler. No task stealing or
// atomic append: each worker owns contiguous ranges of every output table.
#pragma once
#include "device_route_plan.hpp"

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
