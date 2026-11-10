// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/parallel_route_plan.hpp"

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
