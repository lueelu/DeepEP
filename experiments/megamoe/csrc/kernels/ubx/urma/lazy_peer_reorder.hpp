// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "runtime_comm_routes.hpp"

namespace MegaMoeUrma {
__aicore__ inline Chunk RuntimeReceiveChunkByIndex(const Arguments& a, uint32_t index)
{
    const auto* r = RouteInfo(a);
    uint32_t peer = 0U;
    while (peer + 1U < a.config.world && r->recvCvBase[peer + 1U] <= index) ++peer;
    return RuntimeReceiveChunk(a, peer, index - r->recvCvBase[peer]);
}

// Same expert/M256 intersections and destination order as the old span table.
// A span is identified by its first peer-local row. This is unique even when
// one M-group crosses several CVs, and fits the existing one-bit-per-row bound.
struct RuntimePeerSpans {
    const volatile __gm__ DeviceRouteInfo* info;
    uint32_t peer, rank, row, end, expert;
    uint64_t sourcePitch;
    __aicore__ RuntimePeerSpans(const Arguments& a, const Chunk& chunk)
        : info(RouteInfo(a)),
          peer(chunk.peer),
          rank(a.config.rank),
          row(chunk.cv * a.config.cvRows),
          end(row + chunk.rows),
          expert(RuntimePeerRowExpert(a, peer, row)),
          sourcePitch(a.config.sourceRowCapacity)
    {
    }
    __aicore__ inline bool Next(PeerReorderSpan& span, uint32_t& bit)
    {
        if (row >= end) return false;
        while (info->incoming[peer][expert + 1U] <= row) ++expert;
        const uint32_t expertRow = info->sources[expert][peer] + row - info->incoming[peer][expert];
        uint32_t n = RouteMin(end - row, info->incoming[peer][expert + 1U] - row);
        n = RouteMin(n, 256U - expertRow % 256U);
        bit = row;
        span = {uint64_t(info->compactBase[expert]) + expertRow,
                peer == rank ? info->returnBase[peer] + row : uint64_t(peer) * sourcePitch + row,
                info->groupBase[expert] + expertRow / 256U, n};
        row += n;
        return true;
    }
};
__aicore__ inline PeerReorderRange RuntimePeerReorderRange(const Arguments& a, uint32_t index)
{
    RuntimePeerSpans cursor(a, RuntimeReceiveChunkByIndex(a, index));
    PeerReorderSpan span{};
    uint32_t bit, count = 0U;
    while (cursor.Next(span, bit)) ++count;
    // Protocol6 encodes the global CV index here, not a materialized span offset.
    return {index, count};
}
}  // namespace MegaMoeUrma
