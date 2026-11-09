// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "route_runtime.hpp"

namespace MegaMoeUrma {

// Send/Pack read only the early outgoing prefix. Receive helpers require the
// late summary handoff. Both parts are immutable once individually published;
// transport addresses and peer/CV traversal order remain unchanged.
__aicore__ inline uint32_t RuntimeSendCvCount(const Arguments& args, uint32_t peer)
{
    const auto* routes = RouteInfo(args);
    const uint32_t le = args.config.localExperts;
    const uint32_t rows = routes->outgoing[(peer + 1U) * le] - routes->outgoing[peer * le];
    return (rows + args.config.cvRows - 1U) / args.config.cvRows;
}

__aicore__ inline uint32_t RuntimeReceiveCvCount(const Arguments& args, uint32_t peer)
{
    const auto* routes = RouteInfo(args);
    return routes->recvCvBase[peer + 1U] - routes->recvCvBase[peer];
}

__aicore__ inline Chunk RuntimeSendChunk(const Arguments& args, uint32_t peer, uint32_t cv)
{
    if (peer >= args.config.world) return {};
    const auto* routes = RouteInfo(args);
    const uint32_t localExperts = args.config.localExperts;
    const uint32_t base = routes->outgoing[peer * localExperts];
    const uint32_t rows = routes->outgoing[(peer + 1U) * localExperts] - base;
    const uint64_t first = uint64_t(cv) * args.config.cvRows;
    // Pack retains an empty virtual slot for exhausted peers.
    if (first >= rows) return {};
    const uint32_t remaining = rows - static_cast<uint32_t>(first);
    return {peer,
            cv,
            uint64_t(base) + first,
            uint64_t(args.config.rank) * args.config.sourceRowCapacity + first,
            remaining < args.config.cvRows ? remaining : args.config.cvRows,
            0U,
            0U,
            0U};
}

__aicore__ inline Chunk RuntimeReceiveChunk(const Arguments& args, uint32_t peer, uint32_t cv)
{
    if (peer >= args.config.world) return {};
    const auto* routes = RouteInfo(args);
    const uint32_t rows = routes->incoming[peer][args.config.localExperts];
    const uint64_t first = uint64_t(cv) * args.config.cvRows;
    if (first >= rows) return {};
    const uint32_t remaining = rows - static_cast<uint32_t>(first);
    // fragmentBegin/count are intentionally unused: readiness walks the small
    // expert/source prefixes rather than an eagerly generated Link array.
    return {peer,
            cv,
            uint64_t(peer) * args.config.sourceRowCapacity + first,
            routes->returnBase[peer] + first,
            remaining < args.config.cvRows ? remaining : args.config.cvRows,
            0U,
            0U,
            0U};
}

__aicore__ inline uint64_t RuntimePackRoundRows(const Arguments& args, uint32_t round)
{
    uint64_t rows = 0U;
    for (uint32_t peer = 0U; peer < args.config.world; ++peer) rows += RuntimeSendChunk(args, peer, round).rows;
    return rows;
}

// First expert with an upper prefix strictly beyond this peer-local row.
// Equal prefixes belong to empty experts and must be skipped.
__aicore__ inline uint32_t RuntimePeerRowExpert(const Arguments& args, uint32_t peer, uint32_t row)
{
    const auto* routes = RouteInfo(args);
    uint32_t first = 0U, last = args.config.localExperts;
    while (first < last) {
        const uint32_t middle = first + (last - first) / 2U;
        if (routes->incoming[peer][middle + 1U] <= row)
            first = middle + 1U;
        else
            last = middle;
    }
    return first;
}

}  // namespace MegaMoeUrma
