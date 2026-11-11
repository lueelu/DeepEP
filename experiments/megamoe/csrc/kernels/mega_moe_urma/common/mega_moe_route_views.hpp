// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"
#include "kernel_operator.h"

// Device views and iterators over immutable routing metadata.

namespace MegaMoeUrma {
// Outgoing prefixes are immutable after local preparation. The receive portion
// is published later on separate cache lines and must be acquired before use.
__aicore__ inline const volatile __gm__ DeviceRouteInfo* RouteInfo(const Arguments& a)
{
    return reinterpret_cast<const volatile __gm__ DeviceRouteInfo*>(a.routeInfo);
}
__aicore__ inline uint32_t RuntimeExpertForExpandedRow(const Arguments& a, uint64_t row)
{
    const auto* r = RouteInfo(a);
    uint32_t lo = 0U, hi = a.config.world * a.config.localExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (r->outgoing[mid + 1U] <= row)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}
}  // namespace MegaMoeUrma

namespace MegaMoeUrma {

// 当前 M-group 的行段按 source、CV 单调推进。只保存几个游标，不占用
// GMM/反量化正在使用的 UB，也不在 RoutePrepare 生成整张 Segment 表。
struct RuntimeGroupSegments {
    const volatile __gm__ DeviceRouteInfo* info = nullptr;
    uint32_t expert = 0U;
    uint32_t first = 0U;
    uint32_t last = 0U;
    uint32_t world = 0U;
    uint32_t cvRows = 0U;
    uint32_t maxPeerChunks = 0U;
    uint64_t sourcePitch = 0U;

    struct Cursor {
        const volatile __gm__ DeviceRouteInfo* info;
        uint32_t expert, first, last, world, cvRows, maxPeerChunks;
        uint64_t sourcePitch;
        uint32_t source = 0U;
        uint32_t nextRow = 0U;

        __aicore__ inline bool Next(RowSegment& segment)
        {
            while (source < world) {
                const uint32_t sourceBegin = info->sources[expert][source];
                const uint32_t sourceEnd = info->sources[expert][source + 1U];
                const uint32_t lo = RouteMax(first, sourceBegin);
                const uint32_t hi = RouteMin(last, sourceEnd);
                if (nextRow < lo) nextRow = lo;
                if (nextRow >= hi) {
                    ++source;
                    continue;
                }
                const uint32_t peerRow = info->incoming[source][expert] + nextRow - sourceBegin;
                const uint32_t cv = peerRow / cvRows;
                const uint32_t rows = RouteMin(hi - nextRow, cvRows - peerRow % cvRows);
                segment = {source,
                           cv,
                           uint64_t(source) * sourcePitch + peerRow,
                           info->returnBase[source] + peerRow,
                           nextRow - first,
                           rows,
                           source * maxPeerChunks + cv,
                           0U};
                nextRow += rows;
                return true;
            }
            return false;
        }
    };

    // 与通用 SegmentedRowIO 的默认 GM provider 保持相同入口。当前 provider
    // 从计数前缀推导，故不读取历史的 descriptor 指针/count。
    __aicore__ inline Cursor MakeCursor(const __gm__ RowSegment* = nullptr, uint32_t = 0U) const
    {
        return {info, expert, first, last, world, cvRows, maxPeerChunks, sourcePitch, 0U, first};
    }
};

__aicore__ inline uint32_t RuntimeExpertForGroup(const Arguments& args, uint32_t group)
{
    const auto* info = RouteInfo(args);
    uint32_t lo = 0U, hi = args.config.localExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (info->groupBase[mid + 1U] <= group)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}

__aicore__ inline RuntimeGroupSegments RuntimeSegments(const Arguments& args, uint32_t expert, uint32_t first,
                                                       uint32_t rows)
{
    return {RouteInfo(args),
            expert,
            first,
            first + rows,
            args.config.world,
            args.config.cvRows,
            args.config.maxPeerChunks,
            args.config.sourceRowCapacity};
}

__aicore__ inline RuntimeGroupSegments RuntimeSegments(const Arguments& args, const Fragment& fragment)
{
    const auto* info = RouteInfo(args);
    const uint32_t first = (fragment.segmentBegin - info->groupBase[fragment.expert]) * kComputeTileM;
    return RuntimeSegments(args, fragment.expert, first, fragment.rows);
}

__aicore__ inline Fragment RuntimeFragment(const Arguments& args, uint32_t group)
{
    const auto* info = RouteInfo(args);
    const uint32_t expert = RuntimeExpertForGroup(args, group);
    const uint32_t first = (group - info->groupBase[expert]) * kComputeTileM;
    const uint32_t rows = RouteMin(kComputeTileM, info->sources[expert][args.config.world] - first);
    // segmentBegin 在本协议中是 group ID；不是一张预构建 Segment 表的偏移。
    // Cursor自行判断结束，segmentCount=0表示未物化；避免每次轮询重扫全部来源。
    return {expert,
            group,
            uint64_t(info->compactBase[expert]) + first,
            rows,
            0U,
            info->gmm1Base[expert] +
                uint64_t(group - info->groupBase[expert]) * Gmm1ComputeTileCount(kComputeTileM, args.config),
            uint64_t(group) * (args.config.hidden / kComputeTileN)};
}

// 随机访问仅供需要单条 descriptor 的调用方；连续消费应使用 MakeCursor，
// 避免为每个 ordinal 重新从 source0 遍历。
__aicore__ inline RowSegment RuntimeSegment(const Arguments& args, uint32_t group, uint32_t ordinal)
{
    const auto fragment = RuntimeFragment(args, group);
    auto cursor = RuntimeSegments(args, fragment).MakeCursor();
    RowSegment segment{};
    for (uint32_t i = 0U; i <= ordinal; ++i) {
        if (!cursor.Next(segment)) return {};
    }
    return segment;
}

}  // namespace MegaMoeUrma

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
