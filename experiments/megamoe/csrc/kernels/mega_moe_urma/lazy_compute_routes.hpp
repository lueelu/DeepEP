// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_urma/route_runtime.hpp"

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
