// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
#include "../v2.hpp"
#include "ep_memory_server_dedup_combine_config.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "ep_memory_server_dedup_count_layout.h"
namespace UbxCombine {
inline bool BuildLayout(int64_t p, int64_t b, int64_t h, int64_t k, int64_t e, deepep::ubx::PutLaunchPlan& out,
                        uint32_t coreNum, uint32_t ubBytes, int64_t expertRows)
{
    out = {};
    if (expertRows < 0 || !UbxDispatch::LlCoreCountValid(coreNum) || coreNum > deep_ep::kEpServerDedupMaxVecCores || p < 2 || (p < 16 && p != 2 && p != 4 && p != 8) ||
        p > UbxDispatch::DISPATCH_DEDUP_MAX_RANK_SIZE || p % UbxDispatch::LlGroupForWorld(uint32_t(p)) || b <= 0 ||
        b > 65536 || h <= 0 || h > ubBytes / 12 || k < 1 || k > 32 || e < p || e > 1024 ||
        e % p || k > e)
        return false;
    const uint64_t capacity = p * b * (k < e / p ? k : e / p);
    const uint64_t slots = uint64_t(p) * b * k;
    if (capacity > INT32_MAX || slots - 1U > INT32_MAX) return false;
    // An exact Dispatch may return fewer expert rows (including zero). The
    // device checks this same bound before touching expert storage.
    const uint32_t rowLimit = uint32_t(uint64_t(expertRows) < capacity ? uint64_t(expertRows) : capacity);
    const auto peak = PlanPutRows(rowLimit, b, h, k, EpCombineRecvCoreCount(coreNum), ubBytes);
    if (!peak.lanes) return false;
    const uint64_t packet = (h * 2 + 479) / 480 * 512;
    const uint64_t bytes = COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + b * k * packet + slots * packet;
    if (bytes > UbxDispatch::kEpServerDedupWindowUsableBytes ||
        UbxDispatch::DispatchServerDedupCountLayout(e, coreNum).stateBytes >
            COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET)
        return false;
    out.capacityRows = rowLimit;
    // Row metadata grows monotonically. If even the maximum legal count keeps
    // the whole table, every route for this shape can omit the tiled loops.
    out.wholeTable = peak.rows == rowLimit;
    out.remoteFirst = b >= 32;
    return true;
}
}  // namespace UbxCombine
