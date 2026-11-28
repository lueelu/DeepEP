// Copyright (c) 2026, Lu Lu
#include "v2.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "kernels/combine_layout.h"
#include "kernels/ep_memory_server_dedup_combine_tiling.h"

namespace deepep::ubx {
uint32_t combine_ub_bytes(int64_t available)
{
    if (available < 32) throw std::invalid_argument("UBX Combine requires a valid per-vector-core UB size");
    const auto bytes = std::min<int64_t>(available, UbxCombine::kEpServerDedupCombineMaxUbBytes);
    return static_cast<uint32_t>(bytes) / 32U * 32U;
}

PutLaunchPlan v2_plan(int world, int tokens, int hidden, int topk, int experts, uint32_t cores,
                      uint32_t ranks_per_host, uint32_t ub_bytes, int64_t expert_rows)
{
    if (ranks_per_host != UbxDispatch::kLlPhysicalRanksPerHost || world < int(ranks_per_host) ||
        world % ranks_per_host || (world & (world - 1)))
        throw std::invalid_argument("UBX and Dispatch must use the same participating ranks per host");
    PutLaunchPlan plan{};
    if (!UbxCombine::BuildLayout(world, tokens, hidden, topk, experts, plan, cores, ub_bytes, expert_rows))
        throw std::invalid_argument("UBX Combine exceeds topology, shape, UB or workspace limits");
    return plan;
}

std::array<uint8_t, 16> v2_tiling(int world, int tokens, int hidden, int topk, int experts, uint8_t dtype, int capacity,
                                  uint32_t cores, uint32_t ranks_per_host, uint32_t ub_bytes)
{
    v2_plan(world, tokens, hidden, topk, experts, cores, ranks_per_host, ub_bytes);
    if (capacity < tokens || capacity > 65535 ||
        (dtype != UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_BFP16 && dtype != UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_FP16))
        throw std::invalid_argument("Invalid UBX Combine capacity or dtype");
    UbxCombine::EpServerDedupCombineTilingData data{};
    data.bs = tokens;
    data.h = hidden;
    data.topK = topk;
    data.moeExpertNum = experts;
    data.dtype = dtype;
    data.numMaxTokensPerRank = capacity;
    data.tilingKey = UbxDispatch::LlTilingKeyForWorld(world);
    std::array<uint8_t, 16> result{};
    static_assert(sizeof(data) == sizeof(result), "UBX tiling ABI");
    std::memcpy(result.data(), &data, sizeof(data));
    return result;
}
}  // namespace deepep::ubx
