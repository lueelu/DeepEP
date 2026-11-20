// Copyright (c) 2026, Lu Lu
#pragma once

#include <stdexcept>
#include "kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "kernels/mega_moe/common/mega_moe_workspace.hpp"

namespace MegaMoeImpl {
// Control records are plan-private. Every payload class is independently sized
// against the root envelope so a view may use different H, K and packet strides.
inline int64_t LayoutArena(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t h,
                           int64_t ranks_per_node, uint32_t domain, uint32_t configs, MegaMoeArenaOffsets& out)
{
    if (configs < 1 || configs > 4096 || domain >= configs) throw std::invalid_argument("invalid MegaMoE plan domain");
    const bool roce = ranks_per_node == 16 && world == 32;
    const int64_t dump = EXCEPTION_DUMP_REGION_SIZE;
    int64_t offset = dump + PEERMEM_DATA_OFFSET;
    auto reserve = [&](int64_t bytes) {
        const int64_t start = offset;
        offset += Ops::Base::CeilAlign(bytes, int64_t{512});
        return start;
    };
    const int64_t slot =
        CalcRailTopkIdsSlotDataBytes(tokens, topk) + (roce ? CalcCombineWeightsBytes(tokens, topk) : 0);
    const int64_t rails = CalcServerNum(world, RailServerSize(static_cast<uint32_t>(ranks_per_node)));
    out.counts = reserve(CalcMteExpertCountRecvSize(experts, world)) - dump;
    out.rail = reserve(rails * slot) - dump;
    out.control = reserve(roce ? static_cast<int64_t>(HierarchicalSignal::Count) * 512 : 0) - dump;
    out.ready = reserve(roce ? 16 * 512 : 0) - dump;
    out.commit = reserve(roce ? HIERARCHICAL_COMBINE_QPS * 512 : 0) - dump;
    out.compact = reserve(roce ? (1 + 2 * tokens) * 4 : 0) - dump;
    out.gather = reserve(roce ? world * CalcRailTopkIdsSlotDataBytes(tokens, topk) : 0) - dump;
    out.closure = reserve(roce ? (64 + 2 * 16 + 2) * 512 : 0) - dump;
    const int64_t stride = offset;
    const int64_t origin = domain * stride + dump;
    offset = stride * configs;
    const int64_t plain = CalcQuantTokenScaleBytes(h, 1, topk, false);
    const int64_t packet = Ops::Base::CeilDiv(plain, int64_t{480}) * 512;
    out.route = reserve(Ops::Base::CeilAlign(tokens, int64_t{32}) * 4 * experts * world) - origin;
    out.quant = reserve(tokens * plain) - origin;
    out.combine = reserve(tokens * topk * h * 2) - origin;
    out.relay = reserve(roce ? tokens * packet : 0) - origin;
    out.packetSend = reserve(roce ? tokens * packet : 0) - origin;
    out.gateway = reserve(roce ? CalcGatewayCombineBytes(tokens, topk, h) : 0) - origin;
    out.partialSend = reserve(roce ? tokens * h * 4 : 0) - origin;
    out.partialRecv = reserve(roce ? tokens * h * 4 : 0) - origin;
    out.domain = domain;
    out.domainBits = 0;
    while ((1U << out.domainBits) < configs) ++out.domainBits;
    return Ops::Base::CeilAlign(offset, int64_t{2} << 20);
}

}  // namespace MegaMoeImpl
