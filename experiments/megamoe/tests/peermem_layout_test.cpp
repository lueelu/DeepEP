// Copyright (c) 2026, Lu Lu
#include <algorithm>
#include "../csrc_ep32/arena.hpp"
#ifndef __aicore__
#define __aicore__
#endif
#include "../csrc_ep32/kernels/mega_moe/catlass_gmm/a8w4_k_plan.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_workspace.hpp"

using namespace MegaMoeImpl;

using GmmKernel::MakeA8W4KPlan;
static_assert(MakeA8W4KPlan(128, 256, true).kb == 512);
static_assert(MakeA8W4KPlan(129, 256, true).kb == 256);
static_assert(MakeA8W4KPlan(16, 256, true).ka == 4096);
static_assert(MakeA8W4KPlan(64, 256, true).ka == 1024);
static_assert(MakeA8W4KPlan(96, 128, true).ka == 512);
static_assert(MakeA8W4KPlan(64, 256, false).kb == 256);
static_assert(sizeof(MegaMoeArenaOffsets) == 136);
static_assert(sizeof(MegaMoeTilingData) == 368);
void checkPlanDomains()
{
    // Every rank receives one stripe per lane, containing the four source meshes in order.
    for (uint32_t host = 0; host < 2; ++host) {
        uint32_t slot = 16U * host;
        for (uint32_t lane = 0; lane < 4; ++lane)
            for (uint32_t mesh = 0; mesh < 4; ++mesh) {
                const uint32_t source = 16U * host + 4U * mesh + lane;
                assert(RouteGatherSlot(source) == slot++);
            }
    }
    for (uint32_t bits : {0U, 4U, 12U}) {
        uint32_t domain = (1U << bits) - 1U;
        for (uint32_t epoch : {0U, 1U, 127U, 128U, 259U}) {
            uint32_t seq = NextDomainSequence(epoch, domain, bits);
            assert((seq >> bits) == epoch + 1U && (seq & domain) == domain);
        }
        assert(NextDomainSequence((0x7fffffffU >> bits) - 1U, domain, bits) == 0U);
    }
    for (uint32_t bs : {32U, 48U, 64U, 72U, 96U, 128U}) {
        MegaMoeTilingData t{};
        t.topoType = TOPO_TYPE_MTE_ROCE;
        t.epWorldSize = 32;
        t.dispatchRanksPerNode = 16;
        t.aicNum = 28;
        t.blockAivNum = 56;
        t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
        t.combineQuantMode = COMBINE_NO_QUANT;
        t.bs = bs;
        bool dsv = bs >= 96;
        t.h = dsv ? 5120 : 3584;
        t.hiddenDim = dsv ? 4608 : 2560;
        t.topK = dsv ? 6 : 8;
        t.moeExpertPerRank = dsv ? 13 : 12;
        t.sharedExpertNum = dsv ? 1 : 0;
        t.sharedFp8 = dsv;
        assert(UseSharedRouteTable(&t));
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
        assert(UseEarlyClosure(&t) == (bs == 32 || dsv));
#else
        assert(!UseEarlyClosure(&t));
#endif
        assert(UseAiv0CombineOwners(&t) == (bs != 96));
        for (uint32_t count = 1; count <= bs; ++count)
            for (uint32_t lane = 0; lane < 16; ++lane)
                for (uint32_t j = 0; j < count; ++j) {
                    bool owns = bs == 96 ? j % 16 == lane : j >= count * lane / 16 && j < count * (lane + 1) / 16;
                    assert((CompactOwner(&t, j, count) == lane) == owns);
                }
    }
    constexpr uint32_t configs = 17;
    MegaMoeArenaOffsets root{};
    const auto capacity = LayoutArena(32, 13, 128, 8, 5120, 16, 0, configs, root);
    for (uint32_t i = 0; i < configs; ++i) {
        MegaMoeArenaOffsets view{};
        assert(LayoutArena(32, 13, 128, 8, 5120, 16, i, configs, view) == capacity);
        assert(view.domain == i && view.domainBits == 5);
        const int64_t base = root.route - view.route;
        for (auto member : {&MegaMoeArenaOffsets::route, &MegaMoeArenaOffsets::quant, &MegaMoeArenaOffsets::combine,
                            &MegaMoeArenaOffsets::packetSend, &MegaMoeArenaOffsets::relay, &MegaMoeArenaOffsets::gateway,
                            &MegaMoeArenaOffsets::partialSend, &MegaMoeArenaOffsets::partialRecv})
            assert(base + view.*member == root.*member && root.*member < capacity);
        assert(view.counts == root.counts && view.closure == root.closure);
        assert(i == 0 || base >= root.closure + 98 * 512);
    }
}
static_assert(offsetof(MegaMoeTilingData, dispatchRanksPerNode) == 216);
static_assert(offsetof(MegaMoeTilingData, roceScratchUbOffset) == 220);
static_assert(offsetof(MegaMoeTilingData, hierarchicalSequence) == 224);

PeermemSizeParams parameters(const MegaMoeTilingData& t)
{
    return {t.numMaxTokensPerRank,      t.topK, t.h,        t.moeExpertPerRank,       t.epWorldSize,         2, 1,
            t.topkWeightsPrefetch != 0, false,  t.topoType, UseDecodeRailRouting(&t), t.dispatchRanksPerNode};
}

MegaMoeTilingData make_tiling(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch,
                              uint8_t mode = GROUPED_MATMUL_MODE_A8W4)
{
    MegaMoeTilingData t{};
    t.numMaxTokensPerRank = capacity;
    t.bs = std::min(capacity, HIERARCHICAL_MAX_TOKENS);
    t.h = h;
    t.topK = topk;
    t.topkWeightsPrefetch = prefetch;
    t.moeExpertPerRank = 12;
    t.epWorldSize = 32;
    t.dispatchRanksPerNode = 16;
    t.blockAivNum = 48;
    t.groupedMatmulMode = mode;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    return t;
}

void check_combine_owner_policy()
{
    for (uint32_t cores : {16U, 28U, 32U}) {
        auto base = make_tiling(96, 5120, 6, false);
        base.hiddenDim = 4608;
        base.moeExpertPerRank = 13;
        base.combineQuantMode = COMBINE_NO_QUANT;
        base.sharedExpertNum = 1;
        base.sharedFp8 = true;
        base.aicNum = cores;
        base.blockAivNum = cores * 2;
        auto check = [](const MegaMoeTilingData& t, bool odd) {
            const bool hierarchical = t.topoType == TOPO_TYPE_MTE_ROCE;
            const uint32_t lanes = std::min(t.bf16Bs96 ? 24U : 16U, static_cast<uint32_t>(t.aicNum));
            const auto jobs = t.blockAivNum - (hierarchical ? lanes : 0U);
            assert(UnpermuteJobCount(&t) == jobs);
            assert(UseAiv0CombineOwners(&t) == (hierarchical && !odd));
            uint32_t outputJob = 0;
            for (uint32_t j = 0; j < t.blockAivNum; ++j) {
                bool owner = hierarchical && j / 2 < lanes && (j % 2 == static_cast<uint32_t>(odd));
                assert(IsHierarchicalCombineOwner(&t, j) == owner);
                assert(UnpermuteJobIndex(&t, j) == (owner ? jobs : outputJob++));
                if (owner) assert(HierarchicalCombineLane(&t, j) == j / 2);
            }
            assert(outputJob == jobs);
            if (!hierarchical) return;
            for (uint32_t rows : {1U, 7U, 16U, 31U, 96U})
                for (uint32_t lane = 0; lane < lanes; ++lane)
                    for (uint32_t i = 0; i < rows; ++i)
                        assert((CompactOwner(&t, i, rows) == lane) ==
                               (odd ? i % lanes == lane : i >= rows * lane / lanes && i < rows * (lane + 1U) / lanes));
        };
        check(base, true);
        auto candidate = base;
        candidate.bf16Bs96 = UseBf16Bs96(&base, false);
        assert(!UseBf16Bs96(&base, true));
        assert(candidate.bf16Bs96 == (cores == 28 && HIERARCHICAL_COMBINE_QPS == 24));
        check(candidate, true);
        assert(UseInputEarlyClosure(&candidate, false) == candidate.bf16Bs96);
        if (candidate.bf16Bs96) assert(UnpermuteJobCount(&candidate) == 32);
        auto variant = [&](auto field, auto value) {
            auto t = base;
            t.*field = value;
            check(t, false);
        };
        variant(&MegaMoeTilingData::sharedExpertNum, 0);
        variant(&MegaMoeTilingData::sharedExpertNum, 2);
        variant(&MegaMoeTilingData::sharedFp8, false);
        variant(&MegaMoeTilingData::bs, 128);
        variant(&MegaMoeTilingData::h, 3584);
        variant(&MegaMoeTilingData::hiddenDim, 2560);
        variant(&MegaMoeTilingData::topK, 8);
        variant(&MegaMoeTilingData::moeExpertPerRank, 12);
        variant(&MegaMoeTilingData::combineQuantMode, COMBINE_NO_QUANT + 1);
        variant(&MegaMoeTilingData::topkWeightsPrefetch, 1);
        variant(&MegaMoeTilingData::groupedMatmulMode, GROUPED_MATMUL_MODE_GENERAL);
        variant(&MegaMoeTilingData::blockAivNum, cores * 2 + 1);
        for (uint32_t unsupported : {15U, 33U}) {
            auto t = base;
            t.aicNum = unsupported;
            t.blockAivNum = 2 * unsupported;
            check(t, false);
        }
        variant(&MegaMoeTilingData::dispatchRanksPerNode, 8);
        variant(&MegaMoeTilingData::epWorldSize, 16);
        variant(&MegaMoeTilingData::topoType, TOPO_TYPE_MTE);
    }
}

void check_hybrid(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch, uint8_t mode)
{
    auto t = make_tiling(capacity, h, topk, prefetch, mode);
    assert(UseDecodeRailRouting(&t));
    const auto size = CalcPeermemLeastSize(parameters(t));
    std::vector<uint8_t> allocation(size + 511);
    auto* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(allocation.data()) + 511) & ~uintptr_t{511});
    PeermemInfo hybrid(base, &t);
    const auto prefix = HierarchicalMetadataPrefixBytes(&t);
    const auto routeBase = hybrid.railTopkIdsRecvPtr - prefix;
    const std::array<uint8_t*, 13> boundaries = {hybrid.maskRecvPtr,
                                                 hybrid.expertCountRecvPtr,
                                                 routeBase,
                                                 hybrid.quantTokenScalePtr,
                                                 hybrid.combineSendPtr,
                                                 hybrid.relayPayloadPtr,
                                                 hybrid.gatewayCombinePtr,
                                                 hybrid.hierarchicalControlPtr,
                                                 hybrid.gatewayPayloadReadyPtr,
                                                 hybrid.combinePartialSendPtr,
                                                 hybrid.combinePartialRecvPtr,
                                                 hybrid.combineReduceCommitPtr,
                                                 base + size};
    assert(hybrid.rankSyncInWorldPtr == base);
    assert(boundaries.front() == base + PEERMEM_DATA_OFFSET);
    for (size_t i = 0; i < boundaries.size(); ++i) {
        assert(reinterpret_cast<uintptr_t>(boundaries[i]) % 512 == 0);
        assert(boundaries[i] >= base && boundaries[i] <= base + size);
        if (i) assert(boundaries[i - 1] < boundaries[i]);
    }
    const int64_t packetRow = (static_cast<int64_t>(h) * 2 + 507) / 508 * 512;
    assert(hybrid.hierarchicalControlPtr - hybrid.gatewayCombinePtr == capacity * topk * packetRow);
    assert(hybrid.gatewayCombinePtr - hybrid.relayPayloadPtr == hybrid.combineSendPtr - hybrid.quantTokenScalePtr);
    assert(hybrid.gatewayPayloadReadyPtr - hybrid.hierarchicalControlPtr ==
           static_cast<int64_t>(HierarchicalSignal::Count) * 512);
    assert(hybrid.combinePartialSendPtr - hybrid.gatewayPayloadReadyPtr == 16 * 512);
    assert(base + size - hybrid.combineReduceCommitPtr == HIERARCHICAL_COMBINE_QPS * 512);
    assert(hybrid.quantTokenScalePtr - routeBase == 2 * CalcRailTopkIdsSlotBytes(&t));
    assert(routeBase - hybrid.expertCountRecvPtr == CalcMteExpertCountRecvSize(t.moeExpertPerRank, 32));
    assert(CalcSendMaskExpertCapacity(&t, 46) >= 16 * t.moeExpertPerRank);

    for (uint32_t bs : {1U, 7U, 17U, 31U, 32U, 33U, 138U, 144U, 255U, 256U, 1U, 32U}) {
        if (bs > capacity) continue;
        t.bs = bs;
        PeermemInfo next(base, &t);
        assert(CalcPeermemLeastSize(parameters(t)) == size);
        assert(next.railTopkIdsRecvPtr - HierarchicalMetadataPrefixBytes(&t) == routeBase);
        assert(next.quantTokenScalePtr == hybrid.quantTokenScalePtr);
        assert(next.combineSendPtr == hybrid.combineSendPtr);
        assert(next.relayPayloadPtr == hybrid.relayPayloadPtr);
        assert(next.hierarchicalControlPtr == hybrid.hierarchicalControlPtr);
        assert(next.combinePartialSendPtr == hybrid.combinePartialSendPtr);
        assert(next.combinePartialRecvPtr == hybrid.combinePartialRecvPtr);
        assert(next.combineReduceCommitPtr == hybrid.combineReduceCommitPtr);
        const auto payloadBytes = static_cast<int64_t>(bs) * CalcQuantTokenScaleBytes(h, 1, topk, false);
        assert(next.quantTokenScalePtr + payloadBytes <= next.combineSendPtr);
        assert(next.relayPayloadPtr + payloadBytes <= next.gatewayCombinePtr);
    }
    if (capacity == 32 && topk == 32) {
        assert(CalcRailTopkIdsSlotBytes(&t) == (9 + 8) * 512);
        assert(hybrid.quantTokenScalePtr - routeBase == 34 * 512);
        assert(hybrid.hierarchicalControlPtr - hybrid.gatewayCombinePtr == 1024 * packetRow);
    }

    // MTE-only remains independent of the fixed two-host weighted layout.
    t.topoType = TOPO_TYPE_MTE;
    PeermemInfo local(base, &t);
    assert(UseDecodeRailRouting(&t) == (mode == GROUPED_MATMUL_MODE_A8W4));
    assert(local.maskRecvPtr == hybrid.maskRecvPtr);
    assert(local.expertCountRecvPtr == hybrid.expertCountRecvPtr);
    if (UseDecodeRailRouting(&t)) {
        const auto weightBytes = 2 * CalcCombineWeightsBytes(capacity, topk);
        assert(local.railTopkIdsRecvPtr == routeBase);
        assert(local.quantTokenScalePtr + weightBytes == hybrid.quantTokenScalePtr);
        assert(local.combineSendPtr + weightBytes == hybrid.combineSendPtr);
        assert(base + CalcPeermemLeastSize(parameters(t)) + weightBytes == hybrid.relayPayloadPtr);
    } else {
        assert(local.railTopkIdsRecvPtr == nullptr);
        assert(local.quantTokenScalePtr - local.expertCountRecvPtr == CalcExpertCountRecvSize(t.moeExpertPerRank, 32));
        assert(local.quantTokenScalePtr < hybrid.quantTokenScalePtr);
    }
    assert(local.gatewayPayloadReadyPtr == nullptr && local.hierarchicalControlPtr == nullptr);
    assert(local.relayPayloadPtr == nullptr && local.gatewayCombinePtr == nullptr);
}

void check_workspace_ready_region(uint8_t mode)
{
    for (const auto [maxOutput, groups] :
         std::array<std::array<uint32_t, 2>, 5>{{{1, 1}, {256, 1}, {257, 2}, {512, 2}, {513, 3}}}) {
        auto t = make_tiling(32, 1024, 32, false, mode);
        t.maxOutputSize = maxOutput;
        t.hiddenDim = 256;
        t.moeExpertPerRank = 2;
        t.aicNum = 3;
        t.combineQuantMode = COMBINE_NO_QUANT;
        std::vector<uint8_t> allocation(4 * 1024 * 1024 + 511, 0xA5);
        auto* base =
            reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(allocation.data()) + 511) & ~uintptr_t{511});
        WorkspaceInfo hybrid(base, &t);
        assert(hybrid.workspaceSize <= 4 * 1024 * 1024);
        assert(hybrid.gmm2ReadyPtr != nullptr);
        assert(hybrid.gmm2MmadResPtr != nullptr);
        assert(reinterpret_cast<uintptr_t>(hybrid.gmm2ReadyPtr) % 64 == 0);
        assert(hybrid.gmm2ReadyPtr >= hybrid.flagActivationToGmm2Ptr);
        auto* flagsEnd = hybrid.flagActivationToGmm2Ptr + hybrid.flagResetElementCount * sizeof(int32_t);
        const int64_t readyBytes = static_cast<int64_t>(t.moeExpertPerRank) * groups * 64;
        auto* priorFlagsEnd = hybrid.gmm2ReadyPtr + readyBytes;
        assert(priorFlagsEnd == flagsEnd);
        assert(flagsEnd <= hybrid.gmm2MmadResPtr);
        // The reset span must include the final expert's final M-group, while
        // leaving the subsequent data/workspace byte untouched.
        std::fill(hybrid.flagActivationToGmm2Ptr, flagsEnd, uint8_t{0});
        assert(hybrid.gmm2ReadyPtr[readyBytes - 1] == 0);
        assert(flagsEnd[0] == 0xA5);

        t.topoType = TOPO_TYPE_MTE;
        WorkspaceInfo legacy(base, &t);
        const int64_t priorElements = (priorFlagsEnd - hybrid.flagActivationToGmm2Ptr) / sizeof(int32_t);
        if (mode == GROUPED_MATMUL_MODE_GENERAL) {
            assert(legacy.gmm2ReadyPtr == nullptr);
            assert(priorElements - legacy.flagResetElementCount == readyBytes / 4);
        } else {
            assert(legacy.gmm2ReadyPtr == hybrid.gmm2ReadyPtr);
            assert(legacy.flagResetElementCount == priorElements);
        }
        assert(hybrid.flagResetElementCount == priorElements);
    }
}

void check_packet_gateway_layout(uint8_t mode)
{
    auto largest = make_tiling(64U, 8192U, 8U, false, mode);
    std::vector<uint8_t> heap(CalcPeermemLeastSize(parameters(largest)) + 511U, 0xA5);
    auto* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(heap.data()) + 511U) & ~uintptr_t{511});
    for (uint32_t hidden = 1024U; hidden <= 8192U; hidden += 32U) {
        {
            auto t = make_tiling(64U, hidden, 8U, false, mode);
            const auto bytes = CalcPeermemLeastSize(parameters(t));
            assert(base + bytes <= heap.data() + heap.size());
            PeermemInfo info(base, &t);
            const uint64_t rowBytes = (hidden * 2U + 507U) / 508U * 512U;
            const uint64_t rows = 64U * 8U;
            assert(UsePacketizedGatewayCombine(&t));
            const uint64_t inputRowBytes = rowBytes;
            const uint64_t sumBytes = (hidden * 4U + 255U) / 256U * 256U;
            const uint64_t readyBytes = 8U * (rowBytes / 512U) * 32U + 32U;
            assert(CalcGatewayCombineInputUbRowBytes(&t) == static_cast<int64_t>(inputRowBytes));
            assert(CalcGatewayCombineSumUbBytes(&t) == static_cast<int64_t>(sumBytes));
            assert(CalcGatewayCombineReadyUbBytes(&t) == static_cast<int64_t>(readyBytes));
            for (uint32_t inputBuffers : {2U}) {
                const uint64_t reduceBytes = inputRowBytes * inputBuffers + sumBytes + 32U;
                assert(CalcGatewayCombineReduceUbBytes(&t) ==
                       static_cast<int64_t>(reduceBytes > readyBytes ? reduceBytes : readyBytes));
            }

            assert(info.hierarchicalControlPtr == info.gatewayCombinePtr + rows * rowBytes);
            for (uint64_t row = 0; row < rows; ++row)
                assert(reinterpret_cast<uintptr_t>(info.gatewayCombinePtr + row * rowBytes) % 512U == 0U);

            auto* end = info.combineReduceCommitPtr + HIERARCHICAL_COMBINE_QPS * 512U;
            assert(end == base + bytes);
            for (uint32_t bs : {1U, 32U, 64U}) {
                t.bs = bs;
                PeermemInfo next(base, &t);
                assert(CalcPeermemLeastSize(parameters(t)) == bytes);
                assert(next.gatewayCombinePtr == info.gatewayCombinePtr);
                assert(next.hierarchicalControlPtr == info.hierarchicalControlPtr);
                assert(next.combineReduceCommitPtr == info.combineReduceCommitPtr);
            }
        }
    }
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string_view selected(argv[1]);
    assert(selected == "a8w4" || selected == "a8w8");
    const uint8_t mode = selected == "a8w4" ? GROUPED_MATMUL_MODE_A8W4 : GROUPED_MATMUL_MODE_GENERAL;
    assert(HIERARCHICAL_MAX_TOKENS == 256);
    for (uint32_t capacity : {1U, 17U, 31U, 32U, 64U}) {
        for (uint32_t h : {1024U, 7168U, 8192U}) {
            for (uint32_t topk : {1U, 6U, 32U}) {
                for (bool prefetch : {false, true}) check_hybrid(capacity, h, topk, prefetch, mode);
            }
        }
    }
    check_hybrid(4096, 1024, 1, true, mode);
    check_hybrid(256, 3584, 8, false, mode);
    check_hybrid(256, 1024, 32, true, mode);
    check_hybrid(144, 5120, 6, false, mode);
    check_hybrid(144, 1024, 32, true, mode);
    check_workspace_ready_region(mode);
    check_packet_gateway_layout(mode);
    auto legacy = make_tiling(31, 1024, 6, true, mode);
    legacy.topoType = TOPO_TYPE_MTE;
    legacy.dispatchRanksPerNode = 8;
    checkPlanDomains();
    check_combine_owner_policy();
    assert(RailServerSize(&legacy) == 8);
    assert(CalcServerNum(32, RailServerSize(&legacy)) == 4);
    // Frozen EP32 legacy layout: catches accidental size/offset ABI changes.
    assert(CalcPeermemLeastSize(parameters(legacy)) == (mode == GROUPED_MATMUL_MODE_A8W4 ? 531456 : 526848));
    const auto legacy_size = CalcPeermemLeastSize(parameters(legacy));
    legacy.dispatchRanksPerNode = 0;
    assert(CalcPeermemLeastSize(parameters(legacy)) == legacy_size);
    legacy.epWorldSize = 16;
    legacy.dispatchRanksPerNode = 16;
    assert(!UseHierarchicalRoce(&legacy));
    assert(CalcRailTopkIdsRecvSize(&legacy) == CalcRailTopkIdsSlotBytes(&legacy));
    for (uint32_t rank = 0; rank < 32; ++rank) {
        assert(HierarchicalGateway(HierarchicalGateway(rank)) == rank);
        assert(HierarchicalGateway(rank) % 16 == rank % 16);
        assert(HierarchicalHost(HierarchicalGateway(rank)) == 1U - HierarchicalHost(rank));
    }
    assert(HierarchicalPackJob() == 0U);
    assert(HierarchicalSendJob() == 1U);
    // Wire shape is unchanged, but hierarchical metadata must retain every
    // sequence bit so a stale variable-length tail cannot match after 7-bit wrap.
    auto markerTiling = make_tiling(32, 3584, 8, true, mode);
    markerTiling.topoType = TOPO_TYPE_MTE_ROCE;
    for (uint32_t sequence : {1U, 129U, 257U, 0x80000101U, 0xffffffffU}) {
        markerTiling.hierarchicalSequence = sequence;
        for (uint32_t items : {0U, 1U, 127U, 128U, 256U, 1024U}) {
            assert(static_cast<uint32_t>(RailPacketMarker(&markerTiling, 0x81U, items)) == sequence);
        }
    }
    markerTiling.topoType = TOPO_TYPE_MTE;
    assert(static_cast<uint32_t>(RailPacketMarker(&markerTiling, 0x81U, 256U)) == 0x81000100U);
    assert(CalcRailTopkIdsSlotDataBytes(32U, 8U) == 1536);
    assert(CalcRailTopkIdsSlotDataBytes(0U, 8U) == 512);
}
