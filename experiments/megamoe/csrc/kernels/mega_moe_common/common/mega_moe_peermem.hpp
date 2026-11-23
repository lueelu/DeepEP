/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_peermem.h
 * \brief peermem 跨卡对称窗口的布局与尺寸计算。
 *        窗口各区容量按全卡一致的 numMaxTokensPerRank 上界推导。公开 API 要求各 rank 同 BS、
 *        同模式调用；decode/prefill 协议选择也必须一致，跨卡读写共用逐字节一致的偏移。
 *        本文件是 host 侧 tiling 校验与 device 侧地址装配的唯一事实源，两侧不得各写一份公式。
 */

#ifndef MEGA_MOE_PEERMEM_H
#define MEGA_MOE_PEERMEM_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include "kernel_operator.h"
#include "op_common/op_kernel/math_util.h"
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t*
// host 侧命名空间级自由函数需显式 inline 保 ODR；类内成员函数本就隐式 inline，加之无副作用。
#define HOST_DEVICE inline
// host 侧自洽：Ops::Base::CeilAlign/CeilDiv 的提供方，不再依赖包含方的 include 顺序。
#include "op_common/op_host/util/math_util.h"
#endif
#include "kernels/mega_moe_common/mega_moe_tiling.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#ifdef DEEPEP_MEGAMOE_ROCE
#include "../../../../../../csrc/common/megamoe_combine_config.hpp"
#endif

namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE {

// 窗口首部的跨卡同步区（rankSyncInWorld）字节数，数据区从此偏移起。
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
#ifdef DEEPEP_MEGAMOE_ROCE
constexpr int64_t TOPO_TYPE_MTE = 0U;       // mte
constexpr int64_t TOPO_TYPE_URMA = 1U;      // urma
constexpr int64_t TOPO_TYPE_MTE_ROCE = 2U;  // local MTE, paired cross-host RoCE

HOST_DEVICE bool UsesMteLayout(int64_t topology)
{
    return topology == TOPO_TYPE_MTE || topology == TOPO_TYPE_MTE_ROCE;
}

HOST_DEVICE bool UseHierarchicalRoce(const MegaMoeTilingData* t)
{
    return t->topoType == TOPO_TYPE_MTE_ROCE;
}

HOST_DEVICE bool UseHierarchicalCombineReduction(const MegaMoeTilingData* t)
{
    return UseHierarchicalRoce(t);
}

constexpr uint32_t HIERARCHICAL_COMBINE_QPS = deepep::megamoe_config::COMBINE_QPS;
constexpr uint32_t HIERARCHICAL_DISPATCH_QPS = deepep::megamoe_config::DISPATCH_QPS;
static_assert(HIERARCHICAL_DISPATCH_QPS <= HIERARCHICAL_COMBINE_QPS);
constexpr uint32_t HIERARCHICAL_DISPATCH_WORKERS = deepep::megamoe_config::DISPATCH_WORKERS;
constexpr uint32_t HIERARCHICAL_RANKS_PER_HOST = 16U;
constexpr uint32_t HIERARCHICAL_WORLD_SIZE = 32U;
// Covers the DSV41 sweep (original BS 1..24, six tokens per sequence).
// Per-token Combine state has one uint32 top-k mask, not one bit per token.
constexpr uint32_t HIERARCHICAL_MAX_TOKENS = 256U;
constexpr uint32_t DISPATCH_PROBE_ROWS = 8U;  // One delivery bitmap byte per window.
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512U;
constexpr uint32_t CLOSURE_SLOT_COUNT = 64U + 2U * HIERARCHICAL_RANKS_PER_HOST + 2U;
constexpr uint32_t BF16_GMM1_READY_SLOT = CLOSURE_SLOT_COUNT;
constexpr uint32_t PLAN_FLAG_SLOTS = BF16_GMM1_READY_SLOT + 28U;
// The QP backend uses the first 256B; startup consumers reuse this reservation
// for a single strided snapshot of sixteen 32B flag heads (512B total).
constexpr uint32_t HIERARCHICAL_ROCE_UB_BYTES = HIERARCHICAL_RANKS_PER_HOST * 32U;

// Every control flag below owns a full L2 line. Embedded rail markers instead
// travel with their packet's data under a single writer; see RailPacketMarker.
// Distinct flag producers must never share a line.
enum class HierarchicalSignal : uint32_t {
    Sequence,
    LocalRailPacked,
    NetMetaCommit,
    MetaReady,
    NetPayloadCommit,
    NetPayloadCommitLast = NetPayloadCommit + HIERARCHICAL_DISPATCH_QPS - 1U,
    RelayPayloadReady,
    Count
};

constexpr uint32_t DISPATCH_PACKET_DATA = 480U;
HOST_DEVICE uint32_t DispatchPacketCount(uint32_t bytes)
{
    return (bytes + DISPATCH_PACKET_DATA - 1U) / DISPATCH_PACKET_DATA;
}
HOST_DEVICE bool UseRouteAllGather(const MegaMoeTilingData* t)
{
    const bool common = (t->bs == 32U || t->bs == 48U || t->bs == 64U || t->bs == 72U) && t->h == 3584U &&
                        t->topK == 8U && t->moeExpertPerRank == 12U;
    const bool dsv = (t->bs == 96U || t->bs == 128U) && t->h == 5120U && t->topK == 6U && t->moeExpertPerRank == 13U &&
                     t->sharedExpertNum <= 1U;
    return UseHierarchicalRoce(t) && t->blockAivNum >= 8U && (common || dsv || t->bf16Bs96);
}

HOST_DEVICE uint32_t NextDomainSequence(uint32_t previous, uint32_t domain, uint32_t bits)
{
    if (previous > (0x7fffffffU >> bits) - 2U) return 0U;
    return ((previous + 1U) << bits) | domain;
}

HOST_DEVICE uint32_t RouteGatherSlot(uint32_t rank)
{
    return rank / 16U * 16U + rank % 4U * 4U + rank % 16U / 4U;
}

HOST_DEVICE bool UseSharedRouteTable(const MegaMoeTilingData* t)
{
    return UseRouteAllGather(t) && t->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 &&
           t->combineQuantMode == COMBINE_NO_QUANT && (t->sharedExpertNum == 0U || t->h == 5120U);
}

HOST_DEVICE uint32_t HierarchicalHost(uint32_t rank)
{
    return rank / HIERARCHICAL_RANKS_PER_HOST;
}

HOST_DEVICE uint32_t HierarchicalGateway(uint32_t source)
{
    return source ^ HIERARCHICAL_RANKS_PER_HOST;
}

// Physical AIV roles are identical on both hosts, independent of host/gateway IDs.
HOST_DEVICE uint32_t HierarchicalPackJob()
{
    return 0U;
}

HOST_DEVICE uint32_t HierarchicalSendJob()
{
    return 1U;
}

// Input QP1..8 belong to g0..g7 AIV1 until the all-core input barrier.
HOST_DEVICE uint32_t HierarchicalDispatchLane(uint32_t physicalJob)
{
    const uint32_t lane = physicalJob / 2U;
    return (physicalJob & 1U) != 0U && lane < HIERARCHICAL_DISPATCH_QPS ? lane : HIERARCHICAL_DISPATCH_QPS;
}

HOST_DEVICE uint32_t HierarchicalPayloadTokenBegin(uint32_t tokens, uint32_t lane)
{
    return tokens * lane / HIERARCHICAL_DISPATCH_QPS;
}

HOST_DEVICE uint32_t HierarchicalCombineOwnerCount(const MegaMoeTilingData* t)
{
    const uint32_t lanes = t->bf16Bs96 ? HIERARCHICAL_COMBINE_QPS : 16U;
    return t->aicNum != 0U && t->aicNum < lanes ? t->aicNum : lanes;
}

HOST_DEVICE bool IsDsv96CombineShape(const MegaMoeTilingData* t)
{
    return t->bs == 96U && t->h == 5120U && t->hiddenDim == 4608U &&
           (t->moeExpertPerRank == 12U || t->moeExpertPerRank == 13U) &&
           t->topK == 6U && t->sharedExpertNum == 1U &&
           t->dispatchRanksPerNode == HIERARCHICAL_RANKS_PER_HOST &&
           t->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 && t->combineQuantMode == COMBINE_NO_QUANT &&
           t->topkWeightsPrefetch == 0 && t->epWorldSize == 32U && t->aicNum >= 16U && t->aicNum <= 32U &&
           t->blockAivNum == 2U * t->aicNum;
}

HOST_DEVICE bool UseAiv0CombineOwners(const MegaMoeTilingData* t)
{
    return UseHierarchicalRoce(t) && !t->bf16Bs96 &&
           !(t->moeExpertPerRank == 13U && t->sharedFp8 && IsDsv96CombineShape(t));
}

HOST_DEVICE bool SupportsBf16Bs96(const MegaMoeTilingData* t)
{
    return UseHierarchicalRoce(t) && IsDsv96CombineShape(t) && t->aicNum == 28U;
}

HOST_DEVICE bool UseBf16Bs96(const MegaMoeTilingData* t, bool prequantized)
{
    return !prequantized && t->sharedFp8 && SupportsBf16Bs96(t);
}

// Each active Combine QP has one physical owner. The input barrier closes Dispatch's use of QP1..8.
HOST_DEVICE uint32_t HierarchicalCombineLane(const MegaMoeTilingData* t, uint32_t physicalJob)
{
    const uint32_t count = HierarchicalCombineOwnerCount(t);
    const uint32_t lane = physicalJob / 2U;
    return (physicalJob & 1U) == (UseAiv0CombineOwners(t) ? 0U : 1U) && lane < count ? lane : count;
}

HOST_DEVICE uint32_t CompactOwner(const MegaMoeTilingData* t, uint32_t index, uint32_t count)
{
    const uint32_t qps = HierarchicalCombineOwnerCount(t);
    return UseAiv0CombineOwners(t) ? ((index + 1U) * qps - 1U) / count : index % qps;
}
HOST_DEVICE bool UseEarlyClosure(const MegaMoeTilingData* t)
{
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
    return UseSharedRouteTable(t) && t->epWorldSize == 32 && t->aicNum >= 16 && t->aicNum <= 32 &&
           t->blockAivNum == 2 * t->aicNum && !t->topkWeightsPrefetch &&
           ((t->bs == 32 && t->hiddenDim == 2560 && !t->sharedExpertNum) ||
            ((t->bs == 96 || (t->bs == 128 && t->aicNum == 28)) && t->hiddenDim == 4608 && t->sharedExpertNum == 1U &&
             t->sharedFp8));
#else
    (void)t;
    return false;
#endif
}

HOST_DEVICE bool UseInputEarlyClosure(const MegaMoeTilingData* t, bool hasInputScales)
{
    return (hasInputScales || t->bf16Bs96) && UseEarlyClosure(t);
}

HOST_DEVICE bool IsHierarchicalCombineOwner(const MegaMoeTilingData* t, uint32_t physicalJob)
{
    return UseHierarchicalRoce(t) && HierarchicalCombineLane(t, physicalJob) < HierarchicalCombineOwnerCount(t);
}

// Reserve the physical QP owner for completion after the output-ready barrier.
// Host buffer sizing and device token partitioning must use the same job count.
HOST_DEVICE uint32_t UnpermuteJobCount(const MegaMoeTilingData* t)
{
    return UseHierarchicalRoce(t) && t->blockAivNum > HierarchicalSendJob()
               ? t->blockAivNum - HierarchicalCombineOwnerCount(t)
               : t->blockAivNum;
}

HOST_DEVICE uint32_t UnpermuteJobIndex(const MegaMoeTilingData* t, uint32_t physicalJob)
{
    if (UnpermuteJobCount(t) == t->blockAivNum) {
        return physicalJob;
    }
    if (IsHierarchicalCombineOwner(t, physicalJob)) {
        return UnpermuteJobCount(t);  // Invalid logical job: owns no output rows.
    }
    // For a nonowner this counts preceding owners for either even or odd
    // physical jobs; owner jobs themselves returned above.
    const uint32_t before = (physicalJob + (UseAiv0CombineOwners(t) ? 1U : 0U)) / 2U;
    const uint32_t count = HierarchicalCombineOwnerCount(t);
    return physicalJob - (before < count ? before : count);
}

HOST_DEVICE uint64_t HierarchicalSignalOffset(HierarchicalSignal signal)
{
    return static_cast<uint64_t>(signal) * HIERARCHICAL_FLAG_BYTES;
}
#else
constexpr int64_t TOPO_TYPE_MTE = 0U;   // mte
constexpr int64_t TOPO_TYPE_URMA = 1U;  // urma
#endif

// ============================ 各区尺寸（host / device 共用） ============================

// URMA layered 路径仍使用 mask 槽；MTE WAVE 不再经过该布局。
HOST_DEVICE int64_t CalcDispatchMaskAlignSizeBy(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    int64_t alignedRouteCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return Ops::Base::CeilAlign(alignedRouteCount / 8, ALIGN_32);
}

HOST_DEVICE int64_t CalcDispatchMaskAlignSize(const MegaMoeTilingData* tilingData)
{
    return CalcDispatchMaskAlignSizeBy(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                       static_cast<int64_t>(tilingData->topK));
}

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
HOST_DEVICE int64_t CalcDispatchRouteIndexAlignSize(const MegaMoeTilingData* tilingData)
{
    return Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData->numMaxTokensPerRank) * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
}

// compact route 接收区不再携带槽尾 count；count 继续使用下方独立连续表。
HOST_DEVICE int64_t CalcRouteIndexRecvSize(int64_t routeIndexAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * routeIndexAlignSize, ALIGN_512);
}

HOST_DEVICE int64_t CalcMaskRecvSize(int64_t maskAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    int64_t maskSlotSize = maskAlignSize + ALIGN_32;
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, ALIGN_512);
}

// 独立 raw count 表按 [localExpert][sourceRank] 排列，供接收端一次搬入并做前缀和。
HOST_DEVICE int64_t CalcExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);
}

// MTE count 按 [sourceRank][localExpert] 排列，每个 source 段 32B 对齐。
// 同一 owner 发给同一目标卡的 count 连续写入，不再逐专家跨步写 4B。
HOST_DEVICE int64_t CalcMteExpertCountRankStride(int64_t moeExpertPerRank)
{
    return Ops::Base::CeilAlign(moeExpertPerRank, static_cast<int64_t>(ALIGN_32 / sizeof(int32_t)));
}

HOST_DEVICE int64_t CalcMteExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(
        CalcMteExpertCountRankStride(moeExpertPerRank) * epWorldSize * static_cast<int64_t>(sizeof(int32_t)),
        ALIGN_512);
}

// ============================ Rail topkIds 中转区（MTE 路径） ============================
// 每服务器卡数。rank r 的 rail rank 在目标 server u 上为 u * SERVER_SIZE + r % SERVER_SIZE。
constexpr int64_t MEGA_MOE_SERVER_SIZE = 8;
#ifdef DEEPEP_MEGAMOE_ROCE

HOST_DEVICE int64_t RailServerSize(uint32_t ranksPerNode)
{
    return ranksPerNode == HIERARCHICAL_RANKS_PER_HOST ? HIERARCHICAL_RANKS_PER_HOST : MEGA_MOE_SERVER_SIZE;
}

HOST_DEVICE int64_t RailServerSize(const MegaMoeTilingData* t)
{
    return RailServerSize(t->dispatchRanksPerNode);
}
#endif
// Platform contract: an aligned 512B peer-write packet becomes visible as a unit.
// Reserve the last int32 of EVERY packet for (epoch << 24) | totalItems.
// Slot and packet starts must remain 512B aligned; checking only the last packet is unsafe.
constexpr uint32_t RAIL_PACKET_BYTES = 512U;
constexpr uint32_t RAIL_PACKET_WORDS = RAIL_PACKET_BYTES / sizeof(int32_t);
constexpr uint32_t RAIL_PACKET_DATA_WORDS = RAIL_PACKET_WORDS - 1U;

#ifdef DEEPEP_MEGAMOE_ROCE
// QP8 gateway input carries an invocation marker in every aligned 512B packet.
HOST_DEVICE bool UsePacketizedGatewayCombine(const MegaMoeTilingData* t)
{
    return UseHierarchicalRoce(t);
}

HOST_DEVICE int64_t CalcGatewayCombinePacketRowBytes(int64_t hidden)
{
    return Ops::Base::CeilDiv(hidden * 2, static_cast<int64_t>(RAIL_PACKET_DATA_WORDS * sizeof(uint32_t))) *
           RAIL_PACKET_BYTES;
}

HOST_DEVICE int64_t CalcGatewayCombineBytes(int64_t tokens, int64_t topK, int64_t hidden)
{
    return tokens * topK * CalcGatewayCombinePacketRowBytes(hidden);
}

HOST_DEVICE int64_t CalcGatewayCombineInputUbRowBytes(const MegaMoeTilingData* t)
{
    // The VF loads 64 BF16 values at a time, including the masked final chunk.
    return CalcGatewayCombinePacketRowBytes(t->h);
}

HOST_DEVICE int64_t CalcGatewayCombineSumUbBytes(const MegaMoeTilingData* t)
{
    return Ops::Base::CeilAlign(static_cast<int64_t>(t->h) * 4, int64_t{256});
}

HOST_DEVICE int64_t CalcGatewayCombineReadyUbBytes(const MegaMoeTilingData* t)
{
    // Sparse snapshots keep the original top-k row positions. The last 32B
    // hold one ready mask, read by Scalar only after the VF has finished.
    return t->topK * (CalcGatewayCombinePacketRowBytes(t->h) / RAIL_PACKET_BYTES) * ALIGN_32 + ALIGN_32;
}

HOST_DEVICE int64_t CalcGatewayCombineReduceUbBytes(const MegaMoeTilingData* t)
{
    const int64_t reduceBytes = CalcGatewayCombineInputUbRowBytes(t) * 2U + CalcGatewayCombineSumUbBytes(t) +
                                Ops::Base::CeilAlign(static_cast<int64_t>(t->topK) * 4, ALIGN_32);
    const int64_t readyBytes = CalcGatewayCombineReadyUbBytes(t);
    // Readiness and reduction run sequentially in the same drained main UB.
    return reduceBytes > readyBytes ? reduceBytes : readyBytes;
}

HOST_DEVICE int32_t RailPacketMarker(const MegaMoeTilingData* tilingData, uint32_t epoch, uint32_t totalItems)
{
    return static_cast<int32_t>(UseHierarchicalRoce(tilingData) ? tilingData->hierarchicalSequence
                                                                : (epoch << 24) | totalItems);
}

HOST_DEVICE int64_t CalcServerNum(int64_t epWorldSize, int64_t ranksPerServer = MEGA_MOE_SERVER_SIZE)
{
    return Ops::Base::CeilDiv(epWorldSize, ranksPerServer);
#else
HOST_DEVICE int64_t CalcServerNum(int64_t epWorldSize)
{
    return Ops::Base::CeilDiv(epWorldSize, MEGA_MOE_SERVER_SIZE);
#endif
}

// 每个 512B 包含 127 个 topkId 和 1 个标记，空输入也发送一个带标记的包。
HOST_DEVICE int64_t CalcRailTopkIdsSlotDataBytes(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t items = numMaxTokensPerRank * topK;
    int64_t packets =
        Ops::Base::CeilDiv(items > 0 ? items : static_cast<int64_t>(1), static_cast<int64_t>(RAIL_PACKET_DATA_WORDS));
    return packets * RAIL_PACKET_BYTES;
}

#ifdef DEEPEP_MEGAMOE_ROCE
HOST_DEVICE int64_t CalcCombineWeightsBytes(int64_t tokens, int64_t topK)
{
    return Ops::Base::CeilAlign(tokens * topK * static_cast<int64_t>(sizeof(float)), ALIGN_512);
}

// Only the active prefix is transmitted. Capacity fixes the slot stride, while
// actual B fixes the route offset inside each slot, identically on all ranks.
HOST_DEVICE int64_t HierarchicalMetadataPrefixBytes(const MegaMoeTilingData* t)
{
    return UseHierarchicalCombineReduction(t) ? CalcCombineWeightsBytes(t->bs, t->topK) : 0;
}

#endif
// 不再追加独立 arrival 区：每个包的标记随该包的数据一起发送。
HOST_DEVICE int64_t CalcRailTopkIdsSlotBytes(const MegaMoeTilingData* tilingData)
{
#ifdef DEEPEP_MEGAMOE_ROCE
    return CalcRailTopkIdsSlotDataBytes(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                        static_cast<int64_t>(tilingData->topK)) +
           (UseHierarchicalCombineReduction(tilingData)
                ? CalcCombineWeightsBytes(tilingData->numMaxTokensPerRank, tilingData->topK)
                : 0);
#else
    return CalcRailTopkIdsSlotDataBytes(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                        static_cast<int64_t>(tilingData->topK));
#endif
}

// 每卡接收来自各 server 同号卡的 topkIds，槽数 = serverNum（含本 server 自己那份，路径统一）。
HOST_DEVICE int64_t CalcRailTopkIdsRecvSize(const MegaMoeTilingData* tilingData)
{
#ifdef DEEPEP_MEGAMOE_ROCE
    return Ops::Base::CeilAlign(
        CalcServerNum(static_cast<int64_t>(tilingData->epWorldSize), RailServerSize(tilingData)) *
            CalcRailTopkIdsSlotBytes(tilingData),
        ALIGN_512);
#else
    return Ops::Base::CeilAlign(
        CalcServerNum(static_cast<int64_t>(tilingData->epWorldSize)) * CalcRailTopkIdsSlotBytes(tilingData), ALIGN_512);
#endif
}

// Rail grouping follows the donor's eight-rank servers. A partial final server
// falls back to direct routing; single-server EP2..8 is also supported.
HOST_DEVICE bool UseDecodeRailRouting(const MegaMoeTilingData* t)
{
#ifdef DEEPEP_MEGAMOE_ROCE
    const int64_t ranksPerServer = RailServerSize(t);
    const bool decode = UseHierarchicalRoce(t)
                            ? t->bs <= HIERARCHICAL_MAX_TOKENS
                            : UseDecodeInputReady(t->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4, t->bs);
    return UsesMteLayout(t->topoType) && decode &&
           (t->epWorldSize <= ranksPerServer || t->epWorldSize % ranksPerServer == 0) &&
           t->blockAivNum > CalcServerNum(t->epWorldSize, ranksPerServer);
#else
    return t->topoType == TOPO_TYPE_MTE &&
           UseDecodeInputReady(t->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4, t->bs) &&
           (t->epWorldSize <= MEGA_MOE_SERVER_SIZE || t->epWorldSize % MEGA_MOE_SERVER_SIZE == 0) &&
           t->blockAivNum > CalcServerNum(t->epWorldSize);
#endif
}

// Host buffer budgeting and device UB binding must use the same relay bound.
HOST_DEVICE uint32_t CalcSendMaskExpertCapacity(const MegaMoeTilingData* t, uint32_t routeCores)
{
    uint32_t capacity = Ops::Base::CeilDiv(t->epWorldSize * t->moeExpertPerRank, routeCores);
    if (UseDecodeRailRouting(t)) {
#ifdef DEEPEP_MEGAMOE_ROCE
        const uint32_t ranksPerServer = static_cast<uint32_t>(RailServerSize(t));
        uint32_t serverRanks = t->epWorldSize < ranksPerServer ? t->epWorldSize : ranksPerServer;
#else
        uint32_t serverRanks = t->epWorldSize < MEGA_MOE_SERVER_SIZE ? t->epWorldSize : MEGA_MOE_SERVER_SIZE;
#endif
        uint32_t serverExperts = serverRanks * t->moeExpertPerRank;
        capacity = capacity > serverExperts ? capacity : serverExperts;
    }
    return capacity;
}

// 单 token 的量化记录字节数 = 量化数据 + mx scale（prefetch 场景再拼 topk 权重），按 32B 对齐。
HOST_DEVICE int64_t CalcQuantTokenScaleBytes(int64_t h, uint32_t elemsPerByte, int64_t topK, bool topkWeightsPrefetch)
{
    uint32_t mxScaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(h), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(h) / elemsPerByte, static_cast<uint32_t>(ALIGN_256)) *
        static_cast<uint32_t>(sizeof(int8_t));
    uint32_t scaleBytes = mxScaleNum * static_cast<uint32_t>(sizeof(int8_t));
    uint32_t tokenScaleBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
    if (topkWeightsPrefetch) {
        uint32_t weightBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)),
                                                    static_cast<uint32_t>(ALIGN_32));
        tokenScaleBytes = Ops::Base::CeilAlign(tokenScaleBytes + weightBytes, static_cast<uint32_t>(ALIGN_32));
    }
    return static_cast<int64_t>(tokenScaleBytes);
}

// combine 接收区单 token 记录字节数：须与 kernel InitCombineBuffers 的 combine record 布局一致。
HOST_DEVICE int64_t CalcCombineTokenBytes(int64_t h, int64_t yDtypeSize, bool isQuantCombine)
{
    if (!isQuantCombine) {
        return h * yDtypeSize;
    }
    int64_t tokenStorageBytes = Ops::Base::CeilAlign(h, ALIGN_256);
    int64_t scaleCount = Ops::Base::CeilDiv(h, static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
    int64_t storedScaleBytes = Ops::Base::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
    return Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, ALIGN_32);
}

// 计算窗口总大小的入参（供 host 侧 tiling 校验 cclBufferSize 使用）。
struct PeermemSizeParams {
    int64_t numMaxTokensPerRank;
    int64_t topK;
    int64_t h;
    int64_t moeExpertPerRank;
    int64_t epWorldSize;
    int64_t yDtypeSize;
    uint32_t elemsPerByte;
    bool topkWeightsPrefetch;
    bool isQuantCombine;
    int64_t topoType;
    bool reserveDecodeRail = false;
#ifdef DEEPEP_MEGAMOE_ROCE
    uint32_t ranksPerNode = 0U;
#endif
};

/*
 * MTE 路径下 peermem 窗口所需的最小字节数，逐段与下方 PeermemInfo 构造函数的偏移推进同源：
 *   同步区(PEERMEM_DATA_OFFSET) + mask 接收区 + count 接收区 + 量化 token 接收区 + combine 接收区。
 * host 侧 tiling 用它校验用户传入的 cclBufferSize，不再各自手写一份布局公式。
 */
HOST_DEVICE int64_t CalcPeermemLeastSize(const PeermemSizeParams& params)
{
    int64_t routeRecvSize = 0;
#ifdef DEEPEP_MEGAMOE_ROCE
    if (UsesMteLayout(params.topoType)) {
#else
    if (params.topoType == TOPO_TYPE_MTE) {
#endif
        int64_t routeIndexAlignSize =
            Ops::Base::CeilAlign(params.numMaxTokensPerRank * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
        routeRecvSize = CalcRouteIndexRecvSize(routeIndexAlignSize, params.moeExpertPerRank, params.epWorldSize);
    } else {
        int64_t maskAlignSize = CalcDispatchMaskAlignSizeBy(params.numMaxTokensPerRank, params.topK);
        routeRecvSize = CalcMaskRecvSize(maskAlignSize, params.moeExpertPerRank, params.epWorldSize);
    }
#ifdef DEEPEP_MEGAMOE_ROCE
    const bool rail = UsesMteLayout(params.topoType) && params.reserveDecodeRail;
#else
    const bool rail = params.topoType == TOPO_TYPE_MTE && params.reserveDecodeRail;
#endif
    int64_t expertCountRecvSize = rail ? CalcMteExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize)
                                       : CalcExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize);
#ifdef DEEPEP_MEGAMOE_ROCE
    int64_t railBytes = rail ? CalcServerNum(params.epWorldSize, RailServerSize(params.ranksPerNode)) *
                                   (CalcRailTopkIdsSlotDataBytes(params.numMaxTokensPerRank, params.topK) +
                                    (params.topoType == TOPO_TYPE_MTE_ROCE
                                         ? CalcCombineWeightsBytes(params.numMaxTokensPerRank, params.topK)
                                         : 0))
                             : 0;
#else
    int64_t railBytes =
        rail ? CalcServerNum(params.epWorldSize) * CalcRailTopkIdsSlotDataBytes(params.numMaxTokensPerRank, params.topK)
             : 0;
#endif

    int64_t tokenScaleBytes =
        CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK, params.topkWeightsPrefetch);
    int64_t quantTokenScaleSize = Ops::Base::CeilAlign(params.numMaxTokensPerRank * tokenScaleBytes, ALIGN_512);
    int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    int64_t combineSendSize =
        Ops::Base::CeilAlign(params.numMaxTokensPerRank * params.topK * combineTokenBytes, ALIGN_512);
#ifdef DEEPEP_MEGAMOE_ROCE
    int64_t bytes =
        PEERMEM_DATA_OFFSET + routeRecvSize + expertCountRecvSize + railBytes + quantTokenScaleSize + combineSendSize;
    if (params.topoType == TOPO_TYPE_MTE_ROCE) {
        bytes += quantTokenScaleSize + CalcGatewayCombineBytes(params.numMaxTokensPerRank, params.topK, params.h);
        bytes += static_cast<int64_t>(HierarchicalSignal::Count) * HIERARCHICAL_FLAG_BYTES;
        bytes += HIERARCHICAL_RANKS_PER_HOST * HIERARCHICAL_FLAG_BYTES;

        const int64_t partialBytes = Ops::Base::CeilAlign(
            params.numMaxTokensPerRank * params.h * static_cast<int64_t>(sizeof(float)), ALIGN_512);
        bytes += 2 * partialBytes + HIERARCHICAL_COMBINE_QPS * HIERARCHICAL_FLAG_BYTES;
    }
    return bytes;
#else
    return PEERMEM_DATA_OFFSET + routeRecvSize + expertCountRecvSize + railBytes + quantTokenScaleSize +
           combineSendSize;
#endif
}

#ifdef DEEPEP_MEGAMOE_ROCE
// Shared host/device layout; offsets follow CalcPeermemLeastSize above.
struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR maskRecvPtr;
    GM_ADDR expertCountRecvPtr;
    GM_ADDR railTopkIdsRecvPtr{nullptr};
    GM_ADDR quantTokenScalePtr;
    GM_ADDR dispatchRecivePtr;
    GM_ADDR dispatchFlagPtr;
    GM_ADDR combineSendPtr;
    GM_ADDR relayPayloadPtr{nullptr};
    GM_ADDR gatewayCombinePtr{nullptr};
    GM_ADDR hierarchicalControlPtr{nullptr};
    GM_ADDR gatewayPayloadReadyPtr{nullptr};
    GM_ADDR combinePartialSendPtr{nullptr};
    GM_ADDR combinePartialRecvPtr{nullptr};
    GM_ADDR combineReduceCommitPtr{nullptr};
    GM_ADDR dispatchPacketPtr{nullptr};
    GM_ADDR compactPtr{nullptr};
    GM_ADDR gatherPtr{nullptr};
    GM_ADDR closurePtr{nullptr};

    HOST_DEVICE PeermemInfo() = default;
    HOST_DEVICE PeermemInfo(GM_ADDR base, const MegaMoeTilingData* tilingData, uint32_t elemsPerByte = 1,
                            uint32_t serverNum = 1)
    {
        rankSyncInWorldPtr = base;
        if (tilingData->arena.route != 0) {
            maskRecvPtr = base + tilingData->arena.route;
            expertCountRecvPtr = base + tilingData->arena.counts;
            railTopkIdsRecvPtr = base + tilingData->arena.rail + HierarchicalMetadataPrefixBytes(tilingData);
            quantTokenScalePtr = base + tilingData->arena.quant;
            combineSendPtr = base + tilingData->arena.combine;
            relayPayloadPtr = base + tilingData->arena.relay;
            dispatchPacketPtr = base + tilingData->arena.packetSend;
            gatewayCombinePtr = base + tilingData->arena.gateway;
            combinePartialSendPtr = base + tilingData->arena.partialSend;
            combinePartialRecvPtr = base + tilingData->arena.partialRecv;
            hierarchicalControlPtr = base + tilingData->arena.control;
            gatewayPayloadReadyPtr = base + tilingData->arena.ready;
            combineReduceCommitPtr = base + tilingData->arena.commit;
            compactPtr = base + tilingData->arena.compact;
            gatherPtr = base + tilingData->arena.gather;
            closurePtr = base + tilingData->arena.closure;
            return;
        }
        int64_t offset = PEERMEM_DATA_OFFSET;
        maskRecvPtr = base + offset;
        if (UsesMteLayout(tilingData->topoType)) {
            offset += CalcRouteIndexRecvSize(CalcDispatchRouteIndexAlignSize(tilingData),
                                             static_cast<int64_t>(tilingData->moeExpertPerRank),
                                             static_cast<int64_t>(tilingData->epWorldSize));
        } else {
            offset += CalcMaskRecvSize(CalcDispatchMaskAlignSize(tilingData),
                                       static_cast<int64_t>(tilingData->moeExpertPerRank),
                                       static_cast<int64_t>(tilingData->epWorldSize));
        }

        expertCountRecvPtr = base + offset;
        if (UseDecodeRailRouting(tilingData)) {
            offset += CalcMteExpertCountRecvSize(tilingData->moeExpertPerRank, tilingData->epWorldSize);
            railTopkIdsRecvPtr = base + offset + HierarchicalMetadataPrefixBytes(tilingData);
            offset += CalcRailTopkIdsRecvSize(tilingData);
        } else {
            offset += CalcExpertCountRecvSize(tilingData->moeExpertPerRank, tilingData->epWorldSize);
        }

        int64_t tokenScaleBytes =
            CalcQuantTokenScaleBytes(static_cast<int64_t>(tilingData->h), elemsPerByte,
                                     static_cast<int64_t>(tilingData->topK), tilingData->topkWeightsPrefetch == 1);
        if (UsesMteLayout(tilingData->topoType)) {
            quantTokenScalePtr = base + offset;
            offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes,
                                           ALIGN_512);
        } else {
            dispatchRecivePtr = base + offset;
            int64_t relayRecordBytes = Ops::Base::CeilAlign(tokenScaleBytes, ALIGN_512);
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->bs) * relayRecordBytes * static_cast<int64_t>(serverNum), ALIGN_512);
            dispatchFlagPtr = base + offset;
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(serverNum) * tilingData->bs * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);
        }
        combineSendPtr = base + offset;
        if (UseHierarchicalRoce(tilingData)) {
            const int64_t combineBytes = Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK * tilingData->h * 2,
                ALIGN_512);
            offset += combineBytes;
            relayPayloadPtr = base + offset;
            offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes,
                                           ALIGN_512);
            gatewayCombinePtr = base + offset;
            offset += CalcGatewayCombineBytes(tilingData->numMaxTokensPerRank, tilingData->topK, tilingData->h);
            hierarchicalControlPtr = base + offset;
            offset += static_cast<int64_t>(HierarchicalSignal::Count) * HIERARCHICAL_FLAG_BYTES;
            // One persistent, single-writer line per gateway on this host.
            // Never reset at launch: a peer can broadcast before our reset.
            gatewayPayloadReadyPtr = base + offset;
            offset += HIERARCHICAL_RANKS_PER_HOST * HIERARCHICAL_FLAG_BYTES;

            const int64_t partialBytes = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) *
                                                                  tilingData->h * static_cast<int64_t>(sizeof(float)),
                                                              ALIGN_512);
            combinePartialSendPtr = base + offset;
            offset += partialBytes;
            combinePartialRecvPtr = base + offset;
            offset += partialBytes;
            combineReduceCommitPtr = base + offset;
        }
    }
};

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
__aicore__ inline __gm__ int32_t* HierarchicalSignalAddress(const PeermemInfo& info, HierarchicalSignal signal)
{
    return reinterpret_cast<__gm__ int32_t*>(info.hierarchicalControlPtr + HierarchicalSignalOffset(signal));
}
#endif
#else
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// device 侧窗口各区基址；偏移推进顺序即上方 CalcPeermemLeastSize 的分段顺序。
struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR maskRecvPtr;
    GM_ADDR expertCountRecvPtr;
    GM_ADDR railTopkIdsRecvPtr{nullptr};
    GM_ADDR quantTokenScalePtr;
    GM_ADDR dispatchRecivePtr;
    GM_ADDR dispatchFlagPtr;
    GM_ADDR combineSendPtr;

    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData* tilingData, uint32_t elemsPerByte = 1,
                                  uint32_t serverNum = 1)
    {
        rankSyncInWorldPtr = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        maskRecvPtr = base + offset;
        if (tilingData->topoType == TOPO_TYPE_MTE) {
            offset += CalcRouteIndexRecvSize(CalcDispatchRouteIndexAlignSize(tilingData),
                                             static_cast<int64_t>(tilingData->moeExpertPerRank),
                                             static_cast<int64_t>(tilingData->epWorldSize));
        } else {
            offset += CalcMaskRecvSize(CalcDispatchMaskAlignSize(tilingData),
                                       static_cast<int64_t>(tilingData->moeExpertPerRank),
                                       static_cast<int64_t>(tilingData->epWorldSize));
        }

        expertCountRecvPtr = base + offset;
        if (UseDecodeRailRouting(tilingData)) {
            offset += CalcMteExpertCountRecvSize(tilingData->moeExpertPerRank, tilingData->epWorldSize);
            railTopkIdsRecvPtr = base + offset;
            offset += CalcRailTopkIdsRecvSize(tilingData);
        } else {
            offset += CalcExpertCountRecvSize(tilingData->moeExpertPerRank, tilingData->epWorldSize);
        }

        int64_t tokenScaleBytes =
            CalcQuantTokenScaleBytes(static_cast<int64_t>(tilingData->h), elemsPerByte,
                                     static_cast<int64_t>(tilingData->topK), tilingData->topkWeightsPrefetch == 1);
        if (tilingData->topoType == TOPO_TYPE_MTE) {
            quantTokenScalePtr = base + offset;
            offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes,
                                           ALIGN_512);
        } else {
            dispatchRecivePtr = base + offset;
            int64_t relayRecordBytes = Ops::Base::CeilAlign(tokenScaleBytes, ALIGN_512);
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->bs) * relayRecordBytes * static_cast<int64_t>(serverNum), ALIGN_512);
            dispatchFlagPtr = base + offset;
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(serverNum) * tilingData->bs * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);
        }
        combineSendPtr = base + offset;
    }
};
#endif
#endif

}  // namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE

#endif  // MEGA_MOE_PEERMEM_H
