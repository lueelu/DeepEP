// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_SEND_MASK_H
#define MEGA_MOE_SEND_MASK_H

#include "../common/mega_moe_utils.hpp"
#include "mega_moe_route_compact.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

struct SendMaskConfig {
    uint64_t expertCountWinOffset;
    uint64_t routeIndexAlignSize;
    uint64_t routeIndexWinOffset;
    uint64_t railTopkIdsWinOffset;
    uint64_t railTopkIdsSlotBytes;
    uint32_t serverNum;
    MegaMoeSendMaskBufferConfig bufferConfig;
};

// 装配 MTE 路径唯一的 compact route 发送配置。
__aicore__ inline SendMaskConfig CreateSendMaskConfig(const Params& params, uint32_t aivCoreIdx)
{
    uint64_t routeIndexWinOffset =
        static_cast<uint64_t>(params.peermemInfo.maskRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint64_t expertCountWinOffset =
        static_cast<uint64_t>(params.peermemInfo.expertCountRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    const MegaMoeSendMaskBufferConfig& bufferConfig = aivCoreIdx < params.tilingData->sendMaskCoreCountWithExtraExpert
                                                          ? params.tilingData->sendMaskConfigForCoreWithExtraExpert
                                                          : params.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    return {.expertCountWinOffset = expertCountWinOffset,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .routeIndexWinOffset = routeIndexWinOffset,
            .railTopkIdsWinOffset = params.peermemInfo.railTopkIdsRecvPtr == nullptr
                                        ? 0U
                                        : static_cast<uint64_t>(params.peermemInfo.railTopkIdsRecvPtr -
                                                                params.peermemInfo.rankSyncInWorldPtr),
            .railTopkIdsSlotBytes = static_cast<uint64_t>(CalcRailTopkIdsSlotBytes(params.tilingData)),
            .serverNum = static_cast<uint32_t>(CalcServerNum(params.tilingData->epWorldSize)),
            .bufferConfig = bufferConfig};
}

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<int32_t> topkIndexTensor;
    // Contiguous runtime-sized ring. Slot i starts at i * bufferConfig.bufferBytes.
    LocalTensor<uint8_t> routeRingTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

// MTE Wave：把当前 route batch 中命中某专家的全局 topkIndex 压紧后直接写入对端槽。
// 每个 slot 只保存 index；raw count 在所有 ring 写完成后由 PublishExpertCounts 独立发布。
__aicore__ inline void GatherAndSendExpertCompactRouteBatch(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                                            const SendMaskConfig& config, SendMaskScratch& scratch,
                                                            GlobalTensor<int32_t>& topkIdsGm,
                                                            GlobalTensor<int32_t>& dstRouteIndexGm,
                                                            int32_t ownedExpertBegin, int32_t ownedExpertNum,
                                                            int32_t batchIdx)
{
    const MegaMoeSendMaskBufferConfig& bufferConfig = config.bufferConfig;
    const uint32_t compareMaskBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) / BITS_PER_BYTE;
    const int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
    const int32_t realSendTotalNum =
        static_cast<int32_t>(static_cast<uint64_t>(common.tokenNum) * static_cast<uint64_t>(common.topK));
    const int32_t realRemain = realSendTotalNum - batchStart;
    int32_t validLen = bufferConfig.routeItemsPerBatch;
    if (realRemain < validLen) {
        validLen = realRemain > 0 ? realRemain : 0;
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    if (validLen > 0) {
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
        DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
        CreateVecIndex(scratch.topkIndexTensor, batchStart, validLen);
    }

    int32_t batchRingBegin = batchIdx * ownedExpertNum;
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t globalExpertId = ownedExpertBegin + ownedIdx;
        int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        int32_t bufferIdx = (batchRingBegin + ownedIdx) % bufferConfig.bufferCount;
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        uint32_t slotOffset = bufferIdx * bufferConfig.bufferBytes;
        LocalTensor<uint8_t> compareMaskTensor = scratch.routeRingTensor[slotOffset];
        LocalTensor<uint32_t> compareMaskU32Tensor = compareMaskTensor.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> tokValidIndexTensor =
            scratch.routeRingTensor[slotOffset + compareMaskBytes].template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        uint64_t batchMatchedRouteCount = 0U;
        if (validLen > 0) {
            CompareScalar(compareMaskTensor, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ, validLen);
            GatherMask(tokValidIndexTensor, scratch.topkIndexTensor, compareMaskU32Tensor, true,
                       static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        int32_t previousCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        int32_t remainingCapacity = static_cast<int32_t>(common.tokenNum) - previousCount;
        int32_t copiedCount = static_cast<int32_t>(batchMatchedRouteCount);
        if (copiedCount > remainingCapacity) {
            copiedCount = remainingCapacity > 0 ? remainingCapacity : 0;
        }
        scratch.sendCntAccTensor.SetValue(ownedIdx, previousCount + copiedCount);
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        if (copiedCount > 0) {
            uint64_t dstOffset = config.routeIndexWinOffset +
                                 static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                                       static_cast<int32_t>(common.rankId)) *
                                     config.routeIndexAlignSize +
                                 static_cast<uint64_t>(previousCount) * sizeof(int32_t);
            dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(winRankAddr[dstRank] + dstOffset));
            DataCopyPad(dstRouteIndexGm, tokValidIndexTensor,
                        {1U, static_cast<uint32_t>(copiedCount * sizeof(int32_t)), 0U, 0U, 0U});
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

// 将本核连续专家区间的 raw count 发布到各目标 rank 的 [localExpert][sourceRank] 表。
__aicore__ inline void PublishExpertCounts(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                           const SendMaskConfig& config, const SendMaskScratch& scratch,
                                           int32_t ownedExpertBegin, int32_t ownedExpertNum, int32_t arrivalEpoch)
{
    if (ownedExpertNum <= 0) {
        return;
    }
    /*
     * 到达标记内嵌：count 与跨卡同步信号走不同源核/不同通道(MTE3 vs scalar)，互连不保证
     * 到达序，接收端凭同步放行后可能读到在途 count(旧序靠共享 GMM1 时间垫层掩盖)。
     * 判据必须内嵌在数据自身的单次写内：高 8 位写 launch epoch(值域[0x80,0xFF]恒非零,
     * 避开窗口零初值)，低 24 位为真实 count(上限 maxOutputSize 远小于 2^24)。
     * 接收端逐槽校验 epoch 后取低 24 位(见 token_dispatch.h PrepareMoeExpertTokenCountTable)。
     */
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t rawCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        scratch.sendCntAccTensor.SetValue(ownedIdx, (arrivalEpoch << 24) | rawCount);
    }

    int32_t sourceRank = static_cast<int32_t>(common.rankId);
    int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    int32_t worldSize = static_cast<int32_t>(common.worldSize);
    int32_t ownedOffset = 0;
    GlobalTensor<int32_t> dstCountGm;

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    while (ownedOffset < ownedExpertNum) {
        int32_t globalExpertId = ownedExpertBegin + ownedOffset;
        int32_t dstRank = globalExpertId / expertPerRank;
        int32_t localExpertBegin = globalExpertId % expertPerRank;
        int32_t remainingInRank = expertPerRank - localExpertBegin;
        int32_t segmentExpertCount = ownedExpertNum - ownedOffset;
        segmentExpertCount = segmentExpertCount < remainingInRank ? segmentExpertCount : remainingInRank;

        uint64_t dstOffset = config.expertCountWinOffset +
                             static_cast<uint64_t>(localExpertBegin * worldSize + sourceRank) * sizeof(int32_t);
        dstCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(winRankAddr[dstRank] + dstOffset));
        DataCopyExtParams countCopyParams{
            static_cast<uint16_t>(segmentExpertCount), static_cast<uint32_t>(sizeof(int32_t)), 0,
            static_cast<int64_t>(worldSize - 1) * static_cast<int64_t>(sizeof(int32_t)), 0U};

        if (ownedOffset % static_cast<int32_t>(INT32_PER_256B) == 0) {
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendCntAccTensor[ownedOffset],
                                                       countCopyParams);
        } else {
            // compact route 已发送完成，复用 topkIndexTensor 将非对齐 count 段重排到对齐起点。
            for (int32_t expertIdx = 0; expertIdx < segmentExpertCount; ++expertIdx) {
                scratch.topkIndexTensor.SetValue(expertIdx, scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.topkIndexTensor, countCopyParams);
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
        }
        ownedOffset += segmentExpertCount;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// Prefill 和不适用 rail 分组的形状保留直发 route：连续 quotient/remainder
// 专家分配，count 按目标 rank 合并成二维 strided copy。
__aicore__ inline void GatherAndSendExpertCompactRoutesDirect(const AivJobContext& job,
                                                              const MoeStageCommonConfig& common, const Params& params,
                                                              GM_ADDR* winRankAddr, const SendMaskConfig& config,
                                                              SendMaskScratch& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const MegaMoeSendMaskBufferConfig& bufferConfig = config.bufferConfig;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    int32_t totalExperts = static_cast<int32_t>(common.worldSize * common.moeExpertPerRank);
    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    int32_t expertsPerJob = totalExperts / totalJobs;
    int32_t jobCountWithExtraExpert = totalExperts % totalJobs;
    int32_t ownedExpertNum = expertsPerJob + (jobIndex < jobCountWithExtraExpert ? 1 : 0);
    int32_t ownedExpertBegin =
        jobIndex * expertsPerJob + (jobIndex < jobCountWithExtraExpert ? jobIndex : jobCountWithExtraExpert);
    if (ownedExpertNum <= 0) {
        return;
    }

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdxGmAddr));
    GlobalTensor<int32_t> dstRouteIndexGm;
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        GatherAndSendExpertCompactRouteBatch(common, winRankAddr, config, scratch, topkIdsGm, dstRouteIndexGm,
                                             ownedExpertBegin, ownedExpertNum, batchIdx);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    __gm__ int32_t* launchCountSlot =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(GetBlockIdx()) * RANK_SYNC_COUNTER_SLOT_BYTES);
    // 本 launch epoch = 计数槽值+1(跨卡同步在本阶段之后执行,槽值仍为上一 launch)。
    int32_t arrivalEpoch = ((ReadGmByPassDCache(launchCountSlot) + 1) & 0x7F) | 0x80;
    PublishExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum, arrivalEpoch);
}

// 一批 UB 可以容纳的完整 512B 包；每包最后一个 word 留给到达标记。
__aicore__ inline int32_t RailItemsPerBatch(const MegaMoeSendMaskBufferConfig& config)
{
    return config.routeItemsPerBatch / RAIL_PACKET_WORDS * RAIL_PACKET_DATA_WORDS;
}

// 只在所有包标记匹配后消费数据，允许网络以任意次序交付这些包。
// 第一个包的标记还给出实际 item 数，因而可变 BS 和零 token 不必发送定容 padding 包。
__aicore__ inline int32_t WaitRailPacketArrival(__gm__ int32_t* slot, int32_t expectEpoch)
{
    int32_t firstMarker = ReadGmByPassDCache(slot + RAIL_PACKET_DATA_WORDS);
    while ((static_cast<uint32_t>(firstMarker) >> 24) != static_cast<uint32_t>(expectEpoch)) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
        firstMarker = ReadGmByPassDCache(slot + RAIL_PACKET_DATA_WORDS);
    }
    const int32_t totalItems = firstMarker & 0x00FFFFFF;
    const int32_t packetCount = (totalItems + RAIL_PACKET_DATA_WORDS - 1U) / RAIL_PACKET_DATA_WORDS;
    for (int32_t packet = 1; packet < packetCount; ++packet) {
        __gm__ int32_t* marker = slot + packet * RAIL_PACKET_WORDS + RAIL_PACKET_DATA_WORDS;
        while (ReadGmByPassDCache(marker) != firstMarker) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        }
    }
    return totalItems;
}

// Compact MTE2 去掉每包的末尾标记，UB 中仍为原来的连续 topkIds。
// 最后一个包的无效 payload 最多多读 126 个 word；只有 validItems 参与后续计算。
__aicore__ inline void LoadRailTopkIdsBatch(const SendMaskConfig& config, SendMaskScratch& scratch,
                                            GlobalTensor<int32_t>& railGm, int32_t batchIdx, int32_t validItems)
{
    const int32_t packetsPerBatch = config.bufferConfig.routeItemsPerBatch / RAIL_PACKET_WORDS;
    const int32_t packetCount = (validItems + RAIL_PACKET_DATA_WORDS - 1U) / RAIL_PACKET_DATA_WORDS;
    DataCopyExtParams copyParams{static_cast<uint16_t>(packetCount), RAIL_PACKET_DATA_WORDS * sizeof(int32_t),
                                 sizeof(int32_t), 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
    DataCopyPad<int32_t, PaddingMode::Compact>(scratch.topkIdsTensor,
                                               railGm[batchIdx * packetsPerBatch * RAIL_PACKET_WORDS], copyParams, pad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
}

/*
 * 每个目标 server 一个发送核。每个 512B 包携带自身的 epoch/length 标记，
 * 同批数据和标记用同一个连续 MTE3 写入，不再发独立 arrival，也不需要 data -> flag barrier。
 * 这里依赖平台的 512B 包内可见性约定，不依赖不同包或不同 MTE3 指令的完成顺序。
 */
__aicore__ inline void SendTopkIdsToRailServer(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                               const SendMaskConfig& config, const Params& params,
                                               SendMaskScratch& scratch, uint32_t targetServerIdx)
{
    const int32_t itemsPerBatch = RailItemsPerBatch(config.bufferConfig);
    const uint32_t railRank = targetServerIdx * static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE) +
                              common.rankId % static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE);
    const uint32_t srcServerIdx = common.rankId / static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE);
    const uint64_t slotOffset =
        config.railTopkIdsWinOffset + static_cast<uint64_t>(srcServerIdx) * config.railTopkIdsSlotBytes;
    const int32_t totalItems = static_cast<int32_t>(static_cast<uint64_t>(common.tokenNum) * common.topK);
    __gm__ int32_t* launchCountSlot =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(targetServerIdx) * RANK_SYNC_COUNTER_SLOT_BYTES);
    const uint32_t epoch = ((ReadGmByPassDCache(launchCountSlot) + 1) & 0x7F) | 0x80;
    const int32_t marker = static_cast<int32_t>((epoch << 24) | static_cast<uint32_t>(totalItems));
    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdxGmAddr));
    GlobalTensor<int32_t> dstRailGm;
    dstRailGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(winRankAddr[railRank] + slotOffset));

    const int32_t batchCount = totalItems > 0 ? (totalItems + itemsPerBatch - 1) / itemsPerBatch : 1;
    for (int32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
        const int32_t start = batchIdx * itemsPerBatch;
        const int32_t remain = totalItems - start;
        const int32_t valid = remain < itemsPerBatch ? remain : itemsPerBatch;
        const int32_t fullPackets = valid / RAIL_PACKET_DATA_WORDS;
        const int32_t tailItems = valid % RAIL_PACKET_DATA_WORDS;
        const int32_t packetCount = valid > 0 ? fullPackets + (tailItems != 0) : 1;
        if (batchIdx > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
        }
        DataCopyPadExtParams<int32_t> pad{true, 0U, 0U, 0U};
        if (fullPackets > 0) {
            // Normal MTE2 pads each 508B data block to 512B, reserving the marker word.
            DataCopyPad(scratch.topkIdsTensor, topkIdsGm[start],
                        {static_cast<uint16_t>(fullPackets), RAIL_PACKET_DATA_WORDS * sizeof(int32_t), 0U, 0U, 0U},
                        pad);
        }
        if (tailItems > 0) {
            DataCopyPad(scratch.topkIdsTensor[fullPackets * RAIL_PACKET_WORDS],
                        topkIdsGm[start + fullPackets * RAIL_PACKET_DATA_WORDS],
                        {1U, static_cast<uint32_t>(tailItems * sizeof(int32_t)), 0U, 0U, 0U}, pad);
        }
        // MTE2 padding may touch the marker word; stamp it only after input loads complete.
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID0>();
        for (int32_t packet = 0; packet < packetCount; ++packet) {
            scratch.topkIdsTensor.SetValue(packet * RAIL_PACKET_WORDS + RAIL_PACKET_DATA_WORDS, marker);
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
        DataCopyPad(dstRailGm[batchIdx * (itemsPerBatch / RAIL_PACKET_DATA_WORDS) * RAIL_PACKET_WORDS],
                    scratch.topkIdsTensor, {1U, static_cast<uint32_t>(packetCount) * RAIL_PACKET_BYTES, 0U, 0U, 0U});
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 中转核按批处理 rail 槽内 src 卡的 topkIds：逐专家筛选出全局 topkIndex 并写入目标卡 compact 槽。
// 逐专家直发：topkIds 源为本卡 rail 槽；route/count 的 source 下标用 src 卡号。
__aicore__ inline void RelayExpertCompactRouteBatch(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                                    const SendMaskConfig& config, SendMaskScratch& scratch,
                                                    GlobalTensor<int32_t>& topkIdsGm,
                                                    GlobalTensor<int32_t>& dstRouteIndexGm, int32_t ownedExpertBegin,
                                                    int32_t ownedExpertNum, int32_t sourceRank, int32_t srcTokenNum,
                                                    int32_t realSendTotalNum, int32_t batchIdx)
{
    const MegaMoeSendMaskBufferConfig& bufferConfig = config.bufferConfig;
    const uint32_t compareMaskBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) / BITS_PER_BYTE;
    const int32_t batchStart = batchIdx * RailItemsPerBatch(bufferConfig);
    const int32_t realRemain = realSendTotalNum - batchStart;
    int32_t validLen = RailItemsPerBatch(bufferConfig);
    if (realRemain < validLen) {
        validLen = realRemain > 0 ? realRemain : 0;
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    if (validLen > 0) {
        LoadRailTopkIdsBatch(config, scratch, topkIdsGm, batchIdx, validLen);
        CreateVecIndex(scratch.topkIndexTensor, batchStart, validLen);
    }

    int32_t batchRingBegin = batchIdx * ownedExpertNum;
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t globalExpertId = ownedExpertBegin + ownedIdx;
        int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        int32_t bufferIdx = (batchRingBegin + ownedIdx) % bufferConfig.bufferCount;
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        uint32_t slotOffset = bufferIdx * bufferConfig.bufferBytes;
        LocalTensor<uint8_t> compareMaskTensor = scratch.routeRingTensor[slotOffset];
        LocalTensor<uint32_t> compareMaskU32Tensor = compareMaskTensor.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> tokValidIndexTensor =
            scratch.routeRingTensor[slotOffset + compareMaskBytes].template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        uint64_t batchMatchedRouteCount = 0U;
        if (validLen > 0) {
            CompareScalar(compareMaskTensor, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ, validLen);
            GatherMask(tokValidIndexTensor, scratch.topkIndexTensor, compareMaskU32Tensor, true,
                       static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        int32_t previousCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        int32_t remainingCapacity = srcTokenNum - previousCount;
        int32_t copiedCount = static_cast<int32_t>(batchMatchedRouteCount);
        if (copiedCount > remainingCapacity) {
            copiedCount = remainingCapacity > 0 ? remainingCapacity : 0;
        }
        scratch.sendCntAccTensor.SetValue(ownedIdx, previousCount + copiedCount);
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        if (copiedCount > 0) {
            uint64_t dstOffset =
                config.routeIndexWinOffset +
                static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) + sourceRank) *
                    config.routeIndexAlignSize +
                static_cast<uint64_t>(previousCount) * sizeof(int32_t);
            dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(winRankAddr[dstRank] + dstOffset));
            DataCopyPad(dstRouteIndexGm, tokValidIndexTensor,
                        {1U, static_cast<uint32_t>(copiedCount * sizeof(int32_t)), 0U, 0U, 0U});
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

// route 已完成后复用 topkIndex UB，为每个目标 rank 准备独立的 32B 对齐 count 段。
// 所有段一次准备、连续提交、最后统一等待，避免复用同一段 UB 导致逐目标 rank 排空 MTE3。
__aicore__ inline bool TryPublishPackedExpertCounts(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                                    const SendMaskConfig& config, const SendMaskScratch& scratch,
                                                    int32_t ownedExpertBegin, int32_t ownedExpertNum,
                                                    int32_t sourceRank, int32_t arrivalEpoch)
{
    const int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    const int32_t firstLocalExpert = ownedExpertBegin % expertPerRank;
    const int32_t targetRankCount = (firstLocalExpert + ownedExpertNum - 1) / expertPerRank + 1;
    // 每段最多补 7 个 int32；容量不足时继续使用原来的逐段路径。
    if (ownedExpertNum + targetRankCount * (INT32_PER_256B - 1U) >
        static_cast<uint32_t>(config.bufferConfig.routeItemsPerBatch)) {
        return false;
    }
    for (int32_t ownedOffset = 0, packedOffset = 0; ownedOffset < ownedExpertNum;) {
        const int32_t remainingInRank = expertPerRank - (ownedExpertBegin + ownedOffset) % expertPerRank;
        const int32_t remainingOwned = ownedExpertNum - ownedOffset;
        const int32_t segmentCount = remainingOwned < remainingInRank ? remainingOwned : remainingInRank;
        for (int32_t expertIdx = 0; expertIdx < segmentCount; ++expertIdx) {
            const uint32_t rawCount = static_cast<uint32_t>(scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
            scratch.topkIndexTensor.SetValue(
                packedOffset + expertIdx, static_cast<int32_t>((static_cast<uint32_t>(arrivalEpoch) << 24) | rawCount));
        }
        ownedOffset += segmentCount;
        packedOffset += (segmentCount + INT32_PER_256B - 1U) / INT32_PER_256B * INT32_PER_256B;
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    GlobalTensor<int32_t> dstCountGm;
    for (int32_t ownedOffset = 0, packedOffset = 0; ownedOffset < ownedExpertNum;) {
        const int32_t globalExpertId = ownedExpertBegin + ownedOffset;
        const int32_t localExpertBegin = globalExpertId % expertPerRank;
        const int32_t remainingInRank = expertPerRank - localExpertBegin;
        const int32_t remainingOwned = ownedExpertNum - ownedOffset;
        const int32_t segmentCount = remainingOwned < remainingInRank ? remainingOwned : remainingInRank;
        const uint64_t dstOffset =
            config.expertCountWinOffset +
            (static_cast<uint64_t>(sourceRank) * CalcMteExpertCountRankStride(common.moeExpertPerRank) +
             localExpertBegin) *
                sizeof(int32_t);
        dstCountGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(winRankAddr[globalExpertId / expertPerRank] + dstOffset));
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(segmentCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.topkIndexTensor[packedOffset], copyParams);
        ownedOffset += segmentCount;
        packedOffset += (segmentCount + INT32_PER_256B - 1U) / INT32_PER_256B * INT32_PER_256B;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
    return true;
}

// 将一段连续专家的 raw count 发布到各目标 rank 的 [sourceRank][localExpert] 表。
// sourceRank 为该批 route 数据的真实源卡号（rail 中转时是远端 src 卡，而非本卡）。
__aicore__ inline void PublishRailExpertCounts(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                               const SendMaskConfig& config, const SendMaskScratch& scratch,
                                               int32_t ownedExpertBegin, int32_t ownedExpertNum, int32_t sourceRank,
                                               int32_t arrivalEpoch)
{
    if (ownedExpertNum <= 0) {
        return;
    }
    if (TryPublishPackedExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum, sourceRank,
                                     arrivalEpoch)) {
        return;
    }
    /*
     * 到达标记内嵌：count 与跨卡同步信号走不同源核/不同通道(MTE3 vs scalar)，互连不保证
     * 到达序，接收端凭同步放行后可能读到在途 count(旧序靠共享 GMM1 时间垫层掩盖)。
     * 判据必须内嵌在数据自身的单次写内：高 8 位写 launch epoch(值域[0x80,0xFF]恒非零,
     * 避开窗口零初值)，低 24 位为真实 count(上限 maxOutputSize 远小于 2^24)。
     * 接收端逐槽校验 epoch 后取低 24 位(见 token_dispatch.h PrepareMoeExpertTokenCountTable)。
     */
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t rawCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        scratch.sendCntAccTensor.SetValue(ownedIdx, (arrivalEpoch << 24) | rawCount);
    }

    int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    int32_t ownedOffset = 0;
    GlobalTensor<int32_t> dstCountGm;

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    while (ownedOffset < ownedExpertNum) {
        int32_t globalExpertId = ownedExpertBegin + ownedOffset;
        int32_t dstRank = globalExpertId / expertPerRank;
        int32_t localExpertBegin = globalExpertId % expertPerRank;
        int32_t remainingInRank = expertPerRank - localExpertBegin;
        int32_t segmentExpertCount = ownedExpertNum - ownedOffset;
        segmentExpertCount = segmentExpertCount < remainingInRank ? segmentExpertCount : remainingInRank;

        uint64_t dstOffset =
            config.expertCountWinOffset +
            (static_cast<uint64_t>(sourceRank) * CalcMteExpertCountRankStride(expertPerRank) + localExpertBegin) *
                sizeof(int32_t);
        dstCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(winRankAddr[dstRank] + dstOffset));
        DataCopyExtParams countCopyParams{1U, static_cast<uint32_t>(segmentExpertCount * sizeof(int32_t)), 0U, 0U, 0U};

        if (ownedOffset % static_cast<int32_t>(INT32_PER_256B) == 0) {
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendCntAccTensor[ownedOffset],
                                                       countCopyParams);
        } else {
            // compact route 已发送完成，复用 topkIndexTensor 将非对齐 count 段重排到对齐起点。
            for (int32_t expertIdx = 0; expertIdx < segmentExpertCount; ++expertIdx) {
                scratch.topkIndexTensor.SetValue(expertIdx, scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.topkIndexTensor, countCopyParams);
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
        }
        ownedOffset += segmentExpertCount;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 同一 rail 源的接收核按连续专家区间均分；每个 (sourceRank, expert) 只有一个写入者。
// relayCoreIdx 是去掉发送核后的核号。接收核少于源槽时，每个源槽由一个核处理全部专家。
__aicore__ inline void CalcRailRelayExpertRange(uint32_t serverExpertNum, uint32_t relayCoreIdx,
                                                uint32_t relayCoreCount, uint32_t serverNum, uint32_t relayIdx,
                                                uint32_t& expertOffset, uint32_t& expertCount)
{
    const uint32_t splitCount =
        relayCoreCount > relayIdx ? (relayCoreCount - relayIdx + serverNum - 1U) / serverNum : 1U;
    const uint32_t splitIdx = relayCoreIdx / serverNum;
    const uint32_t expertsPerSplit = serverExpertNum / splitCount;
    const uint32_t extraExpertSplits = serverExpertNum % splitCount;
    expertOffset = splitIdx * expertsPerSplit + (splitIdx < extraExpertSplits ? splitIdx : extraExpertSplits);
    expertCount = expertsPerSplit + (splitIdx < extraExpertSplits ? 1U : 0U);
}

// 单批且 UB 足够时，一次 VF 生成本核所有专家的 route/count，再连续提交 route DMA。
// 复用现有 ring 或 topkIndex 区，不增加 UB/tiling 空间；其余规格保持分批 ring 路径。
__aicore__ inline bool TryRelayCompactRoutes(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                             const SendMaskConfig& config, SendMaskScratch& scratch,
                                             GlobalTensor<int32_t>& topkIdsGm, int32_t ownedExpertBegin,
                                             int32_t ownedExpertNum, int32_t sourceRank, int32_t srcTokenNum,
                                             int32_t realSendTotalNum)
{
    const MegaMoeSendMaskBufferConfig& bufferConfig = config.bufferConfig;
    if (realSendTotalNum <= 0 || realSendTotalNum > RailItemsPerBatch(bufferConfig)) {
        return false;
    }
    const uint32_t routeStrideItems =
        (static_cast<uint32_t>(srcTokenNum) + INT32_PER_256B - 1U) / INT32_PER_256B * INT32_PER_256B;
    const uint64_t routeBytes = static_cast<uint64_t>(ownedExpertNum) * routeStrideItems * sizeof(int32_t);
    const uint64_t ringBytes = static_cast<uint64_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    LocalTensor<int32_t> routeTensor = scratch.routeRingTensor.template ReinterpretCast<int32_t>();
    if (routeBytes > ringBytes) {
        if (routeBytes > static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) * sizeof(int32_t)) {
            return false;
        }
        routeTensor = scratch.topkIndexTensor;
    }
    LoadRailTopkIdsBatch(config, scratch, topkIdsGm, 0, realSendTotalNum);
    RelayCompactRoutesVF(reinterpret_cast<__ubuf__ int32_t*>(scratch.topkIdsTensor.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ int32_t*>(routeTensor.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ int32_t*>(scratch.sendCntAccTensor.GetPhyAddr()),
                         static_cast<uint32_t>(realSendTotalNum), common.topK, ownedExpertBegin,
                         static_cast<uint32_t>(ownedExpertNum), routeStrideItems);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
    GlobalTensor<int32_t> dstRouteGm;
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        const int32_t routeCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        if (routeCount == 0) {
            continue;
        }
        const uint32_t globalExpertId = static_cast<uint32_t>(ownedExpertBegin + ownedIdx);
        const uint64_t dstOffset =
            config.routeIndexWinOffset +
            (static_cast<uint64_t>(globalExpertId % common.moeExpertPerRank) * common.worldSize + sourceRank) *
                config.routeIndexAlignSize;
        dstRouteGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(winRankAddr[globalExpertId / common.moeExpertPerRank] + dstOffset));
        DataCopyPad(dstRouteGm, routeTensor[static_cast<uint32_t>(ownedIdx) * routeStrideItems],
                    {1U, static_cast<uint32_t>(routeCount * sizeof(int32_t)), 0U, 0U, 0U});
    }
    // count 发布可能复用 topkIndex，必须等 route DMA 完成后才能改写它。
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
    return true;
}

/*
 * 多个中转核只读同一 rail 槽，每核重建并发布自己独占的连续专家区间。
 * route 保持 expert-major 接收布局，count 使用 source-major 布局；
 * count 的 epoch 同时承载原源卡输入就绪。
 */
__aicore__ inline void RelayRailTopkIdsToExperts(const MoeStageCommonConfig& common, GM_ADDR* winRankAddr,
                                                 const SendMaskConfig& config, const Params& params,
                                                 SendMaskScratch& scratch, uint32_t relayIdx, uint32_t aivCoreIdx,
                                                 int32_t ownedExpertBegin, int32_t ownedExpertNum)
{
    const MegaMoeSendMaskBufferConfig& bufferConfig = config.bufferConfig;
    // railTopkIdsRecvPtr 已是本卡 rail 区绝对基址（见 peermem.h PeermemInfo 布局），
    // 槽内偏移只需 relayIdx * slotBytes；不能再叠加 railTopkIdsWinOffset（那是相对
    // rankSyncInWorldPtr 的区域基址，会双倍偏移导致读不到发送核写入的槽）。
    const uint64_t slotOffset = static_cast<uint64_t>(relayIdx) * config.railTopkIdsSlotBytes;
    __gm__ int32_t* railSlotBase =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr) + slotOffset / sizeof(int32_t);
    // 本轮输入 epoch = 本核计数槽值 +1；与发送核/接收端的预测公式同源。
    // 必须读取本核槽：同源的其他分片核可能已完成并推进了自己的 epoch。
    __gm__ int32_t* launchCountSlot =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(aivCoreIdx) * RANK_SYNC_COUNTER_SLOT_BYTES);
    int32_t expectEpoch = ((ReadGmByPassDCache(launchCountSlot) + 1) & 0x7F) | 0x80;
    const int32_t realSendTotalNum = WaitRailPacketArrival(railSlotBase, expectEpoch);
    const int32_t srcTokenNum = realSendTotalNum / static_cast<int32_t>(common.topK);
    const int32_t sourceRank = static_cast<int32_t>(relayIdx * static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE) +
                                                    common.rankId % static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE));

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer(railSlotBase);
    if (TryRelayCompactRoutes(common, winRankAddr, config, scratch, topkIdsGm, ownedExpertBegin, ownedExpertNum,
                              sourceRank, srcTokenNum, realSendTotalNum)) {
        PublishRailExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum, sourceRank,
                                expectEpoch);
        return;
    }
    GlobalTensor<int32_t> dstRouteIndexGm;
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    const int32_t itemsPerBatch = RailItemsPerBatch(bufferConfig);
    const int32_t actualBatchCount = (realSendTotalNum + itemsPerBatch - 1) / itemsPerBatch;
    for (int32_t batchIdx = 0; batchIdx < actualBatchCount; ++batchIdx) {
        RelayExpertCompactRouteBatch(common, winRankAddr, config, scratch, topkIdsGm, dstRouteIndexGm, ownedExpertBegin,
                                     ownedExpertNum, sourceRank, srcTokenNum, realSendTotalNum, batchIdx);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }

    int32_t arrivalEpoch = expectEpoch;
    PublishRailExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum, sourceRank,
                            arrivalEpoch);
}

/*
 * MTE Wave 的 route 唯一入口（rail 中转版）：
 *   - 核 [0, serverNum)：发送核，把本卡 topkIds 整段发给 server j 的同号卡；
 *   - 核 [serverNum, totalJobs)：全部用于中转，先轮转分配到源 server，再在同源核之间均分专家；
 *   - 64 AIV / 64 rank / 384 expert 时：8 个发送核，56 个中转核，每源 7 核，每核 6 或 7 专家。
 * 需要 totalJobs > serverNum。接收核不足 serverNum 时，每核顺序处理多个源槽。
 */
__aicore__ inline void GatherAndSendExpertCompactRoutesRail(const AivJobContext& job,
                                                            const MoeStageCommonConfig& common, const Params& params,
                                                            GM_ADDR* winRankAddr, const SendMaskConfig& config,
                                                            SendMaskScratch& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    const uint32_t serverNum = config.serverNum;
    const uint32_t myServer = common.rankId / static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE);
    const uint32_t jobIdx = job.jobIndex;

    if (jobIdx < serverNum) {
        SendTopkIdsToRailServer(common, winRankAddr, config, params, scratch, jobIdx);
        return;
    }
    if (job.totalJobs <= serverNum) {
        return;
    }
    const uint32_t relayCoreIdx = jobIdx - serverNum;
    const uint32_t relayCoreCount = job.totalJobs - serverNum;
    const uint32_t serverRankBegin = myServer * static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE);
    const uint32_t remainingRanks = common.worldSize - serverRankBegin;
    const uint32_t serverCardCount = remainingRanks < static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE)
                                         ? remainingRanks
                                         : static_cast<uint32_t>(MEGA_MOE_SERVER_SIZE);
    const uint32_t serverExpertBegin = serverRankBegin * common.moeExpertPerRank;
    const uint32_t serverExpertNum = serverCardCount * common.moeExpertPerRank;
    for (uint32_t relayIdx = relayCoreIdx % serverNum; relayIdx < serverNum; relayIdx += relayCoreCount) {
        uint32_t expertOffset;
        uint32_t expertCount;
        CalcRailRelayExpertRange(serverExpertNum, relayCoreIdx, relayCoreCount, serverNum, relayIdx, expertOffset,
                                 expertCount);
        if (expertCount == 0U) {
            continue;
        }
        RelayRailTopkIdsToExperts(common, winRankAddr, config, params, scratch, relayIdx, jobIdx,
                                  static_cast<int32_t>(serverExpertBegin + expertOffset),
                                  static_cast<int32_t>(expertCount));
    }
}

// Decode uses rail packets and source-major counts; prefill keeps its original wire protocol.
__aicore__ inline void GatherAndSendExpertCompactRoutes(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                        const Params& params, GM_ADDR* winRankAddr,
                                                        const SendMaskConfig& config, SendMaskScratch& scratch)
{
    if (UseDecodeRailRouting(params.tilingData)) {
        GatherAndSendExpertCompactRoutesRail(job, common, params, winRankAddr, config, scratch);
    } else {
        GatherAndSendExpertCompactRoutesDirect(job, common, params, winRankAddr, config, scratch);
    }
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_SEND_MASK_H
