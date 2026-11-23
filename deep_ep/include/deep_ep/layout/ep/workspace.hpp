// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/ep/token.hpp>

// Count participation threshold is independent of the runtime send/receive split.
#ifndef DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD
#define DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD 512U
#endif

namespace deep_ep {
// BS below this threshold uses all launched cores for Count; otherwise only receivers.
constexpr uint32_t kEpServerDedupAllCoreCountBsThreshold = DISPATCH_SERVER_DEDUP_ALL_CORE_COUNT_BS_THRESHOLD;
constexpr uint32_t kEpServerDedupMaxExperts = 1024U;
constexpr uint32_t kEpServerDedupCountRecordBytes = 32U;
constexpr uint32_t kEpServerDedupCountGmStride = 512U;
// Persistent per-AIV state lives only in window 0's prefix, never in the
// selected ping-pong bank. Keep identical count offsets in both windows.
constexpr uint64_t kEpServerDedupMagicStride = 512U;
// Capacity only, matching DeepEP-Ascend's kNumMaxVecCores: one magic slot per
// possible vector core, independent of the count the device actually reports.
constexpr uint32_t kEpServerDedupMaxVecCores = 64U;
constexpr uint64_t kEpServerDedupControlPrefixBytes = kEpServerDedupMaxVecCores * kEpServerDedupMagicStride;
static_assert(kEpServerDedupControlPrefixBytes == 32U * 1024U, "capacity for 64 isolated magic slots");
static_assert(kEpServerDedupCountGmStride % kEpServerDedupCountRecordBytes == 0U &&
                  kEpServerDedupControlPrefixBytes % kEpServerDedupCountGmStride == 0U,
              "GM count slots must be aligned whole records");

// Matrix offsets are relative to cntOffset; cntOffset, counterOffset and
// stateBytes are relative to the selected window. Leading padding is retained.
struct EpServerDedupCountLayout {
    uint64_t cntOffset;
    uint64_t cumsumOffset;
    // 完成标记必须保持“写列、读行、清本行”的转置结构：每个核只读自己那一行，
    // 清理对其它核不可见。合并成共享槽位会让清理影响他核读取，需要额外的 epoch。
    uint64_t cumsumFlagOffset;
    uint64_t prefixOffset;  // One shared, read-only prefix begins here.
    uint64_t prefixStride;  // Retained replica reservation: keep control offsets/capacity unchanged.
    uint64_t counterOffset;  // Retired counter: preserve its reserved layout, never read/write it.
    uint64_t stateBytes;
};

// Keep each rank's core split fixed for the communicator lifetime: prefix storage is not
// cleared and can overlap another split's control flags. All ranks must use the
// same communicator call order and serialize its kernels
// on one stream. Count publication (including zeros) protects two-call reuse.
constexpr __aicore__ __forceinline__ uint64_t DispatchServerDedupWindowOffset(int64_t magic)
{
    return (uint64_t(magic) % 2U) * kEpServerDedupWindowBytes;
}

// Pure shape formula: usable in host and device static_assert and constexpr.
constexpr __aicore__ __forceinline__ EpServerDedupCountLayout DispatchServerDedupCountLayout(uint32_t experts,
                                                                                            uint32_t coreNum)
{
    EpServerDedupCountLayout layout{};
    const uint32_t recvCores = coreNum / 2U;
    // The subtotal scoreboard now uses one writer-owned 32B slot per receiver
    // core at cumsumOffset; the historical matrix footprint stays reserved so
    // prefix and flag offsets, capacities and the protocol layout are unchanged.
    const uint64_t matrixBytes = uint64_t(recvCores) * recvCores * kEpServerDedupCountRecordBytes;
    layout.cntOffset = kEpServerDedupControlPrefixBytes;
    layout.cumsumOffset = uint64_t(experts) * kEpServerDedupCountGmStride;
    layout.prefixOffset = layout.cumsumOffset + matrixBytes;
    layout.prefixStride = (uint64_t(experts) * sizeof(int32_t) + 31U) / 32U * 32U;
    // Only the first prefixStride is used. Retain the old reservation so this
    // optimization does not move readiness controls or change the wire layout.
    layout.cumsumFlagOffset = layout.prefixOffset + recvCores * layout.prefixStride;
    const uint64_t bytes = layout.cntOffset + layout.cumsumFlagOffset + matrixBytes;
    layout.counterOffset = (bytes + 1023U) / 1024U * 1024U;
    layout.stateBytes = layout.counterOffset + 1024U;
    return layout;
}

// Peak sendCnt allocation: three route buffers, work scratch (also the remote
// frame), and local count records/histogram. Only local records remain reserved.
// Sender/receive scratch overlays those four buffers and the histogram after
// filtering; the records remain reserved until the asynchronous writes retire.
__aicore__ __forceinline__ uint64_t DispatchServerDedupCountSendUbBytes(uint64_t routes, uint32_t experts,
                                                                            uint32_t countCores,
                                                                            uint32_t remoteRecords = 0U,
                                                                            bool packedFrames = false)
{
    const uint64_t routeBytes = (routes * sizeof(int32_t) + 255U) / 256U * 256U;
    const uint32_t records = (experts + countCores - 1U) / countCores;
    const uint32_t alignedRecords = (records + 7U) / 8U * 8U;
    const uint64_t remoteBytes = uint64_t((remoteRecords + 7U) / 8U * 8U) * 32U;
    const uint64_t workBytes = routeBytes < remoteBytes ? remoteBytes : routeBytes;
    const uint64_t localBytes = packedFrames ? CountWireBytes(records) : uint64_t(alignedRecords) * 32U;
    return 3U * routeBytes + workBytes + localBytes + uint64_t(alignedRecords) * sizeof(int32_t);
}
}  // namespace deep_ep

namespace deep_ep {
struct EpMemoryServerDedupLayout {
    uint64_t globalBs = 0;
    uint64_t localExpertNum = 0;
    uint64_t capacityRows = 0;
    EpServerDedupInputMode inputMode = EpServerDedupInputMode::Plain16;
    uint64_t xBytes = 0;
    uint64_t inputScalesBytes = 0;
    uint64_t expertIdsBytes = 0;
    uint64_t weightBytes = 0;
    uint64_t expandXBytes = 0;
    uint64_t dynamicScalesBytes = 0;
    uint64_t expertTokenNumsBytes = 0;
    uint64_t sendCountsBytes = 0;
    uint64_t countStateBytes = 0;
    uint64_t tokenTypeBytes = 0;
    uint64_t destinationIndexBytes = 0;
    uint64_t relayReadIndexRowStrideBytes = 0;
    uint64_t relayReadIndexBytes = 0;
    uint64_t sourceMaskBytes = 0;
    EpServerDedupDataLayout data{};
};
}  // namespace deep_ep
