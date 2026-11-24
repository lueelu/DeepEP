// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#pragma once

#include <cstdint>
#include <cfloat>

#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_BF16_HOST_DEVICE
#endif

namespace MegaMoeBF16Impl {

constexpr uint64_t kAlignment32 = 32;
constexpr uint64_t kAlignment128 = 128;
constexpr uint64_t kAlignment256 = 256;
constexpr uint64_t kAlignment512 = 512;
// Alignment of saved metadata, expert-count strides and routed rows, respectively.
constexpr uint64_t kMetaAlignment = kAlignment512;
constexpr uint64_t kTokenCountStrideAlignment = kAlignment128;
constexpr uint64_t kRoutingRowAlignment = kAlignment256;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint32_t kTopkWeightBf16 = 0;
constexpr uint32_t kTopkWeightFp32 = 1;
constexpr uint64_t kInt32Bytes = 4;
constexpr uint64_t kIntCacheline = 16;
constexpr uint32_t kSwigluNHalf = 2U;
constexpr uint64_t kMaxEpRanks = 1024;
constexpr uint64_t kMaxLocalExperts = 32;
constexpr uint64_t kMaxTotalExperts = 2048;

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t TopkWeightBytes(uint32_t type)
{
    return type == kTopkWeightFp32 ? sizeof(float) : kBf16Bytes;
}

// Ordinary BF16 always reserves 64 KiB for one 64-byte slot per supported
// rank and 8 KiB for per-AIV generation counters. The shared 72 KiB header
// keeps Host allocation, forward and backward offsets independent of EP.
constexpr uint64_t kPeerRankSyncSlotBytes = 64;
constexpr uint64_t kPeerSyncCountOffset = 64 * 1024;
constexpr uint64_t kPeerDataOffset = 72 * 1024;
constexpr uint64_t kCompactTokenCountAutoMinEp = 512;
// A 64-byte packet stores up to eight int32 counts in [0, 32) and the
// matching eight exclusive prefixes in [32, 64).
constexpr uint64_t kCompactTokenCountMaxLocalExperts = 8;
constexpr uint64_t kCompactTokenCountPacketBytes = 64;
constexpr uint64_t kCompactTokenCountPrefixOffset = 32;
constexpr uint64_t kCompactTokenCountMaxAivCores = 72;
static_assert(kCompactTokenCountMaxLocalExperts * kInt32Bytes <= kCompactTokenCountPrefixOffset);
static_assert(kCompactTokenCountPrefixOffset + kCompactTokenCountMaxLocalExperts * kInt32Bytes <=
              kCompactTokenCountPacketBytes);

static_assert(kMaxEpRanks * kPeerRankSyncSlotBytes <= kPeerSyncCountOffset);
// A5 exposes at most 72 AIVs (36 AICs in MIX 1:2); 8 KiB is ample for their
// 64-byte generation-counter slots.
static_assert(72 * kPeerRankSyncSlotBytes <= kPeerDataOffset - kPeerSyncCountOffset);

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool CanUseCompactTokenCountExchange(uint64_t worldSize, uint64_t expertPerRank)
{
    return worldSize >= 2 && worldSize <= kMaxEpRanks && expertPerRank > 0 &&
           expertPerRank <= kCompactTokenCountMaxLocalExperts;
}

// Large EP groups exchange one packet of counts and prefixes per peer when the local experts fit.
// Smaller groups use the dense count table.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseCompactTokenCountExchange(uint64_t worldSize, uint64_t expertPerRank)
{
    return worldSize >= kCompactTokenCountAutoMinEp && CanUseCompactTokenCountExchange(worldSize, expertPerRank);
}

// Ordinary BF16 large-EP exchange sends one aligned packet per destination.
// Phase one keeps the existing dense local table and saved-state ABI: packets
// are unpacked into the same source-rank rows consumed by forward/backward.
// Experimental variants pass allowCompact=false and retain their old layout.
struct CompactTokenCountExchangeLayout {
    bool enabled;
    uint64_t packetOffset;
    uint64_t packetBytes;
    uint64_t blockSumsOffset;
    uint64_t blockSumsBytes;
    uint64_t tokenCountsOffset;
    uint64_t tokenCountStride;
    uint64_t tokenCountsBytes;
    uint64_t payloadOffset;

    MEGA_MOE_BF16_HOST_DEVICE constexpr CompactTokenCountExchangeLayout(uint64_t worldSize, uint64_t expertPerRank,
                                                                        bool allowCompact = true)
        : enabled(allowCompact && CanUseCompactTokenCountExchange(worldSize, expertPerRank)),
          packetOffset(kPeerDataOffset),
          packetBytes(enabled ? AlignUp(worldSize * kCompactTokenCountPacketBytes, kAlignment512) : 0),
          blockSumsOffset(packetOffset + packetBytes),
          blockSumsBytes(enabled ? AlignUp(kCompactTokenCountMaxAivCores * kInt32Bytes, kAlignment512) : 0),
          tokenCountsOffset(blockSumsOffset + blockSumsBytes),
          tokenCountStride(AlignUp(worldSize * expertPerRank, kAlignment128)),
          tokenCountsBytes(worldSize * tokenCountStride * kInt32Bytes),
          payloadOffset(tokenCountsOffset + tokenCountsBytes)
    {
    }
};

// One X (forward) or dY (backward) per source token; indices and returns use route order.
struct PeerPayloadLayout {
    uint64_t routeIndexOffset, weightOffset, returnOffset, topkGradOffset, totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr PeerPayloadLayout(uint64_t base, uint64_t m, uint64_t h, uint64_t topK,
                                                          uint64_t weightBytes)
        : routeIndexOffset(base + m * h * kBf16Bytes),
          weightOffset(routeIndexOffset + AlignUp(m * topK * kInt32Bytes, kAlignment512)),
          returnOffset(weightOffset + AlignUp(m * topK * weightBytes, kAlignment512)),
          topkGradOffset(returnOffset + m * topK * h * kBf16Bytes),
          totalBytes(topkGradOffset + AlignUp(m * topK * weightBytes, kAlignment512))
    {
    }
};

// Backward recomputes Gate/Up for none, reads BF16 values for bf16, and decodes packed values for mxfp8.
enum SavedGateUpMode : uint32_t { kSaveNone = 1, kSaveBf16 = 2, kSaveMxFp8 = 3 };

// A complete row uses 6F bytes of BF16 inputs, 4F bytes of FP32 products
// and a padded scalar result. All three save modes stream wider rows through
// one reusable FP32 row per AIV and apply the same reduction tree.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseStreamedTopkGrad(uint32_t mode, uint64_t f)
{
    return (mode == kSaveNone || mode == kSaveBf16 || mode == kSaveMxFp8) && f > 16384 && f % 256 == 0;
}
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseWholeRowTopkGrad(uint32_t mode, uint64_t f)
{
    return UseStreamedTopkGrad(mode, f) ||
           (f > 0 && f <= 16384 &&
            ((mode == kSaveMxFp8 && f % 256 == 0) || ((mode == kSaveBf16 || mode == kSaveNone) && f % 64 == 0)));
}
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseRecomputeDhpRowReady(uint32_t mode, uint64_t f)
{
    return mode == kSaveNone && UseWholeRowTopkGrad(mode, f) && !UseStreamedTopkGrad(mode, f);
}
constexpr uint32_t kWholeRowTopkResultBytes = 8U * 1024U;
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseBackwardStagedWave(uint32_t mode, uint64_t f)
{
    return f > 0 && f <= 16384 && ((mode == kSaveBf16 && f % 64 == 0) || (mode == kSaveMxFp8 && f % 32 == 0));
}
MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t TopkGradBatchRows(uint32_t columns)
{
    if (columns == 0) return 0;
    const uint32_t capacity = (248U * 1024U - kWholeRowTopkResultBytes) / (columns * 10U);
    return capacity < 8U ? capacity : 8U;
}

// CANN's FP32 contiguous AR sum first selects an A tile by core occupancy,
// then splits R to fit its four input buffers. The rank's complete routed
// row count is needed: one-row AR can have a different addition tree.
struct TopkGradReductionPlan {
    uint32_t width;
    uint32_t stripesPerGroup;
};
MEGA_MOE_BF16_HOST_DEVICE constexpr TopkGradReductionPlan MakeTopkGradReductionPlan(uint64_t rows, uint32_t columns,
                                                                                    uint32_t cores)
{
    if (rows == 0 || columns == 0 || cores == 0) return {columns, 1};
    constexpr uint64_t elements = (248U * 1024U - 16U * 1024U - 2U * 512U) / 16U;
    auto ceilDiv = [](uint64_t n, uint64_t d) { return (n + d - 1U) / d; };
    auto occupancy = [&](uint64_t outerA, uint64_t stripes) {
        const uint64_t tasks = outerA * stripes;
        const uint64_t active = tasks / ceilDiv(tasks, cores);
        return active > outerA ? active / outerA * outerA : active;
    };
    // CANN 9.2 ReduceSum selects split sizes above 95% AIV occupancy.
    constexpr uint32_t occupancyPercent = 95U;
    const uint64_t rTasks = ceilDiv(columns, 64U);
    uint64_t best = 0, aTile = 1;
    for (uint64_t candidate = 1; candidate <= rows && candidate <= 128U; ++candidate) {
        const uint64_t active = occupancy(ceilDiv(rows, candidate), rTasks);
        if (active * 100U > cores * occupancyPercent && active > best) aTile = candidate;
        best = active > best ? active : best;
    }
    if (aTile * columns <= elements) return {columns, 1};
    const uint64_t outerA = ceilDiv(rows, aTile);
    const uint32_t capacity = static_cast<uint32_t>(elements / aTile);
    uint32_t width = capacity / 8U * 8U;
    for (uint32_t w = width; w > 64U; w -= 8U) {
        if (occupancy(outerA, ceilDiv(columns, w)) * 100U > cores * occupancyPercent) {
            width = w;
            break;
        }
    }
    // CANN looks for an even divisor down to half the available R capacity,
    // then aligns to a 32-byte block. The last stripe can be shorter.
    for (uint32_t w = width; w >= capacity / 2U; --w) {
        if (columns % w == 0) {
            width = w / 8U * 8U;
            break;
        }
    }
    const uint32_t stripes = static_cast<uint32_t>(ceilDiv(columns, width));
    const uint64_t tasks = outerA * stripes;
    uint64_t blocks = ceilDiv(tasks, ceilDiv(tasks, cores));
    if (blocks > outerA) {
        const uint64_t up = ceilDiv(blocks, outerA) * outerA;
        blocks = up <= cores ? up : blocks / outerA * outerA;
    }
    return {width, static_cast<uint32_t>(ceilDiv(stripes, ceilDiv(blocks, outerA)))};
}

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t BackwardTopkPartialCount(uint32_t mode, uint64_t f)
{
    return UseWholeRowTopkGrad(mode, f) ? 1U : (f + 255U) / 256U;
}
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseGmForBackwardDataGrad(uint32_t savedMode)
{
    return savedMode != kSaveNone;
}

// v2 appends three uint32 bit planes only for explicit MXFP8 with clipping.
// All v1 prefix/data/scale offsets are retained. No GM header or initialization.
enum SavedClipMaskKind : uint32_t { kClipMaskNone = 0, kClipMaskThreeBitplanesV1 = 1 };
MEGA_MOE_BF16_HOST_DEVICE constexpr bool SaveGateUpClipMasks(float tilingLimit, uint32_t mode)
{
    return mode == kSaveMxFp8 && tilingLimit != FLT_MAX;
}

// AIV0 epilogue only. Dispatch and GMM2 own UB in different phases.
template <uint32_t RawRows, uint32_t ChunkRows = RawRows, uint32_t Columns = 256>
struct SavedGateUpQuantUbLayoutFor {
    static constexpr uint32_t kRows = ChunkRows;
    static constexpr uint32_t kColumns = Columns;
    static constexpr uint32_t kElements = kRows * kColumns;
    static constexpr uint32_t kGroups = kElements / 32;
    static constexpr uint32_t gateData = 2 * RawRows * kColumns * kBf16Bytes;
    static constexpr uint32_t upData = gateData + kElements;
    static constexpr uint32_t compactScale = upData + kElements;
    static constexpr uint32_t gateScale = compactScale + kGroups * 2;
    static constexpr uint32_t upScale = gateScale + kRows * 32;
    static constexpr uint32_t maxExp = upScale + kRows * 32;
    static constexpr uint32_t halfScale = maxExp + kGroups * 2;
    static constexpr uint32_t totalBytes = halfScale + kGroups * 2;
    static constexpr uint32_t maskPlaneBytes = kRows * 32;
    static constexpr uint32_t gateHigh = totalBytes;
    static constexpr uint32_t upLow = gateHigh + maskPlaneBytes;
    static constexpr uint32_t upHigh = upLow + maskPlaneBytes;
    static constexpr uint32_t totalBytesWithClipMasks = upHigh + maskPlaneBytes;
};

using SavedGateUpQuantUbLayout = SavedGateUpQuantUbLayoutFor<128>;

// Backward SwiGLU uses three BF16 tiles followed by the MXFP8 decode scratch.
struct BackwardSwigluUbLayout {
    static constexpr uint32_t kTileRows = 128;
    static constexpr uint32_t kTileColumns = 256;
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static constexpr uint32_t kTileBytes = kTileElements * kBf16Bytes;
    static constexpr uint32_t kDecodeRows = 16;
    static constexpr uint32_t kMxDataElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxScaleRowBytes = 32;
    static constexpr uint32_t kMxScaleBytes = kDecodeRows * kMxScaleRowBytes;
    // DIST_INTLV stores ignore the mask and write 512 bytes on A5.
    // Reserve their full footprint before the clipping masks.
    static constexpr uint32_t kMxDualRegisterStoreBytes = 2U * 256U;
    static constexpr uint32_t kMxScaleBf16Elements = kMxDualRegisterStoreBytes / kBf16Bytes;
    static constexpr uint32_t kMxScaleBf16Bytes = kMxScaleBf16Elements * kBf16Bytes;
    static constexpr uint32_t kMxScaleFp32Elements = kMxDualRegisterStoreBytes / sizeof(float);
    static constexpr uint32_t kMxScaleFp32Bytes = kMxScaleFp32Elements * sizeof(float);
    static constexpr uint32_t kWorkTileBytes = 3U * kTileBytes;
    static constexpr uint32_t kMxDataOffset = kWorkTileBytes;
    static constexpr uint32_t kMxScaleOffset = kMxDataOffset + kMxDataElements;
    static constexpr uint32_t kMxScaleBf16Offset = kMxScaleOffset + kMxScaleBytes;
    static constexpr uint32_t kMxScaleFp32Offset = kMxScaleBf16Offset + kMxScaleBf16Bytes;
    static constexpr uint32_t kClipMaskBytes = kTileRows * kMxScaleRowBytes;
    static constexpr uint32_t kGateHighOffset = kMxScaleFp32Offset + kMxScaleFp32Bytes;
    static constexpr uint32_t kUpLowOffset = kGateHighOffset + kClipMaskBytes;
    static constexpr uint32_t kUpHighOffset = kUpLowOffset + kClipMaskBytes;
    static constexpr uint32_t kMxScratchEnd = kUpHighOffset + kClipMaskBytes;
};

// Simultaneous UB allocations during count exchange and Dispatch. GM count
// rows contain all experts; Cumsum gathers only this rank's expert columns.
struct ForwardDispatchUbLayout {
    static constexpr uint64_t kTopkWeightBytes = 4 * 1024;
    static constexpr uint64_t kCountResetChunkBytes = 16 * 1024;
    uint64_t syncBytes;
    uint64_t prevSumBytes;
    uint64_t cumsumBytes;
    uint64_t copyElementsPerPing;
    uint64_t copyBytes;
    uint64_t resetFlagBytes;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardDispatchUbLayout(uint64_t worldSize, uint64_t expertPerRank, uint64_t k,
                                                                uint64_t dispatchRows)
        : syncBytes(AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes),
          // CrossRankSyncCntPerExpert writes and copies exactly one int32
          // prefix per local expert.  It never indexes this buffer by rank or
          // total expert, so reserving syncBytes * expertPerRank was excessive.
          prevSumBytes(AlignUp(expertPerRank * kInt32Bytes, 32)),
          cumsumBytes(worldSize * AlignUp(expertPerRank * kInt32Bytes, 32)),
          copyElementsPerPing(AlignUp(dispatchRows * k, 32)),
          copyBytes(2 * copyElementsPerPing * kBf16Bytes),
          // Preserve the existing scratch reservation, rounded as InitBuffer does.
          resetFlagBytes(AlignUp((expertPerRank * kIntCacheline * kInt32Bytes * 2 + 31) / 32, 32)),
          totalBytes(syncBytes + prevSumBytes + cumsumBytes + copyBytes + kTopkWeightBytes + resetFlagBytes)
    {
    }
};

// Ordinary BF16 Dispatch output. Packed tokens and their packed TopK weights
// share the same row order and allocation lifetime. The framework retains
// both for the matching backward, independently of forward transient scratch.
struct DispatchOutputLayout {
    uint64_t tokensOffset;
    uint64_t tokensBytes;
    uint64_t topkWeightsOffset;
    uint64_t topkWeightsBytes;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr DispatchOutputLayout(uint64_t k, uint64_t maxRecvTokens,
                                                             uint64_t weightBytes = kBf16Bytes)
        : tokensOffset(0),
          tokensBytes(AlignUp(maxRecvTokens * k * kBf16Bytes, kAlignment512)),
          topkWeightsOffset(tokensOffset + tokensBytes),
          topkWeightsBytes(AlignUp(maxRecvTokens * weightBytes, kAlignment512)),
          totalBytes(topkWeightsOffset + topkWeightsBytes)
    {
    }
};

// State retained from forward until the matching backward completes.
struct TrainingSavedStateLayout {
    uint64_t ptrAOffset;
    uint64_t ptrABytes;
    uint64_t cumsumOffset;
    uint64_t expandedRowIdxOffset;
    uint64_t expandedRowIdxGatherOffset;
    uint64_t sumBeforeRankOffset;
    uint64_t savedTokenCountsOffset;
    uint64_t dispatchedTopkWeightsOffset;
    uint64_t dispatchedTopkWeightsBytes;
    // Size of routing metadata before optional saved gate/up planes.
    uint64_t legacyBytes;
    uint64_t gateOffset;
    uint64_t upOffset;
    uint64_t gateUpPlaneBytes;
    uint64_t gateScaleOffset;
    uint64_t upScaleOffset;
    uint64_t gateUpScaleBytes;
    bool gateUpMxFp8;
    bool gateUpSaved;
    uint64_t gateHighOffset;
    uint64_t upLowOffset;
    uint64_t upHighOffset;
    uint64_t clipMaskPlaneBytes;
    uint32_t clipMaskKind;
    uint32_t layoutVersion;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr TrainingSavedStateLayout(uint64_t m, uint64_t k, uint64_t topk,
                                                                 uint64_t worldSize, uint64_t expertPerRank,
                                                                 uint64_t maxRecvTokens, uint64_t n, bool saveMxFp8,
                                                                 bool saveGateUp = true, bool saveClipMasks = false,
                                                                 bool embedDispatchOutput = true,
                                                                 uint64_t weightBytes = kBf16Bytes)
        : ptrAOffset(0),
          ptrABytes(embedDispatchOutput ? AlignUp(maxRecvTokens * k * kBf16Bytes, kAlignment512) : 0),
          cumsumOffset(ptrAOffset + ptrABytes),
          expandedRowIdxOffset(cumsumOffset + AlignUp(worldSize * expertPerRank * kInt32Bytes, kAlignment512)),
          expandedRowIdxGatherOffset(expandedRowIdxOffset + AlignUp(m, kAlignment256) * topk * kInt32Bytes),
          sumBeforeRankOffset(expandedRowIdxGatherOffset + AlignUp(m, kAlignment256) * topk * kInt32Bytes),
          savedTokenCountsOffset(sumBeforeRankOffset + AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes),
          dispatchedTopkWeightsOffset(
              savedTokenCountsOffset +
              AlignUp(worldSize * AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes, kAlignment512)),
          dispatchedTopkWeightsBytes(embedDispatchOutput ? AlignUp(maxRecvTokens * weightBytes, kAlignment512) : 0),
          legacyBytes(dispatchedTopkWeightsOffset + dispatchedTopkWeightsBytes),
          gateOffset(AlignUp(legacyBytes, kAlignment512)),
          upOffset(gateOffset +
                   (saveGateUp ? AlignUp(maxRecvTokens * (n / 2) * (saveMxFp8 ? 1 : kBf16Bytes), kAlignment512) : 0)),
          gateUpPlaneBytes(saveGateUp ? AlignUp(maxRecvTokens * (n / 2) * (saveMxFp8 ? 1 : kBf16Bytes), kAlignment512)
                                      : 0),
          gateScaleOffset(upOffset + gateUpPlaneBytes),
          upScaleOffset(gateScaleOffset +
                        (saveGateUp && saveMxFp8 ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32), kAlignment512) : 0)),
          gateUpScaleBytes(saveGateUp && saveMxFp8 ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32), kAlignment512) : 0),
          gateUpMxFp8(saveGateUp && saveMxFp8),
          gateUpSaved(saveGateUp),
          gateHighOffset(upScaleOffset + gateUpScaleBytes),
          upLowOffset(gateHighOffset + (saveGateUp && saveMxFp8 && saveClipMasks
                                            ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32) * kInt32Bytes, kAlignment512)
                                            : 0)),
          upHighOffset(upLowOffset + (upLowOffset - gateHighOffset)),
          clipMaskPlaneBytes(upLowOffset - gateHighOffset),
          clipMaskKind(clipMaskPlaneBytes ? kClipMaskThreeBitplanesV1 : kClipMaskNone),
          layoutVersion(clipMaskPlaneBytes ? 2 : 1),
          totalBytes(saveGateUp ? upHighOffset + clipMaskPlaneBytes : legacyBytes)
    {
    }
};

// Scratch used only by the current forward call. Host allocation is
// uninitialized; the forward kernel resets all control flags per invocation.
struct ForwardTransientWorkspaceLayout {
    uint64_t ptrA2Offset;
    uint64_t swigluToGmm2FlagOffset;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardTransientWorkspaceLayout(uint64_t n, uint64_t expertPerRank,
                                                                        uint64_t maxRecvTokens)
        : ptrA2Offset(0),
          swigluToGmm2FlagOffset(ptrA2Offset + AlignUp(maxRecvTokens * (n / 2) * kBf16Bytes, kAlignment512)),
          totalBytes(swigluToGmm2FlagOffset + AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128))
    {
    }
};

// Saved Wave and row-ready recompute allocate a counter for each canonical M block.
MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t BackwardDispatchSplitExperts(uint32_t mode, uint64_t f,
                                                                          uint32_t localExperts)
{
    return UseBackwardStagedWave(mode, f) || UseRecomputeDhpRowReady(mode, f) ? localExperts : 0U;
}

}  // namespace MegaMoeBF16Impl

#undef MEGA_MOE_BF16_HOST_DEVICE
