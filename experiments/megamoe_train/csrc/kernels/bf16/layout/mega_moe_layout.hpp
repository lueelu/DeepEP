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

#ifndef DEEPEP_BF16_FORCE_EXPERT_READY
#define DEEPEP_BF16_FORCE_EXPERT_READY 0
#endif
static_assert(DEEPEP_BF16_FORCE_EXPERT_READY == 0 || DEEPEP_BF16_FORCE_EXPERT_READY == 1,
              "DEEPEP_BF16_FORCE_EXPERT_READY must be 0 or 1");

namespace MegaMoeBF16Training {

constexpr uint64_t kAlignment32 = 32;
constexpr uint64_t kAlignment128 = 128;
constexpr uint64_t kAlignment256 = 256;
constexpr uint64_t kAlignment512 = 512;
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
// CalcPeermemSize still includes the common 60 KiB legacy header; ordinary
// BF16 Host allocation adds the difference to the fixed data offset below.
constexpr uint64_t kPeerBaseDataOffset = 60 * 1024;
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

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t PeerSyncCountOffset(uint64_t worldSize)
{
    (void)worldSize;
    return kPeerSyncCountOffset;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t PeerDataOffset(uint64_t worldSize)
{
    (void)worldSize;
    return kPeerDataOffset;
}

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

MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseCompactTokenCountExchangeByDefault(uint64_t worldSize,
                                                                               uint64_t expertPerRank)
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

// Host normalizes the public swiglu_limit=0 to FLT_MAX in tiling.
enum SavedGateUpMode : uint32_t { kSaveAuto = 0, kSaveNone = 1, kSaveBf16 = 2, kSaveMxFp8 = 3 };

MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t ResolveSavedGateUpMode(uint32_t mode, float tilingLimit)
{
    return mode == kSaveAuto ? (tilingLimit == FLT_MAX ? kSaveMxFp8 : kSaveBf16) : mode;
}

// A complete row uses 6F bytes of BF16 inputs, 4F bytes of FP32 products
// and a padded scalar result. Wider rows keep the tiled partial path.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseWholeRowTopkGrad(uint32_t mode, uint64_t f)
{
    return mode == kSaveBf16 && f > 0 && f <= 16384 && f % 64 == 0;
}
// CANN's FP32 contiguous AR sum first selects an A tile by core occupancy,
// then splits R to fit its four input buffers. The rank's complete routed
// row count is needed: one-row AR can have a different addition tree.
MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t TopkGradReductionWidth(uint64_t rows, uint32_t columns, uint32_t cores)
{
    if (rows == 0 || columns == 0 || cores == 0) return columns;
    constexpr uint64_t elements = (248U * 1024U - 16U * 1024U - 2U * 512U) / 16U;
    const uint64_t rTasks = (columns + 63U) / 64U;
    uint64_t best = 0, aTile = 1;
    for (uint64_t candidate = 1; candidate <= rows && candidate <= 128U; ++candidate) {
        const uint64_t outerA = (rows + candidate - 1U) / candidate;
        const uint64_t tasks = outerA * rTasks;
        uint64_t active = tasks / ((tasks + cores - 1U) / cores);
        if (active > outerA) active = active / outerA * outerA;
        if (active * 100U > cores * 85U && active > best) aTile = candidate;
        best = active > best ? active : best;
    }
    if (aTile * columns <= elements) return columns;
    uint32_t width = static_cast<uint32_t>(elements / aTile) / 8U * 8U;
    while (width > 1U && columns % width != 0U) --width;
    return width;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t BackwardTopkPartialCount(uint32_t mode, uint64_t f)
{
    return UseWholeRowTopkGrad(mode, f) ? 1U : (f + 255U) / 256U;
}
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseMappedBackwardInputs(uint32_t mode, uint64_t f, uint32_t groups)
{
    return groups != 0 && UseWholeRowTopkGrad(mode, f);
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseGmForBackwardDataGrad(uint32_t savedMode)
{
    return savedMode != kSaveNone;
}

// The caller supplies the effective (capacity-clipped) local expert rows.
// A single M tile has no inter-row-tile pipeline to overlap with Dispatch.
// ordinal still counts every non-empty expert; a fallback does not promote
// a later expert into the first k selected experts.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseBackwardDispatchRowReady(uint32_t savedMode, uint32_t ordinal,
                                                                     uint32_t splitExperts, uint64_t effectiveRows,
                                                                     uint64_t computeRows)
{
    return UseGmForBackwardDataGrad(savedMode) && ordinal < splitExperts && effectiveRows > computeRows;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool SaveGateUpAsMxFp8(float tilingLimit, uint32_t mode = kSaveAuto)
{
    return ResolveSavedGateUpMode(mode, tilingLimit) == kSaveMxFp8;
}

// v2 appends three uint32 bit planes only for explicit MXFP8 with clipping.
// All v1 prefix/data/scale offsets are retained. No GM header or initialization.
enum SavedClipMaskKind : uint32_t { kClipMaskNone = 0, kClipMaskThreeBitplanesV1 = 1 };
MEGA_MOE_BF16_HOST_DEVICE constexpr bool SaveGateUpClipMasks(float tilingLimit, uint32_t mode)
{
    return SaveGateUpAsMxFp8(tilingLimit, mode) && tilingLimit != FLT_MAX;
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
    uint64_t dispatchToGmm1FlagOffset;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardTransientWorkspaceLayout(uint64_t n, uint64_t expertPerRank,
                                                                        uint64_t maxRecvTokens)
        : ptrA2Offset(0),
          swigluToGmm2FlagOffset(ptrA2Offset + AlignUp(maxRecvTokens * (n / 2) * kBf16Bytes, kAlignment512)),
          dispatchToGmm1FlagOffset(swigluToGmm2FlagOffset +
                                   AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128)),
          totalBytes(dispatchToGmm1FlagOffset + AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128))
    {
    }
};

// Backward-only per-producer row-progress suffix; preceding workspace and
// saved-state offsets remain fixed. Host allocation zeros these counters
// before backward Dispatch publishes readiness. These are synchronization
// counters used for dispatch readiness.
struct BackwardDispatchProgressLayout {
    static constexpr uint64_t kSlotBytes = 64;
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint64_t ExtraBytes(uint64_t experts, uint64_t sources,
                                                                   uint64_t producersPerSource, uint32_t savedMode)
    {
        return UseGmForBackwardDataGrad(savedMode) ? experts * sources * producersPerSource * kSlotBytes : 0;
    }
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint64_t TotalBytes(uint64_t legacyBytes, uint64_t experts,
                                                                   uint64_t sources, uint64_t producersPerSource,
                                                                   uint32_t savedMode)
    {
        return UseGmForBackwardDataGrad(savedMode) && experts != 0
                   ? AlignUp(legacyBytes, kSlotBytes) + ExtraBytes(experts, sources, producersPerSource, savedMode)
                   : legacyBytes;
    }
};

// Backward uses row readiness for up to seven local experts; larger expert counts use whole-expert readiness.
MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t DefaultBackwardDispatchSplitExperts(uint32_t savedMode,
                                                                                 uint32_t localExperts)
{
    return UseGmForBackwardDataGrad(savedMode) && localExperts <= 7U ? localExperts : 0U;
}

}  // namespace MegaMoeBF16Training

#undef MEGA_MOE_BF16_HOST_DEVICE
