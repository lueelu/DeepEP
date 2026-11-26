// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include <cstdint>
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include "kernel_operator.h"
#define MEGA_MOE_WAVE_MATH_HD __host_aicore__
#else
#define MEGA_MOE_WAVE_MATH_HD
#endif
namespace MegaMoeImpl {
// Budget is measured in tasks per AIC. Callers provide payload-specific tile
// counts (paired Gate/Up or quantized output), not a fixed 256-column assumption.
MEGA_MOE_WAVE_MATH_HD constexpr uint32_t WaveGroups(uint32_t cores, uint64_t gmm1Tiles, uint64_t gmm2Tiles,
                                                    uint32_t gmm1Budget, bool constrainGmm2 = true)
{
    if (!cores || !gmm1Tiles || (constrainGmm2 && !gmm2Tiles)) return 1U;
    const uint64_t first = (uint64_t(cores) * gmm1Budget + gmm1Tiles - 1U) / gmm1Tiles;
    const uint64_t second = constrainGmm2 ? (uint64_t(cores) + gmm2Tiles - 1U) / gmm2Tiles : 1U;
    return static_cast<uint32_t>(first > second ? first : second);
}

constexpr uint32_t kWaveScratchSlots = 8U;

// Keep a dense layout when it fits. Wave tails can span several expert slices.
MEGA_MOE_WAVE_MATH_HD constexpr uint64_t WaveScratchRows(uint64_t capacity, uint32_t rowsPerWave)
{
    const uint64_t ring = uint64_t(kWaveScratchSlots) * rowsPerWave;
    return ring && ring < capacity ? ring : capacity;
}

MEGA_MOE_WAVE_MATH_HD constexpr uint64_t WaveScratchRow(uint32_t wave, uint64_t first, uint64_t packed,
                                                        uint32_t slotRows)
{
    return slotRows ? uint64_t(wave % kWaveScratchSlots) * slotRows + (packed - first) : packed;
}

struct WorkRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

// Reverse only the logical owner; the physical AIC/AIV pairing stays fixed.
MEGA_MOE_WAVE_MATH_HD constexpr uint32_t GetGmmFirstOwnedTile(uint32_t physicalBlockIdx, uint32_t blockNum,
                                                              uint32_t startBlockIdx, bool reverseBlocks = false)
{
    const uint32_t logicalBlockIdx = reverseBlocks ? blockNum - 1U - physicalBlockIdx : physicalBlockIdx;
    return (logicalBlockIdx < startBlockIdx ? logicalBlockIdx + blockNum : logicalBlockIdx) - startBlockIdx;
}

MEGA_MOE_WAVE_MATH_HD constexpr uint32_t GetGmmNextStartBlock(uint32_t startBlockIdx, uint32_t tiles, uint32_t blockNum)
{
    return (startBlockIdx + tiles) % blockNum;
}

// Each GMM2 output tile belongs to one AIC. Only AICs with tiles signal Combine.
template <uint32_t TileM, uint32_t TileN>
MEGA_MOE_WAVE_MATH_HD constexpr uint32_t GetGmm2SliceTiles(uint32_t rows, uint32_t columns)
{
    static_assert(TileM > 0 && TileN > 0);
    const uint32_t mTiles = rows / TileM + (rows % TileM != 0);
    const uint32_t nTiles = columns / TileN + (columns % TileN != 0);
    return mTiles * nTiles;
}

MEGA_MOE_WAVE_MATH_HD constexpr uint32_t GetGmm2ProducerCount(uint32_t tiles, uint32_t cores)
{
    return tiles < cores ? tiles : cores;
}

// Replay the producer assignment for one slice, even when this AIC owns no tiles.
MEGA_MOE_WAVE_MATH_HD constexpr bool AdvanceGmmSlice(uint32_t blockIdx, uint32_t blockNum, uint32_t& startBlock,
                                                     uint32_t tiles, bool reverseBlocks = false)
{
    const uint32_t first = GetGmmFirstOwnedTile(blockIdx, blockNum, startBlock, reverseBlocks);
    startBlock = GetGmmNextStartBlock(startBlock, tiles, blockNum);
    return first < tiles;
}

// Give the first remainder workers one extra row.
MEGA_MOE_WAVE_MATH_HD constexpr WorkRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx,
                                                                uint32_t workerCount)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t base = totalTokens / workerCount;
    uint32_t remainder = totalTokens % workerCount;
    uint32_t extraBefore = workerIdx < remainder ? workerIdx : remainder;
    return {workerIdx * base + extraBefore, base + static_cast<uint32_t>(workerIdx < remainder)};
}

// Rotate the owner of the first extra row by the absolute packed-row prefix.
MEGA_MOE_WAVE_MATH_HD constexpr WorkRange GetRotatedBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx,
                                                                       uint32_t workerCount, uint64_t globalRowPrefix)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t firstOwner = static_cast<uint32_t>(globalRowPrefix % workerCount);
    uint32_t logicalWorkerIdx = workerIdx >= firstOwner ? workerIdx - firstOwner : workerIdx + workerCount - firstOwner;
    return GetBalancedTokenRange(totalTokens, logicalWorkerIdx, workerCount);
}
}  // namespace MegaMoeImpl
#undef MEGA_MOE_WAVE_MATH_HD
