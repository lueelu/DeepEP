// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include <cstdint>
#include "csrc/kernels/common/mega_moe_wave_math.hpp"
#include "../../../../../../csrc/kernels/common/mega_moe_peer_rows.hpp"
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include "kernel_operator.h"
#define MEGA_MOE_WAVE_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_WAVE_HOST_DEVICE
#endif

namespace MegaMoeBF16Impl::Wave {

constexpr uint32_t kGmm1Rows = 256;
constexpr uint32_t kGmm2Rows = 256;
constexpr uint32_t kGroupRows = 256;
constexpr uint32_t kTileColumns = 256;
constexpr uint32_t kPairedColumns = kTileColumns / 2U;
static_assert(kGroupRows % kGmm1Rows == 0 && kGroupRows % kGmm2Rows == 0);
constexpr uint32_t kDispatchRowsPerSlot = 1;
constexpr uint32_t kWeightUbBytes = 4096;
constexpr uint32_t kMetadataUbBytes = 4096;

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t DispatchUbBase(uint32_t hidden)
{
    // Two complete BF16 Combine rows precede the Dispatch ping-pong buffers.
    return 4U * ((hidden + 31U) / 32U * 32U);
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t GroupsPerWave(uint32_t cores, uint32_t f, uint32_t h)
{
    const uint64_t gmm1Tiles = (static_cast<uint64_t>(f) + kPairedColumns - 1U) / kPairedColumns;
    const uint64_t gmm2Tiles = (static_cast<uint64_t>(h) + kTileColumns - 1U) / kTileColumns;
    return MegaMoeImpl::WaveGroups(cores, gmm1Tiles, gmm2Tiles, 2U);
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t MetadataBytes(uint32_t world, uint32_t experts)
{
    const uint64_t bytes = 2ULL * ((static_cast<uint64_t>(world) * experts * 4U + 31U) / 32U * 32U);
    return bytes > kMetadataUbBytes ? static_cast<uint32_t>(bytes) : kMetadataUbBytes;
}
MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t MetadataUbBase(uint32_t world, uint32_t experts)
{
    return 248U * 1024U - MetadataBytes(world, experts);
}
MEGA_MOE_WAVE_HOST_DEVICE constexpr bool ScratchFits(uint32_t hidden, uint32_t world = 0, uint32_t experts = 0)
{
    // Combine and Dispatch each own two BF16 rows; their buffers coexist on AIV1.
    return 8ULL * ((hidden + 31U) / 32U * 32U) + kWeightUbBytes + MetadataBytes(world, experts) <= 248ULL * 1024ULL;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr bool MetadataFits(uint32_t world, uint32_t experts)
{
    return MetadataBytes(world, experts) <= 248U * 1024U;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t ReadySlot(uint64_t expertBase, uint32_t expert, uint32_t row)
{
    return expertBase / kGmm1Rows + expert + row / kGmm1Rows;
}

constexpr uint32_t kMaxCores = 36;
MEGA_MOE_WAVE_HOST_DEVICE constexpr bool UseGmm2Lag(uint32_t sourceRows)
{
    return sourceRows >= 512;
}

// Dispatch X and GMM2 output share rows. Interleaved Hp follows those rows;
// staged Hp must survive the entire GMM1 pass.
struct ForwardScratch {
    uint64_t rows, hpRows;
    uint32_t slotRows;
    MEGA_MOE_WAVE_HOST_DEVICE constexpr ForwardScratch(uint64_t capacity, uint32_t groups, uint32_t schedule)
        : rows(MegaMoeImpl::WaveScratchRows(capacity, groups * kGroupRows)),
          hpRows(schedule ? capacity : rows),
          slotRows(rows < capacity ? groups * kGroupRows : 0U)
    {
    }
    MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t Row(uint32_t wave, uint64_t first, uint64_t packed) const
    {
        return MegaMoeImpl::WaveScratchRow(wave, first, packed, slotRows);
    }
    MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t HpRow(uint64_t physical, uint64_t logical) const
    {
        return hpRows == rows ? physical : logical;
    }
};

// Per-slice Hp/Dispatch/output counters followed by per-core issued words.
struct ReadyLayout {
    uint64_t words, dispatchOffset, downReadyOffset, issuedOffset, releaseOffset, inputDoneOffset, outputOffset;
    MEGA_MOE_WAVE_HOST_DEVICE constexpr ReadyLayout(uint64_t capacity, uint64_t experts, bool ring = false)
        : words((capacity + kGmm1Rows - 1U) / kGmm1Rows + experts),
          dispatchOffset((words * sizeof(int32_t) + 127U) / 128U * 128U),
          downReadyOffset(dispatchOffset + (words * sizeof(int32_t) + 127U) / 128U * 128U),
          issuedOffset(downReadyOffset + (words * sizeof(int32_t) + 127U) / 128U * 128U),
          releaseOffset(issuedOffset + kMaxCores * 16U * sizeof(int32_t)),
          inputDoneOffset(releaseOffset + (ring ? MegaMoeImpl::kWaveScratchSlots * 16U * sizeof(int32_t) : 0U)),
          outputOffset((inputDoneOffset + (ring ? sizeof(int32_t) : 0U) + 511U) / 512U * 512U)
    {
    }
};

// Unpermute writes Y after Combine drains. A slice starting before the boundary
// may extend by maxSliceRows-1. Leave one row (at least 512 B) for aligning Y views.
MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t OutputRowsInY(uint64_t capacity, uint64_t sourceRows, uint32_t groups)
{
    if (groups == 0 || sourceRows == 0) return 0;
    if (capacity < sourceRows) return capacity;
    const uint64_t maxSliceRows = uint64_t(groups) * kGroupRows;
    return sourceRows > maxSliceRows ? sourceRows - maxSliceRows : 0U;
}

// Each Cube/Vector pair owns two Gate/Up slots. Credits are returned after the vector consumes the tile.
constexpr uint32_t kRawStages = 2;
constexpr uint64_t kRawTileElements = uint64_t(kGmm1Rows) * kPairedColumns;
MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t RawTileOffset(uint32_t core, uint32_t sequence)
{
    return (core * kRawStages + sequence % kRawStages) * kRawTileElements;
}
struct ForwardWorkspaceLayout {
    uint64_t readyOffset, rowsInY, weightOffset, rawGateOffset, rawUpOffset, totalBytes;
    MEGA_MOE_WAVE_HOST_DEVICE constexpr ForwardWorkspaceLayout(uint64_t baseBytes, uint64_t capacity, uint64_t experts,
                                                               uint64_t hidden, uint32_t weightBytes,
                                                               uint64_t sourceRows, uint32_t groups,
                                                               uint64_t scratchRows)
        : readyOffset((baseBytes + 511U) / 512U * 512U),
          rowsInY(OutputRowsInY(scratchRows, sourceRows, groups)),
          weightOffset(readyOffset + ReadyLayout(capacity, experts, scratchRows < capacity).outputOffset +
                       (scratchRows - rowsInY) * hidden * sizeof(uint16_t)),
          rawGateOffset(weightOffset + (capacity * weightBytes + 511U) / 512U * 512U),
          rawUpOffset(rawGateOffset + RawTileOffset(kMaxCores, 0) * sizeof(uint16_t)),
          totalBytes(rawUpOffset + (rawUpOffset - rawGateOffset))
    {
    }
};

// AIC produces Gate/Up in the same tile order that its paired AIV consumes them.
template <class BlockScheduler, class Visit>
MEGA_MOE_WAVE_HOST_DEVICE inline void ForEachGateUpTile(uint32_t rows, uint32_t ffn, uint32_t hidden, uint32_t core,
                                                        uint32_t cores, uint32_t& startBlock, Visit visit)
{
    BlockScheduler scheduler({rows, ffn, hidden}, {kGmm1Rows, kPairedColumns});
    const uint32_t tasks = scheduler.GetCoreLoops();
    const uint32_t first = MegaMoeImpl::GetGmmFirstOwnedTile(core, cores, startBlock);
    for (uint32_t task = first; task < tasks; task += cores) {
        const auto coord = scheduler.GetBlockCoord(task);
        visit(coord.m() * kGmm1Rows, coord.n() * kPairedColumns, scheduler.GetActualBlockShape(coord));
    }
    startBlock = MegaMoeImpl::GetGmmNextStartBlock(startBlock, tasks, cores);
}

struct Cursor {
    uint32_t expert = 0;
    uint32_t row = 0;
    uint32_t slice = 0;
    uint64_t packed = 0;
    uint32_t wave = 0;
};

struct Slice {
    uint32_t expert;
    uint32_t row;
    uint32_t rows;
    uint32_t ordinal;
    uint32_t fullRows;
    uint64_t packed;
};

struct Range {
    Cursor begin;
    Cursor end;
    bool valid = false;
};

// Counts returns each expert's non-negative row count. Capacity clipping uses
// the same packed expert order as the ordinary forward and saved-state layout.
template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline bool Next(Cursor& cursor, uint32_t& groups, uint32_t experts, uint64_t capacity,
                                           const Counts& counts, Slice& slice)
{
    while (groups && cursor.expert < experts) {
        const uint64_t base = cursor.packed - cursor.row;
        const uint64_t available = base < capacity ? capacity - base : 0;
        const uint64_t raw = counts(cursor.expert);
        const uint32_t rows = static_cast<uint32_t>(raw < available ? raw : available);
        if (cursor.row >= rows) {
            ++cursor.expert;
            cursor.row = cursor.slice = 0;
            continue;
        }
        const uint64_t budget = static_cast<uint64_t>(groups) * kGroupRows;
        const uint32_t left = rows - cursor.row;
        const uint32_t take = left < budget ? left : static_cast<uint32_t>(budget);
        slice = {cursor.expert, cursor.row, take, ++cursor.slice, rows, cursor.packed};
        cursor.row += take;
        cursor.packed += take;
        groups -= (take + kGroupRows - 1U) / kGroupRows;
        if (cursor.row == rows) {
            ++cursor.expert;
            cursor.row = cursor.slice = 0;
        }
        return true;
    }
    return false;
}

template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline Range Plan(const Cursor& begin, uint32_t groups, uint32_t experts, uint64_t capacity,
                                            const Counts& counts)
{
    Range range{begin, begin, false};
    Slice slice{};
    while (Next(range.end, groups, experts, capacity, counts, slice)) range.valid = true;
    return range;
}

template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline uint32_t CountWaves(uint32_t groups, uint32_t experts, uint64_t capacity,
                                                     const Counts& counts)
{
    uint64_t totalGroups = 0;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const uint64_t raw = counts(expert);
        const uint64_t rows = raw < capacity ? raw : capacity;
        totalGroups += (rows + kGroupRows - 1U) / kGroupRows;
        capacity -= rows;
    }
    return static_cast<uint32_t>((totalGroups + groups - 1U) / groups);
}

// Rotate existing complete Wave jobs, including the expert tails and capacity-clipped final job.
// The physical row ranges are unchanged; wave counts execution order across the single wrap.
template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline Range PlanScheduled(const Cursor& begin, uint32_t groups, uint32_t experts,
                                                     uint64_t capacity, uint32_t waves, const Counts& counts)
{
    if (begin.wave >= waves) return {};
    auto range = Plan(begin, groups, experts, capacity, counts);
    if (!range.valid) range = Plan(Cursor{}, groups, experts, capacity, counts);
    range.begin.wave = begin.wave;
    range.end.wave = begin.wave + 1U;
    return range;
}

template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline bool NextScheduledSlice(Cursor& cursor, uint32_t& groups, uint32_t waveGroups,
                                                         uint32_t experts, uint64_t capacity, uint32_t waves,
                                                         const Counts& counts, Slice& slice)
{
    const bool newWave = groups == 0U;
    if (newWave) {
        ++cursor.wave;
        groups = waveGroups;
    }
    if (cursor.wave >= waves) return false;
    if (Next(cursor, groups, experts, capacity, counts, slice)) return true;
    // An expert/capacity tail may end the last physical Wave with unused group credits.
    const uint32_t wave = cursor.wave + static_cast<uint32_t>(!newWave);
    if (wave >= waves) return false;
    cursor = {};
    cursor.wave = wave;
    groups = waveGroups;
    return Next(cursor, groups, experts, capacity, counts, slice);
}

}  // namespace MegaMoeBF16Impl::Wave

#undef MEGA_MOE_WAVE_HOST_DEVICE
