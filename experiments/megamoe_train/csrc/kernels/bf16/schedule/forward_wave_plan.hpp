// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include <cstdint>
#include "../../../../../../csrc/kernels/common/mega_moe_peer_rows.hpp"
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include "kernel_operator.h"
#define MEGA_MOE_WAVE_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_WAVE_HOST_DEVICE
#endif

namespace MegaMoeBF16Training::Wave {

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
    if (!cores || !gmm1Tiles || !gmm2Tiles) return 1U;
    const uint64_t first = (2ULL * cores + gmm1Tiles - 1U) / gmm1Tiles;
    const uint64_t second = (static_cast<uint64_t>(cores) + gmm2Tiles - 1U) / gmm2Tiles;
    return static_cast<uint32_t>(first > second ? first : second);
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

// Per-slice Hp/Dispatch/output counters, per-core issued words, then unique
// packed GMM2 rows. No output slot can be overwritten by a later Wave.
struct ReadyLayout {
    uint64_t words, dispatchOffset, downReadyOffset, issuedOffset, outputOffset, bytes;
    MEGA_MOE_WAVE_HOST_DEVICE constexpr ReadyLayout(uint64_t capacity, uint64_t experts, uint64_t hidden)
        : words((capacity + kGmm1Rows - 1U) / kGmm1Rows + experts),
          dispatchOffset((words * sizeof(int32_t) + 127U) / 128U * 128U),
          downReadyOffset(dispatchOffset + (words * sizeof(int32_t) + 127U) / 128U * 128U),
          issuedOffset(downReadyOffset + (words * sizeof(int32_t) + 127U) / 128U * 128U),
          outputOffset((issuedOffset + kMaxCores * 16U * sizeof(int32_t) + 511U) / 512U * 512U),
          bytes(outputOffset + capacity * hidden * sizeof(uint16_t))
    {
    }
};

// BF16 saves own raw Gate/Up. Other save modes reuse Hp for raw Gate and
// append raw Up after the complete output/control region; it cannot alias
// any in-flight GMM2 or Combine payload.
struct ForwardWorkspaceLayout {
    uint64_t readyOffset, rawUpOffset, rawUpBytes, totalBytes;
    MEGA_MOE_WAVE_HOST_DEVICE constexpr ForwardWorkspaceLayout(uint64_t baseBytes, uint64_t capacity, uint64_t experts,
                                                               uint64_t hidden, uint64_t ffn, bool savedBf16)
        : readyOffset(baseBytes),
          rawUpOffset(baseBytes + ReadyLayout(capacity, experts, hidden).bytes),
          rawUpBytes(savedBf16 ? 0U : (capacity * ffn * sizeof(uint16_t) + 511U) / 512U * 512U),
          totalBytes(rawUpOffset + rawUpBytes)
    {
    }
};

struct ProducerRange {
    uint32_t first;
    uint32_t last;
};

// Cover a non-empty consumer interval with producer tiles.
MEGA_MOE_WAVE_HOST_DEVICE constexpr ProducerRange ReadyProducers(uint32_t row, uint32_t rows, uint32_t producerRows)
{
    return {row / producerRows, (row + rows - 1U) / producerRows};
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

}  // namespace MegaMoeBF16Training::Wave

#undef MEGA_MOE_WAVE_HOST_DEVICE
