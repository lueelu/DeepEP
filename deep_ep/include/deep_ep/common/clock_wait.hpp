// Copyright (c) 2026, Lu Lu

// Host/device ABI of the in-graph wait record table. Kept free of device
// headers so the Host launcher and the device body share one layout.
#pragma once

#include <cstdint>

namespace deep_ep {

// Bumped whenever the record layout changes; a wheel with a different value
// must not be mixed with a Host that allocates the newer storage.
constexpr uint32_t kClockWaitRecordSchema = 2U;

// One record per (call index, logical core) carries four meaningful fields but
// occupies a 128-byte physical slot. With the original 32-byte slot, adjacent
// cores on one cache line overwrote each other and only 28 of 56 records
// survived a 100-call graph. The physical stride is therefore part of the ABI.
constexpr uint32_t kClockWaitRecordWords = 4U;          // valid fields per record
constexpr uint32_t kClockWaitRecordStrideWords = 16U;   // 128-byte physical slot
constexpr uint32_t kClockWaitRecordStrideBytes = kClockWaitRecordStrideWords * sizeof(uint64_t);
constexpr uint32_t kClockWaitRecordAlignmentBytes = kClockWaitRecordStrideBytes;

constexpr uint32_t kClockWaitRecordEnter = 0U;
constexpr uint32_t kClockWaitRecordRelease = 1U;
constexpr uint32_t kClockWaitRecordBatch = 2U;
constexpr uint32_t kClockWaitRecordCore = 3U;

// Serialized argument block of the wait kernel. The device struct must match;
// the Host never builds it, but callers can verify the ABI they launched with.
constexpr uint32_t kClockWaitArgBytes = 40U;

static_assert(kClockWaitRecordStrideWords >= kClockWaitRecordWords,
              "the physical record slot cannot be narrower than its fields");
static_assert(kClockWaitRecordStrideBytes == 128U, "the admitted record slot is 128 bytes");

}  // namespace deep_ep
