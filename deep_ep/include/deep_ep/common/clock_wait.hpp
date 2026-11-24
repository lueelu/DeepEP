// Copyright (c) 2026, Lu Lu

// Host/device ABI of the in-graph wait record table. Kept free of device
// headers so the Host launcher and the device body share one layout.
#pragma once

#include <cstdint>

namespace deep_ep {

// One record per (call index, logical core): enter, release, batch, core.
constexpr uint32_t kClockWaitRecordWords = 4U;
constexpr uint32_t kClockWaitRecordEnter = 0U;
constexpr uint32_t kClockWaitRecordRelease = 1U;
constexpr uint32_t kClockWaitRecordBatch = 2U;
constexpr uint32_t kClockWaitRecordCore = 3U;

}  // namespace deep_ep
