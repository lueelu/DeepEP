#pragma once

#include <cstddef>
#include <cstdint>

#include "../launch.hpp"

namespace deepep {
namespace clock_alignment {

constexpr uint32_t kVersion = 1U;
constexpr uint32_t kSampleWords = 8U;
constexpr uint32_t kMaxSamples = 256U;
constexpr uint32_t kMaxWarmupSamples = 32U;
constexpr uint64_t kCyclesPerSecond = 1000000000ULL;

// The LL workspace reserves the first 2 MiB of each window for controls.
// Count TX ends below 1.29 MiB; clock calibration owns this disjoint 48 KiB.
constexpr uint64_t kRegionOffset = 1536ULL * 1024U;
constexpr uint64_t kSlotBytes = 64U;
constexpr uint64_t kRequestBase = kRegionOffset;
constexpr uint64_t kResponseBase = kRequestBase + 256U * kSlotBytes;
constexpr uint64_t kTimestampBase = kResponseBase + 256U * kSlotBytes;
constexpr uint64_t kRegionEnd = kTimestampBase + 256U * kSlotBytes;
static_assert(kRegionEnd < 2ULL * 1024U * 1024U, "clock controls must remain below LL payload");

enum class ClockState : uint64_t {
    Ready = 1U,
    Request = 2U,
    Response = 3U,
    Timestamp = 4U,
    Ack = 5U,
};

enum class SampleStatus : uint32_t {
    Success = 0U,
    InvalidTiming = 1U,
    Timeout = 2U,
    InvalidConfiguration = 3U,
};

struct alignas(32) Protocol {
    uint64_t magic = 0U;
    uint64_t sequence = 0U;
    uint64_t state = 0U;
    uint64_t t1 = 0U;
    uint64_t t2 = 0U;
    uint64_t reserved0 = 0U;
    uint64_t reserved1 = 0U;
    uint64_t reserved2 = 0U;
};
static_assert(sizeof(Protocol) == kSlotBytes, "clock protocol must occupy one slot");

struct alignas(32) Sample {
    uint64_t t0 = 0U;
    uint64_t t1 = 0U;
    uint64_t t2 = 0U;
    uint64_t t3 = 0U;
    uint64_t roundTripCycles = 0U;
    int64_t offsetCycles = 0;
    uint32_t sequence = 0U;
    uint32_t status = uint32_t(SampleStatus::InvalidConfiguration);
    uint64_t reserved = 0U;
};
static_assert(sizeof(Sample) == kSampleWords * sizeof(uint64_t), "clock sample ABI changed");

}  // namespace clock_alignment
}  // namespace deepep

extern "C" {
DEEPEP_EXPORT void deepep_clock_sync_kernel_do(
    void* stream, uint8_t* comm_args, uint8_t* samples, uint32_t sample_count,
    uint32_t warmup_sample_count, uint64_t magic, uint64_t timeout_cycles);
DEEPEP_EXPORT void deepep_clock_sleep_at_kernel_do(
    void* stream, uint64_t local_deadline, uint8_t* release_cycles);
}
