#pragma once

#include <cstdint>

namespace ascend_deepep::udma {
// Independent of the reusable payload workspace. One cache-line-isolated slot
// per publishing rank; the first slot holds this Buffer's device-side epoch.
inline constexpr uint64_t kOutputSlotBytes = 512;
inline constexpr uint32_t kOutputRecordBytes = 96;
inline constexpr uint32_t kOutputRecordWords = kOutputRecordBytes / sizeof(uint64_t);
inline constexpr uint32_t kOutputEpochWord = 5;
inline constexpr uint32_t kOutputSrcWord = 6;
inline constexpr uint32_t kOutputWeightWord = 7;
inline constexpr uint32_t kOutputTopkWord = 8;
static_assert(kOutputTopkWord < kOutputRecordWords);
static_assert(kOutputRecordBytes % 32 == 0 && kOutputRecordBytes <= kOutputSlotBytes);
#if defined(__CCE_AICORE__)
#define ASCEND_DEEPEP_OUTPUT_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_OUTPUT_FN inline constexpr
#endif
// Pure arithmetic shared by scalar AIV and SIMT VF callers. SIMT permits
// constexpr __aicore__ helpers, but cannot call an ordinary __aicore__ helper.
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline constexpr bool ValidMetadataOutput(uint64_t base, uint32_t rows, uint32_t row_bytes)
{
    if (!row_bytes) return base == 0;
    return (!rows || base != 0) && base <= UINT64_MAX - uint64_t(rows) * row_bytes;
}
ASCEND_DEEPEP_OUTPUT_FN uint64_t OutputControlBytes(uint32_t world)
{
    return (uint64_t(world) + 1U) * kOutputSlotBytes;
}
ASCEND_DEEPEP_OUTPUT_FN uint64_t OutputRecordOffset(uint32_t rank)
{
    return (uint64_t(rank) + 1U) * kOutputSlotBytes;
}
} // namespace ascend_deepep::udma
#undef ASCEND_DEEPEP_OUTPUT_FN
