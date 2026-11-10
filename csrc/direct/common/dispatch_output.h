#pragma once

#include <cstdint>

namespace ascend_deepep::udma {
// Independent of the reusable payload workspace. One cache-line-isolated slot
// per publishing rank; the first slot holds this Buffer's device-side epoch.
inline constexpr uint64_t kOutputSlotBytes = 512;
inline constexpr uint32_t kOutputRecordBytes = 64;
inline constexpr uint32_t kOutputRecordWords = kOutputRecordBytes / sizeof(uint64_t);
inline constexpr uint32_t kOutputEpochWord = 5;
#if defined(__CCE_AICORE__)
#define ASCEND_DEEPEP_OUTPUT_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_OUTPUT_FN inline constexpr
#endif
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
