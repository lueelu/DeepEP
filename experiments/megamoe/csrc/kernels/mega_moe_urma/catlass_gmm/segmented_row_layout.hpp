// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>

// The arithmetic below is shared with the CPU layout test. It deliberately
// contains no Ascend or URMA types.
namespace UrmaMegaMoeImpl::GmmKernel::SegmentedRows {

#if defined(__NPU_DEVICE__)
#define MEGAMOE_SEGMENTED_INLINE __aicore__ inline
#else
#define MEGAMOE_SEGMENTED_INLINE inline
#endif

struct Overlap {
    uint32_t tileRow;
    uint32_t segmentRow;
    uint32_t rows;
};

MEGAMOE_SEGMENTED_INLINE constexpr Overlap Intersect(uint32_t segmentBegin, uint32_t segmentRows, uint32_t tileBegin,
                                                     uint32_t tileRows)
{
    const uint32_t begin = segmentBegin > tileBegin ? segmentBegin : tileBegin;
    const uint32_t segmentEnd = segmentBegin + segmentRows;
    const uint32_t tileEnd = tileBegin + tileRows;
    const uint32_t end = segmentEnd < tileEnd ? segmentEnd : tileEnd;
    return end > begin ? Overlap{begin - tileBegin, begin - segmentBegin, end - begin} : Overlap{0, 0, 0};
}

MEGAMOE_SEGMENTED_INLINE constexpr uint32_t RoundUp16(uint32_t value)
{
    return (value + 15U) / 16U * 16U;
}

// ND2NZ writes the exact N count and pads only its D dimension to C0.
MEGAMOE_SEGMENTED_INLINE constexpr uint64_t AOffset(uint32_t row, uint32_t column, uint32_t tileRows)
{
    return static_cast<uint64_t>(column / 32U) * RoundUp16(tileRows) * 32U + static_cast<uint64_t>(row) * 32U +
           column % 32U;
}

MEGAMOE_SEGMENTED_INLINE constexpr uint64_t ScaleOffset(uint32_t row, uint32_t column, uint32_t scaleColumns)
{
    return static_cast<uint64_t>(row / 16U) * ((scaleColumns + 1U) / 2U * 2U) * 16U +
           static_cast<uint64_t>(column / 2U) * 32U + (row % 16U) * 2U + column % 2U;
}

MEGAMOE_SEGMENTED_INLINE constexpr uint64_t L0COffset(uint32_t row, uint32_t column, uint32_t tileRows)
{
    return static_cast<uint64_t>(column / 16U) * RoundUp16(tileRows) * 16U + static_cast<uint64_t>(row) * 16U +
           column % 16U;
}

MEGAMOE_SEGMENTED_INLINE constexpr uint64_t OutputOffset(uint64_t row, uint32_t column, uint32_t stride)
{
    return row * stride + column;
}

}  // namespace UrmaMegaMoeImpl::GmmKernel::SegmentedRows

#undef MEGAMOE_SEGMENTED_INLINE
