// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_urma/abi.hpp"
#include "kernels/mega_moe_urma/catlass_gmm/segmented_row_layout.hpp"

namespace MegaMoeUrma {
constexpr uint32_t kReorderRows = 128U;
constexpr uint32_t kReorderColumns = 256U;
constexpr uint32_t kReorderSlots = 2U;
constexpr uint32_t kReorderSlotBytes = kReorderRows * kReorderColumns * sizeof(uint16_t);
constexpr uint32_t kReorderUbBytes = kReorderSlots * kReorderSlotBytes;

struct ReorderSpan {
    bool self;
    uint32_t scratchRow;
    uint32_t rows;
    uint64_t outputRow;
};

#if defined(__NPU_DEVICE__)
#define URMA_REORDER_INLINE __aicore__ inline
#else
#define URMA_REORDER_INLINE inline
#endif

URMA_REORDER_INLINE constexpr ReorderSpan PlanReorderSpan(const RowSegment& segment, uint32_t rank, uint32_t sliceBegin,
                                                          uint32_t sliceRows)
{
    const auto part =
        UbxMegaMoeImpl::GmmKernel::SegmentedRows::Intersect(segment.tileRow, segment.rows, sliceBegin, sliceRows);
    const bool self = segment.sourceRank == rank;
    return {self, part.tileRow, part.rows, (self ? segment.returnRow : segment.receiveRow) + part.segmentRow};
}

#undef URMA_REORDER_INLINE
}  // namespace MegaMoeUrma
