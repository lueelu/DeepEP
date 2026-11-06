// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#pragma once

#include <cstdint>

namespace MegaMoeMeta {

constexpr uint64_t kMetaAlignment = 512;
constexpr uint64_t kTokenCountStrideAlignment = 128;
constexpr uint64_t kRoutingRowAlignment = 256;

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// Byte layout returned as the fifth BF16 forward output:
//   [all-rank token counters | cumsumMM | expandedRowIdxGather |
//    sumBeforeRank]
// Every segment starts on a 512-byte boundary so backward/debug consumers can
// reinterpret the uint8 tensor without depending on the internal workspace.
struct SavedMetaLayout {
    uint64_t countsOffset;
    uint64_t countsBytes;
    uint64_t cumsumOffset;
    uint64_t cumsumBytes;
    uint64_t rowIdxGatherOffset;
    uint64_t rowIdxGatherBytes;
    uint64_t sumBeforeRankOffset;
    uint64_t sumBeforeRankBytes;
    uint64_t totalBytes;

    constexpr SavedMetaLayout(uint64_t m, uint64_t topK, uint64_t worldSize, uint64_t expertPerRank)
        : countsOffset(0),
          countsBytes(
              AlignUp(worldSize * AlignUp(worldSize * expertPerRank, kTokenCountStrideAlignment) * sizeof(int32_t),
                      kMetaAlignment)),
          cumsumOffset(countsOffset + countsBytes),
          cumsumBytes(AlignUp(worldSize * expertPerRank * sizeof(int32_t), kMetaAlignment)),
          rowIdxGatherOffset(cumsumOffset + cumsumBytes),
          rowIdxGatherBytes(AlignUp(AlignUp(m, kRoutingRowAlignment) * topK * sizeof(int32_t), kMetaAlignment)),
          sumBeforeRankOffset(rowIdxGatherOffset + rowIdxGatherBytes),
          sumBeforeRankBytes(AlignUp(AlignUp(worldSize * expertPerRank, kTokenCountStrideAlignment) * sizeof(int32_t),
                                     kMetaAlignment)),
          totalBytes(sumBeforeRankOffset + sumBeforeRankBytes)
    {
    }
};

}  // namespace MegaMoeMeta
