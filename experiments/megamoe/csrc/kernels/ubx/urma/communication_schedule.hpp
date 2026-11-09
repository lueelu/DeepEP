// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_COMM_INLINE __aicore__ inline
#else
#define URMA_COMM_INLINE inline
#endif
URMA_COMM_INLINE constexpr uint32_t DispatchBatchLimit(uint32_t step, uint32_t remaining, bool packFinished)
{
    // Preserve the two startup deliveries even when the entire case fits them.
    const uint32_t cap = step < 2U ? 1U : packFinished ? remaining : step == 2U ? 2U : step == 3U ? 4U : 8U;
    return cap < remaining ? cap : remaining;
}

// Base post-submit quota: prepare send-count + 1 original CV quotas.
// Counts are 2,2,3,5,9,... for sends 1,1,2,4,8,...; startup's two rounds
// are already complete. Bound by unprocessed quotas, never repeat a row.
URMA_COMM_INLINE constexpr uint32_t DispatchPackLimit(uint32_t sendCount, uint32_t nextPackRound,
                                                      uint32_t totalPackRounds)
{
    if (nextPackRound >= totalPackRounds) return nextPackRound;
    const uint32_t count = sendCount + 1U;
    const uint32_t left = totalPackRounds - nextPackRound;
    return nextPackRound + (count < left ? count : left);
}

// Run AFTER this batch's X/scale PUTs and SO publications. Startup already
// packed two rounds. The base quota alone is insufficient at the 4->8 ramp:
// 2+2+2+3+5=14 packed, but the next cumulative send needs 16. Extend only the
// post-submit quota to cover that next batch; never Pack before the current PUT.
URMA_COMM_INLINE constexpr uint32_t DispatchPackAfterLimit(uint32_t sendCount, uint32_t sentAfter, uint32_t nextStep,
                                                           uint32_t peerCvs, uint32_t nextPackRound,
                                                           uint32_t totalPackRounds)
{
    uint32_t limit = DispatchPackLimit(sendCount, nextPackRound, totalPackRounds);
    if (sentAfter < peerCvs) {
        const uint32_t nextCount = DispatchBatchLimit(nextStep, peerCvs - sentAfter, false);
        const uint32_t need = sentAfter + nextCount;
        if (limit < need) limit = need;
    }
    // Keep the exhausted startup cursor (2) for an empty/single-CV stream.
    if (nextPackRound >= totalPackRounds) return nextPackRound;
    return limit < totalPackRounds ? limit : totalPackRounds;
}

URMA_COMM_INLINE constexpr bool AdjacentChunks(const Chunk& first, uint64_t rows, uint32_t count, const Chunk& next)
{
    return next.peer == first.peer && next.cv == first.cv + count && next.localRow == first.localRow + rows &&
           next.remoteRow == first.remoteRow + rows;
}
#undef URMA_COMM_INLINE
}  // namespace MegaMoeUrma
