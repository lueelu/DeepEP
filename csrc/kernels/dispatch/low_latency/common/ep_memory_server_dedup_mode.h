// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_MODE_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_MODE_H

#include <cstdint>
#include "ll_compile_attrs.h"

namespace DispatchDedup {
// The dedup resource layout is fixed for the lifetime of the binary. The
// execution path is chosen at runtime from the per-rank capacity instead.
constexpr uint32_t kEpServerDedupDedupThreshold = 32U;
// Only the per-rank capacity and communicator size select the execution path;
// the actual launch B no longer participates.
constexpr __aicore__ __forceinline__ bool EpServerDedupEnabled(uint32_t numMaxTokensPerRank, uint32_t world = 16U)
{
    return world > 16U || numMaxTokensPerRank > kEpServerDedupDedupThreshold;
}
}  // namespace DispatchDedup
#endif
