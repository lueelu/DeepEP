// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/owned_compute_cursor.hpp"

#if defined(__NPU_DEVICE__)
#define URMA_COMPLETION_INLINE __aicore__ inline
#else
#define URMA_COMPLETION_INLINE inline
#endif

namespace MegaMoeUrma {
// Bound one cooperative turn to the single hardware UB ownership credit. An issued
// tile can still be waiting on hardware: never grow this batch while waiting,
// and never cross into GMM1 when a Reorder batch is already in progress.
constexpr uint32_t kDirectReorderBatch = 1U;
template <class Consumer, class Work, class TimerT>
URMA_COMPLETION_INLINE uint32_t ConsumeIssuedReorders(Consumer& completion, Work& work, TimerT& timer)
{
    const uint32_t available = completion.IssuedCount();
    if (!available) return 0U;
    const uint32_t limit = available < kDirectReorderBatch ? available : kDirectReorderBatch;
    work.Before();
    uint32_t count = 0U;
    while (count < limit) {
        if (count && completion.NextIsGmm1()) break;
        RoutedCompletionItem item;
        completion.CompleteIssuedOne(item, timer);  // real hardware wait; retain UB ownership
        ++count;
        if (item.gmm1)
            work.Activate(item, timer);
        else
            work.Reorder(item, timer);
        // Single RX mailbox stays owned through all V/MTE3 reads, including
        // GM-backed GMM1 activation, which also uses this low-UB workspace.
        completion.ReleaseCompleted();
        if (item.gmm1) break;
    }
    return count;
}
}  // namespace MegaMoeUrma
#undef URMA_COMPLETION_INLINE
