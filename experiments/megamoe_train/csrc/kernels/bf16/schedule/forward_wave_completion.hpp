// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"
#include <c_api/asc_simd.h>

namespace MegaMoeBF16Training::Wave {

// Two credits overlap the next Gate/Up write with the preceding epilogue.
// AIV0 returns each credit after the epilogue retires; GM addresses are distinct.
struct GateUpCompletion {
    static constexpr uint32_t kStages = 2, kFull = 12, kEmpty = 14;
    __aicore__ static inline void InitConsumer()
    {
        for (uint32_t i = 0; i < kStages; ++i) asc_sync_intra_arrive(PIPE_MTE3, kEmpty + i);
    }
    __aicore__ static inline void Acquire(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_FIX, kEmpty + sequence % kStages);
    }
    __aicore__ static inline void Publish(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_FIX, kFull + sequence % kStages);
    }
    __aicore__ static inline void Wait(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_MTE2, kFull + sequence % kStages);
    }
    __aicore__ static inline void Release(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_MTE3, kEmpty + sequence % kStages);
    }
    __aicore__ static inline void Drain()
    {
        for (uint32_t i = 0; i < kStages; ++i) asc_sync_intra_wait(PIPE_FIX, kEmpty + i);
        AscendC::SyncFunc<AscendC::HardEvent::FIX_S>();
    }
};

// One FIX completion notification per producing core and expert slice.
// Scalar issued means queued; AIV1 must receive FULL before publishing ready.
struct DownCompletion {
    static constexpr uint32_t kMode = 4, kFull = 0, kEmpty = 1, kAiv1 = 16;
    __aicore__ static inline void InitConsumer()
    {
        AscendC::CrossCoreSetFlag<kMode, PIPE_S>(kEmpty);
    }
    __aicore__ static inline void Publish(__gm__ int32_t* issued, int32_t sequence)
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_FIX>(kAiv1 | kEmpty);
        AscendC::CrossCoreSetFlag<kMode, PIPE_FIX>(kAiv1 | kFull);
        AscendC::WriteGmByPassDCache(issued, sequence);
    }
    __aicore__ static inline bool IsIssued(__gm__ int32_t* issued, int32_t sequence)
    {
        return AscendC::ReadGmByPassDCache(issued) >= sequence;
    }
    __aicore__ static inline void Receive()
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_S>(kFull);
    }
    __aicore__ static inline void Release()
    {
        AscendC::CrossCoreSetFlag<kMode, PIPE_S>(kEmpty);
    }
    __aicore__ static inline void Drain()
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_FIX>(kAiv1 | kEmpty);
        AscendC::SyncFunc<AscendC::HardEvent::FIX_S>();
    }
};
}  // namespace MegaMoeBF16Training::Wave
