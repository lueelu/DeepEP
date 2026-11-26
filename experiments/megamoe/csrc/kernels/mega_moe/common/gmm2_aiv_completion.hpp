// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"

namespace MegaMoeImpl {

// One FIX completion credit per AIC/AIV1 pair. Flags 0/1 are separate from
// the AIV0 weight queues. The submitted sequence only announces a queued event.
struct Gmm2AivCompletion {
    static constexpr uint32_t kMode = 4U;
    static constexpr uint32_t kFull = 0U;
    static constexpr uint32_t kEmpty = 1U;
    static constexpr uint32_t kAiv1 = 16U;

    __aicore__ static inline void InitConsumer()
    {
        AscendC::CrossCoreSetFlag<kMode, PIPE_S>(kEmpty);
    }

    __aicore__ static inline void Submit(__gm__ int32_t* submitted, int32_t sequence)
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_FIX>(kAiv1 | kEmpty);
        AscendC::CrossCoreSetFlag<kMode, PIPE_FIX>(kAiv1 | kFull);
        AscendC::WriteGmByPassDCache(submitted, sequence);
    }

    __aicore__ static inline bool HasPending(__gm__ int32_t* submitted, int32_t completed)
    {
        return AscendC::ReadGmByPassDCache(submitted) > completed;
    }

    // After HasPending, FIX must finish before this AIC is counted as ready for its slice.
    __aicore__ static inline void Acknowledge(__gm__ int32_t* sliceReady, int32_t& completed)
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_S>(kFull);
        AscendC::AtomicAdd(sliceReady, int32_t(1));
        AscendC::CrossCoreSetFlag<kMode, PIPE_S>(kEmpty);
        ++completed;
    }

    __aicore__ static inline void Drain(AscendC::TEventID fenceEvent)
    {
        AscendC::CrossCoreWaitFlag<kMode, PIPE_FIX>(kAiv1 | kEmpty);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(fenceEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(fenceEvent);
    }
};

}  // namespace MegaMoeImpl
