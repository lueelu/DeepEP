// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernel_operator.h"
#include <c_api/asc_simd.h>
namespace MegaMoeUrma {
// Hardware channels follow Yancy/DeepEP's AIC->paired-AIV1 completion protocol.
// B supply uses AIV0 full4..7, empty8..11; these channels use AIV1 full12,
// empty14. One low-UB mailbox stays owned until activation/scatter completes.
struct RoutedFixChannel {
    static constexpr uint32_t Slots = 1U, Full = 12U, Empty = 14U, Aiv1Target = 16U;
    __aicore__ static inline void InitConsumer()
    {
        for (uint32_t i = 0; i < Slots; ++i) asc_sync_intra_arrive(PIPE_MTE3, Aiv1Target | (Empty + i));
    }
    __aicore__ static inline void Acquire(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_FIX, Aiv1Target | (Empty + sequence % Slots));
    }
    __aicore__ static inline void Publish(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_FIX, Aiv1Target | (Full + sequence % Slots));
    }
    __aicore__ static inline void Wait(uint32_t sequence)
    {
        asc_sync_intra_wait(PIPE_MTE2, Full + sequence % Slots);
        // S must not publish a ready flag before the MTE2 hardware wait passes.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0U));
    }
    __aicore__ static inline void Release(uint32_t sequence)
    {
        asc_sync_intra_arrive(PIPE_MTE3, Aiv1Target | (Empty + sequence % Slots));
    }
    __aicore__ static inline void DrainProducer()
    {
        for (uint32_t i = 0; i < Slots; ++i) asc_sync_intra_wait(PIPE_FIX, Aiv1Target | (Empty + i));
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(static_cast<AscendC::TEventID>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(static_cast<AscendC::TEventID>(0U));
    }
};

}  // namespace MegaMoeUrma
