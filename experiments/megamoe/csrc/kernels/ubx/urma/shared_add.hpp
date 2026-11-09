#pragma once
#include "kernel_operator.h"

namespace MegaMoeUrma {
// Six independent input/output slots. Routed accumulation remains FP32 across
// the local join; shared GMM2 is BF16. Like main Unpermute, cast only after Add.
struct SharedAddUb {
    static constexpr uint32_t width = 2048U, slots = 6U;
    static constexpr uint32_t routed = 0U;
    static constexpr uint32_t shared = routed + slots * width * 4U;
    static constexpr uint32_t expanded = shared + slots * width * 2U;
    static constexpr uint32_t output = expanded + slots * width * 4U;
    static constexpr uint32_t end = output + slots * width * 2U;
};
static_assert(SharedAddUb::end == 144U * 1024U);

template <class TraceSink>
__aicore__ inline void SharedAdd(const Arguments& a, uint32_t worker, TraceSink& timer)
{
    using namespace AscendC;
    using U = SharedAddUb;
    static_assert(U::end <= kRequiredUbBytes);
    GlobalTensor<float> routed;
    GlobalTensor<bfloat16_t> shared, output;
    routed.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.sharedRoutedOutput));
    shared.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.sharedOutput));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.output));
    LocalTensor<float> x(TPosition::VECCALC, U::routed, U::slots * U::width);
    LocalTensor<bfloat16_t> y(TPosition::VECCALC, U::shared, U::slots * U::width);
    LocalTensor<float> fy(TPosition::VECCALC, U::expanded, U::slots * U::width);
    LocalTensor<bfloat16_t> result(TPosition::VECCALC, U::output, U::slots * U::width);
    const auto start = timer.TraceBegin();
    const uint64_t total = uint64_t(a.config.bs) * a.config.hidden;
    const uint64_t tiles = (total + U::width - 1U) / U::width;
    const uint64_t count = worker < tiles ? (tiles - 1U - worker) / kLocalWorkers + 1U : 0U;
    for (uint32_t slot = 0; slot < U::slots; ++slot) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
        SetFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }
    uint64_t submitted = 0, consumed = 0;
    while (consumed < count) {
        // Prefill / replenish without waiting for MTE3 on the scalar pipe.
        while (submitted < count && submitted - consumed < U::slots) {
            const uint32_t slot = submitted % U::slots, base = slot * U::width;
            const auto event = static_cast<TEventID>(slot);
            const uint64_t offset = (worker + submitted * kLocalWorkers) * U::width;
            const uint32_t n = total - offset < U::width ? total - offset : U::width;
            WaitFlag<HardEvent::V_MTE2>(event);
            DataCopy(x[base], routed[offset], n);
            DataCopy(y[base], shared[offset], n);
            SetFlag<HardEvent::MTE2_V>(event);
            ++submitted;
        }
        const uint32_t slot = consumed % U::slots, base = slot * U::width;
        const auto event = static_cast<TEventID>(slot);
        const uint64_t offset = (worker + consumed * kLocalWorkers) * U::width;
        const uint32_t n = total - offset < U::width ? total - offset : U::width;
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(fy[base], y[base], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Add(fy[base], x[base], fy[base], n);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(event);   // input slot may be refilled
        WaitFlag<HardEvent::MTE3_V>(event);  // protect previous output in this slot
        Cast(result[base], fy[base], RoundMode::CAST_RINT, n);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(output[offset], result[base], n);
        SetFlag<HardEvent::MTE3_V>(event);
        ++consumed;
    }
    for (uint32_t slot = 0; slot < U::slots; ++slot) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
        WaitFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }
    PipeBarrier<PIPE_ALL>();  // one terminal drain, never a per-tile scalar wait
    timer.TraceEnd(TraceEvent::SharedAdd, start);
}
}  // namespace MegaMoeUrma
