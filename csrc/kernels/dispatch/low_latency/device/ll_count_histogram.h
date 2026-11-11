#pragma once
#include "kernel_operator.h"

namespace DispatchDedup {
// Same MicroAPI/VF calling convention as pinned SHMEM's GetExpertFreq.
// Only BIN0 is needed: normalized IDs are in [0, localExperts+1] <= 33.
static __aicore__ inline void CountHistogramVf(__local_mem__ uint16_t* output,
                                              __local_mem__ uint8_t* input,
                                              uint32_t remaining, uint16_t repeats)
{
    AscendC::MicroAPI::RegTensor<uint8_t> values;
    AscendC::MicroAPI::RegTensor<uint16_t> counts;
    AscendC::MicroAPI::Duplicate(counts, 0);
    for (uint16_t i = 0; i < repeats; ++i) {
        auto mask = AscendC::MicroAPI::UpdateMask<uint8_t>(remaining);
        AscendC::MicroAPI::DataCopy(values, input + uint32_t(i) * 256U);
        AscendC::MicroAPI::Histograms<uint8_t, uint16_t,
            AscendC::MicroAPI::HistogramsBinType::BIN0,
            AscendC::MicroAPI::HistogramsType::FREQUENCY>(counts, values, mask);
    }
    auto outMask = AscendC::MicroAPI::CreateMask<uint16_t>();
    AscendC::MicroAPI::DataCopy(output, counts, outMask);
}

__aicore__ inline void CountRankHistogram(AscendC::LocalTensor<int32_t> ids,
                                         AscendC::LocalTensor<int32_t> normalized,
                                         AscendC::LocalTensor<uint16_t> histogram,
                                         uint32_t routes, uint32_t first, uint32_t experts)
{
    using namespace AscendC;
    // Avoid narrowing IDs: clamp before subtracting, so even INT_MIN/MAX are
    // safe. Valid experts map to 1..experts; outsiders to 0 or experts+1.
    Maxs(normalized, ids, int32_t(-1), routes);
    PipeBarrier<PIPE_V>();
    Mins(normalized, normalized, int32_t(first + experts), routes);
    PipeBarrier<PIPE_V>();
    Adds(normalized, normalized, int32_t(1) - int32_t(first), routes);
    PipeBarrier<PIPE_V>();
    Maxs(normalized, normalized, int32_t(0), routes);
    PipeBarrier<PIPE_V>();
    // Reinterpret nonnegative int32 values as four bytes. Three zero bytes
    // per ID affect only discarded bin0; bins1..experts are exact, regardless
    // of byte order. Existing 190KiB count admission bounds 4*routes < 65536.
    VF_CALL<CountHistogramVf>((__local_mem__ uint16_t*)histogram.GetPhyAddr(),
        (__local_mem__ uint8_t*)normalized.GetPhyAddr(), routes * 4U,
        uint16_t((routes * 4U + 255U) / 256U));
}
}
