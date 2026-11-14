// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "launch.h"
#include "kernel_operator.h"

namespace deepep::hierarchical {
// Test-only AIC consumer: the public kernel has no checksum or validation fields.
struct ChecksumConsumer {
    using Arguments = ChecksumArguments;
    DEEPEP_COMM_DEVICE void operator()(const LaunchArgs& p, uint32_t window)
    {
        if (!p.consumer.enabled) return;
        AscendC::GlobalTensor<uint64_t> checks;
        checks.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(p.consumer.checks));
        uint64_t hash = kHashSeed;
        for (uint32_t source = 0; source < p.layout.world; ++source) {
            hash = HashTensor(p.output + source * p.layout.DataStride(),
                              uint64_t(window) * p.layout.windowM * p.layout.n,
                              uint64_t(p.layout.Rows(window)) * p.layout.n, hash);
            hash = HashTensor(p.outputScale + source * p.layout.ScaleStride(),
                              uint64_t(window) * p.layout.windowM * (p.layout.n / 32),
                              uint64_t(p.layout.Rows(window)) * (p.layout.n / 32), hash);
        }
        checks.SetValue(window, hash);
        AscendC::DataCacheCleanAndInvalid<
            uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(checks[window]);
    }
private:
    DEEPEP_COMM_DEVICE uint64_t HashTensor(uintptr_t base, uint64_t offset, uint64_t bytes, uint64_t hash)
    {
        auto ptr = reinterpret_cast<__gm__ uint8_t*>(base) + offset;
        AscendC::GlobalTensor<uint8_t> tensor;
        tensor.SetGlobalBuffer(ptr);
        for (uint64_t i = 0; i < bytes; ++i) {
            if (i == 0 || ((reinterpret_cast<uintptr_t>(ptr + i) & 63U) == 0)) {
                AscendC::DataCacheCleanAndInvalid<
                    uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(tensor[i]);
            }
            hash = HashByte(hash, tensor.GetValue(i));
        }
        return hash;
    }

};
} // namespace deepep::hierarchical
