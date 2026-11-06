// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

// Backward-only, Host-zeroed progress suffix: one cacheline per producer.
#ifndef MEGA_MOE_BF16_BACKWARD_DISPATCH_READY_HPP
#define MEGA_MOE_BF16_BACKWARD_DISPATCH_READY_HPP
namespace MegaMoeBF16BackwardCatlassImpl {
struct BackwardDispatchReadyView {
    __gm__ int32_t* progressBase{nullptr};
    __gm__ int32_t* cumsum{nullptr};
    uint32_t sources{0}, producersPerSource{0}, experts{0}, expert{0}, rowLimit{0};
    uint32_t progressSlot{0};
    bool enabled{false};
    __aicore__ inline __gm__ int32_t* Slot(uint32_t producer) const
    {
        const uint64_t stride = static_cast<uint64_t>(sources) * producersPerSource;
        return progressBase + (static_cast<uint64_t>(progressSlot) * stride + producer) * 16;
    }
    __aicore__ inline uint32_t End(uint32_t source) const
    {
        const uint32_t raw = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(cumsum + source * experts + expert));
        return raw < rowLimit ? raw : rowLimit;
    }
    __aicore__ inline void WaitRows(uint32_t row, uint32_t rows) const
    {
        const uint32_t end = row + rows;
        while (row < end) {
            // Find the first source whose exclusive end exceeds row. Empty
            // sources are skipped. No full-rank traversal per GEMM tile.
            uint32_t lo = 0, hi = sources;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (End(mid) <= row)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const uint32_t source = lo;
            const uint32_t begin = source == 0 ? 0 : End(source - 1);
            const uint32_t sourceEnd = End(source);
            const uint32_t bandRows = (sourceEnd - begin + producersPerSource - 1) / producersPerSource;
            const uint32_t band = (row - begin) / bandRows;
            const uint32_t bandBegin = begin + band * bandRows;
            const uint32_t bandEnd = bandBegin + bandRows < sourceEnd ? bandBegin + bandRows : sourceEnd;
            const uint32_t stop = end < bandEnd ? end : bandEnd;
            const uint32_t needed = stop - bandBegin;
            auto* progress = Slot(source * producersPerSource + band);
            while (AscendC::ReadGmByPassDCache(progress) < static_cast<int32_t>(needed)) {
            }
            row = stop;
        }
    }
};
}  // namespace MegaMoeBF16BackwardCatlassImpl
#endif
