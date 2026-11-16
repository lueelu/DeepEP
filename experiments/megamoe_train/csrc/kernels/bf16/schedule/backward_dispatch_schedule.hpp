// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#pragma once
#include "schedule/backward_dispatch_schedule_math.hpp"
namespace MegaMoeBF16BackwardCatlassImpl {
using PP = ProducerPhaseMath;
__aicore__ inline uint32_t ProducerPhaseChunkSize(uint32_t s, uint32_t c)
{
    return s < c ? s : c;
}
__aicore__ inline bool ProducerPhaseSupported(const BackwardDispatchReadyView& v, uint32_t c)
{
    return v.enabled && c && v.sources && v.producersPerSource &&
           (v.sources + ProducerPhaseChunkSize(v.sources, c) - 1U) / ProducerPhaseChunkSize(v.sources, c) <= 15U;
}
__aicore__ inline uint32_t ProducerPhaseWord(uint32_t phase)
{
    return phase == PP::kRest ? 1U : phase + 2U;
}
__aicore__ inline void PrepareProducerPhaseSource(const BackwardDispatchReadyView& v, uint32_t c, uint32_t m,
                                                  uint32_t source, __gm__ int32_t* dhpReady)
{
    const uint32_t size = ProducerPhaseChunkSize(v.sources, c);
    (void)dhpReady;  // Static metadata: no payload-side group arrival.
    const uint32_t cb = source / size * size;
    const uint32_t count = v.sources - cb < size ? v.sources - cb : size;
    const uint32_t begin = source ? v.End(source - 1U) : 0U;
    const auto d = PP::Desc::Make(begin, v.End(source), m, v.producersPerSource);
    auto* slot = v.Slot(source * v.producersPerSource);
    AscendC::WriteGmByPassDCache(slot + 12, static_cast<int32_t>(d.begin));
    AscendC::WriteGmByPassDCache(slot + 13, static_cast<int32_t>(d.end));
    AscendC::WriteGmByPassDCache(slot + 14, static_cast<int32_t>(d.band));
    AscendC::WriteGmByPassDCache(slot + 15, static_cast<int32_t>(d.packed));
    uint32_t previous = 0;
    for (uint32_t phase = 0; phase < PP::kLevels; ++phase) {
        const uint32_t upto = d.Up(256U << phase, v.producersPerSource);
        const uint32_t value = upto - previous;
        previous = upto;
        if (value) {
            for (uint32_t i = source - cb + 1U; i <= count; i += i & (0U - i))
                AscendC::AtomicAdd(v.Slot((cb + i - 1U) * v.producersPerSource) + ProducerPhaseWord(phase),
                                   static_cast<int32_t>(value));
        }
    }
    const uint32_t rest = d.Owned() - previous;
    if (rest) {
        for (uint32_t i = source - cb + 1U; i <= count; i += i & (0U - i))
            AscendC::AtomicAdd(v.Slot((cb + i - 1U) * v.producersPerSource) + 1, static_cast<int32_t>(rest));
    }
    // Caller retires all metadata stores once per restore lane and uses
    // the existing AIV restore SyncAll before publishing invocation readiness.
}
struct ProducerPhaseTile {
    uint32_t m, n, source, bandBegin, bandEnd;
    bool known;
};
class ProducerPhaseSchedule {
    BackwardDispatchReadyView v_;

    uint32_t cores_ = 0, m_ = 0, nt_ = 0, chunkSize_ = 0, chunks_ = 0;
    uint32_t beginM_ = 0, endM_ = 0, beginTask_ = 0, endTask_ = 0, cb_ = 0, count_ = 0;
    uint32_t cumulative_[11]{};
    bool enabled_ = false, single_ = false, deferTail_ = false;
    __aicore__ inline uint32_t ClipEnd(uint32_t source) const
    {
        const uint32_t e = v_.End(source);
        return e < m_ ? e : m_;
    }
    __aicore__ inline uint32_t ReadNode(uint32_t i, uint32_t phase) const
    {
        return static_cast<uint32_t>(
            AscendC::ReadGmByPassDCache(v_.Slot((cb_ + i - 1U) * v_.producersPerSource) + ProducerPhaseWord(phase)));
    }
    __aicore__ inline void LoadChunk(uint32_t task)
    {
        uint32_t lo = 0, hi = chunks_;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            const uint32_t last = (mid + 1U) * chunkSize_ < v_.sources ? (mid + 1U) * chunkSize_ - 1U : v_.sources - 1U;
            if (PP::CeilM(ClipEnd(last)) * nt_ <= task)
                lo = mid + 1U;
            else
                hi = mid;
        }
        cb_ = lo * chunkSize_;
        count_ = v_.sources - cb_ < chunkSize_ ? v_.sources - cb_ : chunkSize_;
        beginM_ = cb_ ? PP::CeilM(ClipEnd(cb_ - 1U)) : 0U;
        const uint32_t groupEnd = ClipEnd(cb_ + count_ - 1U);
        endM_ = PP::CeilM(groupEnd);
        // Only a non-final, non-aligned group can own a cross-group tail.
        // The final partial expert tile has no dependency on a later group.
        deferTail_ = groupEnd < m_ && (groupEnd & 127U) != 0U && endM_ > beginM_;
        beginTask_ = beginM_ * nt_;
        endTask_ = endM_ * nt_;
        // All trees are immutable after the P0 restore-ready gate. No group wait.
        uint32_t total = 0, nonempty = 0;
        for (uint32_t phase = 0; phase <= PP::kRest; ++phase) {
            uint32_t value = 0;
            for (uint32_t i = count_; i; i -= i & (0U - i)) value += ReadNode(i, phase);
            // The boundary tile is the last M row of kRest. Exclude only
            // that last ordinal from local counts; published trees stay intact.
            if (phase == PP::kRest && deferTail_) --value;
            nonempty += value != 0;
            total += value;
            cumulative_[phase] = total;
        }
        single_ = nonempty == 1U;
    }

public:
    __aicore__ inline void Init(const BackwardDispatchReadyView& v, __gm__ int32_t* dhpReady, uint32_t cores,
                                uint32_t m, uint32_t n)
    {
        (void)dhpReady;
        v_ = v;
        cores_ = cores;
        m_ = m;
        nt_ = (n + 255U) / 256U;
        enabled_ = ProducerPhaseSupported(v, cores);
        if (enabled_) {
            chunkSize_ = ProducerPhaseChunkSize(v.sources, cores);
            chunks_ = (v.sources + chunkSize_ - 1U) / chunkSize_;
        }
    }
    __aicore__ inline bool Enabled() const
    {
        return enabled_;
    }
    __aicore__ inline ProducerPhaseTile Get(uint32_t task)
    {
        if (task >= endTask_) LoadChunk(task);
        const uint32_t local = task - beginTask_;
        const uint32_t interiorM = endM_ - beginM_ - static_cast<uint32_t>(deferTail_);
        const uint32_t interiorTasks = interiorM * nt_;
        if (deferTail_ && local >= interiorTasks) return {endM_ - 1U, local - interiorTasks, 0, 0, 0, false};
        if (single_) {
            const auto x = PP::Nz3(interiorM, nt_, local);
            return {beginM_ + x.m, x.n, 0, 0, 0, false};
        }
        const uint32_t ordinal = local / nt_;
        uint32_t lo = 0, hi = 11;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (cumulative_[mid] <= ordinal)
                lo = mid + 1U;
            else
                hi = mid;
        }
        const uint32_t phase = lo, prior = phase ? cumulative_[phase - 1U] : 0U;
        const auto x = PP::Nz3(cumulative_[phase] - prior, nt_, local - prior * nt_);
        uint32_t rank = x.m, prefix = 0, bit = 1;
        while ((bit << 1) <= count_) bit <<= 1;
        for (; bit; bit >>= 1) {
            const uint32_t next = prefix + bit;
            if (next <= count_) {
                const uint32_t value = ReadNode(next, phase);
                if (value <= rank) {
                    rank -= value;
                    prefix = next;
                }
            }
        }
        const uint32_t source = cb_ + prefix;
        auto* p = v_.Slot(source * v_.producersPerSource);
        PP::Desc d;
        d.begin = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(p + 12));
        d.end = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(p + 13));
        d.band = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(p + 14));
        d.packed = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(p + 15));
        uint32_t band = 0;
        const uint32_t row = d.Select(phase, rank, v_.producersPerSource, band);
        const uint32_t begin = d.begin + band * d.band, end = begin + d.band < d.end ? begin + d.band : d.end;
        return {row, x.n, source, begin, end, true};
    }
};
__aicore__ inline void WaitProducerPhaseRows(const BackwardDispatchReadyView& v, const ProducerPhaseTile& tile,
                                             uint32_t row, uint32_t rows)
{
    const uint32_t end = row + rows, stop = end < tile.bandEnd ? end : tile.bandEnd;
    const uint32_t sourceBegin = tile.source ? v.End(tile.source - 1U) : 0U;
    const uint32_t sourceEnd = v.End(tile.source);
    const uint32_t band = (sourceEnd - sourceBegin + v.producersPerSource - 1U) / v.producersPerSource;
    const uint32_t owner = (tile.bandBegin - sourceBegin) / band;
    auto* progress = v.Slot(tile.source * v.producersPerSource + owner);
    while (AscendC::ReadGmByPassDCache(progress) < static_cast<int32_t>(stop - tile.bandBegin)) {
    }
    if (stop < end) v.WaitRows(stop, end - stop);
}
}  // namespace MegaMoeBF16BackwardCatlassImpl
