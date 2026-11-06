// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include <cstdint>
#if defined(FPP_HOST_TEST)
#include <cassert>
#define FPP_INLINE inline
#define FPP_GM
#ifndef FPP_TILE_M
#define FPP_TILE_M 192
#endif
#else
#define FPP_INLINE __aicore__ inline
#define FPP_GM __gm__
#define FPP_TILE_M MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows
#endif
namespace MegaMoeBF16ForwardSchedule {
FPP_INLINE uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
FPP_INLINE uint32_t Ceil(uint32_t a, uint32_t b)
{
    return a / b + (a % b != 0U);
}
FPP_INLINE int32_t Read(FPP_GM int32_t* p)
{
#if defined(FPP_HOST_TEST)
    return *p;
#else
    return AscendC::ReadGmByPassDCache(p);
#endif
}
struct Coord {
    uint32_t m, n;
};
struct Band {
    uint32_t first = 0, owned = 0, early0 = 0, early1 = 0;
    FPP_INLINE static Band Make(uint32_t begin, uint32_t end, uint32_t p, uint32_t producer)
    {
        const uint32_t width = Ceil(end - begin, p), b = begin + producer * width, z = Min(end, b + width);
        Band d;
        if (z <= b) return d;
        d.first = Ceil(b, kRows);
        d.owned = Ceil(z, kRows) - d.first;
        const uint32_t a = Min(z, b + kRows) / kRows, t = Min(z, b + 2U * kRows) / kRows;
        d.early0 = a > d.first ? a - d.first : 0U;
        d.early1 = t > d.first + d.early0 ? t - d.first - d.early0 : 0U;
        return d;
    }
    static constexpr uint32_t kTile = FPP_TILE_M, kRows = 2U * kTile;
};
struct View {
    static constexpr uint32_t kTile = Band::kTile, kRows = Band::kRows;
    FPP_GM int32_t *flags = nullptr, *ends = nullptr, *modes = nullptr;
    uint32_t e = 0, s = 0, c = 0, p = 0, size = 0, groups = 0, totalProducers = 0, freeWords = 0, words = 0;
    uint64_t cap = 0, slots = 0;
    bool enabled = false;
    FPP_INLINE void Init(FPP_GM int32_t* f, FPP_GM int32_t* z, FPP_GM int32_t* mode, uint64_t capacity,
                         uint32_t experts, uint32_t sources, uint32_t cores, uint32_t producers, uint32_t hidden)
    {
        flags = f;
        ends = z;
        modes = mode;
        cap = capacity;
        e = experts;
        s = sources;
        c = cores;
        p = producers;
        if (!f || !z || !mode || cap > 0x7FFFFFFFULL || !e || e > 7U || !s || s > 256U || !p || p > c ||
            (c != 28U && c != 32U && c != 36U) || s * p > 256U)
            return;
        size = Min(s, c / p);
        groups = Ceil(s, size);
        if (groups > 10U) return;
        totalProducers = s * p;
        freeWords = 16U - e;
        words = 2U * e * totalProducers;
        slots = cap / 256U + (cap % 256U != 0U);
        const uint32_t alignedWords = Ceil(words, 8U) * 8U, alignedFree = Ceil(freeWords, 8U) * 8U;
        const uint32_t tail = hidden >= alignedWords ? hidden - alignedWords : 0U;
        const uint32_t countScratch = s * Ceil(e, 8U) * 8U;
        enabled = words <= hidden && words <= slots * freeWords && (tail >= alignedFree || countScratch >= alignedFree);
    }
    FPP_INLINE FPP_GM int32_t* At(uint32_t i) const
    {
        return flags + (i / freeWords) * 16U + e + i % freeWords;
    }
    FPP_INLINE uint32_t Index(uint32_t expert, uint32_t group, uint32_t i) const
    {
        return 2U * (expert * totalProducers + group * size * p + i - 1U);
    }
    FPP_INLINE uint32_t Node(uint32_t expert, uint32_t group, uint32_t i, uint32_t phase) const
    {
        const uint32_t index = Index(expert, group, i);
        if (phase == 2U) return static_cast<uint32_t>(Read(At(index + 1U)));
        const uint32_t packed = static_cast<uint32_t>(Read(At(index)));
        return (packed >> (phase * 9U)) & 511U;
    }
    FPP_INLINE uint32_t End(uint32_t source, uint32_t expert, uint32_t m) const
    {
        return Min(static_cast<uint32_t>(Read(ends + source * e + expert)), m);
    }
    template <class Tensor>
    FPP_INLINE void Accumulate(Tensor& nodes, uint32_t expert, uint32_t source, uint32_t begin, uint32_t end) const
    {
        const uint32_t group = source / size, first = group * size, count = Min(size, s - first) * p;
        begin = Min(begin, static_cast<uint32_t>(cap));
        end = Min(end, static_cast<uint32_t>(cap));
        for (uint32_t producer = 0; producer < p; ++producer) {
            const auto d = Band::Make(begin, end, p, producer);
            const uint32_t packed = d.early0 + (d.early1 << 9U), rest = d.owned - d.early0 - d.early1;
            if (!packed && !rest) continue;
            for (uint32_t i = (source - first) * p + producer + 1U; i <= count; i += i & (0U - i)) {
                const uint32_t index = Index(expert, group, i);
                nodes.SetValue(index, nodes.GetValue(index) + static_cast<int32_t>(packed));
                nodes.SetValue(index + 1U, nodes.GetValue(index + 1U) + static_cast<int32_t>(rest));
            }
        }
    }
};
class Schedule {
    View v_;
    uint32_t expert_ = 0, m_ = 0, mt_ = 0, nt_ = 0, width_ = 1, groups_ = 0;
    uint32_t ends_[10]{}, tails_[11]{}, early0_[10]{}, early1_[10]{}, rest_[10]{};
    uint32_t known_ = 0, current_ = 0xFFFFFFFFU, mBegin_ = 0, interior_ = 0, counts_[3]{}, nBudget_ = 0;
    bool enabled_ = false, single_ = false;
    FPP_INLINE uint32_t Prefix(uint32_t group, uint32_t count, uint32_t phase) const
    {
        uint32_t sum = 0;
        for (; count; count -= count & (0U - count)) sum += v_.Node(expert_, group, count, phase);
        return sum;
    }
    FPP_INLINE uint32_t Select(uint32_t group, uint32_t phase, uint32_t rank) const
    {
        const uint32_t first = group * v_.size, count = Min(v_.size, v_.s - first) * v_.p;
        uint32_t bit = 1, prefix = 0;
        while ((bit << 1U) <= count) bit <<= 1U;
        for (; bit; bit >>= 1U) {
            const uint32_t next = prefix + bit;
            if (next <= count) {
                const uint32_t value = v_.Node(expert_, group, next, phase);
                if (value <= rank) {
                    rank -= value;
                    prefix = next;
                }
            }
        }
        const uint32_t source = first + prefix / v_.p, producer = prefix % v_.p;
        const uint32_t begin = source ? v_.End(source - 1U, expert_, m_) : 0U, end = v_.End(source, expert_, m_);
        const auto d = Band::Make(begin, end, v_.p, producer);
        return d.first + (phase == 0U ? 0U : phase == 1U ? d.early0 : d.early0 + d.early1) + rank;
    }
    FPP_INLINE Coord StripeColumns(uint32_t mt, uint32_t columns, uint32_t task) const
    {
        const uint32_t span = mt * width_, stripe = task / span, inside = task % span, n = stripe * width_,
                       w = Min(width_, columns - n);
        return {inside / w, n + inside % w};
    }
    FPP_INLINE uint32_t InteriorEnd(uint32_t g) const
    {
        return ends_[g] - tails_[g + 1U];
    }
    FPP_INLINE void Load(uint32_t group)
    {
        current_ = group;
        mBegin_ = group ? ends_[group - 1U] : 0U;
        const uint32_t tail = tails_[group + 1U] - tails_[group];
        interior_ = ends_[group] - mBegin_ - tail;
        if (!(known_ & (1U << group))) {
            const uint32_t count = Min(v_.size, v_.s - group * v_.size) * v_.p;
            early0_[group] = 2U * Prefix(group, count, 0U);
            early1_[group] = 2U * Prefix(group, count, 1U);
            const uint32_t rest = Prefix(group, count, 2U);
#if defined(FPP_HOST_TEST)
            assert(rest >= static_cast<uint32_t>(tail != 0));
#endif
            rest_[group] = 2U * (rest - static_cast<uint32_t>(tail != 0));
            known_ |= 1U << group;
        }
        counts_[0] = early0_[group];
        counts_[1] = early1_[group];
        counts_[2] = rest_[group];
#if defined(FPP_HOST_TEST)
        assert(counts_[0] + counts_[1] + counts_[2] == interior_);
#endif
        single_ = (counts_[0] != 0) + (counts_[1] != 0) + (counts_[2] != 0) == 1U;
    }

public:
    FPP_INLINE void Init(const View& v, uint32_t expert, uint64_t unused, uint32_t m, uint32_t n, uint32_t width,
                         bool use)
    {
        (void)unused;
        v_ = v;
        expert_ = expert;
        m_ = m;
        mt_ = Ceil(m, View::kTile);
        nt_ = Ceil(n, 256U);
        width_ = width;
        groups_ = v.groups;
        enabled_ = use && v.enabled && m && nt_ && width && Read(v.modes + expert * 16U + 1U) == 1;
        if (!enabled_) return;
        uint32_t prior = 0;
        for (uint32_t g = 0; g < groups_; ++g) {
            const uint32_t source = Min((g + 1U) * v_.size, v_.s) - 1U, end = v_.End(source, expert_, m_);
            ends_[g] = Min(2U * Ceil(end, View::kRows), mt_);
            const uint32_t tail = end % View::kRows && ends_[g] > prior ? ends_[g] - 2U * (end / View::kRows) : 0U;
            tails_[g + 1U] = tails_[g] + tail;
            prior = ends_[g];
        }
    }
    FPP_INLINE bool Enabled() const
    {
        return enabled_;
    }
    FPP_INLINE Coord Get(uint32_t task)
    {
        const uint32_t totalTail = tails_[groups_], totalInterior = mt_ - totalTail;
        if (task >= totalInterior * nt_) {
            const auto xy = StripeColumns(totalTail, nt_, task - totalInterior * nt_);
            uint32_t lo = 0, hi = groups_;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1U;
                if (tails_[mid + 1U] <= xy.m)
                    lo = mid + 1U;
                else
                    hi = mid;
            }
            const uint32_t tail = tails_[lo + 1U] - tails_[lo];
            return {ends_[lo] - tail + xy.m - tails_[lo], xy.n};
        }
        if (groups_ == 1U) {
            if (current_ != 0U) Load(0U);
            if (!nBudget_) {
                const uint32_t early = counts_[0] ? counts_[0] : counts_[1];
                nBudget_ = Min(width_, nt_);
                if (!single_ && early && early * nBudget_ < v_.c) {
                    const uint32_t needed = Ceil(v_.c, early);
                    nBudget_ = Min(nt_, needed > width_ ? needed : width_);
                }
            }
            const uint32_t super = task / (interior_ * nBudget_), begin = super * nBudget_,
                           columns = Min(nBudget_, nt_ - begin);
            uint32_t local = task - super * interior_ * nBudget_, phase = 0;
            while (phase < 2U && local >= counts_[phase] * columns) {
                local -= counts_[phase] * columns;
                ++phase;
            }
            const auto xy = StripeColumns(counts_[phase], columns, local);
            return {single_ ? mBegin_ + xy.m : 2U * Select(0U, phase, xy.m / 2U) + xy.m % 2U, begin + xy.n};
        }
        const auto xy = StripeColumns(totalInterior, nt_, task);
        uint32_t lo = 0, hi = groups_;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1U;
            if (InteriorEnd(mid) <= xy.m)
                lo = mid + 1U;
            else
                hi = mid;
        }
        if (current_ != lo) Load(lo);
        uint32_t ordinal = xy.m - (lo ? InteriorEnd(lo - 1U) : 0U);
        if (single_) return {mBegin_ + ordinal, xy.n};
        uint32_t phase = 0;
        while (phase < 2U && ordinal >= counts_[phase]) {
            ordinal -= counts_[phase];
            ++phase;
        }
        return {2U * Select(lo, phase, ordinal / 2U) + ordinal % 2U, xy.n};
    }
};
}  // namespace MegaMoeBF16ForwardSchedule
#undef FPP_INLINE
#undef FPP_GM
#undef FPP_TILE_M
