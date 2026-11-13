// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#pragma once
#include <cstdint>
#if defined(PRODUCER_PHASE_HOST_TEST)
#define PP_INLINE inline
#else
#define PP_INLINE __aicore__ inline
#endif
namespace MegaMoeBF16BackwardCatlassImpl {
// Same effective-row / non-empty ordinal rules as UpdateGroupParams.
// Used inside the existing restore-source loop; no peer search or sorting.
struct PreparedExpertCursor {
    int64_t previousGroupEnd = 0;
    uint64_t consumedRows = 0;
    uint32_t nextOrdinal = 0;
    PP_INLINE bool Next(int32_t value, uint32_t listType, uint64_t capacity, uint32_t& rows, uint32_t& ordinal)
    {
        const int64_t routed = listType == 0 ? static_cast<int64_t>(value) - previousGroupEnd : value;
        if (listType == 0) previousGroupEnd = value;
        rows = 0;
        if (routed <= 0 || consumedRows >= capacity) return false;
        const uint64_t available = capacity - consumedRows, positive = static_cast<uint64_t>(routed);
        rows = static_cast<uint32_t>(positive < available ? positive : available);
        ordinal = nextOrdinal++;
        consumedRows += rows;
        return true;
    }
};
struct ProducerPhaseMath {
    static constexpr uint32_t kRows = 128, kLevels = 10, kRest = 10;
    struct Coord {
        uint32_t m, n;
    };
    PP_INLINE static uint32_t CeilM(uint32_t x)
    {
        return (x + 127U) >> 7;
    }
    PP_INLINE static Coord Nz3(uint32_t mt, uint32_t nt, uint32_t i)
    {
        const uint32_t stripe = i / (mt * 3U), within = i - stripe * mt * 3U;
        const uint32_t left = nt - 3U * stripe, width = left < 3U ? left : 3U;
        const uint32_t row = within / width;
        return {(stripe & 1U) ? mt - 1U - row : row, 3U * stripe + within - row * width};
    }
    struct Desc {
        uint32_t begin = 0, end = 0, band = 0, packed = 255U;
        PP_INLINE static Desc Make(uint32_t begin, uint32_t procEnd, uint32_t m, uint32_t p)
        {
            Desc d;
            d.begin = begin;
            d.end = procEnd < m ? procEnd : m;
            d.band = procEnd > begin ? (procEnd - begin + p - 1U) / p : 0U;
            if (!d.band || d.end <= begin) return d;
            const uint32_t full = (d.end - begin) / d.band;
            uint32_t power = 0;
            while (power < 7U && ((d.band >> power) & 1U) == 0U) ++power;
            const uint32_t gcd = 1U << power;
            uint32_t first = 255U, shift = 0;
            if ((begin & (gcd - 1U)) == 0U) {
                shift = 7U - power;
                const uint32_t odd = d.band >> power;
                uint32_t inv = 1U;
                inv = (inv * (2U - odd * inv)) & 127U;
                inv = (inv * (2U - odd * inv)) & 127U;
                inv = (inv * (2U - odd * inv)) & 127U;
                first = (0U - (begin >> power) * inv) & ((1U << shift) - 1U);
            }
            d.packed = (full << 16) | (shift << 8) | first;
            return d;
        }
        PP_INLINE uint32_t Owned() const
        {
            return end > begin ? CeilM(end) - CeilM(begin) : 0U;
        }
        PP_INLINE uint32_t Aligned(uint32_t b) const
        {
            const uint32_t first = packed & 255U, shift = (packed >> 8) & 255U;
            return first == 255U || b <= first ? 0U : 1U + ((b - 1U - first) >> shift);
        }
        PP_INLINE uint32_t Up(uint32_t limit, uint32_t b) const
        {
            if (band < 128U || end <= begin) return 0U;
            const uint32_t full = packed >> 16, bf = b < full ? b : full;
            uint32_t sum;
            if (limit < band)
                sum = bf * ((limit >> 7) - 1U) + Aligned(bf);
            else
                sum = ((begin + bf * band) >> 7) - (begin >> 7) - bf + Aligned(bf);
            if (b > full) {
                const uint32_t start = begin + full * band, tail = end - start;
                const uint32_t stop = start + (tail < limit ? tail : limit);
                const uint32_t a = CeilM(start), z = stop >> 7;
                if (z > a) sum += z - a;
            }
            return sum;
        }
        PP_INLINE uint32_t OwnPrefix(uint32_t b) const
        {
            if (end <= begin) return 0U;
            const uint32_t stop = begin + b * band;
            return CeilM(stop < end ? stop : end) - CeilM(begin);
        }
        PP_INLINE uint32_t Prefix(uint32_t phase, uint32_t b) const
        {
            if (phase == kRest) return OwnPrefix(b) - Up(256U << (kLevels - 1U), b);
            return Up(256U << phase, b) - (phase ? Up(256U << (phase - 1U), b) : 0U);
        }
        PP_INLINE uint32_t Select(uint32_t phase, uint32_t rank, uint32_t p, uint32_t& owner) const
        {
            uint32_t lo = 0, hi = p;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (Prefix(phase, mid + 1U) <= rank)
                    lo = mid + 1U;
                else
                    hi = mid;
            }
            owner = lo;
            const uint32_t priorLimit =
                phase == kRest ? (256U << (kLevels - 1U)) : (phase ? (256U << (phase - 1U)) : 0U);
            const uint32_t prior = priorLimit ? Up(priorLimit, lo + 1U) - Up(priorLimit, lo) : 0U;
            return CeilM(begin + lo * band) + prior + rank - Prefix(phase, lo);
        }
    };
};
}  // namespace MegaMoeBF16BackwardCatlassImpl
#undef PP_INLINE
