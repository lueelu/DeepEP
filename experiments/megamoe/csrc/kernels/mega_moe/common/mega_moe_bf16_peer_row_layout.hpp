// Copyright (c) 2026, Lu Lu
// Modified by xixing, guofanfeng 2026

#pragma once
#include "common/mega_moe_bf16_wave_plan.hpp"

namespace MegaMoeBF16Impl::Wave {
template <class Prefix>
struct PeerCounts {
    Prefix prefix;
    uint32_t experts, expert, rows;
    MEGA_MOE_PEER_HOST_DEVICE inline uint32_t Get(uint32_t source) const
    {
        const uint32_t begin = source ? prefix.Get((source - 1U) * experts + expert) : 0U;
        const uint32_t end = prefix.Get(source * experts + expert);
        // Capacity keeps the ordinary forward's source-prefix route selection.
        return begin >= rows ? 0U : (end < rows ? end : rows) - begin;
    }
};

// Dispatch reads and Combine writes the same packed source row for each expert row.
template <class Prefix>
struct PeerRowLayout {
    Prefix countsPrefix, sourcePrefix;
    uint32_t peers, experts, rank;

    MEGA_MOE_PEER_HOST_DEVICE inline uint32_t SourceRow(const Slice& s, const MegaMoeImpl::PeerTransfer& t) const
    {
        return sourcePrefix.Get(t.sourceRank * experts + s.expert) + t.sourceOrdinalBegin;
    }

    MEGA_MOE_PEER_HOST_DEVICE inline uint64_t SavedRow(const Slice& s, const MegaMoeImpl::PeerTransfer& t) const
    {
        const uint32_t before = t.sourceRank ? countsPrefix.Get((t.sourceRank - 1) * experts + s.expert) : 0;
        return s.packed - s.row + before + t.sourceOrdinalBegin;
    }

    template <class Visit>
    MEGA_MOE_PEER_HOST_DEVICE inline void ForEachSpan(const Slice& slice, uint64_t begin, uint64_t end,
                                                      MegaMoeImpl::PeerRowLimitsCache& cache, Visit visit) const
    {
        const uint64_t expertBase = slice.packed - slice.row;
        const PeerCounts<Prefix> counts{countsPrefix, experts, slice.expert, slice.fullRows};
        const auto& limits = MegaMoeImpl::GetCachedPeerRowLimits(cache, counts, slice.expert, peers);
        MegaMoeImpl::PeerTransferCursor<PeerCounts<Prefix>> cursor(
            counts, limits, slice.row, slice.row + slice.rows, static_cast<uint32_t>(begin - expertBase),
            static_cast<uint32_t>(end - expertBase), peers, rank);
        MegaMoeImpl::PeerTransfer transfer{};
        while (cursor.Next(transfer)) {
            visit(transfer, expertBase + transfer.destinationRow);
        }
    }
};

struct PeerExpert {
    uint64_t base;
    uint32_t rows, firstSliceRows;
};

template <class Prefix>
MEGA_MOE_PEER_HOST_DEVICE inline PeerExpert DescribePeerExpert(const Prefix& prefix, uint32_t peers, uint32_t experts,
                                                               uint32_t expert, uint64_t capacity, uint32_t groups)
{
    uint64_t base = 0;
    uint32_t priorGroups = 0;
    for (uint32_t e = 0; e <= expert; ++e) {
        const uint32_t raw = prefix.Get((peers - 1U) * experts + e);
        const uint64_t available = base < capacity ? capacity - base : 0;
        const uint32_t rows = raw < available ? raw : static_cast<uint32_t>(available);
        if (e == expert) {
            const uint32_t budget = (groups - priorGroups % groups) * kGroupRows;
            return {base, rows, rows < budget ? rows : budget};
        }
        base += rows;
        priorGroups += (rows + kGroupRows - 1U) / kGroupRows;
    }
    return {};
}

// An expert can start in the unused groups of its predecessor's Wave.
// Whole-Wave rank rotation changes execution order, never these physical cuts.
struct PeerSlice {
    uint32_t begin, end;
};
MEGA_MOE_PEER_HOST_DEVICE inline PeerSlice PeerSliceAt(uint32_t row, uint32_t rows, uint32_t firstSliceRows,
                                                       uint32_t groups)
{
    const uint32_t waveRows = groups * kGroupRows;
    const uint32_t begin = row < firstSliceRows ? 0U : firstSliceRows + (row - firstSliceRows) / waveRows * waveRows;
    const uint32_t length = begin == 0U ? firstSliceRows : waveRows;
    return {begin, rows - begin < length ? rows : begin + length};
}

// Enumerate the inverse mapping for any contiguous packed row interval.
template <class Counts>
class PeerRowCursor {
public:
    MEGA_MOE_PEER_HOST_DEVICE inline PeerRowCursor(const Counts& counts, uint32_t peers, uint32_t rank, uint32_t groups,
                                                   uint32_t rows, uint32_t firstSliceRows, uint32_t begin, uint32_t end)
        : counts_(counts),
          limits_(MegaMoeImpl::GetPeerRowLimits(counts, peers)),
          peers_(peers),
          rank_(rank),
          groups_(groups),
          rows_(rows),
          firstSliceRows_(firstSliceRows),
          next_(begin),
          end_(end)
    {
    }

    MEGA_MOE_PEER_HOST_DEVICE inline bool Next(MegaMoeImpl::PeerTransfer& transfer)
    {
        while (next_ < end_) {
            const auto slice = PeerSliceAt(next_, rows_, firstSliceRows_, groups_);
            const uint32_t stop = end_ < slice.end ? end_ : slice.end;
            MegaMoeImpl::PeerTransferCursor<Counts> cursor(counts_, limits_, slice.begin, slice.end, next_, stop,
                                                           peers_, rank_);
            if (cursor.Next(transfer)) {
                next_ = transfer.destinationRow + transfer.rowCount;
                return true;
            }
            next_ = stop;
        }
        return false;
    }

private:
    const Counts& counts_;
    MegaMoeImpl::PeerRowLimits limits_;
    uint32_t peers_, rank_, groups_, rows_, firstSliceRows_, next_, end_;
};
}  // namespace MegaMoeBF16Impl::Wave
