// Copyright (c) 2026, Lu Lu
// Modified by xixing, guofanfeng 2026

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "experiments/megamoe_train/csrc/kernels/bf16/schedule/peer_row_layout.hpp"
#include "experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp"

using namespace MegaMoeBF16Training::Wave;
using Row = std::pair<uint32_t, uint32_t>;
struct Prefix {
    const std::vector<uint32_t>& values;
    uint32_t Get(uint32_t i) const
    {
        return values[i];
    }
};

int main()
{
    assert(kGmm1Rows == 256 && kGmm2Rows == 256 && GroupsPerWave(32, 3072, 7168) == 3);
    assert(MetadataFits(128, 4));
    assert(MetadataFits(1024, 2));
    assert(MetadataBytes(8, 4) == 4096 && MetadataBytes(1024, 2) == 16384);
    assert(ScratchFits(7168, 1024, 2));
    assert(!ScratchFits(32768, 8, 4));
    static_assert(MegaMoeBF16Training::BackwardTopkPartialCount(MegaMoeBF16Training::kSaveBf16, 3072) == 1);
    static_assert(MegaMoeBF16Training::BackwardTopkPartialCount(MegaMoeBF16Training::kSaveNone, 3072) == 12);
    static_assert(MegaMoeBF16Training::BackwardTopkPartialCount(MegaMoeBF16Training::kSaveMxFp8, 3072) == 12);
    static_assert(MegaMoeBF16Training::TopkGradReductionWidth(24467, 3072, 64) == 3072);
    static_assert(MegaMoeBF16Training::TopkGradReductionWidth(24258, 3072, 64) == 1536);
    static_assert(MegaMoeBF16Training::TopkGradReductionWidth(24693, 3072, 64) == 1536);
    static_assert(10U * 16384U + 8192U <= 248U * 1024U);
    using Quant = MegaMoeBF16Training::SavedGateUpQuantUbLayoutFor<256, 64, 128>;
    static_assert(Quant::gateData == 2U * 256U * 128U * 2U);
    static_assert(Quant::totalBytesWithClipMasks + 256U * sizeof(float) <= 248U * 1024U);
    for (uint64_t capacity : {0ULL, 1ULL, 255ULL, 256ULL, 257ULL, 24576ULL}) {
        const MegaMoeBF16Training::ForwardTransientWorkspaceLayout transient(6144, 4, capacity);
        const ForwardWorkspaceLayout bf16(transient.totalBytes, capacity, 4, 7168, 3072, true);
        const ForwardWorkspaceLayout other(transient.totalBytes, capacity, 4, 7168, 3072, false);
        assert(bf16.rawUpBytes == 0 && bf16.totalBytes == bf16.rawUpOffset);
        assert(other.rawUpOffset == bf16.rawUpOffset && other.rawUpOffset % 512U == 0);
        assert(other.rawUpBytes >= capacity * 3072U * sizeof(uint16_t));
        assert(other.totalBytes == other.rawUpOffset + other.rawUpBytes);
    }
    std::mt19937 rng(20261116);
    uint64_t cases = 0, checkedRows = 0, sourceBands = 0;
    for (uint32_t test = 0; test < 180; ++test) {
        const uint32_t peers = test == 0       ? 8
                               : test == 1     ? 128
                               : test % 7 == 0 ? 128
                               : test % 5 == 0 ? 64
                                               : 2 + rng() % 15;
        const uint32_t experts = 4, groups = test < 2 ? 3 : 1 + rng() % 6;
        std::vector<uint32_t> counts(peers * experts), prefix(peers * experts);
        for (uint32_t e = 0; e < experts; ++e) {
            uint32_t sum = 0;
            for (uint32_t p = 0; p < peers; ++p) {
                uint32_t n = test == 0 ? 768 : test == 1 ? 12 : rng() % 39;
                if (test > 1 && (p + e + test) % 4 == 0) n = 0;
                if (test > 1 && e == test % experts && p == test % peers) n += 1200;
                counts[p * experts + e] = n;
                prefix[p * experts + e] = sum += n;
            }
        }
        uint64_t total = 0;
        for (uint32_t e = 0; e < experts; ++e) total += prefix[(peers - 1) * experts + e];
        const uint64_t capacity = test < 2 || test % 3 ? total : rng() % (total + 1);
        auto expertCounts = [&](uint32_t e) { return prefix[(peers - 1) * experts + e]; };
        std::vector<std::vector<Slice>> slices(experts);
        Cursor position{};
        for (;;) {
            const auto range = Plan(position, groups, experts, capacity, expertCounts);
            if (!range.valid) break;
            auto cursor = range.begin;
            uint32_t remaining = groups;
            Slice slice{};
            while (Next(cursor, remaining, experts, capacity, expertCounts, slice))
                slices[slice.expert].push_back(slice);
            position = range.end;
        }
        for (const uint32_t rank : {0U, peers / 2, peers - 1}) {
            for (uint32_t e = 0; e < experts; ++e) {
                const auto info = DescribePeerExpert(Prefix{prefix}, peers, experts, e, capacity, groups);
                if (slices[e].empty()) {
                    assert(info.rows == 0);
                    continue;
                }
                assert(info.base == slices[e][0].packed && info.firstSliceRows == slices[e][0].rows);
                const PeerCounts<Prefix> pc{Prefix{prefix}, experts, e, info.rows};
                // Independent oracle: visit one route from every nonempty
                // source per round; within each slice, group by fixed peer order.
                std::vector<Row> roundRobin;
                std::vector<uint32_t> taken(peers);
                while (roundRobin.size() < info.rows) {
                    for (uint32_t ordinal = 0; ordinal < peers && roundRobin.size() < info.rows; ++ordinal) {
                        const uint32_t peer = (rank + ordinal) % peers;
                        const uint32_t begin = peer ? prefix[(peer - 1U) * experts + e] : 0U;
                        const uint32_t admitted =
                            begin >= info.rows ? 0U : std::min(counts[peer * experts + e], info.rows - begin);
                        if (taken[peer] < admitted) roundRobin.emplace_back(peer, taken[peer]++);
                    }
                }
                std::vector<Row> expected;
                for (const auto& slice : slices[e]) {
                    auto begin = roundRobin.begin() + slice.row;
                    auto end = begin + slice.rows;
                    std::stable_sort(begin, end, [=](const Row& a, const Row& b) {
                        return (a.first + peers - rank) % peers < (b.first + peers - rank) % peers;
                    });
                    expected.insert(expected.end(), begin, end);
                    const auto actualSlice = PeerSliceAt(slice.row, info.rows, info.firstSliceRows, groups);
                    assert(actualSlice.begin == slice.row && actualSlice.end == slice.row + slice.rows);
                    const auto limits = MegaMoeImpl::GetPeerRowLimits(pc, peers);
                    std::vector<uint32_t> coverage(slice.rows);
                    for (uint32_t owner = 0; owner < 32; ++owner) {
                        const uint32_t q = slice.rows / 32, r = slice.rows % 32;
                        const uint32_t own = slice.row + owner * q + std::min(owner, r);
                        const uint32_t stop = own + q + (owner < r);
                        MegaMoeImpl::PeerTransferCursor<PeerCounts<Prefix>> cursor(
                            pc, limits, slice.row, slice.row + slice.rows, own, stop, peers, rank);
                        MegaMoeImpl::PeerTransfer transfer{};
                        while (cursor.Next(transfer)) {
                            for (uint32_t j = 0; j < transfer.rowCount; ++j) {
                                const uint32_t row = transfer.destinationRow + j;
                                assert(expected[row] == Row(transfer.sourceRank, transfer.sourceOrdinalBegin + j));
                                ++coverage[row - slice.row];
                            }
                        }
                    }
                    for (auto c : coverage) assert(c == 1);
                }
                assert(expected.size() == info.rows);
                const Row unwritten{UINT32_MAX, UINT32_MAX};
                std::vector<Row> canonical(info.rows, unwritten);
                std::vector<uint32_t> physical(info.rows), runs(info.rows);
                // Packed tiles may cross peer, Wave and partial-slice boundaries.
                for (uint32_t begin = 0; begin < info.rows; begin += 128) {
                    const uint32_t end = std::min(begin + 128, info.rows);
                    PeerRowCursor<PeerCounts<Prefix>> inverse(pc, peers, rank, groups, info.rows, info.firstSliceRows,
                                                              begin, end);
                    MegaMoeImpl::PeerTransfer transfer{};
                    uint32_t next = begin;
                    while (inverse.Next(transfer)) {
                        assert(transfer.destinationRow == next);
                        for (uint32_t j = 0; j < transfer.rowCount; ++j, ++next) {
                            assert(expected[next] == Row(transfer.sourceRank, transfer.sourceOrdinalBegin + j));
                            const uint32_t sourceBase =
                                transfer.sourceRank ? prefix[(transfer.sourceRank - 1U) * experts + e] : 0U;
                            const uint32_t target = sourceBase + transfer.sourceOrdinalBegin + j;
                            assert(target < info.rows && canonical[target] == unwritten);
                            canonical[target] = expected[next];
                            physical[target] = next;
                            runs[target] = transfer.rowCount - j;
                        }
                    }
                    assert(next == end);
                }
                auto canonicalOracle = expected;
                std::sort(canonicalOracle.begin(), canonicalOracle.end());
                assert(canonical == canonicalOracle);
                // Emulate a K256 L1 load using only the saved row/run map.
                // Tile and peer boundaries need not coincide with Wave cuts.
                for (uint32_t tile = 0; tile < info.rows; tile += 256) {
                    const uint32_t end = std::min(tile + 256, info.rows);
                    for (uint32_t row = tile; row < end;) {
                        const uint32_t take = std::min(runs[row], end - row);
                        assert(take > 0 && physical[row] + take <= expected.size());
                        for (uint32_t j = 0; j < take; ++j)
                            assert(expected[physical[row] + j] == canonicalOracle[row + j]);
                        row += take;
                    }
                }
                // Emulate backward source producers and their monotonic progress.
                std::vector<uint32_t> covered(info.rows);
                for (uint32_t source = 0; source < peers; ++source) {
                    const uint32_t producers = std::max(1U, 32U / peers);
                    const uint32_t band = (pc.Get(source) + producers - 1) / producers;
                    for (uint32_t producer = 0; producer < producers; ++producer) {
                        const uint32_t begin = producer * band, end = std::min(begin + band, pc.Get(source));
                        uint32_t next = begin;
                        for (uint32_t row = 0; row < info.rows; ++row) {
                            const auto id = expected[row];
                            if (id.first == source && id.second >= begin && id.second < end) {
                                assert(id.second == next++);
                                ++covered[row];
                            }
                        }
                        ++sourceBands;
                    }
                }
                for (auto c : covered) assert(c == 1);
                checkedRows += info.rows;
                ++cases;
            }
        }
    }
    // Keep the ordinary training capacity cut: [8,8] with capacity 10 keeps [8,2].
    std::vector<uint32_t> prefix{8, 16};
    PeerCounts<Prefix> counts{Prefix{prefix}, 1, 0, 10};
    const auto limits = MegaMoeImpl::GetPeerRowLimits(counts, 2);
    MegaMoeImpl::PeerTransferCursor<PeerCounts<Prefix>> cursor(counts, limits, 0, 10, 0, 10, 2, 0);
    MegaMoeImpl::PeerTransfer transfer{};
    uint32_t perPeer[2]{};
    while (cursor.Next(transfer)) perPeer[transfer.sourceRank] += transfer.rowCount;
    assert(perPeer[0] == 8 && perPeer[1] == 2);
    std::cout << "PASS peer rows: " << cases << " expert/rank cases, " << checkedRows << " rows, " << sourceBands
              << " source bands; EP128, empty/hot/exhausted peers, tails and capacity cuts\n";
}
