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
#include "experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_bf16_peer_row_layout.hpp"
#include "experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_workspace_backward.hpp"
#include "experiments/megamoe/csrc/kernels/mega_moe/stage/mega_moe_staged_wave_backward.hpp"

using namespace MegaMoeBF16Impl::Wave;
using Row = std::pair<uint32_t, uint32_t>;
struct Prefix {
    const std::vector<uint32_t>& values;
    uint32_t Get(uint32_t i) const
    {
        return values[i];
    }
};

// Encode dY as (peer, route ordinal) so both a duplicate and a silent row
// permutation are visible. Check the exact M/N rectangles handed to Cube.
static void CheckBackwardWaveRows()
{
    constexpr uint32_t cores = 32, experts = 4;
    // Each peer falls three rows short of a wave. The first wave still
    // draws one complete canonical tile from each of six different peers.
    const std::vector<uint32_t> peerEnds{765, 1530, 2295, 3060, 3825, 4590, 5355, 6120};
    const MegaMoeBF16Impl::BackwardWavePlan geometry(peerEnds.back(), 3072, cores, 0, 0);
    const auto chunk = geometry.SelectChunkTiles(Prefix{peerEnds}, 8, 1, 0);
    const MegaMoeBF16Impl::BackwardWavePlan firstWave(peerEnds.back(), 3072, cores, 0, 0, chunk);
    for (uint32_t i = 0; i < firstWave.groups; ++i) {
        const auto row = firstWave.TileOffset(i) * firstWave.tileM;
        assert(row >= i * 765 && row + firstWave.tileM <= peerEnds[i]);
    }
    std::mt19937 rng(20261120);
    for (uint32_t ep : {8U, 16U, 32U, 128U}) {
        for (uint32_t scenario = 0; scenario < 10; ++scenario) {
            std::vector<uint32_t> prefix(ep * experts);
            std::vector<std::vector<Row>> expected(experts);
            for (uint32_t peer = 0; peer < ep; ++peer) {
                for (uint32_t e = 0; e < experts; ++e) {
                    uint32_t count = scenario == 0 ? 48U : rng() % 101U;
                    if (scenario == 2 && (e == 1 || peer % 3)) count = 0;
                    if (scenario == 3) count = peer == 4 ? 1 + 128 * e : 0;
                    // Tall, ragged peer ranges, mixed empty/hot experts, and a single active source.
                    if (scenario == 5) count = peer % 8 == 0 ? 1547U : peer % 7;
                    if (scenario == 6) count = peer < 6 ? 4096U : 0;
                    if (scenario == 7) count = peer == ep / 2 ? 8193U : 0;
                    if (scenario == 8) count = (peer + e) % 7 == 0 ? 847U + rng() % 401U : rng() % 29U;
                    if (scenario == 9) count = peer == 0 ? 400U : peer == 1 ? 1600U : peer == 2 ? 8192U : 0U;
                    for (uint32_t j = 0; j < count; ++j) expected[e].emplace_back(peer, j);
                    prefix[peer * experts + e] = expected[e].size();
                }
            }
            for (uint32_t rank : {0U, ep / 2, ep - 1}) {
                uint32_t previous = 0;
                for (uint32_t e = 0; e < experts; ++e) {
                    uint32_t m = expected[e].size();
                    if (scenario == 4) m = std::min(m, 513U);
                    if (scenario == 9) m = 1399;
                    uint32_t n = ep == 16 ? 2304 : 3072, cols = (n + 255) / 256;
                    const uint32_t sourceBegin = rank ? prefix[(rank - 1) * experts + e] : 0;
                    const MegaMoeBF16Impl::BackwardWavePlan geometry(m, n, cores, sourceBegin, previous);
                    const uint32_t chunk = geometry.SelectChunkTiles(Prefix{prefix}, ep, experts, e);
                    if (m < expected[e].size()) {
                        auto admitted = prefix;
                        for (uint32_t peer = 0; peer < ep; ++peer) {
                            auto& end = admitted[peer * experts + e];
                            end = std::min(end, m);
                        }
                        assert(chunk == geometry.SelectChunkTiles(Prefix{admitted}, ep, experts, e));
                    }
                    const MegaMoeBF16Impl::BackwardWavePlan plan(m, n, cores, sourceBegin, previous, chunk);
                    std::vector<unsigned> tileVisits(plan.tiles);
                    for (uint32_t visit = 0; visit < plan.tiles; ++visit) {
                        const auto tile = (plan.pivot + plan.TileOffset(visit)) % plan.tiles;
                        assert(plan.WaveId(tile * 128) == (previous + visit) / plan.groups);
                        assert(++tileVisits[tile] == 1);
                    }
                    std::vector<Row> actual(m);
                    std::vector<unsigned> visited(m), products(plan.tiles * cols), arrivals(plan.tiles);
                    std::vector<unsigned> required(plan.tiles), writers(plan.tiles);
                    for (uint32_t lane = 0; lane < cores; ++lane) {
                        plan.ForEach(lane, [&](uint32_t row, uint32_t rows, uint32_t member, uint32_t owners) {
                            assert(row % 128 == 0 && rows == std::min(128U, m - row));
                            const auto band = MegaMoeImpl::GetBalancedTokenRange(rows, member, owners);
                            if (band.count) {
                                MegaMoeBF16Impl::VisitCanonicalPeerRows(Prefix{prefix}, ep, experts, e,
                                                                        row + band.start, row + band.start + band.count,
                                                                        [&](auto peer, auto j, auto dst, auto len) {
                                                                            for (uint32_t i = 0; i < len; ++i) {
                                                                                actual[dst + i] = {peer, j + i};
                                                                                ++visited[dst + i];
                                                                            }
                                                                        });
                                ++arrivals[row / 128];
                            }
                        });
                        plan.ForEachDhpRow(lane, cols, (uint64_t(previous) * cols) % cores,
                                           [&](uint32_t row, uint32_t, uint32_t firstCol, uint32_t dmaArrivals) {
                                               required[row / 128] = dmaArrivals;
                                               ++writers[row / 128];
                                               for (uint32_t col = firstCol; col < cols; col += cores)
                                                   ++products[row / 128 * cols + col];
                                           });
                    }
                    assert(arrivals == required && std::equal(actual.begin(), actual.end(), expected[e].begin()));
                    assert(std::all_of(visited.begin(), visited.end(), [](unsigned v) { return v == 1; }));
                    assert(std::all_of(products.begin(), products.end(), [](unsigned v) { return v == 1; }));
                    assert(std::all_of(writers.begin(), writers.end(),
                                       [&](unsigned v) { return v == std::min(cols, cores); }));
                    previous += plan.tiles;
                }
            }
        }
    }
}

static void CheckDhpWaveOwnership()
{
    for (uint32_t cores : {24U, 32U}) {
        for (uint32_t cols : {1U, 9U, 12U, 32U, 37U, 67U}) {
            unsigned previous = 0;
            std::vector<unsigned> work(cores);
            for (uint32_t m : {0U, 33U, 511U, 16384U, 65U, 1025U}) {
                const MegaMoeBF16Impl::BackwardWavePlan plan(m, cols * 256, cores, m / 2, previous);
                std::vector<unsigned> visits(plan.tiles * cols);
                for (unsigned lane = 0; lane < cores; ++lane) {
                    plan.ForEachDhpRow(lane, cols, previous * cols % cores,
                                       [&](unsigned row, unsigned rows, unsigned firstCol, unsigned) {
                                           assert(rows == std::min(128U, m - row));
                                           unsigned ordinal = (row / 128 + plan.tiles - plan.pivot) % plan.tiles;
                                           for (unsigned col = firstCol; col < cols; col += cores) {
                                               assert(((previous + ordinal) * cols + col) % cores == lane);
                                               ++visits[row / 128 * cols + col];
                                               ++work[lane];
                                           }
                                       });
                }
                assert(std::all_of(visits.begin(), visits.end(), [](unsigned count) { return count == 1; }));
                previous += plan.tiles;
            }
            const auto limits = std::minmax_element(work.begin(), work.end());
            assert(*limits.second - *limits.first <= 1);
        }
    }

    // EP16: three experts with 16384 rows and F2304 share 3456 tiles across 32 AICs.
    std::vector<unsigned> work(32);
    for (unsigned expert = 0; expert < 3; ++expert) {
        const MegaMoeBF16Impl::BackwardWavePlan plan(16384, 2304, 32, 0, expert * 128);
        for (unsigned lane = 0; lane < 32; ++lane)
            plan.ForEachDhpRow(lane, 9, expert * 128 * 9 % 32, [&](unsigned, unsigned, unsigned firstCol, unsigned) {
                assert(firstCol < 9);
                ++work[lane];
            });
    }
    assert(std::all_of(work.begin(), work.end(), [](unsigned count) { return count == 108; }));
}

static void CheckTokenCountExchange()
{
    using namespace MegaMoeBF16Impl;
    static_assert(!UseCompactTokenCountExchange(8, 4));
    static_assert(UseCompactTokenCountExchange(512, 4));
    static_assert(!UseCompactTokenCountExchange(512, 9));
    constexpr CompactTokenCountExchangeLayout dense(8, 4, false);
    constexpr CompactTokenCountExchangeLayout compact(8, 4, true);
    static_assert(compact.enabled && !dense.enabled);
    static_assert(compact.packetBytes >= 8 * kCompactTokenCountPacketBytes);
    static_assert(compact.blockSumsOffset == compact.packetOffset + compact.packetBytes);
    static_assert(compact.tokenCountsOffset == compact.blockSumsOffset + compact.blockSumsBytes);
    static_assert(compact.payloadOffset - dense.payloadOffset == compact.packetBytes + compact.blockSumsBytes);
    static_assert(compact.tokenCountStride == dense.tokenCountStride &&
                  compact.tokenCountsBytes == dense.tokenCountsBytes);
}

static void CheckForwardPeerTopology()
{
    constexpr uint32_t peers = 16;
    std::vector<uint32_t> countsPrefix(peers), sourcePrefix(peers, 0);
    for (uint32_t peer = 0; peer < peers; ++peer) countsPrefix[peer] = 2 * (peer + 1);
    for (uint32_t ranksPerNode : {0U, 8U}) {
        const PeerRowLayout<Prefix> layout{Prefix{countsPrefix}, Prefix{sourcePrefix}, peers, 1, 0, ranksPerNode};
        MegaMoeImpl::PeerRowLimitsCache cache{};
        for (uint32_t wave = 0; wave < 2; ++wave) {
            const Slice slice{0, wave * peers, peers, wave, 2 * peers, wave * peers};
            uint32_t visited = 0;
            layout.ForEachSpan(
                slice, slice.packed, slice.packed + peers, cache, [&](const auto& transfer, uint64_t packed) {
                    const uint32_t peer = ranksPerNode == 0 ? visited : visited / 2 + (visited % 2 == 0 ? 8 : 0);
                    assert(packed == slice.packed + visited && transfer.rowCount == 1 && transfer.sourceRank == peer);
                    assert(layout.SourceRow(slice, transfer) == wave &&
                           layout.SavedRow(slice, transfer) == 2 * peer + wave);
                    ++visited;
                });
            assert(visited == peers);
        }
    }
}

int main()
{
    CheckForwardPeerTopology();
    CheckTokenCountExchange();
    CheckBackwardWaveRows();
    CheckDhpWaveOwnership();
    // A prefix above 2^32 still rotates the remainder to the correct worker.
    constexpr uint64_t wavePrefix = (uint64_t{1} << 32) + 1;
    static_assert(MegaMoeImpl::GetRotatedBalancedTokenRange(8, 2, 3, wavePrefix).count == 3);
    static_assert(MegaMoeImpl::GetRotatedBalancedTokenRange(8, 1, 3, wavePrefix).start == 6);
    static_assert(MegaMoeImpl::GetBalancedTokenRange(2, 3, 4).count == 0);
    static_assert(MegaMoeImpl::GetGmmFirstOwnedTile(0, 4, 1) == 3);
    static_assert(MegaMoeImpl::GetGmmFirstOwnedTile(0, 4, 1, true) == 2);
    static_assert(MegaMoeImpl::GetGmmNextStartBlock(3, 3, 4) == 2);
    // Source tokens are shared by all TopK slots; indices, weights and returns are per route.
    constexpr uint64_t payloadBase = 8192, sourceRows = 129, hidden = 256, topK = 3;
    constexpr MegaMoeBF16Impl::PeerPayloadLayout bf16Payload(payloadBase, sourceRows, hidden, topK, 2);
    constexpr MegaMoeBF16Impl::PeerPayloadLayout fp32Payload(payloadBase, sourceRows, hidden, topK, 4);
    static_assert(bf16Payload.routeIndexOffset == payloadBase + sourceRows * hidden * 2);
    static_assert(bf16Payload.weightOffset == bf16Payload.routeIndexOffset + 2048);
    static_assert(fp32Payload.weightOffset == bf16Payload.weightOffset);
    static_assert(bf16Payload.returnOffset == bf16Payload.weightOffset + 1024);
    static_assert(fp32Payload.returnOffset == fp32Payload.weightOffset + 2048);
    static_assert(fp32Payload.topkGradOffset == fp32Payload.returnOffset + sourceRows * topK * hidden * 2);
    static_assert(bf16Payload.topkGradOffset == bf16Payload.returnOffset + sourceRows * topK * hidden * 2);
    static_assert(fp32Payload.totalBytes == fp32Payload.topkGradOffset + 2048);
    static_assert(bf16Payload.totalBytes == bf16Payload.topkGradOffset + 1024);
    using MegaMoeBF16Impl::BackwardWorkspaceLayout;
    const BackwardWorkspaceLayout saved(4, 24576, 7168, 6144, 4, MegaMoeBF16Impl::kSaveBf16, 64);
    assert(saved.dzOffset == 24576ULL * 7168U * sizeof(uint16_t));
    assert(saved.stagedReadyOffset != 0);
    assert(saved.totalBytes - saved.stagedReadyOffset == (24576 / 128 + 4) * 64);
    using namespace MegaMoeBF16Impl;
    static_assert(UseBackwardStagedWave(kSaveBf16, 2048) && UseBackwardStagedWave(kSaveMxFp8, 2080));
    static_assert(UseBackwardStagedWave(kSaveBf16, 16384) && UseBackwardStagedWave(kSaveMxFp8, 16384));
    static_assert(!UseBackwardStagedWave(kSaveMxFp8, 16416));
    static_assert(!UseBackwardStagedWave(kSaveBf16, 2080) && !UseBackwardStagedWave(kSaveBf16, 16448));
    static_assert(!UseBackwardStagedWave(kSaveNone, 2048) && !UseBackwardStagedWave(kSaveMxFp8, 0));
    static_assert(UseRecomputeDhpRowReady(kSaveNone, 3072) && !UseRecomputeDhpRowReady(kSaveNone, 16640));
    static_assert(!UseRecomputeDhpRowReady(kSaveNone, 2080) && !UseRecomputeDhpRowReady(kSaveBf16, 3072));
    for (auto mode : {kSaveBf16, kSaveMxFp8, kSaveNone}) {
        for (auto localExperts : {4U, 8U}) {
            assert(BackwardDispatchSplitExperts(mode, 16384, localExperts) == localExperts);
            assert(BackwardDispatchSplitExperts(mode, 16640, localExperts) == 0U);
        }
        const BackwardWorkspaceLayout escape(16, 24576, 7168, 4096, 0, mode, 64);
        const BackwardWorkspaceLayout wave(16, 24576, 7168, 4096, 16, mode, 64);
        assert(!escape.stagedReadyOffset && wave.stagedReadyOffset);
    }
    assert(MetadataFits(128, 4));
    assert(MetadataFits(1024, 2));
    assert(MetadataBytes(8, 4) == 4096 && MetadataBytes(1024, 2) == 16384);
    assert(ScratchFits(7168, 1024, 2));
    assert(!ScratchFits(32768, 8, 4));
    for (auto mode : {kSaveBf16, kSaveNone, kSaveMxFp8}) {
        const BackwardWorkspaceLayout wide(4, 128, 256, 2 * 16640, 0, mode, 64);
        assert(wide.stagedReadyOffset == 0);
        assert(wide.wideDpOffset && wide.controlOffset - wide.wideDpOffset == 64U * 16640U * sizeof(float));
        assert(MegaMoeBF16Impl::BackwardTopkPartialCount(mode, 16640) == 1);
    }
    static_assert(MegaMoeBF16Impl::BackwardTopkPartialCount(MegaMoeBF16Impl::kSaveBf16, 3072) == 1);
    static_assert(MegaMoeBF16Impl::BackwardTopkPartialCount(MegaMoeBF16Impl::kSaveNone, 3072) == 1);
    static_assert(MegaMoeBF16Impl::BackwardTopkPartialCount(MegaMoeBF16Impl::kSaveMxFp8, 3072) == 1);
    // These reduction widths reproduce the reference sum order for dP.
    static_assert(MegaMoeBF16Impl::MakeTopkGradReductionPlan(24467, 3072, 64).width == 3072);
    static_assert(MegaMoeBF16Impl::MakeTopkGradReductionPlan(24258, 3072, 64).width == 1536);
    static_assert(MegaMoeBF16Impl::MakeTopkGradReductionPlan(24693, 3072, 64).width == 1536);
    constexpr auto grouped = MegaMoeBF16Impl::MakeTopkGradReductionPlan(1539, 16640, 64);
    static_assert(grouped.width == 256 && grouped.stripesPerGroup == 33);
    constexpr auto shortTail = MegaMoeBF16Impl::MakeTopkGradReductionPlan(498, 2560, 64);
    static_assert(shortTail.width == 504 && shortTail.stripesPerGroup == 6);
    constexpr auto fourGroups = MegaMoeBF16Impl::MakeTopkGradReductionPlan(486, 1024, 64);
    static_assert(fourGroups.width == 256 && fourGroups.stripesPerGroup == 1);
    for (uint32_t f = 64; f <= 16384; f += 64) {
        const auto batch = MegaMoeBF16Impl::TopkGradBatchRows(f);
        assert(batch >= 1);
        assert(batch * 10U * f + MegaMoeBF16Impl::kWholeRowTopkResultBytes <= 248U * 1024U);
    }
    // Row ownership is independent of the 2*AIC lanes used to choose the sum tree.
    for (uint32_t f : {64U, 3072U, 16384U}) {
        const auto batch = MegaMoeBF16Impl::TopkGradBatchRows(f);
        for (uint32_t readers : {1U, 32U, 36U}) {
            for (uint32_t rows : {0U, 1U, 31U, 32U, 33U, 255U, 256U, 257U, 24991U}) {
                std::vector<uint32_t> visited(rows);
                for (uint32_t reader = 0; reader < readers; ++reader) {
                    for (uint32_t begin = reader * batch; begin < rows; begin += readers * batch)
                        for (uint32_t row = begin; row < std::min(begin + batch, rows); ++row) ++visited[row];
                }
                for (auto count : visited) assert(count == 1);
            }
        }
    }
    using Quant = MegaMoeBF16Impl::SavedGateUpQuantUbLayoutFor<256, 64, 128>;
    static_assert(Quant::gateData == 2U * 256U * 128U * 2U);
    static_assert(Quant::totalBytesWithClipMasks + 256U * sizeof(float) <= 248U * 1024U);
    for (uint64_t capacity : {0ULL, 1ULL, 255ULL, 256ULL, 257ULL, 24576ULL}) {
        const MegaMoeBF16Impl::ForwardTransientWorkspaceLayout transient(6144, 4, capacity);
        const ForwardWorkspaceLayout bf16(transient.totalBytes, capacity, 4, 7168, 2, 4096, 3, capacity);
        const ForwardWorkspaceLayout other(transient.totalBytes, capacity, 4, 7168, 4, 4096, 3, capacity);
        const ForwardWorkspaceLayout dedicated(transient.totalBytes, capacity, 4, 7168, 2, 4096, 0, capacity);
        assert(dedicated.readyOffset == bf16.readyOffset);
        assert(dedicated.totalBytes - bf16.totalBytes == bf16.rowsInY * 7168U * sizeof(uint16_t));
        assert(bf16.rawGateOffset - bf16.weightOffset == (capacity * 2 + 511) / 512 * 512);
        assert(other.rawGateOffset - other.weightOffset == (capacity * 4 + 511) / 512 * 512);
        assert(bf16.rawUpOffset - bf16.rawGateOffset == RawTileOffset(kMaxCores, 0) * sizeof(uint16_t));
        assert(bf16.totalBytes - bf16.rawUpOffset == bf16.rawUpOffset - bf16.rawGateOffset);
        assert(other.totalBytes - other.rawGateOffset == bf16.totalBytes - bf16.rawGateOffset);
    }
    // Interleaved Hp follows the reused physical slot; staged Hp keeps its logical row.
    constexpr uint32_t slots = MegaMoeImpl::kWaveScratchSlots;
    constexpr uint32_t groupsPerWave = 3;
    constexpr uint64_t rowsPerWave = groupsPerWave * kGroupRows;
    constexpr uint64_t ringRows = slots * rowsPerWave;
    constexpr ForwardScratch dense(ringRows, groupsPerWave, 0);
    constexpr ForwardScratch interleaved(ringRows + 1, groupsPerWave, 0), staged(ringRows + 1, groupsPerWave, 1);
    static_assert(dense.slotRows == 0 && dense.Row(slots - 1, ringRows - rowsPerWave, ringRows - 1) == ringRows - 1);
    static_assert(interleaved.rows == ringRows && interleaved.hpRows == ringRows);
    static_assert(staged.rows == ringRows && staged.hpRows == ringRows + 1);
    static_assert(interleaved.Row(slots, ringRows, ringRows) == 0 && staged.Row(slots, ringRows, ringRows) == 0);
    static_assert(interleaved.HpRow(0, ringRows) == 0 && staged.HpRow(0, ringRows) == ringRows);
    for (uint32_t sourceRows : {1U, 128U, 768U, 769U, 4096U}) {
        for (uint32_t capacity : {0U, 1U, 767U, 768U, 4096U, 24576U}) {
            constexpr uint32_t groups = 3;
            const uint64_t cut = OutputRowsInY(capacity, sourceRows, groups);
            assert(cut <= capacity);
            for (uint32_t first = 0; first < capacity; ++first) {
                const uint32_t count = std::min(groups * kGroupRows, capacity - first);
                if (first < cut) {
                    assert(first + count <= sourceRows - 1);
                } else {
                    assert(first - cut + count <= capacity - cut);
                }
            }
        }
    }
    for (uint32_t mode : {1U, 2U, 3U}) {
        for (uint64_t capacity : {1ULL, 127ULL, 128ULL, 129ULL, 24576ULL}) {
            const MegaMoeBF16Impl::BackwardWorkspaceLayout layout(4, capacity, 7168, 6144, 4, mode, 64);
            assert(layout.doOffset == 0 && layout.controlOffset < layout.totalBytes);
            assert(layout.dzOffset % 512 == 0 && layout.hpOffset % 512 == 0 && layout.dpOffset % 512 == 0);
            assert(layout.dhpReadyOffset < layout.dzReadyOffset && layout.dzReadyOffset < layout.dxReadyOffset);
            if (layout.stagedReadyOffset) {
                assert(layout.totalBytes == layout.stagedReadyOffset + ((capacity + 127) / 128 + 4) * 64);
            }
        }
    }
    // Every expert owns disjoint row-counter slots, including empty/tail experts.
    for (uint32_t shift = 0; shift < 256; ++shift) {
        uint64_t base = 0;
        std::vector<uint64_t> slots;
        for (uint32_t expert = 0; expert < 8; ++expert) {
            const uint32_t rows = (shift * (expert + 1U)) % 387U;
            for (uint32_t row = 0; row < rows; row += 128U) slots.push_back(base / 128U + expert + row / 128U);
            base += rows;
        }
        std::sort(slots.begin(), slots.end());
        assert(std::adjacent_find(slots.begin(), slots.end()) == slots.end());
        assert(slots.empty() || slots.back() < (base + 127U) / 128U + 8U);
    }
    // A tail may have fewer producing AICs than the wave's full slices.
    for (uint32_t rows : {0U, 1U, 255U, 256U, 257U, 768U}) {
        const uint32_t tiles = MegaMoeImpl::GetGmm2SliceTiles<256, 256>(rows, 7168);
        const uint32_t expectedTiles = ((rows + 255) / 256) * 28;
        assert(tiles == expectedTiles);
        for (uint32_t start : {0U, 7U, 31U}) {
            for (bool reverse : {false, true}) {
                uint32_t producers = 0;
                for (uint32_t core = 0; core < 32; ++core) {
                    uint32_t cursor = start;
                    bool expected = false;
                    for (uint32_t tile = 0; tile < expectedTiles; ++tile) {
                        const uint32_t owner = (start + tile) % 32;
                        expected |= core == (reverse ? 31 - owner : owner);
                    }
                    const bool owned = MegaMoeImpl::AdvanceGmmSlice(core, 32, cursor, tiles, reverse);
                    assert(owned == expected && cursor == (start + tiles) % 32);
                    producers += owned;
                }
                assert(MegaMoeImpl::GetGmm2ProducerCount(tiles, 32) == producers);
            }
        }
    }
    static_assert(MegaMoeImpl::GetGmm2SliceTiles<128, 64>(129, 65) == 4);
    std::mt19937 rng(20261116);
    uint64_t cases = 0, checkedRows = 0, sourceBands = 0;
    for (uint32_t test = 0; test < 180; ++test) {
        constexpr uint32_t epSizes[] = {8, 16, 32, 64, 128, 256};
        const uint32_t peers = test >= 2 && test < 8 ? epSizes[test - 2]
                               : test == 0           ? 8
                               : test == 1           ? 128
                               : test % 7 == 0       ? 128
                               : test % 5 == 0       ? 64
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
            MegaMoeImpl::PeerRowLimitsCache cache{};
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
                    const auto& limits = MegaMoeImpl::GetCachedPeerRowLimits(cache, pc, e, peers);
                    const auto direct = MegaMoeImpl::GetPeerRowLimits(pc, peers);
                    assert(limits.minimum == direct.minimum && limits.maximum == direct.maximum &&
                           limits.total == direct.total);
                    const PeerRowLayout<Prefix> layout{Prefix{prefix}, Prefix{prefix}, peers, experts, rank};
                    layout.ForEachSpan(
                        slice, slice.packed, slice.packed + slice.rows, cache,
                        [&](const MegaMoeImpl::PeerTransfer& t, uint64_t packed) {
                            const uint32_t preceding = t.sourceRank ? prefix[(t.sourceRank - 1) * experts + e] : 0;
                            for (uint32_t j = 0; j < t.rowCount; ++j) {
                                const auto ref = expected[packed + j - info.base];
                                assert(ref.first == t.sourceRank);
                                assert(layout.SavedRow(slice, t) + j == info.base + preceding + ref.second);
                            }
                        });
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
                                                              begin, info.rows);
                    MegaMoeImpl::PeerTransfer transfer{};
                    uint32_t next = begin;
                    while (inverse.Next(transfer) && transfer.destinationRow < end) {
                        assert(transfer.destinationRow == next);
                        const uint32_t ownedRows = std::min(transfer.rowCount, end - next);
                        for (uint32_t j = 0; j < ownedRows; ++j, ++next) {
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
                // Runs are independent of the parallel writers' 128-row ownership cuts.
                std::vector<uint32_t> wholeRuns(info.rows);
                PeerRowCursor<PeerCounts<Prefix>> whole(pc, peers, rank, groups, info.rows, info.firstSliceRows, 0,
                                                        info.rows);
                MegaMoeImpl::PeerTransfer full{};
                while (whole.Next(full)) {
                    const uint32_t base = full.sourceRank ? prefix[(full.sourceRank - 1U) * experts + e] : 0U;
                    for (uint32_t j = 0; j < full.rowCount; ++j)
                        wholeRuns[base + full.sourceOrdinalBegin + j] = full.rowCount - j;
                }
                assert(runs == wholeRuns);
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
