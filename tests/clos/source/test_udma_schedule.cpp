// Compile on a CPU: c++ -std=c++17 -O2 -D__CCE_KT_TEST__ -Itest/udma_cpu_stubs
// -Iinclude test/test_udma_schedule.cpp -o /tmp/test_udma_schedule
// Executes the production scheduling headers against an ordered SQ test double.
#ifdef NDEBUG
#undef NDEBUG  // The CPU oracle must remain active in optimized builds.
#endif
#include "ascend_deepep/kernels/udma_dispatch_schedule.h"
#include "ascend_deepep/kernels/udma_combine_schedule.h"
#include "ascend_deepep/kernels/udma_token_combine_schedule.h"
#include "ascend_deepep/kernels/udma_dispatch_metadata.h"
#include "ascend_deepep/kernels/notify_recv_prefix.h"
#include "ascend_deepep/kernels/notify_dispatch_plan.h"
#include <iostream>
#include <tuple>
#include <set>
#include <algorithm>

using namespace ascend_deepep;
using namespace ascend_deepep::udma;
using Entry = std::tuple<uint64_t, uint64_t, uint32_t>;
using Expected = std::vector<std::vector<Entry>>;
bool corrupt_oracle = false, corrupt_combine_oracle = false, corrupt_metadata_oracle = false;
bool corrupt_notify_oracle = false;
unsigned pipeline_chunk_test = 0;
static_assert(sizeof(DispatchTilingData) == 264);
static_assert(sizeof(CombineTilingData) == 176);

void CheckRequests(const Expected& expected, uint8_t* workspace, uint8_t* input, uint8_t* side, unsigned rank,
                   unsigned world, unsigned core, unsigned group, uint64_t signal, unsigned order = 0,
                   unsigned ranks_per_group = 0, unsigned lanes = 1, unsigned qps = 1)
{
    std::vector<unsigned> position(world);
    std::set<uint64_t> queues;
    unsigned markers = 0;
    bool marked = false;
    for (const auto& r : model::sent) {
        CHECK(r.peer != rank && Owner(rank, r.peer, world, group, order, ranks_per_group) == core / lanes &&
              r.qp / qps == core);
        queues.insert(model::state(r));
        if (r.order == ACLSHMEMX_UDMA_ODR_SO) {
            marked = true;
            ++markers;
            CHECK(r.dst == workspace + signal + rank * kSignalBytes + ((core % lanes) * qps + r.qp % qps) * 8 &&
                  r.src == r.dst && r.bytes == 8);
        } else {
            CHECK(!marked);
            auto& list = expected[r.peer];
            CHECK(position[r.peer] < list.size());
            const auto [dst, src, bytes] = list[position[r.peer]++];
            CHECK(r.dst == workspace + dst && r.bytes == bytes);
            // High bit chooses the independently generated auxiliary-payload oracle.
            CHECK(r.src == ((src >> 63) ? side : input) + (src & ~(uint64_t{1} << 63)));
        }
    }
    for (unsigned p = 0; p < world; ++p)
        if (p != rank && Owner(rank, p, world, group, order, ranks_per_group) == core / lanes)
            CHECK(position[p] == expected[p].size());
    unsigned count = 0;
    for (unsigned peer = 0; peer < world; ++peer)
        count += peer != rank && Owner(rank, peer, world, group, order, ranks_per_group) == core / lanes;
    CHECK(markers == count * qps && model::quiesced.size() == queues.size());
    std::set<uint64_t> drained;
    for (auto [p, q] : model::quiesced) CHECK(drained.insert(model::contexts[p][q].state_addr).second);
    CHECK(drained == queues && model::staged.empty());
}

// Execute the real notify builder and check it against a scalar token-major
// oracle. Poisoned padding must stay untouched, including empty channels.
std::vector<int32_t> Plan(const std::vector<uint8_t>& routes, DispatchTilingData d, std::vector<int32_t>& head)
{
    const unsigned length = (d.num_tokens + d.num_channels - 1) / d.num_channels;
    const unsigned stride = (length + 7) / 8 * 8;
    std::vector<int32_t> list(size_t(d.world_size) * d.num_channels * stride, -777), gold = list;
    std::vector<int32_t> expected_head(head.size(), -1), counts(d.world_size * d.num_channels);
    for (unsigned t = 0; t < d.num_tokens; ++t)
        for (unsigned p = 0; p < d.world_size; ++p)
            if (routes[t * d.world_size + p]) {
                const unsigned ch = t / length;
                auto& count = counts[p * d.num_channels + ch];
                expected_head[t * d.world_size + p] = count;
                gold[(size_t(p) * d.num_channels + ch) * stride + count++] = t;
            }
    d.rank_stride = (d.world_size + 7) / 8 * 8;
    for (unsigned ch = d.num_channels; ch-- > 0;) {
        Resource resource;
        BuildDispatchPlan(resource, const_cast<uint8_t*>(routes.data()), reinterpret_cast<uint8_t*>(list.data()),
                          reinterpret_cast<uint8_t*>(head.data()), d, ch);
        AscendC::CheckDrained();
        auto actual = resource.GetBufferByByte<int32_t>(0);
        for (unsigned p = 0; p < d.rank_stride; ++p)
            CHECK(actual.GetValue(p) == (p < d.world_size ? counts[p * d.num_channels + ch] : 0));
    }
    CHECK(list == gold && head == expected_head);
    // Legacy cached handles rebuild only the list; no send_head is required.
    std::fill(list.begin(), list.end(), -777);
    for (unsigned ch = 0; ch < d.num_channels; ++ch) {
        Resource resource;
        BuildDispatchPlan(resource, const_cast<uint8_t*>(routes.data()), reinterpret_cast<uint8_t*>(list.data()),
                          nullptr, d, ch);
        AscendC::CheckDrained();
    }
    CHECK(list == gold);
    return list;
}

// Independent peer-list oracle for the production sender model.
std::vector<unsigned> ReferencePeers(unsigned world, unsigned rank, unsigned peers, unsigned order, unsigned size)
{
    std::vector<unsigned> result;
    if (!order) {
        for (unsigned step = 1; step < world; ++step) result.push_back((rank + step) % world);
        return result;
    }
    for (unsigned local = 0; local < size; ++local)
        for (unsigned group = 1; group <= world / size; ++group) {
            const unsigned peer = ((rank / size + group) % (world / size)) * size + (rank % size + local) % size;
            if (peer != rank) result.push_back(peer);
        }
    if (order == 2)
        for (unsigned first = 0; first < result.size(); first += peers) {
            const unsigned count = std::min(peers, unsigned(result.size()) - first);
            const unsigned shift = (rank % size + first / peers) % count;
            std::rotate(result.begin() + first, result.begin() + first + shift, result.begin() + first + count);
        }
    return result;
}

std::vector<unsigned> ReferenceWorker(unsigned world, unsigned rank, unsigned count, unsigned order, unsigned size,
                                      unsigned core)
{
    std::vector<unsigned> result;
    if (order == 3) {
        const unsigned workers = (size + count - 1) / count;
        if (core >= workers) return result;
        // Independent nested group/position enumeration, never compact self.
        for (unsigned step = 1; step <= world / size; ++step)
            for (unsigned local = core; local < size; local += workers) {
                unsigned peer = ((rank / size + step) % (world / size)) * size + local;
                if (peer != rank) result.push_back(peer);
            }
    } else {
        auto all = ReferencePeers(world, rank, count, order, size);
        for (unsigned i = core * count; i < std::min(unsigned(all.size()), (core + 1) * count); ++i)
            result.push_back(all[i]);
    }
    return result;
}

void Case(unsigned world, unsigned rank, unsigned tokens, unsigned group, unsigned visit, bool auxiliary,
          unsigned aliases, unsigned dispatch_order = 0, unsigned combine_order = 0, unsigned ranks_per_group = 0,
          unsigned lanes = 1, unsigned qps = 1)
{
    constexpr unsigned channels = 32, xbytes = 16, sbytes = 8;

    std::vector<uint8_t> route(tokens * world), x(tokens * xbytes + 1), side(tokens * sbytes + 1);
    std::vector<int32_t> prefix(world * world), channel(world * channels);
    Expected expected(world);
    for (unsigned p = 0; p < world; ++p) {
        unsigned ordinal = 0;
        for (unsigned ch = 0; ch < channels; ++ch) {
            for (unsigned t = ch * ((tokens + 31) / 32); t < std::min(tokens, (ch + 1) * ((tokens + 31) / 32)); ++t) {
                // Include empty destinations, skew, and non-aligned short tails.
                if (p % 7 == 3 || (t * 17 + p * 13 + rank) % 5 == 0) continue;
                route[t * world + p] = 1;
                expected[p].emplace_back((ordinal + 7) * xbytes, t * xbytes, xbytes);
                if (auxiliary)
                    expected[p].emplace_back(300000 + (ordinal + 7) * sbytes, (uint64_t{1} << 63) + t * sbytes, sbytes);
                ++ordinal;
            }
            channel[p * channels + ch] = ordinal;
        }
        if (rank)
            prefix[(rank - 1) * world + p] = 7;
        else
            for (auto& e : expected[p]) std::get<0>(e) -= (std::get<1>(e) >> 63) ? 7 * sbytes : 7 * xbytes;
    }
    if (corrupt_oracle) {
        for (unsigned p = 0; p < world; ++p)
            if (p != rank && !expected[p].empty()) {
                ++std::get<0>(expected[p].front());
                break;
            }
    }
    std::vector<uint8_t> workspace(600000 + world * kSignalBytes);
    DispatchTilingData d{};
    d.rank = rank;
    d.world_size = world;
    d.num_tokens = tokens;
    d.num_channels = channels;
    d.hidden = 8;
    d.x_element_bytes = 2;
    d.has_x_scales = auxiliary;
    d.num_scales = 2;
    d.scale_element_bytes = 4;
    d.x_scales_buffer_offset = 300000;
    d.udma_completion_signal_offset = 600000;
    d.udma_peer_order = dispatch_order;
    d.udma_ranks_per_group = ranks_per_group;
    d.udma_peers_per_worker = group;
    d.udma_wqes_per_visit = visit;
    d.udma_workers_per_peer = lanes;
    d.udma_qps_per_worker = qps;
    // This sender-only model supplies already-arrived remote markers. Dedicated
    // tests below exercise delayed and missing markers against the real wait.
    for (unsigned p = 0; p < world; ++p) {
        const uint64_t one = 1;
        for (unsigned lane = 0; lane < lanes * qps; ++lane)
            std::memcpy(workspace.data() + d.udma_completion_signal_offset + p * kSignalBytes + lane * 8, &one,
                        sizeof(one));
    }
    std::vector<int32_t> head(tokens * world, -777);
    auto list = Plan(route, d, head);
    for (unsigned core = 0; core < channels; ++core) {
        Resource resource;
        model::reset(aliases);
        model::active_qp = core * qps;
        model::active_qp_count = qps;
        DispatchRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(list.data()),
                       reinterpret_cast<uint8_t*>(prefix.data()), reinterpret_cast<uint8_t*>(channel.data()),
                       workspace.data(), d, core, channels);
        Expected owned(world);
        for (unsigned peer = 0; peer < world; ++peer)
            for (const auto& e : expected[peer]) {
                const uint64_t src = std::get<1>(e);
                const unsigned token = (src & ~(uint64_t{1} << 63)) / ((src >> 63) ? sbytes : xbytes);
                if ((token / ((tokens + channels - 1) / channels)) % lanes == core % lanes) owned[peer].push_back(e);
            }
        CheckRequests(owned, workspace.data(), x.data(), side.data(), rank, world, core, group, 600000, dispatch_order,
                      ranks_per_group, lanes, qps);
        const auto dispatch_peers = ReferenceWorker(world, rank, group, dispatch_order, ranks_per_group, core / lanes);
        const unsigned dispatch_count = dispatch_peers.size();
        // Exact original input-tile / peer / field rotation, not just per-peer
        // contents. Build queues directly from the dense route matrix oracle.
        std::vector<unsigned> wanted, actual, wanted_qp, actual_qp, next_qp(world);
        for (unsigned ch = core % lanes; ch < channels; ch += lanes) {
            unsigned begin = std::min(tokens, ch * ((tokens + channels - 1) / channels));
            unsigned end = std::min(tokens, (ch + 1) * ((tokens + channels - 1) / channels));
            unsigned tile = std::min(256U, 65536U / world);
            for (unsigned start = begin; start < end; start += tile) {
                std::vector<unsigned> pending(dispatch_count);
                for (unsigned p = 0; p < dispatch_count; ++p) {
                    unsigned peer = dispatch_peers[p];
                    for (unsigned t = start; t < std::min(end, start + tile); ++t)
                        pending[p] += route[t * world + peer] * (auxiliary ? 2 : 1);
                }
                while (std::any_of(pending.begin(), pending.end(), [](unsigned n) { return n != 0; })) {
                    for (unsigned p = 0; p < pending.size(); ++p) {
                        unsigned n = std::min(visit, pending[p]);
                        pending[p] -= n;
                        for (unsigned j = 0; j < n; ++j) {
                            wanted.push_back(dispatch_peers[p]);
                            wanted_qp.push_back(core * qps + next_qp[dispatch_peers[p]]);
                        }
                        if (n) next_qp[dispatch_peers[p]] = (next_qp[dispatch_peers[p]] + 1) % qps;
                    }
                }
            }
        }
        for (const auto& request : model::sent)
            if (request.order != ACLSHMEMX_UDMA_ODR_SO) {
                actual.push_back(request.peer);
                actual_qp.push_back(request.qp);
            }
        CHECK(actual == wanted && actual_qp == wanted_qp);
    }
    // Independent combine handle: ordered per-source rows and exclusive channel begins.
    expected = Expected(world);
    std::vector<int32_t> indices;
    std::fill(prefix.begin(), prefix.end(), 0);
    for (unsigned p = 0; p < world; ++p) {
        const unsigned base = indices.size();
        for (unsigned ch = 0; ch < channels; ++ch) {
            channel[p * channels + ch] = indices.size() - base;
            for (unsigned t = ch * ((tokens + 31) / 32); t < std::min(tokens, (ch + 1) * ((tokens + 31) / 32)); ++t) {
                if (p % 7 == 3 || (t * 11 + p + rank) % 3 == 0) continue;
                const uint64_t row = indices.size(), slot = uint64_t(rank) * tokens + t;
                indices.push_back(t);
                expected[p].emplace_back(slot * xbytes, row * xbytes, xbytes);
                if (auxiliary)
                    expected[p].emplace_back(uint64_t(world) * tokens * xbytes + slot * sbytes,
                                             (uint64_t{1} << 63) + row * sbytes, sbytes);
            }
        }
        prefix[p * world + rank] = indices.size();
    }
    if (corrupt_combine_oracle) {
        for (unsigned p = 0; p < world; ++p)
            if (p != rank && !expected[p].empty()) {
                ++std::get<0>(expected[p].front());
                break;
            }
    }
    x.resize(indices.size() * xbytes + 1);
    side.resize(indices.size() * sbytes + 1);
    workspace.resize(uint64_t(world) * tokens * (xbytes + sbytes) + world * kSignalBytes + 1);
    CombineTilingData c{};
    c.rank = rank;
    c.world_size = world;
    c.num_channels = channels;
    c.hidden = 8;
    c.x_element_bytes = 2;
    c.num_tokens = tokens;
    c.token_capacity = tokens;
    c.num_topk = 2;
    c.has_topk_weights = auxiliary;
    c.x_row_stride = xbytes;
    c.topk_weights_row_stride = sbytes;
    c.topk_weights_buffer_offset = uint64_t(world) * tokens * xbytes;
    c.udma_completion_signal_offset = uint64_t(world) * tokens * (xbytes + sbytes);
    c.udma_peer_order = combine_order;
    c.udma_ranks_per_group = ranks_per_group;
    c.udma_peers_per_worker = group;
    c.udma_wqes_per_visit = visit;
    c.udma_workers_per_peer = lanes;
    c.udma_qps_per_worker = qps;
    for (unsigned p = 0; p < world; ++p) {
        const uint64_t one = 1;
        for (unsigned lane = 0; lane < lanes * qps; ++lane)
            std::memcpy(workspace.data() + c.udma_completion_signal_offset + p * kSignalBytes + lane * 8, &one,
                        sizeof(one));
    }
    const unsigned chunks = (indices.size() + 31) / 32;
    std::vector<uint64_t> ready((chunks + 1) * 64, 0);
    c.pipeline_chunk_tokens = pipeline_chunk_test;
    c.pipeline_ready = reinterpret_cast<uint64_t>(ready.data());
    std::map<uint8_t*, unsigned> polls;
    if (pipeline_chunk_test)
        AscendC::on_invalidate = [&](uint8_t* address) {
            const auto a = reinterpret_cast<uintptr_t>(address), base = reinterpret_cast<uintptr_t>(ready.data());
            if (a >= base && a < base + ready.size() * 8) {
                CHECK((a - base) % 512 == 0);
                if (++polls[address] == 3) *reinterpret_cast<uint64_t*>(address) = 1;
            }
        };
    for (unsigned core = 0; core < channels; ++core) {
        Resource resource;
        model::reset(aliases);
        model::active_qp = core * qps;
        model::active_qp_count = qps;
        if (pipeline_chunk_test)
            CombineRemote<3>(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(indices.data()),
                             reinterpret_cast<uint8_t*>(prefix.data()), nullptr, workspace.data(), c, core);
        else
            CombineRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(indices.data()),
                          reinterpret_cast<uint8_t*>(prefix.data()), reinterpret_cast<uint8_t*>(channel.data()),
                          workspace.data(), c, core);
        Expected owned(world);
        for (unsigned peer = 0; peer < world; ++peer) {
            const unsigned fields = auxiliary ? 2 : 1, rows = expected[peer].size() / fields;
            const unsigned begin = (uint64_t(rows) * (core % lanes) / lanes) * fields;
            const unsigned end = (uint64_t(rows) * (core % lanes + 1) / lanes) * fields;
            owned[peer].assign(expected[peer].begin() + begin, expected[peer].begin() + end);
        }
        CheckRequests(owned, workspace.data(), x.data(), side.data(), rank, world, core, group,
                      c.udma_completion_signal_offset, combine_order, ranks_per_group, lanes, qps);
        // The public visit quota applies across channel/tile boundaries. Derive
        // the exact peer order from the independent per-peer request lists.
        const auto combine_peers = ReferenceWorker(world, rank, group, combine_order, ranks_per_group, core / lanes);
        std::vector<unsigned> wanted, actual, wanted_qp, actual_qp, next_qp(world), pending(combine_peers.size());
        for (unsigned p = 0; p < combine_peers.size(); ++p) pending[p] = owned[combine_peers[p]].size();
        while (std::any_of(pending.begin(), pending.end(), [](unsigned n) { return n != 0; })) {
            for (unsigned p = 0; p < pending.size(); ++p) {
                const unsigned n = std::min(visit, pending[p]);
                pending[p] -= n;
                for (unsigned j = 0; j < n; ++j) {
                    wanted.push_back(combine_peers[p]);
                    wanted_qp.push_back(core * qps + next_qp[combine_peers[p]]);
                }
                if (n) next_qp[combine_peers[p]] = (next_qp[combine_peers[p]] + 1) % qps;
            }
        }
        for (const auto& request : model::sent)
            if (request.order != ACLSHMEMX_UDMA_ODR_SO) {
                actual.push_back(request.peer);
                actual_qp.push_back(request.qp);
            }
        CHECK(actual == wanted && actual_qp == wanted_qp);
    }
    AscendC::on_invalidate = {};
}

// Two-phase production schedules against an independent row/field oracle.
void EarlySingleCases()
{
    unsigned cases = 0;
    using Key = std::tuple<unsigned, unsigned, uint64_t, uint64_t, unsigned>;
    for (bool token_schedule : {false, true})
        for (unsigned world : {2U, 8U, 64U})
            for (unsigned cores : {32U, 64U})
                for (unsigned lanes : {1U, 2U})
                    for (unsigned group : {1U, 2U, 4U, 8U}) {
                        if ((token_schedule && lanes != 1) || (!token_schedule && Workers(world, group) * lanes > 32))
                            continue;
                        for (unsigned visit : {4U, 8U, 16U, 32U})
                            for (unsigned pattern : {0U, 1U, 2U, 3U})
                                for (bool aux : {false, true}) {
                                    const unsigned rank = world / 2, tokens = pattern == 3 ? 0 : 273;
                                    const unsigned xb = 16, wb = 8;
                                    std::vector<int32_t> prefix(world * world), src, slots;
                                    for (unsigned peer = 0; peer < world; ++peer) {
                                        for (unsigned t = 0; t < tokens; ++t) {
                                            if (peer % 7 == 3 || (t + peer) % 5 == 0) continue;
                                            src.push_back(t);
                                            slots.push_back(
                                                pattern == 0 || (pattern == 2 && t % 3) ? int(src.size() * 2) : -1);
                                        }
                                        prefix[peer * world + rank] = src.size();
                                    }
                                    std::vector<uint8_t> x(src.size() * xb + 1), side(src.size() * wb + 1),
                                        expanded((src.size() * 2 + 1) * xb + 1);
                                    CombineTilingData td{};
                                    td.rank = rank;
                                    td.world_size = world;
                                    td.num_channels = cores;
                                    td.hidden = 8;
                                    td.x_element_bytes = 2;
                                    td.token_capacity = std::max(tokens, 1U);
                                    td.num_topk = 2;
                                    td.has_topk_weights = aux;
                                    td.x_row_stride = xb;
                                    td.topk_weights_row_stride = wb;
                                    td.topk_weights_buffer_offset = uint64_t(world) * td.token_capacity * xb;
                                    td.udma_completion_signal_offset = uint64_t(world) * td.token_capacity * (xb + wb);
                                    td.udma_workers_per_peer = lanes;
                                    td.udma_peers_per_worker = group;
                                    td.udma_wqes_per_visit = visit;
                                    td.expanded_x = reinterpret_cast<uint64_t>(expanded.data());
                                    td.single_source_slots = reinterpret_cast<uint64_t>(slots.data());
                                    td.num_expanded_tokens = src.size() * 2 + 1;
                                    std::vector<uint8_t> workspace(td.udma_completion_signal_offset +
                                                                   world * kSignalBytes);
                                    for (unsigned peer = 0; peer < world; ++peer)
                                        for (unsigned w = 0; w < (token_schedule ? TokenWorkers(cores) : lanes); ++w) {
                                            uint64_t one = 1;
                                            std::memcpy(workspace.data() + td.udma_completion_signal_offset +
                                                            peer * kSignalBytes + w * 8,
                                                        &one, 8);
                                        }
                                    std::set<Key> expected_early, expected_late, actual_early, actual_late;
                                    unsigned begin = 0;
                                    for (unsigned peer = 0; peer < world; ++peer) {
                                        unsigned end = prefix[peer * world + rank];
                                        if (peer != rank)
                                            for (unsigned row = begin; row < end; ++row) {
                                                unsigned owner =
                                                    Owner(rank, peer, world, group) * lanes +
                                                    unsigned(lanes == 2 && row >= begin + (end - begin) / 2);
                                                if (token_schedule) {
                                                    const unsigned workers = TokenWorkers(cores);
                                                    owner = (row - begin + workers -
                                                             peer * ((workers + world - 1) / world) % workers) %
                                                            workers;
                                                }
                                                uint64_t dst = (uint64_t(rank) * td.token_capacity + src[row]) * xb;
                                                uint64_t source =
                                                    slots[row] >= 0
                                                        ? reinterpret_cast<uint64_t>(expanded.data() + slots[row] * xb)
                                                        : reinterpret_cast<uint64_t>(x.data() + row * xb);
                                                (slots[row] >= 0 ? expected_early : expected_late)
                                                    .emplace(peer, owner, dst, source, xb);
                                                if (aux)
                                                    expected_late.emplace(
                                                        peer, owner,
                                                        td.topk_weights_buffer_offset +
                                                            (uint64_t(rank) * td.token_capacity + src[row]) * wb,
                                                        reinterpret_cast<uint64_t>(side.data() + row * wb), wb);
                                            }
                                        begin = end;
                                    }
                                    for (unsigned core = 0; core < cores; ++core) {
                                        model::reset(cases % 2 ? 1 : 3);
                                        model::active_qp = core;
                                        Resource resource;
                                        const unsigned syncs = AscendC::local_syncs;
                                        if (token_schedule)
                                            CombineTokenRemote<1>(resource, x.data(), side.data(),
                                                                  reinterpret_cast<uint8_t*>(src.data()),
                                                                  reinterpret_cast<uint8_t*>(prefix.data()), nullptr,
                                                                  workspace.data(), td, core);
                                        else
                                            CombineRemote<1>(resource, x.data(), side.data(),
                                                             reinterpret_cast<uint8_t*>(src.data()),
                                                             reinterpret_cast<uint8_t*>(prefix.data()), nullptr,
                                                             workspace.data(), td, core);
                                        CHECK(model::quiesced.empty() && AscendC::local_syncs == syncs);
                                        AscendC::CheckDrained();
                                        const size_t early = model::sent.size();
                                        for (const auto& r : model::sent) {
                                            CHECK(r.order == ACLSHMEMX_UDMA_ODR_RO);
                                            CHECK(actual_early
                                                      .emplace(r.peer, r.qp, r.dst - workspace.data(),
                                                               reinterpret_cast<uint64_t>(r.src), r.bytes)
                                                      .second);
                                        }
                                        // A fresh UB resource models kernel boundaries, but SQ state is
                                        // intentionally retained. Final quiet must cover the early set.
                                        Resource later;
                                        if (token_schedule)
                                            CombineTokenRemote<2>(later, x.data(), side.data(),
                                                                  reinterpret_cast<uint8_t*>(src.data()),
                                                                  reinterpret_cast<uint8_t*>(prefix.data()), nullptr,
                                                                  workspace.data(), td, core);
                                        else
                                            CombineRemote<2>(later, x.data(), side.data(),
                                                             reinterpret_cast<uint8_t*>(src.data()),
                                                             reinterpret_cast<uint8_t*>(prefix.data()), nullptr,
                                                             workspace.data(), td, core);
                                        std::set<unsigned> marked;
                                        std::set<uint64_t> states, drained;
                                        for (size_t i = 0; i < model::sent.size(); ++i) {
                                            const auto& r = model::sent[i];
                                            states.insert(model::state(r));
                                            if (r.order == ACLSHMEMX_UDMA_ODR_SO) {
                                                CHECK(i >= early && marked.insert(r.peer).second);
                                                continue;
                                            }
                                            CHECK(!marked.count(r.peer));
                                            if (i >= early)
                                                CHECK(actual_late
                                                          .emplace(r.peer, r.qp, r.dst - workspace.data(),
                                                                   reinterpret_cast<uint64_t>(r.src), r.bytes)
                                                          .second);
                                        }
                                        for (auto [p, q] : model::quiesced)
                                            drained.insert(model::contexts[p][q].state_addr);
                                        CHECK(states == drained);
                                        AscendC::CheckDrained();
                                    }
                                    CHECK(actual_early == expected_early && actual_late == expected_late);
                                    ++cases;
                                }
                    }
    std::cout << "PASS early_single_cases=" << cases << " (CPU model, no NPU)\n";
}

void SingleSourceWqeCases()
{
    for (uint32_t step : {1U, 2U})
        for (uint32_t index_step : {1U, step}) {
            int32_t indices[] = {5, 6, 7};
            int32_t slots[] = {3, -1, -1};
            uint64_t spans[] = {reinterpret_cast<uint64_t>(indices), uint64_t(4) << 32, 0, 0};
            uint64_t headers[16]{}, bases[] = {0x100000}, wqes[32]{};
            asc_vf_call<BuildCombineWqes>(dim3(32), wqes, spans, headers, bases, 0x200000, 0x300000, 0, 0x8000, 16, 8,
                                          16, 8, 17, 2, 1, 4, step, index_step, 0x400000,
                                          reinterpret_cast<uint64_t>(slots), 4);
            CHECK(wqes[7] == 0x400000 + 3 * 16);      // single: original expert GM
            CHECK(wqes[15] == 0x300000);              // weights remain rank-row indexed
            CHECK(wqes[23] == 0x200000 + step * 16);  // multi: materialized aggregate
            CHECK(wqes[31] == 0x300000 + step * 8);
            CHECK(wqes[5] == 0x100000 + 5 * 16);
            CHECK(wqes[21] == 0x100000 + uint64_t(indices[index_step]) * 16);
            slots[0] = 4;
            bool rejected = false;
            try {
                asc_vf_call<BuildCombineWqes>(dim3(32), wqes, spans, headers, bases, 0x200000, 0x300000, 0, 0x8000, 16,
                                              8, 16, 8, 17, 2, 1, 4, step, index_step, 0x400000,
                                              reinterpret_cast<uint64_t>(slots), 4);
            } catch (const std::exception&) {
                rejected = true;
            }
            CHECK(rejected);
        }
}

void InvalidCombineIndexCases()
{
    for (int32_t invalid : {-1, 17}) {
        Resource resource;
        model::reset(1);
        model::active_qp = 0;
        std::vector<uint8_t> workspace(4096), x(32), weights(16);
        std::vector<int32_t> prefix{0, 0, 1, 0}, indices{invalid};
        CombineTilingData c{};
        c.rank = 0;
        c.world_size = 2;
        c.num_channels = 1;
        c.hidden = 8;
        c.x_element_bytes = 2;
        c.token_capacity = 17;
        c.num_tokens = 17;
        c.x_row_stride = 16;
        c.udma_peers_per_worker = 1;
        c.udma_wqes_per_visit = 8;
        bool rejected = false;
        try {
            CombineRemote(resource, x.data(), weights.data(), reinterpret_cast<uint8_t*>(indices.data()),
                          reinterpret_cast<uint8_t*>(prefix.data()), nullptr, workspace.data(), c, 0);
        } catch (const std::exception&) {
            rejected = true;
        }
        CHECK(rejected && model::sent.empty());
    }
}

template <class T>
void WriteValue(std::vector<uint8_t>& bytes, size_t offset, T value)
{
    CHECK(offset + sizeof(T) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

// Independent scalar golden, with poisoned remote workspaces and duplicate-store
// detection. Executes the actual metadata helper/VFs (including self MTE copies),
// not a second implementation of their scheduling or tiling loops.
void MetadataCase(unsigned world, unsigned rank, unsigned tokens, unsigned group, unsigned cores, unsigned topk,
                  bool weights, bool scales, bool cached, unsigned xbytes = 14, bool empty = false,
                  bool no_experts = false, bool token_schedule = false)
{
    constexpr unsigned experts = 6, sbytes = 12;
    CHECK((token_schedule || Workers(world, group) <= cores) && cores <= 64);
    std::vector<uint8_t> routes(tokens * world), x(tokens * xbytes), side(tokens * sbytes);
    std::vector<topk_idx_t> indices(tokens * topk);
    std::vector<float> input_weights(tokens * topk);
    std::vector<int32_t> rank_prefix(world * world), channel_prefix(world * cores);
    std::vector<unsigned> counts(world), base(world);
    std::vector<int32_t> heads(tokens * world, -777), expected_heads(heads);
    for (size_t i = 0; i < x.size(); ++i) x[i] = uint8_t(i * 11 + 3);
    for (size_t i = 0; i < side.size(); ++i) side[i] = uint8_t(i * 7 + 19);
    const unsigned channel_tokens = (tokens + cores - 1) / cores;
    for (unsigned t = 0; t < tokens; ++t) {
        for (unsigned k = 0; k < topk; ++k) {
            // Duplicate expert slots, multiple experts on a rank, and -1 routes.
            const auto expert = (t + k) % 11 == 0 ? -1 : int((t * 13 + (k / 2) * 7) % (world * experts));
            indices[t * topk + k] = expert;
            input_weights[t * topk + k] = float((t * 3 + k) % 17) * 0.25F;
            if (expert >= 0) routes[t * world + expert / experts] = 1;
        }
        for (unsigned p = 0; p < world; ++p) {
            if (topk == 0) routes[t * world + p] = (t * 13 + p) % 5 != 0;
            if ((t + p) % 17 == 0) routes[t * world + p] = 1;  // A routed row can have no owned expert.
            if (p % 7 == 3) routes[t * world + p] = 0;         // Empty peer, including otherwise valid top-k.
            if (empty) routes[t * world + p] = 0;
        }
    }
    for (unsigned p = 0; p < world; ++p) {
        base[p] = rank ? 7 + p % 3 : 0;
        if (rank) rank_prefix[(rank - 1) * world + p] = base[p];
        for (unsigned ch = 0; ch < cores; ++ch) {
            unsigned ordinal = 0;
            for (unsigned t = ch * channel_tokens; t < std::min(tokens, (ch + 1) * channel_tokens); ++t) {
                expected_heads[t * world + p] = routes[t * world + p] ? int(ordinal) : -1;
                if (routes[t * world + p]) {
                    ++ordinal;
                    ++counts[p];
                }
            }
            channel_prefix[p * cores + ch] = counts[p];
        }
    }
    DispatchTilingData td{};
    td.rank = rank;
    td.world_size = world;
    td.num_tokens = tokens;
    td.num_channels = cores;
    td.rank_stride = (world + 7) / 8 * 8;
    td.num_topk = topk;
    td.num_experts = no_experts ? 0 : world * experts;
    td.hidden = xbytes;
    td.x_element_bytes = 1;
    td.num_scales = 3;
    td.scale_element_bytes = 4;
    td.has_x_scales = scales;
    td.has_topk_weights = weights;
    td.cached_mode = cached;
    td.udma_token_schedule = token_schedule;
    td.udma_peers_per_worker = group;
    td.udma_wqes_per_visit = 8;
    auto list = Plan(routes, td, heads);
    auto align = [](uint64_t n) { return (n + 31) / 32 * 32; };
    const uint64_t capacity = tokens + 16;
    td.recv_channel_prefix_offset = 0;
    td.src_idx_buffer_offset = align(uint64_t(world) * cores * 4);
    td.x_buffer_offset = align(td.src_idx_buffer_offset + capacity * 4);
    td.x_scales_buffer_offset = align(td.x_buffer_offset + capacity * xbytes);
    td.topk_idx_buffer_offset = align(td.x_scales_buffer_offset + capacity * sbytes);
    td.topk_weights_buffer_offset = align(td.topk_idx_buffer_offset + capacity * topk * sizeof(topk_idx_t));
    td.udma_completion_signal_offset = align(td.topk_weights_buffer_offset + capacity * topk * 4);
    const size_t bytes = td.udma_completion_signal_offset + world * kSignalBytes;
    std::vector<std::vector<uint8_t>> actual(world, std::vector<uint8_t>(bytes, 0xa7));
    std::vector<std::vector<uint8_t>> expected = actual;
    // Initialize exactly the valid receive rows; padding and optional weights
    // stay poisoned. Repeat scatter below after reinitialization to catch stale slots.
    for (unsigned round = 0; round < 2; ++round) {
        if (round) {
            for (auto& i : indices) i = -1;
            std::fill(input_weights.begin(), input_weights.end(), -3.0F);
        }
        for (unsigned p = 0; p < world; ++p) {
            for (unsigned core = 0; core < cores; ++core) {
                Resource resource;
                InitializeDispatchTopk(resource, actual[p].data(), td, base[p] + counts[p], core, cores);
                AscendC::CheckDrained();
            }
            for (unsigned row = 0; row < base[p] + counts[p]; ++row)
                for (unsigned k = 0; k < topk; ++k) {
                    WriteValue(expected[p], td.topk_idx_buffer_offset + (uint64_t(row) * topk + k) * sizeof(topk_idx_t),
                               topk_idx_t(-1));
                    if (weights)
                        WriteValue(expected[p], td.topk_weights_buffer_offset + (uint64_t(row) * topk + k) * 4, 0.0F);
                }
        }
        size_t expected_stores = 0;
        for (unsigned p = 0; p < world; ++p) {
            // Prefix exchange belongs to Notify; the old SHMEM prefix slot
            // must remain poisoned throughout DispatchMetadata.
            unsigned row = base[p];
            for (unsigned t = 0; t < tokens; ++t)
                if (routes[t * world + p]) {
                    if (!cached) {
                        WriteValue(expected[p], td.src_idx_buffer_offset + uint64_t(row) * 4, int32_t(t));
                        ++expected_stores;
                    }
                    for (unsigned k = 0; k < topk; ++k) {
                        const auto expert = indices[t * topk + k];
                        if (!no_experts && expert >= topk_idx_t(p * experts) &&
                            expert < topk_idx_t((p + 1) * experts)) {
                            WriteValue(expected[p],
                                       td.topk_idx_buffer_offset + (uint64_t(row) * topk + k) * sizeof(topk_idx_t),
                                       topk_idx_t(expert - p * experts));
                            ++expected_stores;
                            if (weights) {
                                WriteValue(expected[p], td.topk_weights_buffer_offset + (uint64_t(row) * topk + k) * 4,
                                           input_weights[t * topk + k]);
                                ++expected_stores;
                            }
                        }
                    }
                    if (p == rank) {
                        std::memcpy(expected[p].data() + td.x_buffer_offset + uint64_t(row) * xbytes,
                                    x.data() + uint64_t(t) * xbytes, xbytes);
                        if (scales)
                            std::memcpy(expected[p].data() + td.x_scales_buffer_offset + uint64_t(row) * sbytes,
                                        side.data() + uint64_t(t) * sbytes, sbytes);
                    }
                    ++row;
                }
        }
        if (corrupt_metadata_oracle && tokens) expected_heads[0] ^= 1;
        model::local_workspace = actual[rank].data();
        model::workspace_bytes = bytes;
        model::remote_workspaces.clear();
        for (auto& buf : actual) model::remote_workspaces.push_back(buf.data());
        model::metadata_stores.clear();
        model::metadata_store_bytes = 0;
        const unsigned syncs = model::cross_syncs, local_syncs = AscendC::local_syncs;
        const unsigned first = token_schedule ? TokenWorkers(cores) : MetadataFirst(world, group, cores);
        // Reverse core execution also checks that metadata is not relying on an
        // earlier channel/core having produced its ordinal or destination base.
        for (unsigned core = cores; core-- > 0;) {
            Resource resource;
            const auto stores = model::metadata_stores.size();
            DispatchMetadata(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(indices.data()),
                             weights ? reinterpret_cast<uint8_t*>(input_weights.data()) : nullptr,
                             reinterpret_cast<uint8_t*>(list.data()), reinterpret_cast<uint8_t*>(rank_prefix.data()),
                             reinterpret_cast<uint8_t*>(channel_prefix.data()), actual[rank].data(),
                             reinterpret_cast<uint8_t*>(heads.data()), td, core, cores, 192 * 1024);
            AscendC::CheckDrained();
            if (core < first) CHECK(model::metadata_stores.size() == stores);
        }
        CHECK(model::cross_syncs == syncs && AscendC::local_syncs == local_syncs);
        CHECK(model::metadata_stores.size() == expected_stores);
        CHECK(heads == expected_heads);
        CHECK(actual == expected);  // All bytes, including padding and untouched fields.
    }
    model::local_workspace = nullptr;
    model::remote_workspaces.clear();
}

// Execute the production Notify gather against independently constructed route
// counts. Neither the oracle nor the published rows call the gather helper.
void NotifyPrefixCase(unsigned world, unsigned channels, bool fixed_header = false)
{
    const unsigned stride = (channels + 7) / 8 * 8;
    const size_t offset = size_t(world) * world;
    const size_t header = fixed_header ? world * kSignalBytes : 0;
    const size_t bytes = header + (offset + world * stride + 8) * sizeof(int32_t);
    const int32_t poison = -1234567;
    auto token_count = [](unsigned source) { return std::array<unsigned, 5>{0, 1, 17, 129, 257}[source % 5]; };
    auto routed = [](unsigned source, unsigned token, unsigned dest, unsigned round) {
        // Empty, dense/skewed, and sparse routes, changed on the second use.
        return round ? dest == 0 || (token + source * 3 + dest) % 7 == 0
                     : dest % 3 != 0 && (token * 5 + source + dest) % 4 < 2;
    };
    std::vector<std::vector<uint8_t>> published(world, std::vector<uint8_t>(bytes, 0xa7));
    for (unsigned round = 0; round < 2; ++round) {
        for (unsigned source = 0; source < world; ++source)
            for (unsigned dest = 0; dest < world; ++dest) {
                const unsigned tokens = token_count(source), per_channel = (tokens + channels - 1) / channels;
                int32_t sum = 0;
                for (unsigned ch = 0; ch < channels; ++ch) {
                    for (unsigned token = std::min(ch * per_channel, tokens);
                         token < std::min((ch + 1) * per_channel, tokens); ++token)
                        sum += routed(source, token, dest, round);
                    WriteValue(published[source], header + (offset + dest * stride + ch) * sizeof(int32_t), sum);
                }
            }
        const auto original = published;
        model::remote_workspaces.clear();
        for (auto& buf : published) model::remote_workspaces.push_back(buf.data());
        model::workspace_bytes = bytes;
        for (unsigned rank = 0; rank < world; ++rank) {
            model::local_workspace = published[rank].data();
            // Aligned GM matrix with poison before/after it; compact rows may
            // still be sub-32-byte, so ownership must be checked per line.
            const size_t elements = size_t(world) * channels;
            std::vector<int32_t> storage(elements + 24, poison), expected(elements);
            auto* output =
                reinterpret_cast<int32_t*>((reinterpret_cast<uintptr_t>(storage.data() + 8) + 31) & ~uintptr_t(31));
            for (unsigned source = 0; source < world; ++source)
                for (unsigned ch = 0; ch < channels; ++ch) {
                    const unsigned tokens = token_count(source), per_channel = (tokens + channels - 1) / channels;
                    int32_t count = 0;
                    // Exclusive prefix derived directly from original token routes.
                    for (unsigned token = 0; token < std::min(ch * per_channel, tokens); ++token)
                        count += routed(source, token, rank, round);
                    expected[source * channels + ch] = count;
                }
            if (corrupt_notify_oracle) expected.back() ^= 1;
            std::vector<int> line_owner((elements + 7) / 8, -1);
            std::vector<unsigned> visits(world);
            const unsigned syncs = model::cross_syncs, local_syncs = AscendC::local_syncs;
            for (unsigned core = channels; core-- > 0;) {
                Resource resource;
                std::fill(resource.bytes.begin(), resource.bytes.end(), 0xa7);
                AscendC::copy_records.clear();
                AscendC::record_copies = true;
                GatherRecvChannelPrefixes(model::local_workspace + header, reinterpret_cast<uint8_t*>(output), offset,
                                          rank, world, channels, stride, core, channels,
                                          resource.GetBufferByByte<int32_t>(65536),
                                          resource.GetBufferByByte<int32_t>(0));
                AscendC::CheckDrained();
                AscendC::record_copies = false;
                CHECK(AscendC::copy_records.size() % 2 == 0);
                for (size_t pair = 0; pair < AscendC::copy_records.size(); pair += 2) {
                    const auto& load = AscendC::copy_records[pair];
                    const auto& store = AscendC::copy_records[pair + 1];
                    CHECK(!load.to_gm && store.to_gm && load.bytes == channels * 4 && store.bytes == channels * 4);
                    const auto dst = reinterpret_cast<uintptr_t>(store.dst);
                    const auto base = reinterpret_cast<uintptr_t>(output);
                    CHECK(dst >= base && dst + store.bytes <= base + elements * 4 &&
                          (dst - base) % (channels * 4) == 0);
                    const size_t source = (dst - base) / (channels * 4);
                    CHECK(++visits[source] == 1);  // includes identical-value duplicate writes
                    CHECK(load.src == published[source].data() + header + (offset + rank * stride) * 4);
                    for (size_t i = source * channels; i < (source + 1) * channels; ++i) {
                        auto& owner = line_owner[i / 8];
                        CHECK(owner == -1 || owner == int(core));  // no split cache-line writers
                        owner = int(core);
                    }
                }
            }
            for (unsigned visit : visits) CHECK(visit == 1);
            CHECK(std::equal(expected.begin(), expected.end(), output));
            CHECK(std::all_of(storage.data(), output, [&](int32_t v) { return v == poison; }));
            CHECK(std::all_of(output + elements, storage.data() + storage.size(),
                              [&](int32_t v) { return v == poison; }));
            CHECK(model::cross_syncs == syncs && AscendC::local_syncs == local_syncs);
        }
        CHECK(published == original);  // Remote published rows and padding are read-only.
    }
    model::local_workspace = nullptr;
    model::remote_workspaces.clear();
}

template <bool IsDispatch>
unsigned IncomingWaitCases()
{
    constexpr uint64_t second = 1000000000ULL, deadline = 10 * second;
    static_assert(kIncomingTimeoutCycles == deadline);
    std::vector<uint8_t> workspace(5 * kSignalBytes);
    auto set = [&](unsigned source, uint64_t value) {
        std::memcpy(workspace.data() + source * kSignalBytes, &value, sizeof(value));
    };
    auto reset = [&]() {
        std::fill(workspace.begin(), workspace.end(), 0);
        AscendC::system_cycle = 0;
        AscendC::cycle_step = second;
        AscendC::cycle_reads = 0;
        AscendC::invalidations = 0;
        AscendC::diagnostics.clear();
        AscendC::on_invalidate = {};
    };
    unsigned cases = 0;
    auto run = [&](unsigned rank, unsigned world, unsigned core, unsigned cores, bool timeout, unsigned source = 0,
                   unsigned lanes = 1) {
        const unsigned syncs = AscendC::local_syncs;
        bool trapped = false;
        try {
            WaitIncoming<IsDispatch>(workspace.data(), 0, rank, world, core, cores, lanes);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) != "device guard") throw;
            trapped = true;
        }
        CHECK(trapped == timeout);
        CHECK(AscendC::local_syncs == syncs + unsigned(!timeout));
        if (timeout) {
            CHECK(AscendC::diagnostics.size() == 1);
            const auto& line = AscendC::diagnostics.front();
            CHECK(line.find(IsDispatch ? "op=dispatch" : "op=combine") != std::string::npos);
            CHECK(line.find("stage=wait_incoming") != std::string::npos);
            CHECK(line.find("rank=" + std::to_string(rank) + " ") != std::string::npos);
            CHECK(line.find("core=" + std::to_string(core) + " ") != std::string::npos);
            CHECK(line.find("source=" + std::to_string(source) + " ") != std::string::npos);
            CHECK(line.find("elapsed_us=10000000 timeout_us=10000000") != std::string::npos);
        } else
            CHECK(AscendC::diagnostics.empty());
        ++cases;
    };
    // Immediate success, self skip, and empty AIVs still reach the one barrier.
    reset();
    set(0, 1);
    set(1, 1);
    run(2, 3, 0, 1, false);
    CHECK(AscendC::cycle_reads == 1 && AscendC::invalidations == 2);
    reset();
    run(0, 1, 0, 1, false);
    CHECK(AscendC::invalidations == 0);
    reset();
    run(0, 2, 7, 8, false);
    CHECK(AscendC::invalidations == 0);
    // Success just before expiry; every unsuccessful read must invalidate again.
    reset();
    AscendC::on_invalidate = [&](uint8_t*) {
        if (AscendC::system_cycle >= 9 * second) set(0, 1);
    };
    run(1, 2, 0, 1, false);
    CHECK(AscendC::invalidations == 9);
    // Stuck zero or unexpected non-one value must fail exactly at 10 s.
    reset();
    run(1, 2, 0, 1, true, 0);
    CHECK(AscendC::invalidations == 10);
    CHECK(AscendC::diagnostics[0].find("observed=0 ") != std::string::npos);
    reset();
    set(0, 2);
    run(1, 2, 0, 1, true, 0);
    CHECK(AscendC::diagnostics[0].find("observed=2 ") != std::string::npos);
    // First peer takes 7 s; the second gets the remaining budget, not 10 more.
    reset();
    AscendC::on_invalidate = [&](uint8_t* p) {
        if (p == workspace.data() && AscendC::system_cycle >= 7 * second) set(0, 1);
    };
    run(2, 3, 0, 1, true, 1);
    CHECK(AscendC::system_cycle == 11 * second);
    // Lane 0 cannot satisfy completion of lane 1; no source payload is released early.
    reset();
    set(0, 1);
    run(1, 2, 0, 1, true, 0, 2);
    CHECK(AscendC::diagnostics[0].find("worker=1 ") != std::string::npos);
    reset();
    set(0, 1);
    AscendC::on_invalidate = [&](uint8_t* p) {
        if (p == workspace.data() + 8 && AscendC::system_cycle >= 4 * second) *reinterpret_cast<uint64_t*>(p) = 1;
    };
    run(1, 2, 0, 1, false, 0, 2);
    // Reusing the slot clears both lanes; the unused signal region stays intact.
    reset();
    std::fill(workspace.begin(), workspace.end(), 0xa5);
    for (unsigned core = 0; core < 2; ++core) {
        Resource resource;
        Begin(workspace.data(), 0, 1, 2, core, 2, resource.GetBufferByByte<uint64_t>(0), 2);
        AscendC::CheckDrained();
    }
    for (unsigned source = 0; source < 2; ++source) {
        for (unsigned lane = 0; lane < 2; ++lane)
            CHECK(*reinterpret_cast<uint64_t*>(workspace.data() + source * kSignalBytes + lane * 8) ==
                  unsigned(source == 1));
        CHECK(workspace[source * kSignalBytes + 16] == 0xa5);
    }
    // Unsigned elapsed arithmetic must survive a counter wrap.
    reset();
    AscendC::system_cycle = UINT64_MAX - 4 * second;
    run(1, 2, 0, 1, true, 0);
    CHECK(AscendC::invalidations == 10);
    // Strided ownership: this AIV waits for source 1, then 3, never other slots.
    reset();
    set(1, 1);
    run(4, 5, 1, 2, true, 3);
    reset();
    AscendC::cycle_step = 0;
    return cases;
}

void ParallelNotifyCases()
{
    for (unsigned world : {3, 8, 17, 257, 1024}) {
        const unsigned rank_stride = (world + 7) / 8 * 8;
        for (unsigned channels : {2, 16, 64}) {
            if (channels * rank_stride > 16384) continue;
            Resource resource;
            auto values = resource.GetBufferByByte<int32_t>(65536);
            std::vector<int32_t> expected(channels * rank_stride);
            for (unsigned p = 0; p < rank_stride; ++p) {
                int32_t sum = 0;
                for (unsigned ch = 0; ch < channels; ++ch) {
                    const int32_t n = p < world ? (ch * 7 + p * 11) % 129 : 0;
                    values.SetValue(ch * rank_stride + p, n);
                    expected[ch * rank_stride + p] = sum += n;
                }
            }
            asc_vf_call<PrefixChannelCountsVf>(dim3(128), values.GetPhyAddr(), rank_stride, channels);
            for (unsigned i = 0; i < expected.size(); ++i) CHECK(values.GetValue(i) == expected[i]);
        }
        // Partial 8-token chunks, multiple route tiles, non-power-of-two EP,
        // empty channels and carry into later tiles; Plan uses a scalar oracle.
        for (unsigned tokens : {0, 1, 7, 8, 9, 127, 128, 129, 17001}) {
            if (world > 257 && tokens > 129) continue;
            DispatchTilingData td{};
            td.world_size = world;
            td.num_channels = 2;
            td.num_tokens = tokens;
            std::vector<uint8_t> routes(size_t(tokens) * world);
            for (unsigned t = 0; t < tokens; ++t)
                for (unsigned p = 0; p < world; ++p)
                    routes[size_t(t) * world + p] = ((t * 17 + p * 23) % 11 < 6) && p % 7 != 3;
            std::vector<int32_t> head(routes.size(), -777);
            Plan(routes, td, head);
        }
    }
}

void FixedControlHeaderCases()
{
    // Changing shape/weight presence must never move reset writes into a
    // previous call's notify/payload. Exercise every marker through the
    // maximum 64-entry slot and idle AIVs as well as strided source ownership.
    for (unsigned world : {2, 8, 32, 128}) {
        const unsigned header = world * kSignalBytes;
        for (unsigned cores : {20, 32, 64}) {
            for (unsigned markers : {1, 2, 4, 32, 64}) {
                std::vector<uint8_t> workspace(header + 4096, 0xa5);
                for (unsigned core = 0; core < cores; ++core) {
                    Resource resource;
                    Begin(workspace.data(), 0, world - 1, world, core, cores, resource.GetBufferByByte<uint64_t>(0),
                          markers);
                    AscendC::CheckDrained();
                }
                for (unsigned source = 0; source < world; ++source) {
                    for (unsigned lane = 0; lane < markers; ++lane)
                        CHECK(Signal(workspace.data(), 0, source)[lane] == unsigned(source == world - 1));
                    for (unsigned byte = markers * 8; byte < kSignalBytes; ++byte)
                        CHECK(workspace[source * kSignalBytes + byte] == 0xa5);
                }
                for (unsigned byte = header; byte < workspace.size(); ++byte) CHECK(workspace[byte] == 0xa5);
            }
        }
    }
}

int main(int argc, char** argv)
try {
    FixedControlHeaderCases();
    ParallelNotifyCases();
    InvalidCombineIndexCases();
    SingleSourceWqeCases();
    EarlySingleCases();
    const unsigned incoming_wait_cases = IncomingWaitCases<true>() + IncomingWaitCases<false>();
    corrupt_oracle = argc > 1 && std::string(argv[1]) == "--corrupt-oracle";
    corrupt_combine_oracle = argc > 1 && std::string(argv[1]) == "--corrupt-combine-oracle";
    corrupt_metadata_oracle = argc > 1 && std::string(argv[1]) == "--corrupt-metadata-oracle";
    corrupt_notify_oracle = argc > 1 && std::string(argv[1]) == "--corrupt-notify-oracle";
    model::corrupt_wqe = argc > 1 && std::string(argv[1]) == "--corrupt-wqe";
    CHECK(ValidPeers(1) && !ValidVisit(1));
    CHECK(Workers(8, 1) == 7 && MetadataFirst(8, 1, 32) == 7);
    CHECK(Workers(33, 1) == kQpCount && Workers(34, 1) > kQpCount);
    CHECK(Workers(64, 1) > kQpCount);
    // Include reverse-owner channels even though the local writer only uses
    // its own QP. Never initialize QP 32 for an infeasible one-peer schedule.
    CHECK((PeerMask(0, 32, 33) & (uint64_t{1} << 31)) != 0);
    CHECK((PeerMask(0, 33, 34) & (uint64_t{1} << 32)) == 0);
    for (unsigned rank = 0; rank < 8; ++rank)
        for (unsigned peer = 0; peer < 8; ++peer) {
            if (rank == peer) {
                CHECK(PeerMask(rank, peer, 8) == 0);
                continue;
            }
            const unsigned forward = (peer + 8 - rank - 1) % 8;
            const unsigned reverse = (rank + 8 - peer - 1) % 8;
            const uint64_t mask = PeerMask(rank, peer, 8);
            CHECK(Owner(rank, peer, 8, 1) == forward);
            CHECK((mask & 1) && (mask & (uint64_t{1} << forward)) && (mask & (uint64_t{1} << reverse)));
            CHECK((mask >> 7) == 0);
        }
    uint64_t notify_cases = 0;
    for (unsigned world : {2, 4, 8, 17, 64})
        for (unsigned channels = 2; channels <= 64; channels += 2) {
            NotifyPrefixCase(world, channels);
            ++notify_cases;
        }
    NotifyPrefixCase(257, 6);
    ++notify_cases;
    NotifyPrefixCase(257, 32);
    ++notify_cases;
    for (unsigned world : {2, 8, 32, 128}) {
        NotifyPrefixCase(world, 20, true);
        ++notify_cases;
    }
    unsigned pipeline_cases = 0;
    for (unsigned chunk : {32U, 64U}) {
        pipeline_chunk_test = chunk;
        for (unsigned group : {1U, 2U, 4U, 8U})
            for (unsigned visit : {4U, 8U, 16U, 32U})
                for (unsigned rows : {0U, 1U, 31U, 32U, 33U, 63U, 65U, 129U, 513U})
                    for (bool weights : {false, true}) {
                        Case(8, 4, rows, group, visit, weights, 3);
                        ++pipeline_cases;
                    }
    }
    pipeline_chunk_test = 0;
    {
        uint64_t never_ready = 0;
        CombineTilingData td{};
        td.pipeline_ready = reinterpret_cast<uint64_t>(&never_ready);
        td.pipeline_chunk_tokens = 32;
        AscendC::system_cycle = 0;
        AscendC::cycle_step = kIncomingTimeoutCycles;
        bool rejected = false;
        try {
            WaitAggregateReady(td, 0, 0);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        CHECK(rejected && !AscendC::diagnostics.empty());
        CHECK(AscendC::diagnostics.back().find("stage=aggregate_ready") != std::string::npos);
        AscendC::cycle_step = 0;
        AscendC::system_cycle = 0;
    }
    std::cout << "PASS aggregate_pipeline_scheduler_cases=" << pipeline_cases << " timeout_rejection=1\n";
    unsigned topology_cases = 0;
    for (unsigned world : {2U, 8U, 16U, 32U, 64U, 128U})
        for (unsigned size : {1U, 2U, 8U, 16U}) {
            if (world % size) continue;
            for (unsigned peers : {1U, 4U, 8U}) {
                if (Workers(world, peers) > 32) continue;
                for (unsigned dispatch = 0; dispatch <= 2; ++dispatch)
                    for (unsigned combine = 0; combine <= 2; ++combine) {
                        Case(world, world - 1, 17, peers, 4, true, 3, dispatch, combine, size);
                        ++topology_cases;
                    }
            }
        }
    Case(8, 3, 0, 4, 8, false, 1, 2, 1, 8);
    ++topology_cases;
    Case(64, 31, 513, 4, 8, false, 3, 1, 2, 16);
    ++topology_cases;
    pipeline_chunk_test = 32;
    Case(16, 7, 129, 4, 8, true, 1, 2, 2, 8);
    ++topology_cases;
    pipeline_chunk_test = 0;
    std::cout << "PASS topology_production_sender_cases=" << topology_cases << "\n";
    uint64_t bindings = 0, cases = 0;
    for (unsigned world = 2; world <= 257; ++world)
        for (unsigned rank = 0; rank < world; ++rank) {
            for (unsigned group : {1, 2, 4, 8})
                if (Workers(world, group) <= 32) {
                    std::vector<unsigned> visits(world);
                    for (unsigned owner = 0; owner < Workers(world, group); ++owner)
                        for (unsigned i = 0; i < group; ++i) {
                            if (owner * group + i >= world - 1) break;
                            auto peer = Peer(rank, world, owner * group + i);
                            CHECK(peer != rank && Owner(rank, peer, world, group) == owner);
                            CHECK(PeerMask(rank, peer, world) == PeerMask(peer, rank, world));
                            CHECK(PeerMask(rank, peer, world) & (uint64_t{1} << owner));
                            ++visits[peer];
                            ++bindings;
                        }
                    for (unsigned p = 0; p < world; ++p) CHECK(visits[p] == unsigned(p != rank));
                }
        }
    for (unsigned world : {2, 8, 33, 64, 129, 257})
        for (unsigned group : {1, 2, 4, 8})
            if (Workers(world, group) <= 32)
                for (unsigned visit : {4, 8, 16, 32})
                    for (unsigned tokens : {0, 17, 513})
                        for (bool aux : {false, true}) {
                            Case(world, world / 2, tokens, group, visit, aux, cases % 2 ? 1 : 3);
                            ++cases;
                        }
    for (unsigned world : {8U, 16U, 32U, 64U, 128U})
        for (unsigned size : {4U, 8U, 16U}) {
            if (size > world) continue;
            for (unsigned per : {1U, 2U, 4U}) {
                if (per * (world / size) > 64) continue;
                for (unsigned rank : {0U, size - 1, world - 1})
                    for (bool aux : {false, true}) {
                        Case(world, rank, 17, per, 8, aux, 3, 3, 3, size);
                        ++cases;
                    }
            }
        }
    // One core drives two independent QPs. Dense-oracle WQEs must keep
    // the original peer/row order while alternating lanes once per visit.
    for (unsigned world : {8U, 16U, 32U})
        for (unsigned order : {0U, 1U, 2U, 3U})
            for (unsigned quota : {4U, 8U, 32U})
                for (unsigned tokens : {0U, 1U, 17U, 513U})
                    for (bool aux : {false, true}) {
                        Case(world, world - 1, tokens, world == 32 ? 2 : 1, quota, aux, 3, order, order, 8, 1, 2);
                        ++cases;
                    }
    Case(128, 17, 513, 1, 8, true, 1, 3, 3, 8, 1, 2);
    ++cases;
    Case(8, 7, 8193, 1, 8, true, 1, 0, 0, 0, 1, 2);
    ++cases;
    // Two independent QPs per ordered rank pair: empty, skew, tails and auxiliaries.
    for (unsigned order : {0U, 1U, 2U, 3U})
        for (unsigned tokens : {0U, 1U, 17U, 513U})
            for (bool aux : {false, true}) {
                Case(16, 7, tokens, 2, 8, aux, 3, order, order, 8, 2);
                ++cases;
            }
    Case(128, 17, 513, 2, 8, true, 1, 3, 3, 16, 2);
    pipeline_chunk_test = 32;
    Case(16, 0, 513, 2, 8, true, 3, 3, 3, 8, 2);
    pipeline_chunk_test = 0;
    Case(128, 17, 513, 4, 8, true, 1, 3, 3, 8);
    Case(48, 17, 513, 2, 8, true, 3, 3, 3, 3);
    Case(64, 0, 8193, 4, 4, true, 1);
    ++cases;
    Case(8, 7, 8193, 2, 8, true, 3);
    ++cases;
    Case(8, 7, 8193, 1, 8, true, 1);
    ++cases;
    for (unsigned tokens : {255, 256, 257})
        for (bool aux : {false, true}) {
            Case(8, 7, tokens, 4, 8, aux, 1);
            ++cases;
        }
    uint64_t ownership = 0, metadata_cases = 0;
    for (unsigned cores = 2; cores <= 64; cores += 2)
        for (unsigned world = 2; world <= 257; ++world)
            for (unsigned group : {1, 2, 4, 8})
                if (Workers(world, group) <= std::min(cores, 32U)) {
                    const unsigned first = MetadataFirst(world, group, cores);
                    std::vector<unsigned> channels(cores);
                    for (unsigned core = first; core < cores; ++core)
                        for (unsigned ch = core - first; ch < cores; ch += cores - first) ++channels[ch];
                    for (unsigned count : channels) CHECK(count == 1);
                    CHECK(first == 0 || first == Workers(world, group));
                    ++ownership;
                }
    for (unsigned world : {2, 4, 8, 33, 64, 129, 257})
        for (unsigned group : {1, 2, 4, 8})
            if (Workers(world, group) <= 32)
                for (unsigned tokens : {0, 17, 129, 513})
                    for (unsigned variant = 0; variant < 4; ++variant) {
                        MetadataCase(world, world / 2, tokens, group, 32, variant == 0 ? 0 : 8, variant >= 2,
                                     variant % 2, variant == 3);
                        ++metadata_cases;
                    }
    // Full sender occupancy, one spare core, 64 AIVs, route tile overflow,
    // single-slot self copy, and top-k at its supported upper bound.
    MetadataCase(8, 0, 32769, 4, 2, 8, true, true, false);
    ++metadata_cases;
    MetadataCase(8, 7, 129, 4, 2, 128, true, false, false);
    ++metadata_cases;
    MetadataCase(4, 0, 3, 4, 2, 8, true, true, false, 100002);
    ++metadata_cases;
    MetadataCase(64, 63, 129, 4, 64, 8, true, true, false);
    ++metadata_cases;
    MetadataCase(64, 0, 17, 2, 32, 0, false, false, true);
    ++metadata_cases;
    MetadataCase(8, 0, 129, 4, 32, 8, true, true, false, 14, true);
    ++metadata_cases;
    MetadataCase(8, 0, 129, 4, 32, 8, false, false, true, 14, false, true);
    ++metadata_cases;
    MetadataCase(8, 7, 129, 1, 8, 8, true, true, false);
    ++metadata_cases;
    MetadataCase(9, 0, 129, 1, 8, 8, true, true, false);
    ++metadata_cases;
    MetadataCase(8, 0, 32769, 1, 32, 8, true, true, false);
    ++metadata_cases;
    for (unsigned cores : {2, 8, 32, 64})
        for (unsigned tokens : {0, 1, 17, 129, 513})
            for (unsigned variant = 0; variant < 4; ++variant) {
                MetadataCase(8, 4, tokens, 4, cores, variant == 0 ? 0 : 8, variant >= 2, variant % 2, variant == 3, 14,
                             false, false, true);
                ++metadata_cases;
            }
    std::cout << "PASS bindings=" << bindings << " scheduler_cases=" << cases << " metadata_ownership=" << ownership
              << " metadata_cases=" << metadata_cases << " notify_cases=" << notify_cases
              << " decoded_wqes=" << model::decoded_wqes << " incoming_wait_cases=" << incoming_wait_cases
              << " topk_bits=" << EP_NUM_TOPK_IDX_BITS << " (CPU model, no NPU)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
}
