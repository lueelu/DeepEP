// CPU address/ownership oracle; uses the production scheduling headers.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ascend_deepep/kernels/udma_token_dispatch_schedule.h"
#include "ascend_deepep/kernels/udma_token_combine_schedule.h"
#include <set>
#include <tuple>
#include <iostream>

using namespace ascend_deepep;
using namespace ascend_deepep::udma;
using Key = std::tuple<unsigned, uint64_t, uint64_t, unsigned>;
bool corrupt = false;

void Case(unsigned world, unsigned cores, unsigned tokens, bool aux, unsigned aliases, unsigned visit = 8)
{
    const unsigned rank = world / 2, workers = TokenWorkers(cores), channel_tokens = (tokens + cores - 1) / cores;
    const unsigned stride = (channel_tokens + 7) / 8 * 8, xb = 14, sb = 8;
    const uint64_t side_offset = uint64_t(world) * tokens * xb + 4096;
    const uint64_t signal = side_offset + uint64_t(world) * tokens * sb + 4096;
    std::vector<uint8_t> workspace(signal + world * kSignalBytes + 1), x(uint64_t(world) * tokens * xb + 1),
        side(uint64_t(world) * tokens * sb + 1);
    std::vector<int32_t> prefix(world * world), channels(world * cores), plan(uint64_t(world) * cores * stride, -777),
        src;
    std::set<Key> dispatch, combine;
    for (unsigned peer = 0; peer < world; ++peer) {
        unsigned ordinal = 0;
        for (unsigned ch = 0; ch < cores; ++ch) {
            unsigned count = 0;
            for (unsigned t = ch * channel_tokens; t < std::min(tokens, (ch + 1) * channel_tokens); ++t) {
                if (peer % 7 == 3 || (t * 17 + peer * 13 + rank) % 5 == 0) continue;
                plan[(uint64_t(peer) * cores + ch) * stride + count++] = t;
                if (peer != rank) {
                    dispatch.emplace(peer, (ordinal + 7) * xb, t * xb, xb);
                    if (aux) dispatch.emplace(peer, side_offset + (ordinal + 7) * sb, (uint64_t{1} << 63) + t * sb, sb);
                }
                ++ordinal;
            }
            channels[peer * cores + ch] = ordinal;
        }
        prefix[(rank - 1) * world + peer] = 7;
    }
    DispatchTilingData d{};
    d.rank = rank;
    d.world_size = world;
    d.num_channels = cores;
    d.num_tokens = tokens;
    d.hidden = 7;
    d.x_element_bytes = 2;
    d.has_x_scales = aux;
    d.num_scales = 2;
    d.scale_element_bytes = 4;
    d.x_scales_buffer_offset = side_offset;
    d.udma_completion_signal_offset = signal;
    d.udma_wqes_per_visit = visit;
    d.udma_token_schedule = 1;
    for (unsigned p = 0; p < world; ++p)
        for (unsigned w = 0; w < workers; ++w) {
            uint64_t one = 1;
            std::memcpy(workspace.data() + signal + p * kSignalBytes + w * 8, &one, 8);
        }
    auto check = [&](bool is_dispatch, unsigned core, std::set<Key>& seen) {
        std::set<unsigned> marked;
        std::set<uint64_t> written, drained;
        for (auto& r : model::sent) {
            CHECK(core < workers && r.qp == core && r.peer != rank);
            CHECK(TokenPeerMask(rank, r.peer) & (uint64_t{1} << core));
            written.insert(model::state(r));
            if (r.order == ACLSHMEMX_UDMA_ODR_SO) {
                CHECK(marked.insert(r.peer).second);
                CHECK(r.dst == workspace.data() + signal + rank * kSignalBytes + core * 8 && r.src == r.dst &&
                      r.bytes == 8);
                continue;
            }
            CHECK(!marked.count(r.peer));
            const bool is_side = r.bytes == sb;
            const uint64_t source = r.src - (is_side ? side.data() : x.data());
            const uint64_t row = source / (is_side ? sb : xb);
            if (is_dispatch)
                CHECK((row / channel_tokens) % workers == core);
            else {
                const unsigned begin = r.peer ? prefix[(r.peer - 1) * world + rank] : 0;
                CHECK((row - begin) % workers == (core + r.peer * ((workers + world - 1) / world)) % workers);
            }
            Key k{r.peer, uint64_t(r.dst - workspace.data()), source + (is_side ? (uint64_t{1} << 63) : 0), r.bytes};
            CHECK(seen.insert(k).second);
        }
        CHECK(marked.size() == (core < workers ? world - 1 : 0));
        for (auto [p, q] : model::quiesced) drained.insert(model::contexts[p][q].state_addr);
        CHECK(written == drained && model::staged.empty());
        AscendC::CheckDrained();
    };
    std::set<Key> actual;
    for (unsigned core = 0; core < cores; ++core) {
        model::reset(aliases);
        model::active_qp = core;
        Resource resource;
        DispatchTokenRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(plan.data()),
                            reinterpret_cast<uint8_t*>(prefix.data()), reinterpret_cast<uint8_t*>(channels.data()),
                            workspace.data(), d, core, cores);
        check(true, core, actual);
    }
    if (corrupt && !dispatch.empty()) dispatch.erase(dispatch.begin());
    CHECK(actual == dispatch);
    std::fill(prefix.begin(), prefix.end(), 0);
    for (unsigned peer = 0; peer < world; ++peer) {
        for (unsigned t = 0; t < tokens; ++t) {
            if (peer % 7 == 3 || (t * 11 + peer + rank) % 3 == 0) continue;
            const uint64_t row = src.size(), slot = uint64_t(rank) * tokens + t;
            src.push_back(t);
            if (peer != rank) {
                combine.emplace(peer, slot * xb, row * xb, xb);
                if (aux) combine.emplace(peer, side_offset + slot * sb, (uint64_t{1} << 63) + row * sb, sb);
            }
        }
        prefix[peer * world + rank] = src.size();
    }
    CombineTilingData c{};
    c.rank = rank;
    c.world_size = world;
    c.num_channels = cores;
    c.hidden = 7;
    c.x_element_bytes = 2;
    c.num_tokens = tokens;
    c.token_capacity = tokens;
    c.num_topk = 2;
    c.has_topk_weights = aux;
    c.x_row_stride = xb;
    c.topk_weights_row_stride = sb;
    c.topk_weights_buffer_offset = side_offset;
    c.udma_completion_signal_offset = signal;
    c.udma_wqes_per_visit = visit;
    c.udma_token_schedule = 1;
    actual.clear();
    for (unsigned core = 0; core < cores; ++core) {
        model::reset(aliases);
        model::active_qp = core;
        Resource resource;
        CombineTokenRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(src.data()),
                           reinterpret_cast<uint8_t*>(prefix.data()), nullptr, workspace.data(), c, core);
        check(false, core, actual);
    }
    CHECK(actual == combine);
}

void Markers()
{
    constexpr unsigned world = 8, cores = 64, workers = 32, rank = 3;
    std::vector<uint8_t> bytes(world * kSignalBytes, 0xa5);
    for (unsigned core = 0; core < cores; ++core) {
        Resource r;
        Begin(bytes.data(), 0, rank, world, core, cores, r.ubBuf.GetBufferByByte<uint64_t>(0), workers);
    }
    for (unsigned p = 0; p < world; ++p) {
        for (unsigned w = 0; w < workers; ++w) CHECK(Signal(bytes.data(), 0, p)[w] == unsigned(p == rank));
        for (unsigned b = workers * 8; b < kSignalBytes; ++b) CHECK(bytes[p * kSignalBytes + b] == 0xa5);
    }
    // Arrival of one worker must not satisfy another worker's marker.
    for (unsigned p = 0; p < world; ++p)
        for (unsigned w = 0; w < workers; ++w) Signal(bytes.data(), 0, p)[w] = 1;
    Signal(bytes.data(), 0, 0)[1] = 0;
    AscendC::system_cycle = 0;
    AscendC::cycle_step = kIncomingTimeoutCycles;
    bool trapped = false;
    try {
        WaitIncoming<true>(bytes.data(), 0, rank, world, 1, cores, workers);
    } catch (const std::exception&) {
        trapped = true;
    }
    CHECK(trapped && AscendC::diagnostics.back().find("worker=1") != std::string::npos);
    AscendC::cycle_step = 0;
    unsigned polls = 0;
    AscendC::on_invalidate = [&](uint8_t* p) {
        if (++polls == 3) *reinterpret_cast<uint64_t*>(p) = 1;
    };
    WaitIncoming<true>(bytes.data(), 0, rank, world, 1, cores, workers);
    CHECK(polls >= 3);
    AscendC::on_invalidate = {};
}

int main(int argc, char**)
try {
    corrupt = argc > 1;
    unsigned count = 0;
    for (unsigned world : {2, 4, 8, 9, 33, 64})
        for (unsigned cores : {2, 8, 32, 64})
            for (unsigned t : {0, 1, 17, 129, 513})
                for (bool aux : {false, true}) {
                    Case(world, cores, t, aux, count % 2 ? 1 : 3);
                    ++count;
                }
    // Cross compact-index window, channel, span and 32-WQE batch boundaries.
    // The scalar oracle above independently checks every source/destination.
    for (unsigned cores : {32, 64})
        for (unsigned visit : {4, 8, 16, 32})
            for (bool aux : {false, true}) {
                Case(8, cores, 16385, aux, count % 2 ? 1 : 3, visit);
                ++count;
            }
    Markers();
    std::cout << "PASS token_schedule_cases=" << count << " decoded_wqes=" << model::decoded_wqes << " CPU-only\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
}
