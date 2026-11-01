// CPU-only: c++ -std=c++17 -O2 -Iinclude test/test_udma_peer_order.cpp -o peer_order
#include "ascend_deepep/udma_schedule.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ascend_deepep::udma;
#define CHECK(x)                                \
    do {                                        \
        if (!(x)) throw std::runtime_error(#x); \
    } while (0)

// Independent nested traversal, without the production compact-index formula.
std::vector<unsigned> Reference(unsigned rank, unsigned world, unsigned size)
{
    std::vector<unsigned> result;
    for (unsigned local = 0; local < size; ++local)
        for (unsigned step = 1; step <= world / size; ++step) {
            unsigned peer = ((rank / size + step) % (world / size)) * size + (rank % size + local) % size;
            if (peer != rank) result.push_back(peer);
        }
    return result;
}

int main(int argc, char**)
try {
    uint64_t bindings = 0, masks = 0;
    for (unsigned world = 2; world <= 257; ++world)
        for (unsigned size = 1; size <= world; ++size) {
            if (world % size) continue;
            std::vector<std::vector<unsigned>> inverse(world, std::vector<unsigned>(world));
            for (unsigned source = 0; source < world; ++source) {
                const auto ref = Reference(source, world, size);
                for (unsigned i = 0; i < ref.size(); ++i) inverse[source][ref[i]] = i;
            }
            for (unsigned rank = 0; rank < world; ++rank) {
                const auto reference = Reference(rank, world, size);
                CHECK(reference.size() == world - 1);
                for (unsigned p = 1; p <= 8; p *= 2) {
                    if (Workers(world, p) > 32) continue;
                    for (unsigned order = 0; order <= 2; ++order) {
                        std::vector<unsigned> seen(world);
                        for (unsigned owner = 0; owner < Workers(world, p); ++owner) {
                            unsigned first = owner * p, count = std::min(p, world - 1 - first);
                            for (unsigned i = 0; i < count; ++i) {
                                const unsigned rotated = order == 2 ? (i + rank % size + owner) % count : i;
                                unsigned expected = order ? reference[first + rotated] : (rank + 1 + first + i) % world;
                                if (argc > 1 && !bindings) expected = rank;  // Deliberate bad oracle must fail.
                                const unsigned actual = WorkerPeer(rank, world, owner, p, i, order, size);
                                CHECK(actual == expected && actual != rank);
                                CHECK(Owner(rank, actual, world, p, order, size) == owner);
                                CHECK(!seen[actual]++);
                                ++bindings;
                            }
                        }
                        for (unsigned peer = 0; peer < world; ++peer) CHECK(seen[peer] == unsigned(peer != rank));
                    }
                }
                for (unsigned peer = 0; peer < world; ++peer)
                    for (unsigned dispatch = 0; dispatch <= 2; ++dispatch)
                        for (unsigned combine = 0; combine <= 2; ++combine) {
                            const uint64_t mask = PeerMask(rank, peer, world, dispatch, combine, size);
                            CHECK(mask == PeerMask(peer, rank, world, dispatch, combine, size));
                            uint64_t expected = peer == rank ? 0 : 1;
                            if (peer != rank)
                                for (unsigned p = 1; p <= 8; p *= 2) {
                                    if (Workers(world, p) > 32) continue;
                                    for (unsigned source : {rank, peer}) {
                                        const unsigned dest = source == rank ? peer : rank;
                                        const unsigned group_owner = inverse[source][dest] / p;
                                        const unsigned rank_owner = ((dest + world - source - 1) % world) / p;
                                        expected |= uint64_t{1} << (dispatch ? group_owner : rank_owner);
                                        expected |= uint64_t{1} << (combine ? group_owner : rank_owner);
                                    }
                                }
                            CHECK(mask == expected);
                            if (!dispatch && !combine) CHECK(mask == PeerMask(rank, peer, world));
                            ++masks;
                        }
            }
        }
    std::cout << "PASS peer-order bindings=" << bindings << " symmetric masks=" << masks << " (CPU only)\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
