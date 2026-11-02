#include "common/udma_schedule.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
using namespace ascend_deepep::udma;
int main()
{
    unsigned long long checked = 0;
    for (unsigned world : {2U, 3U, 8U, 12U, 16U, 32U, 48U, 64U, 96U, 128U, 256U})
        for (unsigned g = 1; g <= world; ++g) {
            if (world % g) continue;
            for (unsigned per : {1U, 2U, 4U}) {
                const unsigned workers = (g + per - 1) / per;
                if (workers > 32 || ((g + workers - 1) / workers) * (world / g) > 64) continue;
                for (unsigned rank = 0; rank < world; ++rank) {
                    std::vector<unsigned> seen(world);
                    for (unsigned w = 0; w < workers + 1; ++w) {
                        std::vector<unsigned> expected;
                        if (w < workers)
                            for (unsigned step = 1; step <= world / g; ++step)
                                for (unsigned local = w; local < g; local += workers) {
                                    unsigned peer = ((rank / g + step) % (world / g)) * g + local;
                                    if (peer != rank) expected.push_back(peer);
                                }
                        assert(WorkerCount(rank, world, w, per, 3, g) == expected.size());
                        for (unsigned i = 0; i < expected.size(); ++i) {
                            const unsigned peer = WorkerPeer(rank, world, w, per, i, 3, g);
                            assert(peer == expected[i] && Owner(rank, peer, world, per, 3, g) == w);
                            ++seen[peer];
                            ++checked;
                            for (unsigned lanes : {1U, 2U}) {
                                if (workers * lanes > 32) continue;
                                for (unsigned lane = 0; lane < lanes; ++lane) {
                                    auto mask = PeerMask(rank, peer, world, 3, 3, g, per, lanes);
                                    assert(mask == PeerMask(peer, rank, world, 3, 3, g, per, lanes));
                                    assert(mask & (uint64_t{1} << (w * lanes + lane)));
                                    assert((mask >> 32) == 0);
                                }
                            }
                            for (unsigned other : {0U, 1U, 2U, 3U}) {
                                auto a = PeerMask(rank, peer, world, 3, other, g, per);
                                auto b = PeerMask(peer, rank, world, 3, other, g, per);
                                assert(a == b && (a & (uint64_t{1} << w)));
                            }
                        }
                    }
                    for (unsigned peer = 0; peer < world; ++peer) assert(seen[peer] == unsigned(peer != rank));
                }
            }
        }
    // Both directions and both operations must have every active lane bound.
    for (unsigned world : {8U, 16U, 32U, 64U, 128U})
        for (unsigned a : {0U, 1U, 2U, 3U})
            for (unsigned b : {0U, 1U, 2U, 3U})
                for (unsigned rank = 0; rank < world; ++rank)
                    for (unsigned peer = 0; peer < world; ++peer) {
                        const auto mask = PeerMask(rank, peer, world, a, b, 8, 2, 2);
                        assert(mask == PeerMask(peer, rank, world, a, b, 8, 2, 2));
                        if (rank == peer) {
                            assert(mask == 0);
                            continue;
                        }
                        assert((mask >> 32) == 0);
                        for (unsigned order : {a, b})
                            for (unsigned per : {1U, 2U, 4U, 8U}) {
                                if (order == 3 && per != 2) continue;
                                if (Workers(world, per, order, 8) * 2 > 32) continue;
                                const unsigned owner = Owner(rank, peer, world, per, order, 8);
                                assert((mask & (uint64_t{3} << (owner * 2))) == (uint64_t{3} << (owner * 2)));
                            }
                    }
    std::cout << "PASS group_round_robin mappings=" << checked << " (CPU only)\n";
}
