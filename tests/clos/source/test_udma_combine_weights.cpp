// CPU arithmetic/ownership test; hardware visibility is covered by NPU tests.
// g++ -std=c++17 -O2 -D__CCE_KT_TEST__ -DASCEND_DEEPEP_COMBINE_SIMT_WEIGHTS=1
//   -Itest/udma_cpu_stubs -Iinclude test/test_udma_combine_weights.cpp -o /tmp/test_combine_weights
#include "kernels/combine/udma_combine_schedule.h"
#include <iostream>
#include <set>
#include <tuple>
using namespace ascend_deepep;
using namespace ascend_deepep::udma;
using RequestKey = std::tuple<unsigned, uintptr_t, uintptr_t, unsigned>;

void Run(unsigned world, unsigned cores, unsigned peers, unsigned lanes, unsigned qps, unsigned order, unsigned tokens,
         unsigned topk, bool weights, bool corrupt)
{
    CombineTilingData td{};
    td.rank = world / 2;
    td.world_size = world;
    td.num_channels = cores;
    td.hidden = 8;
    td.x_element_bytes = 2;
    td.token_capacity = tokens + 1;
    td.num_topk = topk;
    td.has_topk_weights = weights;
    td.udma_peers_per_worker = peers;
    td.udma_wqes_per_visit = 8;
    td.udma_workers_per_peer = lanes;
    td.udma_qps_per_worker = qps;
    td.udma_peer_order = order;
    td.udma_ranks_per_group = order ? 4 : 0;
    td.x_row_stride = 16;
    td.topk_weights_row_stride = (topk * 4 + 31) / 32 * 32;
    td.udma_completion_signal_offset = 0;
    td.x_buffer_offset = world * kSignalBytes + 512;  // fixed header + notify
    td.topk_weights_buffer_offset = td.x_buffer_offset + uint64_t(world) * td.token_capacity * 16;
    const size_t bytes =
        td.topk_weights_buffer_offset + uint64_t(world) * td.token_capacity * td.topk_weights_row_stride;
    std::vector<uint8_t> workspace(bytes, 0), x;
    std::vector<std::vector<uint8_t>> remote(world, std::vector<uint8_t>(bytes, 0xCD)), expected = remote;
    std::vector<int32_t> prefix(world * world), indices;
    std::vector<unsigned> row_peer;
    for (unsigned p = 0; p < world; ++p) {
        for (unsigned token = 0; token < tokens; ++token) {
            if (p % 5 == 3 || (token + p) % 3 == 0) continue;
            indices.push_back(token);
            row_peer.push_back(p);
        }
        prefix[p * world + td.rank] = indices.size();
        for (unsigned lane = 0; lane < lanes * qps; ++lane) {
            const uint64_t one = 1;
            std::memcpy(workspace.data() + td.udma_completion_signal_offset + p * kSignalBytes + lane * 8, &one, 8);
        }
    }
    x.resize(indices.size() * 16 + 1);
    std::vector<float> w(indices.size() * topk + 1);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 97) - 48) / 8;
    const bool active = weights && lanes * Workers(world, peers, order, td.udma_ranks_per_group) < cores;
    CHECK(CombineSimtWeightsEnabled(td) == active);
    std::multiset<RequestKey> requests, wanted;
    std::set<std::pair<unsigned, size_t>> stores;
    for (size_t row = 0; row < indices.size(); ++row) {
        unsigned p = row_peer[row];
        if (p == td.rank) continue;
        const uint64_t slot = uint64_t(td.rank) * td.token_capacity + indices[row];
        wanted.emplace(p, uintptr_t(remote[p].data() + td.x_buffer_offset + slot * 16), uintptr_t(x.data() + row * 16),
                       16);
        if (!weights) continue;
        const size_t offset = td.topk_weights_buffer_offset + slot * td.topk_weights_row_stride;
        if (active) {
            for (unsigned k = 0; k < topk; ++k) {
                float value = w[row * topk + k];
                if (corrupt && stores.empty()) value += 1;
                std::memcpy(expected[p].data() + offset + k * 4, &value, 4);
                CHECK(stores.emplace(p, offset + k * 4).second);
            }
        } else
            wanted.emplace(p, uintptr_t(remote[p].data() + offset), uintptr_t(w.data() + row * topk), topk * 4);
    }
    model::local_workspace = workspace.data();
    model::workspace_bytes = bytes;
    model::remote_workspaces.clear();
    for (auto& r : remote) model::remote_workspaces.push_back(r.data());
    model::metadata_stores.clear();
    model::metadata_store_bytes = 0;
    unsigned cross = model::cross_syncs, local = AscendC::local_syncs;
    for (unsigned core = 0; core < cores; ++core) {
        Resource resource;
        model::reset(3);
        model::active_qp = core * qps;
        model::active_qp_count = qps;
        CombineRemote(resource, x.data(), reinterpret_cast<uint8_t*>(w.data()),
                      reinterpret_cast<uint8_t*>(indices.data()), reinterpret_cast<uint8_t*>(prefix.data()), nullptr,
                      workspace.data(), td, core);
        AscendC::CheckDrained();
        for (const auto& r : model::sent)
            if (r.order != ACLSHMEMX_UDMA_ODR_SO) requests.emplace(r.peer, uintptr_t(r.dst), uintptr_t(r.src), r.bytes);
    }
    CHECK(requests == wanted);
    CHECK(model::metadata_stores == stores && model::metadata_store_bytes == stores.size() * 4);
    CHECK(remote == expected);  // Includes unwritten slots, self, and padding.
    CHECK(model::cross_syncs - cross == (active ? cores : 0));
    // WaitIncoming always performs one local sync; the final drain adds
    // another local sync only on the original UDMA-weight path.
    CHECK(AscendC::local_syncs - local == (active ? cores : 2U * cores));
    model::local_workspace = nullptr;
    model::remote_workspaces.clear();
}

int main(int argc, char**)
{
    try {
        unsigned cases = 0;
        for (unsigned world : {2U, 4U, 8U, 16U, 64U})
            for (unsigned cores : {2U, 20U, 32U})
                for (unsigned peers : {1U, 4U, 8U})
                    for (unsigned lanes : {1U, 2U})
                        for (unsigned qps : {1U, 2U})
                            for (unsigned order : {0U, 3U}) {
                                if (order && world < 4) continue;
                                unsigned senders = lanes * Workers(world, peers, order, order ? 4 : 0);
                                if (senders > cores || senders * qps > 32) continue;
                                for (unsigned tokens : {0U, 17U, 273U})
                                    for (unsigned topk : {1U, 8U, 17U}) {
                                        if (world == 64 && tokens == 273) continue;
                                        Run(world, cores, peers, lanes, qps, order, tokens, topk, true, argc > 1);
                                        ++cases;
                                    }
                                Run(world, cores, peers, lanes, qps, order, 17, 8, false, false);
                                ++cases;
                            }
        std::cout << "PASS combine SIMT weights cases=" << cases << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
