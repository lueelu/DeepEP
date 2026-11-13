// CPU protocol model. Exercises the public TileHierarchicalReady template with delayed writes.
#include "comm/tile/tile_hierarchical_ready.hpp"
#include "layout.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace deepep::hierarchical;
using Transport = deepep::comm::CopyTransport;
using deepep::comm::RdmaCompletion;
using deepep::comm::tile::TileHierarchicalReady;
struct Message {
    std::function<void()> deliver;
};
struct Model {
    Layout layout;
    uint32_t group;
    std::mt19937 random;
    using Key = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::map<Key, std::deque<Message>> queues;
    std::vector<std::vector<uint32_t>> control;
    // Record completion of every actual byte. Each output is indexed by source.
    std::vector<std::vector<uint8_t>> data, scale;
    uint32_t rdmaQuiets = 0, udmaQuiets = 0, flags = 0;
    Model(Layout l, uint32_t g, uint32_t seed) : layout(l), group(g), random(seed),
        control(l.world, std::vector<uint32_t>(l.Ready().Bytes() / 4)),
        data(l.world, std::vector<uint8_t>(l.DataBytes(), kPadding)),
        scale(l.world, std::vector<uint8_t>(l.ScaleBytes(), kPadding))
    {
        for (auto& values : control)
            for (uint32_t w = 0; w < l.Windows(); ++w) values[l.Ready().ValueOffset(w) / 4] = w + 1;
    }
    Transport Route(uint32_t src, uint32_t dst) const
    {
        return src == dst ? Transport::Mte : src / group == dst / group ? Transport::Udma : Transport::Rdma;
    }
    bool Progress()
    {
        std::vector<Key> active;
        for (auto& pair : queues) if (!pair.second.empty()) active.push_back(pair.first);
        if (active.empty()) return false;
        Deliver(active[random() % active.size()]);
        return true;
    }
    void Deliver(Key key)
    {
        auto& queue = queues[key];
        // Deliberately reverse UDMA writes; RDMA remains FIFO within each QP only.
        const bool reverse = Route(std::get<0>(key), std::get<1>(key)) == Transport::Udma;
        auto message = reverse ? std::move(queue.back()) : std::move(queue.front());
        if (reverse) queue.pop_back(); else queue.pop_front();
        message.deliver();
    }
    void Payload(uint32_t src, uint32_t dst, uint32_t window, bool isScale)
    {
        const uint32_t columns = isScale ? layout.n / 32 : layout.n;
        const uint64_t count = uint64_t(layout.Rows(window)) * columns;
        const uint64_t offset = uint64_t(window) * layout.windowM * columns;
        const uint64_t stride = isScale ? layout.ScaleStride() : layout.DataStride();
        const auto units = count / 32;
        for (uint32_t qp = 0; qp < layout.qps; ++qp) {
            const auto begin = units * qp / layout.qps * 32;
            const auto end = qp + 1 == layout.qps ? count : units * (qp + 1) / layout.qps * 32;
            if (begin == end) continue;
            auto copy = [=] {
                auto& bytes = isScale ? scale[dst] : data[dst];
                for (auto i = begin; i < end; ++i) bytes[src * stride + offset + i] = Pattern(src, dst, offset + i, isScale);
            };
            if (src == dst) copy(); else queues[{src, dst, qp}].push_back({copy});
        }
    }
    void Verify(uint32_t dst, uint32_t windows) const
    {
        for (uint32_t src = 0; src < layout.world; ++src) {
            for (bool isScale : {false, true}) {
                const auto columns = isScale ? layout.n / 32 : layout.n;
                const auto stride = isScale ? layout.ScaleStride() : layout.DataStride();
                const auto& bytes = isScale ? scale[dst] : data[dst];
                const auto rows = std::min(uint64_t(windows) * layout.windowM, uint64_t(layout.m));
                for (uint64_t i = 0; i < rows * columns; ++i)
                    if (bytes[src * stride + i] != Pattern(src, dst, i, isScale))
                        throw std::runtime_error("Consumer observed incomplete data/scale");
                for (uint64_t i = uint64_t(layout.m) * columns; i < stride; ++i)
                    assert(bytes[src * stride + i] == kPadding);
            }
        }
    }
};
struct Ops {
    using Transport = ::Transport;
    Model& model;
    uint32_t rank;
    void FinishMte() {}
    void Store(uint64_t offset, uint32_t value)
    {
        model.control[rank][offset / 4] = value;
        // Immediate consumer: validates at the actual production publication point.
        if (offset == model.layout.Ready().FrontierOffset()) model.Verify(rank, value);
    }
    void Wait(uint64_t offset, uint32_t value)
    {
        while (model.control[rank][offset / 4] < value)
            if (!model.Progress()) throw std::runtime_error("Deadlock: missing marker");
    }
    void Quiet(uint32_t peer, Transport transport, uint32_t qp)
    {
        if (transport == Transport::Rdma) ++model.rdmaQuiets; else ++model.udmaQuiets;
        auto key = Model::Key{rank, peer, qp};
        while (!model.queues[key].empty()) model.Deliver(key);
    }
    void PutFlag(uint64_t dst, uint64_t src, uint32_t peer, Transport, uint32_t qp)
    {
        ++model.flags;
        // Fetch immutable source at DELIVERY time, not at submission time.
        Model* m = &model;
        const uint32_t sourceRank = rank;
        model.queues[{rank, peer, qp}].push_back({[=] { m->control[peer][dst / 4] = m->control[sourceRank][src / 4]; }});
    }
};
void Test(uint32_t world, uint32_t group, uint32_t qps, uint32_t n, uint32_t seed, RdmaCompletion completion = RdmaCompletion::Deferred)
{
    Layout l{7, n, world, 3, qps}; // 3 + 3 + 1; scale can be smaller than QP alignment.
    Model model(l, group, seed);
    for (uint32_t window = 0; window < l.Windows(); ++window) {
        // Senders run ahead, deliberately overwriting rolling marker DESTINATIONS.
        for (uint32_t src = 0; src < world; ++src) {
            Ops ops{model, src};
            TileHierarchicalReady<Ops> ready(ops, l.Ready(), src);
            for (uint32_t dst = 0; dst < world; ++dst) {
                model.Payload(src, dst, window, false);
                model.Payload(src, dst, window, true);
                if (completion == RdmaCompletion::Deferred) {
                    ready.Notify(dst, model.Route(src, dst), window, completion);
                } else {
                    ready.Notify(dst, model.Route(src, dst), window); // Exercise the conservative default.
                }
                if (seed % 2) model.Progress();
            }
        }
    }
    if (completion == RdmaCompletion::Deferred) assert(model.rdmaQuiets == 0);
    for (uint32_t rank = 0; rank < world; ++rank) {
        Ops ops{model, rank};
        TileHierarchicalReady<Ops> ready(ops, l.Ready(), rank);
        for (uint32_t window = 0; window < l.Windows(); ++window) {
            ready.WaitAll(window);
            ready.WaitReady(window);
        }
    }
    uint32_t cross = 0, same = 0;
    for (uint32_t src = 0; src < world; ++src) {
        Ops ops{model, src};
        for (uint32_t dst = 0; dst < world; ++dst) {
            const auto route = model.Route(src, dst);
            cross += route == Transport::Rdma;
            same += route == Transport::Udma;
            // The caller drains RDMA via HierarchicalCopy::QuietRdma in production.
            if (route == Transport::Rdma)
                for (uint32_t qp = 0; qp < qps; ++qp) ops.Quiet(dst, route, qp);
        }
    }
    assert(model.rdmaQuiets == cross * qps * (completion == RdmaCompletion::Deferred ? 1 : 2 * l.Windows() + 1));
    assert(model.udmaQuiets == same * qps * l.Windows() * 2);
    assert(model.flags == (world * world - world) * qps * l.Windows());
    for (auto& pair : model.queues) assert(pair.second.empty());
    for (uint32_t rank = 0; rank < world; ++rank) model.Verify(rank, l.Windows());
}
void NegativeControls()
{
    // Prove the model detects an unordered flag that overtakes a payload.
    Layout l{1, 32, 2, 1, 1};
    Model m(l, 2, 0);
    m.Payload(0, 1, 0, false);
    m.Payload(0, 1, 0, true);
    m.Payload(1, 1, 0, false);
    m.Payload(1, 1, 0, true);
    Ops ops{m, 0};
    ops.PutFlag(l.Ready().ReadyOffset(0, 0), l.Ready().ValueOffset(0), 1, Transport::Udma, 0);
    m.Deliver({0, 1, 0}); // LIFO UDMA delivers flag first.
    assert(m.control[1][l.Ready().ReadyOffset(0, 0) / 4] == 1);
    bool failed = false;
    try { m.Verify(1, 1); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    // One QP's RDMA marker cannot certify a different QP's data.
    Layout multi{1, 128, 2, 1, 2};
    Model r(multi, 1, 0);
    r.Payload(0, 1, 0, false);
    r.Payload(0, 1, 0, true);
    r.Payload(1, 1, 0, false);
    r.Payload(1, 1, 0, true);
    Ops rops{r, 0};
    rops.PutFlag(multi.Ready().ReadyOffset(0, 0), multi.Ready().ValueOffset(0), 1, Transport::Rdma, 0);
    while (!r.queues[{0, 1, 0}].empty()) r.Deliver({0, 1, 0});
    failed = false;
    try { r.Verify(1, 1); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
int main()
{
    for (uint32_t qps : {1, 2, 4})
        for (uint32_t n : {32, 1024})
            for (uint32_t seed = 0; seed < 8; ++seed) {
                Test(1, 1, qps, n, seed);
                Test(4, 4, qps, n, seed);
                Test(4, 2, qps, n, seed);
                Test(32, 16, qps, n, seed);
                Test(4, 2, qps, n, seed, RdmaCompletion::Wait);
            }
    NegativeControls();
    std::cout << "PASS 192 deferred + 48 default-wait scenarios + 2 negative controls\n";
}
