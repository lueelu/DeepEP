// CPU model: same build flags as test_udma_schedule.cpp. This checks the real
// publication, sender and self-copy code, not hardware memory-coherence behavior.
#include "ascend_deepep/kernels/udma_dispatch_schedule.h"
#include "ascend_deepep/kernels/udma_token_dispatch_schedule.h"
#include "ascend_deepep/kernels/udma_dispatch_metadata.h"
#include <iostream>
#include <set>
#include <tuple>

using namespace ascend_deepep;
using namespace ascend_deepep::udma;
using Bytes = std::vector<uint8_t>;
bool corrupt_oracle = false;

template <class F>
void Reject(F f)
{
    bool rejected = false;
    try {
        f();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected);
    // A simulated trap abandons the stream; the next case is an independent run.
    AscendC::pending_events.clear();
}

void Run(unsigned world, unsigned cores, unsigned tokens, bool scales, bool token_schedule, unsigned group,
         unsigned order = 0, unsigned ranks_per_group = 0, unsigned lanes = 1, unsigned qps = 1)
{
    constexpr unsigned hidden = 14;
    const unsigned xb = scales ? hidden : hidden * 2, sb = scales ? 8 : 0;
    const unsigned length = (tokens + cores - 1) / cores, stride = (length + 7) / 8 * 8;
    const unsigned rank = world - 1;
    DispatchTilingData d{};
    d.rank = rank;
    d.world_size = world;
    d.num_tokens = tokens;
    d.num_channels = cores;
    d.hidden = hidden;
    d.x_element_bytes = scales ? 1 : 2;
    d.has_x_scales = scales;
    d.num_scales = scales ? 2 : 0;
    d.scale_element_bytes = 4;
    d.udma_peers_per_worker = group;
    d.udma_peer_order = order;
    d.udma_ranks_per_group = ranks_per_group;
    d.udma_wqes_per_visit = 8;
    d.udma_token_schedule = token_schedule;
    d.udma_workers_per_peer = lanes;
    d.udma_qps_per_worker = qps;
    d.cached_mode = 1;
    d.udma_completion_signal_offset = 0;
    // Independent dense routing oracle, including one empty destination.
    std::vector<int32_t> channels(world * cores), prefix(world * world), head(tokens * world, -1);
    std::vector<int32_t> plan(size_t(world) * cores * stride, -1);
    std::vector<std::vector<unsigned>> selected(world);
    for (unsigned peer = 0; peer < world; ++peer) {
        for (unsigned ch = 0; ch < cores; ++ch) {
            unsigned count = 0;
            for (unsigned t = ch * length; t < std::min(tokens, (ch + 1) * length); ++t)
                if (peer != 0 && (t * 13 + peer) % 5) {
                    selected[peer].push_back(t);
                    plan[(size_t(peer) * cores + ch) * stride + count++] = t;
                }
            channels[peer * cores + ch] = selected[peer].size();
        }
    }
    Bytes x(std::max(1U, tokens * xb)), side(std::max(1U, tokens * sb));
    for (unsigned i = 0; i < x.size(); ++i) x[i] = i * 7 % 251;
    for (unsigned i = 0; i < side.size(); ++i) side[i] = i * 11 % 241;
    std::vector<Bytes> control(world, Bytes(OutputControlBytes(world)));
    Bytes workspace(world * kSignalBytes, 0);
    for (unsigned p = 0; p < world; ++p)
        for (unsigned w = 0; w < (token_schedule ? TokenWorkers(cores) : lanes * qps); ++w)
            *reinterpret_cast<uint64_t*>(workspace.data() + p * kSignalBytes + w * 8) = 1;
    std::vector<Bytes> old_x, old_scales;
    for (unsigned round = 1; round <= 3; ++round) {
        // Keep all previous output allocations alive; the new pointers must not
        // overwrite retained outputs, even though the routing plan is unchanged.
        std::vector<Bytes> out(world), out_scales(world), expected(world), expected_scales(world);
        for (unsigned p = 0; p < world; ++p) {
            out[p] = Bytes((selected[p].size() + 1) * xb, 0xa5);
            out_scales[p] = Bytes((selected[p].size() + 1) * std::max(sb, 1U), 0xa5);
            expected[p] = out[p];
            expected_scales[p] = out_scales[p];
            for (unsigned row = 0; row < selected[p].size(); ++row) {
                std::memcpy(expected[p].data() + row * xb, x.data() + selected[p][row] * xb, xb);
                if (sb) std::memcpy(expected_scales[p].data() + row * sb, side.data() + selected[p][row] * sb, sb);
            }
        }
        const auto saved_old_x = old_x, saved_old_scales = old_scales;
        model::remote_workspaces.clear();
        for (auto& c : control) model::remote_workspaces.push_back(c.data());
        model::workspace_bytes = OutputControlBytes(world);
        model::metadata_stores.clear();
        for (unsigned p = 0; p < world; ++p) {
            model::local_workspace = control[p].data();
            // Epoch commit is last. Inspect the body at each actual store of it.
            model::on_metadata_store = [&](unsigned dst, size_t offset) {
                if (offset != OutputRecordOffset(p) + kOutputEpochWord * 8) return;
                const auto* rec = reinterpret_cast<uint64_t*>(control[dst].data() + OutputRecordOffset(p));
                CHECK(rec[0] == reinterpret_cast<uint64_t>(out[p].data()));
                CHECK(rec[1] == (sb ? reinterpret_cast<uint64_t>(out_scales[p].data()) : 0));
                CHECK(rec[2] == selected[p].size() && rec[3] == xb && rec[4] == sb && rec[5] == round);
            };
            Resource resource;
            PrepareDispatchOutput(resource, reinterpret_cast<uint64_t>(control[p].data()), p, world, out[p].data(),
                                  sb ? out_scales[p].data() : nullptr, selected[p].size(), selected[p].size(), xb, sb);
            AscendC::CheckDrained();
        }
        model::on_metadata_store = {};
        CHECK(model::metadata_stores.size() == world * world * 6U);
        model::local_workspace = nullptr;  // SQ decoder uses literal remote output VAs.
        model::direct_output = true;
        d.udma_output_control = reinterpret_cast<uint64_t>(control[rank].data());
        d.num_output_tokens = selected[rank].size();
        std::set<std::pair<unsigned, uint64_t>> seen;
        for (unsigned core = 0; core < cores; ++core) {
            Resource resource;
            model::reset(round % 2 ? 1 : 3);
            model::active_qp = core * qps;
            model::active_qp_count = qps;
            DispatchMetadata(resource, x.data(), side.data(), nullptr, nullptr, reinterpret_cast<uint8_t*>(plan.data()),
                             reinterpret_cast<uint8_t*>(prefix.data()), reinterpret_cast<uint8_t*>(channels.data()),
                             workspace.data(), reinterpret_cast<uint8_t*>(head.data()), d, core, cores, 192 * 1024,
                             out[rank].data(), out_scales[rank].data());
            const auto before = model::cross_syncs;
            if (token_schedule)
                DispatchTokenRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(plan.data()),
                                    reinterpret_cast<uint8_t*>(prefix.data()),
                                    reinterpret_cast<uint8_t*>(channels.data()), workspace.data(), d, core, cores);
            else
                DispatchRemote(resource, x.data(), side.data(), reinterpret_cast<uint8_t*>(plan.data()),
                               reinterpret_cast<uint8_t*>(prefix.data()), reinterpret_cast<uint8_t*>(channels.data()),
                               workspace.data(), d, core, cores);
            CHECK(model::cross_syncs == before + 1);
            for (const auto& req : model::sent) {
                if (req.order == ACLSHMEMX_UDMA_ODR_SO) continue;
                const bool scale = req.bytes == sb;
                auto& target = scale ? out_scales[req.peer] : out[req.peer];
                CHECK(req.dst >= target.data() &&
                      req.dst + req.bytes <= target.data() + selected[req.peer].size() * req.bytes);
                CHECK(seen.emplace(req.peer, reinterpret_cast<uint64_t>(req.dst)).second);
                std::memcpy(req.dst, req.src, req.bytes);
            }
            AscendC::CheckDrained();
        }
        if (corrupt_oracle && tokens) expected[rank][0] ^= 1;
        CHECK(out == expected && out_scales == expected_scales);
        CHECK(old_x == saved_old_x && old_scales == saved_old_scales);
        for (auto& a : out) old_x.push_back(std::move(a));
        for (auto& a : out_scales) old_scales.push_back(std::move(a));
        // Each corruption must fail before submission; never accept a stale
        // pointer just because its previous row count and shape still match.
        const unsigned peer =
            token_schedule ? Peer(rank, world, 0) : WorkerPeer(rank, world, 0, group, 0, order, ranks_per_group);
        auto* rec = reinterpret_cast<uint64_t*>(control[rank].data() + OutputRecordOffset(peer));
        for (unsigned field : {0U, 2U, 3U, 4U, 5U}) {
            const auto old = rec[field];
            rec[field] = field == 0 ? UINT64_MAX : field == 2 ? UINT64_MAX : old + 1;
            // peer 0 has zero rows, so test overflow with a non-empty capacity.
            const auto old_rows = rec[2];
            if (field == 0) rec[2] = 1;
            Reject([&] {
                Resource resource;
                model::reset(1);
                model::active_qp = 0;
                if (qps == 2) {
                    DispatchSenderT<kGroupPeerCapacity, 2> sender(d, 0, resource.GetBufferByByte<uint8_t>(0),
                                                                  workspace.data(), x.data(), side.data());
                } else {
                    DispatchSender sender(d, 0, resource.GetBufferByByte<uint8_t>(0), workspace.data(), x.data(),
                                          side.data());
                }
            });
            rec[2] = old_rows;
            rec[field] = old;
        }
    }
    model::direct_output = false;
}

void CheckWideOffsetsAndBounds()
{
    DispatchTilingData d{};
    d.rank = 0;
    d.world_size = 2;
    d.num_tokens = 3;
    d.hidden = 6144;
    d.x_element_bytes = 2;
    d.has_x_scales = 1;
    d.num_scales = 48;
    d.scale_element_bytes = 4;
    d.udma_peers_per_worker = 1;
    std::vector<uint64_t> control(OutputControlBytes(2) / 8, 0);
    control[0] = 7;
    auto* output = control.data() + OutputRecordOffset(1) / 8;
    constexpr uint64_t x_base = uint64_t{1} << 48, scales_base = uint64_t{1} << 47;
    constexpr unsigned rows = 1048576;
    output[0] = x_base;
    output[1] = scales_base;
    output[2] = rows;
    output[3] = 12288;
    output[4] = 192;
    output[5] = 7;
    d.udma_output_control = reinterpret_cast<uint64_t>(control.data());
    int32_t indices[] = {0, 1, 2};
    Bytes workspace(2048), x(3 * 12288), scales(3 * 192);
    model::local_workspace = nullptr;
    model::direct_output = true;
    model::reset(1);
    model::active_qp = 0;
    {
        Resource resource;
        DispatchSender sender(d, 0, resource.GetBufferByByte<uint8_t>(0), workspace.data(), x.data(), scales.data());
        CHECK(sender.PutRows(0, indices, rows - 2, 0, 4) == 4);
        sender.Flush();
        sender.Quiet();
        CHECK(model::sent.size() == 4);
        for (unsigned i = 0; i < 4; ++i) {
            const uint64_t address =
                (i % 2 ? scales_base : x_base) + uint64_t(rows - 2 + i / 2) * (i % 2 ? 192 : 12288);
            CHECK(reinterpret_cast<uint64_t>(model::sent[i].dst) == address);
        }
        AscendC::CheckDrained();
    }
    for (unsigned row : {rows, UINT32_MAX})
        Reject([&] {
            Resource resource;
            model::reset(1);
            DispatchSender sender(d, 0, resource.GetBufferByByte<uint8_t>(0), workspace.data(), x.data(),
                                  scales.data());
            sender.PutRows(0, indices, row, 0, 4);
            sender.Flush();
        });
    model::direct_output = false;
    // Insufficient destination capacity and missing mandatory payload/scales.
    Reject([&] { PublishDispatchOutput(nullptr, 0, 2, 1, 64, 128, 1, 2, 16, 8); });
    Reject([&] { PublishDispatchOutput(nullptr, 0, 2, 1, 0, 128, 1, 1, 16, 8); });
    Reject([&] { PublishDispatchOutput(nullptr, 0, 2, 1, 64, 0, 1, 1, 16, 8); });
}

int main(int argc, char**)
try {
    corrupt_oracle = argc > 1;
    for (unsigned world : {2U, 8U, 16U, 64U})
        for (unsigned cores : {32U, 64U})
            for (unsigned tokens : {0U, 17U, 129U})
                for (bool scales : {false, true})
                    for (bool token : {false, true}) Run(world, cores, tokens, scales, token, world > 32 ? 4 : 1);
    for (unsigned world : {8U, 32U, 128U})
        for (unsigned cores : {32U, 64U})
            for (bool scales : {false, true})
                for (unsigned per : {1U, 2U, 4U}) Run(world, cores, 129, scales, false, per, 3, world == 8 ? 4 : 8);
    uint64_t counter = UINT64_MAX, epoch = 0;
    Reject([&] { AdvanceDispatchEpoch(&counter, &epoch); });
    CHECK(counter == UINT64_MAX);
    CheckWideOffsetsAndBounds();
    for (unsigned order : {0U, 1U, 2U, 3U})
        for (unsigned cores : {32U, 64U})
            for (bool scales : {false, true}) {
                Run(16, cores, 129, scales, false, 2, order, 8, 2);
            }
    for (unsigned world : {8U, 16U, 32U})
        for (unsigned order : {0U, 1U, 2U, 3U})
            for (unsigned cores : {32U, 64U})
                for (unsigned tokens : {0U, 17U, 129U})
                    for (bool scales : {false, true})
                        Run(world, cores, tokens, scales, false, world == 32 ? 2 : 1, order, 8, 1, 2);
    std::cout << "PASS direct output: 292 cases x 3 invocations, descriptors, retained outputs, self/remote, guards\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
