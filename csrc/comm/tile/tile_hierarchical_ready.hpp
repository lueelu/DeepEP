// Copyright (c) 2026, Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once

#include "comm/tile/ready_layout.hpp"

namespace deepep::comm::tile {
// Ops supplies transport-specific submission, completion and cache-aware waits.
// One AIV exclusively owns one peer and all of that peer's QPs.
template <class Ops>
class TileHierarchicalReady {
public:
    using Transport = typename Ops::Transport;
    DEEPEP_COMM_DEVICE TileHierarchicalReady(Ops& ops, ReadyLayout layout, uint32_t rank) : ops(ops), layout(layout), rank(rank) {}

    // Deferred opts into same-QP ordering and immutable ValueOffset(window).
    // Unlike PR #239's QP0-only shortcut, every configured QP has its own marker.
    DEEPEP_COMM_DEVICE void Notify(uint32_t peer, Transport transport, uint32_t window,
                                 RdmaCompletion rdmaCompletion = RdmaCompletion::Wait)
    {
        const uint32_t epoch = window + 1;
        if (transport == Transport::Mte) {
            ops.FinishMte();
            for (uint32_t qp = 0; qp < layout.qps; ++qp) {
                ops.Store(layout.ReadyOffset(rank, qp), epoch);
            }
            return;
        }
        if (transport == Transport::Rdma && rdmaCompletion == RdmaCompletion::Deferred) {
            // Ordered data -> scale -> flag on each QP, with NO per-window quiet.
            // Also notify empty QP slices; the receiver waits for all configured QPs.
            for (uint32_t qp = 0; qp < layout.qps; ++qp) {
                ops.PutFlag(layout.ReadyOffset(rank, qp), layout.ValueOffset(window), peer, transport, qp);
            }
            return;
        }
        // UDMA never relies on ordering. Generic RDMA notification also waits
        // unless the caller explicitly opts into the immutable-epoch contract.
        for (uint32_t qp = 0; qp < layout.qps; ++qp) ops.Quiet(peer, transport, qp);
        for (uint32_t qp = 0; qp < layout.qps; ++qp) {
            ops.PutFlag(layout.ReadyOffset(rank, qp), layout.ValueOffset(window), peer, transport, qp);
        }
        // Complete markers before another epoch overwrites these rolling slots.
        for (uint32_t qp = 0; qp < layout.qps; ++qp) ops.Quiet(peer, transport, qp);
    }

    DEEPEP_COMM_DEVICE void WaitAll(uint32_t window)
    {
        for (uint32_t source = 0; source < layout.world; ++source) {
            for (uint32_t qp = 0; qp < layout.qps; ++qp) {
                ops.Wait(layout.ReadyOffset(source, qp), window + 1);
            }
        }
        ops.Store(layout.FrontierOffset(), window + 1);
    }

    DEEPEP_COMM_DEVICE void WaitReady(uint32_t window) { ops.Wait(layout.FrontierOffset(), window + 1); }

private:
    Ops& ops;
    ReadyLayout layout;
    uint32_t rank;
};
} // namespace deepep::comm::tile
