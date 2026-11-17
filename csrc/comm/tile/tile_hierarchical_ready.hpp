// Copyright (c) 2026, Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once

#include "comm/common.hpp"

namespace deepep::comm::tile {

// One cache line per incoming marker and for the local frontier. The host
// writes the immutable epoch values after the frontier as a packed array.
DEEPEP_COMM_HOST_DEVICE uint64_t GetReadyFrontierOffset(uint32_t rankSize, uint32_t qpNum)
{
    return uint64_t(rankSize) * qpNum * kSlotBytes;
}

DEEPEP_COMM_HOST_DEVICE uint64_t GetReadyValueOffset(uint32_t rankSize, uint32_t qpNum, uint32_t commIdx)
{
    return GetReadyFrontierOffset(rankSize, qpNum) + kSlotBytes + uint64_t(commIdx) * sizeof(uint32_t);
}

DEEPEP_COMM_HOST_DEVICE uint64_t GetReadyWorkspaceSize(uint32_t rankSize, uint32_t qpNum, uint32_t commLoops)
{
    return Align(GetReadyValueOffset(rankSize, qpNum, commLoops));
}

enum class ReadyMode { PerWindow, PerWindowOrderedRdma, FinalOnly };

// Ops supplies transport operations and cache-aware waits.
// One AIV exclusively owns one peer and all of that peer's QPs.
template <class Ops, ReadyMode Mode = ReadyMode::PerWindow>
class TileHierarchicalReady {
public:
    using Transport = typename Ops::Transport;
    static constexpr RdmaCompletion RDMA_COMPLETION =
        Mode == ReadyMode::PerWindow ? RdmaCompletion::Wait : RdmaCompletion::Deferred;

    DEEPEP_COMM_DEVICE
    TileHierarchicalReady(Ops& ops, uint32_t rankIdx, uint32_t rankSize, uint32_t qpNum)
        : ops(ops), rankIdx(rankIdx), rankSize(rankSize), qpNum(qpNum)
    {}

    // Post after all payloads for this communication round. Ordered RDMA uses an immutable
    // epoch and one marker per QP, including QPs with empty payload slices.
    // Callers must select it only on a backend with ordered remote visibility.
    DEEPEP_COMM_DEVICE void NotifyWindow(uint32_t peer, Transport transport, uint32_t commIdx)
    {
        static_assert(Mode != ReadyMode::FinalOnly, "Use FinishPeer to publish final readiness");
        if constexpr (Mode == ReadyMode::PerWindowOrderedRdma) {
            if (transport == Transport::Rdma) {
                PutFlags(peer, transport, commIdx);
                return;
            }
        }
        PublishCompleted(peer, transport, commIdx);
    }

    // The QP owner calls this after its last communication round. FinalOnly requires stable
    // TX storage and no intermediate remote consumer, and publishes on QP0.
    DEEPEP_COMM_DEVICE void FinishPeer(uint32_t peer, Transport transport, uint32_t lastCommIdx)
    {
        if constexpr (Mode == ReadyMode::FinalOnly) {
            PublishCompleted(peer, transport, lastCommIdx);
        } else if constexpr (Mode == ReadyMode::PerWindowOrderedRdma) {
            if (transport == Transport::Rdma) {
                QuietPayload(peer, transport);
            }
        }
    }

    DEEPEP_COMM_DEVICE void WaitAll(uint32_t commIdx)
    {
        for (uint32_t srcRankIdx = 0; srcRankIdx < rankSize; ++srcRankIdx) {
            for (uint32_t qpIdx = 0; qpIdx < MarkerCount(); ++qpIdx) {
                ops.Wait(GetReadyOffset(srcRankIdx, qpIdx), commIdx + 1);
            }
        }
        ops.Store(GetReadyFrontierOffset(rankSize, qpNum), commIdx + 1);
    }

    DEEPEP_COMM_DEVICE void WaitReady(uint32_t commIdx) { ops.Wait(GetReadyFrontierOffset(rankSize, qpNum), commIdx + 1); }

private:
    DEEPEP_COMM_DEVICE uint64_t GetReadyOffset(uint32_t srcRankIdx, uint32_t qpIdx) const
    {
        return (uint64_t(srcRankIdx) * qpNum + qpIdx) * kSlotBytes;
    }

    DEEPEP_COMM_DEVICE uint32_t MarkerCount() const
    {
        return Mode == ReadyMode::FinalOnly ? 1 : qpNum;
    }

    DEEPEP_COMM_DEVICE void QuietPayload(uint32_t peer, Transport transport)
    {
        for (uint32_t qpIdx = 0; qpIdx < qpNum; ++qpIdx) {
            ops.Quiet(peer, transport, qpIdx);
        }
    }

    DEEPEP_COMM_DEVICE void PutFlags(uint32_t peer, Transport transport, uint32_t commIdx)
    {
        for (uint32_t qpIdx = 0; qpIdx < MarkerCount(); ++qpIdx) {
            ops.PutFlag(
                GetReadyOffset(rankIdx, qpIdx), GetReadyValueOffset(rankSize, qpNum, commIdx), peer, transport, qpIdx);
        }
    }

    DEEPEP_COMM_DEVICE void PublishCompleted(uint32_t peer, Transport transport, uint32_t commIdx)
    {
        if (transport == Transport::Mte) {
            ops.FinishMte();
            for (uint32_t qpIdx = 0; qpIdx < MarkerCount(); ++qpIdx) {
                ops.Store(GetReadyOffset(rankIdx, qpIdx), commIdx + 1);
            }
            return;
        }
        QuietPayload(peer, transport);
        PutFlags(peer, transport, commIdx);
        // Complete markers before a later epoch can overwrite these rolling slots.
        for (uint32_t qpIdx = 0; qpIdx < MarkerCount(); ++qpIdx) {
            ops.Quiet(peer, transport, qpIdx);
        }
    }

    Ops& ops;
    uint32_t rankIdx;
    uint32_t rankSize;
    uint32_t qpNum;
};
} // namespace deepep::comm::tile
