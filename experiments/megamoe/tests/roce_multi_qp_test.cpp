// Copyright (c) 2026, Lu Lu

// Production transport helpers with independently scheduled NIC/CQ queues.
#include <cassert>
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"

int main()
{
    using namespace MegaMoeImpl;
    using namespace RoceTest;
    constexpr uint32_t sequenceOffset = 3072U;
    constexpr auto phase = AscendTimer::RdmaPhase::Combine;
    for (uint32_t qps : {4U, 8U}) {
        const uint32_t sequence = 0x80000101U + qps;
        local.fill(0U);
        remote.fill(0U);
        queued.clear();
        extraCommitOffsets.clear();
        std::memcpy(local.data() + sequenceOffset, &sequence, 4U);
        for (uint32_t qp = 1; qp <= qps; ++qp) {
            const uint32_t offset = 1024U + qp * 32U;
            extraCommitOffsets.push_back(offset);
            expectedQp = qp;
            // The last lane is deliberately empty but still posts a commit.
            if (qp != qps) {
                std::fill_n(local.data() + 2048U + qp * 32U, 32U, static_cast<uint8_t>(qp));
                RocePutBytes(local.data() + qp * 32U, local.data() + 2048U + qp * 32U, 32U, kPeer, scratch.data(),
                             nullptr, phase, false, qp);
            }
            RocePublishCommit(local.data() + offset, local.data() + sequenceOffset, kPeer, scratch.data(), nullptr,
                              phase, qp);
        }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        assert(queued.size() == 2U * qps - 1U);
        // Expose QP1 first: its commit cannot release any other QP.
        DeliverPosted(1U);
        for (uint32_t qp = 2; qp <= qps; ++qp) {
            auto* flag = reinterpret_cast<int32_t*>(remote.data() + 1024U + qp * 32U);
            expectedQp = qp;
            cycleStep = static_cast<int64_t>(MEGAMOE_ROCE_WAIT_TIMEOUT_CYCLES);
            signalObserved = scalarEventSet = mteCanRead = false;
            bool rejected = false;
            try {
                RoceWaitCommit(flag, sequence, nullptr, phase, kPeer, nullptr, qp);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            assert(rejected && !mteCanRead && lastWaitQp == qp);
        }
        cycleStep = 1;
        // Empty QP can finish before the remaining nonempty queues.
        for (uint32_t qp = qps; qp > 1; --qp) DeliverPosted(qp);
#endif
        for (uint32_t qp = 1; qp <= qps; ++qp) {
            expectedQp = qp;
            signalObserved = scalarEventSet = mteCanRead = false;
            RoceWaitCommit(reinterpret_cast<int32_t*>(remote.data() + 1024U + qp * 32U), sequence, nullptr, phase,
                           kPeer, scratch.data(), qp);
            assert(mteCanRead);
            if (qp != qps)
                for (uint32_t i = 0; i < 32U; ++i) assert(remote[qp * 32U + i] == qp);
        }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        // Visibility does not release local sources; corruption is still caught.
        expectedQp = qps;
        local[sequenceOffset] ^= 1U;
        bool rejected = false;
        try {
            RoceFinishOutstanding(kPeer, scratch.data(), nullptr, phase, qps);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
        local[sequenceOffset] ^= 1U;
        // Ready-only CQ progress retires exactly this owner's entries.
        expectedQp = 2U;
        const auto pending = queued.size();
        assert(RoceTryProgress(kPeer, scratch.data(), 2U) == 2U);
        assert(queued.size() == pending - 2U);
        assert(std::none_of(queued.begin(), queued.end(), [](const Write& w) { return w.qp == 2U; }));
#endif
        for (uint32_t qp = 1; qp <= qps; ++qp) {
            expectedQp = qp;
            RoceFinishOutstanding(kPeer, scratch.data(), nullptr, phase, qp);
            assert(std::none_of(queued.begin(), queued.end(), [qp](const Write& w) { return w.qp == qp; }));
        }
        assert(queued.empty());
        local[sequenceOffset] ^= 1U;  // All queues have released the common sequence source.
    }
}
