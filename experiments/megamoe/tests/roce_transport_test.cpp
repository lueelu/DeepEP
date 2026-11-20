// Copyright (c) 2026, Lu Lu

#include <cassert>
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"

int main()
{
    using namespace MegaMoeImpl;
    using namespace RoceTest;
    static_assert(MEGAMOE_ROCE_SCRATCH_BYTES == scratch.size());
    constexpr uint32_t sourceOffset = 2048U;
    constexpr uint32_t sequenceOffset = 3072U;
    constexpr uint32_t sequence = 130U;  // Past the old route epoch's 7-bit wrap.

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    remote.fill(0xA5U);
    for (uint32_t i = 0; i < 96U; ++i) {
        local[sourceOffset + i] = static_cast<uint8_t>(i);
    }
    std::memcpy(local.data() + sequenceOffset, &sequence, sizeof(sequence));
    RocePutBytes(local.data(), local.data() + sourceOffset, 32U, kPeer, scratch.data());
    RocePutBytes(local.data() + 64U, local.data() + sourceOffset + 64U, 32U, kPeer, scratch.data());
    assert(remote[0] == 0xA5U);  // Put is asynchronous.
    scratch.fill(0xCCU);         // In-flight operands are GM; SDK scratch can be reused.
    RocePutBytes(nullptr, nullptr, 0U, kPeer, scratch.data());
    assert(writes == 2U);
    uint32_t commitsSeen = 0U;
    onCommitVisible = [&](const Write&) {
        ++commitsSeen;
        // This callback runs at the first instant the marker is visible.
        for (uint32_t i = 0; i < 96U; ++i) {
            assert(remote[i] == (i >= 32U && i < 64U ? 0xA5U : static_cast<uint8_t>(i)));
        }
    };
    RocePublishCommit(local.data() + kCommitOffset, local.data() + sequenceOffset, kPeer, scratch.data());
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    assert(queued.size() == 3U && writes == 3U && quiets == 0U);
    assert((operations == std::vector<int>{1, 1, 1}));
    assert(remote[0] == 0xA5U && commitsSeen == 0U);
    DeliverPosted();  // NIC progress is independent from sender CQ consumption.
    assert(queued.size() == 3U && retired == 0U && commitsSeen == 1U);
#else
    assert(queued.empty());
    assert(writes == 3U && quiets == 2U);
    assert((operations == std::vector<int>{1, 1, 2, 1, 2}));
#endif
    for (uint32_t i = 0; i < 96U; ++i) {
        assert(remote[i] == (i >= 32U && i < 64U ? 0xA5U : static_cast<uint8_t>(i)));
    }
    assert(!mteCanRead);
    RoceWaitCommit(reinterpret_cast<int32_t*>(remote.data() + kCommitOffset), sequence);
    assert(mteCanRead && !scalarEventSet);
    assert(progressCalls == 0U);  // The receiver never consumes the sender's CQ.

#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // Dispatch and Combine share one final completion. Their 4B markers read
    // the same immutable Sequence while both phases may still have live CQEs.
    RocePutBytes(local.data() + 128U, local.data() + sourceOffset + 64U, 32U, kPeer, scratch.data());
    RocePublishCommit(local.data() + kCombineCommitOffset, local.data() + sequenceOffset, kPeer, scratch.data());
    assert(queued.size() == 5U && writes == 5U && quiets == 0U);
    DeliverPosted();
    signalObserved = scalarEventSet = mteCanRead = false;
    RoceWaitCommit(reinterpret_cast<int32_t*>(remote.data() + kCombineCommitOffset), sequence, nullptr,
                   AscendTimer::RdmaPhase::Combine, kPeer, scratch.data());
    assert(mteCanRead && queued.size() == 5U && progressCalls == 0U);
    assert(std::equal(remote.begin() + 128U, remote.begin() + 160U, local.begin() + sourceOffset + 64U));
#endif
    RoceFinishOutstanding(kPeer, scratch.data());
    assert(queued.empty());
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    assert(quiets == 1U && retired == 5U);
    assert((operations == std::vector<int>{1, 1, 1, 1, 1, 2}));
#else
    assert(quiets == 2U && retired == 3U);  // XSCALE did not add another quiet.
#endif
    local[sourceOffset] = 0xEEU;  // Safe only after the completion boundary.
    assert(remote[0] == 0U);

    // Zero payload still commits a fresh epoch; receivers cannot reuse old flags.
    const uint32_t nextSequence = sequence + 1U;
    std::memcpy(local.data() + sequenceOffset, &nextSequence, sizeof(nextSequence));
    RocePublishCommit(local.data() + kCommitOffset, local.data() + sequenceOffset, kPeer, scratch.data());
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    assert(queued.size() == 1U && quiets == 1U);
#endif
    RoceFinishOutstanding(kPeer, scratch.data());
    int32_t observed;
    std::memcpy(&observed, remote.data() + kCommitOffset, sizeof(observed));
    assert(observed == static_cast<int32_t>(nextSequence));
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    assert(writes == 6U && quiets == 2U);

    // A source overwrite before completion is caught for payload and Sequence.
    // Merely posting an SO marker does not grant local buffer reuse.
    for (const uint32_t offset : {sourceOffset, sequenceOffset}) {
        if (offset == sourceOffset) {
            RocePutBytes(local.data() + 256U, local.data() + offset, 32U, kPeer, scratch.data());
        } else {
            RocePublishCommit(local.data() + kCommitOffset, local.data() + offset, kPeer, scratch.data());
        }
        local[offset] ^= 1U;
        bool rejected = false;
        try {
            RoceFinishOutstanding(kPeer, scratch.data());
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
        local[offset] ^= 1U;
        queued.clear();
    }

    // Stale and future epochs both time out without acquiring payload or
    // touching the sender's CQ on a nonowner receive leader.
    auto* flag = reinterpret_cast<int32_t*>(remote.data() + kCommitOffset);
    for (uint32_t wrong : {nextSequence - 1U, nextSequence + 1U}) {
        std::memcpy(flag, &wrong, sizeof(wrong));
        signalObserved = scalarEventSet = mteCanRead = false;
        cycleStep = static_cast<int64_t>(MEGAMOE_ROCE_WAIT_TIMEOUT_CYCLES);
        const auto diagnosticsBefore = timeoutDiagnostics;
        const auto pollsBefore = progressCalls;
        bool rejected = false;
        try {
            RoceWaitCommit(flag, nextSequence, nullptr, AscendTimer::RdmaPhase::Combine, kPeer);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected && timeoutDiagnostics == diagnosticsBefore + 1U);
        assert(lastWaitKind == static_cast<uint32_t>(RoceWaitKind::Commit) && lastWaitPeer == kPeer &&
               lastWaitExpected == nextSequence && lastWaitObserved == wrong);
        assert(progressCalls == pollsBefore && !mteCanRead);
    }

    // A slow owner wait progresses only ready CQEs at the millisecond cadence.
    // It never substitutes a blocking quiet for the failed signal observation.
    cycleStep = static_cast<int64_t>(MEGAMOE_ROCE_PROGRESS_INTERVAL_CYCLES / 2U);
    const auto pollsBefore = progressCalls;
    const auto testsBefore = signalTests;
    const auto quietsBefore = quiets;
    onProgress = [&] { std::memcpy(flag, &nextSequence, sizeof(nextSequence)); };
    RoceWaitCommit(flag, nextSequence, nullptr, AscendTimer::RdmaPhase::Combine, kPeer, scratch.data());
    assert(progressCalls == pollsBefore + 1U && signalTests >= testsBefore + 129U && quiets == quietsBefore);
    onProgress = nullptr;

    // CQ failure during a missing-commit wait must surface before a tail quiet
    // that might never become reachable. The nonowner timeout above never polls.
    const uint32_t stale = nextSequence - 1U;
    std::memcpy(flag, &stale, sizeof(stale));
    signalObserved = scalarEventSet = mteCanRead = false;
    cqError = true;
    cycleStep = static_cast<int64_t>(MEGAMOE_ROCE_PROGRESS_INTERVAL_CYCLES);
    bool errorObserved = false;
    try {
        RoceWaitCommit(flag, nextSequence, nullptr, AscendTimer::RdmaPhase::Combine, kPeer, scratch.data());
    } catch (const std::runtime_error&) {
        errorObserved = true;
    }
    assert(errorObserved && !mteCanRead && quiets == quietsBefore);
    cqError = false;
    cycleStep = 1;
#else
    assert(writes == 4U && quiets == 4U);
#endif
#else
    bool trapped = false;
    try {
        RocePutBytes(local.data(), local.data() + sourceOffset, 32U, kPeer, scratch.data());
    } catch (const std::runtime_error&) {
        trapped = true;
    }
    assert(trapped);
    trapped = false;
    try {
        RoceQuiet(kPeer, scratch.data());
    } catch (const std::runtime_error&) {
        trapped = true;
    }
    assert(trapped);
    (void)sequenceOffset;
    (void)sequence;
#endif
    return 0;
}
