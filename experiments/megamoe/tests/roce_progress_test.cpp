// Copyright (c) 2026, Lu Lu

#include <cassert>
#include <limits>
#define __CCE_KT_TEST__ 1
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce_progress.hpp"

int main()
{
    using namespace MegaMoeImpl;
    using namespace RoceProgressTest;
    auto progress = [] { return RoceProgressReady(kPeer, kQp, scratch.data(), 7U); };
    Reset();
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    assert(progress() == 0U && copies == 0U && doorbells == 0U);

    // A not-yet-published CQE must return after one read, with no CI update.
    head = 1U;
    assert(progress() == 0U && copies == 1U && doorbells == 0U);
    Publish(0U, 31U);  // Matching owner alone does not make an invalid slot ready.
    assert(progress() == 0U && copies == 2U && doorbells == 0U);
    Publish(0U, 4U, false);  // A valid opcode with the previous lap's owner is stale.
    assert(progress() == 0U && copies == 3U && doorbells == 0U);

    Reset();
    head = 3U;
    Publish(0U);
    Publish(1U);
    assert(progress() == 2U && copies == 3U && syncs == 3U);
    assert(sqTail == 2U && cqTail == 2U && doorbells == 1U);
    assert(progress() == 0U && copies == 4U && doorbells == 1U);
    Publish(2U);
    Publish(3U);  // Do not consume entries beyond the submitted head snapshot.
    assert(progress() == 1U && sqTail == 3U && cqTail == 3U);
    assert(copies == 5U && doorbells == 2U);

    // A signaled WQE can acknowledge several SQ entries in a single CQE.
    Reset();
    sqTail = 65534U;
    head = 65539U;
    Publish(0U, 4U, true, 0U, head - 1U);
    assert(progress() == 1U && cqTail == 1U && sqTail == head);
    assert(progress() == 0U && copies == 1U);

    // Use the hardware stride and owner lap at wrap, including the default stride.
    for (uint32_t stride : {0U, 64U, 128U}) {
        Reset(7U, 8U, stride);
        head = 10U;
        Publish(7U);
        Publish(8U);
        Publish(9U);
        assert(progress() == 3U && sqTail == 10U && cqTail == 10U);
        assert(copies == 3U && doorbells == 1U);
    }
    Reset(8U);
    head = 9U;
    Publish(8U, 4U, false);
    assert(progress() == 0U && cqTail == 8U && doorbells == 0U);

    Reset(std::numeric_limits<uint32_t>::max() - 1U);
    head = 1U;
    Publish(std::numeric_limits<uint32_t>::max() - 1U);
    Publish(std::numeric_limits<uint32_t>::max());
    Publish(0U);
    assert(progress() == 3U && sqTail == 1U && cqTail == 1U);

    // Bound a hot CQ so receiver-side flag checks cannot be starved by progress.
    Reset(0U, 64U);
    head = MEGAMOE_ROCE_PROGRESS_MAX_CQES + 2U;
    for (uint32_t index = 0U; index < head; ++index) {
        Publish(index);
    }
    assert(progress() == MEGAMOE_ROCE_PROGRESS_MAX_CQES);
    assert(copies == MEGAMOE_ROCE_PROGRESS_MAX_CQES && doorbells == 1U);
    assert(progress() == 2U && cqTail == head && sqTail == head && doorbells == 2U);

    // Error detection survives partial progress; dump before UB is reused and
    // publish both tails through the failed entry before trapping.
    Reset();
    head = 3U;
    Publish(0U);
    Publish(1U, 30U, true, 0x5aU);
    Publish(2U);
    bool trapped = false;
    try {
        (void)progress();
    } catch (const std::runtime_error&) {
        trapped = true;
    }
    assert(trapped && dumps == 1U && dumpedSyndrome == 0x5aU);
    assert(sqTail == 2U && cqTail == 2U && copies == 2U && doorbells == 1U);
#else
    head = 1U;
    Publish(0U);
    assert(progress() == 0U && copies == 0U && doorbells == 0U);
    assert(sqTail == 0U && cqTail == 0U);
#endif
    return 0;
}
