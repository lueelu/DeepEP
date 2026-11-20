// Copyright (c) 2026, Lu Lu

// Model the SDK host pass, which declares a four-argument void int32_test.
// The device-only three-argument call must not be parsed on this path.
#define MEGAMOE_TEST_HOST_PASS 1
#include <cassert>
#include <type_traits>
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"

int main()
{
    using namespace MegaMoeImpl;
    using namespace RoceTest;
    static_assert(std::is_same_v<decltype(&aclshmem_int32_test), void (*)(int32_t*, int, int32_t, int*)>);
    constexpr int32_t sequence = 137;
    auto* flag = reinterpret_cast<int32_t*>(local.data() + kCommitOffset);
    std::memcpy(flag, &sequence, sizeof(sequence));
    RoceWaitState wait;
    RoceCheckWait(wait, kPeer, scratch.data(), RoceWaitKind::Commit, sequence, sequence);
    assert(!wait.started && progressCalls == 0U);
    RoceWaitCommit(flag, sequence, nullptr, AscendTimer::RdmaPhase::Combine, kPeer, scratch.data());
    assert(mteCanRead && progressCalls == 0U && timeoutDiagnostics == 0U);
    return 0;
}
