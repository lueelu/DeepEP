// Copyright (c) 2026, Lu Lu

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include "kernel_operator.h"

// The separate progress test exercises the device CQ/SQ implementation.
#define MEGA_MOE_ROCE_PROGRESS_H

constexpr int ACLSHMEM_CMP_EQ = 0;

namespace RoceTest {
constexpr uint32_t kPeer = 21U;
constexpr uint32_t kCommitOffset = 1536U;
constexpr uint32_t kCombineCommitOffset = 1024U;
inline std::array<uint8_t, 4096> local{};
inline std::array<uint8_t, 4096> remote{};
inline std::array<uint8_t, 256> scratch{};
struct Write {
    uint32_t destination;
    uint8_t* source;
    uint32_t bytes;
    std::vector<uint8_t> sourceAtPost;
    bool delivered = false;
    bool commit = false;
    uint32_t qp = 1U;
};
inline std::vector<Write> queued;
inline uint32_t writes = 0;
inline uint32_t quiets = 0;
inline uint32_t progressCalls = 0;
inline uint32_t retired = 0;
inline uint32_t signalTests = 0;
inline bool cqError = false;
inline uint32_t expectedQp = 1U;
inline std::vector<uint32_t> extraCommitOffsets;
inline std::function<void()> onProgress;
inline std::function<void(const Write&)> onCommitVisible;

inline void CheckResource(uint32_t peer, uint32_t qp, uint8_t* ub, uint32_t event)
{
    if (peer != kPeer || qp != expectedQp || ub != scratch.data() || event != 7U) {
        throw std::runtime_error("transport used another owner's peer/QP/UB/event");
    }
}

inline uint32_t CheckSymmetric(uint8_t* address, uint32_t bytes)
{
    const auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(local.data());
    if (offset > local.size() || bytes > local.size() - offset) {
        throw std::runtime_error("operand was not a local symmetric address");
    }
    return static_cast<uint32_t>(offset);
}

inline void CheckSource(const Write& write)
{
    if (!std::equal(write.sourceAtPost.begin(), write.sourceAtPost.end(), write.source)) {
        throw std::runtime_error("in-flight GM source changed before local completion");
    }
}

inline void Deliver(Write& write)
{
    CheckSource(write);
    if (!write.delivered) {
        std::copy_n(write.source, write.bytes, remote.data() + write.destination);
        write.delivered = true;
        if (write.commit && onCommitVisible) onCommitVisible(write);
    }
}

// Model the supplied HNS contract: a marker's only fragment is SO, so it
// becomes visible after every preceding WQE. Reverse the data within each
// group to avoid accidentally relying on their individual execution order.
// Delivery does not consume CQEs or release the sender's lifetime contract.
inline void DeliverPosted(uint32_t qp = 1U)
{
    std::vector<size_t> entries;
    for (size_t i = 0; i < queued.size(); ++i)
        if (queued[i].qp == qp) entries.push_back(i);
    size_t begin = 0U;
    while (begin < entries.size()) {
        size_t end = begin;
        while (end < entries.size() && !queued[entries[end]].commit) ++end;
        for (size_t i = end; i > begin; --i) Deliver(queued[entries[i - 1U]]);
        if (end < entries.size()) Deliver(queued[entries[end++]]);
        begin = end;
    }
}
}  // namespace RoceTest

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
template <typename T>
void aclshmemx_roce_qp_put_nbi(T* destination, T* source, T* scratch, uint32_t count, int peer, uint32_t qp,
                               uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace RoceTest;
    CheckResource(static_cast<uint32_t>(peer), qp, scratch, event);
    const uint32_t offset = CheckSymmetric(destination, count);
    CheckSymmetric(source, count);
    const bool commit =
        offset == kCommitOffset || offset == kCombineCommitOffset ||
        std::find(extraCommitOffsets.begin(), extraCommitOffsets.end(), offset) != extraCommitOffsets.end();
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    if (commit && std::any_of(queued.begin(), queued.end(), [qp](const Write& w) { return w.qp == qp; })) {
        throw std::runtime_error("XSCALE commit can overtake outstanding data writes");
    }
#endif
    queued.push_back({offset, source, count, std::vector<uint8_t>(source, source + count), false, commit, qp});
    ++writes;
    operations.push_back(1);
}

template <typename T>
void aclshmemx_roce_qp_quiet(uint32_t peer, uint32_t qp, T* scratch, uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace RoceTest;
    CheckResource(peer, qp, scratch, event);
    if (cqError) throw std::runtime_error("completion error");
    DeliverPosted(qp);
    auto remaining = std::remove_if(queued.begin(), queued.end(), [qp](const Write& w) { return w.qp == qp; });
    retired += static_cast<uint32_t>(queued.end() - remaining);
    queued.erase(remaining, queued.end());
    ++quiets;
    operations.push_back(2);
}
#endif

namespace MegaMoeImpl {
inline uint32_t RoceProgressReady(uint32_t peer, uint32_t qp, uint8_t* scratch, uint32_t event)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    using namespace RoceTest;
    CheckResource(peer, qp, scratch, event);
    ++progressCalls;
    if (cqError) throw std::runtime_error("ready CQE error");
    if (onProgress) onProgress();
    uint32_t consumed = 0U;
    for (auto it = queued.begin(); it != queued.end();) {
        if (it->qp != qp) {
            ++it;
            continue;
        }
        if (!it->delivered) break;
        CheckSource(*it);
        it = queued.erase(it);
        ++consumed;
    }
    retired += consumed;
    return consumed;
#else
    (void)peer;
    (void)qp;
    (void)scratch;
    (void)event;
    return 0U;
#endif
}
}  // namespace MegaMoeImpl

inline int TestSignal(int32_t* address, int comparison, int32_t value)
{
    ++RoceTest::signalTests;
    int32_t observed;
    std::memcpy(&observed, address, sizeof(observed));
    if (comparison != ACLSHMEM_CMP_EQ) throw std::runtime_error("invalid signal comparison");
    if (observed != value) return 0;
    RoceTest::signalObserved = true;
    return 1;
}

#if defined(MEGAMOE_TEST_HOST_PASS)
// The real pinned SDK declares this distinct signature during host compilation.
inline void aclshmem_int32_test(int32_t* address, int comparison, int32_t value, int* result)
{
    *result = TestSignal(address, comparison, value);
}
#else
inline int aclshmem_int32_test(int32_t* address, int comparison, int32_t value)
{
    return TestSignal(address, comparison, value);
}
#endif

inline int32_t aclshmem_signal_wait_until(int32_t* address, int comparison, int32_t value)
{
    if (!TestSignal(address, comparison, value)) {
        throw std::runtime_error("receiver did not wait for its exact invocation");
    }
    return value;
}
