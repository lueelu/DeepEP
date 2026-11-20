// Copyright (c) 2026, Lu Lu

// Run the production local acquisition with asynchronous snapshot/event doubles.
// This checks protocol and UB bounds, not NPU cache coherence or DMA visibility.
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define __aicore__
#define __gm__
#define GM_ADDR uint8_t*
#define MEGA_MOE_UTILS_H
constexpr int AIC = 0;
#ifdef GATEWAY_TEST_AIC
constexpr int g_coreType = AIC;
#else
constexpr int g_coreType = 1;
#endif

namespace GatewayTest {
constexpr uint32_t slots = 16, wordsPerLine = 128, scratchOffset = 4096;
struct State {
    std::mutex mutex;
    std::condition_variable changed;
    alignas(512) std::array<int32_t, (slots + 2) * wordsPerLine> flags{};
    std::array<uint32_t, 32> snapshots{}, finished{};
    std::exception_ptr failure;
    uint32_t groups = 1, sequence = 1;
    bool aborted = false;
    int32_t* Flags()
    {
        return flags.data() + wordsPerLine;
    }
};
inline State* state;
thread_local uint32_t group = 0, lane = 1, reads = 0, mask = 0;
thread_local bool armed = false, pending = false, readable = false, acquired = false;
thread_local std::array<int32_t, 128> ub;
inline void Check(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
}  // namespace GatewayTest

namespace MegaMoeImpl {
using namespace GatewayTest;
constexpr uint32_t HIERARCHICAL_RANKS_PER_HOST = 16;
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512;
constexpr uint32_t HIERARCHICAL_ROCE_UB_BYTES = 512;
struct BlockWorkspaceContext {
    uint32_t blockIdx = 0, blockNum = 1;
};
struct Tiling {
    uint32_t hierarchicalSequence = 1, roceScratchUbOffset = scratchOffset;
    bool hierarchical = true;
};
struct Params {
    Tiling* tilingData;
    struct {
        GM_ADDR gatewayPayloadReadyPtr;
    } peermemInfo;
};
inline uint32_t GetSubBlockIdx()
{
    return lane;
}
inline bool UseHierarchicalRoce(const Tiling* t)
{
    return t->hierarchical;
}
enum class TPosition { VECCALC };
enum class HardEvent { S_MTE2, MTE2_S };
struct DataCopyParams {
    uint16_t blockCount, blockLen, srcStride, dstStride;
};
template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
};
template <typename T>
struct LocalTensor {
    LocalTensor(TPosition, uint32_t offset, uint32_t count)
    {
        Check(offset == scratchOffset && count == 128, "snapshot exceeded reserved UB");
    }
    T GetValue(uint32_t index) const
    {
        Check(readable && index < ub.size() && index % 8 == 0, "invalid/unordered snapshot read");
        const uint32_t bit = 1U << (index / 8);
        Check((reads & bit) == 0, "snapshot read a slot twice");
        reads |= bit;
        if (static_cast<uint32_t>(ub[index]) == state->sequence) mask |= bit;
        return ub[index];
    }
};
inline void DataCopy(LocalTensor<int32_t>, GlobalTensor<int32_t> flags, DataCopyParams p)
{
    Check(armed && !pending && flags.data == state->Flags(), "DMA not local or missing UB reuse ordering");
    Check(p.blockCount == 16 && p.blockLen == 1 && p.srcStride == 15 && p.dstStride == 0,
          "expected one 16-slot strided DMA (512B total)");
    armed = false;
    pending = true;
    acquired = false;
}
template <HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    static_assert(Id == 7);
    if constexpr (Event == HardEvent::S_MTE2) {
        Check(!pending, "payload/reuse acquire preceded DMA completion");
        if (readable) {
            Check(reads == 0xffffU, "consumer did not compare all sixteen slots");
            acquired = mask == 0xffffU;
            readable = false;
        }
        armed = true;
    } else {
        Check(pending, "DMA completion event without DMA");
        auto& s = *state;
        std::unique_lock<std::mutex> lock(s.mutex);
        Check(!s.aborted, "another consumer failed");
        for (uint32_t i = 0; i < 16; ++i)
            for (uint32_t word = 0; word < 8; ++word) ub[i * 8 + word] = s.Flags()[i * 128 + word];
        ++s.snapshots[group];
        s.changed.notify_all();
        pending = false;
        readable = true;
        reads = mask = 0;
        lock.unlock();
        std::this_thread::yield();
    }
}
}  // namespace MegaMoeImpl

#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_gateway_readiness.hpp"

namespace GatewayTest {
using namespace MegaMoeImpl;
struct Fixture {
    State s;
    Tiling t;
    Params p;
    std::vector<std::thread> consumers;
    Fixture(uint32_t groups, uint32_t sequence) : p{&t, {reinterpret_cast<GM_ADDR>(s.Flags())}}
    {
        state = &s;
        s.groups = groups;
        s.sequence = t.hierarchicalSequence = sequence;
        s.flags.fill(0x62626262);
        for (uint32_t slot = 0; slot < slots; ++slot)
            s.Flags()[slot * wordsPerLine] = static_cast<int32_t>(sequence - 128);
    }
    void Publish(uint32_t slot)
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.Flags()[slot * wordsPerLine] = static_cast<int32_t>(s.sequence);
        s.changed.notify_all();
    }
    void Start(uint32_t index)
    {
        consumers.emplace_back([&, index] {
            group = index;
            try {
                AcquireHierarchicalGatewayPayload({index, s.groups}, p);
                Check(acquired && !pending && !readable, "consumer returned before complete snapshot/acquire");
                std::lock_guard<std::mutex> lock(s.mutex);
                ++s.finished[index];
                s.changed.notify_all();
            } catch (...) {
                std::lock_guard<std::mutex> lock(s.mutex);
                if (!s.failure) s.failure = std::current_exception();
                s.aborted = true;
                s.changed.notify_all();
            }
        });
    }
    void WaitForPolls()
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        Check(s.changed.wait_for(lock, std::chrono::seconds(3),
                                 [&] {
                                     if (s.failure) return true;
                                     for (uint32_t i = 0; i < s.groups; ++i)
                                         if (s.snapshots[i] == 0) return false;
                                     return true;
                                 }),
              "consumer did not take a snapshot");
        if (s.failure) std::rethrow_exception(s.failure);
        for (uint32_t i = 0; i < s.groups; ++i) Check(s.finished[i] == 0, "stale/missing slot released consumer");
    }
    void Join()
    {
        for (auto& worker : consumers) worker.join();
        if (s.failure) std::rethrow_exception(s.failure);
        for (uint32_t i = 0; i < s.groups; ++i) Check(s.finished[i] == 1, "consumer lost notification");
        for (uint32_t i = 0; i < s.flags.size(); ++i) {
            const bool slot = i >= wordsPerLine && i < (slots + 1) * wordsPerLine && i % wordsPerLine == 0;
            Check(s.flags[i] == (slot ? static_cast<int32_t>(s.sequence) : 0x62626262),
                  "consumer wrote or cleared flags");
        }
    }
    ~Fixture()
    {
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.aborted = true;
        }
        for (auto& worker : consumers)
            if (worker.joinable()) worker.join();
    }
};
void Run(uint32_t groups, uint32_t sequence, int missing = -1, bool lateConsumer = false)
{
    Fixture f(groups, sequence);
    for (uint32_t slot = 0; slot < slots; ++slot)
        if (static_cast<int>(slot) != missing) f.Publish(slot);
    const uint32_t initial = lateConsumer ? groups - 1 : groups;
    for (uint32_t i = 0; i < initial; ++i) f.Start(i);
    if (missing >= 0) {
        f.WaitForPolls();
        f.Publish(static_cast<uint32_t>(missing));
    }
    if (lateConsumer) f.Start(groups - 1);
    f.Join();
}
void Epochs()
{
    Fixture f(16, 1);
    for (uint32_t sequence : {1U, 129U, 257U, 0x80000101U, 0xffffffffU}) {
        // Reuse the SAME inbox without clearing it, after every prior consumer
        // has left. Old ready values must never release the new invocation.
        f.consumers.clear();
        f.s.snapshots.fill(0);
        f.s.finished.fill(0);
        f.s.sequence = f.t.hierarchicalSequence = sequence;
        for (uint32_t i = 0; i < 16; ++i) f.Start(i);
        f.WaitForPolls();
        for (uint32_t slot = 0; slot < 16; ++slot) f.Publish(slot);
        f.Join();
    }
}
void Guards()
{
    Fixture f(1, 1);
    for (uint32_t subblock : {0U, 1U})
        for (bool hierarchical : {false, true})
            for (auto workspace :
                 {BlockWorkspaceContext{0, 0}, BlockWorkspaceContext{1, 1}, BlockWorkspaceContext{0, 1}}) {
                if (g_coreType != AIC && subblock == 1 && hierarchical && workspace.blockNum && workspace.blockIdx == 0)
                    continue;
                lane = subblock;
                f.t.hierarchical = hierarchical;
                AcquireHierarchicalGatewayPayload(workspace, f.p);
                Check(f.s.snapshots[0] == 0, "inactive core/topology accessed readiness");
            }
}
}  // namespace GatewayTest
int main(int argc, char** argv)
{
    using namespace GatewayTest;
    try {
        Check(argc == 2, "missing scenario");
        const std::string name = argv[1];
        if (name == "guards")
            Guards();
        else if (name == "distribution")
            for (uint32_t groups : {1U, 7U, 16U, 28U}) Run(groups, 257);
        else if (name == "late-arrival")
            for (int slot = 0; slot < 16; ++slot) Run(28, 257, slot);
        else if (name == "epochs")
            Epochs();
        else if (name == "late-consumer")
            Run(28, 257, -1, true);
        else
            throw std::runtime_error("unknown scenario");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
