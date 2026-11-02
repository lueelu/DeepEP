#pragma once
// Synchronous CPU model for scheduler arithmetic only, never a device emulator.
#include <cstdint>
#include <cstring>
#include <cassert>
#include <array>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <vector>
#include <functional>
#include <cstdio>
#include <string>
#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) throw std::runtime_error("CPU check: " #condition); \
    } while (0)
#define __aicore__
#define __gm__
#define __ubuf__
#define __simt_vf__
#define __launch_bounds__(n)
using GM_ADDR = uint8_t*;
enum { EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID4 = 4, EVENT_ID5 = 5, EVENT_ID6 = 6, PIPE_MTE3 = 3, PIPE_V = 2 };
struct dim3 {
    unsigned x;
    explicit dim3(unsigned n) : x(n) {}
};
inline dim3 threadIdx(0);
inline void trap()
{
    throw std::runtime_error("device guard");
}
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S, S_MTE2, MTE2_V, V_S, MTE2_S, S_V, MTE3_MTE2, MTE2_MTE3, V_MTE3, MTE3_V };
inline std::map<std::pair<HardEvent, unsigned>, bool> pending_events;
inline unsigned local_syncs = 0;
template <unsigned>
void PipeBarrier()
{
}  // Arithmetic model executes VFs synchronously.
// Virtual time and invalidation hooks let tests exercise the real polling loop
// without sleeping or claiming to model hardware cache visibility.
inline uint64_t system_cycle = 0, cycle_step = 0;
inline unsigned cycle_reads = 0, invalidations = 0;
inline std::function<void(uint8_t*)> on_invalidate;
inline std::vector<std::string> diagnostics;
inline int64_t GetSystemCycle()
{
    ++cycle_reads;
    const uint64_t now = system_cycle;
    system_cycle += cycle_step;
    return static_cast<int64_t>(now);
}
template <class... A>
void printf(const char* format, A... args)
{
    char buffer[1024];
    const int length = std::snprintf(buffer, sizeof(buffer), format, args...);
    CHECK(length >= 0 && size_t(length) < sizeof(buffer));
    diagnostics.emplace_back(buffer);
}
struct CopyRecord {
    const void* src;
    void* dst;
    size_t bytes;
    bool to_gm;
};
inline bool record_copies = false;
inline std::vector<CopyRecord> copy_records;
template <HardEvent E>
void SetFlag(unsigned id)
{
    auto& pending = pending_events[{E, id}];
    CHECK(!pending);
    pending = true;
}
template <HardEvent E>
void WaitFlag(unsigned id)
{
    auto& pending = pending_events[{E, id}];
    CHECK(pending);
    pending = false;
}
inline void CheckDrained()
{
    for (const auto& entry : pending_events) CHECK(!entry.second);
}
template <bool>
void SyncAll()
{
    CheckDrained();
    ++local_syncs;
}
struct DataCopyExtParams {
    uint32_t count, bytes, srcStride, dstStride, pad;
};
template <class T>
struct DataCopyPadExtParams {
    bool enabled;
    uint32_t left, right;
    T value;
};
template <class T>
struct LocalTensor {
    T* p;
    LocalTensor operator[](size_t i) const
    {
        return {p + i};
    }
    T GetValue(size_t i) const
    {
        return p[i];
    }
    void SetValue(size_t i, T x)
    {
        p[i] = x;
    }
    T* GetPhyAddr() const
    {
        return p;
    }
};
template <class T>
struct GlobalTensor {
    T* p{};
    void SetGlobalBuffer(T* ptr)
    {
        p = ptr;
    }
    GlobalTensor operator[](size_t i) const
    {
        return {p + i};
    }
    T GetValue(size_t i) const
    {
        return p[i];
    }
};
template <class T>
void Duplicate(LocalTensor<T> dst, T value, uint32_t count)
{
    CHECK(reinterpret_cast<uintptr_t>(dst.p) % 32 == 0);
    std::fill(dst.p, dst.p + count, value);
}
template <class T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams c, DataCopyPadExtParams<T>)
{
    if (record_copies) copy_records.push_back({src.p, dst.p, c.bytes, false});
    CHECK(c.count == 1 && reinterpret_cast<uintptr_t>(dst.p) % 32 == 0);
    std::memcpy(dst.p, src.p, c.bytes);
}
template <class T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams c)
{
    if (record_copies) copy_records.push_back({src.p, dst.p, c.bytes, true});
    CHECK(c.count == 1 && reinterpret_cast<uintptr_t>(src.p) % 32 == 0);
    std::memcpy(dst.p, src.p, c.bytes);
}
}  // namespace AscendC
inline void dcci_cacheline(uint8_t* address)
{
    ++AscendC::invalidations;
    if (AscendC::on_invalidate) AscendC::on_invalidate(address);
}
template <auto F, class... A>
void asc_vf_call(dim3 threads, A... args)
{
    for (threadIdx.x = 0; threadIdx.x < threads.x; ++threadIdx.x) F(args...);
}
struct Resource {
    alignas(64) std::array<uint8_t, 216 * 1024> bytes{};
    template <class T>
    AscendC::LocalTensor<T> GetBufferByByte(size_t offset)
    {
        CHECK(offset < bytes.size() && offset % 32 == 0);
        return {reinterpret_cast<T*>(bytes.data() + offset)};
    }
};
