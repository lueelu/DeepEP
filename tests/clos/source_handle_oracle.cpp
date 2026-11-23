// CPU execution of the production Source builder with checked DMA/event mocks.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#define __aicore__
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
#define __ubuf__
#define ASCEND_DEEPEP_ARCH 3510
#include "common/platform.h"
using topk_idx_t = TOPK_TYPE;
using namespace ascend_deepep;
constexpr int EVENT_ID4 = 4;
struct dim3 {
    unsigned x;
    explicit dim3(unsigned n) : x(n) {}
};
dim3 threadIdx(0);
void asc_threadfence() {}

static std::array<bool, 7> events{};
static uint8_t* ub_base;
static const uint8_t *prefix_base, *source_base, *topk_base;
static uint8_t* output_base;
static size_t prefix_bytes, source_bytes, topk_bytes, output_bytes;
static unsigned prefix_loads, active_core;
static std::vector<unsigned> writes;
static std::vector<int> line_owner;

namespace AscendC {
enum class HardEvent { MTE2_S, MTE2_V, V_MTE3, MTE3_V, V_MTE2, MTE3_S, V_S };
template <HardEvent E>
void SetFlag(int id)
{
    assert(id == EVENT_ID4 && !events[size_t(E)]);
    events[size_t(E)] = true;
}
template <HardEvent E>
void WaitFlag(int id)
{
    assert(id == EVENT_ID4 && events[size_t(E)]);
    events[size_t(E)] = false;
}
template <class T>
struct LocalTensor {
    T* data;
    T* GetPhyAddr()
    {
        return data;
    }
    T GetValue(size_t i) const
    {
        return data[i];
    }
};
template <class T>
struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
    GlobalTensor operator[](size_t i) const
    {
        return {data + i};
    }
};
struct DataCopyExtParams {
    uint32_t blocks, bytes, src_stride, dst_stride, reserved;
};
template <class T>
struct DataCopyPadExtParams {
    bool pad;
    uint8_t left, right;
    T value;
};
bool within(const void* p, size_t n, const uint8_t* base, size_t bytes)
{
    const auto address = uintptr_t(p), start = uintptr_t(base);
    return address >= start && n <= bytes && address - start <= bytes - n;
}
template <class T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(copy.blocks == 1 && !copy.src_stride && !copy.dst_stride);
    assert(within(dst.data, copy.bytes, ub_base, platform::kKernelUbBytes));
    assert((reinterpret_cast<uint8_t*>(dst.data) - ub_base) % 32 == 0);
    if (within(src.data, copy.bytes, prefix_base, prefix_bytes))
        ++prefix_loads;
    else
        assert(within(src.data, copy.bytes, source_base, source_bytes) ||
               within(src.data, copy.bytes, topk_base, topk_bytes));
    std::memcpy(dst.data, src.data, copy.bytes);
}
template <class T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    assert(copy.blocks == 1 && !copy.src_stride && !copy.dst_stride);
    assert(within(src.data, copy.bytes, ub_base, platform::kKernelUbBytes));
    assert((reinterpret_cast<uint8_t*>(src.data) - ub_base) % 32 == 0);
    assert(within(dst.data, copy.bytes, output_base, output_bytes));
    const size_t offset = reinterpret_cast<uint8_t*>(dst.data) - output_base;
    for (size_t i = offset / 4; i < (offset + copy.bytes) / 4; ++i) {
        assert(++writes[i] == 1);
        auto& owner = line_owner[i / 8];
        assert(owner == -1 || owner == int(active_core));
        owner = int(active_core);
    }
    std::memcpy(dst.data, src.data, copy.bytes);
}
}  // namespace AscendC
struct Resource {
    std::vector<uint64_t> ub = std::vector<uint64_t>(platform::kKernelUbBytes / 8, 0xccccccccccccccccULL);
    template <class T>
    AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset)
    {
        ub_base = reinterpret_cast<uint8_t*>(ub.data());
        assert(offset % 32 == 0 && offset < platform::kKernelUbBytes);
        return {reinterpret_cast<T*>(ub_base + offset)};
    }
};
template <auto Fn, class... Args>
void asc_vf_call(dim3 threads, Args... args)
{
    for (threadIdx.x = 0; threadIdx.x < threads.x; ++threadIdx.x) Fn(args...);
}

// PRODUCTION

void check(unsigned world, unsigned cores, unsigned k, unsigned valid, unsigned padding, unsigned pattern)
{
    const unsigned rank = world / 2, rows = valid + padding, capacity = 4096;
    std::vector<int32_t> prefix(world), matrix(world * world, -777), src(valid);
    std::vector<topk_idx_t> topk(size_t(valid) * k, -1);
    std::vector<int32_t> expected(size_t(rows) * (k + 2), -1), out(expected.size(), 12345);
    // Uniform, all at last rank (leading empty ranks), and alternating empty ranks.
    for (unsigned p = 0; p < world; ++p) {
        const unsigned count = pattern == 1   ? (p + 1 == world ? valid : 0)
                               : pattern == 2 ? valid * ((p + 1) / 2) / (world / 2)
                                              : valid * (p + 1) / world;
        prefix[p] = count;
        matrix[size_t(p) * world + rank] = count;
    }
    for (unsigned row = 0; row < valid; ++row) {
        src[row] = (row * 17 + 3) % capacity;
        const unsigned first = k ? row % (k + 1) : 0;
        for (unsigned j = first; j < k; ++j) topk[size_t(row) * k + j] = 3 + j;
        const auto peer = unsigned(std::upper_bound(prefix.begin(), prefix.end(), int32_t(row)) - prefix.begin());
        expected[size_t(row) * (k + 2)] = peer * capacity + src[row];
        expected[size_t(row) * (k + 2) + 1] = peer * k + first;
    }
    prefix_base = reinterpret_cast<const uint8_t*>(prefix.data());
    prefix_bytes = prefix.size() * 4;
    source_base = reinterpret_cast<const uint8_t*>(src.data());
    source_bytes = src.size() * 4;
    topk_base = reinterpret_cast<const uint8_t*>(topk.data());
    topk_bytes = topk.size() * sizeof(topk_idx_t);
    output_base = reinterpret_cast<uint8_t*>(out.data());
    output_bytes = out.size() * 4;
    const auto tile = MakeV2SourceTile(world, k, platform::kKernelUbBytes);
    prefix_loads = 0;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        writes.assign(out.size(), 0);
        line_owner.assign((out.size() + 7) / 8, -1);
        std::fill(out.begin(), out.end(), -12345);
        for (active_core = 0; active_core < cores; ++active_core) {
            Resource resource;
            BuildV2HandleSource(resource, prefix.data(), matrix.data(), src.data(), topk.data(), out.data(), rank,
                                world, capacity, k, rows, active_core, cores);
            for (bool pending : events) assert(!pending);
        }
        assert(out == expected);
        if (tile.rows)
            for (unsigned n : writes) assert(n == 1);
    }
    const unsigned per_core = (((rows + 7) / 8) + cores - 1) / cores;
    const unsigned active = rows ? (rows + per_core * 8 - 1) / (per_core * 8) : 0;
    assert(prefix_loads == (tile.rows ? 2 * active : 0));
    // Legacy/fallback encoding must remain exactly identical.
    std::fill(out.begin(), out.end(), 12345);
    for (unsigned core = 0; core < cores; ++core)
        asc_vf_call<V2HandleSourceVf>(dim3(128), matrix.data(), src.data(), topk.data(), out.data(), rank, world,
                                      capacity, k, rows, core, cores);
    assert(out == expected);
}

int main()
{
    for (unsigned world : {8U, 16U, 32U, 128U})
        for (unsigned cores : {8U, 32U, 48U, 64U})
            for (unsigned k : {0U, 1U, 3U, 8U, 17U})
                for (unsigned valid : {0U, 1U, 7U, 65U, 257U, 4099U})
                    for (unsigned pattern = 0; pattern < 3; ++pattern)
                        check(world, cores, k, valid, valid % 9, pattern);
    check(16, 8, 8, 0, 19, 0);
    check(16, 8, 128, 257, 7, 2);
    check(16, 8, 4096, 17, 3, 1);  // Less than eight rows fit: legacy fallback.
}
