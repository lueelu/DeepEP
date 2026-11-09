#pragma once

#include <cstdint>

// This ABI is private to the diagnostic extension, not a DeepEP operator API.
namespace udma_bench {
constexpr uint32_t kLanes = 32;
constexpr uint32_t kTimes = 6;
constexpr uint64_t kSignalStride = 512;
constexpr uint64_t kCyclesPerUs = 1000;  // Ascend950 system counter, not AIV clock.

struct Config {
    uint64_t capacity, bytes, chunk, epoch;
    uint32_t rank, world, cores, peers, visit;
    uint32_t mode, source, destination, delta;
};

#ifdef __CCE_AICORE__
#define BENCH_FN __aicore__ inline
#else
#define BENCH_FN inline
#endif
BENCH_FN bool Edge(const Config& c, uint32_t src, uint32_t dst)
{
    if (src == dst) return false;
    if (c.mode == 0) return src == c.source && dst == c.destination;
    if (c.mode == 1) return dst == (src + c.delta) % c.world;
    return true;
}
BENCH_FN uint64_t SendOffset(const Config& c)
{
    return c.world * kSignalStride;
}
BENCH_FN uint64_t RecvOffset(const Config& c)
{
    return SendOffset(c) + c.world * c.capacity;
}
#undef BENCH_FN
}  // namespace udma_bench

extern "C" void udma_bench_prepare(void*, void*, udma_bench::Config, uint64_t);
extern "C" void udma_bench_transfer(void*, void*, void*, udma_bench::Config, uint64_t);
extern "C" void udma_bench_verify(void*, void*, void*, udma_bench::Config, uint64_t);
