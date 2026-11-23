// Synchronous CPU oracle for production DMA/VF bodies, not a device-ordering proof.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/platform.h"
using namespace ascend_deepep;
using GM_ADDR = uint8_t*;
constexpr int EVENT_ID2 = 2;
struct {
    unsigned x;
} threadIdx;
struct dim3 {
    unsigned x;
    explicit dim3(unsigned n) : x(n) {}
};
inline void trap()
{
    std::abort();
}
namespace __asc_simt_vf {
inline void __trap()
{
    std::abort();
}
}  // namespace __asc_simt_vf
template <auto Fn, class... Args>
void asc_vf_call(dim3 grid, Args... args)
{
    for (threadIdx.x = 0; threadIdx.x < grid.x; ++threadIdx.x) Fn(args...);
}
namespace AscendC {
enum class HardEvent { V_S, MTE2_V, V_MTE2, V_MTE3, MTE3_V, MTE3_S };
bool pending[6][3]{};
template <HardEvent E>
void SetFlag(int id)
{
    assert(id >= 0 && id < 3 && !pending[int(E)][id]);
    pending[int(E)][id] = true;
}
template <HardEvent E>
void WaitFlag(int id)
{
    assert(id >= 0 && id < 3 && pending[int(E)][id]);
    pending[int(E)][id] = false;
}
void CheckDrained()
{
    for (auto& row : pending)
        for (bool flag : row) assert(!flag);
}
template <class T>
struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
    GlobalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
template <class T>
struct LocalTensor {
    T* data{};
    T* GetPhyAddr() const
    {
        return data;
    }
    LocalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
struct DataCopyExtParams {
    uint32_t blocks, bytes, src_stride, dst_stride, reserved;
    DataCopyExtParams(uint32_t n, uint32_t b, uint32_t s, uint32_t d, uint32_t r)
        : blocks(n), bytes(b), src_stride(s), dst_stride(d), reserved(r)
    {
    }
};
template <class T>
struct DataCopyPadExtParams {
    bool pad;
    uint8_t left, right;
    T value;
};
uint8_t* ub_begin;
void CheckUb(const void* ptr, size_t bytes)
{
    auto p = reinterpret_cast<const uint8_t*>(ptr);
    assert(p >= ub_begin && p + bytes <= ub_begin + platform::kKernelUbBytes);
    assert((p - ub_begin) % 32 == 0);
}
template <class T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(copy.bytes && copy.blocks && copy.dst_stride == 0);
    const uint32_t aligned = (copy.bytes + 31) / 32 * 32;
    CheckUb(dst.data, copy.blocks * aligned);
    for (uint32_t row = 0; row < copy.blocks; ++row) {
        auto* d = reinterpret_cast<uint8_t*>(dst.data) + row * aligned;
        auto* s = reinterpret_cast<uint8_t*>(src.data) + uint64_t(row) * (copy.bytes + copy.src_stride);
        std::memcpy(d, s, copy.bytes);
        std::memset(d + copy.bytes, 0, aligned - copy.bytes);
    }
}
template <class T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    assert(copy.blocks == 1 && copy.bytes && copy.src_stride == 0 && copy.dst_stride == 0);
    CheckUb(src.data, copy.bytes);
    std::memcpy(dst.data, src.data, copy.bytes);
}
}  // namespace AscendC
struct Resource {
    std::vector<uint64_t> ub = std::vector<uint64_t>(platform::kKernelUbBytes / 8, ~0ULL);
    Resource()
    {
        AscendC::ub_begin = reinterpret_cast<uint8_t*>(ub.data());
    }
    template <class T>
    AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset)
    {
        assert(offset % 32 == 0 && offset < platform::kKernelUbBytes);
        return {reinterpret_cast<T*>(AscendC::ub_begin + offset)};
    }
};
#include "kernels/dispatch/dispatch_layout_simt.h"

template <class T>
GM_ADDR Address(std::vector<T>& v)
{
    return reinterpret_cast<GM_ADDR>(v.data());
}

void CheckCase(uint32_t world, uint32_t experts, uint32_t tokens, uint32_t k, uint32_t cores, uint32_t route)
{
    DispatchLayoutTilingData td{};
    td.world_size = world;
    td.num_experts = experts;
    td.num_tokens = tokens;
    td.num_topk = k;
    td.block_dim = cores;
    td.rank_stride = (world + 7) / 8 * 8;
    td.expert_stride = (experts + 7) / 8 * 8;
    std::vector<topk_idx_t> indices(std::max(1U, tokens * k), -1);
    // Extra sentinels catch incorrect tail stores and count-padding writes.
    std::vector<uint8_t> mask(tokens * world + 32, 99), ref_mask(tokens * world, 0);
    std::vector<int32_t> ranks(td.rank_stride + 8, -99), expert_counts(td.expert_stride + 8, -99);
    std::vector<int32_t> ref_rank(td.rank_stride, 0), ref_expert(td.expert_stride, 0);
    std::vector<int32_t> scratch(cores * (td.rank_stride + td.expert_stride) + 8, -99);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t j = 0; j < k; ++j) {
            // Explicit signed conversion prevents unsigned -1 oracle corruption.
            int32_t e =
                route == 2 || (t + j) % 11 == 0 ? -1 : int32_t(route == 1 ? (j % 2) : (t * 17 + j * 13) % experts);
            indices[t * k + j] = e;
            if (e >= 0) {
                ++ref_expert[e];
                ref_mask[t * world + uint32_t(e) / (experts / world)] = 1;
            }
        }
        for (uint32_t p = 0; p < world; ++p) ref_rank[p] += ref_mask[t * world + p];
    }
    for (uint32_t core = 0; core < cores; ++core) {
        Resource resource;
        BuildSimtLayoutPartials(resource, Address(indices), Address(mask), Address(scratch), td, core, cores);
        AscendC::CheckDrained();
    }
    // CPU barrier between all producers and reducers; scratch must be fully written.
    for (uint32_t core = 0; core < cores; ++core) {
        Resource resource;
        ReduceSimtLayoutPartials(resource, Address(scratch), Address(ranks), Address(expert_counts), td, core, cores);
        AscendC::CheckDrained();
    }
    assert(std::equal(ref_rank.begin(), ref_rank.end(), ranks.begin()));
    assert(std::equal(ref_expert.begin(), ref_expert.end(), expert_counts.begin()));
    assert(std::equal(ref_mask.begin(), ref_mask.end(), mask.begin()));
    assert(std::all_of(mask.begin() + tokens * world, mask.end(), [](auto x) { return x == 99; }));
    assert(std::all_of(ranks.begin() + td.rank_stride, ranks.end(), [](auto x) { return x == -99; }));
    assert(std::all_of(expert_counts.begin() + td.expert_stride, expert_counts.end(), [](auto x) { return x == -99; }));
    assert(std::all_of(scratch.end() - 8, scratch.end(), [](auto x) { return x == -99; }));
    // Ensure the comparison rejects a deliberately corrupted reference.
    ++ref_rank[0];
    assert(!std::equal(ref_rank.begin(), ref_rank.end(), ranks.begin()));
}

int main()
{
    for (auto world : {4U, 8U, 16U, 32U})
        for (auto cores : {2U, 8U, 16U, 32U, 48U, 64U})
            for (auto tokens : {0U, 1U, 65U, 129U, 4093U, 4096U})
                for (auto route : {0U, 1U, 2U}) CheckCase(world, 256, tokens, 8, cores, route);
    for (auto world : {1U, 3U, 64U, 128U, 256U, 1024U})
        for (auto k : {0U, 1U, 10U, 128U})
            for (auto cores : {8U, 48U}) CheckCase(world, world * 3U, 137, k, cores, 0);
    // Larger expert dimension reduces histogram lanes; larger K forces smaller input tiles.
    CheckCase(8, 16384, 17, 8, 8, 0);
    CheckCase(8, 256, 129, 1024, 8, 0);
    for (auto world : {4U, 32U, 256U, 1024U})
        for (auto experts : {1024U, 8192U, 32768U})
            for (auto k : {0U, 8U, 128U, 1024U}) {
                auto p = MakeSimtLayoutUbPlan(world, experts, k, sizeof(topk_idx_t), platform::kKernelUbBytes);
                if (p.tile) {
                    assert(p.bytes <= platform::kKernelUbBytes);
                    assert(p.histogram_offset % 32 == 0 && p.output_offset % 32 == 0);
                    assert(p.input_bytes >= p.tile * k * sizeof(topk_idx_t));
                    assert(p.mask_bytes >= p.tile * world);
                    assert(p.output_offset >= p.histogram_offset + p.lanes * p.row_elems * 4);
                }
            }
    assert(!MakeSimtLayoutUbPlan(8, 65536, 8, 8, platform::kKernelUbBytes).tile);
    assert(!MakeSimtLayoutUbPlan(0, 256, 8, 8, platform::kKernelUbBytes).tile);
    std::cout << "SIMT layout CPU oracle passed\n";
}
