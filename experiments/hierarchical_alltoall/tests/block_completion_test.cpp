// Exercise the actual public block's submission/completion choices with a fake SDK.
// Device annotations/resources are replaced only in this CPU test translation unit.
#define __aicore__
#define __forceinline__ inline
#define __gm__
#define __ubuf__
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

namespace AscendC {
enum class HardEvent { MTE3_MTE2 };
template <HardEvent> void SetFlag(uint32_t) {}
template <HardEvent> void WaitFlag(uint32_t) {}
template <class T> struct GlobalTensor {
    T* ptr = nullptr;
    void SetGlobalBuffer(T* value) { ptr = value; }
    T* GetPhyAddr() const { return ptr; }
    GlobalTensor operator[](uint64_t offset) const { return {ptr + offset}; }
};
template <class T> struct LocalTensor {
    T* ptr;
    T* GetPhyAddr() const { return ptr; }
};
}
namespace Catlass::Arch {
struct CpuArch { static constexpr uint32_t UB_SIZE = 65536; };
template <class Arch> struct Resource {
    struct Ub {
        std::array<uint8_t, Arch::UB_SIZE> bytes{};
        template <class T> AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset)
        { return {reinterpret_cast<T*>(bytes.data() + offset)}; }
    } ubBuf;
};
}
struct Trace {
    std::map<std::pair<int, uint32_t>, std::vector<std::function<void()>>> pending;
    uint32_t puts = 0, quiets = 0;
} trace;
template <class T>
void aclshmemx_roce_qp_put_nbi(T* dst, T* src, T*, uint32_t count, int peer, uint32_t qp, uint32_t)
{
    ++trace.puts;
    trace.pending[{peer, qp}].push_back([=] { std::memcpy(dst, src, count * sizeof(T)); });
}
template <class T>
void aclshmemx_udma_qp_put_nbi(T* dst, T* src, T* scratch, uint32_t count, int peer, uint32_t qp, uint32_t event)
{
    aclshmemx_roce_qp_put_nbi(dst, src, scratch, count, peer, qp, event);
}
template <class T> void aclshmemx_roce_qp_quiet(int peer, uint32_t qp, T*, uint32_t)
{
    ++trace.quiets;
    for (auto& finish : trace.pending[{peer, qp}]) finish();
    trace.pending[{peer, qp}].clear();
}
#include "comm/block/comm_block_hierarchical_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
struct Tensor {
    using Element = uint8_t;
    using Layout = Catlass::layout::RowMajor;
};
struct NoMte {
    template <class... Args> void operator()(Args&&...) { assert(false && "network-only test"); }
};
using Block = deepep::comm::block::HierarchicalCopy<Catlass::Arch::CpuArch, Tensor, Tensor, NoMte,
    Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle, 2>;
using Transport = deepep::comm::CopyTransport;
using Layout = Catlass::layout::RowMajor;
void Check(uint32_t qps)
{
    trace = {};
    Catlass::Arch::Resource<Catlass::Arch::CpuArch> resource;
    Block block(resource, Block::Params({1U, 256U}));
    std::array<uint8_t, 256> input{}, output{};
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = i + 1;
    AscendC::GlobalTensor<uint8_t> src{input.data()}, dst{output.data()};
    Layout layout{1U, 256U};
    block(src, layout, dst, layout, {1U, 256U}, 1, Transport::Rdma, qps);
    assert(input == output && trace.quiets == qps); // Default is Wait.

    trace = {};
    output.fill(0);
    std::array<uint8_t, 7> scale{1, 2, 3, 4, 5, 6, 7}, scaleOut{};
    Block scaleBlock(resource, Block::Params({1U, 7U}));
    block(src, layout, dst, layout, {1U, 256U}, 1, Transport::Rdma, qps, Block::RdmaCompletion::Deferred);
    scaleBlock({scale.data()}, {1U, 7U}, {scaleOut.data()}, {1U, 7U}, {1U, 7U}, 1,
               Transport::Rdma, qps, Block::RdmaCompletion::Deferred);
    assert(trace.quiets == 0 && input != output && scale != scaleOut);
    block.QuietRdma(1, qps); // Drains both block objects sharing the peer/QPs.
    assert(trace.quiets == qps && input == output && scale == scaleOut);

    trace = {};
    output.fill(0);
    block(src, layout, dst, layout, {1U, 256U}, 1, Transport::Udma, qps);
    assert(trace.quiets == 0 && input != output); // No accidental UDMA completion shortcut.
    trace = {}; // Discard model requests before their source buffers leave scope.

    // Padded rows: all row submissions complete before the default Wait returns.
    output.fill(0xa5);
    Layout padded{2U, 48U, 64U};
    block(src, padded, dst, padded, {2U, 48U}, 1, Transport::Rdma, qps);
    assert(trace.quiets == qps);
    for (uint32_t row = 0; row < 2; ++row)
        for (uint32_t col = 0; col < 64; ++col)
            assert(output[row * 64 + col] == (col < 48 ? input[row * 64 + col] : 0xa5));
}
int main()
{
    for (uint32_t qps : {1, 2, 4}) Check(qps);
    std::cout << "PASS actual block Wait/Deferred/QuietRdma with QP1/2/4 (fake SDK)\n";
}
