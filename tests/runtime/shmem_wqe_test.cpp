// CPU check of the production adapter and extracted v1.8.0 SQE builder.
// Device types/copy operations are doubles, not an AscendC or NPU test.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#define __CCE_KT_TEST__
#define __gm__
#define __ubuf__
#define ACLSHMEM_DEVICE inline
#define ACLSHMEM_DEBUG_FUNC(fn, ...) fn(__VA_ARGS__)
namespace AscendC {
namespace Std { using std::conditional; }
template <class A, class B> using IsSameType = std::is_same<A, B>;
template <class T> struct LocalTensor { T* ptr; };
}
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE, UDMA_OP_WRITE_WITH_REDUCE };
struct aclshmemx_udma_op_config_t { uint32_t flag; };
constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT{0};
constexpr aclshmemx_udma_op_config_t test_config{7};
constexpr uint32_t ACLSHMEM_UDMA_REMOTE_JETTY_TYPE = 1;
struct aclshmemi_sqe_ctx_t { uint64_t words[8]; };
struct aclshmemi_ubmem_info_t {
    uint64_t addr, eid_addr;
    uint32_t token_value_valid, target_hint, tpn, tid, rmt_token_value;
};
struct aclshmemi_udma_addr_ctx_t { uint64_t heap_base, remote_mr_base; };
std::array<uint8_t, 256> heap;
struct State { void* heap_base = heap.data(); uint64_t heap_size = heap.size(); } state;
State* aclshmemi_get_state() { return &state; }
void aclshmemi_kernel_abort(const char*) { assert(false); }
void assert_remote_jetty_type_valid(aclshmemi_ubmem_info_t*) {}
template <aclshmemi_udma_opcode_t, const aclshmemx_udma_op_config_t& C>
constexpr uint32_t aclshmemi_udma_build_flag() { return C.flag; }
template <class T, aclshmemi_udma_opcode_t> constexpr uint32_t aclshmemi_udma_get_reduce_attr() { return 0; }
// The test runner extracts these exact function bodies from the pinned SDK.
#include "sdk_sqe.inc"
uint8_t* pending_dst;
uint8_t* pending_src;
uint32_t pending_bytes;
bool pending = false;
void aclshmemi_udma_copy_wqe_from_ub_async(uint8_t* dst, AscendC::LocalTensor<uint8_t>& src,
                                         uint32_t bytes, uint32_t event)
{
    assert(!pending && event == 3);
    pending = true; pending_dst = dst; pending_src = src.ptr; pending_bytes = bytes;
}
void aclshmemi_udma_wait_wqe_copy_from_ub(uint32_t event)
{
    assert(pending && event == 3);
    std::memcpy(pending_dst, pending_src, pending_bytes);
    pending = false;
}
#include "comm/tile/tile_udma_wqe.hpp"
int main()
{
    uint64_t eid[2]{11, 22};
    // Local symmetric heap, remote MR, and exchanged live output VAs differ.
    aclshmemi_ubmem_info_t memory{0x800000000ULL, reinterpret_cast<uint64_t>(eid), 1, 2, 3, 4, 5};
    for (uint64_t address : {0x800000020ULL, 0x900000088ULL}) {
        aclshmemi_sqe_ctx_t sqe{};
        deepep::comm::tile::FillUdmaRemoteSqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                                             aclshmemi_sqe_ctx_t*, test_config>(
            &sqe, reinterpret_cast<uint8_t*>(address), &memory, 0);
        assert(sqe.words[5] == address); // Never translate an already resolved remote VA again.
        assert(sqe.words[2] == 11 && sqe.words[3] == 22 && sqe.words[4] == 5);
        assert(sqe.words[0] == ((uint64_t(2) << 32) | (1U << 29) | (1U << 28) | (7U << 16)));
    }
    std::array<uint8_t, 16> src{}, dst{};
    src.fill(37);
    AscendC::LocalTensor<uint8_t> tensor{src.data()};
    deepep::comm::tile::CopyUdmaWqe(dst.data(), tensor, src.size(), 3);
    assert(!pending && src == dst); // The caller can now ring the SQ doorbell/reuse UB.
}
