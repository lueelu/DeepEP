// Copyright (c) 2026, Lu Lu

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

#define __gm__
#define __ubuf__
#define __aicore__
#define ACLSHMEM_DEVICE inline
#if TEST_DEBUG
#define ACLSHMEM_DEBUG_FUNC(fn, ...) fn(__VA_ARGS__)
#else
#define ACLSHMEM_DEBUG_FUNC(fn, ...) ((void)0)
#endif
namespace AscendC {
template <typename A, typename B>
using IsSameType = std::is_same<A, B>;
namespace Std = std;
}  // namespace AscendC

enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE = 1, UDMA_OP_READ = 2, UDMA_OP_WRITE_WITH_REDUCE = 3 };
struct aclshmemx_udma_op_config_t {
    uint32_t odr, fence, se, cqe;
};
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT{0, 0, 0, 1};
inline constexpr aclshmemx_udma_op_config_t kPayloadRoNoCqeConfig{1, 0, 0, 0}, terminal{1, 0, 0, 1};
constexpr uint32_t ACLSHMEM_UDMA_FLAG_ODR_MASK = 7, ACLSHMEM_UDMA_FLAG_FENCE_SHIFT = 3, ACLSHMEM_UDMA_FLAG_SE_SHIFT = 4,
                   ACLSHMEM_UDMA_FLAG_CQE_SHIFT = 5, ACLSHMEM_UDMA_FLAG_UDF = 0x80, ACLSHMEM_UDMA_REMOTE_JETTY_TYPE = 1;
struct aclshmemi_sqe_ctx_t {
    uint64_t words[6];
};
struct aclshmemi_sge_ctx_t {
    uint64_t words[2];
};
struct aclshmemi_udma_addr_ctx_t {
    uint64_t heap_base, remote_mr_base;
};
struct aclshmemi_ubmem_info_t {
    uint32_t token_value_valid, target_hint, tpn, tid, rmt_token_value;
    uint64_t eid_addr, addr;
};
struct aclshmemx_submit_state_t {
    uint32_t pending_count = 0;
};
struct aclshmemi_aiv_udma_info_t {};
struct aclshmemi_udma_wq_ctx_t {};
template <typename T, aclshmemi_udma_opcode_t Op>
struct aclshmemi_udma_params_t {};
template <typename T, aclshmemi_udma_opcode_t Op, typename... Args>
void aclshmemi_fill_notify_data(Args&&...)
{
}
struct DeviceState {
    void* heap_base;
    uint64_t heap_size;
} device{reinterpret_cast<void*>(0x10000000ULL), 0x10000ULL};
DeviceState* aclshmemi_get_state()
{
    return &device;
}
void aclshmemi_kernel_abort(const char*)
{
    assert(false && "SDK address validation failed");
}
void assert_not_self_send(uint32_t peer)
{
    assert(peer == 3);
}
void assert_remote_jetty_type_valid(aclshmemi_ubmem_info_t*) {}
uint64_t eid[]{0x0123456789abcdefULL, 0x9876543210fedcbaULL};
aclshmemi_ubmem_info_t remote{1, 0x37, 0x12345, 0x9876, 0x89abcdef, reinterpret_cast<uint64_t>(eid), 0x20000000ULL};
aclshmemi_aiv_udma_info_t info;
aclshmemi_udma_wq_ctx_t context;
aclshmemi_aiv_udma_info_t* aclshmemi_udma_qp_info_fetch()
{
    return &info;
}
uint32_t aclshmemi_udma_compute_slot(uint32_t peer, uint32_t relay)
{
    assert(peer == 3 && relay == uint32_t(-1));
    return peer;
}
aclshmemi_ubmem_info_t* aclshmemi_udma_get_mem_info(aclshmemi_aiv_udma_info_t*, uint32_t slot, uint32_t qp)
{
    assert(slot == 3 && qp == 2);
    return &remote;
}
aclshmemi_udma_wq_ctx_t* aclshmemi_udma_get_qp_ctx(aclshmemi_aiv_udma_info_t*, uint32_t slot, uint32_t qp)
{
    assert(slot == 3 && qp == 2);
    return &context;
}
constexpr uint32_t ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE = 64, kWqeBatchScratchBytes = 4096;
static_assert(sizeof(aclshmemi_sqe_ctx_t) + sizeof(aclshmemi_sge_ctx_t) == ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE);
uint64_t published[32 * 8]{};
uint32_t flushes = 0, published_count = 0;
void aclshmemi_udma_flush_aggregate_wqes(uint8_t* scratch, aclshmemi_udma_wq_ctx_t* ctx, uint32_t slot, uint32_t qp,
                                         uint32_t sync, uint32_t count, uint32_t stride)
{
    assert(ctx == &context && slot == 3 && qp == 2 && sync == 7 && stride == 64);
    ++flushes;
    published_count = count;
    std::memcpy(published, scratch, count * stride);
}

/* PINNED_SHMEM_BUILDERS */

#include "csrc/kernels/dispatch/udma_live_address.hpp"

using GM_ADDR = uint8_t*;
// The old Combine call mapped to a peer-access VA. This differs from both the
// local symmetric VA and the remote NIC MR, so accidental double conversion fails.
void* aclshmem_ptr(void* address, uint32_t)
{
    return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(address) + 0x40000000ULL);
}
constexpr uint32_t kCrossDescriptorsOffset = 2048;
uint64_t* CrossPutHeader(uint8_t* scratch, uint32_t lane, uint32_t index)
{
    return reinterpret_cast<uint64_t*>(scratch + lane * kWqeBatchScratchBytes) + index * 8;
}
template <bool Enabled>
struct ProductionCombineBatch {
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(0x10000000ULL);
    void BuildLane(uint32_t, uint32_t, uint8_t*) { assert(false); }
    uint32_t direct_rows[2]{}, reduced_rows[2]{}, peer[2]{};
    aclshmemx_submit_state_t state[2]{};
    void FlushLane(uint32_t, uint32_t, uint8_t*)
    {
        assert(false && "unexpected full batch");
    }
    /* PRODUCTION_COMBINE_ADD */
};

int main()
{
    using Op = aclshmemi_udma_opcode_t;
    auto* target = reinterpret_cast<uint8_t*>(0x70000120ULL);
    auto* source = reinterpret_cast<uint8_t*>(0x60000400ULL);
    alignas(64) uint64_t scratch[1024]{};
    auto* bytes = reinterpret_cast<uint8_t*>(scratch);
    aclshmemi_sqe_ctx_t expected{};
    const aclshmemi_udma_addr_ctx_t addresses{reinterpret_cast<uint64_t>(device.heap_base), remote.addr};
    aclshmemi_udma_fill_sqe_base_ctx<uint8_t, Op::UDMA_OP_WRITE, aclshmemi_sqe_ctx_t*, kPayloadRoNoCqeConfig>(
        &expected, reinterpret_cast<uint8_t*>(device.heap_base), &remote, 0, addresses);
    ascend_deepep::FillLiveUdmaSqeBase<uint8_t, Op::UDMA_OP_WRITE, aclshmemi_sqe_ctx_t*, kPayloadRoNoCqeConfig>(
        reinterpret_cast<aclshmemi_sqe_ctx_t*>(scratch), target, &remote, 0);
    for (uint32_t word = 0; word < 5; ++word) assert(scratch[word] == expected.words[word]);
    assert(scratch[5] == reinterpret_cast<uint64_t>(target));

    aclshmemx_submit_state_t state{};
    for (uint32_t row = 0; row < 2; ++row) {
        ascend_deepep::StageLiveUdmaWqe<uint8_t, Op::UDMA_OP_WRITE, kPayloadRoNoCqeConfig>(
            target + row * 512U, source + row * 512U, 3, 2, 512, bytes, state);
        assert(state.pending_count == row + 1 && flushes == 0);
    }
    ascend_deepep::SubmitLiveUdmaWqes<uint8_t, Op::UDMA_OP_WRITE, terminal>(target + 1024, source + 1024, 3, 2, 512,
                                                                            bytes, 7, state);
    assert(flushes == 1 && published_count == 3 && state.pending_count == 0);
    for (uint32_t row = 0; row < 3; ++row) {
        auto* wqe = published + row * 8;
        assert(wqe[5] == reinterpret_cast<uint64_t>(target) + row * 512);
        assert(wqe[6] == 512 && wqe[7] == reinterpret_cast<uint64_t>(source) + row * 512);
        assert(((wqe[0] >> (16 + ACLSHMEM_UDMA_FLAG_CQE_SHIFT)) & 1) == uint64_t(row == 2));
        for (uint32_t word = 1; word < 5; ++word) assert(wqe[word] == expected.words[word]);
    }

    ProductionCombineBatch<false> combine;
    CrossPutHeader(bytes, 0, 0)[7] = 0x20000000ULL;
    combine.Add(reinterpret_cast<uint8_t*>(0x10000080ULL), source, 3, 0, 1, 512, bytes, false, 0);
    assert(combine.state[0].pending_count == 1);
    auto* descriptor = reinterpret_cast<uint64_t*>(bytes + kCrossDescriptorsOffset);
    assert(descriptor[1] == 0x20000080ULL);  // Exactly one symmetric -> remote MR conversion.
    assert(descriptor[0] == reinterpret_cast<uint64_t>(source));
}
