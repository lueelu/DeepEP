// Copyright (c) 2026, Lu Lu

// Preserve distinct pointer address spaces during a host Clang compile. These
// SDK declarations model the pinned signatures without requiring a CANN compiler.
#define __gm__ __attribute__((address_space(1)))
#define __ubuf__ __attribute__((address_space(3)))
#define __aicore__
using uint8_t = unsigned char;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
namespace AscendC {
template <typename A, typename B>
struct IsSameType {
    static constexpr bool value = false;
};
template <typename A>
struct IsSameType<A, A> {
    static constexpr bool value = true;
};
}  // namespace AscendC
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE };
struct aclshmemx_udma_op_config_t {};
inline constexpr aclshmemx_udma_op_config_t config{};
struct aclshmemi_sqe_ctx_t {
    uint64_t words[6];
};
struct aclshmemi_udma_addr_ctx_t {
    uint64_t heap_base, remote_mr_base;
};
struct aclshmemi_ubmem_info_t {
    uint64_t addr;
};
struct aclshmemx_submit_state_t {
    uint32_t pending_count;
};
struct DeviceState {
    void* heap_base;
};
struct QpInfo {};
struct QpContext {};
__gm__ DeviceState* aclshmemi_get_state();
__gm__ QpInfo* aclshmemi_udma_qp_info_fetch();
__gm__ QpContext* aclshmemi_udma_get_qp_ctx(__gm__ QpInfo*, uint32_t, uint32_t);
uint32_t aclshmemi_udma_compute_slot(uint32_t, uint32_t);
constexpr uint32_t ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE = 64;
template <typename T, aclshmemi_udma_opcode_t Op, typename SqePtr, const aclshmemx_udma_op_config_t& Config>
void aclshmemi_udma_fill_sqe_base_ctx(SqePtr, __gm__ uint8_t*, __gm__ aclshmemi_ubmem_info_t*, uint32_t,
                                      const aclshmemi_udma_addr_ctx_t&);
template <typename T, aclshmemi_udma_opcode_t Op, const aclshmemx_udma_op_config_t& Config>
void aclshmemi_udma_stage_send_wqe(__gm__ uint8_t*, __gm__ uint8_t*, uint32_t, uint32_t, uint64_t, __ubuf__ uint8_t*,
                                   aclshmemx_submit_state_t&);
template <const aclshmemx_udma_op_config_t& Config>
void aclshmemi_udma_require_submit_cqe();
void aclshmemi_udma_flush_aggregate_wqes(__ubuf__ uint8_t*, __gm__ QpContext*, uint32_t, uint32_t, uint32_t, uint32_t,
                                         uint32_t);

#if TEST_ILLEGAL_CAST
__gm__ uint8_t* invalid_cast(void* pointer)
{
    return reinterpret_cast<__gm__ uint8_t*>(pointer);
}
#endif

#include "csrc/kernels/dispatch/udma_live_address.hpp"

void compile_all_adapters(__ubuf__ uint8_t* scratch, __gm__ uint8_t* target, __gm__ uint8_t* source,
                          __gm__ aclshmemi_ubmem_info_t* remote, aclshmemx_submit_state_t& state)
{
    using Op = aclshmemi_udma_opcode_t;
    ascend_deepep::FillLiveUdmaSqeBase<uint8_t, Op::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*, config>(
        reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(scratch), target, remote, 0);
    ascend_deepep::StageLiveUdmaWqe<uint8_t, Op::UDMA_OP_WRITE, config>(target, source, 3, 2, 512, scratch, state);
    ascend_deepep::SubmitLiveUdmaWqes<uint8_t, Op::UDMA_OP_WRITE, config>(target, source, 3, 2, 512, scratch, 7, state);
}
