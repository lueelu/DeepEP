// Copyright (c) 2026, Lu Lu
#pragma once
#include "shmem.h"

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
namespace deepep::comm::tile {
// Fast paths already resolve a remote VA (including exchanged live outputs).
// v1.8.0's SDK helper instead accepts a local symmetric pointer and translates
// it. Build the routing/control words using a valid symmetric placeholder, then
// install the caller's resolved VA in the UB header before any SQ publication.
// This does not register memory or extend the caller's remote-access rights.
template <typename T, aclshmemi_udma_opcode_t Opcode, typename SqePtr,
          const aclshmemx_udma_op_config_t& Config>
ACLSHMEM_DEVICE void FillUdmaRemoteSqe(SqePtr sqe, __gm__ uint8_t* remote,
                                     __gm__ aclshmemi_ubmem_info_t* memory, uint32_t head)
{
    static_assert(Opcode == aclshmemi_udma_opcode_t::UDMA_OP_WRITE);
    static_assert(AscendC::IsSameType<SqePtr, __ubuf__ aclshmemi_sqe_ctx_t*>::value);
    auto* state = aclshmemi_get_state();
    const aclshmemi_udma_addr_ctx_t address{reinterpret_cast<uint64_t>(state->heap_base), memory->addr};
    aclshmemi_udma_fill_sqe_base_ctx<T, Opcode, SqePtr, Config>(
        sqe, reinterpret_cast<__gm__ uint8_t*>(state->heap_base), memory, head, address);
    reinterpret_cast<__ubuf__ uint64_t*>(sqe)[5] = reinterpret_cast<uint64_t>(remote);
}
// Keep UB reuse and SQ doorbells after MTE3 completion. v1.8.0 split
// the old blocking helper into submission and explicit wait operations.
ACLSHMEM_DEVICE void CopyUdmaWqe(__gm__ uint8_t* destination, AscendC::LocalTensor<uint8_t>& source,
                                uint32_t bytes, uint32_t event)
{
    aclshmemi_udma_copy_wqe_from_ub_async(destination, source, bytes, event);
    aclshmemi_udma_wait_wqe_copy_from_ub(event);
}
} // namespace deepep::comm::tile
#endif
