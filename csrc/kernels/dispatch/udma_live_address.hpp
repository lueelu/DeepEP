// Copyright (c) 2026, Lu Lu

#pragma once

// Match shmem.h: Bisheng's Host pass does not expose Device UDMA types.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
namespace ascend_deepep {
// Dispatch payloads use peer-published framework addresses, outside the symmetric heap.
// SHMEM 87b63bd's internal builders now translate symmetric addresses. Build the
// endpoint header from a valid heap address, then replace only the destination
// word before publishing the WQE. Never translate a live peer VA as a heap offset.
template <typename T, aclshmemi_udma_opcode_t Op, typename SqePtr, const aclshmemx_udma_op_config_t& Config>
__aicore__ inline void FillLiveUdmaSqeBase(SqePtr sqe, __gm__ uint8_t* target, __gm__ aclshmemi_ubmem_info_t* remote,
                                           uint32_t head)
{
    static_assert(Op == aclshmemi_udma_opcode_t::UDMA_OP_WRITE);
    static_assert(AscendC::IsSameType<SqePtr, __ubuf__ aclshmemi_sqe_ctx_t*>::value);
    const uint64_t heap_address = reinterpret_cast<uint64_t>(aclshmemi_get_state()->heap_base);
    auto* heap = reinterpret_cast<__gm__ uint8_t*>(heap_address);
    const aclshmemi_udma_addr_ctx_t addresses{heap_address, remote->addr};
    aclshmemi_udma_fill_sqe_base_ctx<T, Op, SqePtr, Config>(sqe, heap, remote, head, addresses);
    reinterpret_cast<__ubuf__ uint64_t*>(sqe)[5] = reinterpret_cast<uint64_t>(target);
}

template <typename T, aclshmemi_udma_opcode_t Op, const aclshmemx_udma_op_config_t& Config>
__aicore__ inline void StageLiveUdmaWqe(__gm__ uint8_t* target, __gm__ uint8_t* source, uint32_t peer, uint32_t qp,
                                        uint64_t bytes, __ubuf__ uint8_t* scratch, aclshmemx_submit_state_t& state)
{
    static_assert(Op == aclshmemi_udma_opcode_t::UDMA_OP_WRITE);
    const uint32_t pending = state.pending_count;
    const uint64_t heap_address = reinterpret_cast<uint64_t>(aclshmemi_get_state()->heap_base);
    auto* heap = reinterpret_cast<__gm__ uint8_t*>(heap_address);
    aclshmemi_udma_stage_send_wqe<T, Op, Config>(heap, source, peer, qp, bytes, scratch, state);
    auto* words = reinterpret_cast<__ubuf__ uint64_t*>(scratch + pending * ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE);
    words[5] = reinterpret_cast<uint64_t>(target);
}

template <typename T, aclshmemi_udma_opcode_t Op, const aclshmemx_udma_op_config_t& Config>
__aicore__ inline void SubmitLiveUdmaWqes(__gm__ uint8_t* target, __gm__ uint8_t* source, uint32_t peer, uint32_t qp,
                                          uint64_t bytes, __ubuf__ uint8_t* scratch, uint32_t sync,
                                          aclshmemx_submit_state_t& state)
{
    aclshmemi_udma_require_submit_cqe<Config>();
    StageLiveUdmaWqe<T, Op, Config>(target, source, peer, qp, bytes, scratch, state);
    const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
    auto* context = aclshmemi_udma_get_qp_ctx(aclshmemi_udma_qp_info_fetch(), slot, qp);
    aclshmemi_udma_flush_aggregate_wqes(scratch, context, slot, qp, sync, state.pending_count,
                                        ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE);
    state.pending_count = 0;
}
}  // namespace ascend_deepep
#endif
