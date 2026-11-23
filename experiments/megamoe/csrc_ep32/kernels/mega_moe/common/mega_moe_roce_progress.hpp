// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_ROCE_PROGRESS_H
#define MEGA_MOE_ROCE_PROGRESS_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"

namespace MegaMoeImpl {

constexpr uint32_t MEGAMOE_ROCE_PROGRESS_MAX_CQES = 32U;

// Only the selected QP's producer may call this, using its private 256B UB
// workspace and event. Reuse the pinned SHMEM HNS owner and doorbell helpers so
// both the CQ consumer index and the SQ capacity mirror advance together.
// This is completion/error progress, not a remote visibility or source-lifetime
// barrier: an unfinished CQE ends this attempt immediately. The SDK reports the last completed WQE in each CQE, independently of CQ depth.
__aicore__ inline uint32_t RoceProgressReady(uint32_t peer, uint32_t qpIndex, __ubuf__ uint8_t* scratch,
                                             uint32_t syncId)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825) && (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__))
    constexpr uint32_t CQE_OPCODE_SHIFT = 27U;
    constexpr uint32_t CQE_OPCODE_MASK = 0x1fU;
    constexpr uint32_t CQE_ERROR = 0x1eU;
    constexpr uint32_t CQE_INVALID = 0x1fU;
    constexpr uint32_t CQE_DEFAULT_STRIDE = 64U;
    constexpr uint32_t WORKSPACE_SLICE_BYTES = 64U;
    static_assert(sizeof(aclshmemi_hns_1825_cqe_t) <= WORKSPACE_SLICE_BYTES);

    __gm__ aclshmemi_rdma_info* info = aclshmemi_qp_info_fetch();
    const uint64_t slot = static_cast<uint64_t>(peer) * info->qp_num + qpIndex;
    __gm__ aclshmemi_rdma_cq_ctx* cq =
        reinterpret_cast<__gm__ aclshmemi_rdma_cq_ctx*>(info->scq_ptr + slot * sizeof(aclshmemi_rdma_cq_ctx));
    __gm__ aclshmemi_rdma_sq_ctx* sq =
        reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(info->sq_ptr + slot * sizeof(aclshmemi_rdma_sq_ctx));
    const uint32_t targetHead = aclshmemi_hns_1825_read_u32_gm(sq->head_addr);
    const uint32_t initialSqTail = aclshmemi_hns_1825_read_u32_gm(sq->tail_addr);
    uint32_t completedSqTail = initialSqTail;
    uint32_t tail = aclshmemi_hns_1825_read_u32_gm(cq->tail_addr);
    const uint32_t stride = cq->cqe_size == 0U ? CQE_DEFAULT_STRIDE : cq->cqe_size;

    AscendC::LocalTensor<uint32_t> ub32;
    ub32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub32.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch);
    ub32.address_.dataLen = WORKSPACE_SLICE_BYTES;
    AscendC::LocalTensor<uint64_t> ub64;
    ub64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub64.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch) + WORKSPACE_SLICE_BYTES;
    ub64.address_.dataLen = WORKSPACE_SLICE_BYTES;
    __ubuf__ aclshmemi_hns_1825_cqe_t* cqe = reinterpret_cast<__ubuf__ aclshmemi_hns_1825_cqe_t*>(scratch);
    uint32_t reclaimed = 0U;
    bool failed = false;
    while (completedSqTail != targetHead && reclaimed < MEGAMOE_ROCE_PROGRESS_MAX_CQES) {
        __gm__ uint8_t* address =
            reinterpret_cast<__gm__ uint8_t*>(cq->buf_addr + static_cast<uint64_t>(tail & (cq->depth - 1U)) * stride);
        // Refresh before the single read; unlike blocking SDK poll, this call
        // does not retry the same CQE after discovering that it is not ready.
        dcci_cachelines(address, sizeof(aclshmemi_hns_1825_cqe_t));
        AscendC::GlobalTensor<uint32_t> cqeGm;
        cqeGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
        AscendC::DataCopyExtParams copy{1U, sizeof(aclshmemi_hns_1825_cqe_t), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<uint32_t> pad{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(ub32, cqeGm, copy, pad);
        aclshmemi_hns_1825_sync_func<AscendC::HardEvent::MTE2_S>(syncId);
        const uint32_t opcode = (cqe->op_sr_wqebb >> CQE_OPCODE_SHIFT) & CQE_OPCODE_MASK;
        if (opcode == CQE_INVALID || !aclshmemi_roce_hns_1825_check_cqe_owner(cqe, tail, cq->depth)) {
            break;
        }
        const auto previous = static_cast<uint16_t>(completedSqTail - 1U);
        completedSqTail += static_cast<uint16_t>(cqe->wqe_counter - previous);
        ++tail;
        ++reclaimed;
        if (opcode == CQE_ERROR) {
            // Dump before the doorbell helper reuses UB for its CI payload.
            aclshmemi_roce_hns_1825_dump_cqe(cqe);
            failed = true;
            break;
        }
    }
    if (reclaimed != 0U) {
        aclshmemi_roce_ring_cq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(peer, qpIndex, tail, ub64, ub32, syncId);
    }
    if (completedSqTail != initialSqTail) {
        aclshmemi_hns_1825_write_u32_gm(sq->tail_addr, completedSqTail);
    }
    if (failed) trap();
    return reclaimed;
#else
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    (void)syncId;
    return 0U;
#endif
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_ROCE_PROGRESS_H
