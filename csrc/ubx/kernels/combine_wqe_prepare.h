// Copyright (c) 2026, Lu Lu

#pragma once
#include "shmem.h"
#include "ll_transport.h"

namespace UbxCombine {

// Do not publish a valid SQ owner or update PI before the payload wait: the
// NIC may already be walking the SQ from an earlier doorbell.
class PreparedWrite {
    __gm__ aclshmemi_rdma_sq_ctx* sq_ = nullptr;
    AscendC::LocalTensor<uint32_t> wqe_;
    AscendC::LocalTensor<uint64_t> db_;
    uint32_t head_ = 0U;

public:
    __aicore__ inline bool Prepare(uint32_t peer, GM_ADDR remote, GM_ADDR local, uint32_t bytes,
                                   AscendC::LocalTensor<uint8_t> scratch, deepep::CoreContext core)
    {
        static_assert(ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825,
                      "WQE preparation requires the native HNS-1825 SDK");
        auto* info = aclshmemi_qp_info_fetch();
        const uint64_t index = uint64_t(peer) * info->qp_num + core.index;
        sq_ = reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(info->sq_ptr + index * sizeof(*sq_));
        auto* cq =
            reinterpret_cast<__gm__ aclshmemi_rdma_cq_ctx*>(info->scq_ptr + index * sizeof(aclshmemi_rdma_cq_ctx));
        head_ = aclshmemi_hns_1825_read_u32_gm(sq_->head_addr);
        const uint32_t tail = aclshmemi_hns_1825_read_u32_gm(sq_->tail_addr);
        const uint32_t depth = sq_->depth < cq->depth ? sq_->depth : cq->depth;
        // No pollCQ here: it borrows MTE events while payload MTE3_S(7) is live.
        // Fall back to the public SDK put AFTER the payload wait instead.
        if (head_ - tail + 10U >= depth) return false;
        auto* mr = reinterpret_cast<__gm__ aclshmemi_rdma_mem_info*>(info->mem_ptr);
        aclshmemi_rdma_send_wr wr{};
        wr.local_addr = local;
        wr.remote_addr = aclshmemi_remote_mr_addr(remote, aclshmemi_get_state()->heap_base, mr[peer].addr);
        wr.message_len = bytes;
        wr.rkey = mr[peer].rkey;
        wr.lkey = mr[aclshmemi_get_my_pe()].lkey;
        auto ub32 = scratch.ReinterpretCast<uint32_t>();
        wqe_ = aclshmemi_hns_1825_make_wqe_ub(ub32, 64U);
        db_ = scratch[64U].ReinterpretCast<uint64_t>();
        auto* task = aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(reinterpret_cast<__ubuf__ uint8_t*>(wqe_.GetPhyAddr()),
                                                               head_, sq_->depth, true);
        auto* data = aclshmemi_roce_hns_1825_fill_wqe_task_seg(
            task, wr.rkey, wr.remote_addr, wr.lkey, bytes,
            aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE, true);
        aclshmemi_roce_hns_1825_fill_wqe_data_seg(data, wr.lkey, wr.local_addr, bytes);
        return true;
    }
    __aicore__ inline void Publish()
    {
        // Same one-core/one-QP ownership, next-owner invalidation, SQ copy,
        // cache flush and PI/doorbell helper as the pinned SHMEM implementation.
        auto* dst = aclshmemi_roce_hns_1825_get_send_wqe(sq_, head_ & (sq_->depth - 1U));
        aclshmemi_roce_hns_1825_write_invalid_wqebb(sq_->buf_addr, sq_->depth, head_ + 1U);
        aclshmemi_roce_write_ub_to_gm_with_sync(reinterpret_cast<uint64_t>(dst), wqe_, 64U, 7U);
        dcci_cachelines(dst, 64U);
        aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(sq_, head_ + 1U, db_, wqe_, 7U);
    }
};
}  // namespace UbxCombine
