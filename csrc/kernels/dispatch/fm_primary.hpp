// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Each pair is (source row, final peer row). The caller selects input or received GM.
// No per-WQE branch or CQE selection; the header fixes every data WQE to RO/no-CQE.
__simt_vf__ __launch_bounds__(32) inline void DispatchBuildFmWqesVf(__ubuf__ uint64_t* wqes, __ubuf__ uint64_t* header,
                                                                    __ubuf__ int32_t* rows, uint32_t first,
                                                                    uint32_t count, uint32_t row_bytes, uint64_t source,
                                                                    uint64_t target)
{
    for (uint32_t i = threadIdx.x; i < count; i += 32U) {
        auto* sqe = wqes + uint64_t(i) * 8U;
        const uint32_t row = (first + i) * 2U;
        sqe[0] = header[0];
        sqe[1] = header[1];
        sqe[2] = header[2];
        sqe[3] = header[3];
        sqe[4] = header[4];
        sqe[5] = target + uint64_t(rows[row + 1U]) * row_bytes;
        sqe[6] = uint64_t(row_bytes);
        sqe[7] = source + uint64_t(rows[row]) * row_bytes;
    }
}

__aicore__ inline void InitFmPrimaryContext(DeepepPrimaryState& p, uint32_t peer, __ubuf__ uint8_t* scratch)
{
    auto& s = *p.secondary;
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    if (!p.fm_context) {
        auto* info = aclshmemi_udma_qp_info_fetch();
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
        if (!ctx || !ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT ||
            ctx->wqe_size != 64U) {
            aclshmemi_kernel_abort("FM primary SQ ABI mismatch");
            trap();
            return;
        }
        auto* target = PeerHiddenAddress(s.book, peer);
        if (!target) {
            aclshmemi_kernel_abort("FM primary missing live output");
            trap();
            return;
        }
        deepep::comm::tile::FillUdmaRemoteSqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), target,
            aclshmemi_udma_get_mem_info(info, slot, 0U), 0U);
        header[5] = reinterpret_cast<uint64_t>(target);
        p.fm_context = reinterpret_cast<uint64_t>(ctx);
        p.fm_queue = ctx->state_addr;
        p.fm_slot = slot;
    }
}

// Contiguous (input token, final output row) ranges prepared alongside Notify.
// Each metadata tile is read once; each VF builds 32 RO/no-CQE descriptors.
__aicore__ inline void IssueFmPrimaryPlanRange(DeepepPrimaryState& p, uint32_t peer, uint32_t begin, uint32_t count,
                                               __ubuf__ uint8_t* scratch, LocalTensor<uint32_t> records)
{
    if (!count) return;
    auto& s = *p.secondary;
    InitFmPrimaryContext(p, peer, scratch);
    auto* ctx = reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(p.fm_context);
    auto* queue = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(p.fm_queue);
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(scratch + 256U);
    const bool admitted = count <= kFmMaxAdmittedRows;
    if (admitted) AdmitFmRows(s, peer, scratch, queue, count);
    for (uint32_t at = 0U; at < count; at += 128U) {
        const uint32_t tile = count - at < 128U ? count - at : 128U;
        NativeLoad(records, reinterpret_cast<__gm__ uint32_t*>(p.clos_peer_rows) + uint64_t(begin + at) * 2U,
                   tile * 2U);
        if (!admitted) AdmitFmRows(s, peer, scratch, queue, tile);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t first = 0U; first < tile; first += 32U) {
            const uint32_t n = tile - first < 32U ? tile - first : 32U;
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            asc_vf_call<DispatchBuildFmWqesVf>(dim3(32), wqes, header,
                                               reinterpret_cast<__ubuf__ int32_t*>(records.GetPhyAddr()), first, n,
                                               s.row_bytes, reinterpret_cast<uint64_t>(p.input), header[5]);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            PublishFmPreparedRows(reinterpret_cast<__ubuf__ uint8_t*>(wqes), n, ctx, queue);
        }
    }
}
