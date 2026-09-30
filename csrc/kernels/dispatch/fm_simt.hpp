// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

// Included after the secondary readiness helpers. Private peer QP0 only.
// Reuse the 8KiB WQE scratch: header [0,64), WQEs [256,4352).
// Metadata UB: up to 128 row pairs [0,1024), indices [2048,2304),
// world+1 prefix words [4096,5124) at EP256; no new TPipe UB.
constexpr uint32_t kFmSimtMaxWqes = 64U;
constexpr uint32_t kFmSimtWqeOffset = 256U;
constexpr uint32_t kFmSimtIndicesOffset = 2048U;
static_assert(kFmSimtWqeOffset + kFmSimtMaxWqes * 64U <= kUdmaWqeScratchBytes,
              "FM SIMT must fit the existing WQE scratch");
static_assert(kFmSimtIndicesOffset + kFmSimtMaxWqes * 4U <= kSlotSelectUbBytes,
              "FM SIMT metadata must fit the existing slots");
static_assert(128U * 2U * sizeof(uint32_t) <= kFmSimtIndicesOffset &&
                  kFmSimtIndicesOffset + kFmSimtMaxWqes * sizeof(uint32_t) <= 4096U,
              "FM row pairs, indices and prefixes must not overlap");

// Each pair is (received hidden row, final peer row), NOT (input token, row).
// No per-WQE branch. Only the last RO requests a CQE, as in the legacy FM path.
__simt_vf__ __launch_bounds__(64) inline void DispatchBuildFmWqesVf(__ubuf__ uint64_t* wqes, __ubuf__ uint64_t* header,
                                                                    __ubuf__ int32_t* rows, __ubuf__ uint32_t* indices,
                                                                    uint32_t count, uint32_t row_bytes, uint64_t source,
                                                                    uint64_t target, uint64_t cqe_mask)
{
    for (uint32_t i = threadIdx.x; i < count; i += 64U) {
        auto* sqe = wqes + uint64_t(i) * 8U;
        const uint32_t row = indices[i];
        sqe[0] = header[0] | (uint64_t(i + 1U == count) * cqe_mask);
        sqe[1] = header[1];
        sqe[2] = header[2];
        sqe[3] = header[3];
        sqe[4] = header[4];
        sqe[5] = target + uint64_t(rows[row + 1U]) * row_bytes;
        sqe[6] = uint64_t(row_bytes);
        sqe[7] = source + uint64_t(rows[row]) * row_bytes;
    }
}

// One batch => one DB, including SQ wrap. Preserve the legacy terminal-RO CQE
// accounting. No payload quiet here; only WQE copies finish before ringing DB.
__aicore__ inline void PublishFmSimtWqes(__ubuf__ uint8_t* wqes, uint32_t count, __gm__ aclshmemi_udma_wq_ctx_t* ctx,
                                         __gm__ aclshmemi_udma_queue_state_t* state)
{
    const uint32_t head = state->sq_head, index = head % ctx->depth;
    const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
    AscendC::LocalTensor<uint8_t> view;
    view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes);
    aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * ctx->wqe_size),
                                    view, first * ctx->wqe_size, EVENT_ID0);
    if (first != count) {
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes + first * ctx->wqe_size);
        aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view,
                                        (count - first) * ctx->wqe_size, EVENT_ID0);
    }
    aclshmemi_udma_post_send_update_info(head + count, ctx, state);
    ++state->cqe_cnt;
}

// Consume only row pairs whose source-ready has already been acquired.
__aicore__ inline void IssueFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch,
                                   LocalTensor<uint32_t> rows, LocalTensor<uint32_t> indices, uint32_t first,
                                   uint32_t count)
{
    if (!count || count > kFmSimtMaxWqes) {
        aclshmemi_kernel_abort("FM SIMT invalid batch size");
        trap();
        return;
    }
    EnsureDeepepPeerAddress(s, peer);
    auto* info = aclshmemi_udma_qp_info_fetch();
    const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
    auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
    if (!ctx || !ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT || ctx->wqe_size != 64U) {
        aclshmemi_kernel_abort("FM SIMT SQ ABI mismatch");
        trap();
        return;
    }
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
    aclshmemi_udma_poll_before_aggregate_flush(state, slot, 0U, count);
    if (!aclshmemi_udma_sq_has_capacity(state, count)) {
        aclshmemi_kernel_abort("FM SIMT SQ capacity exhausted");
        trap();
        return;
    }
    for (uint32_t i = 0; i < count; ++i) indices.SetValue(i, (first + i) * 2U);
    // Descriptor contains an absolute payload VA; no heap translation here.
    auto* target = PeerHiddenAddress(s.book, peer, s.book_words);
    if (!target) {
        aclshmemi_kernel_abort("FM SIMT missing live output\n");
        trap();
        return;
    }
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kFmSimtWqeOffset);
    aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                     kUdmaDataRoNoCqeConfig>(reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header),
                                                             target, aclshmemi_udma_get_mem_info(info, slot, 0U), 0U);
    constexpr uint32_t ro = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoNoCqeConfig>();
    constexpr uint32_t cqe = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoCqeConfig>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    asc_vf_call<DispatchBuildFmWqesVf>(dim3(64), wqes, header, reinterpret_cast<__ubuf__ int32_t*>(rows.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr()), count, s.row_bytes,
                                       reinterpret_cast<uint64_t>(s.hidden), reinterpret_cast<uint64_t>(target),
                                       uint64_t(ro ^ cqe) << 16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    PublishFmSimtWqes(reinterpret_cast<__ubuf__ uint8_t*>(wqes), count, ctx, state);
}

// One terminal SO on the same private QP0 orders all forwarded RO writes.
// Use an otherwise unused tail of the SHMEM sync area, NOT a framework VA
// and NOT a primary source-ready flag. The caller drains this QP before the
// existing all-sender FinishPeerDispatch completion exchange.
constexpr uint32_t kFmFenceOffset = 57344U;
static_assert(kFmFenceOffset + 8U * 512U <= 65536U);
__aicore__ inline void FinishFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch)
{
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(s.sync + uint64_t(s.world) * kRankSyncSlotBytes + kFmFenceOffset +
                                                    uint64_t(s.rank % 8U) * 512U);
    auto* value = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), kPeerCompletionValueSlot, s.sync);
    aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
        flag, value, reinterpret_cast<__ubuf__ uint64_t*>(scratch), 1U, peer, 0U, EVENT_ID0);
}
