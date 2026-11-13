// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Private FM QP0: data RO never requests a CQE. Admission happens outside
// descriptor construction/publication loops; only SO checkpoints retire SQ.
constexpr uint32_t kFmControlScratchOffset = 4096U;
constexpr uint32_t kFmFenceOffset = 57344U;
constexpr uint32_t kFmControlReserve = 2U;  // source-ready SO and final/checkpoint SO
constexpr uint32_t kFmMaxAdmittedRows =
    shm::UDMA_SQ_BASKBLK_CNT - ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD - kFmControlReserve - 2U;
static_assert(kFmControlScratchOffset + 64U <= kUdmaWqeScratchBytes);
static_assert(kFmFenceOffset + 8U * 512U <= 65536U);
static_assert(get_wqe_bb_cnt<aclshmemi_udma_opcode_t::UDMA_OP_WRITE>() == 1U);
inline constexpr aclshmemx_udma_op_config_t kFmReadySoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};

__aicore__ inline void PublishFmSimtWqes(__ubuf__ uint8_t* wqes, uint32_t count, __gm__ aclshmemi_udma_wq_ctx_t* ctx,
                                         __gm__ aclshmemi_udma_queue_state_t* state)
{
    const uint32_t head = state->sq_head, index = head % ctx->depth;
    const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
    AscendC::LocalTensor<uint8_t> view;
    view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes);
    deepep::comm::tile::CopyUdmaWqe(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * ctx->wqe_size),
                                    view, first * ctx->wqe_size, EVENT_ID0);
    if (first != count) {
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes + first * ctx->wqe_size);
        deepep::comm::tile::CopyUdmaWqe(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view,
                                        (count - first) * ctx->wqe_size, EVENT_ID0);
    }
    aclshmemi_udma_post_send_update_info(head + count, ctx, state);
}

// Land the entire admitted builder packet before
// ringing the same four-WQE doorbells. SQ slots past each published head
// remain invisible to the NIC. A wrap needs at most two bounded copies;
// neither the copy nor the DB loop changes RO/CQE or queue ownership.
__aicore__ inline void PublishFmPreparedRows(__ubuf__ uint8_t* wqes, uint32_t count,
                                             __gm__ aclshmemi_udma_wq_ctx_t* ctx,
                                             __gm__ aclshmemi_udma_queue_state_t* state)
{
    const uint32_t head = state->sq_head, index = head % ctx->depth;
    const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
    AscendC::LocalTensor<uint8_t> view;
    view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes);
    deepep::comm::tile::CopyUdmaWqe(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * ctx->wqe_size),
                                    view, first * ctx->wqe_size, EVENT_ID0);
    if (first != count) {
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes + first * ctx->wqe_size);
        deepep::comm::tile::CopyUdmaWqe(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view,
                                        (count - first) * ctx->wqe_size, EVENT_ID0);
    }
    const uint32_t full = count / 4U;
    for (uint32_t batch = 0U; batch < full; ++batch)
        aclshmemi_udma_post_send_update_info(head + (batch + 1U) * 4U, ctx, state);
    if (count % 4U) aclshmemi_udma_post_send_update_info(head + count, ctx, state);
}

// Separate scratch preserves the cached RO header and any prebuilt descriptors.
// The fence is SHMEM storage, not an ordinary output VA or a source-ready flag.
__aicore__ inline void FinishFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch)
{
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(s.sync + uint64_t(s.world) * kRankSyncSlotBytes + kFmFenceOffset +
                                                    uint64_t(s.rank % 8U) * 512U);
    auto* value = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), kPeerCompletionValueSlot, s.sync);
    aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
        flag, value, reinterpret_cast<__ubuf__ uint64_t*>(scratch + kFmControlScratchOffset), 1U, peer, 0U, EVENT_ID0);
}

__aicore__ inline void AdmitFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch,
                                   __gm__ aclshmemi_udma_queue_state_t* state, uint32_t count)
{
    if (count > kFmMaxAdmittedRows) {
        aclshmemi_kernel_abort("FM admission exceeds SQ capacity");
        trap();
        return;
    }
    if (!aclshmemi_udma_sq_has_credit(state, count + kFmControlReserve)) {
        // RO has no CQEs to poll. Use the reserved slot to create a completion
        // before recycling descriptors; polling the old CQ count cannot free SQ.
        ++s.fm_checkpoints;
        FinishFmRows(s, peer, scratch);
        aclshmemx_udma_qp_quiet(peer, 0U);
    }
    if (!aclshmemi_udma_sq_has_credit(state, count + kFmControlReserve)) {
        aclshmemi_kernel_abort("FM SQ capacity exhausted after SO completion");
        trap();
    }
}

// Remote readers still need the primary-data boundary, but the sender does
// not need its CQE. The terminal SO later covers both this SO and forwarding.
__aicore__ inline void PublishFmSourceReady(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch)
{
    auto* info = aclshmemi_udma_qp_info_fetch();
    const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
    auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
    AdmitFmRows(s, peer, scratch, state, 0U);
    auto* wqe = scratch + kFmControlScratchOffset;
    auto* flag = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), 0U, s.sync);
    auto* value = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), kPeerCompletionValueSlot, s.sync);
    auto* target = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(flag, peer));
    // Immediate SDK put requires CQE=1, so build the unsignaled SO explicitly.
    deepep::comm::tile::FillUdmaRemoteSqe<uint64_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                     kFmReadySoNoCqeConfig>(reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(wqe),
                                                            target, aclshmemi_udma_get_mem_info(info, slot, 0U),
                                                            state->sq_head);
    aclshmemi_fill_sge_header_ctx(reinterpret_cast<__ubuf__ aclshmemi_sge_ctx_t*>(wqe + sizeof(aclshmemi_sqe_ctx_t)),
                                  sizeof(uint64_t), reinterpret_cast<uint64_t>(value));
    PublishFmSimtWqes(wqe, 1U, ctx, state);
}
