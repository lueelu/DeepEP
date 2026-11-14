// Copyright (c) 2026, Lu Lu

#pragma once

// Private FM QP0. Use the existing 8 KiB scratch: header [0,64),
// 32 prebuilt WQEs [256,2304). No payload staging or extra workspace.
constexpr uint32_t kFmBuildRows = 32U;
constexpr uint32_t kFmSimtWqeOffset = 256U;
static_assert(kFmSimtWqeOffset + kFmBuildRows * 64U <= kUdmaWqeScratchBytes);

// Consume only row pairs whose source-ready has already been acquired.
__aicore__ inline void IssueFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch,
                                   LocalTensor<uint32_t> rows, uint32_t first, uint32_t count)
{
    if (!count || count > 32U) {
        aclshmemi_kernel_abort("FM SIMT invalid batch size");
        trap();
        return;
    }
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kFmSimtWqeOffset);
    if (!s.fm_context) {
        EnsureDeepepPeerAddress(s, peer);
        auto* info = aclshmemi_udma_qp_info_fetch();
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
        if (!ctx || !ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT ||
            ctx->wqe_size != 64U) {
            aclshmemi_kernel_abort("FM SIMT SQ ABI mismatch");
            trap();
            return;
        }
        auto* target = PeerHiddenAddress(s.book, peer);
        if (!target) {
            aclshmemi_kernel_abort("FM SIMT missing live output");
            trap();
            return;
        }
        deepep::comm::tile::FillUdmaRemoteSqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), target,
            aclshmemi_udma_get_mem_info(info, slot, 0U), 0U);
        // WQE word5 caches the live peer base for this invocation only.
        header[5] = reinterpret_cast<uint64_t>(target);
        s.fm_context = reinterpret_cast<uint64_t>(ctx);
        s.fm_queue = ctx->state_addr;
        s.fm_slot = slot;
    }
    auto* ctx = reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(s.fm_context);
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(s.fm_queue);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    // Relay row pairs are produced by vector compaction (or the V-pipe
    // carry), already completed by the caller. Do not drain the unrelated
    // next metadata-bank prefetch here. Primary's GM-loaded pairs keep their
    // own MTE2 dependency in IssueFmPrimaryPlanRange.
    asc_vf_call<DispatchBuildFmWqesVf>(dim3(32), wqes, header, reinterpret_cast<__ubuf__ int32_t*>(rows.GetPhyAddr()),
                                       first, count, s.row_bytes, reinterpret_cast<uint64_t>(s.hidden), header[5]);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // The consumer is MTE3, not scalar; the V->MTE3 dependency above covers
    // descriptor visibility. The SDK copy completes before any DB is rung.
    PublishFmPreparedRows(reinterpret_cast<__ubuf__ uint8_t*>(wqes), count, ctx, state);
}
