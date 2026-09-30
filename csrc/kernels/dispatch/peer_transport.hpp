// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

__aicore__ inline uint32_t DeepepTransportQp(const DeepepSecondaryState& s, uint32_t, uint32_t logical)
{
    return ascend_deepep::udma_layout::CrossQp(s.clos_table_owner, logical);
}
__aicore__ inline void EnsureDeepepPeerAddress(DeepepSecondaryState& s, uint32_t peer)
{
    if (!s.address_peer_ready || (s.address_seen[peer / 64U] & (uint64_t(1) << (peer % 64U)))) return;
    // Same single-record protocol as the private C++ peer-ready path. Observing
    // word3 accepts all three live virtual addresses under the target's 32B visibility assumption.
    auto* flag =
        s.book + uint64_t(peer) * ascend_deepep::kPeerAddressRowWords + ascend_deepep::kPeerAddressGenerationWord;
    aclshmem_uint64_wait_until(flag, ACLSHMEM_CMP_EQ, s.magic);
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(s.book + uint64_t(peer) * ascend_deepep::kPeerAddressRowWords),
                    64U);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    s.address_seen[peer / 64U] |= uint64_t(1) << (peer % 64U);
}
template <bool Profile>
__aicore__ inline void PublishClosSimtUbWqes(__ubuf__ uint8_t* wqes, uint32_t count,
                                             __gm__ aclshmemi_udma_wq_ctx_t* ctx,
                                             __gm__ aclshmemi_udma_queue_state_t* state, uint64_t* detail,
                                             uint32_t mode)
{
    // Capacity for the WHOLE batch was admitted before VF. Use the SHMEM copy
    // and DB helpers, not flush_aggregate_wqes: that helper unconditionally
    // counts a terminal data CQE, which this unsignaled path does not generate.
    const uint32_t head = state->sq_head;
    const uint32_t index = head % ctx->depth;
    const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
    AscendC::LocalTensor<uint8_t> view;
    view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes);
    const uint64_t copy_start = DetailClock<Profile>(mode);
    aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * ctx->wqe_size),
                                    view, first * ctx->wqe_size, EVENT_ID0);
    if (first != count) {
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes + first * ctx->wqe_size);
        aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view,
                                        (count - first) * ctx->wqe_size, EVENT_ID0);
    }
    // Both ring spans are complete before ONE DB; no data CQE counter update.
    // SHMEM MTE3_S also makes the UB safe to reuse. This is not payload quiet.
    const uint64_t db_start = DetailClock<Profile>(mode);
    DetailAdd<Profile>(detail, 9U, db_start - copy_start);
    aclshmemi_udma_post_send_update_info(head + count, ctx, state);
    DetailAdd<Profile>(detail, 10U, DetailClock<Profile>(mode) - db_start);
    DetailAdd<Profile>(detail, 1U, 1U);
    DetailAdd<Profile>(detail, 3U, first != count ? 1U : 0U);
}

#pragma once

// Notify peer lists; server dedup is unchanged. Eight Clos
// owners retain BOTH QPs for their peer set (physical lane by default;
// EP64 group-offset uses source-local-rank staggered membership).
struct ClosPeerTablePeer {
    int32_t peer = -1;
    uint32_t begin[2]{}, count[2]{}, cursor[2]{};
    uint32_t queue[2]{};
    bool released = false;
    uint64_t target = 0U, context[2]{}, link[16]{};
};

// Table payloads never request CQEs. At most two SO boundaries are outstanding
// per actual SQ. Each SO owns a known CQ ordinal: observe owner/status, not
// an assumed equality between CQE.entry_idx and the software SQ head.
// Read-only: leave CQ/SQ tails and the CQ doorbell to the later quiet.
template <bool Wait>
__aicore__ inline bool ObserveClosPeerTableSoCqe(uint32_t peer, uint32_t qp, uint64_t expected_state, uint32_t ticket)
{
    auto* info = aclshmemi_udma_qp_info_fetch();
    auto* table = aclshmemi_udma_active_table(info);
    const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
    auto* cq = reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(table->scq_ptr + (uint64_t(slot) * info->qp_num + qp) *
                                                                                      sizeof(aclshmemi_udma_cq_ctx_t));
    constexpr uint32_t depth = shm::UDMA_CQ_DEPTH_DEFAULT;
    if (!expected_state || cq->state_addr != expected_state || cq->depth != depth ||
        cq->cqe_size < sizeof(aclshmemi_jfc_cqe_ctx_t)) {
        aclshmemi_kernel_abort("peer-table SO CQ/state ABI mismatch\n");
        trap();
        return false;
    }
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(expected_state);
    const uint32_t pending = state->cqe_cnt - state->cq_tail;
    if (!pending || pending > 2U || uint32_t(ticket - state->cq_tail) >= pending) {
        aclshmemi_kernel_abort("peer-table SO CQ ticket outside outstanding boundaries\n");
        trap();
        return false;
    }
    auto* entry = reinterpret_cast<__gm__ aclshmemi_jfc_cqe_ctx_t*>(cq->buf_addr +
                                                                    uint64_t(ticket & (depth - 1U)) * cq->cqe_size);
    const uint32_t old_owner = (ticket / depth) & 1U;
    for (uint32_t tries = 0U; tries < (Wait ? MAX_RETRY_TIMES : 1U); ++tries) {
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(entry), sizeof(aclshmemi_jfc_cqe_ctx_t));
        if (entry->owner != old_owner) {
            if (entry->status || entry->substatus) {
                aclshmemi_kernel_abort("peer-table SO CQE failed\n");
                trap();
                return false;
            }
            return true;
        }
    }
    if constexpr (Wait) {
        aclshmemi_kernel_abort("peer-table SO CQE timeout\n");
        trap();
    }
    return false;
}

__aicore__ inline void WaitClosPeerTableSoCqe(uint32_t peer, uint32_t qp, uint64_t expected_state, uint32_t ticket)
{
    ObserveClosPeerTableSoCqe<true>(peer, qp, expected_state, ticket);
}

struct ClosPeerTableQueue {
    uint64_t state = 0U;
    uint32_t peer = 0U, qp = 0U, rows = 0U, mid_ticket = 0U, final_ticket = 0U;
    bool mid_used = false, mid_seen = false;
};

__aicore__ inline __gm__ uint64_t* ClosRelayAssistOwner(uint64_t mailbox, uint32_t owner)
{
    return reinterpret_cast<__gm__ uint64_t*>(mailbox) + uint64_t(owner) * ascend_deepep::kClosRelayAssistOwnerWords;
}

__aicore__ inline bool ClosRelayAssistGeneration(__gm__ uint64_t* line, uint64_t generation)
{
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(line));
    return *reinterpret_cast<__gm__ volatile uint64_t*>(line) == generation;
}

__aicore__ inline void ClosRelayAssistPublishGeneration(__gm__ uint64_t* line, uint64_t generation)
{
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    *line = generation;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(line));
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// Clos publishes immutable CQE snapshots after SO1. AIV7 never fetches/mutates
// the live queue state: Clos may concurrently submit SO2 and change cqe_cnt.
__aicore__ inline void PublishClosRelayAssist(DeepepSecondaryState& s, const ClosPeerTablePeer* peers,
                                              uint32_t peer_count, const ClosPeerTableQueue* queues,
                                              uint32_t queue_count, bool group_offset = false, uint32_t split_task = 0U)
{
    using namespace ascend_deepep;
    auto* task = ClosRelayAssistOwner(s.clos_relay_mailbox, s.clos_split_qp ? split_task : s.core - 8U);
    auto* info = aclshmemi_udma_qp_info_fetch();
    auto* table = aclshmemi_udma_active_table(info);
    // Word3 is in the generation cache line; reused for profile only AFTER ACK.
    task[1] = queue_count;
    task[2] = peer_count;
    task[3] = group_offset ? 1U : 0U;
    for (uint32_t q = 0U; q < queue_count; ++q) {
        const auto& queue = queues[q];
        const uint32_t physical = DeepepTransportQp(s, queue.peer, queue.qp);
        const uint32_t slot = aclshmemi_udma_compute_slot(queue.peer, static_cast<uint32_t>(-1));
        auto* cq = reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(
            table->scq_ptr + (uint64_t(slot) * info->qp_num + physical) * sizeof(aclshmemi_udma_cq_ctx_t));
        constexpr uint32_t depth = shm::UDMA_CQ_DEPTH_DEFAULT;
        if (cq->state_addr != queue.state || cq->depth != depth || cq->cqe_size < sizeof(aclshmemi_jfc_cqe_ctx_t)) {
            aclshmemi_kernel_abort("relay assistant CQ ABI mismatch\n");
            trap();
            return;
        }
        auto* record = task + kClosRelayAssistQueueWord + q * 4U;
        record[0] = cq->buf_addr + uint64_t(queue.mid_ticket & (depth - 1U)) * cq->cqe_size;
        record[1] = (queue.mid_ticket / depth) & 1U;
        record[2] = physical;
        record[3] = queue.mid_ticket;
    }
    for (uint32_t j = 0U; j < peer_count; ++j) {
        auto* record = task + kClosRelayAssistPeerWord + j * 4U;
        record[0] = uint32_t(peers[j].peer);
        record[1] = peers[j].queue[0];
        record[2] = peers[j].queue[1];
        record[3] = 0U;
    }
    // Do not touch the adjacent helper-owned ack line (one writer / cacheline).
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(task), kClosRelayAssistLineWords * 8U);
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(task + kClosRelayAssistQueueWord),
                    (kClosRelayAssistOwnerWords - kClosRelayAssistQueueWord) * 8U);
    ClosRelayAssistPublishGeneration(task, s.magic);
}

struct ClosRelayAssistState {
    uint32_t started = 0U, done = 0U;
    uint32_t pending = 0U;  // owners with enqueued MTE writes, not yet drained
    uint64_t seen[8]{}, released[8]{};
};

// One immutable UB payload for the entire helper invocation. QP0/1 ready
// slots are adjacent 512B regions; one 1024B copy publishes both only AFTER
// both SO1 CQEs have been observed. Fence sinks (slots 2/4) are untouched.
__aicore__ inline void InitClosRelayAssistPayload(DeepepSecondaryState& s,
                                                  AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy)
{
    constexpr uint32_t words = 2U * kPeerSyncFlagBytes / sizeof(uint64_t);
    auto payload = copy.Get<uint64_t>();
    for (uint32_t i = 0U; i < words; ++i) {
        payload.SetValue(i, i % (kPeerSyncFlagBytes / sizeof(uint64_t)) == 0U ? s.magic : 0U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
}

// One finite scan, never blocking on an unfinished owner/CQE. Enqueue every
// eligible destination, then visit the next owner without a per-flag wait.
// RunClosRelayAssist drains one bounded scan of eight owners, not all SO1s.
template <bool Profile>
__aicore__ inline bool StepClosRelayAssist(DeepepSecondaryState& s, ClosRelayAssistState& state, uint32_t owner,
                                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy)
{
    using namespace ascend_deepep;
    const uint32_t bit = 1U << owner;
    if (state.done & bit) {
        return false;
    }
    auto* task = ClosRelayAssistOwner(s.clos_relay_mailbox, owner);
    auto* task1 = s.clos_split_qp ? ClosRelayAssistOwner(s.clos_relay_mailbox, owner + 8U) : task;
    bool progress = false;
    if (!(state.started & bit)) {
        if (!ClosRelayAssistGeneration(task, s.magic)) {
            return false;
        }
        if (s.clos_split_qp && !ClosRelayAssistGeneration(task1, s.magic)) {
            return false;
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(task + kClosRelayAssistQueueWord),
                        (kClosRelayAssistOwnerWords - kClosRelayAssistQueueWord) * 8U);
        if (s.clos_split_qp) {
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(task1 + kClosRelayAssistQueueWord),
                            (kClosRelayAssistOwnerWords - kClosRelayAssistQueueWord) * 8U);
            if (task1[2] != task[2] || task1[3] != task[3] || task[1] > kClosRelayAssistPeers ||
                task1[1] > kClosRelayAssistPeers) {
                aclshmemi_kernel_abort("split-QP helper header mismatch\n");
                trap();
                return false;
            }
            for (uint32_t j = 0U; j < task[2] && j < kClosRelayAssistPeers; ++j) {
                if (task[kClosRelayAssistPeerWord + j * 4U] != task1[kClosRelayAssistPeerWord + j * 4U]) {
                    aclshmemi_kernel_abort("split-QP helper peer mismatch\n");
                    trap();
                    return false;
                }
            }
        }
        // Validate the staggered ownership ONCE, not in the CQ polling loop.
        if (task[3]) {
            if (s.world != 64U || task[2] != 8U) {
                aclshmemi_kernel_abort("relay assistant invalid offset layout\n");
                trap();
                return false;
            }
            for (uint32_t j = 0U; j < 8U; ++j) {
                if (task[kClosRelayAssistPeerWord + j * 4U] != clos_offset64::Peer(s.rank, owner + j * 8U)) {
                    aclshmemi_kernel_abort("relay assistant invalid offset peer\n");
                    trap();
                    return false;
                }
            }
        }
        state.started |= bit;
        progress = true;
    }
    const uint32_t queues0 = static_cast<uint32_t>(task[1]);
    const uint32_t queues = queues0 + (s.clos_split_qp ? static_cast<uint32_t>(task1[1]) : 0U),
                   peers = static_cast<uint32_t>(task[2]);
    if (!queues || queues > kClosRelayAssistQueues || peers != s.world / 8U || peers > kClosRelayAssistPeers) {
        aclshmemi_kernel_abort("relay assistant invalid mailbox\n");
        trap();
        return false;
    }
    for (uint32_t q = 0U; q < queues; ++q) {
        if (state.seen[owner] & (uint64_t(1) << q)) {
            continue;
        }
        auto* record = s.clos_split_qp && q >= queues0 ? task1 + kClosRelayAssistQueueWord + (q - queues0) * 4U
                                                       : task + kClosRelayAssistQueueWord + q * 4U;
        auto* entry = reinterpret_cast<__gm__ aclshmemi_jfc_cqe_ctx_t*>(record[0]);
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(entry), sizeof(aclshmemi_jfc_cqe_ctx_t));
        if (entry->owner == static_cast<uint32_t>(record[1])) {
            continue;
        }
        if (entry->status || entry->substatus) {
            aclshmemi_kernel_abort("relay assistant SO1 CQE failed\n");
            trap();
            return false;
        }
        state.seen[owner] |= uint64_t(1) << q;
        progress = true;
    }
    for (uint32_t j = 0U; j < peers; ++j) {
        if (state.released[owner] & (uint64_t(1) << j)) {
            continue;
        }
        auto* record = task + kClosRelayAssistPeerWord + j * 4U;
        const uint32_t peer = static_cast<uint32_t>(record[0]);
        if (peer >= s.world || (!task[3] && peer % 8U != owner)) {
            aclshmemi_kernel_abort("relay assistant invalid peer\n");
            trap();
            return false;
        }
        if (peer / 8U == s.rank / 8U) {
            state.released[owner] |= uint64_t(1) << j;
            continue;
        }
        const uint32_t q0 = static_cast<uint32_t>(record[1]);
        const uint32_t q1 = s.clos_split_qp
                                ? queues0 + static_cast<uint32_t>(task1[kClosRelayAssistPeerWord + j * 4U + 2U])
                                : static_cast<uint32_t>(record[2]);
        if (q0 >= queues0 || q1 >= queues || (s.clos_split_qp && q1 < queues0)) {
            aclshmemi_kernel_abort("relay assistant invalid queue map\n");
            trap();
            return false;
        }
        const uint64_t mask = (uint64_t(1) << q0) | (uint64_t(1) << q1);
        if ((state.seen[owner] & mask) != mask) {
            continue;
        }
        // Even an empty data peer receives completion fanout. Its signal
        // initialization must precede this publication, not all Clos setup.
        EnsureDeepepPeerAddress(s, peer);
        constexpr uint32_t words = 2U * kPeerSyncFlagBytes / sizeof(uint64_t);
        auto* flags = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), 0U, s.sync);
        AscendC::GlobalTensor<uint64_t> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(flags, peer)), words);
        auto payload = copy.Get<uint64_t>();
        AscendC::DataCopy(target, payload, words);

        state.pending |= bit;
        state.released[owner] |= uint64_t(1) << j;
        progress = true;
    }
    return progress;
}

template <bool Profile>
__aicore__ inline void DrainClosRelayAssist(DeepepSecondaryState& s, ClosRelayAssistState& state)
{
    using namespace ascend_deepep;
    if (!state.pending) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    for (uint32_t owner = 0U; owner < 8U; ++owner) {
        const uint32_t bit = 1U << owner;
        if (!(state.pending & bit)) {
            continue;
        }
        auto* task = ClosRelayAssistOwner(s.clos_relay_mailbox, owner);

        const uint32_t peers = static_cast<uint32_t>(task[2]);
        if (state.released[owner] != (uint64_t(1) << peers) - 1U) {
            continue;
        }
        auto* ack = task + kClosRelayAssistLineWords;

        // CQ slots cannot be retired/reused until this release is observed.
        ClosRelayAssistPublishGeneration(ack, s.magic);
        if (s.clos_split_qp) {
            auto* ack1 = ClosRelayAssistOwner(s.clos_relay_mailbox, owner + 8U) + kClosRelayAssistLineWords;
            ClosRelayAssistPublishGeneration(ack1, s.magic);
        }
        state.done |= bit;
        DispatchProfileMark<Profile>(s.profile, 16U + owner);
    }
    state.pending = 0U;
}

template <bool Profile>
__aicore__ inline void RunClosRelayAssist(DeepepSecondaryState& s, AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy)
{
    ClosRelayAssistState state;
    InitClosRelayAssistPayload(s, copy);
    uint32_t idle_tries = 0U;
    while (state.done != 0xffU) {
        bool progress = false;
        for (uint32_t owner = 0U; owner < 8U; ++owner) {
            progress |= StepClosRelayAssist<Profile>(s, state, owner, copy);
        }
        DrainClosRelayAssist<Profile>(s, state);
        if (progress) {
            idle_tries = 0U;
        } else {
            if (++idle_tries == MAX_RETRY_TIMES) {
                aclshmemi_kernel_abort("relay assistant timeout rank=%u started=%u done=%u pending=%u\n", s.rank,
                                       state.started, state.done, state.pending);
                trap();
                return;
            }
        }
    }
}

__simt_vf__ __launch_bounds__(32) inline void DispatchBuildPeerTableWqesVf(__ubuf__ uint64_t* wqes,
                                                                           __ubuf__ uint64_t* header,
                                                                           __ubuf__ uint32_t* rows, uint32_t count,
                                                                           uint32_t row_bytes, uint64_t source_base,
                                                                           uint64_t target_base)
{
    for (uint32_t index = threadIdx.x; index < count; index += 32U) {
        auto* sqe = wqes + uint64_t(index) * 8U;
        sqe[0] = header[0];
        sqe[1] = header[1];
        sqe[2] = header[2];
        sqe[3] = header[3];
        sqe[4] = header[4];
        sqe[5] = target_base + uint64_t(rows[index * 2U + 1U]) * row_bytes;
        sqe[6] = row_bytes;
        sqe[7] = source_base + uint64_t(rows[index * 2U]) * row_bytes;
    }
}

template <bool Profile, bool Fp8>
__aicore__ inline void IssueClosPeerTableVisit(DeepepPrimaryState& p, ClosPeerTablePeer& peer, uint32_t qp,
                                               __ubuf__ uint8_t* scratch,
                                               AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    const uint32_t remaining = peer.count[qp] - peer.cursor[qp];
    if (!remaining) {
        return;
    }
    auto& s = *p.secondary;
    const uint32_t rotation_wqes = s.rotation_wqes;
    const uint32_t n = remaining < rotation_wqes ? remaining : rotation_wqes;
    DetailAdd<Profile>(s.detail, 0U, n);
    DetailAdd<Profile>(s.detail, 2U, n < rotation_wqes ? 1U : 0U);
    DetailAdd<Profile>(s.detail, 24U + s.detail_phase, n);
    DetailAdd<Profile>(s.detail, 26U + s.detail_phase, 1U);
    const uint64_t ready_start = DetailClock<Profile>(s.detail_mode);
    if (s.address_peer_ready && !peer.target) {
        // Lazy lookup: an unused/future peer must not delay this visit.
        EnsureDeepepPeerAddress(s, peer.peer);
        peer.target = reinterpret_cast<uint64_t>(PeerHiddenAddress(s.book, peer.peer, s.book_words));
        if (!peer.target) {
            aclshmemi_kernel_abort("peer-table missing live output\n");
            trap();
            return;
        }
    }
    DetailAdd<Profile>(s.detail, 11U, DetailClock<Profile>(s.detail_mode) - ready_start);
    const uint64_t load_start = DetailClock<Profile>(s.detail_mode);
    auto rows = slots.Get<uint32_t>();
    auto* table = reinterpret_cast<__gm__ uint32_t*>(p.clos_peer_rows);
    const uint64_t first = peer.begin[qp] + uint64_t(peer.cursor[qp]);
    NativeLoad(rows, table + first * 2U, n * 2U);
    const uint64_t build_start = DetailClock<Profile>(s.detail_mode);
    DetailAdd<Profile>(s.detail, 7U, build_start - load_start);
    auto* ctx = reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(peer.context[qp]);
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
    // Whole-round preflight reserved every data WQE and all SO boundaries.
    // Do not wait for an absent payload CQE.
    if (!aclshmemi_udma_sq_has_credit(state, n + 1U)) {
        aclshmemi_kernel_abort("peer-table reserved SQ capacity lost\n");
        trap();
        return;
    }
    const uint32_t physical = DeepepTransportQp(s, peer.peer, qp);
    auto* info = aclshmemi_udma_qp_info_fetch();
    const uint32_t slot = aclshmemi_udma_compute_slot(peer.peer, static_cast<uint32_t>(-1));
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                     kUdmaDataRoNoCqeConfig>(reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header),
                                                             reinterpret_cast<__gm__ uint8_t*>(peer.target),
                                                             aclshmemi_udma_get_mem_info(info, slot, physical), 0U);
    // Pairs occupy <=128B; WQEs use a disjoint 1024B view in existing slot UB.
    auto wqes = slots.GetWithOffset<uint64_t>(128U, 256U);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    asc_vf_call<DispatchBuildPeerTableWqesVf>(dim3(32U), reinterpret_cast<__ubuf__ uint64_t*>(wqes.GetPhyAddr()),
                                              header, reinterpret_cast<__ubuf__ uint32_t*>(rows.GetPhyAddr()), n,
                                              s.row_bytes, reinterpret_cast<uint64_t>(p.input), peer.target);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    DetailAdd<Profile>(s.detail, 8U, DetailClock<Profile>(s.detail_mode) - build_start);
    PublishClosSimtUbWqes<Profile>(reinterpret_cast<__ubuf__ uint8_t*>(wqes.GetPhyAddr()), n, ctx, state, s.detail,
                                   s.detail_mode);
    const uint64_t bytes = uint64_t(n) * s.row_bytes;
    peer.cursor[qp] += n;
    p.bytes += bytes;
}

// QP0 uses AIV8..15; QP1 uses AIV16..23 for every EP (one owner/core).
// Each SQ has one writer; there is exactly one lane per core.
// Each QP keeps the original table row membership and 4-WQE SIMT/DB visits.
// Cross-QP 3:1 is a DATA ratio, no longer a serial 3-commit/1-commit gate.
struct ClosSplitLane {
    ClosPeerTablePeer peers[ascend_deepep::kClosRelayAssistPeers]{};
    ClosPeerTableQueue queues[ascend_deepep::kClosRelayAssistPeers]{};
    uint32_t owner = 0U, count = 0U;
    uint64_t so[2]{}, seen = 0U, seen_scan = 0U;
};
struct ClosSplitState {
    ClosSplitLane lanes[1]{};
    uint64_t tables[2]{};
    uint32_t qp = 0U, lane_count = 1U, peer_count = 0U, visit_width = 4U;
    uint32_t phase = 0U, window = 0U;
    bool group_offset = false, interleave = false, balanced = false, pod_staged = false;
};

__aicore__ inline __gm__ uint64_t* ClosSplitLinks(uint64_t mailbox, uint32_t task)
{
    return reinterpret_cast<__gm__ uint64_t*>(mailbox + 2U * ascend_deepep::kClosRelayAssistBytes) +
           task * ascend_deepep::kClosRelayAssistPeers * 16U;
}

// Frozen Notify directories, not payload data. Reuse the otherwise idle high
// slot UB on Clos-only AIVs; the low region remains available to SIMT WQE build.
constexpr uint32_t kClosDirectoryOffset = 64U * 1024U;
constexpr uint32_t kClosDirectoryBytes = 256U * 4U * sizeof(uint32_t);
static_assert(kClosDirectoryOffset + 2U * kClosDirectoryBytes <= kSlotSelectUbBytes);
__aicore__ inline AscendC::LocalTensor<uint32_t> ClosDirectory(AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots,
                                                               uint32_t phase)
{
    return slots.GetWithOffset<uint32_t>(kClosDirectoryBytes / 4U, kClosDirectoryOffset + phase * kClosDirectoryBytes);
}
__aicore__ inline void CacheClosDirectories(const ascend_deepep::DispatchTiling& t,
                                            AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    if (t.world <= 8U || GetBlockIdx() < 8U || GetBlockIdx() >= 24U) {
        return;
    }
    auto header = slots.Get<uint32_t>();
    NativeLoad(header, reinterpret_cast<__gm__ uint32_t*>(t.peer_table), 16U);
    if (header.GetValue(4U) != 3U || header.GetValue(5U) != t.world * 4U ||
        header.GetValue(6U) != t.plan_mailbox_offset / 4U) {
        aclshmemi_kernel_abort("invalid cached peer directory\n");
        trap();
        return;
    }
    auto first = ClosDirectory(slots, 0U);
    auto second = ClosDirectory(slots, 1U);
    NativeLoad(first, reinterpret_cast<__gm__ uint32_t*>(t.peer_table + 64U), t.world * 4U);
    NativeLoad(second, reinterpret_cast<__gm__ uint32_t*>(t.peer_table + 64U + t.plan_phase_bytes), t.world * 4U);
}

template <bool Profile>
__aicore__ inline void LoadClosSplitPhase(ClosSplitState& x, DeepepSecondaryState& s,
                                          AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    for (uint32_t l = 0U; l < x.lane_count; ++l) {
        for (uint32_t j = 0U; j < x.peer_count; ++j) {
            auto& peer = x.lanes[l].peers[j];
            if (uint32_t(peer.peer) / 8U == s.rank / 8U) {
                continue;
            }
            auto rows = ClosDirectory(slots, x.phase);
            peer.begin[x.qp] = rows.GetValue(peer.peer * 4U + x.qp * 2U);
            peer.count[x.qp] = rows.GetValue(peer.peer * 4U + x.qp * 2U + 1U);
            peer.cursor[x.qp] = 0U;
            DetailAdd<Profile>(s.detail, 5U, peer.count[x.qp]);
            DetailAdd<Profile>(s.detail, 6U, (peer.count[x.qp] + s.rotation_wqes - 1U) / s.rotation_wqes);
        }
    }
}

template <bool Profile>
__aicore__ inline void InitClosSplit(ClosSplitState& x, DeepepPrimaryState& p,
                                     AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    auto h = slots.Get<uint32_t>();
    NativeLoad(h, reinterpret_cast<__gm__ uint32_t*>(p.clos_peer_table), 16U);
    const uint32_t options = h.GetValue(ascend_deepep::kClosPeerTableOptionsWord);
    const uint32_t qp1_workers = (options & ascend_deepep::kClosPeerTableQp1EightWorkers) ? 8U : 4U;
    if (qp1_workers != ascend_deepep::scale_clos::Qp1Workers(s.world)) {
        aclshmemi_kernel_abort("split-QP worker schedule mismatch\n");
        trap();
        return;
    }
    if (h.GetValue(4U) != 3U || !s.clos_shared_jetty || s.core < 8U || s.core >= 16U + qp1_workers || s.scale_bytes ||
        s.scale_send_offset || (options & ascend_deepep::kClosPeerTablePairedVf) ||
        !(options & ascend_deepep::kClosPeerTableSplitQp) ||
        (s.world != 16U && s.world != 32U && s.world != 64U && s.world != 128U && s.world != 256U)) {
        aclshmemi_kernel_abort("invalid split-QP table\n");
        trap();
        return;
    }
    x.tables[0] = p.clos_peer_table;
    x.tables[1] = p.clos_peer_table + uint64_t(h.GetValue(5U)) * 4U;
    x.qp = s.core >= 16U ? 1U : 0U;
    x.lane_count = x.qp ? 8U / qp1_workers : 1U;
    x.peer_count = s.world / 8U;
    x.group_offset = (options & ascend_deepep::kClosPeerTableGroupOffset64) != 0U;
    x.interleave = (options & ascend_deepep::kClosPeerTablePodInterleave) != 0U;
    x.balanced = (options & ascend_deepep::kClosPeerTableBalancedLayout) != 0U;
    x.pod_staged = (options & ascend_deepep::kClosPeerTablePodStaged) != 0U;
    if ((x.pod_staged && ((s.world != 128U && s.world != 256U) || !x.balanced || x.interleave || x.group_offset)) ||
        (s.world == 256U && !x.pod_staged)) {
        aclshmemi_kernel_abort("invalid split-QP pod-staged order\n");
        trap();
        return;
    }
    if ((x.group_offset && (s.world != 64U || x.interleave || x.balanced)) || (x.interleave && s.world != 128U)) {
        aclshmemi_kernel_abort("invalid split-QP target order\n");
        trap();
        return;
    }
    x.visit_width = x.group_offset ? 2U : (x.interleave || x.pod_staged) ? 8U : 4U;
    for (uint32_t l = 0U; l < x.lane_count; ++l) {
        auto& lane = x.lanes[l];
        lane.owner = x.qp ? s.core - 16U + l * qp1_workers : s.core - 8U;
        s.clos_table_owner = lane.owner;
        for (uint32_t j = 0U; j < x.peer_count; ++j) {
            auto& peer = lane.peers[j];
            peer.peer = ascend_deepep::scale_clos::Peer(s.rank, s.world, lane.owner, j, x.interleave, x.balanced,
                                                        x.group_offset, x.pod_staged);
            if (uint32_t(peer.peer) / 8U == s.rank / 8U) {
                continue;
            }
            // Data addresses are resolved by the first non-empty visit.
            // SO sinks (slots 2/4) retain prior generations across invocations and
            // are not reset by InitializeTokenScatterPeerSignals. The relay
            // assistant separately waits before publishing final flags.
            if (!s.address_peer_ready) {
                peer.target = reinterpret_cast<uint64_t>(PeerHiddenAddress(s.book, peer.peer, s.book_words));
                if (!peer.target) {
                    aclshmemi_kernel_abort("split-QP missing live output\n");
                    trap();
                    return;
                }
            }

            const uint32_t physical = DeepepTransportQp(s, peer.peer, x.qp);
            auto* ctx =
                aclshmemi_udma_get_qp_ctx(aclshmemi_udma_qp_info_fetch(),
                                          aclshmemi_udma_compute_slot(peer.peer, static_cast<uint32_t>(-1)), physical);
            if (!ctx->state_addr || ctx->wqe_size != 64U || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT) {
                aclshmemi_kernel_abort("split-QP SQ ABI mismatch\n");
                trap();
                return;
            }
            peer.context[x.qp] = reinterpret_cast<uint64_t>(ctx);
            uint32_t q = 0U;
            while (q < lane.count && lane.queues[q].state != ctx->state_addr) {
                ++q;
            }
            if (q == lane.count) {
                lane.queues[q].state = ctx->state_addr;
                lane.queues[q].peer = peer.peer;
                lane.queues[q].qp = x.qp;
                ++lane.count;
            }
            peer.queue[x.qp] = q;
            for (uint32_t phase = 0U; phase < 2U; ++phase) {
                auto rows = ClosDirectory(slots, phase);
                lane.queues[q].rows += rows.GetValue(peer.peer * 4U + x.qp * 2U + 1U);
            }
        }
        for (uint32_t q = 0U; q < lane.count; ++q) {
            auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(lane.queues[q].state);
            if (state->cqe_cnt != state->cq_tail || !aclshmemi_udma_sq_has_credit(state, lane.queues[q].rows + 2U)) {
                aclshmemi_kernel_abort("split-QP tables exceed available SQ\n");
                trap();
                return;
            }
        }
    }
    LoadClosSplitPhase<Profile>(x, s, slots);
}

// One sweep over the current stage. No receiver credit/owner barrier added.
template <bool Profile, bool Fp8>
__aicore__ inline bool StepClosSplit(ClosSplitState& x, DeepepPrimaryState& p, __ubuf__ uint8_t* wqe,
                                     AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    if (x.phase == 2U) {
        return true;
    }
    p.clos_peer_table = x.tables[x.phase];
    if constexpr (Profile) s.detail_phase = x.phase;
    bool remaining = false;
    for (uint32_t j = x.window * x.visit_width; j < (x.window + 1U) * x.visit_width; ++j) {
        for (uint32_t l = 0U; l < x.lane_count; ++l) {
            auto& lane = x.lanes[l];
            auto& peer = lane.peers[j];
            if (uint32_t(peer.peer) / 8U == s.rank / 8U) {
                continue;
            }
            s.clos_table_owner = lane.owner;
            IssueClosPeerTableVisit<Profile, Fp8>(p, peer, x.qp, wqe, slots);
            remaining |= peer.cursor[x.qp] != peer.count[x.qp];
        }
    }
    if (remaining || ++x.window < x.peer_count / x.visit_width) {
        return false;
    }
    for (uint32_t l = 0U; l < x.lane_count; ++l) {
        auto& lane = x.lanes[l];
        s.clos_table_owner = lane.owner;
        for (uint32_t q = 0U; q < lane.count; ++q) {
            auto& queue = lane.queues[q];
            auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(queue.state);
            if (uint32_t(state->cqe_cnt - state->cq_tail) != x.phase) {
                aclshmemi_kernel_abort("split-QP unexpected CQ boundary\n");
                trap();
                return true;
            }
            if (x.phase == 0U) {
                queue.mid_ticket = state->cqe_cnt;
            } else {
                queue.final_ticket = state->cqe_cnt;
            }
            DetailAdd<Profile>(s.detail, 4U, 1U);
            aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
                GetPeerCompletionFlag(s.rank, GetGroupRankIndex(queue.peer, s.world), 2U + 2U * x.qp, s.sync),
                GetPeerCompletionFlag(s.rank, GetGroupRankIndex(queue.peer, s.world), kPeerCompletionValueSlot, s.sync),
                reinterpret_cast<__ubuf__ uint64_t*>(wqe), 1U, queue.peer, DeepepTransportQp(s, queue.peer, x.qp),
                EVENT_ID0);
        }

        if (x.phase == 0U) {
            PublishClosRelayAssist(s, lane.peers, x.peer_count, lane.queues, lane.count, x.group_offset,
                                   lane.owner + x.qp * 8U);
        }
    }
    DispatchProfileMark<Profile>(s.profile, 9U + x.phase);
    ++x.phase;
    x.window = 0U;
    if (x.phase < 2U) {
        LoadClosSplitPhase<Profile>(x, s, slots);
    }
    p.clos_peer_table = x.tables[0];
    return x.phase == 2U;
}

template <bool Profile>
__aicore__ inline void FinishClosSplit(ClosSplitState& x, DeepepSecondaryState& s)
{
    uint64_t seen[1]{};
    for (uint32_t scan = 1U;; ++scan) {
        bool done = true;
        for (uint32_t l = 0U; l < x.lane_count; ++l) {
            auto& lane = x.lanes[l];
            s.clos_table_owner = lane.owner;
            for (uint32_t q = 0U; q < lane.count; ++q) {
                auto& queue = lane.queues[q];
                if (!(seen[l] & (uint64_t(1) << q)) &&
                    ObserveClosPeerTableSoCqe<false>(queue.peer, DeepepTransportQp(s, queue.peer, x.qp), queue.state,
                                                     queue.final_ticket)) {
                    seen[l] |= uint64_t(1) << q;
                }
            }
            if (seen[l] != (uint64_t(1) << lane.count) - 1U) {
                done = false;
            }
        }
        if (done) {
            break;
        }
        if (scan == MAX_RETRY_TIMES) {
            aclshmemi_kernel_abort("split-QP final CQ timeout\n");
            trap();
            return;
        }
    }
    DispatchProfileMark<Profile>(s.profile, 11U);
    for (uint32_t l = 0U; l < x.lane_count; ++l) {
        auto* ack = ClosRelayAssistOwner(s.clos_relay_mailbox, x.lanes[l].owner + x.qp * 8U) +
                    ascend_deepep::kClosRelayAssistLineWords;
        uint32_t tries = 0U;
        while (!ClosRelayAssistGeneration(ack, s.magic)) {
            if (++tries == MAX_RETRY_TIMES) {
                const uint32_t owner = x.lanes[l].owner;
                auto* task0 = ClosRelayAssistOwner(s.clos_relay_mailbox, owner);
                auto* task1 = ClosRelayAssistOwner(s.clos_relay_mailbox, owner + 8U);
                ClosRelayAssistGeneration(task0, s.magic);
                ClosRelayAssistGeneration(task1, s.magic);
                auto* ack0 = task0 + ascend_deepep::kClosRelayAssistLineWords;
                auto* ack1 = task1 + ascend_deepep::kClosRelayAssistLineWords;
                ClosRelayAssistGeneration(ack0, s.magic);
                ClosRelayAssistGeneration(ack1, s.magic);
                aclshmemi_kernel_abort(
                    "split-QP assist ACK timeout rank=%u owner=%u qp=%u generation=%llu "
                    "published0=%llu published1=%llu ack0=%llu ack1=%llu\n",
                    s.rank, owner, x.qp, static_cast<unsigned long long>(s.magic),
                    static_cast<unsigned long long>(task0[0]), static_cast<unsigned long long>(task1[0]),
                    static_cast<unsigned long long>(ack0[0]), static_cast<unsigned long long>(ack1[0]));
                trap();
                return;
            }
        }
    }
    DispatchProfileMark<Profile>(s.profile, 12U);
    for (uint32_t l = 0U; l < x.lane_count; ++l) {
        auto& lane = x.lanes[l];
        s.clos_table_owner = lane.owner;
        for (uint32_t q = 0U; q < lane.count; ++q) {
            aclshmemx_udma_qp_quiet(lane.queues[q].peer, DeepepTransportQp(s, lane.queues[q].peer, x.qp));
        }
    }
}

template <bool Profile, bool Fp8>
__aicore__ inline void DispatchClosSplitQp(DeepepPrimaryState& p, __ubuf__ uint8_t* wqe,
                                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& /*copy*/,
                                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    ClosSplitState state;
    InitClosSplit<Profile>(state, p, slots);
    DispatchProfileMark<Profile>(p.secondary->profile, 8U);
    while (!StepClosSplit<Profile, Fp8>(state, p, wqe, slots)) {
    }
    FinishClosSplit<Profile>(state, *p.secondary);
    DispatchProfileMark<Profile>(p.secondary->profile, 13U);
}
