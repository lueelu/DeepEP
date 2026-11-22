#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/device_sync_functions.h"
#include "kernels/dispatch/dispatch_device_trace.h"
#define ASCEND_DEEPEP_DEVICE_SCHEDULE
#include "common/udma_schedule.h"
#undef ASCEND_DEEPEP_DEVICE_SCHEDULE

namespace ascend_deepep::udma {
// Shared sender protocol: Begin resets incoming markers on all AIVs; each QP
// owner batches payload WQEs, Publish appends SO markers, and Quiet releases its
// local send buffers. Schedule-specific final barriers also cover SIMT metadata.
// WQEs [0,2048), descriptors [2048,3072), then 128 bytes per peer/QP.
// Only the first 64 bytes of each header pair are read by VF. The unused
// second header of peer zero is independent pressure-WQE scratch; retaining
// the old stride keeps all caller UB/self-copy layouts unchanged.
inline constexpr uint32_t kRequestsOffset = kBatchWqes * kWqeBytes;
inline constexpr uint32_t kHeadersOffset = kRequestsOffset + kBatchWqes * 4U * sizeof(uint64_t);
inline constexpr uint32_t kSenderScratchBytes = kHeadersOffset + kMaxPeers * 2U * kWqeBytes;
inline constexpr uint32_t kPressureScratchOffset = kHeadersOffset + kWqeBytes;
static_assert(kPressureScratchOffset % 64U == 0 &&
              kPressureScratchOffset + kWqeBytes <= kHeadersOffset + 2U * kWqeBytes);
// Each source owns a 512-byte control slot. Final readiness uses words 0..31;
// words 32..63 are private pressure sinks, one per physical QP. No receiver
// polls these sinks. Begin only initializes readiness; sinks need no reset.
inline constexpr uint32_t kPressureSignalOffset = kQpCount * sizeof(uint64_t);
static_assert(kPressureSignalOffset + kQpCount * sizeof(uint64_t) <= kSignalBytes);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kData{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kDone{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};

__aicore__ inline __gm__ uint64_t* Signal(GM_ADDR base, uint64_t offset, uint32_t source)
{
    return reinterpret_cast<__gm__ uint64_t*>(base + offset + uint64_t(source) * kSignalBytes);
}

// Every launched core participates, including cores with no send peers. Separate
// calls sharing a runtime must be serialized. The fixed header is disjoint from
// both notify and payload: a fast rank can reset its local receive markers while
// a slower rank still consumes the previous payload. Quiet() in the preceding
// operation has released this rank's outgoing marker source. When Begin is the
// only entry barrier (ordinary combine), no payload writes may precede it.
// Dispatch's pre-Begin scale packing/top-k initialization still needs notify.
__aicore__ inline void Begin(
    GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t world, uint32_t core, uint32_t cores,
    AscendC::LocalTensor<uint64_t> scratch, uint32_t marker_count = 1)
{
    if (marker_count > kQpCount) { trap(); return; }
    for (uint32_t source = core; source < world; source += cores) {
        for (uint32_t worker = 0; worker < marker_count; ++worker)
            scratch.SetValue(worker, source == rank ? 1U : 0U);
        AscendC::GlobalTensor<uint64_t> out;
        out.SetGlobalBuffer(Signal(base, offset, source));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
        AscendC::DataCopyExtParams copy{1, marker_count * uint32_t(sizeof(uint64_t)), 0, 0, 0};
        AscendC::DataCopyPad(out, scratch, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    }
    aclshmemx_sync_vec_all();
}

struct PeerQueue {
    uint32_t peer, slot;
    uint64_t state;
    uint64_t remote_base;
    __gm__ aclshmemi_udma_wq_ctx_t* context;
};

// The pinned SDK encodes WRITE as a 48-byte SQE plus a 16-byte SGE. Fixed
// fields come from its header builder, not duplicated bitfield definitions.
static_assert(sizeof(aclshmemi_sqe_ctx_t) == 48 && sizeof(aclshmemi_sge_ctx_t) == 16);
static_assert(ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE == kWqeBytes);
__simt_vf__ __launch_bounds__(32) inline void BuildWqes(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* requests, __ubuf__ uint64_t* headers, uint32_t count)
{
    const uint32_t i = threadIdx.x;
    if (i >= count)
        return;
    auto* request = requests + i * 4U;
    auto* header = headers + request[3] * 2U * 8U;
    auto* wqe = wqes + i * 8U;
    wqe[0] = header[0];
    wqe[1] = header[1];
    wqe[2] = header[2];
    wqe[3] = header[3];
    wqe[4] = header[4];
    wqe[5] = request[0]; // remote VA
    wqe[6] = request[2]; // byte count, reserved upper SGE bits zero
    wqe[7] = request[1]; // local VA
}

// One AIV owns each QP lane. Peers sharing state_addr share physical SQ/CQ
// credits, even when their destination headers differ. Ordinary WRITE batches
// are unsignaled; SO pressure checkpoints and the existing final flags provide
// completion. Only SHMEM updates CQ accounting and reclaims the physical tail.
template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1, class Trace = DisabledDispatchDeviceTrace>
class SenderT {
    static_assert(Capacity > 0 && (Qps == 1 || Qps == 2));
public:
    __aicore__ SenderT(
        uint32_t rank, uint32_t world, uint32_t owner, uint32_t group, __ubuf__ uint8_t* scratch, GM_ADDR workspace,
        uint32_t peer_start = UINT32_MAX, uint32_t rotation = 0,
        uint32_t peer_order = kRankOrder, uint32_t ranks_per_group = 0, uint32_t workers_per_peer = 1)
        : qp_(owner * Qps), qp_base_(owner * Qps), scratch_(scratch), local_base_(reinterpret_cast<uint64_t>(workspace)), rank_(rank)
    {
        auto* info = aclshmemi_udma_qp_info_fetch();
        if (info->qp_num != kQpCount)
            trap();
        const uint32_t peer_owner = owner / workers_per_peer;
        const uint32_t first = peer_start == UINT32_MAX ? owner * group : peer_start;
        const uint32_t count = peer_start == UINT32_MAX
            ? WorkerCount(rank, world, peer_owner, group, peer_order, ranks_per_group)
            : (first < world - 1 ? ((world - 1 - first < group) ? world - 1 - first : group) : 0);
        // Idle/helper AIVs may have owner >= 32, but never access a QP.
        if (count > Capacity || (count && qp_base_ + Qps > kQpCount)) { trap(); return; }
        count_ = count;
        for (uint32_t lane = 0; lane < Qps; ++lane) {
            const uint32_t qp = qp_base_ + lane;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t peer = peer_start == UINT32_MAX
                    ? WorkerPeer(rank, world, peer_owner, group, i, peer_order, ranks_per_group)
                    : Peer(rank, world, (first + i + rotation) % (world - 1));
                const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
                auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, qp);
                if (!ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT ||
                    ctx->wqe_size != kWqeBytes)
                    trap();
                auto* remote = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(workspace, peer));
                if (!remote)
                    trap();
                auto* memory = aclshmemi_udma_get_mem_info(info, slot, qp);
                const uint32_t index = lane * Capacity + i;
                auto* headers = scratch_ + kHeadersOffset + index * 2U * kWqeBytes;
                aclshmemi_udma_fill_sqe_base_ctx<
                    uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*, kData>(
                    reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(headers), remote, memory, 0U);
                peers_[index] = {peer, slot, ctx->state_addr, reinterpret_cast<uint64_t>(remote), ctx};
                // Independent QP lanes must never alias the same SQ state, even
                // when two peer endpoints within one lane share a Jetty.
                if constexpr (Qps == 2) {
                    if (lane) for (uint32_t before = 0; before < count; ++before)
                        if (peers_[before].state == ctx->state_addr) trap();
                }
            }
        }
    }
    __aicore__ uint32_t Count() const
    {
        return count_;
    }
    __aicore__ uint32_t Rank(uint32_t p) const
    {
        return peers_[p].peer;
    }

    __aicore__ void SetTrace(Trace* trace) { trace_ = trace; }
    __aicore__ uint64_t TraceNow() const {
        if constexpr (Trace::enabled) return trace_->Now();
        return 0;
    }
    __aicore__ void TraceAdd(DispatchTraceFields::Field field, uint64_t begin) {
        if constexpr (Trace::enabled) trace_->Add(field, begin);
    }
    __aicore__ void TraceCount(DispatchTraceFields::Field field) {
        if constexpr (Trace::enabled) trace_->Count(field);
    }
    __aicore__ void FinishWqeBuild(uint64_t begin) {
        if constexpr (Trace::enabled) trace_->FinishWqeBuild(begin);
    }
    __aicore__ void EnsureCredit(uint32_t p, uint32_t required_bb) {
        auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(peers_[p].state);
        const uint32_t qp = qp_base_ + p / Capacity;
        const uint64_t start = TraceNow();
        const bool blocked = !aclshmemi_udma_sq_has_credit(state, required_bb);
        if constexpr (Trace::enabled) {
            // Software head/tail snapshot, not hardware queue occupancy.
            const uint32_t used = state->sq_head - state->sq_tail;
            trace_->Count(Trace::CreditChecks);
            trace_->Accumulate(Trace::SqUsedSum, used);
            trace_->Maximum(Trace::SqUsedMax, used);
            if (!used) trace_->Count(Trace::SqEmptyChecks);
            if (blocked) trace_->Count(Trace::CreditBlocked);
        }
        if (blocked) {
            // A completed CQE can reclaim an earlier prefix, not the later
            // unsignaled suffix. Never set sq_tail = sq_head in software.
            if (state->cqe_cnt != state->cq_tail) {
                TraceCount(Trace::CreditQuietCalls);
                aclshmemx_udma_qp_quiet(peers_[p].peer, qp);
            }
            if (!aclshmemi_udma_sq_has_credit(state, required_bb)) {
                if (!aclshmemi_udma_sq_has_credit(state, 1U)) { trap(); return; }
                // Full Clos reserves its control header at workspace offset 0.
                // Own-rank readiness word zero is immutable one until the next
                // Begin. The outgoing source is released by the quiet below.
                auto* one = Signal(reinterpret_cast<GM_ADDR>(local_base_), 0, rank_);
                auto* sink = one + kQpCount + qp;
                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kDone>(
                    sink, one, reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kPressureScratchOffset),
                    1U, peers_[p].peer, qp, EVENT_ID5);
                TraceCount(Trace::CqeRequested);
                TraceCount(Trace::CreditCheckpoints);
                TraceCount(Trace::CreditQuietCalls);
                aclshmemx_udma_qp_quiet(peers_[p].peer, qp);
            }
            if (!aclshmemi_udma_sq_has_credit(state, required_bb)) { trap(); return; }
        }
        if constexpr (Trace::enabled) {
            const uint64_t elapsed = trace_->Now() - start;
            trace_->Accumulate(Trace::CreditCycles, elapsed);
            if (blocked) trace_->Accumulate(Trace::CreditBlockedCycles, elapsed);
            trace_->Maximum(Trace::CreditMaxCycles, elapsed);
        }
    }
    __aicore__ void Admit(uint32_t p) {
        // WRITE is one WQEBB in the pinned SDK. Reserve one additional WQEBB
        // before every batch, so pressure can always post an SO+CQE checkpoint.
        static_assert(get_wqe_bb_cnt<aclshmemi_udma_opcode_t::UDMA_OP_WRITE>() == 1U);
        EnsureCredit(p, kBatchWqes + 1U);
    }
    __aicore__ void SubmitBatch(uint32_t count, uint32_t spans, bool metadata) {
        if constexpr (Trace::enabled) {
            trace_->Accumulate(Trace::Wqes, count);
            trace_->Accumulate(Trace::Spans, spans);
            trace_->Count(Trace::Flushes);
            if (count < kBatchWqes) trace_->Count(Trace::ShortFlushes);
            if (metadata) {
                trace_->Accumulate(Trace::MetadataWqes, count);
                trace_->Count(Trace::MetadataFlushes);
            }
            auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(peers_[last_].state);
            if (!aclshmemi_udma_sq_has_credit(state, count)) trace_->Count(Trace::SubmitNoCredit);
        }
        const uint64_t start = TraceNow();
        auto* ctx = peers_[last_].context;
        auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(peers_[last_].state);
        // Admission preceded VF; no other AIV writes this physical SQ. Do not
        // use aggregate flush: it unconditionally counts a CQE for each batch.
        if (!count || count > kBatchWqes || !aclshmemi_udma_sq_has_credit(state, count + 1U)) {
            trap(); return;
        }
        const uint32_t head = state->sq_head;
        const uint32_t index = head % ctx->depth;
        const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
        AscendC::LocalTensor<uint8_t> view;
        view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch_);
        aclshmemi_udma_copy_wqe_from_ub(
            reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * kWqeBytes),
            view, first * kWqeBytes, EVENT_ID5);
        if (first != count) {
            view.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch_ + first * kWqeBytes);
            aclshmemi_udma_copy_wqe_from_ub(
                reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view, (count - first) * kWqeBytes, EVENT_ID5);
            TraceCount(Trace::SqWraps);
        }
        // Both ring spans are visible before the doorbell. The SDK copy's
        // MTE3_S releases UB, but NOT the payload or metadata GM sources.
        aclshmemi_udma_post_send_update_info(head + count, ctx, state);
        TraceAdd(Trace::WqeSubmitCycles, start);
    }

    __aicore__ void Put(uint32_t p, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t bytes)
    {
        const uint32_t queue = QueuePeer(p);
        PutAbsolute(p, peers_[queue].remote_base + (reinterpret_cast<uint64_t>(dst) - local_base_), src, bytes);
    }
    // Published tensor VAs are already remote addresses; never translate as SHMEM offsets.
    __aicore__ void PutAbsolute(uint32_t p, uint64_t dst, __gm__ uint8_t* src, uint32_t bytes)
    {
        p = QueuePeer(p);
        if (pending_ && active_state_ != peers_[p].state) {
            TraceCount(Trace::StateFlushes);
            Flush();
        }
        if (!pending_) {
            active_state_ = peers_[p].state;
            Admit(p);
        }
        const uint64_t prepared = TraceNow();
        auto* request = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + pending_ * 4U;
        request[0] = dst;
        request[1] = reinterpret_cast<uint64_t>(src);
        request[2] = bytes;
        request[3] = p;
        last_ = p;
        ++pending_;
        TraceAdd(Trace::DescriptorCycles, prepared);
        if (pending_ == kBatchWqes)
            Flush();
    }
    __aicore__ void Flush()
    {
        if (pending_ == 0)
            return;
        const uint64_t built = TraceNow();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        asc_vf_call<BuildWqes>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset), pending_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        FinishWqeBuild(built);
        SubmitBatch(pending_, pending_, true);
        // SDK MTE3_S completion drains WQE reads before scratch/requests reuse.
        pending_ = 0;
    }
    __aicore__ void Publish(GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t worker = 0)
    {
        Flush();
        for (uint32_t lane = 0; lane < Qps; ++lane) {
            auto* one = Signal(base, offset, rank) + worker * Qps + lane;
            for (uint32_t p = 0; p < count_; ++p) {
                EnsureCredit(lane * Capacity + p, 1U);
                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kDone>(
                    one, one, reinterpret_cast<__ubuf__ uint64_t*>(scratch_), 1U,
                    peers_[lane * Capacity + p].peer, qp_base_ + lane, EVENT_ID5);
                TraceCount(Trace::CqeRequested);
            }
        }
    }
    __aicore__ void Quiet()
    {
        // Drain every distinct state in every owned QP. A qp_idx may span
        // several EIDs; two different qp_idx values must not alias an SQ.
        for (uint32_t lane = 0; lane < Qps; ++lane) {
            for (uint32_t p = 0; p < count_; ++p) {
                const auto& peer = peers_[lane * Capacity + p];
                bool seen = false;
                for (uint32_t before = 0; before < p; ++before)
                    seen |= peers_[lane * Capacity + before].state == peer.state;
                if (!seen) {
                    const uint64_t start = TraceNow();
                    aclshmemx_udma_qp_quiet(peer.peer, qp_base_ + lane);
                    TraceAdd(Trace::FinalQuietCycles, start);
                    TraceCount(Trace::FinalQuietCalls);
                }
            }
        }
    }

protected:
    // Caller must flush its WQE/span builder before switching lanes.
    __aicore__ void SelectQp(uint32_t lane) { qp_ = qp_base_ + lane; }
    __aicore__ uint32_t QueuePeer(uint32_t p) const { return (qp_ - qp_base_) * Capacity + p; }
    PeerQueue peers_[Capacity * Qps]{};
    uint32_t qp_, qp_base_, count_ = 0, pending_ = 0;
    __ubuf__ uint8_t* scratch_;
    uint64_t local_base_, active_state_ = 0;
    uint32_t rank_, last_ = 0;
    Trace* trace_ = nullptr;
};

using Sender = SenderT<>;

template <bool IsDispatch, class Tiling>
__aicore__ inline void WaitIncoming(
    GM_ADDR base, const Tiling& td, uint32_t core, uint32_t cores, uint32_t marker_count)
{
    // One deadline for this AIV's entire incoming-marker phase.
    const uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    for (uint32_t item = core; item < td.world_size * marker_count; item += cores) {
        const uint32_t source = item / marker_count;
        const uint32_t worker = item % marker_count;
        if (source == td.rank)
            continue;
        auto* address = Signal(base, td.udma_completion_signal_offset, source) + worker;
        AscendC::GlobalTensor<uint64_t> signal;
        signal.SetGlobalBuffer(address);
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
            const uint64_t value = signal.GetValue(0);
            if (value == 1U)
                break;
            const uint64_t elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - start;
            if (elapsed >= td.npu_timeout_cycles) {
                const auto elapsed_us = static_cast<unsigned long long>(elapsed / kNpuSystemCyclesPerUs);
                const auto timeout_s = static_cast<uint32_t>(td.npu_timeout_cycles / kNpuSystemCyclesPerSecond);
                if constexpr (IsDispatch) {
                    AscendC::printf(
                        "[V1_UDMA_TIMEOUT] op=dispatch stage=wait_incoming rank=%u aiv_core_idx=%u "
                        "peer_rank=%u worker_idx=%u instance_id=%u cached=%u "
                        "signal=0x%llx expected=1 observed=%llu elapsed_us=%llu timeout_s=%u\n",
                        td.rank, core, source, worker, td.instance_id, td.cached_mode,
                        reinterpret_cast<unsigned long long>(address), static_cast<unsigned long long>(value),
                        elapsed_us, timeout_s);
                } else {
                    AscendC::printf(
                        "[V1_UDMA_TIMEOUT] op=combine stage=wait_incoming rank=%u aiv_core_idx=%u "
                        "peer_rank=%u worker_idx=%u instance_id=%u "
                        "signal=0x%llx expected=1 observed=%llu elapsed_us=%llu timeout_s=%u\n",
                        td.rank, core, source, worker, td.instance_id,
                        reinterpret_cast<unsigned long long>(address), static_cast<unsigned long long>(value),
                        elapsed_us, timeout_s);
                }
                // Fail the device task. Returning normally would strand other
                // AIVs at SyncAll or expose incomplete payload as valid output.
                trap();
            }
        }
    }
    AscendC::SyncAll<true>();
}
#endif
} // namespace ascend_deepep::udma
