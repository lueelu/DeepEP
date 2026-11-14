#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "../../../comm/tile/tile_udma_wqe.hpp"
#include "simt_api/device_sync_functions.h"
#define ASCEND_DEEPEP_DEVICE_SCHEDULE
#include "common/udma_schedule.h"
#undef ASCEND_DEEPEP_DEVICE_SCHEDULE

namespace ascend_deepep::udma {
// Shared sender protocol: Begin resets incoming markers on all AIVs; each QP
// owner batches payload WQEs, Publish appends SO markers, and Quiet releases its
// local send buffers. Schedule-specific final barriers also cover SIMT metadata.
// WQEs, scalar request descriptors, then two SDK header templates per peer.
inline constexpr uint32_t kRequestsOffset = kBatchWqes * kWqeBytes;
inline constexpr uint32_t kHeadersOffset = kRequestsOffset + kBatchWqes * 4U * sizeof(uint64_t);
inline constexpr uint32_t kSenderScratchBytes = kHeadersOffset + kMaxPeers * 2U * kWqeBytes;
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kData{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kLast{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
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
    if (marker_count > kSignalBytes / sizeof(uint64_t)) { trap(); return; }
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
    aclshmemi_sync_npu_v4<true>(0);
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
    auto* header = headers + (request[3] * 2U + uint32_t(i + 1U == count)) * 8U;
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

// Cache only for this kernel invocation. The SDK still owns admission, SQ
// wrap, publication, and CQ accounting; every short batch requests one CQE.
template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1>
class SenderT {
    static_assert(Qps == 1 || Qps == 2);
public:
    __aicore__ SenderT(
        uint32_t rank, uint32_t world, uint32_t owner, uint32_t group, __ubuf__ uint8_t* scratch, GM_ADDR workspace,
        uint32_t peer_start = UINT32_MAX, uint32_t rotation = 0,
        uint32_t peer_order = kRankOrder, uint32_t ranks_per_group = 0, uint32_t workers_per_peer = 1)
        : qp_(owner * Qps), qp_base_(owner * Qps), scratch_(scratch), local_base_(reinterpret_cast<uint64_t>(workspace))
    {
        auto* info = aclshmemi_udma_qp_info_fetch();
        if (info->qp_num != kQpCount)
            trap();
        const uint32_t peer_owner = owner / workers_per_peer;
        const uint32_t first = peer_start == UINT32_MAX ? owner * group : peer_start;
        const uint32_t count = peer_start == UINT32_MAX
            ? WorkerCount(rank, world, peer_owner, group, peer_order, ranks_per_group)
            : (first < world - 1 ? ((world - 1 - first < group) ? world - 1 - first : group) : 0);
        if (count > Capacity) { trap(); return; }
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
                deepep::comm::tile::FillUdmaRemoteSqe<
                    uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*, kData>(
                    reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(headers), remote, memory, 0U);
                deepep::comm::tile::FillUdmaRemoteSqe<
                    uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*, kLast>(
                    reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(headers + kWqeBytes), remote, memory, 0U);
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

    __aicore__ void Put(uint32_t p, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t bytes)
    {
        p = QueuePeer(p);
        if (pending_ && active_state_ != peers_[p].state)
            Flush();
        if (!pending_) {
            active_state_ = peers_[p].state;
            auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(active_state_);
            aclshmemi_udma_poll_cq_when_sq_overflow(state, peers_[p].slot, qp_, kBatchWqes);
            if (!aclshmemi_udma_sq_has_credit(state, kBatchWqes))
                trap();
        }
        auto* request = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + pending_ * 4U;
        request[0] = peers_[p].remote_base + (reinterpret_cast<uint64_t>(dst) - local_base_);
        request[1] = reinterpret_cast<uint64_t>(src);
        request[2] = bytes;
        request[3] = p;
        last_ = p;
        ++pending_;
        if (pending_ == kBatchWqes)
            Flush();
    }
    __aicore__ void Flush()
    {
        if (pending_ == 0)
            return;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        asc_vf_call<BuildWqes>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset), pending_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        aclshmemi_udma_flush_aggregate_wqes(
            scratch_, peers_[last_].context, peers_[last_].slot, qp_, EVENT_ID5, pending_, kWqeBytes);
        // SDK MTE3_S completion drains WQE reads before scratch/requests reuse.
        pending_ = 0;
    }
    __aicore__ void Publish(GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t worker = 0)
    {
        Flush();
        for (uint32_t lane = 0; lane < Qps; ++lane) {
            auto* one = Signal(base, offset, rank) + worker * Qps + lane;
            for (uint32_t p = 0; p < count_; ++p) {
                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kDone>(
                    one, one, reinterpret_cast<__ubuf__ uint64_t*>(scratch_), 1U,
                    peers_[lane * Capacity + p].peer, qp_base_ + lane, EVENT_ID5);
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
                if (!seen) aclshmemx_udma_qp_quiet(peer.peer, qp_base_ + lane);
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
    uint32_t last_ = 0;
};

using Sender = SenderT<>;

// Ascend950's system counter is 1 GHz (1000 cycles/us), not the AIV core clock.
inline constexpr uint64_t kIncomingCyclesPerUs = 1000ULL;
inline constexpr uint64_t kIncomingTimeoutCycles = 10ULL * 1000 * 1000 * kIncomingCyclesPerUs;

template <bool IsDispatch>
__aicore__ inline void WaitIncoming(
    GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t world, uint32_t core, uint32_t cores,
    uint32_t marker_count = 1)
{
    // One deadline for this core's entire incoming-marker phase, not 10 s per
    // source. Poll the same cache-invalidated EQ marker as SHMEM's unbounded wait.
    const uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    for (uint32_t item = core; item < world * marker_count; item += cores) {
        const uint32_t source = item / marker_count;
        const uint32_t worker = item % marker_count;
        if (source == rank)
            continue;
        auto* address = Signal(base, offset, source) + worker;
        AscendC::GlobalTensor<uint64_t> signal;
        signal.SetGlobalBuffer(address);
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
            const uint64_t value = signal.GetValue(0);
            if (value == 1U)
                break;
            const uint64_t elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - start;
            if (elapsed >= kIncomingTimeoutCycles) {
                if constexpr (IsDispatch) {
                    AscendC::printf(
                        "[V1_UDMA_TIMEOUT] op=dispatch stage=wait_incoming rank=%u core=%u source=%u worker=%u "
                        "signal=0x%llx expected=1 observed=%llu elapsed_us=%llu timeout_us=10000000\n",
                        rank, core, source, worker, reinterpret_cast<unsigned long long>(address),
                        static_cast<unsigned long long>(value),
                        static_cast<unsigned long long>(elapsed / kIncomingCyclesPerUs));
                } else {
                    AscendC::printf(
                        "[V1_UDMA_TIMEOUT] op=combine stage=wait_incoming rank=%u core=%u source=%u worker=%u "
                        "signal=0x%llx expected=1 observed=%llu elapsed_us=%llu timeout_us=10000000\n",
                        rank, core, source, worker, reinterpret_cast<unsigned long long>(address),
                        static_cast<unsigned long long>(value),
                        static_cast<unsigned long long>(elapsed / kIncomingCyclesPerUs));
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
