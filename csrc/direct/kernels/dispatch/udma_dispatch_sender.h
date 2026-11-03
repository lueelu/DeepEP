#pragma once

#include "kernels/common/udma_shared_sender.h"
#include "common/types.h"
#include "kernels/dispatch/udma_dispatch_output.h"
#include "utils/debug/asc_assert.h"

namespace ascend_deepep::udma {
inline constexpr uint32_t kDispatchBasesOffset = kSenderScratchBytes;
inline constexpr uint32_t kDispatchEpochOffset = kDispatchBasesOffset + kMaxPeers * kOutputRecordBytes;
inline constexpr uint32_t kDispatchSenderBytes = kDispatchEpochOffset + 32;
static_assert(kDispatchSenderBytes % 32 == 0);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Spans reference the immutable GM plan, so a list-window refill never forces
// an extra short SQ batch or leaves SIMT reading overwritten UB indices.
__simt_vf__ __launch_bounds__(32) inline void BuildDispatchWqes(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, __ubuf__ uint64_t* headers, __ubuf__ uint64_t* remote_bases,
    uint64_t x, uint64_t scales, uint32_t x_bytes, uint32_t scale_bytes,
    uint32_t tokens, uint32_t fields, uint32_t span_count, uint32_t count)
{
    const uint32_t i = threadIdx.x;
    if (i >= count)
        return;
    for (uint32_t s = 0; s < span_count; ++s) {
        auto* span = spans + s * 4U;
        const uint32_t begin = static_cast<uint32_t>(span[3]);
        const uint32_t length = static_cast<uint32_t>(span[1] >> 32U);
        if (i < begin || i - begin >= length)
            continue;
        const uint32_t peer = static_cast<uint32_t>(span[2]);
        const uint32_t position = static_cast<uint32_t>(span[2] >> 32U) + i - begin;
        const uint64_t row = uint64_t(static_cast<uint32_t>(span[1])) + position / fields;
        const int32_t token = reinterpret_cast<const __gm__ int32_t*>(span[0])[position / fields];
        if (token < 0 || static_cast<uint32_t>(token) >= tokens) {
            __asc_simt_vf::__trap();
            return;
        }
        const bool scale = position % fields != 0;
        const uint32_t bytes = scale ? scale_bytes : x_bytes;
        auto* output = remote_bases + peer * kOutputRecordWords;
        if (uint64_t(row) >= output[2]) {
            __asc_simt_vf::__trap();
            return;
        }
        auto* header = headers + (peer * 2U + uint32_t(i + 1U == count)) * 8U;
        auto* wqe = wqes + i * 8U;
        wqe[0] = header[0];
        wqe[1] = header[1];
        wqe[2] = header[2];
        wqe[3] = header[3];
        wqe[4] = header[4];
        wqe[5] = output[scale ? 1 : 0] + uint64_t(row) * bytes;
        wqe[6] = bytes;
        wqe[7] = (scale ? scales : x) + uint64_t(token) * bytes;
        return;
    }
    __asc_simt_vf::__trap();
}

template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1>
class DispatchSenderT : public SenderT<Capacity, Qps> {
    using Base = SenderT<Capacity, Qps>;
    using Base::scratch_; using Base::count_; using Base::peers_; using Base::qp_;
    using Base::active_state_; using Base::last_;
    static constexpr uint32_t kDispatchBasesOffset = kHeadersOffset + Capacity * Qps * 2U * kWqeBytes;
    static constexpr uint32_t kDispatchEpochOffset = kDispatchBasesOffset + Capacity * Qps * kOutputRecordBytes;
public:
    __aicore__ DispatchSenderT(
        const DispatchTilingData& td, uint32_t core, AscendC::LocalTensor<uint8_t> scratch, GM_ADDR workspace, GM_ADDR x,
        GM_ADDR scales, uint32_t peer_start = UINT32_MAX, uint32_t rotation = 0)
        : Base(td.rank, td.world_size, core,
              peer_start == UINT32_MAX ? td.udma_peers_per_worker : kMaxPeers,
              reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), workspace, peer_start, rotation, td.udma_peer_order, td.udma_ranks_per_group,
              peer_start == UINT32_MAX ? td.udma_workers_per_peer : 1U),
          td_(td),
          x_(reinterpret_cast<uint64_t>(x)),
          scales_(reinterpret_cast<uint64_t>(scales))
    {
        auto* bases = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset);
        if (!td.udma_output_control) {
            // Internal staging baseline. Native UDMA calls require the
            // mailbox; no public option can skip publication accidentally.
            for (uint32_t lane = 0; lane < Qps; ++lane)
                for (uint32_t i = 0; i < count_; ++i) {
                    const uint32_t p = lane * Capacity + i;
                    bases[p * kOutputRecordWords] = peers_[p].remote_base + td.x_buffer_offset;
                    bases[p * kOutputRecordWords + 1] = peers_[p].remote_base + td.x_scales_buffer_offset;
                    bases[p * kOutputRecordWords + 2] = UINT32_MAX;
                }
            return;
        }
        if (count_ == 0)
            return;
        AscendC::GlobalTensor<uint8_t> control;
        control.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(td.udma_output_control));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::DataCopyPad(scratch[kDispatchEpochOffset], control,
            AscendC::DataCopyExtParams{1, 32, 0, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                AscendC::DataCopyPad(scratch[kDispatchBasesOffset + p * kOutputRecordBytes],
                    control[OutputRecordOffset(peers_[p].peer)],
                    AscendC::DataCopyExtParams{1, kOutputRecordBytes, 0, 0, 0},
                    AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
        const uint64_t epoch = *reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchEpochOffset);
        const uint64_t x_bytes = uint64_t(td.hidden) * td.x_element_bytes;
        const uint64_t scale_bytes = td.has_x_scales ? uint64_t(td.num_scales) * td.scale_element_bytes : 0;
        for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                auto* out = bases + p * kOutputRecordWords;
                if (!epoch || out[kOutputEpochWord] != epoch || out[2] > UINT32_MAX ||
                    out[3] != x_bytes || out[4] != scale_bytes || (out[2] && !out[0]) ||
                    (out[2] && scale_bytes && !out[1]) ||
                    out[0] > UINT64_MAX - out[2] * x_bytes || out[1] > UINT64_MAX - out[2] * scale_bytes) {
                    AscendC::printf("dispatch output descriptor mismatch rank=%u peer=%u epoch=%llu remote=%llu\n",
                        td.rank, peers_[p].peer, static_cast<unsigned long long>(epoch),
                        static_cast<unsigned long long>(out[kOutputEpochWord]));
                    trap();
                }
            }
    }

    __aicore__ uint32_t PutRows(uint32_t p, const __gm__ int32_t* indices, uint32_t row, uint32_t field, uint32_t count)
    {
        p = Base::QueuePeer(p);
        if (rows_pending_ && active_state_ != peers_[p].state)
            Flush();
        if (!rows_pending_) {
            active_state_ = peers_[p].state;
            auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(active_state_);
            aclshmemi_udma_poll_cq_when_sq_overflow(state, peers_[p].slot, qp_, kBatchWqes);
            if (!aclshmemi_udma_sq_has_credit(state, kBatchWqes))
                trap();
        }
        const uint32_t n = count < kBatchWqes - rows_pending_ ? count : kBatchWqes - rows_pending_;
        auto* span = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + spans_ * 4U;
        span[0] = reinterpret_cast<uint64_t>(indices);
        span[1] = uint64_t(row) | (uint64_t(n) << 32U);
        span[2] = uint64_t(p) | (uint64_t(field) << 32U);
        span[3] = rows_pending_;
        ++spans_;
        rows_pending_ += n;
        last_ = p;
        if (rows_pending_ == kBatchWqes)
            Flush();
        return n;
    }

    __aicore__ void Flush()
    {
        if (!rows_pending_)
            return;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        asc_vf_call<BuildDispatchWqes>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset), x_, scales_,
            td_.hidden * td_.x_element_bytes, td_.num_scales * td_.scale_element_bytes,
            td_.num_tokens, td_.has_x_scales && td_.num_scales != 0 && td_.scale_element_bytes != 0 ? 2U : 1U, spans_,
            rows_pending_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        aclshmemi_udma_flush_aggregate_wqes(
            scratch_, peers_[last_].context, peers_[last_].slot, qp_, EVENT_ID5, rows_pending_, kWqeBytes);
        rows_pending_ = 0;
        spans_ = 0;
    }

    // One visit is at most udma_wqes_per_visit WQEs for one peer.
    // Baseline Qps=1 keeps its original batching, with no extra flush.
    __aicore__ void BeginVisit(uint32_t p)
    {
        if constexpr (Qps == 2) {
            Flush();
            Base::SelectQp(next_qp_[p]);
            next_qp_[p] ^= 1U;
        }
    }

    __aicore__ void Publish(GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t worker = 0)
    {
        Flush();
        Base::Publish(base, offset, rank, worker);
    }

private:
    const DispatchTilingData& td_;
    uint64_t x_, scales_;
    uint32_t rows_pending_ = 0, spans_ = 0;
    uint32_t next_qp_[Qps == 2 ? Capacity : 1]{};
};
using DispatchSender = DispatchSenderT<>;
#endif
} // namespace ascend_deepep::udma
