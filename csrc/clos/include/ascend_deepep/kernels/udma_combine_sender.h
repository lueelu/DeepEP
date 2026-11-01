#pragma once

#include "ascend_deepep/kernels/udma_shared_sender.h"
#include "ascend_deepep/kernels/types.h"
#include "utils/debug/asc_assert.h"

namespace ascend_deepep::udma {
// Combine uses the request area for <=32 spans, each describing one contiguous
// run of rows/fields. Index windows remain live until the spans are flushed.
inline constexpr uint32_t kCombineBasesOffset = kSenderScratchBytes;
inline constexpr uint32_t kCombineSenderBytes = kCombineBasesOffset + kMaxPeers * sizeof(uint64_t);
static_assert(kCombineSenderBytes % 32 == 0);
static_assert(kCombineSenderBytes + kMaxPeers * kTileTokens * sizeof(int32_t) <= 216 * 1024);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__simt_vf__ __launch_bounds__(32) inline void BuildCombineWqes(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, __ubuf__ uint64_t* headers, __ubuf__ uint64_t* remote_bases,
    uint64_t x, uint64_t weights, uint64_t x_offset, uint64_t weights_offset, uint32_t x_bytes, uint32_t weight_bytes,
    uint32_t x_stride, uint32_t weight_stride, uint32_t token_capacity, uint32_t fields, uint32_t span_count,
    uint32_t count, uint32_t row_step, uint32_t index_step, uint64_t expanded_x,
    uint64_t single_slots, uint32_t expanded_count)
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
        const uint32_t row = static_cast<uint32_t>(span[1]) + position / fields * row_step;
        const int32_t token = reinterpret_cast<__ubuf__ int32_t*>(span[0])[position / fields * index_step];
        if (token < 0 || static_cast<uint32_t>(token) >= token_capacity) {
            __asc_simt_vf::__trap();
            return;
        }
        const bool weight = position % fields != 0;
        auto* header = headers + (peer * 2U + uint32_t(i + 1U == count)) * 8U;
        auto* wqe = wqes + i * 8U;
        wqe[0] = header[0];
        wqe[1] = header[1];
        wqe[2] = header[2];
        wqe[3] = header[3];
        wqe[4] = header[4];
        wqe[5] = remote_bases[peer] + (weight ? weights_offset : x_offset) +
                 uint64_t(token) * (weight ? weight_stride : x_stride);
        wqe[6] = weight ? weight_bytes : x_bytes;
        uint64_t payload_base = x;
        uint32_t payload_row = row;
        if (!weight && single_slots) {
            const int32_t slot = reinterpret_cast<__gm__ int32_t*>(single_slots)[row];
            if (slot >= 0) {
                if (static_cast<uint32_t>(slot) >= expanded_count) {
                    __asc_simt_vf::__trap();
                    return;
                }
                payload_base = expanded_x;
                payload_row = static_cast<uint32_t>(slot);
            }
        }
        wqe[7] = weight ? weights + uint64_t(row) * weight_bytes :
                         payload_base + uint64_t(payload_row) * x_bytes;
        return;
    }
    __asc_simt_vf::__trap();
}

template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1>
class CombineSenderT : public SenderT<Capacity, Qps> {
    using Base = SenderT<Capacity, Qps>;
    using Base::scratch_; using Base::count_; using Base::peers_; using Base::qp_;
    using Base::active_state_; using Base::last_;
    static constexpr uint32_t kCombineBasesOffset = kHeadersOffset + Capacity * Qps * 2U * kWqeBytes;
public:
    __aicore__ CombineSenderT(
        const CombineTilingData& td, uint32_t core, __ubuf__ uint8_t* scratch, GM_ADDR workspace, GM_ADDR x,
        GM_ADDR weights, uint32_t peer_start = UINT32_MAX, uint32_t rotation = 0, uint32_t row_step = 1,
        uint32_t index_step = 0, bool send_weights = true)
        : Base(
              td.rank, td.world_size, core, peer_start == UINT32_MAX ? td.udma_peers_per_worker : kMaxPeers, scratch,
              workspace, peer_start, rotation, td.udma_peer_order, td.udma_ranks_per_group,
              peer_start == UINT32_MAX ? td.udma_workers_per_peer : 1U),
          td_(td),
          x_(reinterpret_cast<uint64_t>(x)),
          weights_(reinterpret_cast<uint64_t>(weights)),
          row_step_(row_step),
          index_step_(index_step ? index_step : row_step),
          fields_(td.has_topk_weights && send_weights ? 2U : 1U)
    {
        auto* bases = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kCombineBasesOffset);
        for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                bases[p] = peers_[p].remote_base;
            }
    }

    // Returns the WQEs admitted, which can end in the middle of an x/weight pair.
    // The caller retains the remainder of this peer's visit budget across tiles.
    __aicore__ uint32_t PutRows(uint32_t p, __ubuf__ int32_t* indices, uint32_t row, uint32_t field, uint32_t count)
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

    // Phase 0 is the serial baseline; 1 submits single payloads before aggregate;
    // 2 submits the complement and all weight fields. The returned count is the
    // number of candidate fields consumed, including skipped fields. Thus peer
    // visits and token ownership remain identical to the original scheduler.
    template <uint32_t Phase>
    __aicore__ uint32_t PutSelectedRows(
        uint32_t p, __ubuf__ int32_t* indices, uint32_t row, uint32_t field, uint32_t count)
    {
        if constexpr (Phase == 0) {
            return PutRows(p, indices, row, field, count);
        } else {
            AscendC::GlobalTensor<int32_t> slots;
            slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(td_.single_source_slots));
            const uint32_t fields = fields_;
            uint32_t consumed = 0;
            while (consumed < count) {
                const uint32_t pos = field + consumed;
                const uint32_t selected_row = row + pos / fields * row_step_;
                const bool weight = pos % fields != 0;
                const bool single = !weight && slots.GetValue(selected_row) >= 0;
                if ((Phase == 1) == single) {
                    // A one-field span can still share a batched SQ publication
                    // with other peers/rows. SIMT retains the source validation.
                    PutRows(p, indices + pos / fields * index_step_, selected_row, pos % fields, 1U);
                }
                ++consumed;
            }
            return consumed;
        }
    }

    __aicore__ void Flush()
    {
        if (!rows_pending_)
            return;
        const uint64_t rank_slot = uint64_t(td_.rank) * td_.token_capacity;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        asc_vf_call<BuildCombineWqes>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kCombineBasesOffset), x_, weights_,
            td_.x_buffer_offset + rank_slot * td_.x_row_stride,
            td_.topk_weights_buffer_offset + rank_slot * td_.topk_weights_row_stride, td_.hidden * td_.x_element_bytes,
            td_.num_topk * uint32_t(sizeof(float)), td_.x_row_stride, td_.topk_weights_row_stride, td_.token_capacity,
            fields_, spans_, rows_pending_, row_step_, index_step_,
            td_.expanded_x, td_.single_source_slots, td_.num_expanded_tokens);
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
    const CombineTilingData& td_;
    uint64_t x_, weights_;
    uint32_t row_step_, index_step_, fields_;
    uint32_t rows_pending_ = 0, spans_ = 0;
    uint32_t next_qp_[Qps == 2 ? Capacity : 1]{};
};
using CombineSender = CombineSenderT<>;
#endif
} // namespace ascend_deepep::udma
