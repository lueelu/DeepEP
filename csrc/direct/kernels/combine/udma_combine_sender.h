#pragma once

#include "kernels/common/udma_shared_sender.h"
#include "common/platform.h"
#include "common/types.h"
#include "utils/debug/asc_assert.h"

namespace ascend_deepep::udma {
// Combine uses the request area for <=32 spans, each describing one contiguous
// run of rows/fields. Index windows remain live until the spans are flushed.
inline constexpr uint32_t kCombineBasesOffset = kSenderScratchBytes;
inline constexpr uint32_t kCombineSenderBytes = kCombineBasesOffset + kMaxPeers * sizeof(uint64_t);
static_assert(kCombineSenderBytes % 32 == 0);
static_assert(kCombineSenderBytes + kMaxPeers * kTileTokens * sizeof(int32_t) <= platform::kKernelUbBytes);

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
        auto* header = headers + peer * 2U * 8U;
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

// Ordinary contiguous input rows: indices have already been loaded by AIV to UB.
// Keep descriptor order and every WQE byte identical to the general builder.
template <bool Weighted>
__simt_vf__ __launch_bounds__(32) inline void BuildCombineWqesFast(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, __ubuf__ uint64_t* headers, __ubuf__ uint64_t* remote_bases,
    uint64_t x, uint64_t weights, uint64_t x_offset, uint64_t weights_offset, uint32_t x_bytes, uint32_t weight_bytes,
    uint32_t x_stride, uint32_t weight_stride, uint32_t token_capacity, uint32_t span_count, uint32_t count)
{
    const uint32_t i = threadIdx.x;
    if (i >= count) return;
    if (!span_count) { __asc_simt_vf::__trap(); return; }
    // Full single-span batches skip lookup. Short multi-span batches have
    // strictly increasing starts, so use a bounded binary search.
    uint32_t first = 0, last = span_count;
    while (first + 1U < last) {
        const uint32_t mid = first + (last - first) / 2U;
        if (static_cast<uint32_t>(spans[mid * 4U + 3U]) <= i) first = mid;
        else last = mid;
    }
    auto* span = spans + first * 4U;
    const uint32_t begin = static_cast<uint32_t>(span[3]);
    const uint32_t length = static_cast<uint32_t>(span[1] >> 32U);
    if (i < begin || i - begin >= length) { __asc_simt_vf::__trap(); return; }
    const uint32_t peer = static_cast<uint32_t>(span[2]);
    const uint32_t position = static_cast<uint32_t>(span[2] >> 32U) + i - begin;
    const uint32_t index = Weighted ? position >> 1U : position;
    const uint64_t row = uint64_t(static_cast<uint32_t>(span[1])) + index;
    const int32_t token = reinterpret_cast<__ubuf__ int32_t*>(span[0])[index];
    if (token < 0 || static_cast<uint32_t>(token) >= token_capacity) { __asc_simt_vf::__trap(); return; }
    const bool weight = Weighted && (position & 1U);
    auto* header = headers + peer * 2U * 8U;
    auto* wqe = wqes + i * 8U;
    wqe[0] = header[0];
    wqe[1] = header[1];
    wqe[2] = header[2];
    wqe[3] = header[3];
    wqe[4] = header[4];
    wqe[5] = remote_bases[peer] + (weight ? weights_offset : x_offset) +
             uint64_t(token) * (weight ? weight_stride : x_stride);
    wqe[6] = weight ? weight_bytes : x_bytes;
    wqe[7] = (weight ? weights : x) + row * (weight ? weight_bytes : x_bytes);
}

// One lane owns one logical dense row. Warp exchanges compact chunk heads into
// disjoint WQE slots. No lane reads the original token indices. All 32 lanes
// (including tails) participate in every warp operation. Span boundaries retain
// visit/peer/QP/batch and worker ownership.
__simt_vf__ __launch_bounds__(32) inline void BuildCombineWqesCoalesced(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, __ubuf__ uint64_t* headers,
    __ubuf__ uint64_t* remote_bases, __ubuf__ uint32_t* actual_count,
    uint64_t x, uint64_t x_offset, uint32_t x_bytes, uint32_t token_capacity,
    uint32_t span_count, uint32_t count, uint32_t max_rows)
{
    const uint32_t lane = threadIdx.x;
    if (!x_bytes || !max_rows || max_rows > 32U || count > 32U || span_count > count ||
        (count && !span_count)) {
        __asc_simt_vf::__trap(); return;
    }
    if (!count) {
        if (lane == 0) *actual_count = 0;
        return;
    }
    const bool active = lane < count;
    uint32_t first = 0, peer = 0, token = 0;
    uint64_t row = 0;
    bool invalid = false;
    if (active) {
        uint32_t last = span_count;
        while (first + 1U < last) {
            const uint32_t mid = first + (last - first) / 2U;
            if (static_cast<uint32_t>(spans[mid * 4U + 3U]) <= lane) first = mid;
            else last = mid;
        }
        auto* span = spans + first * 4U;
        const uint32_t begin = static_cast<uint32_t>(span[3]);
        const uint32_t length = static_cast<uint32_t>(span[1] >> 32U);
        invalid = span[2] >> 32U || begin > lane || length > count - begin || lane - begin >= length;
        if (!invalid) {
            const uint32_t pos = lane - begin;
            // Dense destination ordinal; no src_idx loads or token-contiguity
            // search. A span never crosses a peer's worker-owned input slice.
            const uint64_t selected = span[0] + pos;
            invalid = selected >= token_capacity;
            token = static_cast<uint32_t>(selected);
            row = uint64_t(static_cast<uint32_t>(span[1])) + pos;
            peer = static_cast<uint32_t>(span[2]);
        }
    }
    // Validate the producer's complete span partition in parallel, including
    // boundaries that happen to have consecutive token numbers on both sides.
    if (lane < span_count) {
        auto* span = spans + lane * 4U;
        const uint64_t begin = span[3], length = span[1] >> 32U;
        const uint64_t end = lane + 1U < span_count ? spans[(lane + 1U) * 4U + 3U] : count;
        invalid |= (!lane && begin != 0) || !length || begin + length != end || end > count;
    }
    if (asc_ballot(invalid)) { __asc_simt_vf::__trap(); return; }
    const uint32_t preceding_token = asc_shfl_up(token, 1, 32);
    const uint32_t preceding_span = asc_shfl_up(first, 1, 32);
    const bool run_head = active && (lane == 0 || first != preceding_span ||
                                    uint64_t(token) != uint64_t(preceding_token) + 1U);
    const uint32_t run_heads = asc_ballot(run_head);
    const uint32_t through_lane = 0xffffffffU >> (31U - lane);
    uint32_t preceding_heads = run_heads & through_lane, run_begin = 0;
    // Index of the nearest preceding run head, in five bounded register steps.
    for (uint32_t shift = 16U; shift; shift >>= 1U) {
        if (preceding_heads >> shift) { preceding_heads >>= shift; run_begin += shift; }
    }
    // Conservative 64-KiB cap, not a claim about the hardware's maximum length.
    const uint32_t byte_cap = x_bytes <= 65536U ? 65536U / x_bytes : 1U;
    const uint32_t limit = max_rows < byte_cap ? max_rows : byte_cap;
    const bool head = active && (lane - run_begin) % limit == 0;
    const uint32_t heads = asc_ballot(head);
    // Portable 32-bit population counts; no additional UB workspace or VF.
    uint32_t total = heads;
    total -= (total >> 1U) & 0x55555555U;
    total = (total & 0x33333333U) + ((total >> 2U) & 0x33333333U);
    total = ((total + (total >> 4U)) & 0x0f0f0f0fU) * 0x01010101U >> 24U;
    if (lane == 0) *actual_count = total;
    if (!head) return;  // No warp operations below this point.
    uint32_t output = heads & (through_lane >> 1U);
    output -= (output >> 1U) & 0x55555555U;
    output = (output & 0x33333333U) + ((output >> 2U) & 0x33333333U);
    output = ((output + (output >> 4U)) & 0x0f0f0f0fU) * 0x01010101U >> 24U;
    uint32_t following = heads & ~through_lane, end = count;
    if (following) {
        following &= 0U - following;  // Lowest following head bit.
        end = 0;
        for (uint32_t shift = 16U; shift; shift >>= 1U) {
            if (following >> shift) { following >>= shift; end += shift; }
        }
    }
    const uint32_t rows = end - lane;
    auto* header = headers + peer * 2U * 8U;
    auto* wqe = wqes + output * 8U;
    for (uint32_t h = 0; h < 5U; ++h) wqe[h] = header[h];
    wqe[5] = remote_bases[peer] + x_offset + uint64_t(token) * x_bytes;
    wqe[6] = uint64_t(rows) * x_bytes;
    wqe[7] = x + uint64_t(row) * x_bytes;
}

template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1, class Trace = DisabledDispatchDeviceTrace>
class CombineSenderT : public SenderT<Capacity, Qps, Trace> {
    using Base = SenderT<Capacity, Qps, Trace>;
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
    __aicore__ uint32_t PutRows(uint32_t p, __ubuf__ int32_t* indices, uint32_t row, uint32_t field, uint32_t count,
                              uint32_t peer_begin = 0)
    {
        p = Base::QueuePeer(p);
        if (rows_pending_ && active_state_ != peers_[p].state) {
            Base::TraceCount(Trace::StateFlushes);
            Flush();
        }
        if (!rows_pending_) {
            active_state_ = peers_[p].state;
            Base::Admit(p);
        }
        const uint64_t prepared = Base::TraceNow();
        const uint32_t n = count < kBatchWqes - rows_pending_ ? count : kBatchWqes - rows_pending_;
        auto* span = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + spans_ * 4U;
        span[0] = td_.combine_payload_max_rows > 1U ? uint64_t(row - peer_begin)
                                                  : reinterpret_cast<uint64_t>(indices);
        span[1] = uint64_t(row) | (uint64_t(n) << 32U);
        span[2] = uint64_t(p) | (uint64_t(field) << 32U);
        span[3] = rows_pending_;
        ++spans_;
        rows_pending_ += n;
        last_ = p;
        Base::TraceAdd(Trace::DescriptorCycles, prepared);
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
        const uint64_t built = Base::TraceNow();
        const uint64_t rank_slot = uint64_t(td_.rank) * td_.token_capacity;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        uint32_t submitted = rows_pending_;
        const bool coalesce = td_.combine_payload_max_rows > 1U && fields_ == 1U &&
            row_step_ == 1U && index_step_ == 1U && !td_.single_source_slots &&
            td_.x_row_stride == uint64_t(td_.hidden) * td_.x_element_bytes;
        if (coalesce) {
            auto* actual_count = reinterpret_cast<__ubuf__ uint32_t*>(
                scratch_ + kCombineBasesOffset + Capacity * Qps * sizeof(uint64_t));
            asc_vf_call<BuildCombineWqesCoalesced>(
                dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kCombineBasesOffset), actual_count,
                x_, td_.x_buffer_offset + rank_slot * td_.x_row_stride,
                td_.hidden * td_.x_element_bytes, td_.token_capacity, spans_, rows_pending_,
                td_.combine_payload_max_rows);
            // VF count is consumed by scalar SQ bookkeeping, not just MTE3.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            submitted = *actual_count;
            if (!submitted || submitted > rows_pending_) { trap(); return; }
        } else if (td_.combine_wqe_fast && row_step_ == 1U && index_step_ == 1U && !td_.single_source_slots) {
            if (fields_ == 2U) BuildFast<true>(rank_slot);
            else BuildFast<false>(rank_slot);
        } else {
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
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        Base::FinishWqeBuild(built);
        // Logical rows/visit accounting above is unchanged. SQ head/credits
        // and trace WQE counts must use the physical descriptor count.
        Base::SubmitBatch(submitted, spans_, false);
        rows_pending_ = 0;
        spans_ = 0;
    }

    __aicore__ void FlushForIndex() {
        if (rows_pending_) Base::TraceCount(Trace::IndexFlushes);
        Flush();
    }

    // Payload spans are drained before reusing the request area for a dense
    // weight descriptor. Base owns physical-SQ credit and sparse CQE tracking.
    __aicore__ void PutCompactWeights(uint32_t p, uint32_t row, uint32_t peer_begin, uint32_t rows)
    {
        Flush();
        const uint32_t row_bytes = td_.num_topk * uint32_t(sizeof(float));
        // Bound each descriptor to 64 KiB, with whole rows and no padding.
        const uint32_t chunk_rows = 65536U / row_bytes;
        while (rows) {
            const uint32_t count = rows < chunk_rows ? rows : chunk_rows;
            const uint64_t slot = uint64_t(td_.rank) * td_.token_capacity + row - peer_begin;
            Base::PutAbsolute(p, peers_[Base::QueuePeer(p)].remote_base + td_.topk_weights_buffer_offset +
                                   slot * row_bytes,
                              reinterpret_cast<__gm__ uint8_t*>(weights_ + uint64_t(row) * row_bytes),
                              count * row_bytes);
            row += count;
            rows -= count;
        }
        // No outstanding base descriptors when the next visit changes QP or
        // the final SO markers reuse scratch. This does not wait for payload.
        Base::Flush();
    }

    // One visit is at most udma_wqes_per_visit WQEs for one peer.
    // Baseline Qps=1 keeps its original batching, with no extra flush.
    __aicore__ void BeginVisit(uint32_t p)
    {
        if constexpr (Qps == 2) {
            if (rows_pending_) Base::TraceCount(Trace::QpFlushes);
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
    template <bool Weighted>
    __aicore__ void BuildFast(uint64_t rank_slot)
    {
        asc_vf_call<BuildCombineWqesFast<Weighted>>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kCombineBasesOffset), x_, weights_,
            td_.x_buffer_offset + rank_slot * td_.x_row_stride,
            td_.topk_weights_buffer_offset + rank_slot * td_.topk_weights_row_stride,
            td_.hidden * td_.x_element_bytes, td_.num_topk * uint32_t(sizeof(float)),
            td_.x_row_stride, td_.topk_weights_row_stride, td_.token_capacity, spans_, rows_pending_);
    }

    const CombineTilingData& td_;
    uint64_t x_, weights_;
    uint32_t row_step_, index_step_, fields_;
    uint32_t rows_pending_ = 0, spans_ = 0;
    uint32_t next_qp_[Qps == 2 ? Capacity : 1]{};
};
using CombineSender = CombineSenderT<>;
#endif
} // namespace ascend_deepep::udma
