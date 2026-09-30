// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
struct PeerScalePlan {
    GM_ADDR destinations;
    GM_ADDR input;
    GM_ADDR output;
    uint32_t address_stride;
};
__aicore__ inline void NativeFill(LocalTensor<int32_t> tensor, int32_t value, uint32_t count)
{
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
    for (uint32_t begin = 0; begin < count; begin += 8192U)
        Duplicate(tensor[begin], value, count - begin < 8192U ? count - begin : 8192U);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
}
#include "aggregate_transport.hpp"
#include "aggregate_packet.hpp"
namespace DeepepScaleAggregate {
__simt_vf__ __launch_bounds__(128) inline void Decode(__ubuf__ int32_t* ranks, __ubuf__ int32_t* slots,
                                                      __ubuf__ int32_t* indices, uint32_t n, uint32_t begin,
                                                      uint32_t stride)
{
    for (uint32_t i = threadIdx.x; i < n; i += 128U) {
        const uint32_t bits = uint32_t(ranks[i]);
        indices[i] = int32_t(begin + i);
        if (bits == 0x80000000U) {
            ranks[i] = -1;
            slots[i] = -1;
            continue;
        }
        const uint32_t flat = bits ^ (0U - (bits >> 31U));
        ranks[i] = int32_t(flat / stride);
        slots[i] = int32_t(flat % stride);
    }
    asc_threadfence();
}

// Bitwise local scatter only: no remote SIMT stores, no floating point conversion.
// Final rows are unique per valid route, even for repeated experts/tokens.
__simt_vf__ __launch_bounds__(128) inline void Expand(__ubuf__ uint32_t* packed, __ubuf__ int32_t* slots,
                                                      __gm__ uint32_t* output, uint32_t n, uint32_t words,
                                                      uint32_t packed_words, uint32_t rows, __ubuf__ uint32_t* errors)
{
    uint32_t bad = errors[threadIdx.x];
    for (uint32_t i = threadIdx.x; i < n * words; i += 128U) {
        const uint32_t row = i / words, column = i % words;
        const int32_t slot = slots[(row / 2U) * 128U + 112U + row % 2U];
        if (slot == -1) {
            continue;
        }
        if (slot < 0 || uint32_t(slot) >= rows) {
            if (!column) {
                ++bad;
            }
            continue;
        }
        output[uint64_t(slot) * words + column] = packed[(row / 2U) * 128U + (row % 2U) * 56U + column];
    }
    errors[threadIdx.x] = bad;
    asc_threadfence();
}

__aicore__ inline void WaitPack(AscendC::TEventID event = EVENT_ID2)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
}
__aicore__ inline void WaitReuse()
{
    DeepepWeightAggregate::WaitLocalPut();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
}
__aicore__ inline void LocalExpand(AscendC::LocalTensor<uint8_t> packed, AscendC::LocalTensor<int32_t> slots,
                                   AscendC::LocalTensor<uint32_t> errors, const DispatchTiling& t,
                                   const PeerScalePlan& plan, uint32_t n)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    asc_vf_call<Expand>(dim3(128), reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int32_t*>(slots.GetPhyAddr()),
                        reinterpret_cast<__gm__ uint32_t*>(plan.output), n, 224U / 4U, 224U / 4U, t.output_rows,
                        reinterpret_cast<__ubuf__ uint32_t*>(errors.GetPhyAddr()));
    DeepepWeightAggregate::WaitVector();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
}

}  // namespace DeepepScaleAggregate
// Scale-only pipeline with independently validated 512B packets.
// Hidden/QP/Notify protocols are unchanged.
namespace DispatchScalePipeline {
using ascend_deepep::scale_pipeline::Layout;
using DeepepScaleAggregate::LocalExpand;
using DeepepScaleAggregate::WaitPack;
using DeepepWeightAggregate::Clock;
using DeepepWeightAggregate::Put;
static_assert(ascend_deepep::scale_pipeline::kScratchBytes <= DispatchDeepep::kSlotSelectUbBytes);
static_assert(2048U * 8U <= DispatchDeepep::kLocalCopyTileBytes);
static_assert(2048U * 4U <= DispatchDeepep::kUdmaWqeScratchBytes);

struct Peer {
    uint32_t rank = 0, buffered = 0, sent = 0, entries = 0, face = 0;
    bool failed = false;
};

// Four MTE3->S tickets (IDs 0..3). EP128/256 share them across 8/16 UB faces,
// retiring an old ticket before rearming its ID. No NativeStore
// or WaitLocalPut may run while these tickets are live (they use MTE3_S ID3).
// Decode/Expand use different event directions; reuse waits also order MTE2.
template <bool Profile>
__aicore__ inline void Acquire(uint32_t buffer, uint32_t& pending, uint64_t* c)
{
    buffer %= ascend_deepep::scale_pipeline::kCompletionEvents;
    if (!(pending & (1U << buffer))) {
        return;
    }
    const uint64_t start = Clock<Profile>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(buffer));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
    const uint64_t elapsed = Clock<Profile>() - start;
    c[14] += elapsed;
    c[4] += elapsed;
    ++c[19];
    pending &= ~(1U << buffer);
}

template <bool Profile>
__aicore__ inline void Send(GM_ADDR workspace, const DispatchTiling& t, const PeerScalePlan& plan, const Layout& layout,
                            AscendC::TBuf<AscendC::QuePosition::VECCALC>& scratch,
                            AscendC::LocalTensor<uint32_t> errors, Peer& peer, uint32_t ordinal, uint32_t& pending,
                            uint64_t* c)
{
    if (!peer.buffered || peer.failed) {
        return;
    }
    const uint32_t count = (peer.buffered + 1U) & ~1U;
    if (uint64_t(peer.sent) + count > ((t.tokens * t.topk + 1U) / 2U * 2U)) {
        peer.failed = true;
        return;
    }
    const uint32_t buffer = ordinal * 2U + peer.face;
    const uint32_t offset = layout.packed_offset + buffer * layout.buffer_bytes;
    auto packed = scratch.GetWithOffset<uint8_t>(layout.slots_in_buffer, offset);
    auto slots = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, offset);
    uint64_t start = Clock<Profile>();
    WaitPack();
    for (uint32_t i = peer.buffered; i < count; ++i) {
        slots.SetValue(DispatchAggregatePacket::ScaleSlot(i), -1);
    }
    DispatchAggregatePacket::Stamp(packed, count / 2U, t.generation);
    c[3] += Clock<Profile>() - start;
    start = Clock<Profile>();
    if (peer.rank == t.rank) {
        LocalExpand(packed, slots, errors, t, plan, count);
        c[6] += Clock<Profile>() - start;
        c[10] += peer.buffered;
    } else {
        // Another peer may have armed this ID while this face was partly filled.
        if (2U * layout.peers > ascend_deepep::scale_pipeline::kCompletionEvents) {
            Acquire<Profile>(buffer, pending, c);
        }
        auto* region = workspace + t.inbox + uint64_t(t.rank) * t.inbox_stride;
        Put(region + 512U + uint64_t(peer.sent / 2U) * 512U, packed, count / 2U * 512U, peer.rank);
        // One transfer contains complete 512B packets, including their tail flags.
        const uint32_t event = buffer % ascend_deepep::scale_pipeline::kCompletionEvents;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(event));
        pending |= 1U << event;
        c[4] += Clock<Profile>() - start;
        c[7] += uint64_t(count / 2U) * 512U;
        c[8] += 1U;
        ++c[18];
    }
    peer.sent += count;
    peer.buffered = 0;
    peer.face ^= 1U;
}

template <bool Profile, typename WqeScratch>
__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerScalePlan& plan,
                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& scratch,
                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& local_scratch, WqeScratch& wqe_scratch,
                           DispatchDeepep::DeepepSecondaryState& /*trace_state*/)
{
    using ascend_deepep::scale_pipeline::Make;
    const auto layout = Make(t.world, 224U);
    if (!layout.batch) {
        aclshmemi_kernel_abort("invalid scale peer-double layout");
        return;
    }
    const uint32_t owner = AscendC::GetBlockIdx() - 32U;
    const uint32_t routes = t.tokens * t.topk;
    const uint32_t peers = owner < t.world ? (t.world - 1U - owner) / 32U + 1U : 0U;
    auto ranks = layout.aux_routes ? local_scratch.GetWithOffset<int32_t>(layout.tile, 0U)
                                   : scratch.GetWithOffset<int32_t>(layout.tile, 0U);
    auto slots = layout.aux_routes ? local_scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 4U)
                                   : scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 4U);
    auto indices = layout.aux_routes ? wqe_scratch.template GetWithOffset<int32_t>(layout.tile, 0U)
                                     : scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 8U);
    auto selected = scratch.GetWithOffset<int32_t>(layout.tile, layout.aux_routes ? 0U : layout.tile * 12U);
    AscendC::LocalTensor<uint8_t> mask =
        scratch.GetWithOffset<uint8_t>(layout.tile / 8U, layout.aux_routes ? layout.tile * 4U : layout.tile * 16U);
    auto headers = scratch.GetWithOffset<uint64_t>(layout.peers * 64U, layout.headers_offset);
    auto errors = scratch.GetWithOffset<uint32_t>(128U, layout.errors_offset);
    Peer state[ascend_deepep::scale_pipeline::kMaxPeers]{};
    uint32_t pending = 0U;
    // Same first 16 diagnostic words as serial. Extra 16..23: decoded tiles,
    // decoded routes, remote batches, buffer waits, peers, batch, tile, UB bytes.
    uint64_t c[24]{};
    c[0] = Clock<Profile>();
    c[20] = peers;
    c[21] = layout.batch;
    c[22] = layout.tile;
    c[23] = layout.bytes;
    for (uint32_t i = 0; i < 128U; ++i) {
        errors.SetValue(i, 0U);
    }
    auto storage =
        scratch.GetWithOffset<int32_t>((layout.headers_offset - layout.packed_offset) / 4U, layout.packed_offset);
    NativeFill(storage, 0, (layout.headers_offset - layout.packed_offset) / 4U);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
    for (uint32_t p = 0; p < peers; ++p) {
        state[p].rank = (owner + p * 32U + t.rank) % t.world;
    }

    // Tile-major: load/decode each route once for all peers, including cross-pod.
    // Partial batches and both UB faces persist across tile boundaries.
    const uint64_t tx_start = Clock<Profile>();
    {
        for (uint32_t begin = 0; begin < routes && peers; begin += layout.tile) {
            const uint32_t n = routes - begin < layout.tile ? routes - begin : layout.tile;
            const uint32_t padded = (n + 63U) & ~63U;
            uint64_t start = Clock<Profile>();
            NativeLoad(ranks, reinterpret_cast<__gm__ int32_t*>(plan.destinations) + begin, n);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<DeepepScaleAggregate::Decode>(
                dim3(128), reinterpret_cast<__ubuf__ int32_t*>(ranks.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(slots.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()), n, begin, plan.address_stride);
            DeepepWeightAggregate::WaitVector();
            for (uint32_t i = n; i < padded; ++i) {
                ranks.SetValue(i, -1);
            }
            ++c[16];
            c[17] += n;
            c[3] += Clock<Profile>() - start;
            for (uint32_t p = 0; p < peers; ++p) {
                auto& peer = state[p];
                if (peer.failed) {
                    continue;
                }
                start = Clock<Profile>();
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                AscendC::Compares(mask, ranks, int32_t(peer.rank), AscendC::CMPMODE::EQ, padded);
                AscendC::PipeBarrier<PIPE_V>();
                uint64_t count = 0;
                AscendC::GatherMask(selected, indices, mask.ReinterpretCast<uint32_t>(), true, n, {1U, 1U, 0U, 0U},
                                    count);
                DeepepWeightAggregate::WaitVector();
                c[3] += Clock<Profile>() - start;
                for (uint32_t i = 0; i < count && !peer.failed; ++i) {
                    const uint32_t buffer = p * 2U + peer.face;
                    if (!peer.buffered) {
                        Acquire<Profile>(buffer, pending, c);
                    }
                    const uint32_t offset = layout.packed_offset + buffer * layout.buffer_bytes;
                    auto packed = scratch.GetWithOffset<uint8_t>(layout.slots_in_buffer, offset);
                    auto packed_slots = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, offset);
                    start = Clock<Profile>();
                    const uint32_t route = uint32_t(selected.GetValue(i));
                    DispatchDeepep::IssueGmBytesToUb(packed[DispatchAggregatePacket::ScaleData(peer.buffered)],
                                                     plan.input + uint64_t(route / t.topk) * 224U, 224U);
                    // Read the slot from the shared decoded tile; no fifth tile array.
                    packed_slots.SetValue(DispatchAggregatePacket::ScaleSlot(peer.buffered),
                                          slots.GetValue(route - begin));
                    ++peer.buffered;
                    ++peer.entries;
                    c[3] += Clock<Profile>() - start;
                    if (peer.buffered == layout.batch) {
                        Send<Profile>(workspace, t, plan, layout, scratch, errors, peer, p, pending, c);
                    }
                }
            }
        }
        for (uint32_t p = 0; p < peers; ++p) {
            auto& peer = state[p];
            Send<Profile>(workspace, t, plan, layout, scratch, errors, peer, p, pending, c);
            auto header = headers[p * 64U];
            for (uint32_t i = 0; i < 64U; ++i) {
                header.SetValue(i, 0U);
            }
            header.SetValue(0U, t.generation);
            header.SetValue(63U, t.generation);
            header.SetValue(1U, peer.failed ? 0U : peer.sent);
            header.SetValue(2U, peer.failed ? 0U : peer.entries);
            header.SetValue(3U, peer.failed ? 1U : 0U);
            c[9] += peer.entries;
            c[12] += peer.failed ? 1U : 0U;
        }
        // Consume tickets before generic MTE helpers reuse event ID3. Headers
        // may overtake payload: receivers validate all packet flags themselves.
        WaitPack();
        for (uint32_t buffer = 0; buffer < 2U * peers; ++buffer) {
            Acquire<Profile>(buffer, pending, c);
        }
        uint64_t start = Clock<Profile>();
        for (uint32_t p = 0; p < peers; ++p) {
            if (state[p].rank == t.rank) {
                continue;
            }
            auto* region = workspace + t.inbox + uint64_t(t.rank) * t.inbox_stride;
            Put(reinterpret_cast<__gm__ uint64_t*>(region), headers[p * 64U], 64U, state[p].rank);
            c[7] += 512U;
            ++c[8];
        }
        aclshmemx_mte_quiet();
        DeepepScaleAggregate::WaitReuse();
        const uint64_t publish = Clock<Profile>() - start;
        c[13] += publish;
        c[4] += publish;
    }
    c[1] = Clock<Profile>();
    uint64_t start = 0;
    // All TX is complete: one released buffer can now service the original
    // source-major receive/expand loop. No new receiver dependency during TX.
    auto packed = scratch.GetWithOffset<uint8_t>(layout.slots_in_buffer, layout.packed_offset);
    auto packed_slots = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, layout.packed_offset);
    for (uint32_t source = owner; source < t.world; source += 32U) {
        if (source == t.rank) {
            continue;
        }
        auto* region = workspace + t.inbox + uint64_t(source) * t.inbox_stride;
        start = Clock<Profile>();
        DispatchAggregatePacket::Wait(region, 1U, t.generation, wqe_scratch);
        NativeLoadPacket(headers, reinterpret_cast<__gm__ uint64_t*>(region), 64U);
        const uint64_t count = headers.GetValue(1U), entries = headers.GetValue(2U);
        if (headers.GetValue(0U) != t.generation || headers.GetValue(3U) ||
            count > ((t.tokens * t.topk + 1U) / 2U * 2U) || entries > count || (count & 1U)) {
            ++c[12];
            continue;
        }
        DispatchAggregatePacket::Wait(region + 512U, uint32_t(count / 2U), t.generation, wqe_scratch);
        c[5] += Clock<Profile>() - start;
        c[10] += entries;
        start = Clock<Profile>();
        for (uint32_t begin = 0; begin < count; begin += layout.batch) {
            const uint32_t n = count - begin < layout.batch ? uint32_t(count - begin) : layout.batch;
            NativeLoadPacket(packed, region + 512U + uint64_t(begin / 2U) * 512U, n / 2U * 512U);
            LocalExpand(packed, packed_slots, errors, t, plan, n);
        }
        c[6] += Clock<Profile>() - start;
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    c[2] = Clock<Profile>();
    c[15] = 1U;
    for (uint32_t i = 0; i < 128U; ++i) {
        if (errors.GetValue(i)) {
            aclshmemi_kernel_abort("scale scatter row out of bounds\n");
            trap();
        }
    }
    if (c[12]) {
        aclshmemi_kernel_abort("scale inbox protocol invalid\n");
        trap();
    }
    if constexpr (Profile) {
        auto* record = reinterpret_cast<__gm__ uint64_t*>(t.profile + uint64_t(AscendC::GetBlockIdx()) * 256U);
        record[14] = c[1];
        record[15] = c[2];
        record[24] = c[5];
        record[25] = c[6];
        record[16] = tx_start;
        record[17] = c[1];
        record[18] = 0U;
        record[19] = 0U;
    }
}
}  // namespace DispatchScalePipeline
