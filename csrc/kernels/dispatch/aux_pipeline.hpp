// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
struct PeerAuxPlan {
    GM_ADDR destinations;
    GM_ADDR weights;
    GM_ADDR scales;
    GM_ADDR output_weights;
    GM_ADDR output_scales;
    uint32_t address_stride;
};
// Rear 32 AIVs: weight-only packets, independent of scale.
// Each 512B packet carries sixty slot+weight pairs and a generation tail.
namespace DispatchAuxPipeline {
using ascend_deepep::aux_pipeline::Layout;
using DeepepScaleAggregate::WaitPack;
using DeepepWeightAggregate::Clock;
using DeepepWeightAggregate::Put;
using DispatchScalePipeline::Acquire;
using DispatchScalePipeline::Peer;

template <bool Scale>
__simt_vf__ __launch_bounds__(128) inline void Expand(__ubuf__ uint32_t* packed, __ubuf__ int32_t* meta,
                                                      __gm__ uint32_t* scales, __gm__ uint32_t* weights, uint32_t n,
                                                      uint32_t words, uint32_t packed_words, uint32_t rows,
                                                      __ubuf__ uint32_t* errors)
{
    uint32_t bad = errors[threadIdx.x];
    if constexpr (Scale) {
        for (uint32_t i = threadIdx.x; i < n * words; i += 128U) {
            const uint32_t row = i / words, column = i % words;
            const int32_t slot = meta[(row / 60U) * 128U + (row % 60U) * 2U];
            if (slot < 0 || uint32_t(slot) >= rows) {
                continue;
            }
            scales[uint64_t(slot) * words + column] = packed[row * packed_words + column];
        }
    }
    for (uint32_t row = threadIdx.x; row < n; row += 128U) {
        const int32_t slot = meta[(row / 60U) * 128U + (row % 60U) * 2U];
        if (slot == -1) {
            continue;
        }
        if (slot < 0 || uint32_t(slot) >= rows) {
            ++bad;
            continue;
        }
        weights[slot] = uint32_t(meta[(row / 60U) * 128U + (row % 60U) * 2U + 1U]);  // preserve FP32 bits
    }
    errors[threadIdx.x] = bad;
    asc_threadfence();
}

template <bool Scale>
__aicore__ inline void LocalExpand(AscendC::LocalTensor<uint8_t> packed, AscendC::LocalTensor<int32_t> meta,
                                   AscendC::LocalTensor<uint32_t> errors, const DispatchTiling& t,
                                   const PeerAuxPlan& plan, uint32_t n)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    asc_vf_call<Expand<Scale>>(
        dim3(128), reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(meta.GetPhyAddr()), reinterpret_cast<__gm__ uint32_t*>(plan.output_scales),
        reinterpret_cast<__gm__ uint32_t*>(plan.output_weights), n, 224U / 4U, (Scale ? 224U : 0U) / 4U, t.output_rows,
        reinterpret_cast<__ubuf__ uint32_t*>(errors.GetPhyAddr()));
    DeepepWeightAggregate::WaitVector();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
}

template <bool Profile, bool Scale>
__aicore__ inline void Send(GM_ADDR region, uint32_t capacity, uint32_t stride, const DispatchTiling& t,
                            const PeerAuxPlan& plan, const Layout& layout,
                            AscendC::TBuf<AscendC::QuePosition::VECCALC>& scratch,
                            AscendC::LocalTensor<uint32_t> errors, Peer& peer, uint32_t ordinal, uint32_t& pending,
                            uint64_t* c)
{
    if (!peer.buffered || peer.failed) {
        return;
    }
    const uint32_t count = (peer.buffered + 59U) / 60U * 60U;
    if (uint64_t(peer.sent) + count > capacity) {
        peer.failed = true;
        return;
    }
    const uint32_t buffer = ordinal * 2U + peer.face;
    const uint32_t offset = layout.packed_offset + buffer * layout.buffer_bytes;
    // Metadata and block-tail generations share one framed weight buffer.
    auto packed = scratch.GetWithOffset<uint8_t>(layout.buffer_bytes, offset);
    auto meta = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, offset);
    uint64_t start = Clock<Profile>();
    WaitPack();
    for (uint32_t i = peer.buffered; i < count; ++i) {
        meta.SetValue(DispatchAggregatePacket::WeightMeta(i), -1);
        meta.SetValue(DispatchAggregatePacket::WeightMeta(i) + 1U, 0);
    }
    DispatchAggregatePacket::Stamp(packed, count / 60U, t.generation);
    c[3] += Clock<Profile>() - start;
    start = Clock<Profile>();
    if (peer.rank == t.rank) {
        LocalExpand<Scale>(packed, meta, errors, t, plan, count);
        c[6] += Clock<Profile>() - start;
        c[10] += peer.buffered;
    } else {
        // Weight packets use the same four local completion tickets as scale.
        if (2U * layout.peers > ascend_deepep::scale_pipeline::kCompletionEvents) {
            Acquire<Profile>(buffer, pending, c);
        }
        Put(region + 512U + uint64_t(peer.sent / 60U) * 512U, packed, count / 60U * 512U, peer.rank);
        const uint32_t event = buffer % ascend_deepep::scale_pipeline::kCompletionEvents;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(event));
        pending |= 1U << event;
        c[4] += Clock<Profile>() - start;
        c[7] += uint64_t(count / 60U) * 512U;
        c[8] += 1U;
        ++c[18];
    }
    peer.sent += count;
    peer.buffered = 0;
    peer.face ^= 1U;
}

template <bool Profile, bool Scale, typename WqeScratch>
__aicore__ inline void Run(GM_ADDR workspace, const DispatchTiling& t, const PeerAuxPlan& plan,
                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& scratch,
                           AscendC::TBuf<AscendC::QuePosition::VECCALC>& local_scratch, WqeScratch& wqe_scratch,
                           DispatchDeepep::DeepepSecondaryState& trace_state)
{
    static_assert(!Scale, "weights and scales must use independent packet transports");
    const uint32_t stride = 0U;
    const auto layout = ascend_deepep::aux_pipeline::Make(t.world, stride);
    if (!layout.batch || !plan.weights || (t.output_rows && !plan.output_weights) ||
        (Scale && (!plan.scales || (t.output_rows && !plan.output_scales)))) {
        aclshmemi_kernel_abort("invalid auxiliary pipeline layout");
        return;
    }
    const uint32_t owner = AscendC::GetBlockIdx() - 32U;
    const uint32_t peers = owner < t.world ? (t.world - 1U - owner) / 32U + 1U : 0U;
    const uint32_t routes = t.tokens * t.topk;
    const uint32_t capacity = t.weight_capacity;
    const uint64_t inbox = t.weight_inbox;
    const uint64_t region_stride = t.weight_inbox_stride;
    auto* tx_region = workspace + inbox + uint64_t(t.rank) * region_stride;
    auto ranks = layout.aux_routes ? local_scratch.GetWithOffset<int32_t>(layout.tile, 0U)
                                   : scratch.GetWithOffset<int32_t>(layout.tile, 0U);
    auto slots = layout.aux_routes ? local_scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 4U)
                                   : scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 4U);
    auto indices = layout.aux_routes ? wqe_scratch.template GetWithOffset<int32_t>(layout.tile, 0U)
                                     : scratch.GetWithOffset<int32_t>(layout.tile, layout.tile * 8U);
    auto selected = scratch.GetWithOffset<int32_t>(layout.tile, layout.aux_routes ? 0U : layout.tile * 12U);
    auto mask =
        scratch.GetWithOffset<uint8_t>(layout.tile / 8U, layout.aux_routes ? layout.tile * 4U : layout.tile * 16U);
    auto weights = layout.aux_routes ? scratch.GetWithOffset<int32_t>(layout.tile, layout.weights_offset)
                                     : local_scratch.GetWithOffset<int32_t>(layout.tile, 0U);
    auto headers = scratch.GetWithOffset<uint64_t>(peers * 64U, layout.headers_offset);
    auto errors = scratch.GetWithOffset<uint32_t>(128U, layout.errors_offset);
    Peer state[ascend_deepep::scale_pipeline::kMaxPeers]{};
    uint32_t pending = 0U;
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
    // One GM route/weight load per tile, independent of the number of peers.
    for (uint32_t begin = 0; begin < routes; begin += layout.tile) {
        const uint32_t n = routes - begin < layout.tile ? routes - begin : layout.tile;
        const uint32_t padded = (n + 63U) & ~63U;
        uint64_t start = Clock<Profile>();
        NativeLoad(ranks, reinterpret_cast<__gm__ int32_t*>(plan.destinations) + begin, n);
        NativeLoad(weights, reinterpret_cast<__gm__ int32_t*>(plan.weights) + begin, n);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        asc_vf_call<DeepepScaleAggregate::Decode>(dim3(128), reinterpret_cast<__ubuf__ int32_t*>(ranks.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ int32_t*>(slots.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()), n, begin,
                                                  plan.address_stride);
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
            AscendC::GatherMask(selected, indices, mask.ReinterpretCast<uint32_t>(), true, n, {1U, 1U, 0U, 0U}, count);
            DeepepWeightAggregate::WaitVector();
            c[3] += Clock<Profile>() - start;
            for (uint32_t i = 0; i < count && !peer.failed; ++i) {
                const uint32_t buffer = p * 2U + peer.face;
                if (!peer.buffered) {
                    Acquire<Profile>(buffer, pending, c);
                }
                const uint32_t offset = layout.packed_offset + buffer * layout.buffer_bytes;
                auto packed = scratch.GetWithOffset<uint8_t>(layout.buffer_bytes, offset);
                auto meta = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, offset);
                start = Clock<Profile>();
                const uint32_t route = uint32_t(selected.GetValue(i));
                if constexpr (Scale) {
                    DispatchDeepep::IssueGmBytesToUb(packed[peer.buffered * stride],
                                                     plan.scales + uint64_t(route / t.topk) * 224U, 224U);
                }
                meta.SetValue(DispatchAggregatePacket::WeightMeta(peer.buffered), slots.GetValue(route - begin));
                meta.SetValue(DispatchAggregatePacket::WeightMeta(peer.buffered) + 1U, weights.GetValue(route - begin));
                ++peer.buffered;
                ++peer.entries;
                c[3] += Clock<Profile>() - start;
                if (peer.buffered == layout.batch) {
                    Send<Profile, Scale>(tx_region, capacity, stride, t, plan, layout, scratch, errors, peer, p,
                                         pending, c);
                }
            }
        }
    }
    for (uint32_t p = 0; p < peers; ++p) {
        auto& peer = state[p];
        Send<Profile, Scale>(tx_region, capacity, stride, t, plan, layout, scratch, errors, peer, p, pending, c);
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
    WaitPack();
    for (uint32_t buffer = 0; buffer < 2U * peers; ++buffer) {
        Acquire<Profile>(buffer, pending, c);
    }
    uint64_t start = Clock<Profile>();
    // Packet flags gate remote consumption; no payload-before-header quiet.
    for (uint32_t p = 0; p < peers; ++p) {
        if (state[p].rank == t.rank) continue;
        Put(reinterpret_cast<__gm__ uint64_t*>(tx_region), headers[p * 64U], 64U, state[p].rank);
        c[7] += 512U;
        ++c[8];
    }
    aclshmemx_mte_quiet();
    DeepepScaleAggregate::WaitReuse();
    c[13] = Clock<Profile>() - start;
    c[4] += c[13];
    c[1] = Clock<Profile>();
    auto packed = scratch.GetWithOffset<uint8_t>(layout.buffer_bytes, layout.packed_offset);
    auto meta = scratch.GetWithOffset<int32_t>(layout.buffer_bytes / 4U, layout.packed_offset);
    for (uint32_t source = owner; source < t.world; source += 32U) {
        if (source == t.rank) {
            continue;
        }
        auto* region = workspace + inbox + uint64_t(source) * region_stride;
        start = Clock<Profile>();
        DispatchAggregatePacket::Wait(region, 1U, t.generation, wqe_scratch);
        NativeLoadPacket(headers, reinterpret_cast<__gm__ uint64_t*>(region), 64U);
        const uint64_t count = headers.GetValue(1U), entries = headers.GetValue(2U);
        if (headers.GetValue(0U) != t.generation || headers.GetValue(3U) || count > capacity || entries > count ||
            (count % 60U)) {
            ++c[12];
            continue;
        }
        DispatchAggregatePacket::Wait(region + 512U, uint32_t(count / 60U), t.generation, wqe_scratch);
        c[5] += Clock<Profile>() - start;
        c[10] += entries;
        start = Clock<Profile>();
        for (uint32_t begin = 0; begin < count; begin += layout.batch) {
            const uint32_t n = count - begin < layout.batch ? uint32_t(count - begin) : layout.batch;
            NativeLoadPacket(packed, region + 512U + uint64_t(begin / 60U) * 512U, n / 60U * 512U);
            LocalExpand<Scale>(packed, meta, errors, t, plan, n);
        }
        c[6] += Clock<Profile>() - start;
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    c[2] = Clock<Profile>();
    c[15] = 1U;
    for (uint32_t i = 0; i < 128U; ++i) {
        if (errors.GetValue(i)) {
            aclshmemi_kernel_abort("auxiliary scatter out of bounds");
            trap();
        }
    }
    if (c[12]) {
        aclshmemi_kernel_abort("auxiliary inbox protocol invalid");
        trap();
    }
    if constexpr (Profile) {
        if (t.weight_profile) {
            auto* diag = reinterpret_cast<__gm__ uint64_t*>(t.weight_profile) + owner * 16U;
            auto result = scratch.Get<uint64_t>();
            for (uint32_t i = 0; i < 16U; ++i) result.SetValue(i, c[i]);
            NativeStore(diag, result, 16U);
        }
        // Version 3: one shared weight(+scale) interval, never double counted.
        if (t.profile) {
            auto* record = reinterpret_cast<__gm__ uint64_t*>(t.profile) + (owner + 32U) * 32U;
            record[14] = c[1];
            record[15] = c[2];
            record[24] = c[5];
            record[25] = c[6];
        }
    }
}
}  // namespace DispatchAuxPipeline
