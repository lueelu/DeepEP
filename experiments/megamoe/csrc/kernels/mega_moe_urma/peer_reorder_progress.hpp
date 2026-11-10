// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_urma/peer_reorder.hpp"

namespace MegaMoeUrma {

constexpr uint32_t kPeerReorderBitmapWords = (kMaxPeerReorderSpans + 31U) / 32U;
constexpr uint32_t kPeerReorderBitmapBytes = kPeerReorderBitmapWords * sizeof(uint32_t);
constexpr uint32_t kReorderCvSubmitted = UINT32_MAX;
constexpr uint32_t kCombinePutRows = 128U;

__aicore__ inline PeerReorderRange LoadPeerReorderRange(const Arguments& args, uint32_t chunkIndex)
{
    return RuntimePeerReorderRange(args, chunkIndex);
}

// Each peer owns a first-row-addressed bitmap. Self owners have
// separate bitmaps and only touch their statically assigned CVs.
__aicore__ inline PeerReorderRange PeerReorderSpanRange(const Arguments& args, uint32_t begin, uint32_t end)
{
    if (begin == end) return {0U, 0U};
    const auto first = RuntimeReceiveChunkByIndex(args, begin);
    const uint32_t rows = RouteInfo(args)->incoming[first.peer][args.config.localExperts];
    return {0U, rows};  // sparse span bits use peer-local first row, not dense span ordinal
}

__aicore__ inline void ClearReorderBitmap(AscendC::LocalTensor<uint32_t> copied, uint32_t spans)
{
    const uint32_t words = (spans + 31U) / 32U;
    if (words == 0U) return;
    AscendC::Duplicate(copied, uint32_t{0}, words);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(static_cast<AscendC::TEventID>(0U));
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(static_cast<AscendC::TEventID>(0U));
}

// The lazy cursor partitions a CV in destination-row order. Bits are consumed
// only after ReorderReadyPeerSpans has drained its MTE3 writes. A count of
// ready rows is insufficient: an unready hole must never enter a payload PUT.
__aicore__ inline uint32_t ReorderedPeerPrefixRows(const Arguments& args, const PeerReorderRange& range,
                                                   uint32_t peerSpanBegin, AscendC::LocalTensor<uint32_t> copied)
{
    (void)peerSpanBegin;
    uint32_t rows = 0U, bit;
    RuntimePeerSpans cursor(args, RuntimeReceiveChunkByIndex(args, range.spanBegin));
    PeerReorderSpan span{};
    while (cursor.Next(span, bit)) {
        if ((copied.GetValue(bit / 32U) & (1U << (bit % 32U))) == 0U) break;
        rows += span.rows;
    }
    return rows;
}

// Single-writer UB bits skip spans already copied in earlier scans. Unready
// producer groups never block another span/CV. All groups have M<=256, so a
// full-width span needs ceil(H/256) completed N tiles regardless of its rows.
// Return the number of NEW spans whose local GM writes have completed.
__aicore__ inline uint32_t ReorderReadyPeerSpans(const Arguments& args, const PeerReorderRange& range,
                                                 const Chunk& chunk, uint32_t peerSpanBegin,
                                                 AscendC::LocalTensor<uint32_t> copied, Timer& timer, bool& waiting,
                                                 uint64_t waitStart, const TraceMeta& waitMeta)
{
    using namespace AscendC;
    const int32_t required = static_cast<int32_t>(ComputeTileCount(kComputeTileM, args.config.hidden));
    PeerReorderPipeline pipeline;
    uint32_t copiedSpans = 0U;
    uint32_t copiedRows = 0U;
    uint64_t start = 0U;
    uint64_t onlyFragment = kTraceUnknown;
    (void)range;
    (void)peerSpanBegin;
    RuntimePeerSpans cursor(args, chunk);
    PeerReorderSpan span{};
    uint32_t bitIndex;
    while (cursor.Next(span, bitIndex)) {
        const uint32_t mask = 1U << (bitIndex % 32U);
        const uint32_t word = copied.GetValue(bitIndex / 32U);
        if (word & mask) continue;
        const uint32_t group = span.fragmentIndex;
        auto* done = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + static_cast<uint64_t>(group) * kFlagBytes);
        if (ReadGmByPassDCache(done) < required) continue;
        if (copiedSpans == 0U) {
            if (waiting) {
                waiting = false;
            }
            start = timer.TraceBegin();
            pipeline.Init(args, chunk);
            onlyFragment = group;
        } else if (onlyFragment != group) {
            onlyFragment = kTraceUnknown;
        }
        pipeline.Copy(span);
        copied.SetValue(bitIndex / 32U, word | mask);
        copiedRows += span.rows;
        ++copiedSpans;
    }
    if (copiedSpans != 0U) {
        // One drain per ready batch, never per span. No PUT or self ready can
        // be published until these stores and all previous batches completed.
        pipeline.Finish();
        TraceMeta meta;
        meta.peer = chunk.peer;
        meta.cv = chunk.cv;
        meta.fragment = onlyFragment;
        // A batch may contain nonadjacent rows; do not claim a contiguous row.
        meta.rows = copiedRows;
        meta.columns = args.config.hidden;
        meta.bytes = static_cast<uint64_t>(copiedRows) * args.config.hidden * sizeof(bfloat16_t);
        timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
    }
    return copiedSpans;
}
}  // namespace MegaMoeUrma
