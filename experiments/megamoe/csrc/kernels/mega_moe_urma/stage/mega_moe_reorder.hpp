// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man

#pragma once

#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/catlass_gmm/segmented_row_layout.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_route_views.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_timer.hpp"

// GMM2 output placement and peer/CV reorder progress.

namespace MegaMoeUrma {
constexpr uint32_t kReorderRows = 128U;
constexpr uint32_t kReorderColumns = 256U;
constexpr uint32_t kReorderSlots = 2U;
constexpr uint32_t kReorderSlotBytes = kReorderRows * kReorderColumns * sizeof(uint16_t);
constexpr uint32_t kReorderUbBytes = kReorderSlots * kReorderSlotBytes;

struct ReorderSpan {
    bool self;
    uint32_t scratchRow;
    uint32_t rows;
    uint64_t outputRow;
};

#if defined(__NPU_DEVICE__)
#define URMA_REORDER_INLINE __aicore__ inline
#else
#define URMA_REORDER_INLINE inline
#endif

URMA_REORDER_INLINE constexpr ReorderSpan PlanReorderSpan(const RowSegment& segment, uint32_t rank, uint32_t sliceBegin,
                                                          uint32_t sliceRows)
{
    const auto part =
        UrmaMegaMoeImpl::GmmKernel::SegmentedRows::Intersect(segment.tileRow, segment.rows, sliceBegin, sliceRows);
    const bool self = segment.sourceRank == rank;
    return {self, part.tileRow, part.rows, (self ? segment.returnRow : segment.receiveRow) + part.segmentRow};
}

#undef URMA_REORDER_INLINE
}  // namespace MegaMoeUrma

namespace MegaMoeUrma {
static_assert(kReorderUbBytes == ActivationUbContract::inputEnd);

// The paired AIC already filled low128KiB via FIX. Consume it in place and
// drain MTE3 before the caller releases the single RX ownership credit.
__aicore__ inline void ReorderGmm2Tile(const Arguments& args, const Fragment& fragment, uint32_t fragmentIndex,
                                       uint32_t tile, Timer& timer)
{
    using namespace AscendC;
    const uint32_t h = args.config.hidden;
    const uint32_t n = tile * kReorderColumns;  // M-group never exceeds M256.
    const uint32_t columns = h - n < kReorderColumns ? h - n : kReorderColumns;
    GlobalTensor<bfloat16_t> remote, self;
    remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.gmm2Output));
    self.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.symmetric + args.config.returnYOffset));
    LocalTensor<bfloat16_t> source(TPosition::VECCALC, 0U, kComputeTileM * kReorderColumns);
    const uint64_t start = timer.TraceBegin();
    auto cursor = RuntimeSegments(args, fragment).MakeCursor();
    RowSegment segment{};
    while (cursor.Next(segment)) {
        const bool local = segment.sourceRank == args.config.rank;
        const uint64_t row = local ? segment.returnRow : segment.receiveRow;
        const DataCopyExtParams copy{static_cast<uint16_t>(segment.rows), columns * 2U,
                                     (kReorderColumns - columns) * 2U / 32U, (h - columns) * 2U, 0U};
        if (local)
            DataCopyPad(self[row * h + n], source[segment.tileRow * kReorderColumns], copy);
        else
            DataCopyPad(remote[row * h + n], source[segment.tileRow * kReorderColumns], copy);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    // Preserve the existing consumer contract: one count means one N tile
    // has reached its FINAL source-major or self-return layout, not scratch.
    auto* done = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + fragmentIndex * kFlagBytes);
    AtomicAdd(done, 1);
    TraceMeta meta;
    meta.fragment = fragmentIndex;
    meta.expert = fragment.expert;
    meta.row = fragment.rowBegin;
    meta.rows = fragment.rows;
    meta.tileM = 0U;
    meta.tileN = n;
    meta.columns = columns;
    meta.bytes = static_cast<uint64_t>(fragment.rows) * columns * 2U;
    timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
}

}  // namespace MegaMoeUrma

namespace MegaMoeUrma {

// Full-H rows with two 64KiB slots. Copy() extends the same pipeline across
// ready spans; Finish() certifies GM stores once before a PUT or UB handoff.
class PeerReorderPipeline {
public:
    __aicore__ inline void Init(const Arguments& args, const Chunk& chunk)
    {
        using namespace AscendC;
        rowBytes_ = args.config.hidden * sizeof(bfloat16_t);
        rowsPerSlot_ = kSlotBytes / rowBytes_;
        iteration_ = 0U;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.gmm2Contiguous));
        input_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
            chunk.peer == args.config.rank ? args.symmetric + args.config.returnYOffset : args.gmm2Output));
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void Copy(const PeerReorderSpan& span)
    {
        using namespace AscendC;
        for (uint32_t row = 0U; row < span.rows; row += rowsPerSlot_) {
            const uint32_t rows = span.rows - row < rowsPerSlot_ ? span.rows - row : rowsPerSlot_;
            const uint32_t slot = iteration_++ % 2U;
            const auto event = static_cast<TEventID>(slot);
            LocalTensor<uint8_t> buffer(TPosition::VECCALC, slot * kSlotBytes, kSlotBytes);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopyPad(buffer, input_[(span.sourceRow + row) * rowBytes_], {1U, rows * rowBytes_, 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            DataCopyPad(output_[(span.destinationRow + row) * rowBytes_], buffer, {1U, rows * rowBytes_, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
    }

    __aicore__ inline void Finish()
    {
        using namespace AscendC;
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        // Fence consumption of the final MTE3 credits before reuse or PUT.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

private:
    static constexpr uint32_t kSlotBytes = 64U * 1024U;
    static_assert(2U * kSlotBytes <= ActivationUbContract::inputEnd);
    AscendC::GlobalTensor<uint8_t> input_, output_;
    uint32_t rowBytes_, rowsPerSlot_, iteration_;
};

// Whole-ready-CV entry retained for the DMA contract checks. The communication
// loop uses ReorderReadyPeerSpans to prepare partial CVs before all groups finish.
__aicore__ inline void ReorderPeerChunk(const Arguments& args, uint32_t chunkIndex, const Chunk& chunk, Timer& timer)
{
    (void)chunkIndex;
    const uint64_t start = timer.TraceBegin();
    PeerReorderPipeline pipeline;
    pipeline.Init(args, chunk);
    RuntimePeerSpans cursor(args, chunk);
    PeerReorderSpan span{};
    uint32_t bit;
    while (cursor.Next(span, bit)) pipeline.Copy(span);
    pipeline.Finish();
    TraceMeta meta;
    meta.peer = chunk.peer;
    meta.cv = chunk.cv;
    meta.row = chunk.localRow;
    meta.rows = chunk.rows;
    meta.columns = args.config.hidden;
    // Reorder copies raw BF16 storage; bfloat16_t is not in AscendC on this SDK.
    meta.bytes = static_cast<uint64_t>(chunk.rows) * args.config.hidden * sizeof(uint16_t);
    timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
}
}  // namespace MegaMoeUrma

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
