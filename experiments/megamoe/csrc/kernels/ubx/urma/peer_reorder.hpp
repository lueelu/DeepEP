// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "lazy_peer_reorder.hpp"
#include "timer.hpp"

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
