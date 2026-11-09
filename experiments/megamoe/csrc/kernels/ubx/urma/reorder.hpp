// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "reorder_plan.hpp"
#include "compute_contract.hpp"
#include "lazy_compute_routes.hpp"
#include "timer.hpp"

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
