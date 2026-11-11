// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernel_operator.h"

// Device timing records and nonblocking Dispatch receive observation.

namespace MegaMoeUrma {
enum Phase : uint32_t { Pack = 0, Dispatch, Gmm1, Activation, Gmm2, Combine, Barrier, ReducePhase };

// Each engine/phase owns a 128B record. Words 0..6 contain first start, last
// end, accumulated ticks, submitted bytes, interval count, timing kind and
// schema. Word 7 records outbound completion, or Dispatch receive observation
// for kind 2; words 8..9 hold received bytes/rows for kind 2. No cross-rank
// clock subtraction is valid.
struct NoopTimer {
    __aicore__ inline uint64_t TraceWaitBegin() const
    {
        return 0U;
    }
    __aicore__ inline uint64_t TraceBegin() const
    {
        return 0U;
    }
    __aicore__ inline void TraceEnd(TraceEvent, uint64_t, const TraceMeta& = TraceMeta{}) {}
    __aicore__ inline void Begin(Phase) {}
    __aicore__ inline void End(Phase, uint64_t = 0U) {}
};

enum class CoarsePhase : uint32_t {
    Pack,
    SyncStart,
    Gmm,
    Dequant,
    Dispatch,
    Combine,
    Activation,
    Reduce,
    SyncEnd,
    SharedSum,
    Route,
    RouteSummary,
    RouteWait,
    Count
};

class Timer {
public:
    __aicore__ inline bool DiagnosticEnabled() const
    {
        return address_ != 0U || traceAddress_ != 0U;
    }

    // Retain the successful observation in scalar state. Trace/aggregate GM
    // writes happen only at FinishTrace, outside the cooperative/Reduce loop.
    __aicore__ inline void RecordDispatchReceive(uint32_t peer, uint32_t cv, uint32_t rows, uint64_t bytes)
    {
        if (!DiagnosticEnabled()) return;
        dispatchTick_ = AscendC::GetSystemCycle();
        dispatchPeer_ = peer;
        dispatchCv_ = cv;
        dispatchRows_ = rows;
        dispatchBytes_ = bytes;
    }

    __aicore__ inline void SharedMode()
    {
        shared_ = true;
    }
    __aicore__ inline void Init(uint64_t address, uint64_t traceAddress = 0U, uint64_t traceCapacity = 0U,
                                bool coarse = false)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        coarse_ = coarse;
        address_ = coarse ? 0U : address;
        traceAddress_ = traceAddress;
        traceCapacity_ = traceCapacity;
        if constexpr (g_coreType == AscendC::AIV) {
            core_ = (AscendC::GetBlockIdx() / AscendC::GetTaskRation()) * 3U + AscendC::GetSubBlockIdx() + 1U;
        } else {
            core_ = AscendC::GetBlockIdx() * 3U;
        }
#else
        (void)address;
        (void)traceAddress;
        (void)traceCapacity;
        (void)coarse;
#endif
    }

    __aicore__ inline uint64_t TraceWaitBegin() const
    {
        return 0U;
    }
    __aicore__ inline uint64_t TraceBegin() const
    {
        return traceAddress_ == 0U || coarse_ ? 0U : AscendC::GetSystemCycle();
    }

    // This interval measures the instrumented call, not a synthesized DMA or
    // Cube hardware duration. Submission and existing completion waits are
    // separate events; recording never introduces an extra pipeline drain.
    __aicore__ inline void TraceEnd(TraceEvent event, uint64_t start, const TraceMeta& meta = TraceMeta{})
    {
        if (traceAddress_ == 0U || coarse_) return;
        const uint64_t end = AscendC::GetSystemCycle();
        WriteTrace(event, start, end, meta);
    }

    __aicore__ inline void WriteTrace(TraceEvent event, uint64_t start, uint64_t end, const TraceMeta& meta)
    {
        if (traceCount_ >= traceCapacity_) {
            ++traceDropped_;
            return;
        }
        auto* record = TraceHeader() + kTraceHeaderElements + traceCount_ * kTraceRecordElements;
        record[0] = start;
        record[1] = end;
        record[2] = static_cast<uint64_t>(event);
        record[3] = meta.peer;
        record[4] = meta.cv;
        record[5] = meta.fragment;
        record[6] = meta.expert;
        record[7] = meta.row;
        record[8] = meta.rows;
        record[9] = meta.tileM;
        record[10] = meta.tileN;
        record[11] = meta.columns;
        record[12] = meta.bytes;
        record[13] = 0U;
        record[14] = 0U;
        record[15] = 0U;
        FlushTraceLine(record);
        ++traceCount_;
    }

    __aicore__ inline void CoarseBegin(CoarsePhase phase)
    {
        if (!coarse_ || traceAddress_ == 0U) return;
        const uint32_t p = static_cast<uint32_t>(phase);
        if (coarseStart_[p] == 0U) coarseStart_[p] = AscendC::GetSystemCycle();
    }
    __aicore__ inline void CoarseEnd(CoarsePhase phase)
    {
        if (!coarse_ || traceAddress_ == 0U) return;
        const uint32_t p = static_cast<uint32_t>(phase);
        if (coarseStart_[p] != 0U) coarseEnd_[p] = AscendC::GetSystemCycle();
    }

    // Called once by every AIC/AIV engine, including engines with zero events.
    // An unfinished kernel leaves a zero header and is rejected by the reader.
    __aicore__ inline void FinishTrace()
    {
        if (dispatchTick_ != 0U && address_ != 0U) {
            auto* record = reinterpret_cast<__gm__ uint64_t*>(address_) +
                           (static_cast<uint64_t>(core_) * kTimerPhases + Phase::Dispatch) * kTimerRecordElements;
            // Timing kind2: outgoing first PUT -> incoming last Dispatch.
            // Empty receive edges have no end tick but remain explicitly known.
            record[5] = 2U;
            record[6] = kProtocol;
            record[7] = dispatchRows_ ? dispatchTick_ : 0U;
            record[8] = dispatchBytes_;
            record[9] = dispatchRows_;
            FlushTraceLine(record);
            FlushTraceLine(record + 8);
        }
        if (traceAddress_ == 0U) return;
        // All engines have passed the final local join. Emit only now, sorted
        // by start so the ordinary raw trace decoder keeps its ordering checks.
        if (coarse_) {
            constexpr uint32_t count = static_cast<uint32_t>(CoarsePhase::Count);
            for (uint32_t n = 0U; n < count; ++n) {
                uint32_t selected = count;
                for (uint32_t p = 0U; p < count; ++p)
                    if (coarseStart_[p] && (selected == count || coarseStart_[p] < coarseStart_[selected]))
                        selected = p;
                if (selected == count) break;
                TraceMeta meta;
                const auto event =
                    selected == static_cast<uint32_t>(CoarsePhase::RouteSummary) ? TraceEvent::RouteSummary
                    : selected == static_cast<uint32_t>(CoarsePhase::RouteWait)  ? TraceEvent::WaitRouteSummary
                    : selected == static_cast<uint32_t>(CoarsePhase::Route)      ? TraceEvent::RoutePrepare
                    : selected == static_cast<uint32_t>(CoarsePhase::SharedSum)  ? TraceEvent::SharedAdd
                    : shared_ && selected == static_cast<uint32_t>(CoarsePhase::Combine)
                        ? TraceEvent::CoarseSharedCombineSubmit
                    : selected == static_cast<uint32_t>(CoarsePhase::Dispatch)
                        ? TraceEvent::CoarseDispatchSubmit
                        : static_cast<TraceEvent>(static_cast<uint32_t>(TraceEvent::CoarsePack) + selected);
                WriteTrace(event, coarseStart_[selected], coarseEnd_[selected], meta);
                coarseStart_[selected] = 0U;
            }
        }
        // Observation may happen inside a Reduce/quiet envelope. Append this
        // independent point after ordinary scopes, keeping its original tick.
        // The decoder allows only this zero-duration ID65 point out of order.
        if (dispatchTick_ != 0U) {
            TraceMeta meta;
            meta.peer = dispatchPeer_;
            meta.cv = dispatchCv_;
            meta.rows = dispatchRows_;
            meta.bytes = dispatchBytes_;
            WriteTrace(TraceEvent::DispatchReceiveComplete, dispatchTick_, dispatchTick_, meta);
        }
        auto* header = TraceHeader();
        header[0] = kTraceSchema;
        header[1] = traceCapacity_;
        header[2] = traceCount_;
        header[3] = traceDropped_;
        header[4] = core_;
        FlushTraceLine(header);
    }

    __aicore__ inline void Begin(Phase phase)
    {
        if (address_ == 0U) return;
        start_[phase] = AscendC::GetSystemCycle();
        if (count_[phase] == 0U) first_[phase] = start_[phase];
    }

    __aicore__ inline void End(Phase phase, uint64_t bytes = 0U, bool submissionOnly = false)
    {
        if (address_ == 0U) return;
        const uint64_t end = AscendC::GetSystemCycle();
        total_[phase] += end - start_[phase];
        bytes_[phase] += bytes;
        ++count_[phase];
        auto* record = reinterpret_cast<__gm__ uint64_t*>(address_) +
                       (static_cast<uint64_t>(core_) * kTimerPhases + phase) * kTimerRecordElements;
        record[0] = first_[phase];
        record[1] = end;
        record[2] = total_[phase];
        record[3] = bytes_[phase];
        record[4] = count_[phase];
        record[5] = submissionOnly ? 1U : 0U;
        record[6] = kProtocol;
        AscendC::GlobalTensor<uint64_t> tensor;
        tensor.SetGlobalBuffer(record);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

    __aicore__ inline void MarkSendComplete(Phase phase)
    {
        if (address_ == 0U) return;
        auto* record = reinterpret_cast<__gm__ uint64_t*>(address_) +
                       (static_cast<uint64_t>(core_) * kTimerPhases + phase) * kTimerRecordElements;
        record[7] = AscendC::GetSystemCycle();
        FlushTraceLine(record);
    }

private:
    uint64_t dispatchTick_ = 0U;
    uint64_t dispatchBytes_ = 0U;
    uint32_t dispatchPeer_ = 0U, dispatchCv_ = 0U, dispatchRows_ = 0U;
    bool shared_ = false;
    __aicore__ inline __gm__ uint64_t* TraceHeader() const
    {
        return reinterpret_cast<__gm__ uint64_t*>(traceAddress_) +
               static_cast<uint64_t>(core_) * (kTraceHeaderElements + traceCapacity_ * kTraceRecordElements);
    }

    __aicore__ inline void FlushTraceLine(__gm__ uint64_t* record) const
    {
        AscendC::GlobalTensor<uint64_t> tensor;
        tensor.SetGlobalBuffer(record);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

    bool coarse_ = false;
    uint64_t coarseStart_[static_cast<uint32_t>(CoarsePhase::Count)]{};
    uint64_t coarseEnd_[static_cast<uint32_t>(CoarsePhase::Count)]{};
    uint64_t address_ = 0U;
    uint64_t traceAddress_ = 0U;
    uint64_t traceCapacity_ = 0U;
    uint64_t traceCount_ = 0U;
    uint64_t traceDropped_ = 0U;
    uint32_t core_ = 0U;
    uint64_t start_[kTimerPhases]{};
    uint64_t first_[kTimerPhases]{};
    uint64_t total_[kTimerPhases]{};
    uint64_t bytes_[kTimerPhases]{};
    uint64_t count_[kTimerPhases]{};
};
}  // namespace MegaMoeUrma

namespace MegaMoeUrma {
// Diagnostic-only observer of peer -> THIS rank. Every peer submits CVs in
// order on one QP; its final SO epoch certifies all prior X/scale PUTs to us.
// No CQ access, remote loads, credit updates, new signals or quiet.
class DispatchReceiveObserver {
public:
    __aicore__ inline void Init(Timer& timer)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (timer.DiagnosticEnabled()) timer_ = &timer;
#else
        (void)timer;
#endif
    }
    // Arm only after route summary is acquired and local Dispatch submission
    // is finished, when the owner enters its Combine phase. Empty incoming
    // edges are explicitly recorded, never polled as CV UINT_MAX.
    __aicore__ inline void Arm(const Arguments& a, uint32_t peer, uint32_t rows)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (!timer_) return;
        peer_ = peer;
        rows_ = rows;
        epoch_ = a.config.epoch;
        bytes_ = uint64_t(rows) * (a.config.hidden + a.config.hidden / 32U);
        if (rows == 0U) {
            timer_->RecordDispatchReceive(peer, 0U, 0U, 0U);
            return;
        }
        cv_ = (rows - 1U) / a.config.cvRows;
        flag_ = reinterpret_cast<__gm__ uint64_t*>(a.symmetric + a.config.dispatchReadyOffset +
                                                   (uint64_t(peer) * a.config.maxPeerChunks + cv_) * kFlagBytes);
        pending_ = true;
#else
        (void)a;
        (void)peer;
        (void)rows;
#endif
    }
    __aicore__ inline void Poll()
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (!pending_ || AscendC::ReadGmByPassDCache(flag_) != epoch_) return;
        timer_->RecordDispatchReceive(peer_, cv_, rows_, bytes_);
        pending_ = false;
#endif
    }

private:
#ifdef DEEPEP_MEGAMOE_TIMER
    Timer* timer_ = nullptr;
    __gm__ uint64_t* flag_ = nullptr;
    uint64_t epoch_ = 0U, bytes_ = 0U;
    uint32_t peer_ = 0U, cv_ = 0U, rows_ = 0U;
    bool pending_ = false;
#endif
};
}  // namespace MegaMoeUrma
