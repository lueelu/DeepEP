#pragma once

#include "common/types.h"
#include "kernel_operator.h"

namespace ascend_deepep::udma {
// One row per AIV, 512-byte stride, no shared writers. Timestamps are offsets
// from this AIV's entry; do not compare absolute clocks across devices.
struct DispatchTraceFields {
    enum Field {
        Version, Rank, Core, Cached, PayloadPeers, WholeChannel, SrcThreads, TopkThreads,
        Entry, InitDone, BeginDone, MetadataDone, PayloadQueued, Published,
        IncomingDone, QuietDone, SyncDone, OutputDone, HandleDone, WorkDone,
        SrcCycles, TopkCycles, SelfCycles, ListCycles, ListWindows, PutRowsCalls,
        DescriptorCycles, WqeBuildCycles, WqeSubmitCycles, CreditCycles, CreditBlockedCycles,
        CreditChecks, CreditBlocked, CreditMaxCycles, SqUsedMax, SqUsedSum, SqEmptyChecks,
        Wqes, Flushes, ShortFlushes, Spans, StateFlushes, QpFlushes, IndexFlushes,
        MetadataWqes, MetadataFlushes, SubmitNoCredit, SelfBegin, SelfEnd, Operation, IncomingBegin, TopkInitCycles, SourceCycles, SlotsCycles, SrcOutputBytes, TopkOutputBytes, WeightOutputBytes,
        CqeRequested, CreditCheckpoints, CreditQuietCalls, FinalQuietCycles, FinalQuietCalls, SqWraps,
        Fields
    };
};
struct DispatchDeviceTrace : DispatchTraceFields {
    static constexpr bool enabled = true;
    uint64_t address;
    uint64_t values[64]{};
    static_assert(Fields <= 64);

    __aicore__ explicit DispatchDeviceTrace(uint64_t output) : address(output) {}
    __aicore__ uint64_t Now() const { return address ? uint64_t(AscendC::GetSystemCycle()) : 0; }
    __aicore__ void Set(Field field, uint64_t value) { if (address) values[field] = value; }
    __aicore__ void Mark(Field field) { if (address) values[field] = Now() - values[Entry]; }
    __aicore__ void Add(Field field, uint64_t begin) { if (address) values[field] += Now() - begin; }
    __aicore__ void Count(Field field) { if (address) ++values[field]; }
    __aicore__ void Accumulate(Field field, uint64_t value) { if (address) values[field] += value; }
    __aicore__ void Maximum(Field field, uint64_t value) {
        if (address && value > values[field]) values[field] = value;
    }
    __aicore__ void SelfDone(uint64_t begin) {
        if (!address) return;
        const uint64_t end = Now();
        if (!values[SelfBegin]) values[SelfBegin] = begin - values[Entry];
        values[SelfEnd] = end - values[Entry];
        values[SelfCycles] += end - begin;
    }
    __aicore__ void FinishWqeBuild(uint64_t begin) {
        // V_MTE3 only orders MTE3; the scalar clock needs V_S to measure completed VF work.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
        Add(WqeBuildCycles, begin);
    }
    __aicore__ void FinishOutput()
    {
        // Existing MTE3_MTE2 drains protect UB reuse, not the scalar timer.
        // This local diagnostic-only edge measures completed output DMA.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        Mark(OutputDone);
    }

    template <class Resource>
    __aicore__ void Save(Resource& resource, uint32_t core)
    {
        if (!address) return;
        // All original readers have finished. Reuse UB [0,512) only at exit.
        // MTE3_S protects the final output-copy source from scalar overwrite.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        Mark(WorkDone);
        auto scratch = resource.template GetBufferByByte<uint64_t>(0);
        for (uint32_t i = 0; i < 64; ++i) scratch.SetValue(i, values[i]);
        AscendC::GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address + uint64_t(core) * 512U));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
        AscendC::DataCopyExtParams copy{1, 512U, 0, 0, 0};
        AscendC::DataCopyPad(output, scratch, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
    }
};
// Compiled-out specialization: default launches carry no trace array or clock reads.
struct DisabledDispatchDeviceTrace : DispatchTraceFields {
    static constexpr bool enabled = false;
    __aicore__ explicit DisabledDispatchDeviceTrace(uint64_t) {}
    __aicore__ uint64_t Now() const { return 0; }
    __aicore__ void Set(Field, uint64_t) {}
    __aicore__ void Mark(Field) {}
    __aicore__ void Add(Field, uint64_t) {}
    __aicore__ void Count(Field) {}
    __aicore__ void Accumulate(Field, uint64_t) {}
    __aicore__ void Maximum(Field, uint64_t) {}
    __aicore__ void SelfDone(uint64_t) {}
    __aicore__ void FinishWqeBuild(uint64_t) {}
    __aicore__ void FinishOutput() {}
    template <class Resource>
    __aicore__ void Save(Resource&, uint32_t) {}
};
} // namespace ascend_deepep::udma
