#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ascend_deepep::ops {
// One host clock for Python entry/return, native stages and benchmark completion.
// Diagnostic only: no device synchronization, no hot-path output, calling-thread local.
enum HostStage {
    ApiEntry,
    NativeEntry,
    NotifySubmitBegin,
    NotifySubmitted,
    CountWaitBegin,
    TotalObserved,
    CountsObserved,
    OutputReady,
    CommSubmitBegin,
    CommSubmitted,
    NativeReturn,
    ApiReturn,
    Complete,
    Fused,
    HostPrep,
    DeferredPlan,
    MappedCounts,
    HostStageCount
};
using HostTimingRow = std::array<int64_t, HostStageCount>;
struct DispatchHostState {
    std::vector<HostTimingRow> rows;
    int64_t begin = 0;
};
inline DispatchHostState& GetDispatchHostState()
{
    static thread_local DispatchHostState state;
    return state;
}

inline int64_t HostNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void BeginDispatchHostTiming()
{
    auto& state = GetDispatchHostState();
    if (state.rows.size() >= 65536) throw std::runtime_error("drain dispatch host timing rows before more calls");
    state.begin = HostNowNs();
}

inline void MarkDispatchHostTiming(HostStage stage)
{
    auto& rows = GetDispatchHostState().rows;
    if (rows.empty()) throw std::runtime_error("no completed native dispatch timing row");
    rows.back()[stage] = HostNowNs();
}

inline std::vector<HostTimingRow> TakeDispatchHostTimings()
{
    std::vector<HostTimingRow> rows;
    auto& state = GetDispatchHostState();
    rows.swap(state.rows);
    state.begin = 0;
    return rows;
}

class DispatchHostTiming {
public:
    DispatchHostTiming() : enabled_(GetDispatchHostState().begin != 0)
    {
        auto& state = GetDispatchHostState();
        row_[ApiEntry] = state.begin;
        state.begin = 0;
        Mark(NativeEntry);
    }
    void Mark(HostStage stage)
    {
        if (enabled_) row_[stage] = HostNowNs();
    }
    void Value(HostStage stage, int64_t value)
    {
        if (enabled_) row_[stage] = value;
    }
    int64_t* TotalSlot()
    {
        return enabled_ ? &row_[TotalObserved] : nullptr;
    }
    void Finish()
    {
        if (enabled_) {
            Mark(NativeReturn);
            GetDispatchHostState().rows.push_back(row_);
        }
    }

private:
    bool enabled_;
    HostTimingRow row_{};
};
}  // namespace ascend_deepep::ops
