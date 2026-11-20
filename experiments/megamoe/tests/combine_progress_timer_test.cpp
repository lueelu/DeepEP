// Copyright (c) 2026, Lu Lu

// Host doubles exercise the real timer bookkeeping, not NPU/NIC visibility.
#include <cassert>
#include "kernel_operator.h"
#include "../csrc_ep32/kernels/mega_moe/common/ascend_timer_v2_device.hpp"

int main()
{
    using namespace AscendTimer;
    using namespace RoceTest;
    std::vector<int64_t> buffer(TOTAL_BUFFER_SIZE, 0);
    AscendTimerDevice timer;
    timer.Init(reinterpret_cast<uint8_t*>(buffer.data()));
    constexpr int core = 1;  // group 0 / AIV0
    auto logical = [](int type, int slot) { return FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER + slot; };
    auto at = [&](int timing, int end = 0) {
        return buffer[core * N_TIMING_COUNTER_PER_CORE_ALIGN + timing * 2 + end];
    };
    auto count = [&](int type) { return buffer[N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE_ALIGN + type]; };
    auto meta = [&](int type, int slot, int word) {
        return buffer[TASK_METADATA_OFFSET + ((core * DYNAMIC_TYPE_COUNT + type) * MAX_DYNAMIC_ITER + slot) * 2 + word];
    };
    auto wave = [&](int timing) { return buffer[WAVE_ID_OFFSET + core * WAVE_ID_PER_CORE_ALIGN + timing]; };

#ifdef DEEPEP_MEGAMOE_TIMER
    {
        const auto before = pipelineBarriers;
        timer.SetWave(0);
        int first = timer.BeginCombineRow(2, 16, 0, CombineRowPhase::Lifetime);
        timer.SetWave(1);
        int second = timer.BeginCombineRow(6, 3, 1, CombineRowPhase::Issue);
        int reuse = timer.BeginCombineRowAtWave(2, 16, 0, CombineRowPhase::ReuseWait, 0);
        timer.EndCombineRow(reuse);
        timer.EndCombineRow(second);
        timer.EndCombineRow(first);
        assert(wave(logical(AIV_COMBINE_ASYNC, first)) == 0);
        assert(wave(logical(AIV_COMBINE_ASYNC, second)) == 1);
        assert(wave(logical(AIV_COMBINE_ASYNC, reuse)) == 0);
        assert(meta(AIV_COMBINE_ASYNC, first, 0) == ((3LL << 32) | 16));
        assert(meta(AIV_COMBINE_ASYNC, second, 1) == ((2LL << 32) | 2));
        assert(pipelineBarriers == before);
    }
    // Independent handles preserve the live tile/wait and capture their own
    // scheduler wave even when SetWave changes before Finish.
    timer.SetWave(7);
    timer.SetSlice(3, 256, 32);
    assert(timer.StartTile(AIV_COMBINE, 0, static_cast<uint32_t>(-1), 32));
    assert(timer.BeginWait(AIV_COMBINE_WAIT));
    auto barriers = pipelineBarriers;
    {
        AscendTimerFixedScope init(&timer, AIV_COMBINE_INIT);
        cycle += 100;  // Initialization precedes progress and its scan segments.
    }
    {
        AscendTimerCombineProgressScope progress(&timer, AIV_COMBINE_PROGRESS);
        progress.StartScan();
        cycle += 10;
        auto savedCycle = cycle;
        progress.StartScan();  // Repeated starts/stops cannot double count.
        assert(cycle == savedCycle);
        progress.StopScan();  // 11 cycles including the closing timestamp read.
        savedCycle = cycle;
        progress.StopScan();
        assert(cycle == savedCycle);
        {
            AscendTimerRdmaScope rdma(&timer, AIV_RDMA_POST, RdmaPhase::Combine, 17, 4096, false);
            cycle += 200;  // Posting time must be excluded from scan/control.
        }
        cycle += 300;  // Backoff while paused is excluded as well.
        timer.SetWave(99);
        progress.StartScan();
        cycle += 7;
        progress.Finish(1024, 925, 3, 128);  // Closes the remaining 8-cycle segment.
        savedCycle = cycle;
        progress.Finish(0, 0, 0, 0);
        progress.StartScan();
        progress.StopScan();
        assert(cycle == savedCycle);
    }
    assert(pipelineBarriers == barriers);
    timer.EndWait();
    timer.EndTile();
    timer.FlushDynamicCounts();
    int progress = logical(AIV_COMBINE_PROGRESS, 0);
    assert(count(AIV_COMBINE_PROGRESS) == 1);
    assert(at(AIV_COMBINE_INIT) > 0 && at(AIV_COMBINE_INIT, 1) < at(progress));
    assert(at(progress) < at(logical(AIV_RDMA_POST, 0)) && at(logical(AIV_RDMA_POST, 0), 1) < at(progress, 1));
    assert(at(progress, 1) - at(progress) > 500 && meta(AIV_COMBINE_PROGRESS, 0, 1) == 19);
    assert(meta(AIV_COMBINE_PROGRESS, 0, 0) ==
           (int64_t(1024) | (int64_t(925) << 16) | (int64_t(3) << 32) | (int64_t(128) << 48)));
    assert(wave(progress) == 7);
    assert(wave(logical(AIV_RDMA_POST, 0)) == NO_WAVE);
    assert(wave(logical(AIV_COMBINE, 0)) == 7 && wave(logical(AIV_COMBINE_WAIT, 0)) == 7);
    assert(meta(AIV_COMBINE, 0, 0) == ((int64_t(4) << 32) | 256));
    assert(meta(AIV_COMBINE_WAIT, 0, 0) == meta(AIV_COMBINE, 0, 0));
    assert(at(logical(AIV_COMBINE_WAIT, 0), 1) == at(logical(AIV_COMBINE, 0)));
    assert(at(logical(AIV_COMBINE, 0), 1) > at(logical(AIV_COMBINE, 0)));

    // An enclosing span also survives progress without an unexpected split.
    timer.SetWave(8);
    timer.StartSpan(AIV_DISPATCH);
    {
        AscendTimerCombineProgressScope nested(&timer, AIV_COMBINE_PROGRESS);
        nested.StartScan();
        cycle += 3;
        nested.Finish(9, 8, 1, 4);
    }
    timer.EndSpan(AIV_DISPATCH);
    timer.FlushDynamicCounts();
    assert(count(AIV_DISPATCH) == 1 && at(logical(AIV_DISPATCH, 0), 1) > at(logical(AIV_DISPATCH, 0)));
    assert(wave(logical(AIV_DISPATCH, 0)) == 8 && wave(logical(AIV_COMBINE_PROGRESS, 1)) == 8);
    assert(meta(AIV_COMBINE_PROGRESS, 1, 1) == 4);

    // Drain has no scan counter and preserves a NO_WAVE stage-level record.
    timer.SetWave(NO_WAVE);
    {
        AscendTimerFixedScope hostBarrier(&timer, AIV_HOST_COMBINE_BARRIER_WAIT);
    }
    barriers = pipelineBarriers;
    {
        AscendTimerCombineProgressScope drain(&timer, AIV_COMBINE_DRAIN);
        auto savedCycle = cycle;
        drain.StartScan();
        drain.StopScan();
        assert(cycle == savedCycle);
        {
            AscendTimerRdmaScope quiet(&timer, AIV_RDMA_QUIET_WAIT, RdmaPhase::Combine, 17, 0, false);
        }
        drain.Finish(8, 0, 99, 888);  // Unused fields are forced to zero.
    }
    assert(pipelineBarriers == barriers);
    timer.FlushDynamicCounts();
    int drain = logical(AIV_COMBINE_DRAIN, 0);
    assert(count(AIV_COMBINE_DRAIN) == 1 && at(drain, 1) > at(drain));
    assert(at(AIV_HOST_COMBINE_BARRIER_WAIT, 1) < at(drain));
    assert(meta(AIV_COMBINE_DRAIN, 0, 0) == 8 && meta(AIV_COMBINE_DRAIN, 0, 1) == 0);
    assert(wave(drain) == NO_WAVE);
    assert(wave(AIV_COMBINE_INIT) == NO_WAVE && wave(AIV_HOST_COMBINE_BARRIER_WAIT) == NO_WAVE);

    // A forgotten Finish still closes its interval without touching scheduling.
    {
        AscendTimerCombineProgressScope fallback(&timer, AIV_COMBINE_PROGRESS);
        fallback.StartScan();
        cycle += 2;
    }
    timer.FlushDynamicCounts();
    assert(count(AIV_COMBINE_PROGRESS) == 3);
    assert(at(logical(AIV_COMBINE_PROGRESS, 2), 1) > at(logical(AIV_COMBINE_PROGRESS, 2)));
    assert(meta(AIV_COMBINE_PROGRESS, 2, 0) == 0 && meta(AIV_COMBINE_PROGRESS, 2, 1) == 3);

    // Retain the true overflowing count for host rejection. Rejected slots
    // perform no timestamp reads and cannot corrupt the neighboring family.
    auto drainStart = at(drain);
    auto drainEnd = at(drain, 1);
    for (int i = 3; i <= MAX_DYNAMIC_ITER; ++i) {
        auto savedCycle = cycle;
        {
            AscendTimerCombineProgressScope overflow(&timer, AIV_COMBINE_PROGRESS);
            overflow.StartScan();
            overflow.StopScan();
            overflow.Finish(1, 0, 1, 1);
        }
        if (i == MAX_DYNAMIC_ITER) {
            assert(cycle == savedCycle);
        }
    }
    timer.FlushDynamicCounts();
    assert(count(AIV_COMBINE_PROGRESS) == MAX_DYNAMIC_ITER + 1);
    assert(at(drain) == drainStart && at(drain, 1) == drainEnd);
    assert(meta(AIV_COMBINE_DRAIN, 0, 0) == 8 && wave(drain) == NO_WAVE);
#else
    auto savedCycle = cycle;
    {
        AscendTimerFixedScope init(&timer, AIV_COMBINE_INIT);
        AscendTimerFixedScope hostBarrier(&timer, AIV_HOST_COMBINE_BARRIER_WAIT);
        AscendTimerCombineProgressScope progress(&timer, AIV_COMBINE_PROGRESS);
        AscendTimerCombineProgressScope drain(&timer, AIV_COMBINE_DRAIN);
        progress.StartScan();
        progress.StopScan();
        progress.Finish(1024, 0, 3, 128);
        drain.StartScan();
        drain.StopScan();
        drain.Finish(9, 0, 0, 0);
    }
    timer.FlushDynamicCounts();
    assert(cycle == savedCycle && pipelineBarriers == 0);
    for (auto value : buffer) {
        assert(value == 0);
    }
    (void)logical;
    (void)at;
    (void)count;
    (void)meta;
    (void)wave;
#endif
    // Both null scopes and a disabled timer must be completely inert, including
    // explicit Finish and destructor fallback, in timer-on and timer-off builds.
    AscendTimerDevice disabled;
    disabled.Init(nullptr);
    auto savedDisabledCycle = cycle;
    auto savedDisabledBarriers = pipelineBarriers;
    for (AscendTimerDevice* candidate : {static_cast<AscendTimerDevice*>(nullptr), &disabled}) {
        AscendTimerCombineProgressScope inert(candidate, AIV_COMBINE_PROGRESS);
        inert.StartScan();
        inert.StopScan();
        inert.Finish(1024, 0, 3, 128);
        AscendTimerCombineProgressScope fallback(candidate, AIV_COMBINE_DRAIN);
    }
    assert(cycle == savedDisabledCycle && pipelineBarriers == savedDisabledBarriers);
    return 0;
}
