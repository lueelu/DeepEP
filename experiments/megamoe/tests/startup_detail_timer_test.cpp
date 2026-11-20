// Copyright (c) 2026, Lu Lu
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
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(timer.CountSnapshotAddress() ==
           reinterpret_cast<uint8_t*>(buffer.data() + COUNT_SNAPSHOT_OFFSET + COUNT_SNAPSHOT_SLOTS_PER_CORE));
#endif
    constexpr int core = 1;
    const int counters = N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE_ALIGN;
    const int diagnostic = COUNT_DIAGNOSTIC_OFFSET + core * COUNT_DIAGNOSTIC_COUNT;
    uint64_t values[COUNT_DIAGNOSTIC_COUNT]{};
    values[COUNT_PROFILE_VALID] = 1;
    values[COUNT_SNAPSHOT_ATTEMPTS] = 3;
    values[COUNT_SNAPSHOT_FAILURES] = 2;
    values[COUNT_LAST_STALE_SOURCE_MASK] = 1ULL << 31;
    timer.SetWave(4);
    timer.StartSpan(AIV_DISPATCH);
    const auto barriers = pipelineBarriers;
    {
        AscendTimerRelayScope relay(&timer, RelayStep::RouteBuild, 30U, 168U, 12U);
        cycle += 100;
    }
    timer.RecordDetail(AIV_COMBINE_RUN, 1000, 1010, (uint64_t(7) << 32) | 2U, 0x302U);
    timer.WriteCountDiagnostics(values);
    assert(pipelineBarriers == barriers);
    timer.EndSpan(AIV_DISPATCH);
    timer.FlushDynamicCounts();
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(buffer[counters + AIV_RELAY_STEP] == 1 && buffer[counters + AIV_COMBINE_RUN] == 1);
    assert(buffer[counters + AIV_DISPATCH] == 1);
    const int metadata = TASK_METADATA_OFFSET + (core * DYNAMIC_TYPE_COUNT + AIV_RELAY_STEP) * MAX_DYNAMIC_ITER * 2;
    assert(buffer[metadata] == ((int64_t(2) << 32) | 31U));
    assert(buffer[metadata + 1] == ((int64_t(168) << 32) | 12U));
    const int logical = FIXED_TIMING_COUNT + AIV_COMBINE_RUN * MAX_DYNAMIC_ITER;
    const int timing = core * N_TIMING_COUNTER_PER_CORE_ALIGN + logical * 2;
    assert(buffer[timing] == 1000 && buffer[timing + 1] == 1010);
    assert(buffer[WAVE_ID_OFFSET + core * WAVE_ID_PER_CORE_ALIGN + logical] == 4);
    assert(buffer[diagnostic + COUNT_PROFILE_VALID] == 1);
    assert(buffer[diagnostic + COUNT_SNAPSHOT_FAILURES] == 2);
    assert(buffer[diagnostic + COUNT_LAST_STALE_SOURCE_MASK] == (1LL << 31));
    timer.Init(reinterpret_cast<uint8_t*>(buffer.data()));
    assert(buffer[diagnostic + COUNT_PROFILE_VALID] == 0);  // No stale profile after a later non-snapshot launch.
#else
    assert(buffer[counters + AIV_RELAY_STEP] == 0 && buffer[counters + AIV_COMBINE_RUN] == 0);
    assert(buffer[diagnostic + COUNT_PROFILE_VALID] == 0 && pipelineBarriers == 0);
#endif
}
