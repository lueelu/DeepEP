// Copyright (c) 2026, Lu Lu

// Table-driven device wait used by the in-graph clock alignment protocol.
// One compiled kernel serves every call index: the deadline lives in a GM
// table captured by address, so a replay only refreshes the table contents.
#pragma once

#include <cstddef>
#include <cstdint>

#include <kernel_operator.h>
#include <shmem.h>

#include <deep_ep/common/clock_wait.hpp>
#include <kernels/common/core_context.h>

namespace deep_ep {

// Runtime argument block of the SuperKernel sub-entry. One field per explicit
// kernel parameter and in the same order, so the captured launch and the SK
// bind record share one ABI.
struct ClockWaitArgs {
    __gm__ uint8_t* deadlines;
    int64_t index;
    __gm__ uint8_t* records;
    int64_t batch;
    int32_t cores;
    int32_t diagnostic;
};
static_assert(sizeof(ClockWaitArgs) == 40U, "clock wait argument ABI changed");
static_assert(offsetof(ClockWaitArgs, index) == 8U, "clock wait argument ABI changed");
static_assert(offsetof(ClockWaitArgs, records) == 16U, "clock wait argument ABI changed");
static_assert(offsetof(ClockWaitArgs, batch) == 24U, "clock wait argument ABI changed");
static_assert(offsetof(ClockWaitArgs, cores) == 32U, "clock wait argument ABI changed");
static_assert(offsetof(ClockWaitArgs, diagnostic) == 36U, "clock wait argument ABI changed");

__aicore__ inline uint64_t ClockWaitNow()
{
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
}

// Every participating logical core waits. Records are written to a private
// 32-byte slot so no core can overwrite another core's evidence.
__aicore__ inline void clock_wait_body(const ClockWaitArgs& args, CoreContext core)
{
    if (args.deadlines == nullptr || args.index < 0 || args.cores <= 0) return;
    if (core.count == 0U || core.index >= core.count) return;
    const uint32_t cores = static_cast<uint32_t>(args.cores);
    if (core.index >= cores) return;

    auto* table = reinterpret_cast<__gm__ uint64_t*>(args.deadlines);
    const uint64_t deadline = table[args.index];
    const uint64_t enter = ClockWaitNow();

    __gm__ uint64_t* slot = nullptr;
    if (args.diagnostic != 0 && args.records != nullptr) {
        slot = reinterpret_cast<__gm__ uint64_t*>(args.records) +
            (uint64_t(args.index) * cores + core.index) * kClockWaitRecordWords;
        slot[kClockWaitRecordEnter] = enter;
        slot[kClockWaitRecordBatch] = uint64_t(args.batch);
        slot[kClockWaitRecordCore] = core.index;
    }

    uint64_t release = enter;
    while (release < deadline) release = ClockWaitNow();
    if (slot != nullptr) slot[kClockWaitRecordRelease] = release;
}

// One SuperKernel sub-entry template. `kSkSlot` only distinguishes the four
// symbols the bind table requires; every logical core executes the same wait.
template <uint32_t kSkSlot>
__aicore__ inline void clock_wait_impl_sk(const ClockWaitArgs* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    const CoreContext core{uint32_t(system_args->SkGetBlockIdx()), uint32_t(system_args->SkGetNumBlocks())};
    clock_wait_body(*args, core);
}

}  // namespace deep_ep
