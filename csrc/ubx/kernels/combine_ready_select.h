// Copyright (c) 2026, Lu Lu

#pragma once
#include "kernel_operator.h"

namespace UbxCombine {
constexpr uint32_t kReadyTaskDone = 1U << 16U;
constexpr uint32_t kReadyTaskLocal = 1U << 17U;

// Stable Vector compaction. A completed task never becomes runnable again;
// dependency bits only cover staged members, not the local expert contribution.
__aicore__ inline uint32_t SelectReadyCombineTasks(AscendC::LocalTensor<int32_t> deps,
                                                   AscendC::LocalTensor<int32_t> work,
                                                   AscendC::LocalTensor<int32_t> tasks,
                                                   AscendC::LocalTensor<uint8_t> mask, uint32_t count, uint32_t missing,
                                                   bool remoteOnly)
{
    using namespace AscendC;
    if (!count) return 0U;
    const uint32_t rounded = (count + 63U) / 64U * 64U;
    SetFlag<HardEvent::S_V>(7U);
    WaitFlag<HardEvent::S_V>(7U);
    Duplicate(work, int32_t(missing | kReadyTaskDone | (remoteOnly ? kReadyTaskLocal : 0U)), rounded);
    PipeBarrier<PIPE_V>();
    And(work.ReinterpretCast<uint16_t>(), deps.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(),
        rounded * 2U);
    PipeBarrier<PIPE_V>();
    Compares(mask, work, int32_t(0), CMPMODE::EQ, rounded);
    PipeBarrier<PIPE_V>();
    CreateVecIndex(work, int32_t(0), count);
    PipeBarrier<PIPE_V>();
    uint64_t selected = 0U;
    GatherMask(tasks, work, mask.ReinterpretCast<uint32_t>(), true, count, {1U, 1U, 0U, 0U}, selected);
    SetFlag<HardEvent::V_S>(7U);
    WaitFlag<HardEvent::V_S>(7U);
    return uint32_t(selected);
}
}  // namespace UbxCombine
