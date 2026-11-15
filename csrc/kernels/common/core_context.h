// Copyright (c) 2026, Lu Lu

#pragma once

#include "kernel_operator.h"

namespace deepep {
// Logical execution context of one kernel launch.
//
// The ordinary <<<>>> entry fills this from AscendC::GetBlockIdx() and
// AscendC::GetBlockNum(). The SuperKernel entry fills it from
// sk::SkSystemArgs::skBlockIdx/skNumBlocks instead: inside a fused task those
// AscendC accessors describe the fused grid, not this sub-task. Every nested
// component that sizes partitions, owns a QP or addresses per-core staging must
// use this context rather than the raw AscendC accessors.
struct CoreContext {
    uint32_t index = 0U;
    uint32_t count = 0U;
};

__aicore__ inline CoreContext OrdinaryCoreContext()
{
    return {uint32_t(AscendC::GetBlockIdx()), uint32_t(AscendC::GetBlockNum())};
}
}  // namespace deepep
