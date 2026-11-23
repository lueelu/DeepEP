// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once
#include "kernel_operator.h"

namespace MegaMoeImpl {
// One cacheline per expert-local M tile. Word 0 counts completed dO transfers;
// word 1 counts the dHp producers that SwiGLUGrad waits for.
// Packed expert bases are not necessarily tile-aligned, so reserve one gap
// per expert instead of rounding the complete receive buffer down.
struct RowSliceReady {
    __gm__ int32_t* slots{nullptr};
    uint64_t first{0};
    uint32_t tileRows{128};
    __aicore__ inline RowSliceReady(GM_ADDR base, uint64_t expertBase, uint32_t expert, uint32_t m)
        : slots(reinterpret_cast<__gm__ int32_t*>(base)), first(expertBase / m + expert), tileRows(m)
    {
    }
    __aicore__ inline __gm__ int32_t* Slot(uint32_t row) const
    {
        return slots + (first + row / tileRows) * 16U;
    }
};
}  // namespace MegaMoeImpl
