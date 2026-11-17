// Copyright (c) 2026, Lu Lu
#pragma once
#include "ll_dispatch_c_api.h"
#include "../common/ep_memory_server_dedup_count_layout.h"

namespace DispatchDedup {
// Each core owns one local 512B magic slot. Retire its 32B scratch before Counts uses UB0.
__aicore__ inline uint32_t RotateDispatchServerDedupMagic(const __gm__ CommArgs* args, uint32_t core)
{
    auto* state = LlDispatchC::Ub<uint32_t>(0U);
    auto* slot = args->peerMems[args->rank] + uint64_t(core) * kEpServerDedupMagicStride;
    LlDispatchC::Load(state, slot, 32U);
    LlDispatchC::Sync<PIPE_MTE2, PIPE_S>();
    const uint32_t magic = state[0] ? 0U : 1U;
    state[0] = magic;
    LlDispatchC::Sync<PIPE_S, PIPE_MTE3>();
    LlDispatchC::Store(slot, state, 32U);
    LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
    return magic;
}
}  // namespace DispatchDedup
