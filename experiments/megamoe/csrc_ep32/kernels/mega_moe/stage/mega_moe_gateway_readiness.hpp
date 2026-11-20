// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_GATEWAY_READINESS_H
#define MEGA_MOE_GATEWAY_READINESS_H

#include "../common/mega_moe_utils.hpp"

namespace MegaMoeImpl {

// Every gateway broadcasts its complete invocation sequence before its input
// barrier. Consumers read only their own persistent peer-memory slots. The
// host-wide output barrier drains all Dispatch readers before another launch
// may overwrite these slots; sequence equality rejects notifications left by
// earlier launches. No routing-dependent mask or shared atomic RMW is needed.
__aicore__ inline void AcquireHierarchicalGatewayPayload(const BlockWorkspaceContext& workspace, const Params& params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || !UseHierarchicalRoce(params.tilingData) || workspace.blockNum == 0U ||
        workspace.blockIdx >= workspace.blockNum) {
        return;
    }
    constexpr uint32_t blockBytes = 32U;
    constexpr uint32_t wordsPerBlock = blockBytes / sizeof(int32_t);
    constexpr uint32_t fullReadyMask = (uint32_t(1) << HIERARCHICAL_RANKS_PER_HOST) - 1U;
    static_assert(HIERARCHICAL_ROCE_UB_BYTES >= HIERARCHICAL_RANKS_PER_HOST * blockBytes);
    GlobalTensor<int32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.gatewayPayloadReadyPtr));
    LocalTensor<int32_t> snapshot(TPosition::VECCALC, params.tilingData->roceScratchUbOffset,
                                  HIERARCHICAL_RANKS_PER_HOST * wordsPerBlock);
    uint32_t readyMask;
    do {
        SyncFuncStatic<HardEvent::S_MTE2, 7U>();
        // One DMA, sixteen independent 512B lines; only each line's first 32B
        // is copied. Strides are in 32B units, not bytes.
        DataCopy(snapshot, flags,
                 DataCopyParams{static_cast<uint16_t>(HIERARCHICAL_RANKS_PER_HOST), 1U,
                                static_cast<uint16_t>(HIERARCHICAL_FLAG_BYTES / blockBytes - 1U), 0U});
        SyncFuncStatic<HardEvent::MTE2_S, 7U>();
        readyMask = 0U;
        for (uint32_t slot = 0U; slot < HIERARCHICAL_RANKS_PER_HOST; ++slot) {
            if (static_cast<uint32_t>(snapshot.GetValue(slot * wordsPerBlock)) ==
                params.tilingData->hierarchicalSequence) {
                readyMask |= uint32_t(1) << slot;
            }
        }
    } while (readyMask != fullReadyMask);
    // Order the successful snapshot/scalar comparison before payload DMA and
    // before this AIV reuses the reserved UB for later RoCE/Combine operations.
    SyncFuncStatic<HardEvent::S_MTE2, 7U>();
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_GATEWAY_READINESS_H
