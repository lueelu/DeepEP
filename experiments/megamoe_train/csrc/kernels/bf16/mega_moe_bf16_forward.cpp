// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*
 * Standalone DeepEP MegaMoe BF16 entry.  The implementation uses DeepEP's
 * routing/dispatch/combine pipeline and only references CATLASS primitives
 * for BF16 MMAD; it does not call the CATCCOS DispatchFFNCombine operator.
 */
#include "kernel_operator.h"
#include "mega_moe_quant_constants.hpp"
#include "mega_moe_tiling.hpp"
#include "mega_moe_bf16_forward.hpp"
#include "csrc/bf16/kernel_launch.hpp"

using namespace AscendC;
using namespace MegaMoeImpl;

template <typename TopkWeightT>
__mix__(1, 2) __global__ __aicore__
    void mega_moe_bf16_forward(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                               GM_ADDR weight2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR trainingSavedStateGM,
                               GM_ADDR dispatchOutputGM, GM_ADDR forwardTransientWorkspaceGM,
                               MegaMoeTilingData tilingData)
{
    InitSocState();
    MegaMoeBF16Forward<TopkWeightT> op;
    op.Init(context, x, topkIds, topkWeights, weight1, weight2, nullptr, nullptr, nullptr, nullptr, yOut,
            expertTokenNumsOut, nullptr, nullptr, nullptr, trainingSavedStateGM, &tilingData,
            forwardTransientWorkspaceGM, dispatchOutputGM);
    op.EnableDispatchTileReady(forwardTransientWorkspaceGM);
    op.Process();
}

extern "C" ASCEND_MEGAMOE_EXPORT void deepep_train_mega_moe_bf16_forward_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* x, uint8_t* topk_ids, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* training_saved_state,
    uint8_t* dispatch_output, uint8_t* forward_transient_workspace, const uint8_t* tiling_host, uint64_t ffts_addr)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    if (tiling.topkWeightType == 1U) {
        mega_moe_bf16_forward<float><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, tiling);
    } else {
        mega_moe_bf16_forward<bfloat16_t><<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2, y, expert_token_nums, training_saved_state,
            dispatch_output, forward_transient_workspace, tiling);
    }
}

// The Host queries the loaded SO, so changing a build option cannot leave
// device execution and transient capacity using different policies.
extern "C" ASCEND_MEGAMOE_EXPORT int deepep_train_mega_moe_bf16_swiglu_decoupled(uint64_t hidden)
{
    // Legacy Hosts know only hidden. Return the union of saved-mode policies
    // so they may over-allocate at the boundary, but never under-allocate.
    return MegaMoeBF16Training::ForwardUsesGmSwiglu(hidden, MegaMoeBF16Training::kSaveMxFp8) ? 1 : 0;
}

extern "C" ASCEND_MEGAMOE_EXPORT int deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(uint64_t hidden,
                                                                                                uint32_t savedMode)
{
    return MegaMoeBF16Training::ForwardUsesGmSwiglu(hidden, savedMode) ? 1 : 0;
}
