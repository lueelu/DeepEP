// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_bf16_context.hpp
 * \brief BF16 kernel parameters and peer window addresses.
 */

#ifndef MEGA_MOE_CONTEXT_HPP
#define MEGA_MOE_CONTEXT_HPP

#include "common/mega_moe_kernel_utils.hpp"
#include "timer/ascend_timer_device.hpp"
#include "mega_moe_bf16_tiling.hpp"
#include "common/mega_moe_bf16_workspace_view.hpp"

using namespace AscendC;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer[1024];
};

struct Bf16GMMAddrInfo {
    MegaMoeTrainTimerDevice* timer = nullptr;
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR cGlobal;
    __gm__ int32_t* groupFlagList;
    __gm__ int32_t* groupFlagList2;
};

struct Bf16Params {
    MegaMoeTrainTimerDevice* timer = nullptr;
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    GM_ADDR permutedInputOutGmAddr;
    GM_ADDR actualRecvTokensOutGmAddr;
    GM_ADDR savedMetaOutGmAddr;
    ForwardWorkspaceInfo workspaceInfo;
    Bf16PeermemInfo peermemInfo;
    MegaMoeBf16TilingData* tilingData;
    // Rebuilt from the matching launch policy; zero retains source-major rows.
    GM_ADDR forwardRawGate{nullptr};
    GM_ADDR forwardRawUp{nullptr};
    uint32_t peerRowsGroups{0};
};

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

#endif
