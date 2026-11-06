// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_context.hpp
 * \brief
 */

#ifndef MEGA_MOE_CONTEXT_HPP
#define MEGA_MOE_CONTEXT_HPP

#include "lib/std/tuple.h"
#include "mega_moe_tiling.hpp"
#include "routing/moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h"
#include "layout/mega_moe_workspace_view.hpp"

using namespace AscendC;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer[1024];
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR cGlobal;
    __gm__ int32_t* groupFlagList;
    __gm__ int32_t* groupFlagList2;
};

struct Params {
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
    PeermemInfo peermemInfo;
    MegaMoeTilingData* tilingData;
};

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

#endif
