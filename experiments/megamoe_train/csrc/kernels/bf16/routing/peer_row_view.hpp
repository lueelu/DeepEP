// Copyright (c) 2026, Lu Lu
// Modified by xixing, guofanfeng 2026

#pragma once
#include "layout/mega_moe_context.hpp"
#include "schedule/peer_row_layout.hpp"

namespace MegaMoeBF16Training::Wave {
struct GmPeerPrefix {
    __gm__ int32_t* data;
    __aicore__ inline uint32_t Get(uint32_t index) const
    {
        return static_cast<uint32_t>(AscendC::ReadGmByPassDCache(data + index));
    }
};
struct UbPeerPrefix {
    AscendC::LocalTensor<int32_t> data;
    __aicore__ inline uint32_t Get(uint32_t index) const
    {
        return data.GetValue(index);
    }
};
__aicore__ inline PeerExpert GetPeerExpert(const Params& params, uint32_t expert)
{
    return DescribePeerExpert(GmPeerPrefix{reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM)},
                              params.tilingData->epWorldSize, params.tilingData->expertPerRank, expert,
                              params.tilingData->maxOutputSize, params.peerRowsGroups);
}
}  // namespace MegaMoeBF16Training::Wave
