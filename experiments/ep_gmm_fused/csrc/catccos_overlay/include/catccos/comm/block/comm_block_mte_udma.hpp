/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/catccos_overlay/LICENSE in the source tree and licenses/catccos/LICENSE in the wheel.
 */
// Modified by zhu-mingzhe71 2026

#ifndef CATCCOS_COMM_BLOCK_MTE_UDMA_HPP
#define CATCCOS_COMM_BLOCK_MTE_UDMA_HPP

#include "catccos/catccos.hpp"

#include "shmem.h"

namespace Catccos::Comm::Block {

/// Presents the regular dynamic CommBlock interface while selecting MTE for a
/// self copy and UDMA for a remote peer. This lets existing communication and
/// GEMM schedulers use UDMA without changing their parameter ABI.
template <class LocalCommBlock_, class RemoteCommBlock_>
class CommBlockMteUdma {
public:
    using LocalCommBlock = LocalCommBlock_;
    using RemoteCommBlock = RemoteCommBlock_;
    using Params = typename LocalCommBlock::Params;
    using ElementSrc = typename LocalCommBlock::ElementSrc;
    using LayoutSrc = typename LocalCommBlock::LayoutSrc;
    using ElementDst = typename LocalCommBlock::ElementDst;
    using LayoutDst = typename LocalCommBlock::LayoutDst;
    using TileRemoteCopy = typename LocalCommBlock::TileRemoteCopy;

    template <class ArchTag>
    CATLASS_DEVICE CommBlockMteUdma(Catlass::Arch::Resource<ArchTag>& resource, Params const& params)
        : localComm(resource, params), remoteComm(resource, params)
    {
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        localComm.InitBlockLoop();
        remoteComm.InitBlockLoop();
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        // Complete every deferred UDMA request before the caller publishes its
        // remote-completion signal for this communication round.
        remoteComm.FinalizeBlockLoop();
        localComm.FinalizeBlockLoop();
    }

    /// Complete this worker's pending copies before a software signal without
    /// closing the local MTE event loop. In SHMEM 1.6.0 a generic fence only
    /// drains local pipes/cache, so UDMA must first be completed by its owner.
    CATLASS_DEVICE
    void CompleteForSignal()
    {
        remoteComm.FinalizeBlockLoop();
        aclshmemx_mte_quiet();
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
                    AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
                    Catlass::MatrixCoord const& actualCommBlockShape, uint32_t peerIdx)
    {
        if (peerIdx == static_cast<uint32_t>(shmem_my_pe())) {
            localComm(gmSrc, layoutSrc, gmDst, layoutDst, actualCommBlockShape, peerIdx);
        } else {
            remoteComm(gmSrc, layoutSrc, gmDst, layoutDst, actualCommBlockShape, peerIdx);
        }
    }

private:
    LocalCommBlock localComm;
    RemoteCommBlock remoteComm;
};

}  // namespace Catccos::Comm::Block

#endif  // CATCCOS_COMM_BLOCK_MTE_UDMA_HPP
