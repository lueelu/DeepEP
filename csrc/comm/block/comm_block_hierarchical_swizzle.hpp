/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/comm/LICENSE.catccos for the full text of the License.
 */
#ifndef DEEPEP_COMM_BLOCK_HIERARCHICAL_SWIZZLE_HPP
#define DEEPEP_COMM_BLOCK_HIERARCHICAL_SWIZZLE_HPP

#include <cstdint>
#include "comm/common.hpp"
#include "catlass/matrix_coord.hpp"

namespace deepep::comm::block {

// Consecutive ranks form a group (one server/UB domain); GROUP_SIZE is fixed at compile time.
// One AIV owns one peer; the communication block partitions
// the current peer tensor across that peer's QPs. The self peer is copied in blocks with MTE;
// every other same-group peer uses UDMA and every peer in another group uses RDMA.
// There is no automatic fallback: the host verifies the transport routes
// before launching the kernel.
template <uint32_t CoreCount_ = 32, uint32_t GroupSize_ = 16, uint32_t CommBlockM_ = 64>
class HierarchicalCommSwizzle {
public:
    static_assert(CoreCount_ > 0, "Core count must be positive");
    static constexpr uint32_t CORE_COUNT = CoreCount_;
    static constexpr uint32_t GROUP_SIZE = GroupSize_;
    static constexpr uint32_t COMM_BLOCK_M = CommBlockM_;
    static_assert(COMM_BLOCK_M > 0, "Communication block rows must be positive");
    static_assert(GROUP_SIZE > 0 && GROUP_SIZE <= CORE_COUNT, "Group size must be in [1, CORE_COUNT]");
    using CopyTransport = comm::CopyTransport;
    using MatrixCoord = Catlass::MatrixCoord;

    // problemShape = {rows per peer, columns per peer, world size}.
    // Rows and columns describe the current communication range: a complete peer tensor or
    // a caller-selected fusion window. The caller owns the outer window loop and its base addresses.
    // aicoreIdx is the physical AIV index; with one AIV per peer it equals the peer rank.
    // COMM_BLOCK_M is the compile-time number of rows moved by one MTE round (commIdx).
    // UDMA/RDMA schedule the entire current range in one round, regardless of COMM_BLOCK_M.
    // GROUP_SIZE is the number of consecutive ranks in one UDMA domain.
    CATLASS_HOST_DEVICE
    HierarchicalCommSwizzle(CommCoord problemShape, uint32_t rank, uint32_t aicoreIdx)
        : problemShape(problemShape), rank(rank), aicoreIdx(aicoreIdx), peer(aicoreIdx)
    {
        if (peer == rank) {
            transport = CopyTransport::Mte;
        } else if (peer / GROUP_SIZE == rank / GROUP_SIZE) {
            transport = CopyTransport::Udma;
        } else {
            transport = CopyTransport::Rdma;
        }
        const bool hasWork = IsActiveCore() && problemShape.column() != 0 && problemShape.row() != 0;
        if (!hasWork) {
            coreLoops = 0;
        } else if (transport == CopyTransport::Mte) {
            coreLoops = problemShape.row() / COMM_BLOCK_M + (problemShape.row() % COMM_BLOCK_M != 0);
        } else {
            coreLoops = 1;
        }
    }

    CATLASS_HOST_DEVICE
    bool IsValidProblem() const
    {
        return problemShape.rank() > 0 && problemShape.rank() <= CORE_COUNT && rank < problemShape.rank();
    }

    CATLASS_HOST_DEVICE
    bool IsActiveCore() const { return IsValidProblem() && peer < problemShape.rank(); }

    // Query transport only for an active core; inactive cores schedule no blocks.
    CATLASS_HOST_DEVICE
    CopyTransport GetTransport() const { return transport; }

    CATLASS_HOST_DEVICE
    uint32_t GetPeer() const { return peer; }

    // First QP of this peer context; additional QPs are managed by the communication block.
    CATLASS_HOST_DEVICE
    uint32_t GetQpIndex() const { return 0; }

    CATLASS_HOST_DEVICE
    uint32_t GetCommBlockM() const { return COMM_BLOCK_M; }

    CATLASS_HOST_DEVICE
    uint32_t GetGroupSize() const { return GROUP_SIZE; }

    CATLASS_HOST_DEVICE
    uint32_t GetAicoreIdx() const { return aicoreIdx; }

    CATLASS_HOST_DEVICE
    uint32_t GetRealCore() const { return CORE_COUNT; }

    // Number of communication rounds (commIdx) for this AIV's current problemShape, not fusion windows.
    // For valid nonempty work: ceil(rows / COMM_BLOCK_M) for MTE, one for UDMA/RDMA; otherwise zero.
    CATLASS_HOST_DEVICE
    uint32_t GetCoreLoop() const { return coreLoops; }

    CATLASS_HOST_DEVICE
    CommCoord GetBlockCoord(uint32_t commIdx) const { return {commIdx, 0, peer}; }

    CATLASS_HOST_DEVICE
    bool IsValidBlock(CommCoord block) const
    {
        return block.rank() == peer && block.column() == 0 && block.row() < GetCoreLoop();
    }

    // Offset relative to the current communication range; the caller adds any outer window offset.
    CATLASS_HOST_DEVICE
    MatrixCoord GetBlockOffsetInRank(MatrixCoord block) const
    {
        if (block.column() != 0 || block.row() >= GetCoreLoop()) {
            return {UINT32_MAX, UINT32_MAX};
        }
        return {transport == CopyTransport::Mte ? block.row() * COMM_BLOCK_M : 0U, 0};
    }

    CATLASS_HOST_DEVICE
    MatrixCoord GetActualBlockShape(MatrixCoord block) const
    {
        if (block.column() != 0 || block.row() >= GetCoreLoop()) {
            return {0U, 0U};
        }
        if (transport != CopyTransport::Mte) {
            return {problemShape.row(), problemShape.column()};
        }
        const uint32_t remaining = problemShape.row() - block.row() * COMM_BLOCK_M;
        return {remaining < COMM_BLOCK_M ? remaining : COMM_BLOCK_M, problemShape.column()};
    }

private:
    CommCoord problemShape;
    uint32_t rank;
    uint32_t aicoreIdx;
    uint32_t peer = 0;
    CopyTransport transport;
    uint32_t coreLoops = 0;
};

} // namespace deepep::comm::block
#endif
