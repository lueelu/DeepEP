/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/comm/LICENSE.catccos for the full text of the License.
 */
#ifndef DEEPEP_COMM_BLOCK_HIERARCHICAL_COPY_HPP
#define DEEPEP_COMM_BLOCK_HIERARCHICAL_COPY_HPP

#include <type_traits>

#include "comm/common.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/numeric_size.hpp"

// from shmem
#include "shmem.h"

// Adapted from CatCCOS: Put-only, with explicit RDMA completion control.
namespace deepep::comm::block {

using Catlass::MatrixCoord;

// One block handles one peer's current communication range. The block owns UB,
// MTE tile scheduling and network QP slices; TileRemoteCopy is the existing MTE tile.
// The caller owns outer fusion windows and the per-peer QPs.
template <class ArchTag_, class SrcType_, class DstType_, class TileRemoteCopy_, class TileSwizzle_,
          uint32_t UbStages = 1>
class HierarchicalCopy {
public:
    static constexpr uint32_t UB_STAGES = UbStages;
    using ArchTag = ArchTag_;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using TileRemoteCopy = TileRemoteCopy_;
    using TileSwizzle = TileSwizzle_;
    using CopyTransport = comm::CopyTransport;
    using RdmaCompletion = comm::RdmaCompletion;
    static constexpr uint32_t UB_STAGE_BYTES = 32 * 1024;
    static constexpr uint32_t WQE_BYTES = 256;
    static constexpr uint32_t MAX_QP_NUM = 4;
    static constexpr uint64_t MAX_NETWORK_BYTES = 256ULL * 1024 * 1024;
    static_assert(UB_STAGES > 0 && UB_STAGES <= 8, "Use 1 to 8 UB stages/event IDs");
    static_assert(UB_STAGE_BYTES % 32 == 0, "UB stages must be aligned");
    static_assert(UB_STAGES <= ArchTag::UB_SIZE / UB_STAGE_BYTES, "Communication scratch exceeds UB");
    static_assert(std::is_same_v<ElementSrc, ElementDst>, "Source and destination types must match");
    static_assert(
        std::is_same_v<LayoutSrc, Catlass::layout::RowMajor> && std::is_same_v<LayoutDst, Catlass::layout::RowMajor>,
        "Hierarchical copy requires row-major layouts");
    static_assert(
        Catlass::SizeOfBits<ElementSrc>::value == sizeof(ElementSrc) * 8, "Packed sub-byte types are unsupported");
    static_assert(
        UB_STAGE_BYTES >= MAX_QP_NUM * WQE_BYTES && WQE_BYTES >= ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE,
        "Insufficient WQE scratch");

    struct Params {
        MatrixCoord blockShape;
        CATLASS_HOST_DEVICE
        explicit Params(MatrixCoord shape) : blockShape(shape) {}
    };

    CATLASS_DEVICE
    HierarchicalCopy(Catlass::Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        size_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubSList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
            ubOffset += UB_STAGE_BYTES;
            copyEventIdList[i] = i;
        }
    }

    // gmSrc/gmDst include the outer window/block offsets; actualCommBlockShape is
    // this call's extent. MTE waits before UB reuse; RDMA waits by default.
    // UDMA NBI consumes its UB scratch on return, but GM data remains in flight.
    // UDMA completion remains caller-owned. Deferred RDMA requires stable GM
    // sources and a final QuietRdma before reuse/return. Only Put is supported.
    CATLASS_DEVICE
    bool operator()(
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        MatrixCoord const& actualCommBlockShape, uint32_t rankIdx, CopyTransport transport, uint32_t qpNum = 1,
        RdmaCompletion rdmaCompletion = RdmaCompletion::Wait)
    {
        if (actualCommBlockShape.row() == 0 || actualCommBlockShape.column() == 0) {
            return false;
        }

        auto ubS = ubSList[ubListId];
        uint32_t copyEventId = copyEventIdList[ubListId];
        bool submitted = false;
        switch (transport) {
            case CopyTransport::Mte:
                CopyMte(gmSrc, layoutSrc, gmDst, layoutDst, actualCommBlockShape, ubS, copyEventId, rankIdx);
                break;
            case CopyTransport::Udma:
                submitted = CopyNetwork<CopyTransport::Udma>(
                    gmSrc, layoutSrc, gmDst, layoutDst, actualCommBlockShape, ubS, copyEventId, rankIdx, qpNum);
                break;
            case CopyTransport::Rdma:
                submitted = CopyNetwork<CopyTransport::Rdma>(
                    gmSrc, layoutSrc, gmDst, layoutDst, actualCommBlockShape, ubS, copyEventId, rankIdx, qpNum);
                break;
        }
        if (submitted && transport == CopyTransport::Rdma && rdmaCompletion == RdmaCompletion::Wait) {
            QuietRdma(rankIdx, qpNum);
        }
        ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        return submitted;
    }

    // Drain every data/scale/flag request on this peer's QPs, regardless of which
    // block object submitted it. Scratch/event must be idle; the AIV owns the QPs.
    CATLASS_DEVICE
    void QuietRdma(uint32_t rankIdx, uint32_t qpNum = 1)
    {
        auto scratch = reinterpret_cast<__ubuf__ uint8_t*>(ubSList[0].GetPhyAddr());
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            aclshmemx_roce_qp_quiet(rankIdx, qp, scratch + qp * WQE_BYTES, copyEventIdList[0]);
        }
    }

private:
    // Use the original GM strides for both contiguous and padded layouts. The
    // MTE tile pads each UB row to 32 bytes, including narrow scale rows.
    CATLASS_DEVICE
    void CopyMte(
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        MatrixCoord const& actualCommBlockShape, AscendC::LocalTensor<ElementSrc> const& ubS, uint32_t copyEventId,
        uint32_t rankIdx)
    {
        constexpr uint32_t maxTileN = UB_STAGE_BYTES / sizeof(ElementSrc);
        const uint32_t tileN = actualCommBlockShape.column() < maxTileN ? actualCommBlockShape.column() : maxTileN;
        const uint32_t tileRowBytes = (tileN * sizeof(ElementSrc) + 31) / 32 * 32;
        const uint32_t maxTileM = UB_STAGE_BYTES / tileRowBytes;
        const uint32_t tileM = actualCommBlockShape.row() < maxTileM ? actualCommBlockShape.row() : maxTileM;
        const MatrixCoord tileShape{tileM, tileN};
        TileSwizzle tileSwizzle{actualCommBlockShape, tileShape};
        uint32_t tileLoops = tileSwizzle.GetLoops();

        for (uint32_t tileIdx = 0; tileIdx < tileLoops; tileIdx++) {
            auto tileCoord = tileSwizzle.GetTileCoord(tileIdx);
            auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;

            // Get the data and layout of input
            auto gmTileSrc = gmSrc[layoutSrc.GetOffset(tileOffsetInBlock)];
            auto layoutTileSrc = layoutSrc.GetTileLayout(actualTileShape);

            // Get the data and layout of output
            auto gmTileDst = gmDst[layoutDst.GetOffset(tileOffsetInBlock)];
            auto layoutTileDst = layoutDst.GetTileLayout(actualTileShape);

            tileRemoteCopy(
                gmTileDst, layoutTileDst, gmTileSrc, layoutTileSrc, actualTileShape, ubS, copyEventId, rankIdx);
            // Complete this tile's MTE3 stores before UB reuse or publishing ready.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventId);
        }
    }

    // UDMA/RDMA copy: partition the current contiguous communication range across QPs. For padded layouts,
    // partition each logical row separately so padding remains untouched.
    // Returns true when at least one request was submitted.
    template <CopyTransport Transport>
    CATLASS_DEVICE bool CopyNetwork(
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        MatrixCoord const& actualCommBlockShape, AscendC::LocalTensor<ElementSrc> const& ubS, uint32_t copyEventId,
        uint32_t rankIdx, uint32_t qpNum)
    {
        if (layoutSrc.stride(0) == actualCommBlockShape.column() &&
            layoutDst.stride(0) == actualCommBlockShape.column()) {
            CopyNetworkSpan<Transport>(
                gmSrc, gmDst,
                static_cast<uint32_t>(
                    static_cast<uint64_t>(actualCommBlockShape.row()) * actualCommBlockShape.column()),
                ubS, copyEventId, rankIdx, qpNum);
        } else {
            for (uint32_t row = 0; row < actualCommBlockShape.row(); ++row) {
                CopyNetworkSpan<Transport>(
                    gmSrc[static_cast<uint64_t>(row) * layoutSrc.stride(0)],
                    gmDst[static_cast<uint64_t>(row) * layoutDst.stride(0)], actualCommBlockShape.column(), ubS,
                    copyEventId, rankIdx, qpNum);
            }
        }
        // Completion is selected by operator(); this helper only submits requests.
        return true;
    }

    template <CopyTransport Transport>
    CATLASS_DEVICE void CopyNetworkSpan(
        AscendC::GlobalTensor<ElementSrc> gmSrc, AscendC::GlobalTensor<ElementDst> gmDst, uint32_t elements,
        AscendC::LocalTensor<ElementSrc> const& ubS, uint32_t copyEventId, uint32_t rankIdx, uint32_t qpNum)
    {
        // Keep boundaries element-aligned for generic element types, as well as 32-byte aligned
        // for FP8 data and scales. Small spans may leave some QPs empty.
        constexpr uint32_t alignElements = sizeof(ElementSrc) < 32 ? 32 / sizeof(ElementSrc) : 1;
        const uint64_t units = elements / alignElements;
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            const uint64_t begin = units * qp / qpNum * alignElements;
            const uint64_t end = qp + 1 == qpNum ? elements : units * (qp + 1) / qpNum * alignElements;
            CopyNetworkQp<Transport>(
                gmSrc, gmDst, begin, static_cast<uint32_t>(end - begin), ubS, copyEventId, rankIdx, qp);
        }
    }

    template <CopyTransport Transport>
    CATLASS_DEVICE void CopyNetworkQp(
        AscendC::GlobalTensor<ElementSrc> gmSrc, AscendC::GlobalTensor<ElementDst> gmDst, uint64_t offset,
        uint32_t elements, AscendC::LocalTensor<ElementSrc> const& ubS, uint32_t copyEventId, uint32_t rankIdx,
        uint32_t qp)
    {
        constexpr uint32_t maxElements = static_cast<uint32_t>(MAX_NETWORK_BYTES / sizeof(ElementSrc));
        auto scratch = reinterpret_cast<__ubuf__ uint8_t*>(ubS.GetPhyAddr()) + qp * WQE_BYTES;
        uint32_t done = 0;
        while (done < elements) {
            const uint32_t count = elements - done < maxElements ? elements - done : maxElements;
            // GlobalTensor exposes a const view; SHMEM's Get/Put pointer overloads use mutable pointers.
            auto dstPtr = const_cast<__gm__ uint8_t*>(
                reinterpret_cast<const __gm__ uint8_t*>(gmDst.GetPhyAddr() + offset + done));
            auto srcPtr = const_cast<__gm__ uint8_t*>(
                reinterpret_cast<const __gm__ uint8_t*>(gmSrc.GetPhyAddr() + offset + done));
            const uint32_t bytes = count * static_cast<uint32_t>(sizeof(ElementSrc));
            if constexpr (Transport == CopyTransport::Udma) {
                aclshmemx_udma_qp_put_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, rankIdx, qp, copyEventId);
            } else {
                aclshmemx_roce_qp_put_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, rankIdx, qp, copyEventId);
            }
            done += count;
        }
    }

    Params params;
    AscendC::LocalTensor<ElementSrc> ubSList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
};

} // namespace deepep::comm::block
#endif // DEEPEP_COMM_BLOCK_HIERARCHICAL_COPY_HPP
