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

#ifndef CATCCOS_COMM_BLOCK_REMOTE_COPY_HPP
#define CATCCOS_COMM_BLOCK_REMOTE_COPY_HPP

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/detail/remote_copy_type.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

// from shmem
#include "shmem.h"

namespace Catccos::Comm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

// 婢舵氨娣弫鐗堝祦閻ㄥ嫯绻欑粩顖炩偓姘繆鐎圭偟骞?// for matmul fusion kernel
template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_, class SrcType_, class DstType_, class BlockShape_,
          class TileRemoteCopy_, class TileSwizzle_>
class CommBlock<AtlasCommRemoteCopy<ArchTag_, UB_STAGES_, IsDynamic_>, SrcType_, DstType_, BlockShape_, TileRemoteCopy_,
                TileSwizzle_> {
public:
    // Type aliases
    using DispatchPolicy = AtlasCommRemoteCopy<ArchTag_, UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using BlockShape = BlockShape_;
    using TileRemoteCopy = TileRemoteCopy_;
    using TileSwizzle = TileSwizzle_;
    using TileParams = typename TileRemoteCopy::Params;
    static constexpr detail::CopyDirect RemoteCopyDirect = TileRemoteCopy::RemoteCopyDirect;

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord BlockShape()
        {
            return BlockShape::ToCoord();
        }
        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileRemoteCopy::TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord blockShape;
        TileParams tileParams;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord blockShape_, const TileParams& tileParams_)
            : blockShape(blockShape_), tileParams(tileParams_)
        {
        }

        CATLASS_DEVICE
        MatrixCoord BlockShape() const
        {
            return blockShape;
        }
        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileParams.TileShape();
        }
    };

    using Params = ParamsBase<IsDynamic>;

    CATLASS_DEVICE
    CommBlock(Catlass::Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        size_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubSList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
            ubOffset += params.TileShape().row() * params.TileShape().column() * sizeof(ElementDst);
        }
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
        ubListId = 0;
    }

    CATLASS_DEVICE
    ~CommBlock() {}

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
                    AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
                    MatrixCoord const& actualCommBlockShape, uint32_t rankIdx)
    {
        if (actualCommBlockShape.row() == 0) {
            return;
        }

        auto tileShape = params.TileShape();
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

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            tileRemoteCopy(gmTileDst, layoutTileDst, gmTileSrc, layoutTileSrc, actualTileShape, ubSList[ubListId],
                           copyEventIdList[ubListId], rankIdx);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    AscendC::LocalTensor<ElementDst> ubSList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
};

// 娑撯偓缂佸瓨鏆熼幑顔炬畱鏉╂粎顏柅姘繆鐎圭偟骞?// for comm fusion kernel
template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_, class SrcType_, class DstType_, class BlockShape_>
// This specialization handles 1D contiguous input and performs cross-rank remote copy
// without the tile/layout remapping used by the TileRemoteCopy-based specialization above.
class CommBlock<AtlasCommRemoteCopy<ArchTag_, UB_STAGES_, IsDynamic_>, SrcType_, DstType_, BlockShape_> {
public:
    // Type aliases
    using DispatchPolicy = AtlasCommRemoteCopy<ArchTag_, UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    static constexpr uint32_t flagSize = 2 * 1024;  // Reserved bytes at UB start for hardware flags
    static constexpr uint32_t ubAlignSize = 64;     // UB buffer alignment in elements
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    static_assert(std::is_same_v<ElementSrc, ElementDst>,
                  "This 1D remote-copy CommBlock expects the same source and destination element type.");
    using BlockShape = BlockShape_;
    // Epilogue params definition
    struct Params {
        uint32_t blockShape;
        CATLASS_DEVICE
        Params() {}
        CATLASS_DEVICE
        Params(uint32_t blockShape_) : blockShape(blockShape_) {}
    };

    CATLASS_DEVICE
    CommBlock(Catlass::Arch::Resource<ArchTag>& resource, Params const&)
    {
        tileElements = (ArchTag::UB_SIZE - flagSize) / UB_STAGES / sizeof(ElementSrc) / ubAlignSize * ubAlignSize;
        size_t ubOffset = flagSize;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubSrcList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
            ubOffset += tileElements * sizeof(ElementSrc);
        }
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
        ubListId = 0;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc>& gmSrc, AscendC::GlobalTensor<ElementDst>& gmDst,
                    uint32_t const& actualCommBlockShape, uint32_t rankIdx)
    {
        if (actualCommBlockShape == 0) {
            return;
        }
        uint32_t tileLoops = AscendC::CeilDivision(actualCommBlockShape, tileElements);
        for (uint32_t tileIdx = 0; tileIdx < tileLoops; tileIdx++) {
            uint32_t offset = tileIdx * tileElements;
            uint32_t processNum =
                (tileIdx == (tileLoops - 1)) ? (actualCommBlockShape - tileIdx * tileElements) : tileElements;
            CopyTile(gmSrc, gmDst, offset, processNum, rankIdx);
        }
    }

private:
    CATLASS_DEVICE
    void CopyTile(AscendC::GlobalTensor<ElementSrc>& gmSrc, AscendC::GlobalTensor<ElementDst>& gmDst, uint32_t offset,
                  uint32_t processNum, uint32_t rankIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
        non_contiguous_copy_param copyParams;
        copyParams.repeat = 1;
        copyParams.length = processNum;
        copyParams.src_ld = processNum;
        copyParams.dst_ld = processNum;
        aclshmemx_mte_get_nbi(gmDst[offset], gmSrc[offset], ubSrcList[ubListId], copyParams, rankIdx,
                              copyEventIdList[ubListId]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
        ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
    }
    AscendC::LocalTensor<ElementSrc> ubSrcList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    uint32_t tileElements{0};
};

template <class ArchTag_, uint32_t UB_STAGES_, class SrcType_, class DstType_, class TileRemoteCopy_>
class CommBlock<AtlasCommUdmaRemoteCopy<ArchTag_, UB_STAGES_>, SrcType_, DstType_, TileRemoteCopy_> {
public:
    // Type aliases
    using DispatchPolicy = AtlasCommUdmaRemoteCopy<ArchTag_, UB_STAGES_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using TileRemoteCopy = TileRemoteCopy_;

    CATLASS_DEVICE
    CommBlock() = default;

    template <class ResourceArch, class Params>
    CATLASS_DEVICE CommBlock(Catlass::Arch::Resource<ResourceArch>& resource, Params const&)
    {
        // Reserve the end of UB for WQE publication so it cannot alias the
        // local MTE communication tiles placed from offset zero. Existing
        // UDMA tiles either ignore this tensor or provide their own scratch.
        ubSList[0] = resource.ubBuf.template GetBufferByByte<ElementDst>(ResourceArch::UB_SIZE -
                                                                         TileRemoteCopy::UDMA_WQE_SCRATCH_BYTES);
    }

    CATLASS_DEVICE
    ~CommBlock() {}

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        hasPendingCopy = false;
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        if constexpr (TileRemoteCopy::DEFER_QUIET) {
            if (hasPendingCopy) {
                tileRemoteCopy.Finalize(activePeerIdx);
            }
        }
        hasPendingCopy = false;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
                    AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
                    MatrixCoord const& actualCommBlockShape, uint32_t rankIdx)
    {
        if (actualCommBlockShape.row() == 0) {
            return;
        }

        if constexpr (TileRemoteCopy::DEFER_QUIET) {
            // Current collective schedules one peer per block loop. Keep this
            // guard so future schedulers can switch peers without mixing two
            // completion domains in one deferred quiet.
            if (hasPendingCopy && activePeerIdx != rankIdx) {
                tileRemoteCopy.Finalize(activePeerIdx);
                hasPendingCopy = false;
            }
        }

        tileRemoteCopy(gmDst, layoutDst, gmSrc, layoutSrc, actualCommBlockShape, ubSList[ubListId],
                       copyEventIdList[ubListId], rankIdx);
        if constexpr (TileRemoteCopy::DEFER_QUIET) {
            activePeerIdx = rankIdx;
            hasPendingCopy = true;
        }
    }

private:
    AscendC::LocalTensor<ElementDst> ubSList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    uint32_t activePeerIdx{0};
    bool hasPendingCopy{false};
    TileRemoteCopy tileRemoteCopy;
};

}  // namespace Catccos::Comm::Block

#endif  // CATCCOS_COMM_BLOCK_REMOTE_COPY_HPP
