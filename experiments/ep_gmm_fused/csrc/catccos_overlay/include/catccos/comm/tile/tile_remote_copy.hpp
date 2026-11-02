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

#ifndef CATCCOS_TILE_REMOTE_COPY_HPP
#define CATCCOS_TILE_REMOTE_COPY_HPP

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/tile/copy_int4_rowmajor.hpp"

// from catlass
#include "catlass/catlass.hpp"

// from shmem
#include "shmem.h"

namespace Catccos::Comm::Tile {

using Catlass::MatrixCoord;

template <
    /// Tag indicating architecture
    class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_, detail::CopyDirect CopyDirect_,
    detail::CopyTransport CopyTransport_>
struct TileRemoteCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Get,
                      detail::CopyTransport::Mte> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        if constexpr (std::is_same_v<ElementSrc, AscendC::int4b_t>) {
            using Catlass::layout::RowMajor;
            RowMajor layoutUb{copyShape, Catlass::MakeCoord<int64_t>(copyShape.column(), 1)};

            auto remotePtr = shmem_ptr(Int4GmVoidAddr(srcTensor), peerIdx);
            AscendC::GlobalTensor<ElementSrc> gmRemoteSrc;
            gmRemoteSrc.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSrc*>(remotePtr));

            CopyInt4GmToUbRowMajor(tmpUb, gmRemoteSrc, static_cast<RowMajor const&>(layoutUb),
                                   static_cast<RowMajor const&>(srcLayout));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);

            CopyInt4UbToGmRowMajor(dstTensor, tmpUb, static_cast<RowMajor const&>(dstLayout),
                                   static_cast<RowMajor const&>(layoutUb));
            return;
        }

        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);

        auto ptr = aclshmem_ptr((__gm__ void*)srcTensor.GetPhyAddr(), peerIdx);

        AscendC::GlobalTensor<ElementSrc> remoteBuff;
        remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSrc*>(ptr));

        uint64_t ELE_NUM_PER_UNIT = Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<ElementSrc>::value;
        uint64_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
        AscendC::DataCopyExtParams dataCopyParamsGm2ub(
            copyParams.repeat,
            copyParams.length * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value,
            (copyParams.src_ld - copyParams.length) * Catlass::SizeOfBits<ElementSrc>::value /
                Catlass::SizeOfBits<uint8_t>::value,
            (ubStride - copyParams.length) / ELE_NUM_PER_UNIT, 0);
        aclshmemi_copy_gm2ub(tmpUb, remoteBuff, dataCopyParamsGm2ub);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);

        AscendC::DataCopyExtParams dataCopyParamsUb2gm(
            copyParams.repeat,
            copyParams.length * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value,
            (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
            (copyParams.dst_ld - copyParams.length) * Catlass::SizeOfBits<ElementSrc>::value /
                Catlass::SizeOfBits<uint8_t>::value,
            0);
        aclshmemi_copy_ub2gm(dstTensor, tmpUb, dataCopyParamsUb2gm);
    }
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Put,
                      detail::CopyTransport::Mte> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);
        auto ptr = aclshmem_ptr((__gm__ void*)dstTensor.GetPhyAddr(), peerIdx);

        AscendC::GlobalTensor<ElementSrc> remoteBuff;
        remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSrc*>(ptr));

        uint64_t ELE_NUM_PER_UNIT = Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<ElementSrc>::value;
        uint64_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
        AscendC::DataCopyExtParams dataCopyParamsGm2ub(
            copyParams.repeat,
            copyParams.length * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value,
            (copyParams.src_ld - copyParams.length) * Catlass::SizeOfBits<ElementSrc>::value /
                Catlass::SizeOfBits<uint8_t>::value,
            (ubStride - copyParams.length) / ELE_NUM_PER_UNIT, 0);
        aclshmemi_copy_gm2ub(tmpUb, srcTensor, dataCopyParamsGm2ub);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(peerIdx);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(peerIdx);

        AscendC::DataCopyExtParams dataCopyParamsUb2gm(
            copyParams.repeat,
            copyParams.length * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value,
            (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
            (copyParams.dst_ld - copyParams.length) * Catlass::SizeOfBits<ElementSrc>::value /
                Catlass::SizeOfBits<uint8_t>::value,
            0);
        aclshmemi_copy_ub2gm(remoteBuff, tmpUb, dataCopyParamsUb2gm);
    }
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Get,
                      detail::CopyTransport::Rdma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        AscendC::LocalTensor<uint32_t> ubLocal32;
        ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
        ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
        AscendC::LocalTensor<uint64_t> ubLocal64;
        ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
        ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen =
            repeat * stride * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value;
        auto ptr = shmem_ptr((__gm__ void*)srcTensor.GetPhyAddr(), peerIdx);
        aclshmemi_roce_read((__gm__ uint8_t*)(dstTensor.GetPhyAddr()), (__gm__ uint8_t*)ptr, peerIdx, 0, messageLen,
                            ubLocal64, ubLocal32, 0);

        aclshmemi_roce_quiet(peerIdx, 0, ubLocal64, ubLocal32, 0);
    }
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Put,
                      detail::CopyTransport::Rdma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        AscendC::LocalTensor<uint32_t> ubLocal32;
        ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
        ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
        AscendC::LocalTensor<uint64_t> ubLocal64;
        ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
        ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen =
            repeat * stride * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value;
        auto ptr = shmem_ptr((__gm__ void*)dstTensor.GetPhyAddr(), peerIdx);
        aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(srcTensor.GetPhyAddr()), peerIdx, 0, messageLen,
                             ubLocal64, ubLocal32, 0);

        aclshmemi_roce_quiet(peerIdx, 0, ubLocal64, ubLocal32, 0);
    }
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Get,
                      detail::CopyTransport::Udma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;
    static constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
    static constexpr bool DEFER_QUIET = false;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
        pipe.InitBuffer(buf, UDMA_WQE_SCRATCH_BYTES);
        AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_WQE_SCRATCH_BYTES, 0);
        constexpr uint32_t SYNC_ID = 0;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen =
            repeat * stride * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value;
        aclshmemx_udma_get_nbi((__gm__ uint8_t*)dstTensor.GetPhyAddr(), (__gm__ uint8_t*)srcTensor.GetPhyAddr(),
                               (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), messageLen, peerIdx, SYNC_ID);

        aclshmemx_udma_quiet(peerIdx);
    }
};

template <class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Put,
                      detail::CopyTransport::Udma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;
    static constexpr bool IsDynamic = IsDynamic_;
    static constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
    static constexpr bool DEFER_QUIET = false;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape()
        {
            return TileShape::ToCoord();
        }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const
        {
            return tileShape;
        }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
                    AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutDst const& srcLayout,
                    MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb, uint32_t copyEventId,
                    uint32_t peerIdx)
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
        pipe.InitBuffer(buf, UDMA_WQE_SCRATCH_BYTES);
        AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_WQE_SCRATCH_BYTES, 0);
        constexpr uint32_t SYNC_ID = 0;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen =
            repeat * stride * Catlass::SizeOfBits<ElementSrc>::value / Catlass::SizeOfBits<uint8_t>::value;
        aclshmemx_udma_put_nbi((__gm__ uint8_t*)dstTensor.GetPhyAddr(), (__gm__ uint8_t*)srcTensor.GetPhyAddr(),
                               (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), messageLen, peerIdx, SYNC_ID);

        aclshmemx_udma_quiet(peerIdx);
    }
};

}  // namespace Catccos::Comm::Tile

#endif  // CATCCOS_TILE_REMOTE_COPY_HPP
