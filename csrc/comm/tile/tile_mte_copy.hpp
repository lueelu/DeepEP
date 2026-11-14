/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See csrc/comm/LICENSE.catccos for the full text of the License.
 */
#pragma once
#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"
#include "shmem.h"

namespace deepep::comm::tile {
// Dynamic row-major Put tile. The block chooses a UB-fitting tile and waits
// for MTE3 completion before reusing its UB or publishing ready.
template <class SrcType, class DstType = SrcType>
struct TileMteCopy {
    using ElementSrc = typename SrcType::Element;
    using ElementDst = typename DstType::Element;
    using LayoutSrc = typename SrcType::Layout;
    using LayoutDst = typename DstType::Layout;
    static_assert(std::is_same_v<ElementSrc, ElementDst>);
    static_assert(std::is_same_v<LayoutSrc, Catlass::layout::RowMajor> &&
                  std::is_same_v<LayoutDst, Catlass::layout::RowMajor>);
    static_assert(Catlass::SizeOfBits<ElementSrc>::value == sizeof(ElementSrc) * 8,
                  "Packed sub-byte elements are unsupported");

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& dstTensor, LayoutDst const& dstLayout,
        AscendC::GlobalTensor<ElementSrc> const& srcTensor, LayoutSrc const& srcLayout,
        Catlass::MatrixCoord const& copyShape, AscendC::LocalTensor<ElementSrc> const& tmpUb,
        uint32_t copyEventId, uint32_t peerIdx)
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

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventId);

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
} // namespace deepep::comm::tile
