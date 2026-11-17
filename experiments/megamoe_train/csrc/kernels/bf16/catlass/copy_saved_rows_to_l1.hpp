// Copyright (c) 2026, Lu Lu
// Modified by xixing, guofanfeng 2026

#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/tensor.hpp"

namespace MegaMoeBF16BackwardCatlassImpl {

// GetTile retains the underlying tensor, so each nested K tile carries the
// same logical-to-saved row map without modifying the CATLASS MMAD pipeline.
struct SavedRowGlobalTensor : AscendC::GlobalTensor<bfloat16_t> {
    __gm__ uint32_t* rowMap = nullptr;
    uint64_t rowBase = 0;
};

template <class TensorSrc, class TensorL1>
struct CopySavedRowsToL1 {
    template <class Dst, class Src>
    CATLASS_DEVICE void operator()(Dst const& dst, Src const& src)
    {
        using Storage = std::remove_cv_t<std::remove_reference_t<decltype(src.data())>>;
        if constexpr (std::is_same_v<Storage, SavedRowGlobalTensor>) {
            if (src.data().rowMap != nullptr) {
                CopyMapped(dst, src);
                return;
            }
            AscendC::GlobalTensor<bfloat16_t> data;
            data.SetGlobalBuffer(const_cast<__gm__ bfloat16_t*>(src.data().GetPhyAddr()));
            auto plain = tla::MakeTensor(data, src.layout(), src.coord(), Catlass::Arch::PositionGM{});
            Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(plain), Dst>{}(dst, plain);
        } else {
            Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, Src, Dst>{}(dst, src);
        }
    }

private:
    template <class Dst, class Src>
    CATLASS_DEVICE void CopyMapped(Dst const& dst, Src const& src)
    {
        constexpr bool rowMajor = tla::detail::isRowMajor<typename Src::Layout>::value;
        const uint32_t rows = rowMajor ? tla::get<0>(src.originShape()) : tla::get<1>(src.originShape());
        const uint32_t columns = rowMajor ? tla::get<1>(src.originShape()) : tla::get<0>(src.originShape());
        const uint32_t logicalBegin = rowMajor ? tla::get<0>(src.coord()) : tla::get<1>(src.coord());
        const uint32_t column = rowMajor ? tla::get<1>(src.coord()) : tla::get<0>(src.coord());
        const uint32_t stride = rowMajor ? tla::get<0>(src.stride()) : tla::get<1>(src.stride());
        for (uint32_t offset = 0; offset < rows;) {
            const uint64_t index = src.data().rowBase + logicalBegin + offset;
            const uint32_t physical = AscendC::ReadGmByPassDCache(src.data().rowMap + 2U * index);
            const uint32_t run = AscendC::ReadGmByPassDCache(src.data().rowMap + 2U * index + 1U);
            const uint32_t take = run < rows - offset ? run : rows - offset;
            AscendC::Nd2NzParams p{};
            p.ndNum = 1;
            p.nValue = take;
            p.dValue = columns;
            p.srcDValue = stride;
            uint64_t destination;
            if constexpr (rowMajor) {
                p.dstNzC0Stride = tla::get<1, 1>(dst.stride()) / 16U;
                p.dstNzNStride = tla::get<0, 0>(dst.stride()) / 16U;
                destination = dst.layout()(tla::MakeCoord(tla::get<0>(dst.coord()) + offset, tla::get<1>(dst.coord())));
            } else {
                p.dstNzC0Stride = tla::get<0, 1>(dst.stride()) / 16U;
                p.dstNzNStride = tla::get<1, 0>(dst.stride()) / 16U;
                destination = dst.layout()(tla::MakeCoord(tla::get<0>(dst.coord()), tla::get<1>(dst.coord()) + offset));
            }
            const uint64_t source = (physical - src.data().rowBase) * stride + column;
            AscendC::DataCopy(dst.data()[destination], src.data()[source], p);
            offset += take;
        }
    }
};

}  // namespace MegaMoeBF16BackwardCatlassImpl
