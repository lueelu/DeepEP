/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ALLTOALLV_GMM_EXPERT_ROWS_HPP
#define CATCCOS_DGEMM_KERNEL_ALLTOALLV_GMM_EXPERT_ROWS_HPP

#include "catlass/gemm/block/block_mmad.hpp"
#include "catccos/catccos.hpp"

namespace Catccos::DGemm::Kernel::ExpertRows {

struct Span {
    uint32_t rows;
    uint32_t inputRow;
    uint32_t outputRow;
};

// A tile is a borrowed interval of one expert's source spans. K panels reuse
// the first span and both copy engines see precisely the same logical rows.
struct RowWindow {
    const Span* next;
    uint32_t skip;
    uint32_t rows;

    CATLASS_DEVICE Span Take()
    {
        Span part = *next++;
        part.inputRow += skip;
        part.outputRow += skip;
        part.rows = Min(part.rows - skip, rows);
        rows -= part.rows;
        skip = 0;
        return part;
    }
};

template <uint32_t MaxSources>
struct RowMap {
    Span spans[MaxSources];

    template <class Scheduler>
    CATLASS_DEVICE uint32_t ReadExpert(Scheduler& scheduler, uint32_t expert, uint32_t stage, uint32_t stageRows)
    {
        const uint32_t ranks = scheduler.problemShape.rankSize();
        uint32_t count = 0;
        uint32_t total = 0;
        for (uint32_t peer = 0; peer < ranks; ++peer) {
            const auto index = Scheduler::TableIndex(expert, peer);
            const uint32_t rows = scheduler.commContext.tokenNumList[index];
            if (rows != 0) {
                const uint32_t inputRow = (stage * ranks + peer) * stageRows +
                    static_cast<uint32_t>(scheduler.commContext.tokenOffsetList[index]);
                const uint32_t outputRow = static_cast<uint32_t>(scheduler.outputOffsetList[index] +
                    scheduler.commContext.inGroupOffsetList[index]);
                spans[count++] = {rows, inputRow, outputRow};
                total += rows;
            }
        }
        return total;
    }

    CATLASS_DEVICE RowWindow Slice(uint32_t first, uint32_t rows) const
    {
        const Span* part = spans;
        while (first >= part->rows) {
            first -= part->rows;
            ++part;
        }
        return {part, first, rows};
    }
};

struct Input {
    __gm__ bfloat16_t* base;
    RowWindow window;
    uint32_t stride;
    uint32_t column;
    uint32_t columns;
};

template <class Coord, class Shape>
CATLASS_DEVICE Input GetTile(const Input& input, const Coord& coord, const Shape& shape)
{
    // BlockMmad's GM A prefetch selects K slabs; M already describes one tile.
    return {input.base, input.window, input.stride,
            input.column + static_cast<uint32_t>(tla::get<1>(coord)), static_cast<uint32_t>(tla::get<1>(shape))};
}

struct GatherToL1 {
    template <class L1Tensor>
    CATLASS_DEVICE void operator()(const L1Tensor& dst, const Input& input) const
    {
        AscendC::GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(input.base);
        // RowMajor BF16 -> zN, matching Ascend950 CATLASS GM-to-L1. One
        // logical M-row in this K panel starts on a full 32-byte C0 boundary.
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.dValue = input.columns;
        copy.srcDValue = input.stride;
        copy.dstNzC0Stride = tla::get<1, 1>(dst.stride()) / 16;
        copy.dstNzNStride = tla::get<0, 0>(dst.stride()) / 16;
        uint32_t row = 0;
        auto window = input.window;
        while (window.rows != 0) {
            const auto part = window.Take();
            copy.nValue = part.rows;
            const auto l1Offset = dst.layout()(tla::MakeCoord(row, 0U));
            AscendC::DataCopy(dst.data()[l1Offset],
                              gm[uint64_t(part.inputRow) * input.stride + input.column], copy);
            row += part.rows;
        }
    }
};

struct Output {
    __gm__ bfloat16_t* base;
    RowWindow window;
    uint32_t stride;
    uint32_t column;
    uint32_t columns;
};

struct ScatterFromL0C {
    template <class L0Tensor>
    CATLASS_DEVICE void operator()(const Output& output, const L0Tensor& src) const
    {
        AscendC::GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(output.base);
        AscendC::DataCopyCO12DstParams copy{};
        copy.nSize = output.columns;
        copy.dstStride = output.stride;
        copy.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
        copy.quantPre = QuantMode_t::F322BF16;
        copy.nz2ndEn = true;
        copy.unitFlag = 0;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        uint32_t row = 0;
        auto window = output.window;
        while (window.rows != 0) {
            const auto part = window.Take();
            uint32_t rows = part.rows;
            while (window.rows != 0 && window.next->outputRow == part.outputRow + rows) {
                rows += window.Take().rows;
            }
            copy.mSize = rows;
            const auto l0Offset = src.layout()(tla::Add(src.coord(), tla::MakeCoord(row, 0U)));
            AscendC::DataCopy(gm[uint64_t(part.outputRow) * output.stride + output.column],
                              src.data()[l0Offset], copy);
            row += rows;
        }
    }
};

template <class OriginalCopy>
struct TileCopy : OriginalCopy {
    template <class Tensor> using CopyGmToL1A = GatherToL1;
    template <class Tensor> using CopyL0CToDst = ScatterFromL0C;
};

template <class OriginalBlock>
using Block = Catlass::Gemm::Block::BlockMmadTla<
    Catlass::Gemm::MmadPingpong<typename OriginalBlock::ArchTag, false>,
    typename OriginalBlock::L1TileShape, typename OriginalBlock::L0TileShape,
    typename OriginalBlock::ElementA, typename OriginalBlock::ElementB,
    typename OriginalBlock::ElementC, void, TileCopy<typename OriginalBlock::TileCopy>>;

}  // namespace Catccos::DGemm::Kernel::ExpertRows

#endif
