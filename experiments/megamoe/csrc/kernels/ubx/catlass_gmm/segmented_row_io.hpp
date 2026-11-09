// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "segmented_row_layout.hpp"
#include "catlass/catlass.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace UbxMegaMoeImpl::GmmKernel {

// The default policy compiles out all segmented paths in the main MegaMoE
// instantiations. It does not add runtime branches or alter their unit flags.
struct ContiguousRowIO {
    static constexpr bool kGatherInput = false;
    static constexpr bool kScatterOutput = false;
    static constexpr bool kMayOutputToUb = false;
};

// Segment is a caller-owned descriptor with sourceRank, receiveRow, returnRow,
// tileRow and rows fields. This layer has no dependency on the URMA wire ABI.
template <class Segment>
struct GmSegmentSource {
    struct Cursor {
        const __gm__ Segment* segments;
        uint32_t count;
        uint32_t index = 0U;
        __aicore__ inline bool Next(Segment& value)
        {
            if (index == count) return false;
            const auto* segment = segments + index++;
            value.sourceRank = segment->sourceRank;
            value.receiveRow = segment->receiveRow;
            value.returnRow = segment->returnRow;
            value.tileRow = segment->tileRow;
            value.rows = segment->rows;
            return true;
        }
    };
    __aicore__ inline Cursor MakeCursor(const __gm__ Segment* segments, uint32_t count) const
    {
        return {segments, count, 0U};
    }
};

// A provider may instead derive the current group's segments from immutable
// count prefixes. The default provider retains the original GM-table policy.
template <class Segment, bool GatherInput, bool ScatterOutput, class SegmentSource = GmSegmentSource<Segment>>
struct SegmentedRowIO {
    static constexpr bool kGatherInput = GatherInput;
    static constexpr bool kScatterOutput = ScatterOutput;
    static constexpr bool kMayOutputToUb = false;

    const __gm__ Segment* segments = nullptr;
    uint32_t count = 0;
    uint32_t rank = 0;
    GM_ADDR inputX = nullptr;
    // Scale rows must already be contiguous in the caller's aScaleGlobal.
    // An arbitrary M boundary would put a scale pair at a 2-byte L1 offset,
    // whereas even DataCopyPad<Compact> requires a 32-byte LocalTensor start.
    // These two descriptive fields are retained alongside the input row ABI;
    // the ordinary CATLASS MX-scale loader consumes aScaleGlobal instead.
    GM_ADDR inputScale = nullptr;
    uint32_t inputStride = 0;
    uint32_t scaleStride = 0;
    GM_ADDR remoteOutput = nullptr;
    GM_ADDR selfReturn = nullptr;
    uint32_t outputStride = 0;
    SegmentSource source{};

    template <class TensorL1>
    __aicore__ inline void Gather(TensorL1& dst, uint32_t tileBegin, uint32_t tileRows, uint32_t columnBegin,
                                  uint32_t columns) const
    {
        static_assert(sizeof(typename TensorL1::Element) == 1, "Segmented A gather expects an 8-bit operand");
        AscendC::GlobalTensor<uint8_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(inputX));
        auto output = dst.data().template ReinterpretCast<uint8_t>();
        const uint32_t c0Stride = tla::get<1, 1>(dst.stride()) / 32U;
        auto cursor = source.MakeCursor(segments, count);
        Segment segment{};
        while (cursor.Next(segment)) {
            const auto part = SegmentedRows::Intersect(segment.tileRow, segment.rows, tileBegin, tileRows);
            if (part.rows == 0) {
                continue;
            }
            AscendC::Nd2NzParams params{};
            params.ndNum = 1;
            params.nValue = part.rows;
            params.dValue = columns;
            params.srcNdMatrixStride = 0;
            params.srcDValue = inputStride;
            params.dstNzC0Stride = c0Stride;
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 0;
            // ND2NZ pads D (K here) to a 32-byte C0; N is the exact row
            // count, including 43/85. Its destination must be 32-byte
            // aligned, not M16 aligned. These C0 rows are disjoint even
            // when adjacent segments share an M16 fractal. Do not rely on
            // overlapping writes being ordered: that is explicitly forbidden.
            // CANN API: DataCopy_GMToL1_ND2NZ.md (Nd2NzParams/Constraints).
            const auto offset = SegmentedRows::AOffset(part.tileRow, 0U, tileRows);
            const uint64_t sourceRow = segment.receiveRow + part.segmentRow;
            AscendC::DataCopy(output[offset], input[sourceRow * inputStride + columnBegin], params);
        }
    }

    template <class Copy, class TensorC, class TensorL0C>
    __aicore__ inline void Scatter(Copy& copy, TensorC& prototype, TensorL0C& src, uint32_t tileBegin,
                                   uint32_t tileRows, uint32_t columnBegin, uint32_t columns) const
    {
        using Element = typename TensorC::Element;
        auto cursor = source.MakeCursor(segments, count);
        Segment segment{};
        while (cursor.Next(segment)) {
            const auto part = SegmentedRows::Intersect(segment.tileRow, segment.rows, tileBegin, tileRows);
            if (part.rows == 0) {
                continue;
            }
            const bool self = segment.sourceRank == rank;
            const uint64_t row = (self ? segment.returnRow : segment.receiveRow) + part.segmentRow;
            AscendC::GlobalTensor<Element> output;
            output.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(self ? selfReturn : remoteOutput) +
                                   SegmentedRows::OutputOffset(row, columnBegin, outputStride));
            auto tensor =
                tla::MakeTensor(output, tla::MakeLayout<Element, Catlass::layout::RowMajor>(tileRows, outputStride),
                                Catlass::Arch::PositionGM{});
            auto destination = tla::GetTile(tensor, tla::MakeCoord(0U, 0U), tla::MakeShape(part.rows, columns));
            auto source = tla::GetTile(src, tla::MakeCoord(part.tileRow, 0U), tla::MakeShape(part.rows, columns));
            // L0C row starts are 16 accumulators (64 bytes) apart, satisfying
            // the FIX source alignment even for M=43/85 segment boundaries.
            // The block caller fences MMAD/FIX and keeps L0C owned until ALL
            // stores finish; no segment is allowed to release a unit flag.
            copy(destination, source, 0U);
        }
    }
};

}  // namespace UbxMegaMoeImpl::GmmKernel
