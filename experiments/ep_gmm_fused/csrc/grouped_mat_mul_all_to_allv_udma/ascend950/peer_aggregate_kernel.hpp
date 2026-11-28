/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See csrc/catccos_overlay/LICENSE.
 */
#ifndef DEEPEP_EP_GMM_PEER_AGGREGATE_KERNEL_HPP
#define DEEPEP_EP_GMM_PEER_AGGREGATE_KERNEL_HPP

#include "large_ep_pipeline_launch.hpp"
#include "peer_aggregate_round_schedule.hpp"

namespace deepep::ep_gmm_fused {
namespace PeerAggregate {

struct OutputSegment {
    uint32_t sourceRow;
    uint32_t targetRow;
    uint32_t rows;
};

template <uint32_t Capacity>
struct SegmentedOutputFor {
    __gm__ bfloat16_t* stage;
    uint32_t columns;
    uint32_t columnOffset;
    uint32_t tileColumns;
    uint32_t segmentCount;
    OutputSegment segments[Capacity];
};

struct CopySegments {
    template <uint32_t Capacity, class TensorSrc>
    CATLASS_DEVICE void operator()(const SegmentedOutputFor<Capacity>& dst, const TensorSrc& src) const
    {
        static_assert(std::is_same_v<typename TensorSrc::Element, float>, "FP32 L0C accumulator required");
        AscendC::GlobalTensor<bfloat16_t> stage;
        stage.SetGlobalBuffer(dst.stage);
        AscendC::DataCopyCO12DstParams copy;
        copy.nSize = dst.tileColumns;
        copy.dstStride = dst.columns;
        // Keep the parent accumulator's N-block stride. Rebuilding an M18
        // layout here would address the wrong columns of the M144 result.
        copy.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
        copy.quantPre = QuantMode_t::F322BF16;
        copy.nz2ndEn = true;
        copy.reluPre = false;
        copy.unitFlag = 0;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        for (uint32_t i = 0; i < dst.segmentCount; ++i) {
            const auto& segment = dst.segments[i];
            copy.mSize = segment.rows;
            const auto sourceOffset = src.layout()(tla::Add(src.coord(), tla::MakeCoord(segment.sourceRow, 0U)));
            const uint64_t targetOffset = static_cast<uint64_t>(segment.targetRow) * dst.columns + dst.columnOffset;
            AscendC::DataCopy(stage[targetOffset], src.data()[sourceOffset], copy);
        }
        // BlockMmad places FIX_M after this entire sequence and waits for that
        // event before the next tile can reuse L0C. No per-fragment UnitFlag.
    }
};

struct TileCopy : LargeEpN128Config::TileCopy {
    template <class TensorDst>
    using CopyL0CToDst = CopySegments;
};

}  // namespace PeerAggregate
}  // namespace deepep::ep_gmm_fused
#endif
