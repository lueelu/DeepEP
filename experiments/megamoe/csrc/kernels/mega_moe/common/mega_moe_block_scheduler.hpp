// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "catlass/gemm/block/block_swizzle.hpp"
#include "tensor_api/tensor.h"

namespace MegaMoeImpl::GmmKernel {
// Shared by W4A8 and generic GEMM. CATLASS returns tile indices; the existing Wave stages consume element offsets.
class BlockScheduler {
public:
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = ProblemShape;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t>;
    struct Params {
        AscendC::Te::Shape<int64_t, int64_t> tileShape;
    };
    CATLASS_DEVICE BlockScheduler(const ProblemShape& shape, const Params& params)
        : tileM_(AscendC::Std::get<0>(params.tileShape)),
          tileN_(AscendC::Std::get<1>(params.tileShape)),
          swizzle_(
              {static_cast<uint32_t>(AscendC::Std::get<0>(shape)), static_cast<uint32_t>(AscendC::Std::get<1>(shape)),
               static_cast<uint32_t>(AscendC::Std::get<2>(shape))},
              {tileM_, tileN_})
    {
    }
    CATLASS_DEVICE uint32_t GetTileNum() const
    {
        return swizzle_.GetCoreLoops();
    }
    CATLASS_DEVICE BlockCoord GetBlockCoord(uint32_t index)
    {
        const auto coord = swizzle_.GetBlockCoord(index);
        return {coord.m() * tileM_, coord.n() * tileN_, 0};
    }
    CATLASS_DEVICE BlockShape GetBlockShape(const BlockCoord& coord)
    {
        const auto shape =
            swizzle_.GetActualBlockShape({static_cast<uint32_t>(AscendC::Std::get<0>(coord)) / tileM_,
                                          static_cast<uint32_t>(AscendC::Std::get<1>(coord)) / tileN_, 0});
        return {shape.m(), shape.n(), shape.k()};
    }

private:
    uint32_t tileM_, tileN_;
    Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0> swizzle_;
};
}  // namespace MegaMoeImpl::GmmKernel
