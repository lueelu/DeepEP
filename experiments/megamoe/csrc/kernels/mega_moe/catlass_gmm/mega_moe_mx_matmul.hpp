// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_CATLASS_MX_MATMUL_H
#define MEGA_MOE_CATLASS_MX_MATMUL_H

#include "tensor_api/tensor.h"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "block_mmad_mx.hpp"

namespace MegaMoeImpl::GmmKernel {

// The stage supplies either GM or AIV0 UB output. Select the CATLASS Fixpipe
// primitive by destination; UB must explicitly target AIV0, without splitting.
template <class Base>
struct MxStageTileCopy : Base {
    template <class TensorC>
    struct CopyL0CToDst {
        template <class TensorL0C>
        __aicore__ inline void operator()(const TensorC& dst, const TensorL0C& src, uint8_t unitFlag = 0)
        {
            if constexpr (TensorC::position == AscendC::TPosition::VECCALC) {
                Catlass::Gemm::Tile::CopyL0CToUBTla<Catlass::Arch::Ascend950, TensorL0C, TensorC,
                                                    Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                                                    Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>
                    copy;
                copy(dst, src, false, unitFlag);
            } else {
                typename Base::template CopyL0CToDst<TensorC> copy;
                copy(dst, src, unitFlag);
            }
        }
    };
};

// Adapter for the stage's tensor-API views. Slices already carry their byte
// offsets; reconstruct TLA views with the parent strides, never compact a tile.
template <class Config>
class CatlassMxMatmul {
    using A = typename Config::ElementAType;
    using B = typename Config::ElementBType;
    using C = typename Config::ElementCType;
    using SA = typename Config::ElementMxScaleAType;
    using SB = typename Config::ElementMxScaleBType;
    using Arch = Catlass::Arch::Ascend950;
    using Row = Catlass::layout::RowMajor;
    using Col = Catlass::layout::ColumnMajor;
    using BLayout = std::conditional_t<Config::IS_WEIGHT_NZ, Catlass::layout::nZ, Catlass::layout::ColumnMajor>;
    using SALayout = decltype(tla::MakeMxScaleLayout<SA, Row, false>(uint32_t{}, uint32_t{}));
    // Weight scales are [N, ceil(K/64), 2]: contiguous K-scale pairs for each N.
    using SBLayout = decltype(tla::MakeMxScaleLayout<SB, Col, true>(uint32_t{}, uint32_t{}));
    using TileCopy = MxStageTileCopy<
        Catlass::Gemm::Tile::PackedMxTileCopyTla<Arch, A, Row, B, BLayout, SA, SALayout, SB, SBLayout, C, Row, void>>;
    using Policy = Catlass::Gemm::MmadMx<Arch, true, 2, 1, false, 2, 2, 2, 2>;
    using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using Block = Catlass::Gemm::Block::MegaMoeBlockMmadMx<Policy, L1, L0, A, B, C, void, TileCopy>;
    Catlass::Gemm::Block::MegaMoeResource resource_;
    Block block_;
    uint32_t k_ = 0;
    uint32_t n_ = 0;

    template <class Element, class Tensor, class Layout>
    __aicore__ inline auto GlobalView(const Tensor& src, const Layout& layout)
    {
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(src.Data().Get()));
        const bool bypass =
            src.Engine().GetCacheMode() == static_cast<uint8_t>(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        gm.SetL2CacheHint(bypass ? AscendC::CacheMode::CACHE_MODE_DISABLE : AscendC::CacheMode::CACHE_MODE_NORMAL);
        return tla::MakeTensor(gm, layout, Catlass::Arch::PositionGM{});
    }

public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = BlockShape;
    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint32_t l1BufNum = 2;
    };

    __aicore__ inline CatlassMxMatmul() : block_(resource_) {}
    __aicore__ inline ~CatlassMxMatmul()
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    }
    __aicore__ inline void Init(const ProblemShape& shape, const BlockShape&, const L1Params&, bool, bool)
    {
        n_ = AscendC::Std::get<1>(shape);
        k_ = AscendC::Std::get<2>(shape);
    }

    template <class TA, class TB, class TSA, class TSB, class TBias, class TC>
    __aicore__ inline void operator()(TA& a, TB& b, TSA& sa, TSB& sb, TBias&, TC& c, const BlockShape& shape)
    {
        const uint32_t rows = AscendC::Std::get<0>(shape);
        const uint32_t cols = AscendC::Std::get<1>(shape);
        const uint32_t k = AscendC::Std::get<2>(shape);
        const uint32_t scaleK = ::CeilDiv(k_, 64U) * 2U;
        auto av = GlobalView<A>(a, tla::MakeLayout<A, Row>(rows, k_));
        auto bv = GlobalView<B>(b, tla::MakeLayout<B, BLayout>(k_, n_));
        auto sav = GlobalView<SA>(sa, tla::MakeMxScaleLayout<SA, Row, false>(rows, scaleK));
        auto sbv = GlobalView<SB>(sb, tla::MakeMxScaleLayout<SB, Col, true>(scaleK, n_));
        auto at = tla::GetTile(av, tla::MakeCoord(0, 0), tla::MakeShape(rows, k));
        auto bt = tla::GetTile(bv, tla::MakeCoord(0, 0), tla::MakeShape(k, cols));
        auto sat = tla::GetTile(sav, tla::MakeCoord(0, 0), tla::MakeShape(rows, scaleK));
        auto sbt = tla::GetTile(sbv, tla::MakeCoord(0, 0), tla::MakeShape(scaleK, cols));
        if constexpr (std::is_same_v<AscendC::Te::GetMemLocation<TC>, AscendC::Te::Location::UB>) {
            AscendC::LocalTensor<C> ub(AscendC::TPosition::VECCALC, reinterpret_cast<uint64_t>(c.Data().Get()),
                                       rows * 256U);
            auto cv = tla::MakeTensor(ub, tla::MakeLayout<C, Row>(rows, 256U), Catlass::Arch::PositionUB{});
            auto ct = tla::GetTile(cv, tla::MakeCoord(0, 0), tla::MakeShape(rows, cols));
            block_(at, bt, ct, Catlass::GemmCoord{rows, cols, k}, sat, sbt);
        } else {
            auto cv = GlobalView<C>(c, tla::MakeLayout<C, Row>(rows, n_));
            auto ct = tla::GetTile(cv, tla::MakeCoord(0, 0), tla::MakeShape(rows, cols));
            block_(at, bt, ct, Catlass::GemmCoord{rows, cols, k}, sat, sbt);
        }
    }
};
}  // namespace MegaMoeImpl::GmmKernel
#endif
