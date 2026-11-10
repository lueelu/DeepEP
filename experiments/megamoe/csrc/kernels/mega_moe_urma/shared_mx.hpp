#pragma once
#include "kernels/mega_moe_common/catlass_gmm/mega_moe_mx_matmul.hpp"

namespace MegaMoeUrma {
// Shared-only FP8xFP8. Raw weights are contiguous ND [N,K], interpreted as
// column-major [K,N]. AIC loads B directly; no AIV0 prologue or UB mailbox.
// One context lives across all tiles/groups in a shared GMM, with UnitFlag.
template <bool First>
class SharedMxGmm {
    using Arch = Catlass::Arch::Ascend950;
    using Row = Catlass::layout::RowMajor;
    using Col = Catlass::layout::ColumnMajor;
    using F = fp8_e4m3fn_t;
    using S = fp8_e8m0_t;
    using C = bfloat16_t;
    using SA = decltype(tla::MakeMxScaleLayout<S, Row, false>(uint32_t{}, uint32_t{}));
    using SB = decltype(tla::MakeMxScaleLayout<S, Col, true>(uint32_t{}, uint32_t{}));
    using Copy = Catlass::Gemm::Tile::PackedMxTileCopyTla<Arch, F, Row, F, Col, S, SA, S, SB, C, Row, void>;
    using Policy = Catlass::Gemm::MmadMx<Arch, true, 2, 1, false, 2, 2, 2, 2>;
    using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using Block = Catlass::Gemm::Block::MegaMoeBlockMmadMx<Policy, L1, L0, F, F, C, void, Copy>;
    Catlass::Gemm::Block::MegaMoeResource resource_;
    Block block_;
    UbxMegaMoeImpl::GMMAddrInfo addr_{};
    uint32_t m_ = 0, n_ = 0, k_ = 0, scaleK_ = 0;
    template <class Element, class Layout>
    __aicore__ inline auto View(GM_ADDR p, const Layout& layout)
    {
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p));
        return tla::MakeTensor(gm, layout, Catlass::Arch::PositionGM{});
    }
    __aicore__ inline void Projection(uint32_t m, uint32_t n, uint32_t rows, uint32_t columns)
    {
        auto av = View<F>(addr_.aGlobal, tla::MakeLayout<F, Row>(m_, k_));
        auto bv = View<F>(addr_.bGlobal, tla::MakeLayout<F, Col>(k_, n_));
        auto sav = View<S>(addr_.aScaleGlobal, tla::MakeMxScaleLayout<S, Row, false>(m_, scaleK_));
        auto sbv = View<S>(addr_.bScaleGlobal, tla::MakeMxScaleLayout<S, Col, true>(scaleK_, n_));
        auto cv = View<C>(First ? addr_.gmm1OutGlobal : addr_.gmm2OutGlobal, tla::MakeLayout<C, Row>(m_, n_));
        auto a = tla::GetTile(av, tla::MakeCoord(m, 0), tla::MakeShape(rows, k_));
        auto b = tla::GetTile(bv, tla::MakeCoord(0, n), tla::MakeShape(k_, columns));
        auto sa = tla::GetTile(sav, tla::MakeCoord(m, 0), tla::MakeShape(rows, scaleK_));
        auto sb = tla::GetTile(sbv, tla::MakeCoord(0, n), tla::MakeShape(scaleK_, columns));
        auto c = tla::GetTile(cv, tla::MakeCoord(m, n), tla::MakeShape(rows, columns));
        block_(a, b, c, Catlass::GemmCoord{rows, columns, k_}, sa, sb);
    }

public:
    __aicore__ inline SharedMxGmm() : block_(resource_) {}
    template <class Problem>
    __aicore__ inline void UpdateProblem(const Problem& p, const UbxMegaMoeImpl::GMMAddrInfo& addr)
    {
        m_ = p.m;
        n_ = p.n;
        k_ = p.k;
        scaleK_ = p.scaleK;
        addr_ = addr;
    }
    __aicore__ inline void Begin() {}
    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t rows, uint32_t cols)
    {
        Projection(m, n, rows, cols);
        if constexpr (First) Projection(m, n + n_ / 2U, rows, cols);
    }
    __aicore__ inline void End(bool = false)
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
    }
};
}  // namespace MegaMoeUrma
