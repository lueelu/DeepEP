// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_CATLASS_GROUPED_MATMUL_H
#define MEGA_MOE_CATLASS_GROUPED_MATMUL_H

#include "../common/mega_moe_types.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "block_mmad_a8w4.hpp"
#include "block_prologue_a8w4.hpp"

namespace MegaMoeImpl::GmmKernel {

// Shared CATLASS physical layout choices for grouped matmul.
struct CatlassGroupedMatmulLayouts {
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::RowMajor;
    using LayoutScaleA = Catlass::layout::RowMajor;
    using LayoutScaleB = Catlass::layout::ColumnMajor;
    using LayoutBias = Catlass::layout::RowMajor;
    using LayoutDequantizedB = Catlass::layout::nZ;
    template <bool IsA8W4, bool IsWeightNZ>
    using LayoutB =
        std::conditional_t<IsA8W4, Catlass::layout::Weight4BitnZ,
                           std::conditional_t<IsWeightNZ, Catlass::layout::nZ, Catlass::layout::ColumnMajor>>;
};

// Shared AIC/AIV0 pipeline mechanics; projection semantics belong to GMM1/GMM2 below.
namespace detail {
template <class Config>
class A8W4MatmulPipeline {
    using C = typename Config::KernelConfig;
    using A = typename C::ElementAType;
    using W = typename C::ElementBType;
    using Out = typename C::ElementCType;
    using SA = typename C::ElementMxScaleAType;
    using SB = typename C::ElementMxScaleBType;
    using Arch = Catlass::Arch::Ascend950;
    using LayoutA = CatlassGroupedMatmulLayouts::LayoutA;
    using LayoutB = CatlassGroupedMatmulLayouts::LayoutB<true, true>;
    using LayoutC = CatlassGroupedMatmulLayouts::LayoutC;
    using LayoutScaleA = CatlassGroupedMatmulLayouts::LayoutScaleA;
    using LayoutScaleB = CatlassGroupedMatmulLayouts::LayoutScaleB;
    using LayoutDequantizedB = CatlassGroupedMatmulLayouts::LayoutDequantizedB;
    using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using ScaleALayout = decltype(tla::MakeMxScaleLayout<SA, LayoutScaleA, false>(uint32_t{}, uint32_t{}));
    using ScaleBLayout = decltype(tla::MakeMxScaleLayout<SB, LayoutScaleB, true>(uint32_t{}, uint32_t{}));
    using TileCopy =
        Catlass::Gemm::Tile::PackedMxA8W4TileCopyTla<Arch, A, LayoutA, W, LayoutB, A, LayoutDequantizedB, SA,
                                                     ScaleALayout, SB, ScaleBLayout, Out, LayoutC, void, false,
                                                     Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using Policy = Catlass::Gemm::MmadA8W4Mx<Arch, true, false, 16, 1, 2, 2, 2, 2>;
    using Mmad = Catlass::Gemm::Block::MegaMoeBlockMmadA8W4<Policy, L1, L0, A, A, Out, void, TileCopy>;
    using Prologue =
        Catlass::Gemm::Block::MegaMoeBlockPrologueA8W4<Arch, 2, Catlass::Gemm::GemmType<W, LayoutB>,
                                                       Catlass::Gemm::GemmType<A, LayoutDequantizedB>, L1, TileCopy>;
    using Resource = Catlass::Gemm::Block::MegaMoeResource;

    struct Cube : Mmad {
        __aicore__ inline Cube(Resource& resource) : Mmad(resource, 0, false) {}
        __aicore__ inline void Begin()
        {
            this->Activate();
            for (uint16_t i = 0; i < 2; ++i) {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(8 + i);
            }
        }
        __aicore__ inline void End(bool outputDrained)
        {
            // Finish GM stores before the base restores the MM layout and the next slice reuses L0C.
            if (!outputDrained) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            }
            this->Deactivate();
        }
    };
    struct Vector {
        typename Prologue::Params params;
        Prologue op;
        __aicore__ inline Vector(Resource& resource)
            : params{L1{}, tla::MakeLayout<W, LayoutB>(uint32_t{}, uint32_t{}), false, resource}, op(params, false)
        {
        }
        __aicore__ inline void Begin()
        {
            op.Activate();
        }
        __aicore__ inline void End(bool /*outputDrained*/)
        {
            // Consume the final free-buffer tokens before another slice reuses the flags/L1.
            for (uint16_t i = 0; i < 2; ++i) {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(8 + i);
            }
            op.Deactivate();
        }
    };
    Resource resource_;
    std::conditional_t<g_coreType == AscendC::AIC, Cube, Vector> block_;
    Config config_{};
    struct OperandAddresses {
        GM_ADDR aGlobal = nullptr;
        GM_ADDR bGlobal = nullptr;
        GM_ADDR aScaleGlobal = nullptr;
        GM_ADDR bScaleGlobal = nullptr;
    } addr_;
    GM_ADDR output_ = nullptr;
    bool active_ = false;

    template <class Element, class Layout>
    __aicore__ inline auto Tensor(GM_ADDR address, const Layout& layout)
    {
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(address));
        return tla::MakeTensor(gm, layout, Catlass::Arch::PositionGM{});
    }

public:
    // Construct fixed buffer views once, without touching shared hardware events.
    __aicore__ inline A8W4MatmulPipeline() : block_(resource_) {}
    __aicore__ inline A8W4MatmulPipeline(const Config& config, const GMMAddrInfo& addr, GM_ADDR output)
        : A8W4MatmulPipeline()
    {
        UpdateProblem(config, addr, output);
        Begin();
    }
    A8W4MatmulPipeline(const A8W4MatmulPipeline&) = delete;
    A8W4MatmulPipeline& operator=(const A8W4MatmulPipeline&) = delete;
    __aicore__ inline ~A8W4MatmulPipeline()
    {
        End();
    }

    // Scheduling policies (shared expert, Wave, gate/up) stay in the caller.
    // Only dimensions and GM addresses are consumed by this block adapter.
    template <class ProblemConfig>
    __aicore__ inline void UpdateProblem(const ProblemConfig& config, const GMMAddrInfo& addr, GM_ADDR output)
    {
        using Next = typename ProblemConfig::KernelConfig;
        static_assert(std::is_same_v<A, typename Next::ElementAType> &&
                          std::is_same_v<W, typename Next::ElementBType> &&
                          std::is_same_v<Out, typename Next::ElementCType> &&
                          std::is_same_v<SA, typename Next::ElementMxScaleAType> &&
                          std::is_same_v<SB, typename Next::ElementMxScaleBType>,
                      "Cannot change GMM element types");
        config_.m = config.m;
        config_.n = config.n;
        config_.k = config.k;
        config_.scaleK = config.scaleK;
        addr_ = {addr.aGlobal, addr.bGlobal, addr.aScaleGlobal, addr.bScaleGlobal};
        output_ = output;
        if constexpr (g_coreType == AscendC::AIV) {
            block_.params.layoutPrologueB = tla::MakeLayout<W, LayoutB>(config.k, config.n);
        }
    }

    __aicore__ inline void Begin()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            if (AscendC::GetSubBlockIdx() != 0) {
                return;
            }
        }
        if (!active_) {
            block_.Begin();
            active_ = true;
        }
    }

    // Preserve the old slice boundary: GMM2/other stages may reuse the same
    // event IDs and L1/L0/UB storage while this C++ object remains alive.
    __aicore__ inline void End(bool outputDrained = false)
    {
        if (active_) {
            block_.End(outputDrained);
            active_ = false;
        }
    }

protected:
    __aicore__ inline void ComputeProjection(uint32_t m, uint32_t weightN, uint32_t rows, uint32_t cols,
                                             uint32_t outputN)
    {
        auto a = Tensor<A>(addr_.aGlobal, tla::MakeLayout<A, LayoutA>(config_.m, config_.k));
        auto sa =
            Tensor<SA>(addr_.aScaleGlobal, tla::MakeMxScaleLayout<SA, LayoutScaleA, false>(config_.m, config_.scaleK));
        auto sb =
            Tensor<SB>(addr_.bScaleGlobal, tla::MakeMxScaleLayout<SB, LayoutScaleB, true>(config_.scaleK, config_.n));
        auto c = Tensor<Out>(output_, tla::MakeLayout<Out, LayoutC>(config_.m, config_.n));
        auto tileA = tla::GetTile(a, tla::MakeCoord(m, 0), tla::MakeShape(rows, config_.k));
        auto tileSA = tla::GetTile(sa, tla::MakeCoord(m, 0), tla::MakeShape(rows, config_.scaleK));
        auto tileSB = tla::GetTile(sb, tla::MakeCoord(0, weightN), tla::MakeShape(config_.scaleK, cols));
        auto tileC = tla::GetTile(c, tla::MakeCoord(m, outputN), tla::MakeShape(rows, cols));
        block_(tileA, tileC, Catlass::GemmCoord{rows, cols, config_.k}, tileSA, tileSB, config_.m < 256U);
    }

    __aicore__ inline void DequantizeWeight(uint32_t weightN, uint32_t rows, uint32_t cols)
    {
        auto b = Tensor<W>(addr_.bGlobal, block_.params.layoutPrologueB);
        auto tileB = tla::GetTile(b, tla::MakeCoord(0, weightN), tla::MakeShape(config_.k, cols));
        block_.op(tileB, Catlass::GemmCoord{rows, cols, config_.k}, block_.params, config_.m < 256U);
    }
};

}  // namespace detail

// Routed GMM1 consumes offline-packed [gate128, up128] in one N256 projection.
// Shared GMM1 retains planar weights and its independently scheduled N192 tiles.
// The object may remain alive across expert slices; Begin/End delimit scratch ownership.
template <class Config>
class CatlassGmm1 : private detail::A8W4MatmulPipeline<Config> {
    bool fused_ = false;
    bool planarSingle_ = false;
    uint32_t halfN_ = 0;
    using Pipeline = detail::A8W4MatmulPipeline<Config>;

public:
    __aicore__ inline CatlassGmm1() = default;
    using Pipeline::Begin;
    using Pipeline::End;

    template <class ProblemConfig>
    __aicore__ inline void UpdateProblem(const ProblemConfig& config, const GMMAddrInfo& addr)
    {
        Pipeline::UpdateProblem(config, addr, addr.gmm1OutGlobal);
        fused_ = ProblemConfig::FUSED_GATE_UP;
        planarSingle_ = config.sharedPolicy.decode;
        halfN_ = config.outputN;
    }
    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t rows, uint32_t cols)
    {
        // Logical activation columns map to contiguous gate/up pairs in GM.
        if (fused_) {
            Pipeline::ComputeProjection(m, n * 2U, rows, cols * 2U, n * 2U);
        } else {
            Pipeline::ComputeProjection(m, n, rows, cols, n);
            if (!planarSingle_) {
                Pipeline::ComputeProjection(m, n + halfN_, rows, cols, n + halfN_);
            }
        }
    }
    __aicore__ inline void Dequantize(uint32_t n, uint32_t rows, uint32_t cols)
    {
        if (fused_) {
            Pipeline::DequantizeWeight(n * 2U, rows, cols * 2U);
        } else {
            Pipeline::DequantizeWeight(n, rows, cols);
            if (!planarSingle_) {
                Pipeline::DequantizeWeight(n + halfN_, rows, cols);
            }
        }
    }
};

// GMM2 owns the down projection and always writes row-major output.
template <class Config>
class CatlassGmm2 : private detail::A8W4MatmulPipeline<Config> {
    using Pipeline = detail::A8W4MatmulPipeline<Config>;

public:
    __aicore__ inline CatlassGmm2(const Config& config, const GMMAddrInfo& addr)
        : Pipeline(config, addr, addr.gmm2OutGlobal)
    {
    }

    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t rows, uint32_t cols)
    {
        Pipeline::ComputeProjection(m, n, rows, cols, n);
    }
    __aicore__ inline void Dequantize(uint32_t n, uint32_t rows, uint32_t cols)
    {
        Pipeline::DequantizeWeight(n, rows, cols);
    }
};

}  // namespace MegaMoeImpl::GmmKernel
#endif
