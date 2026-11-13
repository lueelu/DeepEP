// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*
 * BF16 grouped GEMM wrappers for the DeepEP MegaMoe backward pipeline.
 *
 * Expert-local stages use N=2F and hidden width K. Canonical weights are
 * W1 [N,K] and W2 [K,F]; legacy uses their transposed physical layouts.
 *
 *   P0-C GMM1 recompute : canonical Xpacked @ W1.T -> Gate/Up in GM;
 *                         legacy Xpacked @ W1. Only save=none recomputes.
 *   P1    GMM2 Grad     : canonical dHp = dO @ W2; legacy dO @ W2.T.
 *                         BF16 dHp goes to saved-mode GM or save=none UB.
 *   P2    GMM1 Grad     : canonical dX = dZ @ W1; legacy dZ @ W1.T.
 *   P3    GMM1 dW       : canonical dZ.T @ Xpacked -> [N,K], or legacy
 *                         Xpacked.T @ dZ -> [K,N].
 *   P4    GMM2 dW       : canonical dO.T @ Hp -> [K,F], or legacy
 *                         Hp.T @ dO -> [F,K].
 * h=SiLU_FP32(gate)*up and dh=p*dHp stay FP32; Hp=BF16(h*p). The P1
 * epilogue derives dZ from dh and reduces FP32 dHp*h into row partials.
 * Saved modes run each expert's dW2 before P2 reuses dO as dX; none
 * retains dO until P4. Stage labels describe dependencies, not a barrier.
 * Weight gradients use ColumnMajor A and A5's native transpose-A path:
 * GM ColumnMajor -> L1 nZ -> L0A zN (copy_l1_to_l0a.hpp). No AIV
 * transpose materialization is required; operand layouts follow transW.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_HPP
#define DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/mc2_kernel_utils.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
// Exclude unused HSTU specializations that collide in tuple lookup with CANN 9.2.
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "epilogue/block_epilogue_swiglu_backward_bf16.hpp"
#include "catlass/copy_l0c_to_ub.hpp"
#include "layout/mega_moe_context.hpp"
#include "schedule/backward_dispatch_ready.hpp"
#include "schedule/backward_dispatch_schedule.hpp"
#include "routing/mega_moe_combine_send.hpp"

namespace MegaMoeBF16BackwardCatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using RowMajor = Catlass::layout::RowMajor;
using ColumnMajor = Catlass::layout::ColumnMajor;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
static_assert(L1_TILE_M == ProducerPhaseMath::kRows, "Backward Dispatch scheduling must use the GEMM M tile size");
constexpr uint32_t SWIGLU_N_HALF = MegaMoeBF16Training::kSwigluNHalf;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t DUAL_AIV_TARGET = 2;
static_assert(2U * MAX_SINGLE_MN * sizeof(bfloat16_t) <= ArchTag::UB_SIZE, "The two BF16 GMM tiles must fit in A5 UB");

// Save=none P1 uses one BlockMmad instance (one CATLASS L1/L0 pipeline),
// but Fixpipe must be able to select either paired AIV at runtime. Encode the
// target AIV in the destination tensor's row coordinate: row 0 means AIV0,
// row L1_TILE_M means AIV1. The physical UB offset remains zero on the chosen
// AIV; the extra logical row band is only a control marker for this copier.
template <class TensorSrc, class TensorDst>
struct CopyL0CToDualAivUBTla {
    using ElementDst = typename TensorDst::Element;
    using ElementSrc = typename TensorSrc::Element;
    static constexpr auto quantPre =
        Catlass::Gemm::Tile::CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
                                                   Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;

    template <class DstTensor, class SrcTensor>
    CATLASS_DEVICE void operator()(DstTensor const& dstTensor, SrcTensor const& srcTensor, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.reluEn = false;
        intriParams.unitFlag = unitFlag;

        const uint32_t logicalRow = static_cast<uint32_t>(tla::get<0>(dstTensor.coord()));
        intriParams.subBlockId = logicalRow >= L1_TILE_M ? 1U : 0U;

        // Remove the logical AIV selector from the physical UB row offset.
        const auto logicalDstOffset = dstTensor.layout()(dstTensor.coord());
        const auto dstOffset = logicalDstOffset % MAX_SINGLE_MN;
        const auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(dstTensor.data()[dstOffset],
                                                                   srcTensor.data()[srcOffset], intriParams);
    }
};

// Backward TileCopy selects A/B layouts for transW and the GEMM phase.
// Data gradients use RowMajor A; weight gradients use ColumnMajor A.
// GmOutput selects L0C -> GM for recompute, saved-mode data gradients
// and weight gradients; otherwise Fixpipe targets the paired AIV UB
// for save=none data gradients.
template <class LayoutA, class LayoutB, uint32_t TargetUBIdx, bool GmOutput>
struct BackwardBF16TileCopy : Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB,
                                                                     bfloat16_t, RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t,
                                                        RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        std::conditional_t<TargetUBIdx == DUAL_AIV_TARGET, CopyL0CToDualAivUBTla<TensorL0C, TensorC>,
                           Catlass::Gemm::Tile::CopyL0CToTargetUBTla<
                               ArchTag, TensorL0C, TensorC, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                               Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, TargetUBIdx, false>>;
};

// L0C -> GM specialization for recompute, saved data gradients and dW.
template <class LayoutA, class LayoutB, uint32_t TargetUBIdx>
struct BackwardBF16TileCopy<LayoutA, LayoutB, TargetUBIdx, true>
    : Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t, RowMajor,
                                             void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t,
                                                        RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC,
                                                             Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
};

template <class LayoutA, class LayoutB, uint32_t TargetUBIdx = 0, bool GmOutput = false>
class BackwardGroupMatmulParam {
public:
    using TileCopy = BackwardBF16TileCopy<LayoutA, LayoutB, TargetUBIdx, GmOutput>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, bfloat16_t,
                                                         bfloat16_t, bfloat16_t, void, TileCopy>;

    __aicore__ inline BackwardGroupMatmulParam() : blockMmad(resource)
    {
        blockNum = AscendC::GetBlockNum();
        subBlockIdx = AscendC::GetSubBlockIdx();
        blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        l0cOutUbSecond = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MAX_SINGLE_MN * sizeof(bfloat16_t));
    }

    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
};

// Save=none P0-C: canonical Z=Xpacked@W1.T; legacy Z=Xpacked@W1.
// The two weight halves are independent [M,F] GEMMs whose Fixpipe output
// goes directly to the Gate/Up GM planes. No AIV UB handoff is needed,
// so recompute can overlap P0 gather/dispatch on the vector cores.
template <class WeightLayout>
__aicore__ inline void GroupMatmulRecompute(BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& gmm1Params,
                                            const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                            const GMMAddrInfo& gmmAddrInfo, GM_ADDR gatePlane, GM_ADDR upPlane,
                                            uint32_t& startBlockIdx, uint32_t expertIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));
    const uint32_t outputN = n / SWIGLU_N_HALF;

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> gateGlobal;
    AscendC::GlobalTensor<bfloat16_t> upGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    gateGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gatePlane);
    upGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)upPlane);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, outputN);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorGate = tla::MakeTensor(gateGlobal, layoutC, Catlass::Arch::PositionGM{});
    auto tensorUp = tla::MakeTensor(upGlobal, layoutC, Catlass::Arch::PositionGM{});

    // Match the save=none P1 scheduler and startBlockIdx evolution. The
    // AIC that writes a Gate/Up tile later produces its dHp and publishes
    // the P1 PIPE_FIX notification. The selected AIV's WaitForCube orders
    // both outputs before it loads Gate/Up from GM.
    BlockScheduler scheduler({m, outputN, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                                                 tla::MakeShape(actualShape.k(), actualShape.n()));
                auto& tensorC = weightBlock == 0 ? tensorGate : tensorUp;
                auto tensorBlockC =
                    tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
                gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
            }
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

// The saved-mode GM path uses unique output addresses to decouple the AIC
// producer from AIV UB reuse. Publish one arrival per AIC after the whole
// expert's Fixpipe writes retire; AIVs use alternating tile ownership.
// There are no per-tile cross-core events here, so an AIC may run ahead without
// exhausting event credits or waiting for a consumer acknowledgement.
// dHpBase aliases this expert's Hp plane: each tile has one AIV owner, which
// loads the complete dHp into UB before replacing it with Hp. All AIC writes
// retire before either lane starts consuming this expert. Other experts use
// disjoint bands; each expert's dW2 retires before P2 overwrites its dO.
template <class WeightLayout, typename TopkWeightT>
__aicore__ inline void GMM2GradSwiGLUGradGM(
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& gmm2Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape, TopkWeightT>& epilogueOp,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const GMMAddrInfo& gmmAddrInfo,
    GM_ADDR dHpBase, uint32_t& startBlockIdx, uint32_t& nextTargetAiv, uint32_t expertIdx, int64_t expertRowBase,
    __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget, const BackwardDispatchReadyView& rowReady,
    __gm__ int32_t* dHpReadyFlag)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = Get<N_VALUE>(problemShape);
    const uint32_t k = Get<K_VALUE>(problemShape);
    AscendC::GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmmAddrInfo.aGlobal));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmmAddrInfo.bGlobal));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dHpBase));
    auto tensorA = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(m, k), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, RowMajor>(m, n), Catlass::Arch::PositionGM{});
    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t loops = scheduler.GetCoreLoops();
    const uint32_t first =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;
    ProducerPhaseSchedule producerSchedule;
    if constexpr (g_coreType == AscendC::AIC) {
        producerSchedule.Init(rowReady, dHpReadyFlag, gmm2Params.blockNum, m, n);
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (first < loops && !rowReady.enabled) {
            while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
            }
        }
    } else {
        epilogueOp.UpdateGlobalAddr(expertRowBase);

        while (AscendC::ReadGmByPassDCache(dHpReadyFlag) < static_cast<int32_t>(gmm2Params.blockNum)) {
        }
    }
    for (uint32_t index = first; index < loops; index += gmm2Params.blockNum, ++nextTargetAiv) {
        auto coord = scheduler.GetBlockCoord(index);
        ProducerPhaseTile producerTile{};
        if constexpr (g_coreType == AscendC::AIC) {
            if (producerSchedule.Enabled()) {
                producerTile = producerSchedule.Get(index);
                coord = Catlass::GemmCoord(producerTile.m, producerTile.n, 0);
            }
        }
        const auto shape = scheduler.GetActualBlockShape(coord);
        const uint32_t row = coord.m() * L1_TILE_M;
        const uint32_t col = coord.n() * L1_TILE_N;
        const uint32_t depth = coord.k() * L1_TILE_K;
        if constexpr (g_coreType == AscendC::AIC) {
            if (rowReady.enabled) {
                if (producerSchedule.Enabled() && producerTile.known) {
                    WaitProducerPhaseRows(rowReady, producerTile, row, shape.m());
                } else {
                    rowReady.WaitRows(row, shape.m());
                }
            }
            auto tileA = tla::GetTile(tensorA, tla::MakeCoord(row, depth), tla::MakeShape(shape.m(), shape.k()));
            auto tileB = tla::GetTile(tensorB, tla::MakeCoord(depth, col), tla::MakeShape(shape.k(), shape.n()));
            auto tileC = tla::GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(shape.m(), shape.n()));
            gmm2Params.blockMmad(tileA, tileB, tileC, shape);
        } else if (gmm2Params.subBlockIdx == (nextTargetAiv & 1U)) {
            epilogueOp.LoadDHpFromGM(dHpBase + (static_cast<int64_t>(row) * n + col) * sizeof(bfloat16_t), shape.m(),
                                     shape.n(), n);
            AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epiShape{shape.m(), shape.n(), 0, 0};
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{
                static_cast<int64_t>(row) * n + col, 0, 0, 0, 0, 0};
            epilogueOp.SwiGLUGrad(epiShape, offset);
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        // Retire Fixpipe writes before publishing readiness, including AICs with zero tiles.
        AscendC::PipeBarrier<PIPE_ALL>();
        AtomicAdd(dHpReadyFlag, int32_t(1));
    }
    startBlockIdx = (startBlockIdx + loops) % gmm2Params.blockNum;
}

// Save=none P1: AIC alternates completed GMM2 dHp tiles between
// its two paired AIVs. Each AIV owns an independent UB slot, so Cube can
// produce tile i+1 while the other AIV consumes tile i. A slot is reused only
// after its AIV has published completion through the matching cross-core
// event.
template <class WeightLayout, typename TopkWeightT>
__aicore__ inline void GMM2GradSwiGLUGradDualAiv(
    BackwardGroupMatmulParam<RowMajor, WeightLayout, DUAL_AIV_TARGET>& gmm2Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape, TopkWeightT>& epilogueOp, const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, int32_t& outstandingMask, uint32_t& nextTargetAiv, uint32_t expertIdx,
    int64_t expertRowBase, __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget, uint32_t p1Event)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // F
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // K
    (void)params;
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(2 * L1_TILE_M, L1_TILE_N);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbDual = tla::MakeTensor(gmm2Params.l0cOutUbFirst, layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;

    if constexpr (g_coreType == AscendC::AIV) {
        epilogueOp.UpdateGlobalAddr(expertRowBase);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm2Params.blockNum, ++nextTargetAiv) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        // Keep parity across experts so single-tile experts alternate between both AIVs.
        const uint32_t targetAiv = nextTargetAiv & 1U;
        const int32_t targetBit = 1 << targetAiv;
        const uint16_t targetEvent = static_cast<uint16_t>(p1Event + (targetAiv == 0 ? 0 : FLAG_ID_MAX));

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {
                    }
                }
            }
            if ((outstandingMask & targetBit) != 0) {
                WaitForVector(targetEvent);
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB =
                tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb = tla::GetTile(tensorUbDual, tla::MakeCoord(targetAiv * L1_TILE_M, 0),
                                              tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(targetEvent);
            outstandingMask |= targetBit;
        }

        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm2Params.subBlockIdx == targetAiv) {
                AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0,
                                                                                 0};
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * n + nLoc, 0, 0, 0, 0, 0};

                WaitForCube(p1Event);

                epilogueOp.SwiGLUGrad(epilogueShape, epilogueOffset);
                NotifyCube(p1Event);
            }
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

// The caller must retire dZ/Hp producers and dW2 readers before dX overwrites dO.
template <class WeightLayout>
__aicore__ inline void GMM1GradCombineGM(BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& mm,
                                         const Params& params,
                                         const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                         const GMMAddrInfo& addr, GM_ADDR dxBase, uint32_t& startBlockIdx,
                                         uint32_t expertIdx, __gm__ int32_t* dxReady)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = Get<N_VALUE>(problemShape);
    const uint32_t k = Get<K_VALUE>(problemShape);
    if constexpr (g_coreType == AscendC::AIC) {
        // The caller establishes this expert's input readiness before P2.

    } else if (mm.subBlockIdx == 0) {
        while (AscendC::ReadGmByPassDCache(dxReady) < static_cast<int32_t>(mm.blockNum)) {
        }
    }
    AscendC::GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.aGlobal));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.bGlobal));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dxBase));
    auto tensorA = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(m, k), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, RowMajor>(m, n), Catlass::Arch::PositionGM{});
    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t loops = scheduler.GetCoreLoops();
    const uint32_t first = (mm.blockIdx < startBlockIdx ? mm.blockIdx + mm.blockNum : mm.blockIdx) - startBlockIdx;
    for (uint32_t index = first; index < loops; index += mm.blockNum) {
        const auto coord = scheduler.GetBlockCoord(index);
        auto shape = scheduler.GetActualBlockShape(coord);
        const uint32_t row = coord.m() * L1_TILE_M;
        const uint32_t col = coord.n() * L1_TILE_N;
        const uint32_t depth = coord.k() * L1_TILE_K;
        if constexpr (g_coreType == AscendC::AIC) {
            auto tileA = tla::GetTile(tensorA, tla::MakeCoord(row, depth), tla::MakeShape(shape.m(), shape.k()));
            auto tileB = tla::GetTile(tensorB, tla::MakeCoord(depth, col), tla::MakeShape(shape.k(), shape.n()));
            auto tileC = tla::GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(shape.m(), shape.n()));
            mm.blockMmad(tileA, tileB, tileC, shape);
        } else if (mm.subBlockIdx == 0) {
            // Retire MTE3's UB read before MTE2 refills the same tile.
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(shape.m()),
                                            static_cast<uint32_t>(shape.n() * sizeof(bfloat16_t)),
                                            static_cast<uint32_t>((n - shape.n()) * sizeof(bfloat16_t)), 0, 0};
            AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0U};
            AscendC::DataCopyPad(mm.l0cOutUbFirst, c[static_cast<int64_t>(row) * n + col], copy, pad);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                row, col, n, expertIdx, 0, params.tilingData->rankId, mm.l0cOutUbFirst, shape, params);
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        AscendC::PipeBarrier<PIPE_ALL>();  // Fixpipe GM writes before ready
        AtomicAdd(dxReady, int32_t(1));    // also for zero-tile cores

    } else if (mm.subBlockIdx == 0) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    startBlockIdx = (startBlockIdx + loops) % mm.blockNum;
}

// Save=none P2: canonical dX=dZ@W1; legacy dX=dZ@W1.T. AIC writes
// AIV0 UB, and AIV0 returns each dX tile to its source (CombineTokens).
// Ping-pong ownership uses the forward GMM2's cross-core protocol.
template <uint32_t TargetUBIdx, class WeightLayout>
__aicore__ inline void GMM1GradCombine(BackwardGroupMatmulParam<RowMajor, WeightLayout, TargetUBIdx>& gmm1Params,
                                       const Params& params,
                                       const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                       const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                       uint32_t expertIdx, int64_t expertRowBase, uint16_t& pingpongIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // K
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // N

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(L1_TILE_M, L1_TILE_N);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm1Params.l0cOutUbFirst, layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm1Params.l0cOutUbSecond, layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    constexpr uint16_t targetEventOffset = TargetUBIdx == 0 ? 0 : FLAG_ID_MAX;
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto outputUb = pingpongIdx == 0 ? gmm1Params.l0cOutUbFirst : gmm1Params.l0cOutUbSecond;

        if constexpr (g_coreType == AscendC::AIC) {
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx + targetEventOffset);
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB =
                tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb =
                tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(pingpongIdx + targetEventOffset);
        }

        ++vecSetSyncCom;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == TargetUBIdx) {
                WaitForCube(pingpongIdx);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                    mLoc, nLoc, n, expertIdx, expertRowBase, params.tilingData->rankId, outputUb, actualShape, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                NotifyCube(pingpongIdx);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

// View row-major expert bands as ColumnMajor transposes without moving GM data.
__aicore__ inline void GMM1DW(BackwardGroupMatmulParam<ColumnMajor, RowMajor, 0, true>& gmm1Params,
                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                              const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);                         // canonical N / legacy K
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // canonical K / legacy N
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // Me

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> cGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    cGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.cGlobal);

    // A is canonical dZ.T or legacy Xpacked.T, viewed as ColumnMajor.
    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(cGlobal, layoutC, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorBlockA =
            tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockB =
            tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockC =
            tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

// GMM2 dW: canonical dO.T@Hp -> [K,F]; legacy Hp.T@dO -> [F,K].
// Saved modes run this before per-expert P2 dO reuse; none runs it in P4.
__aicore__ inline void GMM2DW(BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true>& gmm2Params,
                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                              const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);                         // canonical K / legacy F
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // canonical F / legacy K
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // Me

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> cGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    cGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.cGlobal);

    // A is canonical dO.T or legacy Hp.T, viewed as ColumnMajor.
    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(cGlobal, layoutC, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm2Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorBlockA =
            tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockB =
            tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockC =
            tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

}  // namespace MegaMoeBF16BackwardCatlassImpl

#endif  // DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_HPP
