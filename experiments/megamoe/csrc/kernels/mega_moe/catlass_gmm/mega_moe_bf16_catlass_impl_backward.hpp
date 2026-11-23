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
#include <new>

#include "kernel_operator.h"
#include "stage/mega_moe_slice_ready.hpp"
#include "mc2_common/mc2_kernel_utils.hpp"
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
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "epilogue/block_epilogue_swiglu_bf16_backward.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "common/mega_moe_bf16_context.hpp"
#include "stage/mega_moe_dispatch_ready_backward.hpp"
#include "stage/mega_moe_dispatch_schedule_backward.hpp"
#include "stage/mega_moe_staged_wave_backward.hpp"
#include "stage/mega_moe_combine_send.hpp"

namespace MegaMoeBF16BackwardCatlassImpl {

using MegaMoeImpl::Get;
using MegaMoeImpl::NotifyCube;
using MegaMoeImpl::NotifyVector;
using MegaMoeImpl::WaitForCube;
using MegaMoeImpl::WaitForVector;

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
    struct CopyL0CToDst {
        template <class TensorDst, class TensorSrc>
        CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, uint8_t unitFlag = 0)
        {
            using Copy = Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, TensorC,
                                                             Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                                                             Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
            if constexpr (TargetUBIdx == DUAL_AIV_TARGET) {
                // The second logical M band selects AIV1. Each AIV has its own UB tile at the same offset.
                const bool aiv1 = static_cast<uint32_t>(tla::get<0>(dst.coord())) >= L1_TILE_M;
                const auto ubOffset = dst.layout()(dst.coord()) % MAX_SINGLE_MN;
                const auto ubTile = tla::MakeTensor(dst.data()[ubOffset], dst.layout(), Catlass::Arch::PositionUB{});
                Copy{}(ubTile, src, aiv1, unitFlag);
            } else {
                Copy{}(dst, src, TargetUBIdx == 1U, unitFlag);
            }
        }
    };
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
    using BaseBlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, bfloat16_t,
                                                             bfloat16_t, bfloat16_t, void, TileCopy>;

    struct BlockMmad : BaseBlockMmad {
        __aicore__ inline explicit BlockMmad(Catlass::Arch::Resource<ArchTag>& r) : BaseBlockMmad(r) {}
        __aicore__ inline static void* operator new(size_t, void* address)
        {
            return address;
        }
    };
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
                                            const Bf16GMMAddrInfo& gmmAddrInfo, GM_ADDR gatePlane, GM_ADDR upPlane,
                                            uint32_t& startBlockIdx, uint32_t expertIdx)
{
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

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
    // AIC that writes a Gate/Up tile later produces its dHp. The UB path
    // orders both outputs through the P1 PIPE_FIX notification. The GM path
    // publishes recompute readiness only after both Gate/Up writes retire.
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
        MegaMoeTrainTimerTileScope timerTile(g_coreType == AscendC::AIC ? gmmAddrInfo.timer : nullptr,
                                             MegaMoeTrainTimer::AIC_BW_RECOMPUTE, expertIdx, mLoc, nLoc,
                                             actualShape.m(), actualShape.n(), actualShape.k(), L1_TILE_M, L1_TILE_N);

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

// The two counters of each canonical M tile retire dO DMA and dHp Fixpipe writes, respectively.
template <class WeightLayout, typename TopkWeightT, class Borrow>
__aicore__ inline void GMM2GradStagedWave(
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape, TopkWeightT>& epilogue,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problem, const Bf16GMMAddrInfo& addr, GM_ADDR dHp,
    uint32_t expert, uint64_t base, const MegaMoeBF16Training::BackwardWavePlan& plan,
    const MegaMoeImpl::RowSliceReady& ready, uint32_t startBlock, Borrow borrow)
{
    static_assert(L1_TILE_M == MegaMoeBF16Training::BackwardWavePlan::tileM);
    if constexpr (g_coreType == AscendC::AIV) {
        if (params.subBlockIdx == 0) return;
    }
    const uint32_t m = Get<M_VALUE>(problem), n = Get<N_VALUE>(problem), k = Get<K_VALUE>(problem);
    const uint32_t columns = (n + L1_TILE_N - 1U) / L1_TILE_N;
    AscendC::GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.aGlobal));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.bGlobal));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dHp));
    auto tensorA = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(m, k), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, RowMajor>(m, n), Catlass::Arch::PositionGM{});
    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    epilogue.SetTimer(addr.timer);
    if constexpr (g_coreType == AscendC::AIC) {
        plan.ForEachDhpRow(
            params.blockIdx, columns, startBlock,
            [&](uint32_t row, uint32_t rows, uint32_t firstCol, uint32_t arrivals) __aicore__ {
                auto* slot = ready.Slot(row);
                if (addr.timer) addr.timer->SetWave(plan.WaveId(row));
                if (AscendC::ReadGmByPassDCache(slot) < static_cast<int32_t>(arrivals)) {
                    while (borrow(slot, arrivals)) {
                        if (AscendC::ReadGmByPassDCache(slot) >= static_cast<int32_t>(arrivals)) break;
                    }
                }
                // Borrowing changes the Timer's expert to the dW2 donor. Tag only
                // the remaining dO wait with the current dHp expert and row.
                if (AscendC::ReadGmByPassDCache(slot) < static_cast<int32_t>(arrivals)) {
                    if (addr.timer) addr.timer->SetSlice(expert, row, rows);
                    MegaMoeTrainTimerWaitScope wait(addr.timer, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                    do {
                    } while (AscendC::ReadGmByPassDCache(slot) < static_cast<int32_t>(arrivals));
                }
                for (uint32_t colTile = firstCol; colTile < columns; colTile += plan.cores) {
                    const uint32_t col = colTile * L1_TILE_N;
                    const auto shape = scheduler.GetActualBlockShape(Catlass::GemmCoord(row / L1_TILE_M, colTile, 0));
                    MegaMoeTrainTimerTileScope tile(addr.timer, MegaMoeTrainTimer::AIC_BW_DHP, expert, row, col,
                                                    shape.m(), shape.n(), shape.k(), L1_TILE_M, L1_TILE_N, 3U);
                    auto tileA = tla::GetTile(tensorA, tla::MakeCoord(row, 0U), tla::MakeShape(shape.m(), shape.k()));
                    auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, col), tla::MakeShape(shape.k(), shape.n()));
                    auto tileC = tla::GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(shape.m(), shape.n()));
                    params.blockMmad(tileA, tileB, tileC, shape);
                }
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                AscendC::AtomicAdd(slot + 1, int32_t(1));
            });
    } else {
        const uint32_t writers = MegaMoeImpl::GetGmm2ProducerCount(columns, plan.cores);
        auto waitForDhp = [&](uint32_t row, uint32_t rows) __aicore__ {
            auto* slot = ready.Slot(row);
            if (addr.timer) {
                addr.timer->SetWave(plan.WaveId(row));
                addr.timer->SetSlice(expert, row, rows);
            }
            if (AscendC::ReadGmByPassDCache(slot + 1) < static_cast<int32_t>(writers)) {
                MegaMoeTrainTimerWaitScope wait(addr.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD_WAIT);
                while (AscendC::ReadGmByPassDCache(slot + 1) < static_cast<int32_t>(writers)) {
                }
            }
        };
        if (epilogue.SavedMode() == MegaMoeBF16Training::kSaveMxFp8) {
            // Keep MXFP8's original N-tile decoding and dP partials. Each tile
            // has one AIV1 owner and reads dHp before overwriting that tile with Hp.
            epilogue.UpdateGlobalAddr(base);
            plan.ForEachDhpRow(
                params.blockIdx, columns, startBlock,
                [&](uint32_t row, uint32_t rows, uint32_t firstCol, uint32_t) __aicore__ {
                    waitForDhp(row, rows);
                    for (uint32_t colTile = firstCol; colTile < columns; colTile += plan.cores) {
                        const uint32_t col = colTile * L1_TILE_N;
                        const uint32_t cols = n - col < L1_TILE_N ? n - col : L1_TILE_N;
#ifdef DEEPEP_MEGAMOE_TIMER
                        MegaMoeTrainTimerTileScope tile(addr.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD, expert, row,
                                                        col, rows, cols, k, L1_TILE_M, L1_TILE_N, 3U);
#endif
                        const int64_t offset = static_cast<int64_t>(row) * n + col;
                        epilogue.LoadDHpFromGM(dHp + offset * sizeof(bfloat16_t), rows, cols, n);
                        AscendC::Shape<int64_t, int64_t, int64_t, int64_t> shape{rows, cols, 0, 0};
                        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> pos{offset, 0, 0, 0, 0, 0};
                        epilogue.SwiGLUGrad(shape, pos);
                    }
                });
            return;
        }
        plan.ForEach(params.blockIdx, [&](uint32_t row, uint32_t rows, uint32_t member, uint32_t owners) __aicore__ {
            waitForDhp(row, rows);
            MegaMoeTrainTimerSpanScope span(addr.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD);
            epilogue.UpdateGlobalAddr(base + row);
            epilogue.SwiGLUGradRows(dHp + uint64_t(row) * n * sizeof(bfloat16_t), rows, member, owners);
        });
    }
}

// The saved-mode GM path uses unique output addresses to decouple the AIC
// producer from AIV UB reuse. Publish one arrival per AIC after the whole
// expert's Fixpipe writes retire. Whole-row dP uses AIV1 row ownership;
// the other paths use alternating tile ownership.
// There are no per-tile cross-core events here, so an AIC may run ahead without
// exhausting event credits or waiting for a consumer acknowledgement.
// dHpBase aliases this expert's Hp plane: each row or tile has one AIV owner,
// which reads its complete dHp before replacing it with Hp. All AIC writes
// retire before the consumer starts this expert. Other experts use disjoint bands;
// each expert's dW2 retires before P2 overwrites its dO.
template <class WeightLayout, typename TopkWeightT>
__aicore__ inline void GMM2GradSwiGLUGradGM(
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& gmm2Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape, TopkWeightT>& epilogueOp,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const Bf16GMMAddrInfo& gmmAddrInfo,
    GM_ADDR dHpBase, uint32_t& startBlockIdx, uint32_t& nextTargetAiv, uint32_t expertIdx, int64_t expertRowBase,
    __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget, const BackwardDispatchReadyView& rowReady,
    __gm__ int32_t* dHpReadyFlag)
{
    epilogueOp.SetTimer(gmmAddrInfo.timer);
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

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
            {
                MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
                }
            }
        }
    } else {
        epilogueOp.UpdateGlobalAddr(expertRowBase);

        {
            MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD_WAIT);
            while (AscendC::ReadGmByPassDCache(dHpReadyFlag) < static_cast<int32_t>(gmm2Params.blockNum)) {
            }
        }
        if (MegaMoeBF16Training::UseWholeRowTopkGrad(epilogueOp.SavedMode(), n)) {
            // AIV1 owns whole rows while AIV0 dispatches later experts. Row-local
            // ownership orders each raw dHp read before its in-place Hp write.
            if (gmm2Params.subBlockIdx == 1U) {
                MegaMoeTrainTimerSpanScope timerRows(gmmAddrInfo.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD);
                epilogueOp.SwiGLUGradRows(dHpBase, m, gmm2Params.blockIdx, gmm2Params.blockNum);
            }
            nextTargetAiv += first < loops ? (loops - first + gmm2Params.blockNum - 1U) / gmm2Params.blockNum : 0U;
            startBlockIdx = (startBlockIdx + loops) % gmm2Params.blockNum;
            return;
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
        MegaMoeTrainTimerTileScope timerTile(
            (g_coreType == AscendC::AIC || gmm2Params.subBlockIdx == (nextTargetAiv & 1U)) ? gmmAddrInfo.timer
                                                                                           : nullptr,
            g_coreType == AscendC::AIC ? MegaMoeTrainTimer::AIC_BW_DHP : MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD,
            expertIdx, row, col, shape.m(), shape.n(), shape.k(), L1_TILE_M, L1_TILE_N, rowReady.enabled ? 3U : 2U);

        if constexpr (g_coreType == AscendC::AIC) {
            if (rowReady.enabled) {
                if (producerSchedule.Enabled() && producerTile.known) {
                    {
                        MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                        WaitProducerPhaseRows(rowReady, producerTile, row, shape.m());
                    }
                } else {
                    {
                        MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                        rowReady.WaitRows(row, shape.m());
                    }
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
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape, TopkWeightT>& epilogueOp, const Bf16Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const Bf16GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, int32_t& outstandingMask, uint32_t& nextTargetAiv, uint32_t expertIdx,
    int64_t expertRowBase, __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget, uint32_t p1Event)
{
    epilogueOp.SetTimer(gmmAddrInfo.timer);
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

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
        MegaMoeTrainTimerTileScope timerTile(
            (g_coreType == AscendC::AIC || gmm2Params.subBlockIdx == (nextTargetAiv & 1U)) ? gmmAddrInfo.timer
                                                                                           : nullptr,
            g_coreType == AscendC::AIC ? MegaMoeTrainTimer::AIC_BW_DHP : MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD,
            expertIdx, mLoc, nLoc, actualShape.m(), actualShape.n(), actualShape.k(), L1_TILE_M, L1_TILE_N);

        // Keep parity across experts so single-tile experts alternate between both AIVs.
        const uint32_t targetAiv = nextTargetAiv & 1U;
        const int32_t targetBit = 1 << targetAiv;
        const uint16_t targetEvent = static_cast<uint16_t>(p1Event + (targetAiv == 0 ? 0 : FLAG_ID_MAX));

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DHP_WAIT);
                    while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {
                        }
                    }
                }
            }
            if ((outstandingMask & targetBit) != 0) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DHP_UB_WAIT);
                    WaitForVector(targetEvent);
                }
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

                {
                    MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD_WAIT);
                    WaitForCube(p1Event);
                }

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
                                         const Bf16Params& params,
                                         const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                         const Bf16GMMAddrInfo& addr, GM_ADDR dxBase, uint32_t& startBlockIdx,
                                         uint32_t expertIdx, __gm__ int32_t* dxReady)
{
    if (addr.timer != nullptr) {
        addr.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        addr.timer->SetTaskContext(-1);
    }

    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = Get<N_VALUE>(problemShape);
    const uint32_t k = Get<K_VALUE>(problemShape);
    if constexpr (g_coreType == AscendC::AIC) {
        // The caller establishes this expert's input readiness before P2.

    } else if (mm.subBlockIdx == 0) {
        {
            MegaMoeTrainTimerWaitScope timerWait(addr.timer, MegaMoeTrainTimer::AIV_BW_COMBINE_WAIT);
            while (AscendC::ReadGmByPassDCache(dxReady) < static_cast<int32_t>(mm.blockNum)) {
            }
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
        MegaMoeTrainTimerTileScope timerTile(
            (g_coreType == AscendC::AIC || mm.subBlockIdx == 0) ? addr.timer : nullptr,
            g_coreType == AscendC::AIC ? MegaMoeTrainTimer::AIC_BW_DX : MegaMoeTrainTimer::AIV_BW_COMBINE, expertIdx,
            row, col, shape.m(), shape.n(), shape.k(), L1_TILE_M, L1_TILE_N, 2U);

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
                                       const Bf16Params& params,
                                       const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                       const Bf16GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                       int32_t& vecSetSyncCom, uint32_t expertIdx, int64_t expertRowBase,
                                       uint16_t& pingpongIdx)
{
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

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
        MegaMoeTrainTimerTileScope timerTile(
            (g_coreType == AscendC::AIC || gmm1Params.subBlockIdx == 0) ? gmmAddrInfo.timer : nullptr,
            g_coreType == AscendC::AIC ? MegaMoeTrainTimer::AIC_BW_DX : MegaMoeTrainTimer::AIV_BW_COMBINE, expertIdx,
            mLoc, nLoc, actualShape.m(), actualShape.n(), actualShape.k(), L1_TILE_M, L1_TILE_N);

        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto outputUb = pingpongIdx == 0 ? gmm1Params.l0cOutUbFirst : gmm1Params.l0cOutUbSecond;

        if constexpr (g_coreType == AscendC::AIC) {
            if (vecSetSyncCom >= 2) {
                {
                    MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIC_BW_DX_UB_WAIT);
                    WaitForVector(pingpongIdx + targetEventOffset);
                }
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
                {
                    MegaMoeTrainTimerWaitScope timerWait(gmmAddrInfo.timer, MegaMoeTrainTimer::AIV_BW_COMBINE_WAIT);
                    WaitForCube(pingpongIdx);
                }
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
                              const Bf16GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx)
{
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

    const uint32_t m = Get<M_VALUE>(problemShape);                         // canonical N / legacy K
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // canonical K / legacy N
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // Me

    AscendC::GlobalTensor<bfloat16_t> aGlobal, bGlobal;
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
        MegaMoeTrainTimerTileScope timerTile(g_coreType == AscendC::AIC ? gmmAddrInfo.timer : nullptr,
                                             MegaMoeTrainTimer::AIC_BW_DW1, expertIdx, mLoc, nLoc, actualShape.m(),
                                             actualShape.n(), actualShape.k(), L1_TILE_M, L1_TILE_N);

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
                              const Bf16GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx,
                              uint32_t firstLocal = 0, uint32_t count = ~0U, __gm__ int32_t* priority = nullptr,
                              int32_t arrivals = 0, uint32_t* completed = nullptr)
{
    if (gmmAddrInfo.timer != nullptr) {
        gmmAddrInfo.timer->SetSlice(expertIdx, 0, Get<M_VALUE>(problemShape));
        gmmAddrInfo.timer->SetTaskContext(-1);
    }

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

    uint32_t done = 0;
    for (uint32_t loopIdx = startLoopIdx + firstLocal * gmm2Params.blockNum; loopIdx < coreLoops && done < count;
         loopIdx += gmm2Params.blockNum) {
        if (priority && done && AscendC::ReadGmByPassDCache(priority) >= arrivals) break;
        ++done;
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        MegaMoeTrainTimerTileScope timerTile(g_coreType == AscendC::AIC ? gmmAddrInfo.timer : nullptr,
                                             MegaMoeTrainTimer::AIC_BW_DW2, expertIdx, mLoc, nLoc, actualShape.m(),
                                             actualShape.n(), actualShape.k(), L1_TILE_M, L1_TILE_N);

        auto tensorBlockA =
            tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockB =
            tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockC =
            tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
    }
    if (completed) *completed = done;
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

}  // namespace MegaMoeBF16BackwardCatlassImpl

#endif  // DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_HPP
