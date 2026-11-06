// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*
 * DeepEP-native BF16 grouped GEMM stages for MegaMoe. BF16 x BF16 MMAD
 * accumulates in L0C/FP32. GMM1 delivers BF16 Gate/Up either through paired
 * UB tiles with a per-tile handshake or through raw GM planes. On the GM
 * path, AIV waits for all AIC writes for the expert, then loads its tiles
 * before SwiGLU. GMM2 delivers BF16 output tiles through UB for combine.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_FORWARD_CATLASS_IMPL_HPP
#define DEEPEP_MEGA_MOE_BF16_FORWARD_CATLASS_IMPL_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
// CATLASS 2b4 and CANN 9.2 still parse unused HSTU templates whose
// unqualified tuple helpers collide with AscendC::Std. Exclude only those
// unused specializations in this BF16 translation unit.
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
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "epilogue/block_epilogue_swiglu_forward_bf16.hpp"
#include "catlass/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "layout/mega_moe_context.hpp"
#include "routing/mega_moe_combine_send.hpp"
#include "schedule/forward_prepared_phase.hpp"

namespace MegaMoeBF16ForwardCatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using Gmm1L1TileShape = tla::Shape<tla::Int<MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows>,
                                   tla::Int<256>, tla::Int<256>>;
using Gmm1L0TileShape = tla::Shape<tla::Int<MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows>,
                                   tla::Int<256>, tla::Int<64>>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using Gmm2BlockScheduler = BlockScheduler;

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t SWIGLU_N_HALF = MegaMoeBF16Training::kSwigluNHalf;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t GMM1_TILE_M = MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows;
constexpr uint32_t GMM1_TILE_ELEMENTS = GMM1_TILE_M * L1_TILE_N;
__aicore__ inline void ConsiderGmm1NStripeWidth(uint32_t width, uint32_t lanes, uint32_t mTiles, uint32_t nTiles,
                                                uint32_t minWaveColumns, uint32_t& bestWidth,
                                                uint32_t& bestAddressUnits, uint32_t& bestBRun)
{
    if (width > nTiles) {
        return;
    }
    const uint32_t activeM = mTiles < lanes ? mTiles : lanes;
    const uint32_t candidateActiveB = width < minWaveColumns ? minWaveColumns : width;
    const uint32_t activeB = candidateActiveB < nTiles ? candidateActiveB : nTiles;
    const uint32_t addressUnits = activeM * GMM1_TILE_M + activeB * SWIGLU_N_HALF * L1_TILE_N;
    const uint32_t bRun = Ops::Base::CeilDiv(mTiles, lanes);
    if (addressUnits < bestAddressUnits || (addressUnits == bestAddressUnits && bRun > bestBRun)) {
        bestWidth = width;
        bestAddressUnits = addressUnits;
        bestBRun = bRun;
    }
}

__aicore__ inline uint32_t SelectGmm1NStripeWidth(uint32_t m, uint32_t outputN, uint32_t blockNum)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    if (mTiles == 0 || nTiles == 0 || blockNum == 0) {
        return 0;
    }

    // The deployed platforms use 28, 32 or 36 AICs. Keep selection bounded:
    // each core evaluates only the widths that divide its AIC count.
    if (blockNum != 28 && blockNum != 32 && blockNum != 36) {
        return 0;
    }

    const uint32_t minWaveColumns = Ops::Base::CeilDiv(blockNum, mTiles);
    uint32_t bestWidth = 1;
    const uint32_t bestActiveM = mTiles < blockNum ? mTiles : blockNum;
    const uint32_t bestActiveB = minWaveColumns < nTiles ? minWaveColumns : nTiles;
    uint32_t bestAddressUnits = bestActiveM * GMM1_TILE_M + bestActiveB * SWIGLU_N_HALF * L1_TILE_N;
    uint32_t bestBRun = Ops::Base::CeilDiv(mTiles, blockNum);
    if (blockNum == 28) {
        ConsiderGmm1NStripeWidth(2, 14, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(4, 7, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    } else if (blockNum == 32) {
        ConsiderGmm1NStripeWidth(2, 16, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(4, 8, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    } else {
        ConsiderGmm1NStripeWidth(2, 18, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(3, 12, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(4, 9, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(6, 6, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    }
    return bestWidth;
}

__aicore__ inline Catlass::GemmCoord GetGmm1AdaptiveNStripeCoord(uint32_t taskIdx, uint32_t m, uint32_t outputN,
                                                                 uint32_t stripeWidth)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    const uint32_t fullStripeTasks = mTiles * stripeWidth;
    const uint32_t stripe = taskIdx / fullStripeTasks;
    const uint32_t taskInStripe = taskIdx % fullStripeTasks;
    const uint32_t stripeStart = stripe * stripeWidth;
    const uint32_t remainingN = nTiles - stripeStart;
    const uint32_t actualWidth = remainingN < stripeWidth ? remainingN : stripeWidth;
    return Catlass::GemmCoord{taskInStripe / actualWidth, stripeStart + taskInStripe % actualWidth, 0};
}

__aicore__ inline uint32_t GetGmm1ReadyFirstMTile(uint32_t mOrdinal, uint32_t mTiles,
                                                  uint32_t readyBlocksPerProducerPartition)
{
    constexpr uint32_t tilesPerReadyBlock =
        MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kReadyRows / GMM1_TILE_M;
    static_assert(tilesPerReadyBlock == 2, "ready-first mapping requires two GMM1 M tiles per Dispatch-ready block");
    if (readyBlocksPerProducerPartition <= 1 || mTiles % tilesPerReadyBlock != 0) {
        return mOrdinal;
    }
    const uint32_t readyBlocks = mTiles / tilesPerReadyBlock;
    if (readyBlocks % readyBlocksPerProducerPartition != 0) {
        return mOrdinal;
    }
    const uint32_t producerPartitions = readyBlocks / readyBlocksPerProducerPartition;
    const uint32_t blockOrdinal = mOrdinal / tilesPerReadyBlock;
    const uint32_t tileInBlock = mOrdinal % tilesPerReadyBlock;
    const uint32_t producerRound = blockOrdinal / producerPartitions;
    const uint32_t producerPartition = blockOrdinal % producerPartitions;
    return (producerPartition * readyBlocksPerProducerPartition + producerRound) * tilesPerReadyBlock + tileInBlock;
}

__aicore__ inline Catlass::GemmCoord GetGmm1AdaptiveReadyFirstCoord(uint32_t taskIdx, uint32_t m, uint32_t outputN,
                                                                    uint32_t stripeWidth,
                                                                    uint32_t readyBlocksPerProducerPartition)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    const uint32_t fullStripeTasks = mTiles * stripeWidth;
    const uint32_t stripe = taskIdx / fullStripeTasks;
    const uint32_t taskInStripe = taskIdx % fullStripeTasks;
    const uint32_t stripeStart = stripe * stripeWidth;
    const uint32_t remainingN = nTiles - stripeStart;
    const uint32_t actualWidth = remainingN < stripeWidth ? remainingN : stripeWidth;
    const uint32_t mOrdinal = taskInStripe / actualWidth;
    return Catlass::GemmCoord{GetGmm1ReadyFirstMTile(mOrdinal, mTiles, readyBlocksPerProducerPartition),
                              stripeStart + taskInStripe % actualWidth, 0};
}
static_assert(2U * GMM1_TILE_ELEMENTS * sizeof(bfloat16_t) <= ArchTag::UB_SIZE,
              "The two BF16 GMM1 output tiles must fit in A5 UB");

template <uint32_t TargetUBIdx, class WeightLayout>
struct PackedBF16TileCopyTlaToTargetUB
    : public Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                    WeightLayout, bfloat16_t, Catlass::layout::RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                        WeightLayout, bfloat16_t, Catlass::layout::RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        Catlass::Gemm::Tile::CopyL0CToTargetUBTla<ArchTag, TensorL0C, TensorC,
                                                  Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                                                  Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, TargetUBIdx, false>;
};

// Ordinary forward can write raw Gate/Up to GM without an AIV handshake.
template <class WeightLayout>
struct ForwardGmTileCopy
    : public Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                    WeightLayout, bfloat16_t, Catlass::layout::RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                        WeightLayout, bfloat16_t, Catlass::layout::RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;
    template <class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC,
                                                             Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
};

template <uint32_t GmmIndex, bool TransW = true, bool GmOutput = false>
class ForwardGroupMatmulParam {
public:
    static_assert(GmmIndex < 2, "GMM1/GMM2 must target AIV0/AIV1");
    using WeightLayout = std::conditional_t<TransW, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using TileCopy = std::conditional_t<GmOutput, ForwardGmTileCopy<WeightLayout>,
                                        PackedBF16TileCopyTlaToTargetUB<GmmIndex, WeightLayout>>;
    using MmadL1TileShape = std::conditional_t<GmmIndex == 0, Gmm1L1TileShape, L1TileShape>;
    using MmadL0TileShape = std::conditional_t<GmmIndex == 0, Gmm1L0TileShape, L0TileShape>;
    static constexpr uint32_t kTileRows = GmmIndex == 0 ? GMM1_TILE_M : L1_TILE_M;
    using CatlassBlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, MmadL1TileShape, MmadL0TileShape,
                                                                bfloat16_t, bfloat16_t, bfloat16_t, void, TileCopy>;
    using BlockMmad = CatlassBlockMmad;
    using LayoutB = decltype(tla::MakeLayout<bfloat16_t, WeightLayout>(uint32_t{}, uint32_t{}));
    using LayoutC = decltype(tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(uint32_t{}, uint32_t{}));

    __aicore__ inline ForwardGroupMatmulParam(uint32_t nValue, uint32_t kValue, uint32_t dispatchBlockNumPerEPValue = 0)
        : blockMmad(resource),
          blockNum(AscendC::GetBlockNum()),
          subBlockIdx(AscendC::GetSubBlockIdx()),
          blockIdx(AscendC::GetBlockIdx() / AscendC::GetTaskRation()),
          dispatchBlockNumPerEP(dispatchBlockNumPerEPValue),
          n(nValue),
          k(kValue),
          outputN(nValue / SWIGLU_N_HALF),
          layoutB(tla::MakeLayout<bfloat16_t, WeightLayout>(kValue, nValue)),
          layoutC(tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(kTileRows, L1_TILE_N))
    {
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        l0cOutUbSecond =
            resource.ubBuf.template GetBufferByByte<bfloat16_t>(kTileRows * L1_TILE_N * sizeof(bfloat16_t));
    }

    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
    uint32_t dispatchBlockNumPerEP;
    uint32_t n;
    uint32_t k;
    uint32_t outputN;
    LayoutB layoutB;
    LayoutC layoutC;
};

template <bool DispatchTileReady = false, bool WaitForFullExpert = false,
          uint32_t SaveMode = MegaMoeBF16Training::kSaveMxFp8, bool TransW = true, typename TopkWeightT = bfloat16_t>
__aicore__ inline void GroupMatmulSwiglu(
    ForwardGroupMatmulParam<0, TransW>& gmm1Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluForwardBF16<Gmm1L1TileShape, true, TopkWeightT>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom, uint32_t expertIdx,
    bool useTileReady, uint32_t readyBlocksPerProducerPartition)
{
    using ReadyLayout = MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
    static_assert(ReadyLayout::kReadyRows % GMM1_TILE_M == 0,
                  "Dispatch ready rows must contain whole BF16 GMM1 M tiles");
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm1Params.n;
    const uint32_t k = gmm1Params.k;

    AscendC::GlobalTensor<int32_t> dispatchFlags;
    dispatchFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    if (Ops::Base::CeilDiv(m, GMM1_TILE_M) == 1) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (Ops::Base::CeilDiv(n, L1_TILE_N) == 1) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm1Params.layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm1Params.l0cOutUbFirst, gmm1Params.layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm1Params.l0cOutUbSecond, gmm1Params.layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler({m, gmm1Params.outputN, k}, Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;
    const uint32_t nStripeWidth = SelectGmm1NStripeWidth(m, gmm1Params.outputN, gmm1Params.blockNum);

    MegaMoeBF16ForwardSchedule::Schedule phaseSchedule;
    MegaMoeBF16ForwardSchedule::View phaseView;
    phaseView.Init(gmmAddrInfo.groupFlagList2 - expertIdx,
                   reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM),
                   reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1),
                   params.tilingData->maxOutputSize, params.tilingData->expertPerRank, params.tilingData->epWorldSize,
                   gmm1Params.blockNum, gmm1Params.dispatchBlockNumPerEP, params.tilingData->k);
    const bool active = g_coreType == AscendC::AIC || gmm1Params.subBlockIdx == 0;
    phaseSchedule.Init(phaseView, expertIdx, 0, m, gmm1Params.outputN, nStripeWidth, useTileReady && active);
    bool dispatchExpertReady = false;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        Catlass::GemmCoord blockCoord{0, 0, 0};
        if (phaseSchedule.Enabled()) {
            const auto xy = phaseSchedule.Get(loopIdx);
            blockCoord = Catlass::GemmCoord{xy.m, xy.n, 0};
        } else {
            blockCoord =
                nStripeWidth > 0
                    ? (useTileReady ? GetGmm1AdaptiveReadyFirstCoord(loopIdx, m, gmm1Params.outputN, nStripeWidth,
                                                                     readyBlocksPerProducerPartition)
                                    : GetGmm1AdaptiveNStripeCoord(loopIdx, m, gmm1Params.outputN, nStripeWidth))
                    : scheduler.GetBlockCoord(loopIdx);
        }
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * GMM1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
            }
            // Automatic readiness checks each task's input when tile-ready is
            // selected, since static swizzling can visit non-adjacent M tiles.
            // FORCE_EXPERT_READY waits once for this expert's complete input.
            if constexpr (DispatchTileReady && WaitForFullExpert) {
                // Forced expert mode publishes partition completion only.
                // The row-counter storage remains allocated; this path waits
                // once on the separate expert-completion counter.
                if (loopIdx == startLoopIdx) {
                    __gm__ int32_t* expertFlag =
                        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {
                        }
                    }
                }
            } else if constexpr (DispatchTileReady) {
                if (!useTileReady) {
                    if (loopIdx == startLoopIdx) {
                        __gm__ int32_t* expertFlag =
                            reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                            expertIdx * INT_CACHELINE;
                        const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                        while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {
                            }
                        }
                    }
                } else if (!dispatchExpertReady) {
                    __gm__ int32_t* expertFlag =
                        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t expertTarget = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    dispatchExpertReady = expertTarget == AscendC::ReadGmByPassDCache(expertFlag);
                    if (!dispatchExpertReady) {
                        const uint32_t readyBlock = mLoc / ReadyLayout::kReadyRows;
                        __gm__ int32_t* flag = gmmAddrInfo.groupFlagList2 + readyBlock * INT_CACHELINE;
                        const int32_t target = static_cast<int32_t>(ReadyLayout::RowsInReadyBlock(readyBlock, m));
                        while (target != AscendC::ReadGmByPassDCache(flag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {
                            }
                        }
                    }
                }
            } else if (loopIdx == startLoopIdx) {
                const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                __gm__ int32_t* flag = (__gm__ int32_t*)dispatchFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {
                    }
                }
            }
            if (vecSetSyncCom) {
                WaitForVector();
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockUbFirst =
                tla::GetTile(tensorUbFirst, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond =
                tla::GetTile(tensorUbSecond, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc + weightBlock * gmm1Params.outputN),
                                                 tla::MakeShape(actualShape.k(), actualShape.n()));
                auto& tensorBlockUb = weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond;
                gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            }
            NotifyVector();
        }

        vecSetSyncCom = 1;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == 0) {
                AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0,
                                                                                 0};
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * gmm1Params.outputN + nLoc, 0, 0, 0, 0, 0};
                if (loopIdx == startLoopIdx) {
                }
                WaitForCube();
                if (loopIdx == startLoopIdx) {
                }
                epilogueOp.template Run<SaveMode>(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm1Params.subBlockIdx == 0 && startLoopIdx < coreLoops) {
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

template <bool DispatchTileReady = false, bool WaitForFullExpert = false,
          uint32_t SaveMode = MegaMoeBF16Training::kSaveMxFp8, bool TransW = true, typename TopkWeightT = bfloat16_t>
__aicore__ inline void GroupMatmulSwigluGm(
    ForwardGroupMatmulParam<0, TransW, true>& gmm1Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluForwardBF16<Gmm1L1TileShape, true, TopkWeightT>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx, bool useTileReady,
    uint32_t readyBlocksPerProducerPartition, uint64_t savedElementOffset)
{
    using ReadyLayout = MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
    static_assert(ReadyLayout::kReadyRows % GMM1_TILE_M == 0,
                  "Dispatch ready rows must contain whole BF16 GMM1 M tiles");
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm1Params.n;
    const uint32_t k = gmm1Params.k;

    AscendC::GlobalTensor<int32_t> dispatchFlags;
    dispatchFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    if (Ops::Base::CeilDiv(m, GMM1_TILE_M) == 1) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (Ops::Base::CeilDiv(n, L1_TILE_N) == 1) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm1Params.layoutB, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler({m, gmm1Params.outputN, k}, Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;
    const uint32_t nStripeWidth = SelectGmm1NStripeWidth(m, gmm1Params.outputN, gmm1Params.blockNum);

    // Word 1 publishes raw Gate/Up readiness; ResetWorkSpaceFlagList clears this
    // entire cache line before Dispatch. Word 0 remains the Hp task counter.
    __gm__ int32_t* gmmReady = gmmAddrInfo.groupFlagList + 1;
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm1Params.subBlockIdx == 1) {
            startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
            return;
        }

        while (AscendC::ReadGmByPassDCache(gmmReady) != gmm1Params.blockNum) {
            const int64_t start = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - start < 100) {
            }
        }
    }
    AscendC::GlobalTensor<bfloat16_t> savedGate, savedUp;
    // BF16 already owns persistent raw planes. none/mxfp8 reuse Hp for
    // raw Gate; only Up is appended after the unchanged ready-counter region.
    // Every AIC drains Fixpipe before publishing this expert's gmmReady.
    // Consumers wait for all publishers, then own disjoint tiles. Each Gate
    // tile is fully loaded before its Hp store overwrites that same tile.
    constexpr bool reuseHp = SaveMode != MegaMoeBF16Training::kSaveBf16;
    const ReadyLayout rawLayout(params.tilingData->n, params.tilingData->expertPerRank,
                                params.tilingData->maxOutputSize);
    GM_ADDR rawGate = reuseHp ? params.workspaceInfo.ptrA2 : params.workspaceInfo.ptrSavedGate;
    GM_ADDR rawUp = reuseHp ? params.workspaceInfo.ptrA2 + rawLayout.totalBytes : params.workspaceInfo.ptrSavedUp;
    savedGate.SetGlobalBuffer((__gm__ bfloat16_t*)rawGate + savedElementOffset);
    savedUp.SetGlobalBuffer((__gm__ bfloat16_t*)rawUp + savedElementOffset);
    savedGate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    savedUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    auto gmLayout = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, gmm1Params.outputN);
    auto gmGate = tla::MakeTensor(savedGate, gmLayout, Catlass::Arch::PositionGM{});
    auto gmUp = tla::MakeTensor(savedUp, gmLayout, Catlass::Arch::PositionGM{});

    MegaMoeBF16ForwardSchedule::Schedule phaseSchedule;
    MegaMoeBF16ForwardSchedule::View phaseView;
    phaseView.Init(gmmAddrInfo.groupFlagList2 - expertIdx,
                   reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM),
                   reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1),
                   params.tilingData->maxOutputSize, params.tilingData->expertPerRank, params.tilingData->epWorldSize,
                   gmm1Params.blockNum, gmm1Params.dispatchBlockNumPerEP, params.tilingData->k);
    const bool active = g_coreType == AscendC::AIC || gmm1Params.subBlockIdx == 0;
    phaseSchedule.Init(phaseView, expertIdx, 0, m, gmm1Params.outputN, nStripeWidth, useTileReady && active);
    bool dispatchExpertReady = false;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        Catlass::GemmCoord blockCoord{0, 0, 0};
        if (phaseSchedule.Enabled()) {
            const auto xy = phaseSchedule.Get(loopIdx);
            blockCoord = Catlass::GemmCoord{xy.m, xy.n, 0};
        } else {
            blockCoord =
                nStripeWidth > 0
                    ? (useTileReady ? GetGmm1AdaptiveReadyFirstCoord(loopIdx, m, gmm1Params.outputN, nStripeWidth,
                                                                     readyBlocksPerProducerPartition)
                                    : GetGmm1AdaptiveNStripeCoord(loopIdx, m, gmm1Params.outputN, nStripeWidth))
                    : scheduler.GetBlockCoord(loopIdx);
        }
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * GMM1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
            }
            // Automatic readiness checks each task's input when tile-ready is
            // selected, since static swizzling can visit non-adjacent M tiles.
            // FORCE_EXPERT_READY waits once for this expert's complete input.
            if constexpr (DispatchTileReady && WaitForFullExpert) {
                // Forced expert mode publishes partition completion only.
                // The row-counter storage remains allocated; this path waits
                // once on the separate expert-completion counter.
                if (loopIdx == startLoopIdx) {
                    __gm__ int32_t* expertFlag =
                        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {
                        }
                    }
                }
            } else if constexpr (DispatchTileReady) {
                if (!useTileReady) {
                    if (loopIdx == startLoopIdx) {
                        __gm__ int32_t* expertFlag =
                            reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                            expertIdx * INT_CACHELINE;
                        const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                        while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {
                            }
                        }
                    }
                } else if (!dispatchExpertReady) {
                    __gm__ int32_t* expertFlag =
                        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t expertTarget = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    dispatchExpertReady = expertTarget == AscendC::ReadGmByPassDCache(expertFlag);
                    if (!dispatchExpertReady) {
                        const uint32_t readyBlock = mLoc / ReadyLayout::kReadyRows;
                        __gm__ int32_t* flag = gmmAddrInfo.groupFlagList2 + readyBlock * INT_CACHELINE;
                        const int32_t target = static_cast<int32_t>(ReadyLayout::RowsInReadyBlock(readyBlock, m));
                        while (target != AscendC::ReadGmByPassDCache(flag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {
                            }
                        }
                    }
                }
            } else if (loopIdx == startLoopIdx) {
                const uint32_t target = params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                __gm__ int32_t* flag = (__gm__ int32_t*)dispatchFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {
                    }
                }
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockGate =
                tla::GetTile(gmGate, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUp =
                tla::GetTile(gmUp, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc + weightBlock * gmm1Params.outputN),
                                                 tla::MakeShape(actualShape.k(), actualShape.n()));
                auto& tensorBlockOutput = weightBlock == 0 ? tensorBlockGate : tensorBlockUp;
                gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockOutput, actualShape);
            }
        }

        // No UB producer/consumer handshake on the GM path.
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == 0) {
                AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0,
                                                                                 0};
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * gmm1Params.outputN + nLoc, 0, 0, 0, 0, 0};
                epilogueOp.LoadRawTile(rawGate, rawUp, savedElementOffset, actualShape.m(), actualShape.n(),
                                       Get<0>(epilogueOffset));
                // Quantized persistent state is produced from unmodified UB
                // values before ComputeSwiglu. BF16 raw planes are already saved.
                constexpr uint32_t epilogueSave = SaveMode == MegaMoeBF16Training::kSaveMxFp8
                                                      ? MegaMoeBF16Training::kSaveMxFp8
                                                      : MegaMoeBF16Training::kSaveNone;
                epilogueOp.template Run<epilogueSave>(epilogueShape, epilogueOffset);
            }
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        // Include every AIC, including those with no tasks for this expert.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::AtomicAdd(gmmReady, 1);
        if (startLoopIdx < coreLoops) {
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (startLoopIdx < coreLoops) {
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

template <bool TransW = true>
__aicore__ inline void GroupMatmul2(ForwardGroupMatmulParam<1, TransW>& gmm2Params, const Params& params,
                                    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                    uint32_t groupIdx, uint32_t expertRowOffset, uint16_t& pingpongIdx, int32_t rankId)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm2Params.n;
    const uint32_t k = gmm2Params.k;

    AscendC::GlobalTensor<int32_t> swigluFlags;
    swigluFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm2Params.layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm2Params.l0cOutUbFirst, gmm2Params.layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm2Params.l0cOutUbSecond, gmm2Params.layoutC, Catlass::Arch::PositionUB{});

    Gmm2BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
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
        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto outputUb = pingpongIdx == 0 ? gmm2Params.l0cOutUbFirst : gmm2Params.l0cOutUbSecond;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmm1Scheduler({m, k, n}, Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
                const uint32_t target = gmm1Scheduler.GetCoreLoops();
                __gm__ int32_t* flag = (__gm__ int32_t*)swigluFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {
                    }
                }
            }
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx + FLAG_ID_MAX);
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB =
                tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb =
                tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(pingpongIdx + FLAG_ID_MAX);
        }

        ++vecSetSyncCom;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm2Params.subBlockIdx == 1) {
                if (loopIdx == startLoopIdx) {
                }
                WaitForCube(pingpongIdx);
                if (loopIdx == startLoopIdx) {
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                    mLoc, nLoc, n, groupIdx, expertRowOffset, rankId, outputUb, actualShape, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                NotifyCube(pingpongIdx);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm2Params.subBlockIdx == 1 && startLoopIdx < coreLoops) {
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

}  // namespace MegaMoeBF16ForwardCatlassImpl

#endif  // DEEPEP_MEGA_MOE_BF16_FORWARD_CATLASS_IMPL_HPP
