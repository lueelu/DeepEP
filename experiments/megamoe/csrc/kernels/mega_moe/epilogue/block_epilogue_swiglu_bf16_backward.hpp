// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*
 * SwiGLU backward epilogues for MegaMoe BF16.
 *
 * H=SiLU_FP32(gate)*up and dH=p*dHp stay in FP32 registers.
 * Hp is rounded to BF16 for GMM2/dW2, while activation derivatives
 * stay in FP32 until dGate/dUp are written as BF16 dZ.
 *
 * The row owner reads dHp from GM, computes FP32 dHp*h, and replaces dHp
 * with BF16 Hp after consuming it. Gate/Up come from saved state or GMM1
 * recomputation. Tiled callers stage dHp in the first UB work tile.
 *
 * Backward reconstructs the same FP32 H used for forward route weighting.
 * Saved MXFP8 reconstructs H from decoded Gate/Up, so it may differ from
 * the activation computed from the original forward Gate/Up.
 *
 * Tile addressing convention (matches the forward GMM1 epilogue): the caller
 * passes a single flattened block offset `mLoc * F + nLoc` (rows * plane
 * width + tile column base) in Get<0>(blockCoord).  Row strides are handled
 * by DataCopyPad.
 */
#ifndef DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP
#define DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "stage/mega_moe_slice_ready.hpp"
#include <adv_api/reduce/reduce.h>
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "common/mega_moe_bf16_context.hpp"
#include "common/mega_moe_bf16_layout.hpp"

namespace MegaMoeBF16Impl {

using MegaMoeImpl::Get;

static constexpr AscendC::Reg::DivSpecificMode kBackwardDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
static constexpr AscendC::Reg::CastTrait kBackwardCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kBackwardCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Saturate nonzero clamp differences to form a 0/1 range indicator.
static constexpr float kClampMaskScale = 1.0e30F;

template <typename L0TileShape_, typename TopkWeightT = bfloat16_t>
class BlockEpilogueSwigluBackwardBF16 {
public:
    __aicore__ inline void SetTimer(MegaMoeTrainTimerDevice* timer)
    {
        timer_ = timer;
    }
    __aicore__ inline void SetReductionRows(uint64_t rows)
    {
        reductionRows_ = rows;
    }
    __aicore__ inline uint32_t SavedMode() const
    {
        return params_.savedGateUpMode;
    }
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    struct Arguments {
        GM_ADDR gatePlaneGmAddr{nullptr};
        GM_ADDR upPlaneGmAddr{nullptr};
        GM_ADDR gateScaleGmAddr{nullptr};
        GM_ADDR upScaleGmAddr{nullptr};
        GM_ADDR gateHighGmAddr{nullptr};
        GM_ADDR upLowGmAddr{nullptr};
        GM_ADDR upHighGmAddr{nullptr};
        GM_ADDR gradZGmAddr{nullptr};
        GM_ADDR hpGmAddr{nullptr};
        GM_ADDR dpProductsGmAddr{nullptr};
        GM_ADDR packedTopkWeightsGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        uint32_t savedGateUpMode{MegaMoeBF16Training::kSaveNone};
        GM_ADDR wideDpGmAddr{nullptr};
    };

    using Params = Arguments;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            params_ = params;
            gatePlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gatePlaneGmAddr);
            upPlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.upPlaneGmAddr);
            gateMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.gatePlaneGmAddr);
            upMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.upPlaneGmAddr);
            gateScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateScaleGmAddr);
            upScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.upScaleGmAddr);
            gateHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateHighGmAddr);
            upLowGm_.SetGlobalBuffer((__gm__ int8_t*)params.upLowGmAddr);
            upHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.upHighGmAddr);
            gradZGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gradZGmAddr);
            hpGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.hpGmAddr);
            dpPartialsGm_.SetGlobalBuffer((__gm__ float*)params.dpProductsGmAddr);
            packedTopkWeightsGm_.SetGlobalBuffer((__gm__ TopkWeightT*)params.packedTopkWeightsGmAddr);
        }
    }

    // baseRows: cumulative packed expert row offset for all GM planes.
    // Gate/Up/Hp use row width F, gradZ uses N, and dP partials use
    // dpPartialTiles_; each plane applies this same row base.
    __aicore__ inline void UpdateGlobalAddr(int64_t baseRows)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            rowBase_ = baseRows;
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
        outputF_ = outputN_ / SWIGLU_N_HALF;
        dpPartialTiles_ = MegaMoeBF16Training::BackwardTopkPartialCount(params_.savedGateUpMode, outputF_);
    }

    // One AIV owns a complete row: read dHp/Gate/Up once, form dP, Hp and dZ,
    // then replace that row's dHp with Hp. No other AIV consumes its columns.
    // The AR primitive retains the tensor reduction's FP32 addition tree.
    __aicore__ inline void SwiGLUGradRows(GM_ADDR dHpBase, uint32_t rows, uint32_t readerIndex, uint32_t readers)
    {
        if (MegaMoeBF16Training::UseStreamedTopkGrad(params_.savedGateUpMode, outputF_)) {
            SwiGLUGradWideRows(dHpBase, rows, readerIndex, readers);
            return;
        }
        const uint16_t f = static_cast<uint16_t>(outputF_);
        const uint32_t rowBytes = static_cast<uint32_t>(f) * sizeof(bfloat16_t);
        // Stage adjacent routes together; each row keeps its own reduction tree.
        constexpr uint32_t kResultBytes = MegaMoeBF16Training::kWholeRowTopkResultBytes;
        const uint32_t bytesPerRow = rowBytes * 3U + static_cast<uint32_t>(f) * sizeof(float);
        const uint32_t batchRows = MegaMoeBF16Training::TopkGradBatchRows(f);
        const uint32_t inputBytes = rowBytes * batchRows;
        AscendC::LocalTensor<bfloat16_t> d{AscendC::TPosition::VECIN, 0, f * batchRows};
        AscendC::LocalTensor<bfloat16_t> g{AscendC::TPosition::VECIN, inputBytes, f * batchRows};
        AscendC::LocalTensor<bfloat16_t> u{AscendC::TPosition::VECIN, inputBytes * 2U, f * batchRows};
        AscendC::LocalTensor<float> product{AscendC::TPosition::VECCALC, inputBytes * 3U, f * batchRows};
        AscendC::LocalTensor<float> result{AscendC::TPosition::VECCALC, bytesPerRow * batchRows,
                                           kResultBytes / sizeof(float)};
        AscendC::GlobalTensor<bfloat16_t> dhp;
        dhp.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dHpBase));
        // Keep the tensor reduction plan tied to all AIV lanes; work ownership is independent.
        const auto plan =
            MegaMoeBF16Training::MakeTopkGradReductionPlan(reductionRows_, f, 2U * AscendC::GetBlockNum());
        const uint32_t width = plan.width;
        const uint16_t parts = f / width;
        uint16_t mainParts = 1, folds = 0;
        while (2U * mainParts < parts) {
            mainParts *= 2U;
            ++folds;
        }
        const uint16_t tailParts = parts - mainParts;
        for (uint32_t row = readerIndex * batchRows; row < rows; row += readers * batchRows) {
            const uint32_t take = rows - row < batchRows ? rows - row : batchRows;
            // Retire the previous dP write before reusing the batch buffers.
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            LoadPackedWeights(take, row);
            const uint64_t index = static_cast<uint64_t>(row) * f;
            LoadRowInputs(dhp, index, (rowBase_ + row) * f, take * f, d, g, u);
            ComputeRowOutputs(d, g, u, product, take, f);
            AscendC::PipeBarrier<PIPE_V>();
            if (f % width || plan.stripesPerGroup < parts) {
                // The input planes now hold output. Retire their writes so the
                // grouped reducer can reuse that UB while the products stay live.
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                StoreRowOutputs(row, f, take, d, g, u);
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                for (uint32_t i = 0; i < take; ++i) {
                    ReduceGroupedRow(product[i * f], f, plan, rowBase_ + row + i);
                }
                continue;
            }
            if (mainParts == 1U && width % 64U == 0U) {
                if (tailParts != 0U) {
                    // Fold matching columns, then pack rows for one AR call.
                    auto* pp = reinterpret_cast<__ubuf__ float*>(product.GetPhyAddr());
                    __VEC_SCOPE__
                    {
                        for (uint16_t offset = 0; offset < take; ++offset) {
                            uint32_t remaining = width;
                            for (uint16_t col = 0; col < width; col += 64U) {
                                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                                AscendC::Reg::RegTensor<float> left, right, sum;
                                AscendC::Reg::DataCopy(left, pp + offset * f + col);
                                AscendC::Reg::DataCopy(right, pp + offset * f + width + col);
                                AscendC::Reg::Add(sum, left, right, mask);
                                AscendC::Reg::DataCopy(pp + offset * width + col, sum, mask);
                            }
                            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE,
                                                      AscendC::Reg::MemType::VEC_LOAD>();
                        }
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                uint32_t shape[2] = {take, width};
                auto scratch = product.template ReinterpretCast<uint8_t>();
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(result, product, scratch, shape, false);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(dpPartialsGm_[rowBase_ + row], result,
                                     AscendC::DataCopyExtParams{1U, take * sizeof(float), 0U, 0U, 0U});
            } else {
                for (uint32_t offset = 0; offset < take; ++offset) {
                    auto rowProduct = product[offset * f];
                    auto rowResult = result[offset * mainParts * 8U];
                    auto scratch = rowProduct.template ReinterpretCast<uint8_t>();
                    for (uint16_t part = 0; part < mainParts; ++part) {
                        if (part < tailParts) {
                            AscendC::Add(rowProduct[part * width], rowProduct[part * width],
                                         rowProduct[(part + mainParts) * width], width);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        uint32_t shape[2] = {1U, width};
                        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                            rowResult[part * 8U], rowProduct[part * width], scratch, shape, false);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    auto* partial = reinterpret_cast<__ubuf__ float*>(rowResult.GetPhyAddr());
                    __VEC_SCOPE__
                    {
                        auto one = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
                        for (uint16_t level = 0; level < folds; ++level) {
                            const uint16_t pairs = mainParts >> (level + 1U);
                            for (uint16_t pair = 0; pair < pairs; ++pair) {
                                AscendC::Reg::RegTensor<float> left, right, sum;
                                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                                    left, partial + pair * 16U);
                                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                                    right, partial + pair * 16U + 8U);
                                AscendC::Reg::Add(sum, left, right, one);
                                AscendC::Reg::DataCopy(partial + pair * 8U, sum, one);
                            }
                            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE,
                                                      AscendC::Reg::MemType::VEC_LOAD>();
                        }
                    }
                }
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(dpPartialsGm_[rowBase_ + row], result,
                                     AscendC::DataCopyExtParams{static_cast<uint16_t>(take), sizeof(float),
                                                                static_cast<uint32_t>(mainParts - 1U), 0U, 0U});
            }
            StoreRowOutputs(row, f, take, d, g, u);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    // Saved-mode GM path: each AIV loads its assigned dHp tile into its
    // private first UB work tile before calling the shared SwiGLU epilogue.
    __aicore__ inline void LoadDHpFromGM(GM_ADDR src, uint32_t rows, uint32_t columns, uint32_t globalStride)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src));
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),
                                        static_cast<uint32_t>(columns * sizeof(bfloat16_t)),
                                        static_cast<uint32_t>((globalStride - columns) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(l0cOutUbFirst_, gm, copy, pad);
        // Complete the dHp read before its in-place replacement with Hp.
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    // ---- P1: dHp is already in the first work tile (UB offset 0).
    // Full 128x256 tiles fuse Hp, dP partials and dZ after one Gate/Up load.
    // Tail tiles store Hp/partials, reload Gate/Up, then compute dZ because
    // their staged operations reuse the same three UB work tiles. ----
    __aicore__ inline void SwiGLUGrad(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }
        const int64_t flatOffset = Get<0>(blockCoord);
        const uint32_t mLoc = static_cast<uint32_t>(flatOffset / outputF_);
        const uint32_t nLoc = static_cast<uint32_t>(flatOffset % outputF_);
        const uint32_t rowStrideInPlane = Ops::Base::CeilAlign(
            static_cast<uint32_t>(singleN), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t)));

        // One contiguous p load feeds FP32 H*p and dHp*p without narrowing FP32 weights.
        LoadPackedWeights(singleM, mLoc);

        // For direct UB output, the caller's PIPE_V WaitForCube protects Fixpipe dHp;
        // GM callers already loaded dHp with LoadDHpFromGM. Bridge V to MTE2
        // before reusing the other two work tiles for Gate/Up: none reads
        // recompute GM, while saved modes read forward-persistent GM.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        // Load Gate/Up while dHp remains in the first UB work tile.
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // Full 128x256 tiles consume this Gate/Up load without a second GM read.
        if (singleM == UbLayout::kTileRows && singleN == UbLayout::kTileColumns && rowStrideInPlane == singleN) {
            RunFusedFullTile(singleM, singleN, mLoc, nLoc);
            return;
        }
        ComputeForwardAux(singleM, singleN, rowStrideInPlane);

        // tile1 holds FP32 partials and tile2 holds Hp until MTE3 consumes both.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (!MegaMoeBF16Training::UseWholeRowTopkGrad(params_.savedGateUpMode, outputF_)) {
            const uint32_t dpTileIdx = nLoc / BACKWARD_DP_PARTIAL_TILE_N;
            AscendC::DataCopyExtParams partialCopyParams{
                static_cast<uint16_t>(singleM), static_cast<uint32_t>(sizeof(float)),
                // srcStride counts 32-byte blocks; the 4-byte partial occupies one padded block.
                static_cast<uint32_t>(rowStrideInPlane * sizeof(bfloat16_t) / AscendC::ONE_BLK_SIZE - 1U),
                static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0};
            AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + dpTileIdx],
                                 l0cOutUbSecond_.template ReinterpretCast<float>(), partialCopyParams);
        }
        AscendC::DataCopyExtParams auxCopyParams{static_cast<uint16_t>(singleM),
                                                 static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                                 static_cast<uint32_t>((outputF_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(hpGm_[rowBase_ * outputF_ + flatOffset], extraA_, auxCopyParams);

        // Retire both MTE3 sources before reloading gate/up.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // Read raw dHp and apply p in FP32 inside the derivative.
        ComputeSwigluGrad(singleM, singleN, rowStrideInPlane);

        // Retire V's up-tile reads before the next MTE2 refill.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        // dGate and dUp occupy column planes 0 and F.
        AscendC::DataCopyExtParams gradCopyParams{static_cast<uint16_t>(singleM),
                                                  static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                                  static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + nLoc], l0cOutUbFirst_, gradCopyParams);
        AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + outputF_ + nLoc], l0cOutUbSecond_,
                             gradCopyParams);
        // Retire dUp's MTE3 read before the next gate load reuses this tile.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        // Retire dZ writes before the caller signals completion to the consumer.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

private:
    MegaMoeTrainTimerDevice* timer_ = nullptr;
    using UbLayout = MegaMoeBF16Training::BackwardSwigluUbLayout;
    static_assert(tla::get<0>(L0TileShape_{}) == UbLayout::kTileRows &&
                      tla::get<1>(L0TileShape_{}) == UbLayout::kTileColumns,
                  "Backward SwiGLU tile must match its UB layout");
    static_assert(UbLayout::kMxScratchEnd <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The backward SwiGLU work tiles and MXFP8 decoder must fit in A5 UB");
    static constexpr uint32_t kWeightBytes = (UbLayout::kTileRows * sizeof(TopkWeightT) + 31U) / 32U * 32U;
    static constexpr uint32_t kWeightOffset = Catlass::Arch::Ascend950::UB_SIZE - kWeightBytes;
    static_assert(kWeightOffset >= UbLayout::kMxScratchEnd,
                  "Prefetched weights must not alias live MXFP8 decode or clip masks");
    static_assert(kWeightOffset >= UbLayout::kMxScratchEnd + UbLayout::kTileRows * 8U * sizeof(float),
                  "Prefetched weights must not alias the SwiGLU dP scratch");

    using Bf16Tile = AscendC::LocalTensor<bfloat16_t>;
    using Fp32Tile = AscendC::LocalTensor<float>;

    __aicore__ inline void StoreRowOutputs(uint32_t row, uint32_t f, uint32_t take, Bf16Tile d, Bf16Tile g, Bf16Tile u)
    {
        const uint32_t rowBytes = f * sizeof(bfloat16_t);
        AscendC::DataCopyPad(hpGm_[(rowBase_ + row) * f], u,
                             AscendC::DataCopyExtParams{1U, take * rowBytes, 0U, 0U, 0U});
        const AscendC::DataCopyExtParams gradCopy{static_cast<uint16_t>(take), rowBytes, 0U, rowBytes, 0U};
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + row) * outputN_], d, gradCopy);
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + row) * outputN_ + f], g, gradCopy);
    }

    __aicore__ inline void LoadRowInputs(const AscendC::GlobalTensor<bfloat16_t>& dhp, uint64_t dhpIndex,
                                         uint64_t first, uint32_t count, Bf16Tile d, Bf16Tile g, Bf16Tile u)
    {
        const AscendC::DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(d, dhp[dhpIndex], copy, {});

        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8) {
            MegaMoeTrainTimerSpanScope decode(timer_, MegaMoeTrainTimer::AIV_BW_DECODE_GATE_UP);
            LoadMxWholeRows(first, count, g, true);
            LoadMxWholeRows(first, count, u, false);
        } else {
            AscendC::DataCopyPad(g, gatePlaneGm_[first], copy, {});
            AscendC::DataCopyPad(u, upPlaneGm_[first], copy, {});
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void ComputeRowOutputs(Bf16Tile d, Bf16Tile g, Bf16Tile u, Fp32Tile product, uint32_t take,
                                             uint16_t f)
    {
        const bool clip = params_.swigluLimit != kMegaMoeSwigluLimitDisabled;
        const bool mxClip = params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8;
        const float limit = params_.swigluLimit;
        auto* dp = reinterpret_cast<__ubuf__ bfloat16_t*>(d.GetPhyAddr());
        auto* gp = reinterpret_cast<__ubuf__ bfloat16_t*>(g.GetPhyAddr());
        auto* up = reinterpret_cast<__ubuf__ bfloat16_t*>(u.GetPhyAddr());
        auto* pp = reinterpret_cast<__ubuf__ float*>(product.GetPhyAddr());
        auto* weights = reinterpret_cast<__ubuf__ TopkWeightT*>(packedWeightUb_.GetPhyAddr());
        __VEC_SCOPE__
        {
            auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<float> one, zero, lim, negLim;
            AscendC::Reg::Duplicate(one, 1.0F, all);
            if (clip) {
                AscendC::Reg::Duplicate(zero, 0.0F, all);
                AscendC::Reg::Duplicate(lim, limit, all);
                AscendC::Reg::Duplicate(negLim, -limit, all);
            }
            for (uint16_t batchRow = 0; batchRow < take; ++batchRow) {
                AscendC::Reg::RegTensor<float> pFp;
                if constexpr (std::is_same_v<TopkWeightT, bfloat16_t>) {
                    AscendC::Reg::RegTensor<bfloat16_t> pBf;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + batchRow);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights + batchRow);
                }
                uint32_t remaining = f;
                for (uint16_t col = 0; col < f; col += 64U) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = batchRow * static_cast<uint32_t>(f) + col;
                    AscendC::Reg::RegTensor<bfloat16_t> db, gb, ub;
                    AscendC::Reg::RegTensor<float> df, gf, uf, neg, exponent, denom, silu, h, value;
                    AscendC::Reg::RegTensor<float> gateInd, upInd, clippedGate, clippedUp, diff;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(db, dp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gb, gp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(ub, up + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(df, db, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gf, gb, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(uf, ub, mask);
                    if (clip) {
                        AscendC::Reg::Min(clippedGate, gf, lim, mask);
                        AscendC::Reg::Max(clippedUp, uf, negLim, mask);
                        AscendC::Reg::Min(clippedUp, clippedUp, lim, mask);
                        if (mxClip) {
                            // Saved masks turn clipped values into signed infinities.
                            // Finite decoded overshoots still receive a gradient.
                            AscendC::Reg::MaskReg active;
                            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(active, gf, 0x1.fep127F, mask);
                            AscendC::Reg::Select(gateInd, one, zero, active);
                            AscendC::Reg::Abs(diff, uf, mask);
                            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(active, diff, 0x1.fep127F, mask);
                            AscendC::Reg::Select(upInd, one, zero, active);
                        } else {
                            AscendC::Reg::Sub(diff, gf, clippedGate, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, one, mask);
                            AscendC::Reg::Sub(gateInd, one, diff, mask);
                            AscendC::Reg::Sub(diff, uf, clippedUp, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, one, mask);
                            AscendC::Reg::Sub(upInd, one, diff, mask);
                        }
                        gf = clippedGate;
                        uf = clippedUp;
                    }
                    AscendC::Reg::Muls(neg, gf, -1.0F, mask);
                    AscendC::Reg::Exp(exponent, neg, mask);
                    AscendC::Reg::Adds(denom, exponent, 1.0F, mask);
                    AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gf, denom, mask);
                    AscendC::Reg::Mul(h, silu, uf, mask);
                    AscendC::Reg::Mul(value, df, h, mask);
                    AscendC::Reg::DataCopy(pp + offset, value, mask);
                    AscendC::Reg::RegTensor<float> hpFp, dh, sig, gradSwish, oneMinusS, inner, term, dGateFp, dUpFp;
                    AscendC::Reg::RegTensor<bfloat16_t> hpOut, dGateOut, dUpOut;
                    AscendC::Reg::Mul(hpFp, h, pFp, mask);
                    AscendC::Reg::Mul(dh, df, pFp, mask);
                    AscendC::Reg::Div<float, &kBackwardDivMode>(sig, one, denom, mask);
                    AscendC::Reg::Mul(dUpFp, dh, silu, mask);
                    AscendC::Reg::Mul(gradSwish, dh, uf, mask);
                    AscendC::Reg::Muls(oneMinusS, sig, -1.0F, mask);
                    AscendC::Reg::Adds(oneMinusS, oneMinusS, 1.0F, mask);
                    AscendC::Reg::Mul(inner, gf, oneMinusS, mask);
                    AscendC::Reg::Adds(inner, inner, 1.0F, mask);
                    AscendC::Reg::Mul(term, sig, inner, mask);
                    AscendC::Reg::Mul(dGateFp, gradSwish, term, mask);
                    if (clip) {
                        AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                        AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                    }
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpOut, hpFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                    // Inputs are dead at this offset; reuse their planes for outputs.
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dp + offset, dGateOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gp + offset, dUpOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(up + offset, hpOut,
                                                                                               mask);
                }
            }
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }
    }

    // One active AIV reuses a single GM row across all of its routes. The
    // per-column results are written once, before the FP32 reduction reads them.
    __aicore__ inline void SwiGLUGradWideRows(GM_ADDR dHpBase, uint32_t rows, uint32_t owner, uint32_t owners)
    {
        constexpr uint32_t stripe = 4096;
        const uint32_t f = outputF_;
        const auto plan =
            MegaMoeBF16Training::MakeTopkGradReductionPlan(reductionRows_, f, 2U * AscendC::GetBlockNum());
        AscendC::GlobalTensor<float> products;
        products.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params_.wideDpGmAddr) +
                                 static_cast<uint64_t>(AscendC::GetBlockIdx()) * f);
        AscendC::GlobalTensor<bfloat16_t> dhp;
        dhp.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dHpBase));
        Bf16Tile d{AscendC::TPosition::VECIN, 0, stripe};
        Bf16Tile g{AscendC::TPosition::VECIN, 2 * stripe, stripe};
        Bf16Tile u{AscendC::TPosition::VECIN, 4 * stripe, stripe};
        Fp32Tile product{AscendC::TPosition::VECCALC, 6 * stripe, stripe};
        for (uint32_t row = owner; row < rows; row += owners) {
            LoadPackedWeights(1, row);
            for (uint32_t col = 0; col < f; col += stripe) {
                const uint32_t count = f - col < stripe ? f - col : stripe;
                const uint64_t src = static_cast<uint64_t>(row) * f + col;
                const uint64_t at = rowBase_ * f + src;
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                LoadRowInputs(dhp, src, at, count, d, g, u);
                ComputeRowOutputs(d, g, u, product, 1, count);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                const AscendC::DataCopyExtParams bf{1, count * 2U, 0, 0, 0}, fp{1, count * 4U, 0, 0, 0};
                const uint64_t dz = (rowBase_ + row) * outputN_ + col;
                AscendC::DataCopyPad(gradZGm_[dz], d, bf);
                AscendC::DataCopyPad(gradZGm_[dz + f], g, bf);
                AscendC::DataCopyPad(hpGm_[at], u, bf);
                AscendC::DataCopyPad(products[col], product, fp);
            }
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            ReduceGroupedRow(products, f, plan, rowBase_ + row);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    // With a partial final stripe, CANN pads missing columns with the sum identity.
    template <class Tensor>
    __aicore__ inline void LoadReductionStripe(Fp32Tile dst, Tensor source, uint32_t col, uint32_t f, uint32_t width)
    {
        const uint32_t count = f - col < width ? f - col : width;
        if (count < width) {
            AscendC::Duplicate(dst, 0.0F, width);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same_v<Tensor, Fp32Tile>) {
            AscendC::Muls(dst, source[col], 1.0F, count);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPad(dst, source[col], AscendC::DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U}, {});
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
    }

    // Match the contiguous AR kernel: each logical core reduces consecutive
    // stripes, then the RA pass sums those group results for the same route.
    template <class Tensor>
    __aicore__ inline void ReduceGroupedRow(Tensor products, uint32_t f,
                                            MegaMoeBF16Training::TopkGradReductionPlan plan, uint64_t row)
    {
        const uint32_t width = plan.width, stripes = (f + width - 1U) / width;
        const uint32_t aligned = (width + 7U) / 8U * 8U;
        Fp32Tile left{AscendC::TPosition::VECCALC, 0, aligned};
        Fp32Tile right{AscendC::TPosition::VECCALC, 4U * aligned, aligned};
        Fp32Tile sums{AscendC::TPosition::VECCALC, 8U * aligned, 33 * 8};
        const uint32_t count = (stripes + plan.stripesPerGroup - 1U) / plan.stripesPerGroup;
        Fp32Tile groupSums{AscendC::TPosition::VECCALC, 8U * aligned + 33U * 32U, count * 8U};
        auto scratch = right.template ReinterpretCast<uint8_t>();
        uint32_t arShape[2] = {1U, width};
        for (uint32_t group = 0; group < count; ++group) {
            const uint32_t begin = group * plan.stripesPerGroup;
            const uint32_t size = stripes - begin < plan.stripesPerGroup ? stripes - begin : plan.stripesPerGroup;
            uint32_t main = 1;
            while (2U * main < size) main *= 2U;
            for (uint32_t i = 0; i < main; ++i) {
                LoadReductionStripe(left, products, (begin + i) * width, f, width);
                if (i + main < size) {
                    LoadReductionStripe(right, products, (begin + i + main) * width, f, width);
                    AscendC::Add(left, left, right, width);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sums, left, scratch, arShape, false);
                AscendC::PipeBarrier<PIPE_V>();
                uint32_t level = 0;
                for (uint32_t carry = i; carry & 1U; carry >>= 1U) {
                    AscendC::Add(sums, sums[(level + 1U) * 8U], sums, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                    ++level;
                }
                if (i + 1U < main) {
                    AscendC::Muls(sums[(level + 1U) * 8U], sums, 1.0F, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            if (count > 1U) {
                AscendC::Muls(groupSums[group * 8U], sums, 1.0F, 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if (count > 1U) {
            uint32_t raShape[2] = {count, 8U};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sums, groupSums, scratch, raShape, false);
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(dpPartialsGm_[row], sums, AscendC::DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void LoadPackedWeights(uint32_t rows, uint32_t mLoc)
    {
        // Each lane retires its own overlapping UB accesses before refilling p.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams p{1U, static_cast<uint32_t>(rows * sizeof(TopkWeightT)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<TopkWeightT> pad{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(packedWeightUb_, packedTopkWeightsGm_[rowBase_ + mLoc], p, pad);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void LoadGateUpTiles(uint32_t rowStart, uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                           AscendC::LocalTensor<bfloat16_t>& gateDst,
                                           AscendC::LocalTensor<bfloat16_t>& upDst)
    {
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8) {
            MegaMoeTrainTimerSpanScope timerDecode(timer_, MegaMoeTrainTimer::AIV_BW_DECODE_GATE_UP);
            DecodeMxFp8Plane(gateMxFp8Gm_, gateScaleGm_, gateHighGm_, gateHighGm_, rowStart, nLoc, singleM, singleN,
                             gateDst, true);
            DecodeMxFp8Plane(upMxFp8Gm_, upScaleGm_, upLowGm_, upHighGm_, rowStart, nLoc, singleM, singleN, upDst,
                             false);
            return;
        }
        const int64_t index = static_cast<int64_t>(rowStart) * outputF_ + nLoc;
        CopyPlaneTileToUb(gatePlaneGm_, index, singleM, singleN, outputF_, gateDst);
        CopyPlaneTileToUb(upPlaneGm_, index, singleM, singleN, outputF_, upDst);
    }

    __aicore__ inline void CopyMxRowsToUb(const AscendC::GlobalTensor<int8_t>& src, int64_t index, uint16_t rows,
                                          uint32_t rowBytes, uint32_t srcRowBytes, AscendC::LocalTensor<int8_t> dst)
    {
        AscendC::DataCopyExtParams copyParams{rows, rowBytes, srcRowBytes - rowBytes, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dst, src[index], copyParams, padParams);
    }

    // Flatten whole rows into contiguous 256-column stripes. The row path owns
    // all columns and may use the product plane as decode scratch until H is formed.
    __aicore__ inline void LoadMxWholeRows(uint64_t first, uint32_t count, AscendC::LocalTensor<bfloat16_t> dst,
                                           bool gate)
    {
        constexpr uint32_t cols = UbLayout::kTileColumns;
        constexpr uint32_t chunk = UbLayout::kDecodeRows * cols;
        static_assert(248U * 1024U * 6U / 10U < UbLayout::kMxDataOffset);
        for (uint32_t offset = 0; offset < count; offset += chunk) {
            uint32_t size = count - offset < chunk ? count - offset : chunk;
            uint16_t rows = size / cols;
            uint64_t at = first + offset;
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(gate ? gateMxFp8Gm_ : upMxFp8Gm_, at, 1, size, size, mxData_);
            CopyMxRowsToUb(gate ? gateScaleGm_ : upScaleGm_, at / 32, rows, cols / 32, cols / 32, mxScale_);
            if (params_.gateHighGmAddr != nullptr) {
                CopyMxRowsToUb(gate ? gateHighGm_ : upHighGm_, at / 8, 1, size / 8, size / 8,
                               gate ? gateHighMask_ : upHighMask_);
                if (!gate) CopyMxRowsToUb(upLowGm_, at / 8, 1, size / 8, size / 8, upLowMask_);
            }
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            auto decoded = dst[offset];
            DecodeMxFp8Rows(rows, cols, cols / 32, decoded);
            AscendC::PipeBarrier<PIPE_V>();
            if (params_.gateHighGmAddr != nullptr) {
                ApplyClipSentinels(decoded, rows, cols, gate, 0);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void DecodeMxFp8Plane(const AscendC::GlobalTensor<int8_t>& dataGm,
                                            const AscendC::GlobalTensor<int8_t>& scaleGm,
                                            const AscendC::GlobalTensor<int8_t>& lowMaskGm,
                                            const AscendC::GlobalTensor<int8_t>& highMaskGm, uint32_t rowStart,
                                            uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                            AscendC::LocalTensor<bfloat16_t>& dst, bool gatePlane)
    {
        const uint32_t scaleWidth = outputF_ / 32;
        const uint32_t groups = singleN / 32;
        const bool hasClipMask = params_.gateHighGmAddr != nullptr;
        for (uint32_t chunkRow = 0; chunkRow < singleM; chunkRow += UbLayout::kDecodeRows) {
            const uint16_t rows = static_cast<uint16_t>(
                singleM - chunkRow > UbLayout::kDecodeRows ? UbLayout::kDecodeRows : singleM - chunkRow);
            const uint32_t globalRow = rowStart + chunkRow;
            const int64_t dataIndex = static_cast<int64_t>(globalRow) * outputF_ + nLoc;
            const int64_t scaleIndex = static_cast<int64_t>(globalRow) * scaleWidth + nLoc / 32;
            // Retire this chunk's V consumers before MTE2 reuses scratch.
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(dataGm, dataIndex, rows, singleN, outputF_, mxData_);
            CopyMxRowsToUb(scaleGm, scaleIndex, rows, groups, scaleWidth, mxScale_);
            if (hasClipMask) {
                const uint32_t maskWidthBytes = scaleWidth * sizeof(uint32_t);
                const int64_t maskIndex =
                    static_cast<int64_t>(globalRow) * maskWidthBytes + (nLoc / 32) * sizeof(uint32_t);
                if (gatePlane) {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   gateHighMask_[chunkRow * UbLayout::kMxScaleRowBytes]);
                } else {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upLowMask_[chunkRow * UbLayout::kMxScaleRowBytes]);
                }
                if (!gatePlane) {
                    CopyMxRowsToUb(highMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upHighMask_[chunkRow * UbLayout::kMxScaleRowBytes]);
                }
            }
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
            auto decoded = dst[chunkRow * singleN];
            DecodeMxFp8Rows(rows, singleN, groups, decoded);
            AscendC::PipeBarrier<PIPE_V>();
            if (hasClipMask) {
                // Restore forward's clipping decision, not a decision recomputed from lossy MXFP8.
                // An out-of-range sentinel makes the FP32 clamp restore the exact finite limit.
                ApplyClipSentinels(decoded, rows, singleN, gatePlane, chunkRow);
            }
        }
    }

    __aicore__ inline void DecodeMxFp8Rows(uint16_t rows, uint32_t columns, uint32_t groups,
                                           AscendC::LocalTensor<bfloat16_t> dst)
    {
        constexpr uint32_t bf16PerReg = 256U / sizeof(bfloat16_t);
        constexpr uint32_t fp32PerReg = 256U / sizeof(float);
        auto data = (__ubuf__ fp8_e4m3fn_t*)mxData_.GetPhyAddr();
        auto scales = (__ubuf__ fp8_e8m0_t*)mxScale_.GetPhyAddr();
        auto scaleBf16 = (__ubuf__ bfloat16_t*)mxScaleBf16_.GetPhyAddr();
        auto scaleFp32 = (__ubuf__ float*)mxScaleFp32_.GetPhyAddr();
        auto decoded = (__ubuf__ bfloat16_t*)dst.GetPhyAddr();
        __VEC_SCOPE__
        {
            static constexpr AscendC::Reg::CastTrait fp8ToWide = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t scaleRemaining = groups;
                const uint16_t scaleRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(groups, bf16PerReg));
                for (uint16_t i = 0; i < scaleRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<bfloat16_t>(scaleRemaining);
                    AscendC::Reg::RegTensor<fp8_e8m0_t> src;
                    AscendC::Reg::RegTensor<bfloat16_t> value;
                    AscendC::Reg::DataCopy<fp8_e8m0_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(
                        src, scales + row * UbLayout::kMxScaleRowBytes + i * bf16PerReg);
                    AscendC::Reg::Cast<bfloat16_t, fp8_e8m0_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_INTLV_B16>(
                        scaleBf16 + i * bf16PerReg * 2, value, value, mask);
                }
                AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                uint32_t scaleWideRemaining = groups * 2;
                const uint16_t scaleWideRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(groups * 2, fp32PerReg));
                for (uint16_t i = 0; i < scaleWideRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<float>(scaleWideRemaining);
                    AscendC::Reg::RegTensor<bfloat16_t> src;
                    AscendC::Reg::RegTensor<float> value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        src, scaleBf16 + i * fp32PerReg);
                    AscendC::Reg::Cast<float, bfloat16_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(
                        scaleFp32 + i * fp32PerReg * 2, value, value, mask);
                }
                AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                uint32_t dataRemaining = columns;
                const uint16_t dataRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(columns, fp32PerReg));
                for (uint16_t i = 0; i < dataRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<float>(dataRemaining);
                    AscendC::Reg::RegTensor<float> scale;
                    AscendC::Reg::RegTensor<fp8_e4m3fn_t> src;
                    AscendC::Reg::RegTensor<float> value;
                    AscendC::Reg::RegTensor<float> result;
                    AscendC::Reg::RegTensor<bfloat16_t> rounded;
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_E2B_B32>(scale, scaleFp32 + i * 8);
                    AscendC::Reg::DataCopy<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                        src, data + row * columns + i * fp32PerReg);
                    AscendC::Reg::Cast<float, fp8_e4m3fn_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::Mul(result, scale, value, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(rounded, result, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        decoded + row * columns + i * fp32PerReg, rounded, mask);
                }
            }
        }
    }

    __aicore__ inline void SetClippedValues(AscendC::LocalTensor<bfloat16_t> dst, AscendC::LocalTensor<int8_t> clipMask,
                                            uint16_t rows, uint32_t columns, uint32_t rowStride, uint16_t replacement)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t lanes = 256U / sizeof(uint16_t);
        const uint16_t repeats = static_cast<uint16_t>(Ops::Base::CeilDiv(columns, lanes));
        auto* dstBits = reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr());
        auto* maskBits = reinterpret_cast<__ubuf__ uint8_t*>(clipMask.GetPhyAddr());
        __VEC_SCOPE__
        {
            auto all = CreateMask<uint16_t, MaskPattern::ALL>();
            RegTensor<uint16_t> value;
            Duplicate(value, replacement, all);
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t remaining = columns;
                for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                    const uint32_t column = repeat * lanes;
                    MaskReg clipped;
                    // Each packed bit selects one BF16 element (two predicate bits).
                    DataCopy<uint8_t, MaskDist::DIST_US>(clipped,
                                                         maskBits + row * UbLayout::kMxScaleRowBytes + column / 8U);
                    auto valid = UpdateMask<uint16_t>(remaining);
                    And(clipped, clipped, valid, all);
                    DataCopy(dstBits + row * rowStride + column, value, clipped);
                }
            }
        }
    }

    __aicore__ inline void ApplyClipSentinels(AscendC::LocalTensor<bfloat16_t> dst, uint16_t rows, uint32_t columns,
                                              bool gatePlane, uint32_t maskRow)
    {
        constexpr uint16_t positiveInfinityBf16 = 0x7f80U;
        constexpr uint16_t negativeInfinityBf16 = 0xff80U;
        const uint32_t offset = maskRow * UbLayout::kMxScaleRowBytes;
        if (gatePlane) {
            SetClippedValues(dst, gateHighMask_[offset], rows, columns, columns, positiveInfinityBf16);
        } else {
            SetClippedValues(dst, upHighMask_[offset], rows, columns, columns, positiveInfinityBf16);
            AscendC::PipeBarrier<PIPE_V>();
            SetClippedValues(dst, upLowMask_[offset], rows, columns, columns, negativeInfinityBf16);
        }
    }

    __aicore__ inline void CopyPlaneTileToUb(const AscendC::GlobalTensor<bfloat16_t>& srcPlane, int64_t index,
                                             uint32_t singleM, uint32_t singleN, uint32_t planeWidth,
                                             AscendC::LocalTensor<bfloat16_t>& dstUb)
    {
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((planeWidth - singleN) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstUb, srcPlane[index], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    // Tail tiles reconstruct H in FP32, producing BF16 Hp and FP32 dp partials.
    // Gate is dead after its row is consumed, so its first FP32 slot stores the partial.
    __aicore__ inline void ComputeForwardAux(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t lanes = 256U / sizeof(float);
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        auto* dHp = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        auto* weights = (__ubuf__ TopkWeightT*)packedWeightUb_.GetPhyAddr();
        auto* partials = (__ubuf__ float*)l0cOutUbSecond_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                auto oneLane = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
                auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::RegTensor<float> accumulated, pFp, limitFp, negLimitFp;
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
                if constexpr (std::is_same_v<TopkWeightT, bfloat16_t>) {
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + row);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights + row);
                }
                if (limitEnabled) {
                    AscendC::Reg::Duplicate(limitFp, limit, all);
                    AscendC::Reg::Duplicate(negLimitFp, -limit, all);
                }
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = static_cast<uint32_t>(row) * rowStride + col;
                    AscendC::Reg::RegTensor<bfloat16_t> gateBf, upBf, dHpBf, hpBf;
                    AscendC::Reg::RegTensor<float> gateFp, upFp, dHpFp, neg, exp, denom, silu;
                    AscendC::Reg::RegTensor<float> hFp, hpFp, productFp, rowSum;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gateBf, gate + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(upBf, up + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dHpBf, dHp + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp, gateBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp, upBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHpFp, dHpBf, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Min(gateFp, gateFp, limitFp, mask);
                        AscendC::Reg::Max(upFp, upFp, negLimitFp, mask);
                        AscendC::Reg::Min(upFp, upFp, limitFp, mask);
                    }
                    AscendC::Reg::Muls(neg, gateFp, -1.0F, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, 1.0F, mask);
                    AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateFp, denom, mask);
                    AscendC::Reg::Mul(hFp, silu, upFp, mask);
                    if (!MegaMoeBF16Training::UseWholeRowTopkGrad(params_.savedGateUpMode, outputF_)) {
                        AscendC::Reg::Mul(productFp, dHpFp, hFp, mask);
                        AscendC::Reg::ReduceSum(rowSum, productFp, mask);
                        AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
                    }
                    AscendC::Reg::Mul(hpFp, hFp, pFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpBf, hpFp, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(up + offset, hpBf, mask);
                }
                AscendC::Reg::DataCopy(partials + static_cast<uint32_t>(row) * (rowStride / 2), accumulated, oneLane);
            }
        }
    }

    // Tail derivative: dH=p*dHp remains FP32 until dGate/dUp are cast.
    // With s=sigmoid(clamped gate), before applying clipping masks:
    //   dUp   = dh * SiLU_FP32(gate)
    //   dGate = dh * up * s * (1 + gate * (1 - s))
    // dGate overwrites the first UB dHp tile; dUp overwrites the Gate tile.
    // Gate/Up chunks are read before the same offsets are overwritten.
    // Forward clipping decisions zero the corresponding gradient lanes.
    __aicore__ inline void ComputeSwigluGrad(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            // Use saved forward bit planes for derivatives; decoded values may cross the clipping limit.
            ComputeSwigluGradImpl<true>(mSize, nSize, rowStride);
        } else {
            ComputeSwigluGradImpl<false>(mSize, nSize, rowStride);
        }
    }

    template <bool UseSavedClipMask>
    __aicore__ inline void ComputeSwigluGradImpl(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        uint16_t vectorRows = mSize;
        uint16_t vectorColumns = nSize;
        uint32_t vectorStride = rowStride;
        constexpr uint16_t kElementsPerRepeat = 128U;
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(vectorColumns), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* dHp = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        __ubuf__ bfloat16_t* dGate = dHp;
        __ubuf__ bfloat16_t* dUp = gate;  // reuse gate tile

        auto* weights = (__ubuf__ TopkWeightT*)packedWeightUb_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < vectorRows; ++row) {
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                AscendC::Reg::RegTensor<float> pFp;
                auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                if constexpr (std::is_same_v<TopkWeightT, bfloat16_t>) {
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + row);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights + row);
                }
                uint32_t remaining = vectorColumns;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> dH0, dH1, gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> dHFp0, dHFp1, gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * vectorStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(dH0, dHp + offset);
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(dH0, dH1, dH0, dH1);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp0, dH0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp1, dH1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = 2U;
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto dHFp = part == 0 ? dHFp0 : dHFp1;
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig, silu;
                        AscendC::Reg::RegTensor<float> gradSwish;
                        AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                        AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                        AscendC::Reg::RegTensor<float> gateC, upC, gateInd, upInd, diff;
                        AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
                        AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                        AscendC::Reg::Duplicate(oneFp, kOne, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Muls(limitReg, oneFp, limit, mask);
                            AscendC::Reg::Muls(negLimitReg, oneFp, -limit, mask);
                            AscendC::Reg::Min(gateC, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upC, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upC, upC, limitReg, mask);
                            if constexpr (UseSavedClipMask) {
                                // Preserve the forward mask; decoded values must not introduce new clipped lanes.
                                gateInd = oneFp;
                                upInd = oneFp;
                            } else {
                                AscendC::Reg::Sub(diff, gateFp, gateC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(gateInd, oneFp, diff, mask);
                                AscendC::Reg::Sub(diff, upFp, upC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(upInd, oneFp, diff, mask);
                            }
                        } else {
                            gateC = gateFp;
                            upC = upFp;
                        }
                        AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateC, denom, mask);
                        AscendC::Reg::Mul(dHFp, dHFp, pFp, mask);
                        AscendC::Reg::Mul(dUpFp, dHFp, silu, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                        }
                        // The derivative keeps dS=dh*up in FP32; no BF16 dS
                        // is materialized before the gate derivative.
                        AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                        AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                        AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                        AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                        AscendC::Reg::Adds(inner, inner, kOne, mask);
                        AscendC::Reg::Mul(term, sig, inner, mask);
                        AscendC::Reg::Mul(dGateFp, gradSwish, term, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dUp + offset + part * kPartSize, dUpOut, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dGate + offset + part * kPartSize, dGateOut, mask);
                    }
                }
            }
        }
        if constexpr (UseSavedClipMask) {
            AscendC::PipeBarrier<PIPE_V>();
            ApplySavedClipGradientMask(mSize, nSize, rowStride);
        }
    }

    template <bool UseSavedClipMask>
    __aicore__ inline void ComputeFusedFullTile()
    {
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        auto* dGate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* dUp = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        auto* hp = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        auto* weights = (__ubuf__ TopkWeightT*)packedWeightUb_.GetPhyAddr();
        auto* partials = (__ubuf__ float*)fusedDpUb_.GetPhyAddr();
        constexpr uint16_t rows = UbLayout::kTileRows, columns = UbLayout::kTileColumns;
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < rows; ++row) {
                auto mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                auto oneLane = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
                AscendC::Reg::RegTensor<float> accumulated, pFp;
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
                if constexpr (std::is_same_v<TopkWeightT, bfloat16_t>) {
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + row);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, mask);
                } else {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights + row);
                }
                for (uint16_t col = 0; col < columns; col += 64U) {
                    const uint32_t offset = static_cast<uint32_t>(row) * UbLayout::kTileColumns + col;
                    AscendC::Reg::RegTensor<bfloat16_t> gBf, uBf, dHpBf;
                    AscendC::Reg::RegTensor<float> gateFp, upFp, dHpFp, dHFp;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gBf, dUp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(uBf, hp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dHpBf, dGate + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp, gBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp, uBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHpFp, dHpBf, mask);
                    AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig;
                    AscendC::Reg::RegTensor<float> gradSwish;
                    AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                    AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                    AscendC::Reg::RegTensor<float> gateC, upC, gateInd, upInd, diff;
                    AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
                    AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                    AscendC::Reg::Duplicate(oneFp, kOne, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Muls(limitReg, oneFp, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneFp, -limit, mask);
                        AscendC::Reg::Min(gateC, gateFp, limitReg, mask);
                        AscendC::Reg::Max(upC, upFp, negLimitReg, mask);
                        AscendC::Reg::Min(upC, upC, limitReg, mask);
                        if constexpr (UseSavedClipMask) {
                            // Preserve the forward mask; decoded values must not introduce new clipped lanes.
                            gateInd = oneFp;
                            upInd = oneFp;
                        } else {
                            AscendC::Reg::Sub(diff, gateFp, gateC, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, oneFp, mask);
                            AscendC::Reg::Sub(gateInd, oneFp, diff, mask);
                            AscendC::Reg::Sub(diff, upFp, upC, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, oneFp, mask);
                            AscendC::Reg::Sub(upInd, oneFp, diff, mask);
                        }
                    } else {
                        gateC = gateFp;
                        upC = upFp;
                    }
                    AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, kOne, mask);
                    // h reconstruction uses gate/den, while the derivative
                    // retains (1/den)*gate. These are separate accurate Divs.
                    AscendC::Reg::RegTensor<float> auxSilu, hFp;
                    AscendC::Reg::RegTensor<float> hpFp, productFp, rowSum;
                    AscendC::Reg::RegTensor<bfloat16_t> hpBf;
                    AscendC::Reg::Div<float, &kBackwardDivMode>(auxSilu, gateC, denom, mask);
                    // Match forward's FP32 H for dp and Hp reconstruction.
                    AscendC::Reg::Mul(hFp, auxSilu, upC, mask);
                    if (!MegaMoeBF16Training::UseWholeRowTopkGrad(params_.savedGateUpMode, outputF_)) {
                        AscendC::Reg::Mul(productFp, dHpFp, hFp, mask);
                        AscendC::Reg::ReduceSum(rowSum, productFp, mask);
                        AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
                    }
                    AscendC::Reg::Mul(hpFp, hFp, pFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpBf, hpFp, mask);
                    AscendC::Reg::Mul(dHFp, dHpFp, pFp, mask);

                    AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                    // Reuse the FP32 SiLU used to reconstruct forward h.
                    AscendC::Reg::Mul(dUpFp, dHFp, auxSilu, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                    }
                    // Keep dS in FP32, matching the fused forward's tensor boundaries.
                    AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                    AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                    AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                    AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                    AscendC::Reg::Adds(inner, inner, kOne, mask);
                    AscendC::Reg::Mul(term, sig, inner, mask);
                    AscendC::Reg::Mul(dGateFp, gradSwish, term, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                    }
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);

                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dGate + offset, dGateOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dUp + offset, dUpOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(hp + offset, hpBf, mask);
                }
                // Reduce each row's four 64-element chunks in column order.
                AscendC::Reg::DataCopy(partials + row * 8U, accumulated, oneLane);
            }
        }
        if constexpr (UseSavedClipMask) {
            AscendC::PipeBarrier<PIPE_V>();
            ApplySavedClipGradientMask(UbLayout::kTileRows, UbLayout::kTileColumns, UbLayout::kTileColumns);
        }
    }

    __aicore__ inline void RunFusedFullTile(uint32_t rows, uint32_t columns, uint32_t mLoc, uint32_t nLoc)
    {
        static_assert(UbLayout::kMxScratchEnd + UbLayout::kTileRows * 32U <= kWeightOffset,
                      "Fused partials overlap neither masks nor weights");
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            ComputeFusedFullTile<true>();
        } else {
            ComputeFusedFullTile<false>();
        }
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        if (!MegaMoeBF16Training::UseWholeRowTopkGrad(params_.savedGateUpMode, outputF_)) {
            AscendC::DataCopyExtParams partialCopy{static_cast<uint16_t>(rows), sizeof(float), 0U,
                                                   static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0U};
            AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + nLoc / BACKWARD_DP_PARTIAL_TILE_N],
                                 fusedDpUb_, partialCopy);
        }
        AscendC::DataCopyExtParams hpCopy{static_cast<uint16_t>(rows),
                                          static_cast<uint32_t>(columns * sizeof(bfloat16_t)), 0U,
                                          static_cast<uint32_t>((outputF_ - columns) * sizeof(bfloat16_t)), 0U};
        AscendC::DataCopyPad(hpGm_[(rowBase_ + mLoc) * outputF_ + nLoc], extraA_, hpCopy);
        AscendC::DataCopyExtParams gradCopy{static_cast<uint16_t>(rows),
                                            static_cast<uint32_t>(columns * sizeof(bfloat16_t)), 0U,
                                            static_cast<uint32_t>((outputN_ - columns) * sizeof(bfloat16_t)), 0U};
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + mLoc) * outputN_ + nLoc], l0cOutUbFirst_, gradCopy);
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + mLoc) * outputN_ + outputF_ + nLoc], l0cOutUbSecond_, gradCopy);
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void ApplySavedClipGradientMask(uint16_t rows, uint32_t columns, uint32_t rowStride)
    {
        SetClippedValues(l0cOutUbFirst_, gateHighMask_, rows, columns, rowStride, 0U);
        SetClippedValues(l0cOutUbSecond_, upHighMask_, rows, columns, rowStride, 0U);
        AscendC::PipeBarrier<PIPE_V>();
        SetClippedValues(l0cOutUbSecond_, upLowMask_, rows, columns, rowStride, 0U);
    }

    // h = BF16(SiLU_FP32(gate)*up), stored in extraA_ at UB offset 2*UbLayout::kTileBytes.
    __aicore__ inline void ComputeSwigluRecompute(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* h = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<float> oneReg, limitReg, negLimitReg;
                    if (limitEnabled) {
                        AscendC::Reg::Duplicate(oneReg, kOne, mask);
                        AscendC::Reg::Muls(limitReg, oneReg, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneReg, -limit, mask);
                    }
                    AscendC::Reg::RegTensor<bfloat16_t> gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * rowStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> neg, exp, denom, silu, result;
                        AscendC::Reg::RegTensor<bfloat16_t> output;
                        // Reconstruct h with FP32 SiLU and Up, then cast to BF16.
                        // Apply the dsv4 swiglu_limit clamp before SiLU and Mul
                        // to preserve the forward activation boundary.
                        // Skipped entirely when the limit is disabled.
                        if (limitEnabled) {
                            AscendC::Reg::Min(gateFp, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upFp, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upFp, upFp, limitReg, mask);
                        }
                        AscendC::Reg::Muls(neg, gateFp, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateFp, denom, mask);
                        AscendC::Reg::Mul(result, silu, upFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(output, result, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            h + offset + part * kPartSize, output, mask);
                    }
                }
            }
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, UbLayout::kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, UbLayout::kTileBytes,
                                                     UbLayout::kTileElements};
    AscendC::LocalTensor<bfloat16_t> extraA_{AscendC::TPosition::VECIN, 2 * UbLayout::kTileBytes,
                                             UbLayout::kTileElements};
    AscendC::LocalTensor<TopkWeightT> packedWeightUb_{AscendC::TPosition::VECIN, kWeightOffset, UbLayout::kTileRows};
    AscendC::LocalTensor<float> fusedDpUb_{AscendC::TPosition::VECCALC, UbLayout::kMxScratchEnd,
                                           UbLayout::kTileRows * 8U};
    AscendC::LocalTensor<int8_t> mxData_{AscendC::TPosition::VECIN, UbLayout::kMxDataOffset, UbLayout::kMxDataElements};
    AscendC::LocalTensor<int8_t> mxScale_{AscendC::TPosition::VECIN, UbLayout::kMxScaleOffset, UbLayout::kMxScaleBytes};
    AscendC::LocalTensor<bfloat16_t> mxScaleBf16_{AscendC::TPosition::VECCALC, UbLayout::kMxScaleBf16Offset,
                                                  UbLayout::kMxScaleBf16Elements};
    AscendC::LocalTensor<float> mxScaleFp32_{AscendC::TPosition::VECCALC, UbLayout::kMxScaleFp32Offset,
                                             UbLayout::kMxScaleFp32Elements};
    AscendC::LocalTensor<int8_t> gateHighMask_{AscendC::TPosition::VECIN, UbLayout::kGateHighOffset,
                                               UbLayout::kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upLowMask_{AscendC::TPosition::VECIN, UbLayout::kUpLowOffset,
                                            UbLayout::kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upHighMask_{AscendC::TPosition::VECIN, UbLayout::kUpHighOffset,
                                             UbLayout::kClipMaskBytes};

    AscendC::GlobalTensor<bfloat16_t> gatePlaneGm_;
    AscendC::GlobalTensor<bfloat16_t> upPlaneGm_;
    AscendC::GlobalTensor<int8_t> gateMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> upMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> gateScaleGm_;
    AscendC::GlobalTensor<int8_t> upScaleGm_;
    AscendC::GlobalTensor<int8_t> gateHighGm_;
    AscendC::GlobalTensor<int8_t> upLowGm_;
    AscendC::GlobalTensor<int8_t> upHighGm_;
    AscendC::GlobalTensor<bfloat16_t> gradZGm_;
    AscendC::GlobalTensor<bfloat16_t> hpGm_;
    AscendC::GlobalTensor<float> dpPartialsGm_;
    AscendC::GlobalTensor<TopkWeightT> packedTopkWeightsGm_;
    uint64_t reductionRows_{0};
    int64_t rowBase_{0};
    Params params_{};
    int64_t outputN_{0};
    int64_t outputF_{0};
    int64_t dpPartialTiles_{0};
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP
