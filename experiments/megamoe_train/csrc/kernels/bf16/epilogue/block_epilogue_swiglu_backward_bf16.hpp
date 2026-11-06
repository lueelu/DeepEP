// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*
 * SwiGLU backward epilogues for MegaMoe BF16.
 *
 * H=SiLU_FP32(gate)*up and dH=p*dHp stay in FP32 registers.
 * Hp is rounded to BF16 for GMM2/dW2, while activation derivatives
 * stay in FP32 until dGate/dUp are written as BF16 dZ.
 *
 * SwiGLUGrad enters with dHp in the first UB tile: saved GM callers load
 * it with LoadDHpFromGM; save=none callers wait for the AIC Fixpipe tile.
 * Gate/Up and p produce Hp, FP32 row partials of dHp*h, and dZ.
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
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "layout/mega_moe_context.hpp"
#include "layout/mega_moe_layout.hpp"

namespace MegaMoeBF16Impl {

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
        dpPartialTiles_ =
            Ops::Base::CeilDiv(static_cast<uint64_t>(outputF_), static_cast<uint64_t>(BACKWARD_DP_PARTIAL_TILE_N));
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

        // In save=none, the caller's PIPE_V WaitForCube protects Fixpipe dHp;
        // saved modes already loaded dHp with LoadDHpFromGM. Bridge V to MTE2
        // before reusing the other two work tiles for Gate/Up: none reads
        // recompute GM, while saved modes read forward-persistent GM.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        // Load Gate/Up while dHp remains in the first UB work tile.
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // Full 128x256 tiles consume this Gate/Up load without a second GM read.
        if (singleM == kTileRows && singleN == kTileColumns && rowStrideInPlane == singleN) {
            RunFusedFullTile(singleM, singleN, mLoc, nLoc);
            return;
        }
        ComputeForwardAux(singleM, singleN, rowStrideInPlane);

        // tile1 holds FP32 partials and tile2 holds Hp until MTE3 consumes both.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint32_t dpTileIdx = nLoc / BACKWARD_DP_PARTIAL_TILE_N;
        AscendC::DataCopyExtParams partialCopyParams{
            static_cast<uint16_t>(singleM), static_cast<uint32_t>(sizeof(float)),
            // srcStride counts 32-byte blocks; the 4-byte partial occupies one padded block.
            static_cast<uint32_t>(rowStrideInPlane * sizeof(bfloat16_t) / AscendC::ONE_BLK_SIZE - 1U),
            static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0};
        AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + dpTileIdx],
                             l0cOutUbSecond_.template ReinterpretCast<float>(), partialCopyParams);
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
        // Retire dZ MTE3 writes before returning. The save=none caller then
        // issues PIPE_V NotifyCube; saved callers publish GM completion.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

private:
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static constexpr uint32_t kTileBytes = kTileElements * sizeof(bfloat16_t);
    static constexpr uint32_t kDecodeRows = 16;
    static constexpr uint32_t kMxDataElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxScaleRowBytes = 32;
    static constexpr uint32_t kMxScaleBytes = kDecodeRows * kMxScaleRowBytes;
    // DIST_INTLV stores ignore the mask and write 512 bytes on A5.
    // Reserve their full footprint so scale stores cannot corrupt mxDecoded_.
    static constexpr uint32_t kMxDualRegisterStoreBytes = 2U * 256U;
    static constexpr uint32_t kMxScaleBf16Elements = kMxDualRegisterStoreBytes / sizeof(bfloat16_t);
    static constexpr uint32_t kMxScaleBf16Bytes = kMxScaleBf16Elements * sizeof(bfloat16_t);
    static constexpr uint32_t kMxScaleFp32Elements = kMxDualRegisterStoreBytes / sizeof(float);
    static constexpr uint32_t kMxScaleFp32Bytes = kMxScaleFp32Elements * sizeof(float);
    static constexpr uint32_t kMxDecodedElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxDecodedBytes = kMxDecodedElements * sizeof(float);
    static constexpr uint32_t kMxDataOffset = 3U * kTileBytes;
    static constexpr uint32_t kMxScaleOffset = kMxDataOffset + kMxDataElements;
    static constexpr uint32_t kMxScaleBf16Offset = kMxScaleOffset + kMxScaleBytes;
    static constexpr uint32_t kMxScaleFp32Offset = kMxScaleBf16Offset + kMxScaleBf16Bytes;
    static constexpr uint32_t kMxDecodedOffset = kMxScaleFp32Offset + kMxScaleFp32Bytes;
    static constexpr uint32_t kClipMaskBytes = kTileRows * kMxScaleRowBytes;
    static constexpr uint32_t kGateHighOffset = kMxDecodedOffset + kMxDecodedBytes;
    static constexpr uint32_t kUpLowOffset = kGateHighOffset + kClipMaskBytes;
    static constexpr uint32_t kUpHighOffset = kUpLowOffset + kClipMaskBytes;
    static constexpr uint32_t kMxScratchEnd = kUpHighOffset + kClipMaskBytes;
    static_assert(kMxScratchEnd <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The backward SwiGLU work tiles and MXFP8 decoder must fit in A5 UB");
    static constexpr uint32_t kWeightBytes = (kTileRows * sizeof(TopkWeightT) + 31U) / 32U * 32U;
    static constexpr uint32_t kWeightOffset = Catlass::Arch::Ascend950::UB_SIZE - kWeightBytes;
    static_assert(kWeightOffset >= kMxScratchEnd, "Prefetched weights must not alias live MXFP8 decode or clip masks");

    __aicore__ inline void LoadPackedWeights(uint32_t rows, uint32_t mLoc)
    {
        // This AIV has finished its P0 staging before loading weights. In
        // save=none, paired AIC Fixpipe targets only the first work tile.
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

    __aicore__ inline void DecodeMxFp8Plane(const AscendC::GlobalTensor<int8_t>& dataGm,
                                            const AscendC::GlobalTensor<int8_t>& scaleGm,
                                            const AscendC::GlobalTensor<int8_t>& lowMaskGm,
                                            const AscendC::GlobalTensor<int8_t>& highMaskGm, uint32_t rowStart,
                                            uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                            AscendC::LocalTensor<bfloat16_t>& dst, bool gatePlane)
    {
        const uint32_t scaleWidth = outputF_ / 32;
        const uint32_t groups = singleN / 32;
        const bool hasLowMask = params_.gateHighGmAddr != nullptr;
        for (uint32_t chunkRow = 0; chunkRow < singleM; chunkRow += kDecodeRows) {
            const uint16_t rows =
                static_cast<uint16_t>(singleM - chunkRow > kDecodeRows ? kDecodeRows : singleM - chunkRow);
            const uint32_t globalRow = rowStart + chunkRow;
            const int64_t dataIndex = static_cast<int64_t>(globalRow) * outputF_ + nLoc;
            const int64_t scaleIndex = static_cast<int64_t>(globalRow) * scaleWidth + nLoc / 32;
            // Retire this chunk's V consumers before MTE2 reuses scratch.
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(dataGm, dataIndex, rows, singleN, outputF_, mxData_);
            CopyMxRowsToUb(scaleGm, scaleIndex, rows, groups, scaleWidth, mxScale_);
            if (hasLowMask) {
                const uint32_t maskWidthBytes = scaleWidth * sizeof(uint32_t);
                const int64_t maskIndex =
                    static_cast<int64_t>(globalRow) * maskWidthBytes + (nLoc / 32) * sizeof(uint32_t);
                if (gatePlane) {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   gateHighMask_[chunkRow * kMxScaleRowBytes]);
                } else {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upLowMask_[chunkRow * kMxScaleRowBytes]);
                }
                if (!gatePlane) {
                    CopyMxRowsToUb(highMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upHighMask_[chunkRow * kMxScaleRowBytes]);
                }
                AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
            }
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
            DecodeMxFp8Rows(rows, singleN, groups);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst[chunkRow * singleN], mxDecoded_, AscendC::RoundMode::CAST_RINT,
                          static_cast<uint32_t>(rows) * singleN);
            if (hasLowMask) {
                // Restore forward's clipping decision, not a decision recomputed from lossy MXFP8.
                // An out-of-range sentinel makes the FP32 clamp restore the exact finite limit.
                AscendC::SyncFunc<AscendC::HardEvent::V_S>();
                ApplyClipSentinels(dst[chunkRow * singleN], rows, singleN, groups, gatePlane, chunkRow);
                AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                AscendC::SyncFunc<AscendC::HardEvent::S_V>();
            }
        }
    }

    __aicore__ inline void DecodeMxFp8Rows(uint16_t rows, uint32_t columns, uint32_t groups)
    {
        constexpr uint32_t bf16PerReg = 256U / sizeof(bfloat16_t);
        constexpr uint32_t fp32PerReg = 256U / sizeof(float);
        auto data = (__ubuf__ fp8_e4m3fn_t*)mxData_.GetPhyAddr();
        auto scales = (__ubuf__ fp8_e8m0_t*)mxScale_.GetPhyAddr();
        auto scaleBf16 = (__ubuf__ bfloat16_t*)mxScaleBf16_.GetPhyAddr();
        auto scaleFp32 = (__ubuf__ float*)mxScaleFp32_.GetPhyAddr();
        auto decoded = (__ubuf__ float*)mxDecoded_.GetPhyAddr();
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
                        src, scales + row * kMxScaleRowBytes + i * bf16PerReg);
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
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_E2B_B32>(scale, scaleFp32 + i * 8);
                    AscendC::Reg::DataCopy<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                        src, data + row * columns + i * fp32PerReg);
                    AscendC::Reg::Cast<float, fp8_e4m3fn_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::Mul(result, scale, value, mask);
                    AscendC::Reg::DataCopy(decoded + row * columns + i * fp32PerReg, result, mask);
                }
            }
        }
    }

    __aicore__ inline void ApplyClipSentinels(AscendC::LocalTensor<bfloat16_t> dst, uint16_t rows, uint32_t columns,
                                              uint32_t groups, bool gatePlane, uint32_t maskRow)
    {
        auto dstBits = dst.template ReinterpretCast<uint16_t>();
        auto gateHighWords = gateHighMask_.template ReinterpretCast<uint32_t>();
        auto upLowWords = upLowMask_.template ReinterpretCast<uint32_t>();
        auto highWords = upHighMask_.template ReinterpretCast<uint32_t>();
        // Infinity is an internal sentinel beyond every finite limit; the following clamp removes it.
        constexpr uint16_t positiveInfinityBf16 = 0x7f80U;
        constexpr uint16_t negativeInfinityBf16 = 0xff80U;
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t low = gatePlane ? gateHighWords.GetValue((maskRow + row) * 8 + group)
                                               : upLowWords.GetValue((maskRow + row) * 8 + group);
                const uint32_t high = gatePlane ? 0U : highWords.GetValue((maskRow + row) * 8 + group);
                if ((low | high) == 0U) {
                    continue;
                }
                for (uint32_t bit = 0; bit < 32; ++bit) {
                    const uint32_t index = row * columns + group * 32 + bit;
                    if ((low & (1U << bit)) != 0U) {
                        dstBits.SetValue(index, gatePlane ? positiveInfinityBf16 : negativeInfinityBf16);
                    } else if ((high & (1U << bit)) != 0U) {
                        dstBits.SetValue(index, positiveInfinityBf16);
                    }
                }
            }
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
                    AscendC::Reg::Mul(productFp, dHpFp, hFp, mask);
                    AscendC::Reg::ReduceSum(rowSum, productFp, mask);
                    AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
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
            // Retire V stores before scalar mask clears; order those clears before MTE2/MTE3.
            AscendC::SyncFunc<AscendC::HardEvent::V_S>();
            ApplySavedClipGradientMask(mSize, nSize, rowStride);
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
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
        constexpr uint16_t rows = kTileRows, columns = kTileColumns;
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
                    const uint32_t offset = static_cast<uint32_t>(row) * kTileColumns + col;
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
                    AscendC::Reg::Mul(productFp, dHpFp, hFp, mask);
                    AscendC::Reg::ReduceSum(rowSum, productFp, mask);
                    AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
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
            AscendC::SyncFunc<AscendC::HardEvent::V_S>();
            ApplySavedClipGradientMask(kTileRows, kTileColumns, kTileColumns);
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
    }

    __aicore__ inline void RunFusedFullTile(uint32_t rows, uint32_t columns, uint32_t mLoc, uint32_t nLoc)
    {
        static_assert(kMxScratchEnd + kTileRows * 32U <= kWeightOffset,
                      "Fused partials overlap neither masks nor weights");
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            ComputeFusedFullTile<true>();
        } else {
            ComputeFusedFullTile<false>();
        }
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams partialCopy{static_cast<uint16_t>(rows), sizeof(float), 0U,
                                               static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0U};
        AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + nLoc / BACKWARD_DP_PARTIAL_TILE_N],
                             fusedDpUb_, partialCopy);
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
        auto dGateBits = l0cOutUbFirst_.template ReinterpretCast<uint16_t>();
        auto dUpBits = l0cOutUbSecond_.template ReinterpretCast<uint16_t>();
        auto gateHighWords = gateHighMask_.template ReinterpretCast<uint32_t>();
        auto upLowWords = upLowMask_.template ReinterpretCast<uint32_t>();
        auto upHighWords = upHighMask_.template ReinterpretCast<uint32_t>();
        const uint32_t groups = columns / 32;
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t gateHigh = gateHighWords.GetValue(row * 8 + group);
                const uint32_t upClipped = upLowWords.GetValue(row * 8 + group) | upHighWords.GetValue(row * 8 + group);
                if ((gateHigh | upClipped) == 0U) {
                    continue;
                }
                for (uint32_t bit = 0; bit < 32; ++bit) {
                    const uint32_t index = row * rowStride + group * 32 + bit;
                    if ((gateHigh & (1U << bit)) != 0U) {
                        dGateBits.SetValue(index, 0U);
                    }
                    if ((upClipped & (1U << bit)) != 0U) {
                        dUpBits.SetValue(index, 0U);
                    }
                }
            }
        }
    }

    // h = BF16(SiLU_FP32(gate)*up), stored in extraA_ at UB offset 2*kTileBytes.
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

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileBytes, kTileElements};
    AscendC::LocalTensor<bfloat16_t> extraA_{AscendC::TPosition::VECIN, 2 * kTileBytes, kTileElements};
    AscendC::LocalTensor<TopkWeightT> packedWeightUb_{AscendC::TPosition::VECIN, kWeightOffset, kTileRows};
    AscendC::LocalTensor<float> fusedDpUb_{AscendC::TPosition::VECCALC, kMxScratchEnd, kTileRows * 8U};
    AscendC::LocalTensor<int8_t> mxData_{AscendC::TPosition::VECIN, kMxDataOffset, kMxDataElements};
    AscendC::LocalTensor<int8_t> mxScale_{AscendC::TPosition::VECIN, kMxScaleOffset, kMxScaleBytes};
    AscendC::LocalTensor<bfloat16_t> mxScaleBf16_{AscendC::TPosition::VECCALC, kMxScaleBf16Offset,
                                                  kMxScaleBf16Elements};
    AscendC::LocalTensor<float> mxScaleFp32_{AscendC::TPosition::VECCALC, kMxScaleFp32Offset, kMxScaleFp32Elements};
    AscendC::LocalTensor<float> mxDecoded_{AscendC::TPosition::VECCALC, kMxDecodedOffset, kMxDecodedElements};
    AscendC::LocalTensor<int8_t> gateHighMask_{AscendC::TPosition::VECIN, kGateHighOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upLowMask_{AscendC::TPosition::VECIN, kUpLowOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upHighMask_{AscendC::TPosition::VECIN, kUpHighOffset, kClipMaskBytes};

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
    int64_t rowBase_{0};
    Params params_{};
    int64_t outputN_{0};
    int64_t outputF_{0};
    int64_t dpPartialTiles_{0};
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP
