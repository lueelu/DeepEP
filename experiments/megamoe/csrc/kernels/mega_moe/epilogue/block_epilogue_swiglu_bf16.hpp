// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*
 * BF16 SwiGLU epilogue for the DeepEP MegaMoe pipeline. At entry, BF16
 * Gate/Up occupy two UB tiles, filled from GMM1's GM output through
 * LoadRawTile() after its completion flag becomes ready.
 * With R denoting BF16 rounding and Gate/Up clipped as configured,
 * H = SiLU_FP32(gate) * up; route weighting produces
 * Hp = R(H * p) before GMM2. Internal arithmetic stays FP32,
 * and dispatched BF16/FP32 route weights are prefetched from GM.
 */
#ifndef DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
#define DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP

#include "kernel_operator.h"
#include "epilogue/activation/swiglu_math.hpp"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "common/mega_moe_bf16_context.hpp"
#include "epilogue/block_saved_gate_up_mxfp8.hpp"
#include <type_traits>

namespace MegaMoeBF16Impl {

using MegaMoeImpl::Get;

static constexpr AscendC::Reg::CastTrait kCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// BF16 and unsaved activations need no MXFP8 encoding scratch.
struct NoSavedGateUpQuant {
    struct Layout {
        static constexpr uint32_t totalBytesWithClipMasks = 0;
    };
};

template <typename L0TileShape_, typename TopkWeightT = bfloat16_t, bool QuantizedSave = false>
class BlockEpilogueSwigluForwardBF16 {
public:
    __aicore__ inline void SetTimer(MegaMoeTrainTimerDevice* timer)
    {
        timer_ = timer;
    }
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR routedTopkWeightGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        GM_ADDR savedGateGmAddr{nullptr};
        GM_ADDR savedUpGmAddr{nullptr};
        GM_ADDR savedGateScaleGmAddr{nullptr};
        GM_ADDR savedUpScaleGmAddr{nullptr};
        GM_ADDR savedGateHighGmAddr{nullptr};
        GM_ADDR savedUpLowGmAddr{nullptr};
        GM_ADDR savedUpHighGmAddr{nullptr};
        bool applyTopkWeight{true};
    };

    using Params = Arguments;
    // Counts are read by Scalar while Gate/Up, encoded activations and p occupy UB.
    __aicore__ inline AscendC::LocalTensor<int32_t> SavedRowPrefix(uint32_t elements) const
    {
        constexpr auto maxElements = MegaMoeBF16Training::kMaxTotalExperts;
        constexpr uint32_t offset = kWeightOffset - maxElements * sizeof(int32_t);
        static_assert(kWeightOffset >= maxElements * sizeof(int32_t) + 2U * kTileElements * sizeof(bfloat16_t));
        static_assert(kWeightOffset >= maxElements * sizeof(int32_t) + QuantSave::Layout::totalBytesWithClipMasks);
        return AscendC::LocalTensor<int32_t>{AscendC::TPosition::VECCALC, offset, elements};
    }
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            params_ = params;
        }
    }

    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            outputGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.yGmAddr + Get<0>(baseOffset));

            groupFlagListGmAddr_ = (__gm__ int32_t*)params_.groupFlagListGmAddr + Get<5>(baseOffset);
            if (params_.applyTopkWeight) {
                routedTopkWeightGlobal_.SetGlobalBuffer((__gm__ TopkWeightT*)params_.routedTopkWeightGmAddr +
                                                        Get<3>(baseOffset));
                routedTopkWeightGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
    }

    __aicore__ inline void Run(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }

        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }

        const uint32_t tileRow = static_cast<uint32_t>(Get<0>(blockCoord) / outputN_);
        ComputeSwigluDirect(singleM, singleN, tileRow);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                              static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(outputGlobal_[Get<0>(blockCoord)], l0cOutUbFirst_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::AtomicAdd(groupFlagListGmAddr_ + tileRow / params_.baseM, 1);
    }

    __aicore__ inline void LoadRawTile(GM_ADDR gate, GM_ADDR up, uint64_t expertOffset, uint16_t mSize, uint16_t nSize)
    {
        AscendC::GlobalTensor<bfloat16_t> rawGate, rawUp;
        rawGate.SetGlobalBuffer((__gm__ bfloat16_t*)gate + expertOffset);
        rawUp.SetGlobalBuffer((__gm__ bfloat16_t*)up + expertOffset);
        rawGate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        rawUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams copyParams{mSize, static_cast<uint32_t>(nSize * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((kTileColumns - nSize) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPad(l0cOutUbFirst_, rawGate, copyParams, {});
        AscendC::DataCopyPad(l0cOutUbSecond_, rawUp, copyParams, {});
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    template <uint32_t Mode>
    __aicore__ inline void SaveRows(uint32_t row, uint16_t rows, uint16_t cols, uint64_t dst)
    {
        const uint32_t stride = Ops::Base::CeilAlign(uint32_t(cols), uint32_t(16));
        auto gate = l0cOutUbFirst_[row * stride];
        auto up = l0cOutUbSecond_[row * stride];
        if constexpr (Mode == MegaMoeBF16Training::kSaveMxFp8) {
            quantSave_.Save(gate, up, params_.savedGateGmAddr, params_.savedUpGmAddr, params_.savedGateScaleGmAddr,
                            params_.savedUpScaleGmAddr, params_.savedGateHighGmAddr, params_.savedUpLowGmAddr,
                            params_.savedUpHighGmAddr, params_.swigluLimit, rows, cols, outputN_, dst);
        } else {
            AscendC::GlobalTensor<bfloat16_t> savedGate, savedUp;
            savedGate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params_.savedGateGmAddr) + dst);
            savedUp.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params_.savedUpGmAddr) + dst);
            const AscendC::DataCopyExtParams cp{rows, uint32_t(cols * sizeof(bfloat16_t)), 0,
                                                uint32_t((outputN_ - cols) * sizeof(bfloat16_t)), 0};
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopyPad(savedGate, gate, cp);
            AscendC::DataCopyPad(savedUp, up, cp);
        }
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

private:
    MegaMoeTrainTimerDevice* timer_ = nullptr;
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    using QuantSave = std::conditional_t<QuantizedSave, SavedGateUpMxFp8<kTileRows, kTileColumns>, NoSavedGateUpQuant>;
    static constexpr uint32_t kWeightBytes = (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32;
    static constexpr uint32_t kWeightOffset = Catlass::Arch::Ascend950::UB_SIZE - kWeightBytes;
    static_assert(2U * kTileElements * sizeof(bfloat16_t) <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The BF16 SwiGLU input tiles must fit in A5 UB");

    __aicore__ inline void ComputeSwigluDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t lanes = 256 / sizeof(float);
        const uint32_t stride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
        const bool clip = params_.swigluLimit != kMegaMoeSwigluLimitDisabled;
        const float limit = params_.swigluLimit;
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        // Place weights beyond every MXFP8 data/scale/clamp-mask buffer: Save
        // may still have MTE3 readers. No alias and no early pipeline drain.
        static_assert(kWeightOffset >= 2U * kTileElements * sizeof(bfloat16_t));
        static_assert(kWeightOffset >= QuantSave::Layout::totalBytesWithClipMasks);
        AscendC::LocalTensor<TopkWeightT> weightUb{AscendC::TPosition::VECIN, kWeightOffset, kTileRows};
        if (params_.applyTopkWeight) {
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams p{1, uint32_t(mSize * sizeof(TopkWeightT)), 0, 0, 0};
            AscendC::DataCopyPad(weightUb, routedTopkWeightGlobal_[tileRow], p, {});
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        auto* weights = (__ubuf__ TopkWeightT*)weightUb.GetPhyAddr();
        // Direct B16 unpack follows the reference implementation. Arithmetic,
        // accurate division and clipping remain in FP32 until the output cast.
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> lim, negLim;
            if (clip) {
                AscendC::Reg::Duplicate(lim, limit);
                AscendC::Reg::Duplicate(negLim, -limit);
            }
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> weightFp;
                if (params_.applyTopkWeight) {
                    if constexpr (std::is_same_v<TopkWeightT, float>) {
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weightFp, weights + row);
                    } else {
                        AscendC::Reg::RegTensor<bfloat16_t> weightBf;
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(weightBf,
                                                                                                 weights + row);
                        auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(weightFp, weightBf, all);
                    }
                }
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> g, u, output;
                    AscendC::Reg::RegTensor<float> gf, uf, neg, exp, denom, silu, value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        g, gate + row * stride + col);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        u, up + row * stride + col);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gf, g, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(uf, u, mask);
                    if (clip) {
                        AscendC::Reg::Min(gf, gf, lim, mask);
                        AscendC::Reg::Max(uf, uf, negLim, mask);
                        AscendC::Reg::Min(uf, uf, lim, mask);
                    }
                    MegaMoeImpl::Activation::SwiGluFp32<true>(value, gf, uf, neg, exp, denom, silu, mask);
                    if (params_.applyTopkWeight) AscendC::Reg::Mul(value, value, weightFp, mask);
                    // Keep H in FP32 through route weighting; only the GEMM input Hp is BF16.
                    AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, value, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        gate + row * stride + col, output, mask);
                }
            }
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileElements * sizeof(bfloat16_t),
                                                     kTileElements};
    AscendC::GlobalTensor<bfloat16_t> outputGlobal_;
    AscendC::GlobalTensor<TopkWeightT> routedTopkWeightGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_{nullptr};
    Params params_{};
    int64_t outputN_{0};
    QuantSave quantSave_;
    static_assert(QuantSave::Layout::totalBytesWithClipMasks <= Catlass::Arch::Ascend950::UB_SIZE,
                  "Saved gate/up MXFP8 scratch must fit AIV0 UB");
    static_assert(kTileRows == 256 && kTileColumns == 128, "SwiGLU consumes M256 Gate128/Up128 pairs");
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
