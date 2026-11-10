// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/compute.hpp"

namespace MegaMoeUrma {

// FIX writes an ND M256 x [gate128, up128] BF16 matrix to the paired AIV1's
// low128KiB mailbox. The interleaved epilogue compacts its BF16 result in place
// to M256x128, then reuses dead input space for FP8 data and E8M0 scales.
// The caller must have consumed real FIX full and must NOT return UB empty
// until Run() returns (its final activation/scale MTE3 stores are complete).
class UbRoutedActivation {
    using Op = UbxMegaMoeImpl::BlockEpilogueActivationMxQuant<fp8_e4m3fn_t, bfloat16_t, 256U, 256U, false, true, true>;
    static constexpr auto offsets =
        UbxMegaMoeImpl::Detail::BuildActivationMxQuantUbOffsets<bfloat16_t, 256U, 256U, true>();
    static_assert(offsets.firstInputElementCapacity * sizeof(bfloat16_t) == 128U * 1024U);
    static_assert(offsets.reciprocalScaleOffsetBytes + offsets.scaleElementCapacity * sizeof(uint16_t) <= 128U * 1024U);

public:
    static constexpr uint32_t kColumns = 128U;
    __aicore__ inline void Init(const Arguments& a)
    {
        op_.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(a.activation),
                  .yScaleGmAddr = reinterpret_cast<GM_ADDR>(a.activationScale),
                  .clampLimit = FLT_MAX,
                  .ubOffsetBytes = 0U});
        op_.UpdateNextProblem({0, a.config.intermediate, 0, 0});
    }

    template <class TimerT>
    __aicore__ inline void Run(const Arguments& a, const Fragment& f, uint32_t group, uint32_t tile, TimerT& timer)
    {
        const uint32_t n = tile * kColumns;
        // I is256-aligned, so fused N128 task columns have no partial tail.
        // Only the M tail changes; each UB input row remains256 BF16 elements.
        op_.UpdateGlobalAddr({static_cast<int64_t>(f.rowBegin * a.config.intermediate),
                              static_cast<int64_t>(f.rowBegin * (a.config.intermediate / 32U)), 0, 0, 0, 0});
        auto meta = ComputeTileTraceMeta(FragmentTraceMeta(f, group), 0U, n, f.rows, kColumns);
        timer.Begin(Phase::Activation);
        const auto start = timer.TraceBegin();
        AscendC::SetCtrlSpr<60, 60>(0);
        op_({f.rows, kColumns, 0, 0}, {n, n / 32U, 0, 0, 0, 0}, 0U);
        // operator() retains the existing SeparateSigmoid FP32 sequence,
        // BF16 rounding and MX groups. It drains both output DMA planes.
        AscendC::AtomicAdd(ComputeFlag(a.activationDone, group), 1);
        timer.TraceEnd(TraceEvent::ActivationSlice, start, meta);
        timer.End(Phase::Activation);
    }

private:
    Op op_;
};
}  // namespace MegaMoeUrma
