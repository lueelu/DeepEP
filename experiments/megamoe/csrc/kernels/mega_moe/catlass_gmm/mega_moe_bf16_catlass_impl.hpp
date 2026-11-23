// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

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

#include "epilogue/block_epilogue_swiglu_bf16.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "common/mega_moe_bf16_context.hpp"
#include "stage/mega_moe_combine_send.hpp"
#include "common/mega_moe_bf16_wave_plan.hpp"
#include "csrc/kernels/common/mega_moe_wave_math.hpp"
#include "common/mega_moe_bf16_wave_completion.hpp"

namespace MegaMoeBF16ForwardCatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
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

using WaveBlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using WaveL1TileShape = tla::Shape<tla::Int<MegaMoeBF16Training::Wave::kGmm1Rows>, tla::Int<256>, tla::Int<256>>;
using WaveL0TileShape = tla::Shape<tla::Int<MegaMoeBF16Training::Wave::kGmm1Rows>, tla::Int<256>, tla::Int<64>>;
using WaveEpilogueTileShape = tla::Shape<tla::Int<MegaMoeBF16Training::Wave::kGmm1Rows>, tla::Int<128>, tla::Int<256>>;

template <class Tensor>
struct WaveGateUpWeight : Tensor {
    __aicore__ inline explicit WaveGateUpWeight(const Tensor& tensor) : Tensor(tensor) {}
};

template <class Tensor>
struct IsWaveGateUpWeight : std::false_type {};
template <class Tensor>
struct IsWaveGateUpWeight<WaveGateUpWeight<Tensor>> : std::true_type {};

template <class TensorB, class WeightLayout>
struct WaveWeightCopy {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(const TensorDst& dst, const TensorSrc& src)
    {
        using Copy = typename ForwardGmTileCopy<WeightLayout>::template CopyGmToL1B<TensorSrc>;
        Copy copy;
        if constexpr (IsWaveGateUpWeight<TensorB>::value) {
            using namespace MegaMoeBF16Training::Wave;
            const uint32_t k = tla::get<0>(src.coord());
            const uint32_t n = tla::get<1>(src.coord()) / 2U;
            const uint32_t f = [&]() __aicore__ {
                if constexpr (std::is_same_v<WeightLayout, Catlass::layout::ColumnMajor>) {
                    return static_cast<uint32_t>(src.data().GetSize() / (2ULL * tla::get<1>(src.stride())));
                } else {
                    return static_cast<uint32_t>(tla::get<0>(src.stride()) / 2U);
                }
            }();
            const auto halfShape = tla::MakeShape(tla::get<0>(src.originShape()), kPairedColumns);
            const auto rawLayout = tla::MakeLayout(halfShape, src.stride());
            auto gate = tla::MakeTensor(src.data(), rawLayout, tla::MakeCoord(k, n), Catlass::Arch::PositionGM{});
            auto up = tla::MakeTensor(src.data(), rawLayout, tla::MakeCoord(k, f + n), Catlass::Arch::PositionGM{});
            auto left = tla::GetTile(dst, tla::MakeCoord(0U, 0U), halfShape);
            auto right = tla::GetTile(dst, tla::MakeCoord(0U, kPairedColumns), halfShape);
            // Both loads precede BlockMmad's MTE2_MTE1 event, for the first K window and prefetches alike.
            copy(left, gate);
            copy(right, up);
        } else {
            copy(dst, src);
        }
    }
};

template <class Tensor>
struct WaveGateUpOutput : Tensor {
    using BaseTensor = Tensor;
    Tensor up;
    __aicore__ inline WaveGateUpOutput(const Tensor& gate, const Tensor& upTensor) : Tensor(gate), up(upTensor) {}
};

template <class Tensor>
struct IsWaveGateUpOutput : std::false_type {};
template <class Tensor>
struct IsWaveGateUpOutput<WaveGateUpOutput<Tensor>> : std::true_type {};

template <class TensorDst>
struct WaveOutputCopy {
    template <class TensorSrc>
    __aicore__ inline void operator()(const TensorDst& dst, const TensorSrc& src, uint8_t unitFlag = 0)
    {
        if constexpr (IsWaveGateUpOutput<TensorDst>::value) {
            using BaseTensor = typename TensorDst::BaseTensor;
            const auto halfShape = tla::MakeShape(tla::get<0>(dst.originShape()), tla::get<1>(dst.originShape()));
            auto gate = tla::GetTile(src, tla::MakeCoord(0U, 0U), halfShape);
            auto up = tla::GetTile(src, tla::MakeCoord(0U, MegaMoeBF16Training::Wave::kPairedColumns), halfShape);
            Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(gate), BaseTensor,
                                                Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>
                copy;
            // Each N128 half owns separate L0C storage units and retains its completion handshake.
            copy(static_cast<const BaseTensor&>(dst), gate, unitFlag);
            copy(dst.up, up, unitFlag);
        } else {
            Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorSrc, TensorDst,
                                                Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>
                copy;
            copy(dst, src, unitFlag);
        }
    }
};

template <class WeightLayout>
struct WaveGmTileCopy : ForwardGmTileCopy<WeightLayout> {
    template <class TensorB>
    using CopyGmToL1B = WaveWeightCopy<TensorB, WeightLayout>;
    template <class TensorC>
    using CopyL0CToDst = WaveOutputCopy<TensorC>;
};

// Both projections write BF16 to GM. One persistent MMAD retains its L1/L0
// buffers and event sequence when switching between W1 and W2.
template <bool TransW>
class WaveMatmul {
public:
    static constexpr uint32_t kRows = MegaMoeBF16Training::Wave::kGmm1Rows;
    static constexpr uint32_t kDownRows = MegaMoeBF16Training::Wave::kGmm2Rows;
    using WeightLayout = std::conditional_t<TransW, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, WaveL1TileShape, WaveL0TileShape, bfloat16_t,
                                                         bfloat16_t, bfloat16_t, void, WaveGmTileCopy<WeightLayout>>;
    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;

    __aicore__ inline WaveMatmul() : blockMmad(resource) {}

    __aicore__ inline void GateUp(const Bf16Params& params, const MegaMoeBF16Training::Wave::Slice& slice, GM_ADDR x,
                                  __gm__ int32_t* dispatchReady, uint32_t& startBlock, uint32_t& sequence)
    {
        using namespace MegaMoeBF16Training::Wave;
        const uint32_t h = params.tilingData->k, f = params.tilingData->n / 2U;
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        AscendC::GlobalTensor<bfloat16_t> a, b, gate, up;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        b.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(params.bGmAddr) + static_cast<uint64_t>(slice.expert) * 2U * f * h,
            2ULL * f * h);
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, h),
                                  Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(h, 2U * f), Catlass::Arch::PositionGM{});
        bool ready = false;
        ForEachGateUpTile<WaveBlockScheduler>(
            slice.rows, f, h, core, cores, startBlock, [&](uint32_t m, uint32_t n, const auto& shape) __aicore__ {
                if (!ready) {
                    MegaMoeTrainTimerPassiveScope wait(params.timer, MegaMoeTrainTimer::AIC_GMM1_WAIT, slice.expert,
                                                       slice.row + m, shape.m(), n, kRows, kPairedColumns, h, f);
                    while (AscendC::ReadGmByPassDCache(dispatchReady) < cores) {
                    }
                    AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                    ready = true;
                }
                MegaMoeTrainTimerPassiveScope tile(params.timer, MegaMoeTrainTimer::AIC_GMM1, slice.expert,
                                                   slice.row + m, shape.m(), n, kRows, kPairedColumns, h, f);
                GateUpCompletion::Acquire(sequence);
                auto tileA = tla::GetTile(ta, tla::MakeCoord(m, 0U), tla::MakeShape(shape.m(), shape.k()));
                auto tileB = tla::GetTile(tb, tla::MakeCoord(0U, 2U * n), tla::MakeShape(shape.k(), 2U * shape.n()));
                WaveGateUpWeight<decltype(tileB)> pairedWeight(tileB);
                static_assert(kRawStages == GateUpCompletion::kStages);
                const uint64_t slot = RawTileOffset(core, sequence);
                gate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.forwardRawGate) + slot);
                up.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.forwardRawUp) + slot);
                gate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                up.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                auto layout = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(kRows, kPairedColumns);
                auto tg = tla::MakeTensor(gate, layout, Catlass::Arch::PositionGM{});
                auto tu = tla::MakeTensor(up, layout, Catlass::Arch::PositionGM{});
                auto tileGate = tla::GetTile(tg, tla::MakeCoord(0U, 0U), tla::MakeShape(shape.m(), shape.n()));
                auto tileUp = tla::GetTile(tu, tla::MakeCoord(0U, 0U), tla::MakeShape(shape.m(), shape.n()));
                WaveGateUpOutput<decltype(tileGate)> output(tileGate, tileUp);
                blockMmad(tileA, pairedWeight, output, Catlass::GemmCoord{shape.m(), 2U * shape.n(), shape.k()});
                GateUpCompletion::Publish(sequence++);
            });
    }

    __aicore__ inline bool Down(const Bf16Params& params, const MegaMoeBF16Training::Wave::Slice& slice, GM_ADDR output,
                                uint64_t scratchRow, __gm__ int32_t* hpFlags, uint32_t& startBlock)
    {
        const uint32_t h = params.tilingData->k, f = params.tilingData->n / 2U;
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        AscendC::GlobalTensor<bfloat16_t> a, b, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrA2) + scratchRow * f);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.b2GmAddr) +
                          static_cast<uint64_t>(slice.expert) * f * h);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        c.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, f),
                                  Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(f, h), Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, h),
                                  Catlass::Arch::PositionGM{});
        WaveBlockScheduler scheduler({slice.rows, h, f},
                                     Catlass::MakeCoord(kDownRows, MegaMoeBF16Training::Wave::kTileColumns));
        const uint32_t tasks = scheduler.GetCoreLoops();
        const uint32_t first = MegaMoeImpl::GetGmmFirstOwnedTile(core, cores, startBlock);
        const bool owns = first < tasks;
        uint32_t readyM = 0xffffffffU;
        for (uint32_t task = first; task < tasks; task += cores) {
            const auto coord = scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * kDownRows, n = coord.n() * MegaMoeBF16Training::Wave::kTileColumns;
            const uint32_t rows = shape.m(), columns = shape.n();
            if (m != readyM) {
                MegaMoeTrainTimerPassiveScope wait(params.timer, MegaMoeTrainTimer::AIC_GMM2_WAIT, slice.expert,
                                                   slice.row + m, rows, n, kDownRows,
                                                   MegaMoeBF16Training::Wave::kTileColumns, f, h);
                const uint32_t activationColumns = MegaMoeBF16Training::Wave::kPairedColumns;
                while (AscendC::ReadGmByPassDCache(hpFlags + m / kRows) < Ops::Base::CeilDiv(f, activationColumns)) {
                }
                AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                readyM = m;
            }
            MegaMoeTrainTimerPassiveScope tile(params.timer, MegaMoeTrainTimer::AIC_GMM2, slice.expert, slice.row + m,
                                               rows, n, kDownRows, MegaMoeBF16Training::Wave::kTileColumns, f, h);
            auto tileA = tla::GetTile(ta, tla::MakeCoord(m, 0U), tla::MakeShape(rows, f));
            auto tileB = tla::GetTile(tb, tla::MakeCoord(0U, n), tla::MakeShape(f, columns));
            auto tileC = tla::GetTile(tc, tla::MakeCoord(m, n), tla::MakeShape(rows, columns));
            blockMmad(tileA, tileB, tileC, Catlass::GemmCoord{rows, columns, f});
        }
        startBlock = MegaMoeImpl::GetGmmNextStartBlock(startBlock, tasks, cores);
        return owns;
    }
};

}  // namespace MegaMoeBF16ForwardCatlassImpl
