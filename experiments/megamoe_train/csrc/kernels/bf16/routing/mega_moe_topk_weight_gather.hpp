// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"
#include "layout/mega_moe_topk_weight_gather_layout.hpp"
#include <type_traits>

namespace MegaMoeBF16Routing {

// Produce safe byte offsets before Gather: an index outside this window reads
// element zero, never an out-of-range element of a short second window.
__aicore__ inline void MakeWeightWindowOffsets(AscendC::LocalTensor<int32_t> indices,
                                               AscendC::LocalTensor<int32_t> offsets, uint32_t count, bool second)
{
    auto in = (__ubuf__ int32_t*)indices.GetPhyAddr();
    auto out = (__ubuf__ int32_t*)offsets.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63) / 64);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> boundary, zero;
        AscendC::Reg::Duplicate(boundary, int32_t{65536});
        AscendC::Reg::Duplicate(zero, int32_t{0});
        uint32_t remaining = count;
        for (uint16_t i = 0; i < blocks; ++i) {
            auto valid = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::RegTensor<int32_t> index, local, safe;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, in + i * 64);
            if (second) {
                AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GE>(belongs, index, boundary, valid);
                AscendC::Reg::Sub(local, index, boundary, valid);
            } else {
                AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(belongs, index, boundary, valid);
                AscendC::Reg::Adds(local, index, int32_t{0}, valid);
            }
            AscendC::Reg::Select<int32_t>(safe, local, zero, belongs);
            AscendC::Reg::Muls(safe, safe, int32_t{2}, valid);
            AscendC::Reg::DataCopy(out + i * 64, safe, valid);
        }
    }
}

// Select raw BF16 bits; there is no float Cast or numerical round trip.
__aicore__ inline void MergeWeightWindows(AscendC::LocalTensor<int32_t> indices, AscendC::LocalTensor<uint16_t> first,
                                          AscendC::LocalTensor<uint16_t> second, uint32_t count)
{
    auto idx = (__ubuf__ int32_t*)indices.GetPhyAddr();
    auto a = (__ubuf__ uint16_t*)first.GetPhyAddr();
    auto b = (__ubuf__ uint16_t*)second.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63) / 64);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> boundary;
        AscendC::Reg::Duplicate(boundary, int32_t{65536});
        uint32_t remaining = count;
        for (uint16_t i = 0; i < blocks; ++i) {
            auto valid = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::RegTensor<int32_t> index;
            AscendC::Reg::RegTensor<uint16_t> av, bv, result;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, idx + i * 64);
            AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(belongs, index, boundary, valid);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(av, a + i * 64);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bv, b + i * 64);
            AscendC::Reg::Select<uint32_t>((AscendC::Reg::RegTensor<uint32_t>&)result,
                                           (AscendC::Reg::RegTensor<uint32_t>&)av,
                                           (AscendC::Reg::RegTensor<uint32_t>&)bv, belongs);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(a + i * 64, result, valid);
        }
    }
}

__aicore__ inline bool GatherTopkWeightsTwoWindows(GM_ADDR weightsAddr, GM_ADDR sortedIndicesAddr, GM_ADDR outputAddr,
                                                   uint64_t totalRoutes, const TopkWeightGatherPlan& plan)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<> sourceBuf, indicesBuf, offsetsBuf, firstBuf, secondBuf;
    pipe.InitBuffer(sourceBuf, plan.sourceBytes);  // First allocation stays at UB zero.
    pipe.InitBuffer(indicesBuf, plan.kIndexBytes);
    pipe.InitBuffer(offsetsBuf, plan.kIndexBytes);
    pipe.InitBuffer(firstBuf, plan.kOutputBytes);
    pipe.InitBuffer(secondBuf, plan.kOutputBytes);
    auto source = sourceBuf.Get<uint16_t>();
    auto indices = indicesBuf.Get<int32_t>();
    auto offsets = offsetsBuf.Get<int32_t>();
    auto first = firstBuf.Get<uint16_t>();
    auto second = secondBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> weightsGm, outputGm;
    GlobalTensor<int32_t> indicesGm;
    weightsGm.SetGlobalBuffer((__gm__ uint16_t*)weightsAddr);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)sortedIndicesAddr);
    outputGm.SetGlobalBuffer((__gm__ uint16_t*)outputAddr);
    for (uint32_t offset = 0; offset < plan.count; offset += plan.kBatchRoutes) {
        const uint32_t count = plan.BatchCount(offset);
        SyncFunc<HardEvent::V_MTE2>();
        SyncFunc<HardEvent::MTE3_V>();
        DataCopyPad(indices, indicesGm[plan.begin + offset], DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        DataCopyPad(source, weightsGm, DataCopyExtParams{1U, plan.kMaxRoutes * 2U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        MakeWeightWindowOffsets(indices, offsets, count, false);
        PipeBarrier<PIPE_V>();
        Gather(first, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
        // Gather must finish reading the first window before MTE2 overwrites it.
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(source, weightsGm[plan.kMaxRoutes],
                    DataCopyExtParams{1U, static_cast<uint32_t>(totalRoutes - plan.kMaxRoutes) * 2U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        MakeWeightWindowOffsets(indices, offsets, count, true);
        PipeBarrier<PIPE_V>();
        Gather(second, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
        PipeBarrier<PIPE_V>();
        MergeWeightWindows(indices, first, second, count);
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(outputGm[plan.begin + offset], first, DataCopyExtParams{1U, count * 2U, 0U, 0U, 0U});
    }
    SyncFunc<HardEvent::MTE3_S>();
    pipe.Destroy();
    return true;
}

template <uint32_t ElementBytes>
__aicore__ inline void MakeWeightWindowOffsetsFor(AscendC::LocalTensor<int32_t> indices,
                                                  AscendC::LocalTensor<uint32_t> offsets, uint32_t count,
                                                  uint32_t windowStart, uint32_t windowRows)
{
    auto* in = (__ubuf__ uint32_t*)indices.GetPhyAddr();
    auto* out = (__ubuf__ uint32_t*)offsets.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63U) / 64U);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> start, rows, zero;
        AscendC::Reg::Duplicate(start, windowStart);
        AscendC::Reg::Duplicate(rows, windowRows);
        AscendC::Reg::Duplicate(zero, uint32_t{0});
        uint32_t remaining = count;
        for (uint16_t block = 0; block < blocks; ++block) {
            auto valid = AscendC::Reg::UpdateMask<uint32_t>(remaining);
            AscendC::Reg::RegTensor<uint32_t> index, local, safe;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, in + block * 64U);
            AscendC::Reg::Sub(local, index, start, valid);
            // Unsigned subtraction also excludes indices below this window.
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(belongs, local, rows, valid);
            AscendC::Reg::Select<uint32_t>(safe, local, zero, belongs);
            AscendC::Reg::Muls(safe, safe, uint32_t{ElementBytes}, valid);
            AscendC::Reg::DataCopy(out + block * 64U, safe, valid);
        }
    }
}

// Merge raw bits so neither BF16 nor FP32 weights undergo a numerical cast.
template <typename WeightBitsT>
__aicore__ inline void MergeWeightWindow(AscendC::LocalTensor<int32_t> indices,
                                         AscendC::LocalTensor<WeightBitsT> accumulated,
                                         AscendC::LocalTensor<WeightBitsT> candidate, uint32_t count,
                                         uint32_t windowStart, uint32_t windowRows)
{
    auto* idx = (__ubuf__ uint32_t*)indices.GetPhyAddr();
    auto* dst = (__ubuf__ WeightBitsT*)accumulated.GetPhyAddr();
    auto* src = (__ubuf__ WeightBitsT*)candidate.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63U) / 64U);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> start, rows;
        AscendC::Reg::Duplicate(start, windowStart);
        AscendC::Reg::Duplicate(rows, windowRows);
        uint32_t remaining = count;
        for (uint16_t block = 0; block < blocks; ++block) {
            auto valid = AscendC::Reg::UpdateMask<uint32_t>(remaining);
            AscendC::Reg::RegTensor<uint32_t> index, local;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, idx + block * 64U);
            AscendC::Reg::Sub(local, index, start, valid);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(belongs, local, rows, valid);
            AscendC::Reg::RegTensor<WeightBitsT> prior, next, result;
            if constexpr (sizeof(WeightBitsT) == sizeof(uint16_t)) {
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(prior, dst + block * 64U);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(next, src + block * 64U);
                AscendC::Reg::Select<uint32_t>((AscendC::Reg::RegTensor<uint32_t>&)result,
                                               (AscendC::Reg::RegTensor<uint32_t>&)next,
                                               (AscendC::Reg::RegTensor<uint32_t>&)prior, belongs);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + block * 64U, result,
                                                                                         valid);
            } else {
                AscendC::Reg::DataCopy(prior, dst + block * 64U);
                AscendC::Reg::DataCopy(next, src + block * 64U);
                AscendC::Reg::Select<uint32_t>(result, next, prior, belongs);
                AscendC::Reg::DataCopy(dst + block * 64U, result, valid);
            }
        }
    }
}

template <typename WeightBitsT>
__aicore__ inline bool GatherTopkWeightsWindows(GM_ADDR weightsAddr, GM_ADDR sortedIndicesAddr, GM_ADDR outputAddr,
                                                uint64_t totalRoutes,
                                                const TopkWeightGatherPlanFor<sizeof(WeightBitsT)>& plan)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<> sourceBuf, indicesBuf, offsetsBuf, accumulatedBuf, candidateBuf;
    pipe.InitBuffer(sourceBuf, plan.sourceBytes);
    pipe.InitBuffer(indicesBuf, plan.kIndexBytes);
    pipe.InitBuffer(offsetsBuf, plan.kIndexBytes);
    pipe.InitBuffer(accumulatedBuf, plan.kOutputBytes);
    pipe.InitBuffer(candidateBuf, plan.kOutputBytes);
    auto source = sourceBuf.Get<WeightBitsT>();
    auto indices = indicesBuf.Get<int32_t>();
    auto offsets = offsetsBuf.Get<uint32_t>();
    auto accumulated = accumulatedBuf.Get<WeightBitsT>();
    auto candidate = candidateBuf.Get<WeightBitsT>();
    GlobalTensor<WeightBitsT> weightsGm, outputGm;
    GlobalTensor<int32_t> indicesGm;
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightBitsT*>(weightsAddr));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndicesAddr));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightBitsT*>(outputAddr));
    for (uint32_t offset = 0; offset < plan.count; offset += plan.kBatchRoutes) {
        const uint32_t count = plan.BatchCount(offset);
        SyncFunc<HardEvent::V_MTE2>();
        SyncFunc<HardEvent::MTE3_V>();
        DataCopyPad(indices, indicesGm[plan.begin + offset], DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        for (uint32_t window = 0; window < plan.windowCount; ++window) {
            const uint32_t first = window * plan.kMaxRoutes;
            const uint32_t rows = plan.WindowRows(totalRoutes, window);
            // Drain Gather before reusing the same source window for MTE2.
            SyncFunc<HardEvent::V_MTE2>();
            DataCopyPad(source, weightsGm[first],
                        DataCopyExtParams{1U, rows * uint32_t{sizeof(WeightBitsT)}, 0U, 0U, 0U},
                        DataCopyPadExtParams<WeightBitsT>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_V>();
            MakeWeightWindowOffsetsFor<sizeof(WeightBitsT)>(indices, offsets, count, first, rows);
            PipeBarrier<PIPE_V>();
            if (window == 0) {
                Gather(accumulated, source, offsets, 0U, count);
            } else {
                Gather(candidate, source, offsets, 0U, count);
                PipeBarrier<PIPE_V>();
                MergeWeightWindow(indices, accumulated, candidate, count, first, rows);
            }
        }
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(outputGm[plan.begin + offset], accumulated,
                    DataCopyExtParams{1U, count * uint32_t{sizeof(WeightBitsT)}, 0U, 0U, 0U});
    }
    SyncFunc<HardEvent::MTE3_S>();
    pipe.Destroy();
    return true;
}

// The gather pipe is released before Dispatch/transport allocates its UB.
// The BF16 two-window implementation retains its established instruction path.
template <typename TopkWeightT = bfloat16_t>
__aicore__ inline bool TryGatherTopkWeights(GM_ADDR weightsAddr, GM_ADDR sortedIndicesAddr, GM_ADDR outputAddr,
                                            uint64_t totalRoutes, uint32_t coreIndex, uint32_t coreCount,
                                            bool allowMultipleWindows = false)
{
    using namespace AscendC;
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    using WeightBitsT = std::conditional_t<sizeof(TopkWeightT) == sizeof(uint16_t), uint16_t, uint32_t>;
    using Plan = TopkWeightGatherPlanFor<sizeof(TopkWeightT)>;
    const Plan plan(totalRoutes, coreIndex, coreCount, allowMultipleWindows);
    if (!plan.enabled) {
        return false;
    }
    if (plan.count == 0) {
        return true;
    }
    if constexpr (sizeof(TopkWeightT) == sizeof(uint16_t)) {
        if (plan.twoWindows) {
            return GatherTopkWeightsTwoWindows(weightsAddr, sortedIndicesAddr, outputAddr, totalRoutes, plan);
        }
    }
    if (plan.windowCount > 1) {
        return GatherTopkWeightsWindows<WeightBitsT>(weightsAddr, sortedIndicesAddr, outputAddr, totalRoutes, plan);
    }

    TPipe pipe;
    TBuf<> sourceBuf;
    TBuf<> indicesBuf;
    TBuf<> outputBuf;
    pipe.InitBuffer(sourceBuf, plan.sourceBytes);
    pipe.InitBuffer(indicesBuf, Plan::kIndexBytes);
    pipe.InitBuffer(outputBuf, Plan::kOutputBytes);
    auto source = sourceBuf.Get<WeightBitsT>();
    auto indices = indicesBuf.Get<int32_t>();
    auto output = outputBuf.Get<WeightBitsT>();
    GlobalTensor<WeightBitsT> weightsGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<WeightBitsT> outputGm;
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightBitsT*>(weightsAddr));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndicesAddr));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightBitsT*>(outputAddr));

    // Copy only valid GM bytes; the UB allocation alone is rounded up.
    DataCopyPad(source, weightsGm,
                DataCopyExtParams{1U, static_cast<uint32_t>(totalRoutes * sizeof(WeightBitsT)), 0U, 0U, 0U},
                DataCopyPadExtParams<WeightBitsT>{false, 0U, 0U, 0U});
    for (uint32_t offset = 0; offset < plan.count; offset += Plan::kBatchRoutes) {
        const uint32_t count = plan.BatchCount(offset);
        // Previous Gather must finish reading indices before MTE2 reuses them.
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(indices, indicesGm[plan.begin + offset],
                    DataCopyExtParams{1U, count * uint32_t{sizeof(int32_t)}, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        // Gather addresses raw weight bits by byte offset.
        Muls(indices, indices, static_cast<int32_t>(sizeof(WeightBitsT)), count);
        PipeBarrier<PIPE_V>();
        SyncFunc<HardEvent::MTE3_V>();
        Gather(output, source, indices.ReinterpretCast<uint32_t>(), 0U, count);
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(outputGm[plan.begin + offset], output,
                    DataCopyExtParams{1U, count * uint32_t{sizeof(WeightBitsT)}, 0U, 0U, 0U});
    }
    // Count notification must not overtake the MTE3 publication to SHMEM.
    // Release this UB phase before FirstBuffInit allocates Dispatch buffers.
    SyncFunc<HardEvent::MTE3_S>();
    pipe.Destroy();
    return true;
}

}  // namespace MegaMoeBF16Routing
