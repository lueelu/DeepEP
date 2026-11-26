// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"
#include <type_traits>
#include "common/mega_moe_utils.hpp"

namespace MegaMoeBF16Routing {

struct UnpermutePlan {
    static constexpr uint32_t kIndexElements = 1024;
    static constexpr uint32_t kIndexBytes = kIndexElements * sizeof(int32_t);
    static constexpr uint32_t kBf16Rows = 4;
    static constexpr uint32_t kFp32Rows = 2;
    bool enabled = false;
    uint32_t rowBytes = 0;
    uint32_t batchTokens = 0;

    __host_aicore__ constexpr UnpermutePlan(uint32_t hidden, uint32_t topK, uint32_t ubBytes)
    {
        const uint64_t bytes = uint64_t{hidden} * sizeof(uint16_t);
        const uint64_t scratchBytes = kBf16Rows * bytes + uint64_t{kFp32Rows} * hidden * sizeof(float);
        if (hidden == 0 || bytes % 32 != 0 || topK == 0 || topK > kIndexElements ||
            scratchBytes + kIndexBytes > ubBytes) {
            return;
        }
        enabled = true;
        rowBytes = static_cast<uint32_t>(bytes);
        batchTokens = kIndexElements / topK;
    }

    __host_aicore__ constexpr uint32_t BatchCount(uint32_t remaining) const
    {
        return remaining < batchTokens ? remaining : batchTokens;
    }
};

__aicore__ inline void LoadUnpermuteInput(AscendC::LocalTensor<bfloat16_t> input,
                                          AscendC::GlobalTensor<bfloat16_t> expandedX, int32_t row, uint32_t hidden,
                                          uint32_t slot)
{
    using namespace AscendC;
    const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(input, expandedX[uint64_t{static_cast<uint32_t>(row)} * hidden], hidden);
    SetFlag<HardEvent::MTE2_V>(event);
}

__aicore__ inline void AccumulateUnpermuteRoute(AscendC::LocalTensor<float> sum, AscendC::LocalTensor<float> converted,
                                                AscendC::LocalTensor<bfloat16_t> input, uint32_t length, bool first,
                                                bool applyWeight = false, float weight = 1.0F)
{
    using namespace AscendC;
    auto value = first ? sum : converted;
    Cast(value, input, RoundMode::CAST_NONE, length);
    if (!first) PipeBarrier<PIPE_V>();
    if (applyWeight)
        MegaMoeImpl::AccumulateRoute<true, true>(sum, value, length, first, weight);
    else
        MegaMoeImpl::AccumulateRoute<false, true>(sum, value, length, first);
    PipeBarrier<PIPE_V>();
}

template <typename TopkWeightT>
__aicore__ inline float RouteWeight(GM_ADDR address, uint64_t route)
{
    if constexpr (std::is_same_v<TopkWeightT, float>) {
        AscendC::GlobalTensor<float> weights;
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
        return weights.GetValue(route);
    } else {
        AscendC::GlobalTensor<uint16_t> weights;
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(address));
        union FloatBits {
            uint32_t bits;
            float value;
        };
        FloatBits word{static_cast<uint32_t>(weights.GetValue(route)) << 16U};
        return word.value;
    }
}

// Called only after Combine completion and the existing cross-rank barrier.
// False selects the bounded-scratch path below.
template <typename TopkWeightT = bfloat16_t>
__aicore__ inline bool TryUnpermute(GM_ADDR indicesAddr, GM_ADDR expandedXAddr, GM_ADDR outputAddr, uint32_t hidden,
                                    uint32_t topK, uint32_t tokenBegin, uint32_t tokenCount, uint32_t ubBytes,
                                    GM_ADDR weightsAddr = nullptr, bool applyWeights = false)
{
    using namespace AscendC;
    const UnpermutePlan plan(hidden, topK, ubBytes);
    if (!plan.enabled) {
        return false;
    }
    if (tokenCount == 0) {
        return true;
    }

    TPipe pipe;
    TBuf<> rowsBuf;
    TBuf<> fp32Buf;
    TBuf<> indicesBuf;
    pipe.InitBuffer(rowsBuf, UnpermutePlan::kBf16Rows * plan.rowBytes);
    pipe.InitBuffer(fp32Buf, UnpermutePlan::kFp32Rows * hidden * sizeof(float));
    pipe.InitBuffer(indicesBuf, UnpermutePlan::kIndexBytes);
    auto rows = rowsBuf.Get<bfloat16_t>();
    auto sum = fp32Buf.Get<float>();
    auto converted = sum[hidden];
    auto indices = indicesBuf.Get<int32_t>();
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<bfloat16_t> expandedX;
    GlobalTensor<bfloat16_t> output;
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indicesAddr));
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(expandedXAddr));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(outputAddr));

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (uint32_t offset = 0; offset < tokenCount;) {
        const uint32_t count = plan.BatchCount(tokenCount - offset);
        // Scalar must finish reading the old index batch before MTE2 reuses UB.
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(indices, indicesGm[(uint64_t{tokenBegin} + offset) * topK],
                    DataCopyExtParams{1U, count * topK * uint32_t{sizeof(int32_t)}, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t token = 0; token < count; ++token) {
            const uint32_t outputSlot = (offset + token) % 2;
            const auto outputEvent = outputSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            auto result = rows[outputSlot * hidden];
            const uint32_t indexBase = token * topK;

            // Prime two inputs before consuming either. The other input can
            // arrive while Vector consumes this one, including for odd TopK.
            LoadUnpermuteInput(rows[2 * hidden], expandedX, indices.GetValue(indexBase), hidden, 0);
            if (topK > 1) {
                LoadUnpermuteInput(rows[3 * hidden], expandedX, indices.GetValue(indexBase + 1), hidden, 1);
            }
            for (uint32_t expert = 0; expert < topK; ++expert) {
                const uint32_t inputSlot = expert % 2;
                const auto inputEvent = inputSlot == 0 ? EVENT_ID0 : EVENT_ID1;
                auto input = rows[(2 + inputSlot) * hidden];
                WaitFlag<HardEvent::MTE2_V>(inputEvent);
                const float weight =
                    applyWeights
                        ? RouteWeight<TopkWeightT>(weightsAddr, (uint64_t{tokenBegin} + offset + token) * topK + expert)
                        : 1.0F;
                AccumulateUnpermuteRoute(sum, converted, input, hidden, expert == 0, applyWeights, weight);
                SetFlag<HardEvent::V_MTE2>(inputEvent);
                if (expert + 2 < topK) {
                    LoadUnpermuteInput(input, expandedX, indices.GetValue(indexBase + expert + 2), hidden, inputSlot);
                }
            }
            // Reusing a BF16 writeback slot must wait for its previous DMA.
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            Cast(result, sum, RoundMode::CAST_RINT, hidden);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopy(output[(uint64_t{tokenBegin} + offset + token) * hidden], result, hidden);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        offset += count;
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    // Waits above retire both rings; drain the pipes before releasing their UB.
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
    return true;
}

template <typename TopkWeightT = bfloat16_t>
__aicore__ inline void UnpermuteByHiddenChunk(GM_ADDR indicesAddr, GM_ADDR expandedXAddr, GM_ADDR outputAddr,
                                              uint32_t hidden, uint32_t topK, uint32_t tokenBegin, uint32_t tokenCount,
                                              GM_ADDR weightsAddr = nullptr, bool applyWeights = false)
{
    using namespace AscendC;
    if (tokenCount == 0) {
        return;
    }
    constexpr uint32_t kChunkCols = 4096;
    const uint32_t chunkCols = hidden < kChunkCols ? hidden : kChunkCols;
    TPipe pipe;
    TBuf<> bf16Buf;
    TBuf<> fp32Buf;
    pipe.InitBuffer(bf16Buf, chunkCols * sizeof(bfloat16_t));
    pipe.InitBuffer(fp32Buf, 2 * chunkCols * sizeof(float));
    auto row = bf16Buf.Get<bfloat16_t>();
    auto sum = fp32Buf.Get<float>();
    auto converted = sum[chunkCols];
    GlobalTensor<int32_t> indices;
    GlobalTensor<bfloat16_t> expandedX;
    GlobalTensor<bfloat16_t> output;
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indicesAddr));
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(expandedXAddr));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(outputAddr));

    for (uint64_t token = tokenBegin; token < uint64_t{tokenBegin} + tokenCount; ++token) {
        for (uint32_t col = 0; col < hidden; col += chunkCols) {
            const uint32_t remaining = hidden - col;
            const uint32_t count = remaining < chunkCols ? remaining : chunkCols;
            for (uint32_t route = 0; route < topK; ++route) {
                const auto index = static_cast<uint32_t>(indices.GetValue(token * topK + route));
                DataCopy(row, expandedX[uint64_t{index} * hidden + col], count);
                SyncFunc<HardEvent::MTE2_V>();
                const float weight = applyWeights ? RouteWeight<TopkWeightT>(weightsAddr, token * topK + route) : 1.0F;
                AccumulateUnpermuteRoute(sum, converted, row, count, route == 0, applyWeights, weight);
                SyncFunc<HardEvent::V_MTE2>();
            }
            Cast(row, sum, RoundMode::CAST_RINT, count);
            SyncFunc<HardEvent::V_MTE3>();
            DataCopy(output[token * hidden + col], row, count);
            SyncFunc<HardEvent::MTE3_MTE2>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}

}  // namespace MegaMoeBF16Routing
