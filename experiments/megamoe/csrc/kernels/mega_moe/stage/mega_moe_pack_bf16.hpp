// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

#pragma once

#include "common/mega_moe_forward_save.hpp"

namespace MegaMoeImpl::Bf16Route {
using namespace AscendC;

constexpr uint32_t kIndexBatch = 64;

// Publish this call's route table, including when backward follows a newer microbatch.
__aicore__ inline void PublishDispatchInput(GM_ADDR input, GM_ADDR sourceTokens, GM_ADDR rows, GM_ADDR routeIndices,
                                            uint32_t m, uint32_t hidden, uint32_t topK, uint32_t core, uint32_t cores,
                                            LocalTensor<uint8_t> scratch)
{
    MegaMoeForwardSave::CopyContiguous(sourceTokens, input, uint64_t(m) * hidden * sizeof(bfloat16_t), core, cores,
                                       scratch);
    MegaMoeForwardSave::CopyContiguous(routeIndices, rows, uint64_t(m) * topK * sizeof(int32_t), core, cores, scratch);
}

template <class Visit>
__aicore__ inline void ForEachSourceToken(GlobalTensor<int32_t> routeIds, LocalTensor<int32_t> indices, uint32_t rows,
                                          uint32_t topK, Visit visit)
{
    for (uint32_t begin = 0; begin < rows; begin += kIndexBatch) {
        const uint32_t count = rows - begin < kIndexBatch ? rows - begin : kIndexBatch;
        DataCopyPad(indices, routeIds[begin], {1U, count * uint32_t(sizeof(int32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        MegaMoeForwardSave::SyncCopyStage<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < count; ++i) {
            visit(begin + i, static_cast<uint32_t>(indices.GetValue(i)) / topK);
        }
        MegaMoeForwardSave::SyncCopyStage<HardEvent::S_MTE2>();
    }
}
}  // namespace MegaMoeImpl::Bf16Route
