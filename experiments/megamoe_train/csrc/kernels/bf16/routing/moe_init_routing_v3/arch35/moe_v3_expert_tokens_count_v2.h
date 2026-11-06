/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_expert_tokens_count_v2.h
 * \brief ExpertTokensCount V2: 直接对 expertIdx 执行原子加，无需排序
 */

#ifndef MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE
#define MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class ExpertTokensCountV2 {
public:
    __aicore__ inline ExpertTokensCountV2(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCount,
                                const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expertTokensCountGm_;
    AscendC::LocalTensor<int32_t> tmpSum{AscendC::TPosition::VECIN, 0, 0};
    TPipe* pipe_;

    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
};

__aicore__ inline void ExpertTokensCountV2::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCount,
                                                 const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    // 使用与 V1 相同的 Tiling 参数
    auto* tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    needCoreNum_ = tokensCountTiling->needCoreNum;
    perCoreElements_ = tokensCountTiling->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;

    // 计算当前 block 处理的元素数
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = tokensCountTiling->lastCoreElements;
    } else {
        curCoreElements_ = perCoreElements_;
    }

    // 设置 GM 地址
    // expertIdx: 展平后的数组，总长度 = n * k = M * topK
    // 这里假设 expertIdx 已经展平为一维数组
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx + blockIdx_ * perCoreElements_);

    // expertTokensCount: 每个 expert 的 token 数量
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int32_t*)expertTokensCount);
}

__aicore__ inline void ExpertTokensCountV2::Process()
{
    // 仅需要的 core 执行
    if (blockIdx_ >= needCoreNum_) {
        return;
    }
    AscendC::Duplicate(tmpSum, 0, actualExpertNum_);
    SetWaitFlag<HardEvent::V_S>();

    // 直接遍历当前 block 负责的 expertIdx 元素
    // 对每个元素执行原子加
    for (int64_t i = 0; i < curCoreElements_; i++) {
        int32_t expertId = expertIdxGm_(i);
        tmpSum(expertId) = tmpSum(expertId) + 1;
    }
    SetWaitFlag<HardEvent::S_V>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertTokensCountGm_, tmpSum, copyParams);
    // DataCopy(expertTokensCountGm_, tmpSum, actualExpertNum_);
    SetAtomicNone();
}

}  // namespace MoeInitRoutingV3

#endif  // MOE_V3_EXPERT_TOKENS_COUNT_V2_H_REGBASE
