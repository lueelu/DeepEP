// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_COMBINE_VF_H
#define MEGA_MOE_COMBINE_VF_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

constexpr AscendC::Reg::CastTrait GATEWAY_COMBINE_CAST_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

// Each selected row owns packets consecutive 32B tails in the snapshot. Holes
// retain their row positions but must never be read. Only lane zero of ready
// accumulates bits; no packet/row result crosses back to Scalar inside this VF.
__simd_vf__ inline void CheckGatewayCombinePacketsVF(__ubuf__ uint32_t* tails, __ubuf__ uint32_t* status,
                                                     uint32_t packets, uint32_t topK, uint32_t mask, uint32_t sequence)
{
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    uint32_t single = 1U;
    auto firstLane = AscendC::Reg::UpdateMask<uint32_t>(single);
    AscendC::Reg::RegTensor<uint32_t> values, staleLanes, total, one, bit, ready;
    AscendC::Reg::RegTensor<int32_t> indices;
    AscendC::Reg::Duplicate(one, 1U, full);
    AscendC::Reg::Duplicate(ready, 0U, full);
    for (uint32_t topk = 0U; topk < topK; ++topk) {
        const uint32_t rowBit = uint32_t{1} << topk;
        if ((mask & rowBit) == 0U) continue;
        AscendC::Reg::Duplicate(staleLanes, 0U, full);
        for (uint32_t packet = 0U; packet < packets; packet += lanes) {
            uint32_t remaining = packets - packet;
            auto active = AscendC::Reg::UpdateMask<uint32_t>(remaining);
            AscendC::Reg::Arange(indices, static_cast<int32_t>(packet));
            AscendC::Reg::Muls(indices, indices, 8, full);
            AscendC::Reg::Adds(indices, indices, 7, full);
            AscendC::Reg::Gather(values, tails + topk * packets * 8U,
                                 reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(indices), active);
            AscendC::Reg::MaskReg stale;
            AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(stale, values, sequence, active);
            AscendC::Reg::Add<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(staleLanes, staleLanes, one, stale);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(total, staleLanes, full);
        AscendC::Reg::MaskReg rowReady;
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(rowReady, total, 0U, firstLane);
        AscendC::Reg::Duplicate(bit, rowBit, full);
        AscendC::Reg::Add<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(ready, ready, bit, rowReady);
    }
    AscendC::Reg::UnalignRegForStore store;
    AscendC::Reg::StoreUnAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(status, ready, store, 1U);
    AscendC::Reg::StoreUnAlignPost(status, store, 0);
}

// One expert row per VF preserves the existing input prefetch schedule and
// top-k order. Keep Mul and Add separate: contracting them into FMA or adding
// the first product to zero changes the original FP32 rounding/signed zero.
template <bool First>
__simd_vf__ inline void AccumulateGatewayCombineRowVF(__ubuf__ bfloat16_t* input, __ubuf__ float* sum,
                                                      __ubuf__ float* weight, uint32_t hidden)
{
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(float);
    AscendC::Reg::RegTensor<bfloat16_t> bf16;
    AscendC::Reg::RegTensor<float> value, scale, accumulated;
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(scale, weight);
    for (uint32_t offset = 0U; offset < hidden; offset += lanes) {
        uint32_t remaining = hidden - offset;
        auto active = AscendC::Reg::UpdateMask<float>(remaining);
        // Input/sum slots reserve a full final vector even for h % 64 == 32.
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16, input + offset);
        AscendC::Reg::Cast<float, bfloat16_t, GATEWAY_COMBINE_CAST_TO_FP32>(value, bf16, active);
        AscendC::Reg::Mul(value, value, scale, active);
        if constexpr (!First) {
            AscendC::Reg::LoadAlign(accumulated, sum + offset);
            AscendC::Reg::Add(value, accumulated, value, active);
        }
        AscendC::Reg::StoreAlign(sum + offset, value, active);
    }
}

}  // namespace MegaMoeImpl
#endif
