// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_COUNT_ARRIVAL_H
#define MEGA_MOE_COUNT_ARRIVAL_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// Gather the source-major wire table into the expert-major prefix input.
// Every count includes its own epoch; padded experts must never participate.
// A snapshot with any stale word is rejected, even when the last word arrived first.
__simd_vf__ inline void CheckTransposeExpertCountsVF(__ubuf__ uint32_t* sourceCounts, __ubuf__ uint32_t* expertCounts,
                                                     __ubuf__ uint32_t* staleCount, uint32_t worldSize,
                                                     uint32_t expertCount, uint32_t sourceStride,
                                                     uint32_t expectedEpoch)
{
    constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> values, epochs, lowMask, one, staleLanes, total;
    AscendC::Reg::RegTensor<int32_t> indices;
    AscendC::Reg::Duplicate(lowMask, 0x00FFFFFFU, full);
    AscendC::Reg::Duplicate(one, 1U, full);
    AscendC::Reg::Duplicate(staleLanes, 0U, full);
    AscendC::Reg::UnalignRegForStore packedStore;
    __ubuf__ uint32_t* dst = expertCounts;
    for (uint32_t expert = 0; expert < expertCount; ++expert) {
        for (uint32_t source = 0; source < worldSize; source += LANES) {
            uint32_t count = worldSize - source;
            count = count < LANES ? count : LANES;
            uint32_t remaining = count;
            auto active = AscendC::Reg::UpdateMask<uint32_t>(remaining);
            AscendC::Reg::Arange(indices, static_cast<int32_t>(source));
            AscendC::Reg::Muls(indices, indices, static_cast<int32_t>(sourceStride), full);
            AscendC::Reg::Adds(indices, indices, static_cast<int32_t>(expert), full);
            AscendC::Reg::Gather(values, sourceCounts, reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(indices),
                                 active);
            AscendC::Reg::ShiftRights(epochs, values, static_cast<int16_t>(24), active);
            AscendC::Reg::MaskReg stale;
            AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(stale, epochs, expectedEpoch, active);
            AscendC::Reg::Add<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(staleLanes, staleLanes, one, stale);
            AscendC::Reg::And(values, values, lowMask, active);
            AscendC::Reg::StoreUnAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, values, packedStore,
                                                                                              count);
        }
    }
    AscendC::Reg::StoreUnAlignPost(dst, packedStore, 0);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(total, staleLanes, full);
    AscendC::Reg::UnalignRegForStore statusStore;
    AscendC::Reg::StoreUnAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(staleCount, total, statusStore,
                                                                                      1U);
    AscendC::Reg::StoreUnAlignPost(staleCount, statusStore, 0);
}

}  // namespace MegaMoeImpl
#endif
