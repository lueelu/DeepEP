// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernels/mega_moe_urma/abi.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {
constexpr uint32_t kExpertRouteBatch = 4096U;  // preserve the E<=384 path
constexpr uint32_t kExpertWideBatch = 3072U;
constexpr uint32_t kExpertRouteStorageItems = 10U * kExpertWideBatch;
static_assert(7U * kExpertRouteBatch <= kExpertRouteStorageItems);
constexpr uint32_t kExpertRoutesPerWorker = (kExperts + kLocalWorkers - 1U) / kLocalWorkers;

// A worker owns complete experts. Each output slot can hold the entire input
// batch, including repeated expert IDs in one token's top-k list. Flat route
// IDs retain the top-k slot; only Pack divides by top-k to read X.
__simd_vf__ inline void CompactOwnedExpertsVF(__ubuf__ int32_t* ids, __ubuf__ int32_t* routes, __ubuf__ int32_t* counts,
                                              uint32_t valid, uint32_t flatBegin, uint32_t expertBegin,
                                              uint32_t expertCount, uint32_t outputStride)
{
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(int32_t);
    static_assert(kExpertRoutesPerWorker <= lanes);
    auto full = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<int32_t> idsReg, flat, compact, ones, totals, reduced, broadcast, packed, laneIndex;
    AscendC::Reg::RegTensor<uint32_t> zeroIndex;
    AscendC::Reg::Duplicate(ones, 1, full);
    AscendC::Reg::Duplicate(zeroIndex, 0U, full);
    AscendC::Reg::Duplicate(packed, 0, full);
    AscendC::Reg::Arange(laneIndex, 0);
    for (uint32_t e = 0; e < expertCount; ++e) {
        AscendC::Reg::Duplicate(totals, 0, full);
        auto* dst = routes + e * outputStride;
        AscendC::Reg::UnalignRegForStore tail;
        AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        for (uint32_t offset = 0; offset < valid; offset += lanes) {
            uint32_t remaining = valid - offset;
            auto active = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::LoadAlign(idsReg, ids + offset);
            AscendC::Reg::Arange(flat, static_cast<int32_t>(flatBegin + offset));
            AscendC::Reg::MaskReg match;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(match, idsReg, static_cast<int32_t>(expertBegin + e),
                                                                  active);
            AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(compact, flat, match);
            AscendC::Reg::StoreUnAlign(dst, compact, tail);
            AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(totals, totals, ones, match);
        }
        if (valid) AscendC::Reg::StoreUnAlignPost(dst, tail);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reduced, totals, full);
        AscendC::Reg::Gather(broadcast, reduced, zeroIndex);
        AscendC::Reg::MaskReg lane;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(lane, laneIndex, static_cast<int32_t>(e), full);
        AscendC::Reg::Select(packed, broadcast, packed, lane);
    }
    AscendC::Reg::StoreAlign(counts, packed, full);
}
}  // namespace MegaMoeUrma
