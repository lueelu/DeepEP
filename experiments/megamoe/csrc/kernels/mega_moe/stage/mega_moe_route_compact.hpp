// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_ROUTE_COMPACT_H
#define MEGA_MOE_ROUTE_COMPACT_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// One VF builds every owned expert's stable compact route. The caller supplies
// a separate, 32B-aligned route slice per expert and enough capacity for one
// occurrence per source token (top-k expert IDs are distinct within a token).
// topkIds is padded to a full vector; only itemCount entries participate.
// Counts stay in vector registers until their UB write, so no per-expert
// GetSpr()/V-to-scalar round trip is needed.
__simd_vf__ inline void RelayCompactRoutesVF(__ubuf__ int32_t* topkIds, __ubuf__ int32_t* routeIndices,
                                             __ubuf__ int32_t* counts, uint32_t itemCount, uint32_t topK,
                                             int32_t expertBegin, uint32_t expertCount, uint32_t routeStrideItems)
{
    (void)topK;
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(int32_t);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<int32_t> idsReg;
    AscendC::Reg::RegTensor<int32_t> indexReg;
    AscendC::Reg::RegTensor<int32_t> compactReg;
    AscendC::Reg::RegTensor<int32_t> laneCountsReg;
    AscendC::Reg::RegTensor<int32_t> countReg;
    AscendC::Reg::RegTensor<int32_t> oneReg;
    AscendC::Reg::UnalignRegForStore countStore;
    __ubuf__ int32_t* countDst = counts;
    AscendC::Reg::Duplicate(oneReg, 1, fullMask);

    for (uint32_t expert = 0U; expert < expertCount; ++expert) {
        __ubuf__ int32_t* routeDst = routeIndices + expert * routeStrideItems;
        AscendC::Reg::UnalignRegForStore routeStore;
        AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        AscendC::Reg::Duplicate(laneCountsReg, 0, fullMask);
        for (uint32_t offset = 0U; offset < itemCount; offset += ELEMENTS_PER_VECTOR) {
            uint32_t remaining = itemCount - offset;
            AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::LoadAlign(idsReg, topkIds + offset);
            AscendC::Reg::Arange(indexReg, static_cast<int32_t>(offset));
            AscendC::Reg::MaskReg matchMask;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(
                matchMask, idsReg, expertBegin + static_cast<int32_t>(expert), activeMask);
            AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(compactReg, indexReg, matchMask);
            // The implicit AR offset appends precisely the squeezed lanes;
            // chunks and lanes are visited in increasing top-k index order.
            AscendC::Reg::StoreUnAlign(routeDst, compactReg, routeStore);
            AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(laneCountsReg, laneCountsReg, oneReg,
                                                                             matchMask);
        }
        if (itemCount != 0U) {
            AscendC::Reg::StoreUnAlignPost(routeDst, routeStore);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(countReg, laneCountsReg, fullMask);
        // Explicit one-element stores pack counts densely even when the next
        // expert's count address is not 32B-aligned. This does not use AR.
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(countDst, countReg, countStore,
                                                                                         1U);
    }
    if (expertCount != 0U) {
        AscendC::Reg::StoreUnAlignPost(countDst, countStore, 0);
    }
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_ROUTE_COMPACT_H
