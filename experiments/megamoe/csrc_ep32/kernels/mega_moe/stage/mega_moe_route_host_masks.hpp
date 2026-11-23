#pragma once

#include "kernel_operator.h"

namespace MegaMoeImpl {
// One vector lane per token. The unsigned subtraction also rejects negative and out-of-range IDs.
template <bool Compact>
__simd_vf__ inline void RouteHostMasksVF(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* masks, __ubuf__ int32_t* tokens,
                                         __ubuf__ int32_t* count, uint32_t rows, uint32_t topK, uint32_t first,
                                         uint32_t width)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> totals, one;
    Duplicate(totals, 0, full);
    Duplicate(one, 1, full);
    UnalignRegForStore stream;
    if constexpr (Compact) ClearSpr<SpecialPurposeReg::AR>();
    for (uint32_t base = 0; base < rows; base += lanes) {
        uint32_t left = rows - base;
        auto active = UpdateMask<uint32_t>(left);
        RegTensor<int32_t> index, offsets;
        Arange(index, static_cast<int32_t>(base));
        Muls(offsets, index, static_cast<int32_t>(topK), full);
        RegTensor<uint32_t> mask, localMask, value, bit;
        Duplicate(mask, 0U, full);
        if constexpr (!Compact) Duplicate(localMask, 0U, full);
        for (uint32_t k = 0; k < topK; ++k) {
            Gather(value, ids + k, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
            Duplicate(bit, 1U << k, full);
            if constexpr (!Compact) {
                RegTensor<uint32_t> localId;
                Adds(localId, value, 0U - (width - first), active);
                MaskReg local;
                Compares<uint32_t, AscendC::CMPMODE::LT>(local, localId, width, active);
                Add<uint32_t, MaskMergeMode::MERGING>(localMask, localMask, bit, local);
            }
            Adds(value, value, 0U - first, active);
            MaskReg match;
            Compares<uint32_t, AscendC::CMPMODE::LT>(match, value, width, active);
            Add<uint32_t, MaskMergeMode::MERGING>(mask, mask, bit, match);
        }
        StoreAlign(masks + base, mask, active);
        if constexpr (!Compact) StoreAlign(masks + (rows + 7U) / 8U * 8U + base, localMask, active);
        if constexpr (Compact) {
            MaskReg selected;
            Compares<uint32_t, AscendC::CMPMODE::NE>(selected, mask, 0U, active);
            RegTensor<int32_t> packed;
            GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(packed, index, selected);
            StoreUnAlign(tokens, packed, stream);
            Add<int32_t, MaskMergeMode::MERGING>(totals, totals, one, selected);
        }
    }
    if constexpr (Compact) {
        StoreUnAlignPost(tokens, stream);
        RegTensor<int32_t> sum;
        Reduce<ReduceType::SUM>(sum, totals, full);
        UnalignRegForStore out;
        StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(count, sum, out, 1U);
        StoreUnAlignPost(count, out, 0);
    }
}

}  // namespace MegaMoeImpl
