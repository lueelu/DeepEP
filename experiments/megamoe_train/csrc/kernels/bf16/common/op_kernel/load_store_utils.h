// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

/*!
 * \file load_store_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_
#define OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace ops {
using namespace AscendC;

constexpr AscendC::Reg::CastTrait LOAD_CAST_TRAIT_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> inputFp16;
        Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(inputFp16, input + offset);
        Reg::Cast<float, half, LOAD_CAST_TRAIT_B16_TO_B32>(dst, inputFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> inputBf16;
        Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(inputBf16, input + offset);
        Reg::Cast<float, bfloat16_t, LOAD_CAST_TRAIT_B16_TO_B32>(dst, inputBf16, preg);
    } else {
        Reg::DataCopy(dst, (__local_mem__ float*)(input) + offset);
    }
}

}  // namespace ops

#endif  // OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_
