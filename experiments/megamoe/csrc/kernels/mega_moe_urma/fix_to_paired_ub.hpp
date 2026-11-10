// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "kernel_operator.h"
#include "tla/tensor.hpp"
#include <type_traits>

namespace UbxMegaMoeImpl::GmmKernel {
// Pinned CATLASS 2b4bd7c implements only NO_QUANT for L0C->UB. This adapter
// follows its NO_SPLIT stride/subblock handling, but preserves the existing
// W4 GM-output PER_TENSOR *64 restoration before the BF16 cast.
constexpr AscendC::FixpipeConfig kPairedUbFixConfig{AscendC::CO2Layout::ROW_MAJOR, true};
template <class TensorL0C>
__aicore__ inline void StoreUb(const TensorL0C& src, uint32_t rows, uint32_t columns, uint8_t unitFlag)
{
    using Accumulator = typename TensorL0C::Element;
    static_assert(std::is_same_v<Accumulator, float>);
    AscendC::LocalTensor<bfloat16_t> dst(AscendC::TPosition::VECCALC, 0U, 256U * 256U);
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> p{};
    p.mSize = rows;
    p.nSize = columns;
    p.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
    p.dstStride = 256U;
    p.quantPre = ::QuantMode_t::QF322BF16_PRE;
    // Bit representation of float(64), identical to CATLASS PER_TENSOR's
    // static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale)).
    p.deqScalar = 0x42800000ULL;
    p.reluEn = false;
    p.unitFlag = unitFlag;
    p.dualDstCtl = 0U;
    p.subBlockId = 1U;
    const auto offset = src.layout()(src.coord());
    AscendC::Fixpipe<bfloat16_t, Accumulator, kPairedUbFixConfig>(dst, src.data()[offset], p);
}
}  // namespace UbxMegaMoeImpl::GmmKernel
