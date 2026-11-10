// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernel_operator.h"
#include "tla/tensor.hpp"
#include <type_traits>
#include "kernels/mega_moe_urma/catlass_gmm/mega_moe_grouped_matmul.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_route_views.hpp"

// Continuous routed matmul and paired-UB FIX output.

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

namespace MegaMoeUrma {

// One physical pipeline owns every L1/L0 slot and AIC/AIV0 credit for both
// projections. Updating the expert or stage must never reset those indices.
// Only GMM1 gathers A; both stages use one continuous Unit Flag output.
struct ContinuousComputeRows
    : UbxMegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false, RuntimeGroupSegments> {
    static constexpr bool kMayOutputToUb = true;
    bool gather = true;
    bool storeToUb = false;
    template <class TensorL0C>
    __aicore__ inline void StoreUb(const TensorL0C& source, uint32_t rows, uint32_t columns) const
    {
        UbxMegaMoeImpl::GmmKernel::StoreUb(source, rows, columns, 0b11);
    }

    template <class TensorL1>
    __aicore__ inline void Gather(TensorL1& dst, uint32_t tileBegin, uint32_t tileRows, uint32_t columnBegin,
                                  uint32_t columns) const
    {
        if (gather) {
            UbxMegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false, RuntimeGroupSegments>::Gather(
                dst, tileBegin, tileRows, columnBegin, columns);
            return;
        }
        AscendC::GlobalTensor<uint8_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(inputX));
        AscendC::Nd2NzParams params{};
        params.ndNum = 1;
        params.nValue = tileRows;
        params.dValue = columns;
        params.srcDValue = inputStride;
        params.dstNzC0Stride = tla::get<1, 1>(dst.stride()) / 32U;
        params.dstNzNStride = 1;
        AscendC::DataCopy(dst.data().template ReinterpretCast<uint8_t>(),
                          input[static_cast<uint64_t>(tileBegin) * inputStride + columnBegin], params);
    }
};

template <class Config>
class ContinuousMatmul
    : private UbxMegaMoeImpl::GmmKernel::detail::A8W4MatmulPipeline<Config, false, ContinuousComputeRows> {
    using Pipeline = UbxMegaMoeImpl::GmmKernel::detail::A8W4MatmulPipeline<Config, false, ContinuousComputeRows>;
    bool gateUp_ = false;
    bool fused_ = false;
    uint32_t halfN_ = 0U;

public:
    using Pipeline::Begin;
    using Pipeline::End;
    using Pipeline::SetRowIO;

    template <class Problem>
    __aicore__ inline void UpdateProblem(const Problem& config, const UbxMegaMoeImpl::GMMAddrInfo& addr, bool gmm1)
    {
        fused_ = gmm1 && Problem::FUSED_GATE_UP;
        Pipeline::UpdateProblem(config, addr, gmm1 ? addr.gmm1OutGlobal : addr.gmm2OutGlobal);
        gateUp_ = gmm1;
        halfN_ = config.outputN;
    }

    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t rows, uint32_t columns)
    {
        if (fused_) {
            // n/columns describe the logical gate (and up) width. The packed
            // weight columns and physical result contain both adjacent halves.
            Pipeline::ComputeProjection(m, n * 2U, rows, columns * 2U, n * 2U);
        } else {
            Pipeline::ComputeProjection(m, n, rows, columns, n);
            if (gateUp_) Pipeline::ComputeProjection(m, n + halfN_, rows, columns, n + halfN_);
        }
    }

    __aicore__ inline void Dequantize(uint32_t n, uint32_t rows, uint32_t columns)
    {
        if (fused_)
            Pipeline::DequantizeWeight(n * 2U, rows, columns * 2U);
        else {
            Pipeline::DequantizeWeight(n, rows, columns);
            if (gateUp_) Pipeline::DequantizeWeight(n + halfN_, rows, columns);
        }
    }
};

}  // namespace MegaMoeUrma
