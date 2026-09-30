/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_CATLASS_BLOCK_PROLOGUE_A8W4_HPP
#define MEGA_MOE_CATLASS_BLOCK_PROLOGUE_A8W4_HPP

#include "resource.hpp"
#include "a8w4_k_plan.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {
template <class ArchTag, uint32_t L1B_STAGES_, class InType_, class OutType_, class TileShapeL1_, class TileCopy_>
struct MegaMoeBlockPrologueA8W4 {
public:
    using DispatchPolicy = MxA8W4Prologue<ArchTag, L1B_STAGES_>;
    using ElementIn = typename InType_::Element;
    using ElementOut = typename OutType_::Element;
    using LayoutIn = typename InType_::Layout;
    using LayoutOut = typename OutType_::Layout;
    using TileShapeL1 = TileShapeL1_;
    using TileCopy = TileCopy_;
    using LayoutPrologueB = typename TileCopy::LayoutPrologueB;

    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;

    struct Params {
        TileShapeL1 tileShapeL1;
        LayoutPrologueB layoutPrologueB;
        bool hasBias;
        MegaMoeResource& resource;
    };

    struct VfParamsNz {
        uint32_t loopKNum;
        uint32_t innerLoopNum;
        uint32_t loopKDstStride;
        uint32_t innerDstStride;
        uint32_t nRealSizeAlign;
        __ubuf__ ElementIn* weightInUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr;
    };

    static_assert(std::is_same_v<LayoutIn, layout::Weight4BitnZ>, "MegaMoE requires packed NZ weights");
    static_assert(tla::get<1>(TileShapeL1{}) * tla::get<2>(TileShapeL1{}) <= 64 * 1024,
                  "Single-AIV weight output exceeds its UB stage");
    static_assert(L1B_STAGES == 2, "MegaMoE uses two weight stages");

    CATLASS_DEVICE
    MegaMoeBlockPrologueA8W4(const Params& params, bool activate = true)
    {
        // Persistent GMM1 contexts also exist on AIV1, which owns no prologue resources.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList[i] = params.resource.l1Buf.template GetBufferByByte<ElementOut>(i * 384U * 1024U);
        }
        // Packed inputs occupy [0,64KiB); the four
        // output slots interleave 256-byte vectors throughout [64,192KiB).
        for (uint32_t i = 0; i < UB_STAGES; i++) {
            ubCastInTensor[i] = params.resource.ubBuf.template GetBufferByByte<ElementIn>(i * 16U * 1024U);
            ubCastOutTensor[i] = params.resource.ubBuf.template GetBufferByByte<ElementOut>(64U * 1024U + i * 256U);
            ubEventList[i] = i;
        }
        if (activate) {
            Activate();
        }
    }

    CATLASS_DEVICE
    void Activate()
    {
        if (active_ || AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        l1BListId = ubListId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
        }
        active_ = true;
    }

    CATLASS_DEVICE
    void Deactivate()
    {
        if (!active_) {
            return;
        }
        for (uint32_t i = 0; i < UB_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
        }
        active_ = false;
    }

    CATLASS_DEVICE
    ~MegaMoeBlockPrologueA8W4()
    {
        Deactivate();
    }

    template <class TensorBIn, class ActualBlockShape>
    CATLASS_DEVICE void operator()(const TensorBIn& bGlobal, const ActualBlockShape& actualBlockShape,
                                   const Params& params, bool allowDynamicB = false)
    {
        uint32_t kSize = tla::get<0>(params.layoutPrologueB.originShape());
        const auto kPlan =
            MegaMoeImpl::GmmKernel::MakeA8W4KPlan(actualBlockShape.m(), actualBlockShape.n(), allowDynamicB);
        const uint32_t kL1Size = kPlan.kb;
        const uint32_t halfK = kL1Size / 2U;
        const uint32_t kGmLoop = CeilDiv(kSize, kL1Size);

        // AIV1 runs dispatch/epilogue concurrently and must never enter this pipeline.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (uint32_t kLoopIdx = 0; kLoopIdx < kGmLoop; ++kLoopIdx) {
            uint32_t actualK = Min(kL1Size, kSize - kLoopIdx * kL1Size);
            auto layout = tla::MakeLayout<ElementOut, LayoutOut>(kL1Size, actualBlockShape.n());
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], layout, Arch::PositionL1{});
            for (uint32_t offset = 0; offset < actualK; offset += halfK) {
                const uint32_t count = Min(halfK, actualK - offset);
                auto gmTile = GetTile(bGlobal, tla::MakeCoord(kLoopIdx * kL1Size + offset, 0),
                                      tla::MakeShape(count, actualBlockShape.n()));
                auto l1Tile =
                    GetTile(tensorL1B, tla::MakeCoord(offset, 0), tla::MakeShape(count, actualBlockShape.n()));
                ProcessL1Nz(l1Tile, gmTile, offset == 0U);
                ubListId = (ubListId + 1U) % UB_STAGES;
            }
            // Publish only after both halves are stored in the same L1 B slot.
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + l1BListId);
            l1BListId = (l1BListId + 1) % L1B_STAGES;
        }
    }

    template <class TensorOut, class TensorIn>
    __aicore__ inline void ProcessL1Nz(const TensorOut& tensorOut, const TensorIn& tensorIn, bool firstHalf)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[ubListId]);
        uint32_t shapeN;
        uint32_t shapeK;

        shapeN = tla::get<1, 0>(tensorIn.shape()) * tla::get<1, 1>(tensorIn.shape());  // nZ
        shapeK = tla::get<0, 0>(tensorIn.shape()) * tla::get<0, 1>(tensorIn.shape());

        auto layoutCastIn = tla::MakeLayout<ElementOut, layout::Weight4BitnZ>(shapeK, shapeN);
        auto layoutCastOut = tla::MakeLayout<ElementOut, layout::nZ>(shapeK, shapeN);

        auto tensorCastIn = tla::MakeTensor(ubCastInTensor[ubListId], layoutCastIn, Catlass::Arch::PositionUB{});
        auto tensorCastOut = tla::MakeTensor(ubCastOutTensor[ubListId], layoutCastOut, Catlass::Arch::PositionUB{});

        CopyNzInWeightTensor(tensorCastIn, tensorIn);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[ubListId]);

        if (firstHalf) {
            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + l1BListId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[ubListId]);

        AntiQuantComputeNz(tensorCastOut, tensorCastIn);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[ubListId]);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[ubListId]);

        CopyUb2L1(tensorOut, tensorCastOut);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[ubListId]);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyNzInWeightTensor(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        uint32_t blockLen = tla::get<0, 1>(dstTensor.stride()) >> 1;
        AscendC::DataCopyExtParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = (tla::get<0, 1>(srcTensor.stride()) - tla::get<0, 1>(dstTensor.stride())) >> 1;
        repeatParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPadExtParams<typename TensorDst::Element> padParams;
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams, padParams);
    }

    template <class TensorCastOut, class TensorCastIn>
    __aicore__ inline void AntiQuantComputeNz(const TensorCastOut& tensorCastOut, const TensorCastIn& tensorCastIn)
    {
        VfParamsNz params;
        params.weightInUbAddr = (__ubuf__ ElementIn*)tensorCastIn.data().GetPhyAddr();
        params.weightOutUbAddr = (__ubuf__ ElementOut*)tensorCastOut.data().GetPhyAddr();

        params.loopKNum = tla::get<0, 1>(tensorCastIn.shape());
        params.nRealSizeAlign = tla::get<1, 1>(tensorCastIn.shape()) * AscendC::BLOCK_CUBE;
        params.innerDstStride = AscendC::GetVecLen() * UB_STAGES;
        params.innerLoopNum = (params.nRealSizeAlign * tla::get<0, 0>(tensorCastIn.shape())) /
                              static_cast<uint64_t>(AscendC::GetVecLen());
        params.loopKDstStride = params.innerLoopNum * params.innerDstStride;

        RegComputeNz<TensorCastOut, TensorCastIn>(params);
    }

    template <class TensorCastOut, class TensorCastIn>
    CATLASS_DEVICE_SIMD_VF void RegComputeNz(VfParamsNz params)
    {
        AscendC::Reg::RegTensor<int8_t> wShrReg;
        AscendC::Reg::RegTensor<int8_t> wShlReg;
        AscendC::Reg::RegTensor<int8_t> wAndReg;
        AscendC::Reg::RegTensor<int8_t> wLoad;
        AscendC::Reg::RegTensor<int8_t> wShl;
        AscendC::Reg::RegTensor<int8_t> wShr0;
        AscendC::Reg::RegTensor<int8_t> wShr1;
        AscendC::Reg::RegTensor<int8_t> wSel;
        AscendC::Reg::RegTensor<int8_t> wAnd;

        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

        for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
            for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
                // DIST_US_B8 load mode expands each packed B4 byte into lane-aligned B8 slots.
                // Packed B4 address offset (bytes) = logical element index >> 1.
                AscendC::Reg::AddrReg aregWeightB8In = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, (C0_SIZE_B8 * params.nRealSizeAlign) >> 1, innerLoopIdx, AscendC::GetVecLen() >> 1);
                AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
                    (AscendC::Reg::RegTensor<uint8_t>&)wLoad, (__ubuf__ uint8_t*&)params.weightInUbAddr,
                    aregWeightB8In);

                AscendC::Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
                AscendC::Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
                AscendC::Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
                AscendC::Reg::Select(wSel, wShr1, wShr0, pregVsel);
                AscendC::Reg::And(wAnd, wSel, wAndReg, preg);

                AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
                AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    (__ubuf__ uint8_t*&)params.weightOutUbAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd,
                    aregWeightB8Out, preg);
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyUb2L1(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        // VF stores one 256-byte vector every 1024 bytes. Gather this slot's
        // vectors into contiguous NZ L1, skipping the other three UB slots.
        constexpr uint32_t VECTOR_BYTES = 256U;
        constexpr uint32_t BLOCK_BYTES = 32U;
        const uint32_t elements = tla::get<0, 1>(srcTensor.shape()) * C0_SIZE_B8 * tla::get<1, 0>(srcTensor.shape()) *
                                  tla::get<1, 1>(srcTensor.shape());
        AscendC::DataCopyParams dataCopyParams(elements / VECTOR_BYTES, VECTOR_BYTES / BLOCK_BYTES,
                                               (UB_STAGES - 1U) * VECTOR_BYTES / BLOCK_BYTES, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data(), dataCopyParams);
    }

    static constexpr int32_t C0_SIZE_B8 = 32;

    static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
    static constexpr uint32_t E2M1_AND_MASK = 0x9C;
    static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;

    static constexpr int32_t SYNC_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;

    static constexpr int64_t UB_STAGES = 4;

    bool active_ = false;
    uint32_t l1BListId{0};
    uint32_t ubListId{0};
    int32_t ubEventList[UB_STAGES];

    AscendC::LocalTensor<ElementIn> ubCastInTensor[UB_STAGES];
    AscendC::LocalTensor<ElementOut> ubCastOutTensor[UB_STAGES];
    AscendC::LocalTensor<ElementOut> l1BTensorList[L1B_STAGES];
};
}  // namespace Catlass::Gemm::Block

#endif
