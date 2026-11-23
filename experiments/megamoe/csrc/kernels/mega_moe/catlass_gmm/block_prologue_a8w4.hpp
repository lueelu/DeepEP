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

#include "kernels/mega_moe_common/catlass_gmm/resource.hpp"
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
struct PackedNzFp4 {
    static constexpr uint32_t UB_STAGES = 4;

    __aicore__ static inline void Decode(__ubuf__ uint8_t* in, __ubuf__ uint8_t* out, uint16_t vectors)
    {
#ifdef DEEPEP_MEGAMOE_ROCE
        if (vectors > 1U && vectors % 2U == 0U) {
            RegComputeNz<true>(in, out, vectors / 2U);
            return;
        }
#endif
        RegComputeNz<false>(in, out, vectors);
    }

    template <class Element>
    __aicore__ static inline void StoreL1(const AscendC::LocalTensor<Element>& dst,
                                         const AscendC::LocalTensor<Element>& src, uint16_t vectors)
    {
        constexpr uint16_t blocks = AscendC::GetVecLen() / 32U;
        AscendC::DataCopyParams gather(vectors, blocks, (UB_STAGES - 1U) * blocks, 0);
        AscendC::DataCopy(dst, src, gather);
    }

private:
    template <bool Pair>
    CATLASS_DEVICE_SIMD_VF void RegComputeNz(__ubuf__ uint8_t* in, __ubuf__ uint8_t* out, uint16_t iterations)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t inStep = AscendC::GetVecLen() / 2U;
        constexpr uint32_t outStep = AscendC::GetVecLen() * UB_STAGES;
        // For a single vector in slot 3, even forming out + outStep can leave the slot.
        auto in1 = in + (Pair ? inStep : 0U);
        auto out1 = out + (Pair ? outStep : 0U);
        auto all = CreateMask<uint8_t, MaskPattern::ALL>();
        auto alternate = CreateMask<uint16_t, MaskPattern::ALL>();
        RegTensor<int8_t> four, zero, shifts, right, bits, a, b;
        Duplicate<int8_t, MaskMergeMode::ZEROING>(four, 4, all);
        Duplicate<int8_t, MaskMergeMode::ZEROING>(zero, 0, all);
        Duplicate<int8_t, MaskMergeMode::ZEROING>(right, 2, all);
        Duplicate<int8_t, MaskMergeMode::ZEROING>(bits, static_cast<int8_t>(0x9C), all);
        Select(shifts, four, zero, alternate);
        for (uint16_t i = 0; i < iterations; ++i) {
            auto src = CreateAddrReg<uint8_t>(i, (Pair ? 2U : 1U) * inStep);
            auto dst = CreateAddrReg<uint8_t>(i, (Pair ? 2U : 1U) * outStep);
            LoadAlign<uint8_t, LoadDist::DIST_US_B8>(reinterpret_cast<RegTensor<uint8_t>&>(a), in, src);
            if constexpr (Pair) {
                LoadAlign<uint8_t, LoadDist::DIST_US_B8>(reinterpret_cast<RegTensor<uint8_t>&>(b), in1, src);
            }
            ShiftLeft(a, a, shifts, all);
            if constexpr (Pair) ShiftLeft(b, b, shifts, all);
            ShiftRight(a, a, right, all);
            if constexpr (Pair) ShiftRight(b, b, right, all);
            And(a, a, bits, all);
            if constexpr (Pair) And(b, b, bits, all);
            StoreAlign<uint8_t, StoreDist::DIST_NORM_B8>(out, reinterpret_cast<RegTensor<uint8_t>&>(a), dst, all);
            if constexpr (Pair) {
                StoreAlign<uint8_t, StoreDist::DIST_NORM_B8>(out1, reinterpret_cast<RegTensor<uint8_t>&>(b), dst, all);
            }
        }
    }
};

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
#ifdef DEEPEP_MEGAMOE_ROCE
            l1BTensorList[i] = params.resource.l1Buf.template GetBufferByByte<ElementOut>(i * 256U * 1024U);
#else
            l1BTensorList[i] = params.resource.l1Buf.template GetBufferByByte<ElementOut>(i * 384U * 1024U);
#endif
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
#ifdef DEEPEP_MEGAMOE_ROCE
        const uint32_t n16 = CeilDiv(static_cast<uint32_t>(actualBlockShape.n()), 16U) * 16U;
        const uint32_t chunkK = Min(kL1Size / 2U, 64U * (32768U / (64U * n16)));
#else
        const uint32_t halfK = kL1Size / 2U;
#endif
        const uint32_t kGmLoop = CeilDiv(kSize, kL1Size);

        // AIV1 runs dispatch/epilogue concurrently and must never enter this pipeline.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        for (uint32_t kLoopIdx = 0; kLoopIdx < kGmLoop; ++kLoopIdx) {
            uint32_t actualK = Min(kL1Size, kSize - kLoopIdx * kL1Size);
            auto layout = tla::MakeLayout<ElementOut, LayoutOut>(kL1Size, actualBlockShape.n());
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], layout, Arch::PositionL1{});
#ifdef DEEPEP_MEGAMOE_ROCE
            for (uint32_t offset = 0; offset < actualK; offset += chunkK) {
                const uint32_t count = Min(chunkK, actualK - offset);
                auto gmTile = GetTile(bGlobal, tla::MakeCoord(kLoopIdx * kL1Size + offset, 0),
                                      tla::MakeShape(count, actualBlockShape.n()));
                auto l1Tile =
                    GetTile(tensorL1B, tla::MakeCoord(offset, 0), tla::MakeShape(count, actualBlockShape.n()));
                ProcessL1Nz(l1Tile, gmTile, offset == 0U);
                ubListId = (ubListId + 1U) % UB_STAGES;
            }
            // Publish only after all chunks are stored in the same L1 B slot.
#else
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
#endif
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

        auto tensorCastIn = tla::MakeTensor(ubCastInTensor[ubListId], layoutCastIn, Catlass::Arch::PositionUB{});

        CopyNzInWeightTensor(tensorCastIn, tensorIn);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[ubListId]);

        if (firstHalf) {
            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + l1BListId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[ubListId]);

        const auto& shape = tensorCastIn.shape();
        const uint16_t vectors = tla::get<0, 0>(shape) * tla::get<0, 1>(shape) * tla::get<1, 0>(shape) *
                                 tla::get<1, 1>(shape) / AscendC::GetVecLen();
        PackedNzFp4::Decode(reinterpret_cast<__ubuf__ uint8_t*>(tensorCastIn.data().GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint8_t*>(ubCastOutTensor[ubListId].GetPhyAddr()), vectors);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[ubListId]);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[ubListId]);

        auto dstOffset = tensorOut.layout()(tensorOut.coord());
        PackedNzFp4::StoreL1(tensorOut.data()[dstOffset], ubCastOutTensor[ubListId], vectors);

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

    static constexpr int32_t SYNC_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;

    static constexpr uint32_t UB_STAGES = PackedNzFp4::UB_STAGES;

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
