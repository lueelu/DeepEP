/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP
#define MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP

#include "resource.hpp"
#include "a8w4_k_plan.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

// Based on CATLASS example 74. Only AIV0 produces the complete L1 B tile.
template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_ =
              Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct MegaMoeBlockMmadA8W4 {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementMxScaleA = typename TileCopy::ElementMxScaleA;
    using ElementMxScaleB = typename TileCopy::ElementMxScaleB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using ElementL0A = typename helper::GetL0Element<ElementA, true>::Element;
    using ElementL0B = typename helper::GetL0Element<ElementB, true>::Element;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL1MxScaleA = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1MxScaleB = typename TileCopy::LayoutTagL1MxScaleB;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
                  "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
                  "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    // Two 128-KiB A slots provide an independent kaL1 window.
    static constexpr uint32_t L1A_TILE_SIZE = 128U * 1024U;
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * SizeOfBits<ElementB>::value / 8;
    static constexpr uint32_t L1SCALEA_TILE_SIZE =
        L1_TILE_M * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleA) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1SCALEB_TILE_SIZE =
        L1_TILE_N * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleB) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1_USED_SIZE = L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES +
                                             L1SCALEA_TILE_SIZE * L1A_STAGES + L1SCALEB_TILE_SIZE * L1B_STAGES;
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * SizeOfBits<ElementL0A>::value / 8;
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * SizeOfBits<ElementL0B>::value / 8;
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static_assert(L1_TILE_M == 256 && L1_TILE_N == 256 && L1_TILE_K == 256 && L1A_STAGES == 2 && L1B_STAGES == 2 &&
                      L1_SCALE_FACTOR_K == 16 && SizeOfBits<ElementA>::value == 8 && SizeOfBits<ElementB>::value == 8 &&
                      !HAS_BIAS,
                  "Adaptive A8W4 requires two A128KiB/B64KiB/scale32KiB slots and no bias");

    // Check L0C_STAGES
    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                        std::is_same_v<ElementC, float>) &&
                       tla::detail::iszN<ElementC, LayoutC>::value),
                  "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape::K and L0TileShape::K
    static_assert(L1_TILE_K % MX_BASEK_FACTOR == 0 && L0_TILE_K % MX_BASEK_FACTOR == 0,
                  "L1TileShape::K and L0TileShape::K must be multiples of 64!");

    // Check L1TileShape
    static_assert(L1_USED_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    static_assert((!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7),
                  "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");

    static_assert((!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7),
                  "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1SCALEA_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
        tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{});
    static constexpr auto L1SCALEB_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
        tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // When enabling L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if tow blockmmad instances need to be consecutively invoked at the kernel layer,
    // RestoreStatus() must be inserted between them.
    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
        l1MxScaleAListId = 0;
        l1MxScaleBListId = 0;
    }

    /// Construct
    CATLASS_DEVICE
    MegaMoeBlockMmadA8W4(MegaMoeResource& resource, uint32_t l1BufAddrStart = 0, bool activate = true)
    {
        // L1 map: B=0/384KiB, A=128/256KiB, SA=64/448KiB, SB=96/480KiB.
        uint32_t l1Offset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + i * 384U * 1024U);
            l1Offset += L1B_TILE_SIZE;
            // Assign event ID for each stages
            l1BEventList[i] = i + L1A_STAGES;
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + 128U * 1024U + i * L1A_TILE_SIZE);
            l1Offset += L1A_TILE_SIZE;
            // Assign event ID for each stages
            l1AEventList[i] = i;
        }
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(L0A_TILE_SIZE * i);
            // Assign event ID for each stages
            l0AEventList[i] = i;
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(L0B_TILE_SIZE * i);
            //  Assign event ID for each stages
            l0BEventList[i] = i;  // A/B advance together and share one free token.
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                l0CEventList[i] = i;
            }
        } else {
            l0CTensorList[l0CListId] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * l0CListId);
        }
        if constexpr (HAS_BIAS) {
            l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1Offset);
            l1Offset += L1_TILE_N * sizeof(ElementBias);
            l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            l1MxScaleATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(
                l1BufAddrStart + 64U * 1024U + i * 384U * 1024U);
            l1Offset += L1SCALEA_TILE_SIZE;
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            l1MxScaleBTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(
                l1BufAddrStart + 96U * 1024U + i * 384U * 1024U);
            l1Offset += L1SCALEB_TILE_SIZE;
        }

        if (activate) {
            Activate();
        }
    }

    // Buffer views/event IDs are initialized once; stages sharing this hardware
    // still acquire/release the event tokens at each problem boundary.
    CATLASS_DEVICE
    void Activate()
    {
        if (active_) {
            return;
        }
        l1AListId = l1BListId = l0AListId = l0BListId = l0CListId = 0;
        RestoreStatus();
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
        if constexpr (HAS_BIAS) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
        active_ = true;
    }

    CATLASS_DEVICE
    void Deactivate()
    {
        if (!active_) {
            return;
        }
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(false);
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
        if constexpr (HAS_BIAS) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
        active_ = false;
    }

    CATLASS_DEVICE
    ~MegaMoeBlockMmadA8W4()
    {
        Deactivate();
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorC, class TensorMxScaleA = EmptyClass, class TensorMxScaleB = EmptyClass,
              class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorC& tensorC, GemmCoord const& actualShape,
                                   TensorMxScaleA const& tensorMxScaleA = {}, TensorMxScaleB const& tensorMxScaleB = {},
                                   bool allowDynamicB = false, TensorBias const& tensorBias = {})
    {
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE,
                          "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(L1_USED_SIZE + L1BIAS_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
        }

        // datacopy tile
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1MxScaleA = typename TileCopy::template CopyGmToL1MxScaleA<TensorMxScaleA>;
        using CopyGmToL1MxScaleB = typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1MxScaleA copyGmToL1MxScaleA;
        CopyGmToL1MxScaleB copyGmToL1MxScaleB;
        CopyL0CToDst copyL0CToDst;

        uint32_t l1ScaleTileK = L1_TILE_K * L1_SCALE_FACTOR_K;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        const auto kPlan = MegaMoeImpl::GmmKernel::MakeA8W4KPlan(mBlockActual, nBlockActual, allowDynamicB);
        const uint32_t ka = kPlan.ka;
        const uint32_t kb = kPlan.kb;
        const uint32_t kL1Loop = CeilDiv(kBlockActual, kb);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            const uint32_t kOffset = kL1Idx * kb;
            const uint32_t kL1Actual = Min(kb, kBlockActual - kOffset);
            const uint32_t aStart = kOffset / ka * ka;
            const uint32_t aActual = Min(ka, kBlockActual - aStart);
            // Scale buffers have their own lifetime: neither A nor B window release
            // permits overwriting scales that are still used by this 4096-K window.
            const uint32_t scaleStart = kOffset / l1ScaleTileK * l1ScaleTileK;
            const uint32_t scaleActual = Min(l1ScaleTileK, kBlockActual - scaleStart);
            const uint32_t scaleCount = CeilDiv<MX_BASEK_FACTOR>(scaleActual) * 2U;
            auto scaleLayoutA =
                tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(mL1Actual, scaleCount);
            auto scaleLayoutB =
                tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(scaleCount, nL1Actual);
            if (kOffset == scaleStart) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1MxScaleAListId]);
                auto tensorL1MxScaleA =
                    tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], scaleLayoutA, Arch::PositionL1{});
                auto tensorL1MxScaleB =
                    tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], scaleLayoutB, Arch::PositionL1{});
                auto tileScaleA = GetTile(tensorMxScaleA, tla::MakeCoord(0, scaleStart / MX_SCALE_GROUP_NUM),
                                          tla::MakeShape(mBlockActual, scaleCount));
                auto tileScaleB = GetTile(tensorMxScaleB, tla::MakeCoord(scaleStart / MX_SCALE_GROUP_NUM, 0),
                                          tla::MakeShape(scaleCount, nBlockActual));
                copyGmToL1MxScaleA(tensorL1MxScaleA, tileScaleA);
                copyGmToL1MxScaleB(tensorL1MxScaleB, tileScaleB);
            }

            // Compact runtime shape uses the enlarged A slot without padding M to 256.
            auto tensorL1A =
                tla::MakeTensor(l1ATensorList[l1AListId],
                                tla::MakeLayout<ElementA, LayoutTagL1A>(mL1Actual, RoundUp<MX_BASEK_FACTOR>(aActual)),
                                Arch::PositionL1{});
            if (kOffset == aStart) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, aStart), tla::MakeShape(mBlockActual, aActual));
                copyGmToL1A(tensorL1A, tensorTileA);
                InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, aActual));
            }

            auto tensorL1B = tla::MakeTensor(
                l1BTensorList[l1BListId], tla::MakeLayout<ElementB, LayoutTagL1B>(kb, nL1Actual), Arch::PositionL1{});
            // Only an incomplete 64-K MX group needs MTE2 padding after AIV0's store.
            // Aligned blocks wait on MTE1, leaving MTE2 free to prefetch A/scales.
            if (kL1Actual % MX_BASEK_FACTOR != 0U) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIV_SYNC_AIC_FLAG + l1BListId);
                InitZeroInL1B(tensorL1B, tla::MakeShape(kL1Actual, nL1Actual));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
            } else {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIV_SYNC_AIC_FLAG + l1BListId);
            }
            // One fence covers scale and A transfers at each K-window boundary.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            auto tensorL1MxScaleA =
                tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], scaleLayoutA, Arch::PositionL1{});
            auto tensorL1MxScaleB =
                tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], scaleLayoutB, Arch::PositionL1{});
            const uint32_t l0kOffset = kOffset - scaleStart;
            const uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                // mmad k must be multiples of 64 when mx type
                kL0Actual = RoundUp<MX_BASEK_FACTOR>(kL0Actual);

                // Locate the current tile on L0A
                auto layoutAInL0 = tla::MakeLayout<ElementL0A, LayoutTagL0A>(mL1Actual, kL0Actual);
                auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});

                // Locate the current tile of matrix A on L1
                auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, kOffset - aStart + kL0Idx * L0_TILE_K),
                                             tla::MakeShape(mL1Actual, kL0Actual));

                // Locate the current tile of matrix mxScaleA on L1
                auto tensorTileL1MxScaleA =
                    GetTile(tensorL1MxScaleA, tla::MakeCoord(0, (l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM),
                            tla::MakeShape(mL1Actual, CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual)));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                // Load current tile from L1 to L0A
                copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);

                // Locate the current tile on L0B
                auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL1Actual);
                auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});

                // Locate the current tile of matrix mxScaleB on L1
                auto tensorTileL1B =
                    GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, nL1Actual));

                // Locate the current tile of matrix mxScaleB on L1
                auto tensorTileL1MxScaleB =
                    GetTile(tensorL1MxScaleB, tla::MakeCoord((l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM, 0),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), nL1Actual));

                // Wait for mmad finished
                // The shared A/B slot was acquired before loading A.

                // Load current tile from L1 to L0B
                copyL1ToL0B(tensorL0B, tensorTileL1B, tensorTileL1MxScaleB);

                bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                    if (initC) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                        auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                        auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                        auto tensorTileL1Bias = GetTile(tensorL1Bias, tla::MakeCoord(0), tla::MakeShape(nL1Actual));
                        // Load bias to l0 biastable
                        copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                    }
                }

                // Notify to do mmad
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                // If the unit flag is enabled, the unit flag is set according to the calculation progress
                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    if ((kL1Idx == kL1Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        unitFlag = 0b11;
                    } else {
                        unitFlag = 0b10;
                    }
                }
                if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                    if (initC) {
                        tileMmad(tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, mL1Actual, nL1Actual, kL0Actual, initC,
                                 unitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                    } else {
                        tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                    }
                } else {
                    tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                }

                // Notify to move the next L0B tile
                // One M_MTE1 token releases both operands after MMAD.
                l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }

            const uint32_t nextK = kOffset + kL1Actual;
            if (nextK == kBlockActual || nextK % ka == 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                l1AListId = (l1AListId + 1) % L1A_STAGES;
            }
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + l1BListId);
            l1BListId = (l1BListId + 1) % L1B_STAGES;
            if (nextK == kBlockActual || nextK % l1ScaleTileK == 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1MxScaleAListId]);
                l1MxScaleAListId = (l1MxScaleAListId + 1) % L1A_STAGES;
                l1MxScaleBListId = (l1MxScaleBListId + 1) % L1B_STAGES;
            }
        }
        // copy block out
        copyL0CToDst.params.scale = 64;
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        } else {
            copyL0CToDst(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1A(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t mL1Actual = tla::get<0>(actualShape);
        const uint32_t kL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;

            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = mL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            } else {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(mL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<0, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(0, validKL1));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1B(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t kL1Actual = tla::get<0>(actualShape);
        const uint32_t nL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::isnZ<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;

            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(nL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<1, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            } else {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = nL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(validKL1, 0));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementMxScaleA> l1MxScaleATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementMxScaleB> l1MxScaleBTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementL0A> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementL0B> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    bool active_ = false;

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    uint32_t l1MxScaleAListId{0};  // scale id only used in Mmad3
    uint32_t l1MxScaleBListId{0};

    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Block

#endif  // MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP
