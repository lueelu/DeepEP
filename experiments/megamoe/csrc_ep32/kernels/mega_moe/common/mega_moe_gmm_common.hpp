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

#ifndef MEGA_MOE_GMM_COMMON_H
#define MEGA_MOE_GMM_COMMON_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "mega_moe_constants.hpp"
#include "mega_moe_types.hpp"
#include "mega_moe_utils.hpp"
#include "../catlass_gmm/mega_moe_mx_matmul.hpp"
#include "../catlass_gmm/mega_moe_grouped_matmul.hpp"
#include "mega_moe_gmm_utils.hpp"
#include "mega_moe_block_scheduler.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

namespace GmmKernel {

constexpr uint32_t L1_TILE_K = 256U;

// 根据数据类型路径选择对应的 BlockMmad 实现。
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type = CatlassMxMatmul<C>;
};

// A8W4 execution uses CatlassGmm1/Gmm2 directly; only tiling metadata is shared.
struct A8W4BlockDescriptor {
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
    };
};
template <typename C>
struct BlockMmadSelector<true, C> {
    using type = A8W4BlockDescriptor;
};

// 汇总 GMM1/GMM2 在不同量化路径下共用的类型、shape 和 layout 配置。
template <bool IsA8W4, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          bool FuseGateUp = false>
struct Config {
    static_assert(!FuseGateUp || (IsA8W4 && !IsShared && !IsLayered && !IsGmm1Interleaved),
                  "Fused gate/up is restricted to routed local W4A8");
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool IS_WEIGHT_NZ = IsWeightNZ;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;
    using BiasType = float;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;
    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_BIAS = AuxGetC0Size<BiasType>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutB = Std::conditional_t<IsA8W4, Te::ZNLayoutPtn,
                                       Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;
    using MakeLayoutBias = Te::FrameLayoutFormat<LayoutBias, Std::Int<C0_SIZE_BIAS>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(MakeLayoutBias{}(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;
    struct BlockMmadTilingConfig {
        uint32_t tileM;
        uint32_t tileN;
        typename BlockMmad::L1Params l1Params;
    };

    static __aicore__ inline BlockMmadTilingConfig SelectBlockMmadTilingConfig(uint32_t, uint32_t, uint32_t tileM)
    {
        // A8W4 selects kaL1/kbL1 from the actual slice M/N inside the CATLASS blocks.
        // Generic MX uses the source adapter's fixed K256 / scale-K512 plan.
        return {tileM, L1_TILE_N, {L1_TILE_K, IsA8W4 ? 4096U : 512U}};
    }

    struct ProblemConfig {
        using KernelConfig = Config;
        static constexpr bool SOURCE_GMM1_INTERLEAVED = IsGmm1Interleaved;
        static constexpr bool IS_WAVE_FLAG_GRAINED = IsWaveFlagGrained;
        static constexpr bool FUSED_GATE_UP = FuseGateUp;
        static constexpr uint32_t GMM1_TILE_N = FuseGateUp ? GMM1_FUSED_TILE_N : L1_TILE_N;

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t schedulerN = 0;
        uint32_t tileN = L1_TILE_N;
        SharedExpertPolicy sharedPolicy{};
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        uint32_t activationTileM = L1_TILE_M_256;
        BlockMmadTilingConfig blockMmadTiling{};
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline void FinalizeProblemConfig(ProblemConfig& config, const BlockJobContext& blockJob,
                                                        uint32_t gmm1TileM)
    {
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
        config.sharedPolicy = SelectSharedExpertPolicy(IsA8W4 && !IsLayered, IsShared ? 1U : 0U, config.m);
        config.blockMmadTiling = SelectBlockMmadTilingConfig(config.m, config.k, config.tileM);
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape& problemShape,
                                                                  const BlockJobContext& blockJob, uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<N_VALUE>(problemShape);
        config.k = Get<K_VALUE>(problemShape);
        config.outputN = config.n / ACTIVATION_N_HALF;
        config.tileM = gmm1TileM;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        config.schedulerN = (IsGmm1Interleaved || config.sharedPolicy.decode) ? config.n : config.outputN;
        // Fused routed tasks are logical N128 with physical N256 MMAD/prologue.
        config.tileN = FuseGateUp ? GMM1_FUSED_TILE_N : config.sharedPolicy.gmm1TileN;
        config.blockMmadTiling.tileN = FuseGateUp ? L1_TILE_N : config.tileN;
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape& problemShape,
                                                                  const BlockJobContext& blockJob, uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<K_VALUE>(problemShape);
        config.k = Get<N_VALUE>(problemShape) / ACTIVATION_N_HALF;
        config.outputN = config.n;
        config.schedulerN = config.outputN;
        config.tileM = L1_TILE_M_256;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        config.tileN = config.sharedPolicy.gmm2TileN;
        config.blockMmadTiling.tileN = config.tileN;
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig& config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        layouts.c = MakeLayoutC{}(config.m, config.n);
        layouts.bias = MakeLayoutBias{}(1U, config.n);
        return layouts;
    }
};

// 统一描述一次 BlockMmad::Init 的可变配置，避免 Context 分散保存并逐字段比较。
// 通用 MX BlockMmad 的 l1BufNum 始终固定为 2，A8W4 L1Params 则没有该字段，因此不纳入运行时状态。
struct BlockMmadInitConfig {
    uint32_t problemK = 0U;
    uint32_t problemN = 0U;
    uint32_t tileM = 0U;
    uint32_t tileN = 0U;
    uint64_t kL1 = 0U;
    uint64_t scaleKL1 = 0U;

    __aicore__ inline bool operator!=(const BlockMmadInitConfig& other) const
    {
        return problemK != other.problemK || problemN != other.problemN || tileM != other.tileM ||
               tileN != other.tileN || kL1 != other.kL1 || scaleKL1 != other.scaleKL1;
    }
};

// Wave 流程可持有同一个 BlockMmad；普通路径也复用该结构完成一致的初始化。
template <typename BlockMmad>
struct BlockMmadContext {
    BlockMmad blockMmad;
    BlockMmadInitConfig initConfig{};
    bool initialized = false;
};

// N 是 CATLASS GM 父矩阵的行距；改变 N 时也必须刷新 Init，实际 tile M/N 由每次 MMAD 传入。
template <typename MmadContext, typename ProblemConfig>
__aicore__ inline void InitBlockMmad(MmadContext& context, const ProblemConfig& config)
{
    using BlockMmad = decltype(context.blockMmad);
    const auto& tilingConfig = config.blockMmadTiling;
    const BlockMmadInitConfig requestedConfig{config.k,
                                              config.n,
                                              tilingConfig.tileM,
                                              tilingConfig.tileN,
                                              tilingConfig.l1Params.kL1,
                                              tilingConfig.l1Params.scaleKL1};
    if (!context.initialized || context.initConfig != requestedConfig) {
        typename BlockMmad::BlockShape l0TileShape{tilingConfig.tileM, tilingConfig.tileN, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        constexpr bool enableL0CPingPong = false;
        context.blockMmad.Init(matmulShape, l0TileShape, tilingConfig.l1Params, false, enableL0CPingPong);
        context.initConfig = requestedConfig;
        context.initialized = true;
    }
}

/*
 * CACHE_MODE_DISABLE 会使读取不填入 L2，仅适用于后续不会复用的数据。
 * 当前 problem 必须覆盖完整专家且 M 方向只有一个 tile，B/scaleB 才不会被后续 M tile 或 Wave 再读。
 * 热点专家被切到多个 Wave 时必须保留正常 L2 cache。
 * B 与 scaleB 的物理行跨度不同，二者必须独立判定 128B cache-line 对齐，不能共用一个结果。
 */
template <bool IsWeightNz, typename MatmulConfig, typename TensorB, typename TensorScaleB>
__aicore__ inline void SetWaveWeightL2CacheHint(const typename MatmulConfig::ProblemConfig& config,
                                                bool coversWholeExpert, TensorB& gmB, TensorScaleB& gmScaleB)
{
    constexpr uint64_t cacheLineBytes = 128U;
    // coversWholeExpert 防止跨 Wave 复用；m <= tileM 防止同一 problem 内跨 M tile 复用。
    bool hasNoLaterWeightReuse = coversWholeExpert && config.m <= config.tileM;

    /*
     * NZ 已由分形布局保证物理块对齐。ND/DN(transB) 的连续维是 K，每一行的起点只有在
     * K 方向物理字节数为 128B 整数倍时才允许绕过 L2。统一要求 256 个 logical element：
     * packed FP4 恰为 128B，FP8 为 256B；对 FP8 比最低 128-element 要求更保守，但不会误开 bypass。
     */
    bool bypassWeightL2 = hasNoLaterWeightReuse;
    if constexpr (!IsWeightNz) {
        bypassWeightL2 = bypassWeightL2 && config.k % 256U == 0U;
    }
    gmB.SetL2CacheHint(bypassWeightL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);

    /*
     * ScaleBDN 的连续维是 N，每个 logical N 含 C0_SIZE_SCALE 个 scale。这里检查完整 N 行跨度；
     * tile-N 固定为 256，其 tile 行跨度天然满足 128B 对齐，因此无需再做动态 baseN 检查。
     */
    constexpr uint64_t scaleTileNStrideBytes = static_cast<uint64_t>(L1_TILE_N) * MatmulConfig::C0_SIZE_SCALE *
                                               sizeof(typename MatmulConfig::ElementMxScaleBType);
    static_assert(scaleTileNStrideBytes % cacheLineBytes == 0U, "ScaleB tile-N stride must be cache-line aligned");
    uint64_t scaleNStrideBytes = static_cast<uint64_t>(config.n) * MatmulConfig::C0_SIZE_SCALE *
                                 sizeof(typename MatmulConfig::ElementMxScaleBType);
    bool bypassScaleL2 = hasNoLaterWeightReuse && scaleNStrideBytes % cacheLineBytes == 0U;
    gmScaleB.SetL2CacheHint(bypassScaleL2 ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
}

// 保存 GMM 执行所需的全部 tensor；当 bias 或 C 无实际存储时，调用方传入零地址占位 tensor。
template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
struct GroupMatmulWorkSet {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorBias& gmBias;
    TensorC& gmC;
};

}  // namespace GmmKernel
}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_GMM_COMMON_H
