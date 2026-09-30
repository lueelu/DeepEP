/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#ifndef DEEPEP_EP_GMM_FUSED_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_HPP
#define DEEPEP_EP_GMM_FUSED_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_HPP

#include "../../tiling_validation.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_mte_udma.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_udma_2d.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp"

namespace deepep::ep_gmm_fused {

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, uint32_t M0,
          uint32_t N0, uint32_t K0, bool ExportAllToAllV = false, uint32_t WorkspaceStages = 2, uint32_t MSplit = 1>
struct AllToAllVGroupedMatMulUdmaConfig {
    static_assert(WorkspaceStages == 2 || WorkspaceStages == 3,
                  "Ascend950 BF16 UDMA MoE supports two or three workspace stages");
    using ArchTag = Catlass::Arch::Ascend950;
    static_assert(ArchTag::UB_SIZE == TilingValidation::kCompiledUbBytes, "Update the Host and Python UB budgets");
    static constexpr uint32_t kM0 = M0;
    static constexpr uint32_t kN0 = N0;
    static constexpr uint32_t kK0 = K0;
    static constexpr uint32_t kWorkspaceStages = WorkspaceStages;
    static constexpr uint32_t kMSplit = MSplit;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    static constexpr bool kIsDynamic = true;
    static constexpr uint32_t kUbStages = 2;
    using TileLocalCopy =
        Catccos::Comm::Tile::TileRemoteCopy<ArchTag, kIsDynamic, AType, AType, void, Catccos::detail::CopyDirect::Put,
                                            Catccos::detail::CopyTransport::Mte>;
    using TileExportCopy =
        Catccos::Comm::Tile::TileRemoteCopy<ArchTag, kIsDynamic, AType, AType, void, Catccos::detail::CopyDirect::Put,
                                            Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopy =
        Catccos::Comm::Tile::TileRemoteCopyUdma2D<ArchTag, AType, AType, Catccos::detail::CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using LocalDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, kUbStages, kIsDynamic>;
    using RemoteDispatch = Catccos::Comm::AtlasCommUdmaRemoteCopy<ArchTag, kUbStages>;
    using BlockLocalComm =
        Catccos::Comm::Block::CommBlock<LocalDispatch, AType, AType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Catccos::Comm::Block::CommBlock<RemoteDispatch, AType, AType, TileRemoteCopy>;
    using BlockComm = Catccos::Comm::Block::CommBlockMteUdma<BlockLocalComm, BlockRemoteComm>;
    using ExportDispatch = Catccos::Comm::AtlasCommLocalCopy<ArchTag, kUbStages, kIsDynamic>;
    using BlockExport =
        Catccos::Comm::Block::CommBlock<ExportDispatch, AType, AType, void, TileExportCopy, TileScheduler>;

    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, 64>;
    using BlockMmadScheduler = Catccos::DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, MSplit>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel =
        Catccos::DGemm::Kernel::Ascend950AllToAllVGroupedMatmul<ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler,
                                                                BlockScheduler, WorkspaceStages, true, ExportAllToAllV,
                                                                BlockExport>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <uint32_t M0, uint32_t N0, uint32_t K0, bool ExportAllToAllV = false, uint32_t WorkspaceStages = 2,
          uint32_t MSplit = 1>
using Bf16AllToAllVGroupedMatMulUdmaConfig =
    AllToAllVGroupedMatMulUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, M0, N0, K0, ExportAllToAllV,
                                     WorkspaceStages, MSplit>;

using AllToAllVGroupedMatMulUdmaDefaultConfig =
    AllToAllVGroupedMatMulUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 128, 256, 256>;

using AllToAllVGroupedMatMulUdmaM0_256Config =
    AllToAllVGroupedMatMulUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 256, 128, 256>;

using AllToAllVGroupedMatMulUdmaDefaultExportConfig =
    AllToAllVGroupedMatMulUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 128, 256, 256, true>;

using AllToAllVGroupedMatMulUdmaM0_256ExportConfig =
    AllToAllVGroupedMatMulUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 256, 128, 256, true>;

}  // namespace deepep::ep_gmm_fused

#endif  // DEEPEP_EP_GMM_FUSED_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_HPP
