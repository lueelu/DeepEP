/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#ifndef DEEPEP_EP_GMM_FUSED_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_HPP
#define DEEPEP_EP_GMM_FUSED_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_HPP

#include "info.h"
#include "../../tiling_validation.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_mte_udma.hpp"
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_udma_2d.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_tla.hpp"

namespace deepep::ep_gmm_fused {

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, uint32_t M0,
          uint32_t N0, uint32_t K0>
struct GroupedMatMulAllToAllVUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    static_assert(ArchTag::UB_SIZE == TilingValidation::kCompiledUbBytes, "Update the Host and Python UB budgets");
    static constexpr uint32_t kM0 = M0;
    static constexpr uint32_t kN0 = N0;
    static constexpr uint32_t kK0 = K0;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, 32>;
    static constexpr bool kIsDynamic = true;
    static constexpr uint32_t kUbStages = 2;
    static constexpr uint32_t kWorkspaceStages = 2;
    static constexpr uint64_t kSymmetricDataBytes = 200UL * 1024 * 1024 * sizeof(uint16_t);
    using BlockMmadScheduler = Catccos::DGemm::Block::BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
    using BlockCommScheduler =
        Catccos::CommEpilogue::Block::BlockCommSchedulerGmmAllToAllVTargetOrder<MoeConstraints, kIsDynamic>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using TileLocalCopy =
        Catccos::Comm::Tile::TileRemoteCopy<ArchTag, kIsDynamic, CType, CType, void, Catccos::detail::CopyDirect::Get,
                                            Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopy =
        Catccos::Comm::Tile::TileRemoteCopyUdma2D<ArchTag, CType, CType, Catccos::detail::CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using LocalDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, kUbStages, kIsDynamic>;
    using RemoteDispatch = Catccos::Comm::AtlasCommUdmaRemoteCopy<ArchTag, kUbStages>;
    using BlockLocalComm =
        Catccos::Comm::Block::CommBlock<LocalDispatch, CType, CType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Catccos::Comm::Block::CommBlock<RemoteDispatch, CType, CType, TileRemoteCopy>;
    using BlockComm = Catccos::Comm::Block::CommBlockMteUdma<BlockLocalComm, BlockRemoteComm>;

    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel = Catccos::DGemm::Kernel::GroupedMatmulAllToAllVTla<ProblemShape, BlockMmad, BlockComm,
                                                                     BlockMmadScheduler, BlockCommScheduler,
                                                                     kWorkspaceStages, kSymmetricDataBytes, true>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

using GroupedMatMulAllToAllVUdmaDefaultConfig =
    GroupedMatMulAllToAllVUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 128, 256, 256>;

using GroupedMatMulAllToAllVUdmaM0_256Config =
    GroupedMatMulAllToAllVUdmaConfig<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
                                     bfloat16_t, Catlass::layout::RowMajor, 256, 128, 256>;

}  // namespace deepep::ep_gmm_fused

#endif  // DEEPEP_EP_GMM_FUSED_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_HPP
