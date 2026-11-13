// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#include "checksum.h"
#include "comm/block/comm_block_hierarchical_copy.hpp"
#include "comm/block/comm_block_hierarchical_swizzle.hpp"
#include "comm/tile/tile_mte_copy.hpp"
#include "comm/kernel/hierarchical_alltoall.hpp"
#include "comm/kernel/hierarchical_barrier.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace deepep::hierarchical {
using ArchTag = Catlass::Arch::Ascend950;
using TileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
using Scheduler = comm::block::HierarchicalCommSwizzle<kCoreCount, HIERARCHICAL_GROUP_SIZE, HIERARCHICAL_COMM_BLOCK_M>;
template <class T> using Tensor = Catlass::Gemm::GemmType<T, Catlass::layout::RowMajor>;
template <class T> using BlockCopy = comm::block::HierarchicalCopy<
    ArchTag, Tensor<T>, Tensor<T>, comm::tile::TileMteCopy<Tensor<T>>, TileSwizzle>;

// Explicit MIX: two physical AIV workers per AIC. Peer index is GetBlockIdx(),
// not the AIC block index or the AIV sub-block index.
template <class Operator>
__schedmode__(1) __global__ __mix__(1, 2) void RunKernel(typename Operator::Params params, uint64_t ffts)
{
    AscendC::SetSyncBaseAddr(ffts);
    Operator op;
    op(params);
}

void LaunchAllToAll(const LaunchArgs& args, uint32_t blocks, void* stream, uint64_t ffts)
{
    using namespace AscendC;
    using Operator = comm::kernel::HierarchicalAllToAll<
        BlockCopy<float8_e4m3_t>, BlockCopy<float8_e8m0_t>, Scheduler, ChecksumConsumer>;
    static_assert(std::is_same_v<typename Operator::Params, LaunchArgs>);
    RunKernel<Operator><<<blocks, nullptr, stream>>>(args, ffts);
}

using Barrier = comm::kernel::HierarchicalBarrier;
uint64_t BarrierBytes(uint32_t world)
{
    return Barrier::GetWorkspaceSize({0, world, HIERARCHICAL_GROUP_SIZE, 1});
}
void LaunchBarrier(uint32_t rank, uint32_t world, uint32_t epoch, void* workspace, void* stream, uint64_t ffts)
{
    // Parameter conversion is a host operation, never called from device code.
    auto params = Barrier::ToUnderlyingArguments(
        {rank, world, HIERARCHICAL_GROUP_SIZE, epoch}, static_cast<uint8_t*>(workspace));
    RunKernel<Barrier><<<1, nullptr, stream>>>(params, ffts);
}
} // namespace deepep::hierarchical
