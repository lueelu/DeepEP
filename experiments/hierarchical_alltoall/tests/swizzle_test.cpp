// Host arithmetic checks against the public scheduler and actual Catlass TileSwizzle.
// These annotations are empty only for this CPU test; no device code is emulated.
#define __aicore__
#define __forceinline__ inline
#include "comm/block/comm_block_hierarchical_swizzle.hpp"
#include "comm/kernel/alltoall_arguments.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include <cassert>
#include <iostream>
#include <type_traits>

using namespace deepep::comm;
template <uint32_t Group, uint32_t BlockM>
void CheckScheduler()
{
    using Scheduler = block::HierarchicalCommSwizzle<32, Group, BlockM>;
    for (uint32_t world : {1U, Group, 2 * Group})
        for (uint32_t rank = 0; rank < world; ++rank)
            for (uint32_t peer = 0; peer < 64; ++peer)
                for (uint32_t rows : {1, 7, 64, 257}) {
                    Scheduler scheduler({rows, 1024, world}, rank, peer);
                    assert(scheduler.GetPeer() == peer);
                    if (peer >= world) {
                        assert(!scheduler.IsActiveCore() && scheduler.GetCoreLoop() == 0);
                        continue;
                    }
                    auto transport = peer == rank ? CopyTransport::Mte :
                        peer / Group == rank / Group ? CopyTransport::Udma : CopyTransport::Rdma;
                    assert(scheduler.GetTransport() == transport);
                    uint32_t coveredRows = 0;
                    for (uint32_t loop = 0; loop < scheduler.GetCoreLoop(); ++loop) {
                        auto coord = scheduler.GetBlockCoord(loop);
                        assert(scheduler.IsValidBlock(coord));
                        Catlass::MatrixCoord block{coord.row(), coord.column()};
                        auto offset = scheduler.GetBlockOffsetInRank(block);
                        auto shape = scheduler.GetActualBlockShape(block);
                        assert(offset.row() == coveredRows && offset.column() == 0);
                        assert(shape.column() == 1024 && shape.row() > 0);
                        coveredRows += shape.row();
                        if (transport == CopyTransport::Mte) {
                            assert(shape.row() <= BlockM);
                            Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle tiles(shape, {32U, 1024U});
                            uint32_t tileRows = 0;
                            for (uint32_t i = 0; i < tiles.GetLoops(); ++i) {
                                auto tile = tiles.GetActualTileShape(tiles.GetTileCoord(i));
                                assert(tile.row() * tile.column() <= 32768);
                                tileRows += tile.row();
                            }
                            assert(tileRows == shape.row());
                        } else {
                            assert(scheduler.GetCoreLoop() == 1 && shape.row() == rows);
                        }
                    }
                    assert(coveredRows == rows);
                }
}
int main()
{
    CheckScheduler<16, 64>();
    CheckScheduler<2, 17>();
    using Layout = kernel::AllToAllLayout<2, 4, 16>;
    Layout layout{7, 32, 4, 3, 2};
    assert(layout.Windows() == 3 && layout.Rows(2) == 1);
    assert(layout.DataStride() == 448 && layout.ScaleStride() == 64);
    assert(layout.Ready().world == 4 && layout.Ready().qps == 2 && layout.Ready().windows == 3);
    assert(layout.Ready().Bytes() == 640);
    assert(layout.Ready().ValueOffset(1) - layout.Ready().ValueOffset(0) == sizeof(uint32_t));
    assert(layout.Ready().ValueOffset(0) >= layout.Ready().FrontierOffset() + kSlotBytes);
    static_assert(std::is_standard_layout_v<kernel::AllToAllArguments<Layout>>);
    std::cout << "PASS public swizzle, MTE tile coverage and independent typed layout\n";
}
