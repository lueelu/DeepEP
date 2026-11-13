// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include <type_traits>
#include "comm/kernel/alltoall_arguments.hpp"
#include "comm/tile/tile_hierarchical_ready.hpp"
#include "comm/tile/tile_hierarchical_signals.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace deepep::comm::kernel {
struct WaitOnlyConsumer {
    using Arguments = EmptyConsumerArguments;
    template <class Params>
    DEEPEP_COMM_DEVICE void operator()(const Params&, uint32_t) {}
};

// Physical AIVs [0, world) own peers; AIV world only gathers local readiness.
// Launch at least world+1 AIVs so peer 0 can submit independently of receive progress.
// Only AIC0 invokes Consumer(params, window), after ALL incoming QPs are ready.
// Consumer may read that window immediately; it must not recycle communication
// storage or introduce collectives requiring idle AICs. Host owns final teardown.
// RDMA requires same-QP ordered Put visibility. UDMA never relies on that ordering.
template <class BlockData_, class BlockScale_, class Scheduler_, class Consumer = WaitOnlyConsumer,
          uint32_t ScaleGroup = 32>
class HierarchicalAllToAll {
public:
    using BlockData = BlockData_;
    using BlockScale = BlockScale_;
    using Scheduler = Scheduler_;
    using ArchTag = typename BlockData::ArchTag;
    using MatrixCoord = Catlass::MatrixCoord;
    static_assert(std::is_same_v<ArchTag, typename BlockScale::ArchTag>);
    using Layout = AllToAllLayout<
        sizeof(typename BlockData::ElementSrc), sizeof(typename BlockScale::ElementSrc), ScaleGroup>;
    using Params = AllToAllArguments<Layout, typename Consumer::Arguments>;

    template <int32_t CoreType = g_coreType>
    CATLASS_DEVICE void operator()(const Params& p);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params& p)
    {
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        if (aicoreIdx > p.layout.world) return;
        if (aicoreIdx == p.layout.world) {
            tile::TileHierarchicalSignals signals(p.control, nullptr);
            tile::TileHierarchicalReady<tile::TileHierarchicalSignals> ready(signals, p.layout.Ready(), p.rank);
            for (uint32_t window = 0; window < p.layout.Windows(); ++window) ready.WaitAll(window);
            return;
        }
        Catlass::Arch::Resource<ArchTag> resource;
        auto ub = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
        tile::TileHierarchicalSignals signals(p.control, reinterpret_cast<__ubuf__ uint32_t*>(ub.GetPhyAddr()));
        tile::TileHierarchicalReady<tile::TileHierarchicalSignals> ready(signals, p.layout.Ready(), p.rank);
        for (uint32_t window = 0; window < p.layout.Windows(); ++window) {
            const uint32_t rows = p.layout.Rows(window);
            const uint32_t rowOffset = window * p.layout.windowM;
            Scheduler scheduler({rows, p.layout.n, p.layout.world}, p.rank, aicoreIdx);
            Copy<BlockData>(resource, scheduler, p, rowOffset, p.layout.n,
                            p.input, p.output, p.layout.DataStride());
            Scheduler scaleScheduler({rows, p.layout.ScaleN(), p.layout.world}, p.rank, aicoreIdx);
            Copy<BlockScale>(resource, scaleScheduler, p, rowOffset, p.layout.ScaleN(),
                             p.inputScale, p.outputScale, p.layout.ScaleStride());
            ready.Notify(scheduler.GetPeer(), scheduler.GetTransport(), window, RdmaCompletion::Deferred);
        }
        Scheduler scheduler({p.layout.m, p.layout.n, p.layout.world}, p.rank, aicoreIdx);
        if (scheduler.GetTransport() == CopyTransport::Rdma) {
            typename BlockData::Params params(scheduler.GetActualBlockShape({0U, 0U}));
            BlockData blockCopy(resource, params);
            blockCopy.QuietRdma(scheduler.GetPeer(), p.layout.qps);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params& p)
    {
        if (AscendC::GetBlockIdx() != 0) return;
        tile::TileHierarchicalSignals signals(p.control, nullptr);
        tile::TileHierarchicalReady<tile::TileHierarchicalSignals> ready(signals, p.layout.Ready(), p.rank);
        Consumer consume;
        for (uint32_t window = 0; window < p.layout.Windows(); ++window) {
            ready.WaitReady(window);
            consume(p, window);
        }
    }

private:
    template <class BlockCopy>
    DEEPEP_COMM_DEVICE void Copy(Catlass::Arch::Resource<ArchTag>& resource, const Scheduler& scheduler,
                        const Params& p, uint32_t rowOffset, uint32_t n,
                        uintptr_t input, uintptr_t output, uint64_t peerStride)
    {
        using T = typename BlockCopy::ElementSrc;
        typename BlockCopy::Params params(scheduler.GetActualBlockShape({0U, 0U}));
        BlockCopy blockCopy(resource, params);
        AscendC::GlobalTensor<T> gmSrc, gmDst;
        gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input + scheduler.GetPeer() * peerStride) + uint64_t(rowOffset) * n);
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output + p.rank * peerStride) + uint64_t(rowOffset) * n);
        const Catlass::layout::RowMajor layout(p.layout.m, n);
        for (uint32_t commIdx = 0; commIdx < scheduler.GetCoreLoop(); ++commIdx) {
            const auto coord = scheduler.GetBlockCoord(commIdx);
            const MatrixCoord blockCoord{coord.row(), coord.column()};
            const auto blockOffsetSrc = scheduler.GetBlockOffsetInRank(blockCoord);
            const auto blockOffsetDst = blockOffsetSrc;
            const auto actualCommBlockShape = scheduler.GetActualBlockShape(blockCoord);
            blockCopy(gmSrc[layout.GetOffset(blockOffsetSrc)], layout.GetTileLayout(actualCommBlockShape),
                      gmDst[layout.GetOffset(blockOffsetDst)], layout.GetTileLayout(actualCommBlockShape),
                      actualCommBlockShape, scheduler.GetPeer(), scheduler.GetTransport(), p.layout.qps,
                      BlockCopy::RdmaCompletion::Deferred);
        }
    }
};
} // namespace deepep::comm::kernel
