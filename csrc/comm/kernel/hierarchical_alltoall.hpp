// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#ifndef DEEPEP_COMM_KERNEL_HIERARCHICAL_ALLTOALL_HPP
#define DEEPEP_COMM_KERNEL_HIERARCHICAL_ALLTOALL_HPP

#include "comm/tile/tile_hierarchical_ready.hpp"

#if defined(__CCE__)
#include <type_traits>
#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/numeric_size.hpp"
#include "comm/tile/tile_hierarchical_signals.hpp"
#endif

namespace deepep::comm::kernel {

DEEPEP_COMM_HOST_DEVICE uint32_t GetCommLoops(uint32_t m, uint32_t commSizeM)
{
    return m / commSizeM + (m % commSizeM != 0);
}

DEEPEP_COMM_HOST_DEVICE uint32_t GetActualCommSizeM(uint32_t m, uint32_t commSizeM, uint32_t commIdx)
{
    const uint32_t remainingM = m - commIdx * commSizeM;
    return remainingM < commSizeM ? remainingM : commSizeM;
}

#if defined(__CCE__)
// One physical AIV owns each remote rank's QPs. The extra AIV aggregates
// incoming ready flags, so it cannot block senders from submitting later rounds.
template <
    class BlockComm_, class BlockCommScale_, class CommScheduler_,
    uint32_t ScaleGroup = 32, tile::ReadyMode Mode = tile::ReadyMode::PerWindowOrderedRdma>
class HierarchicalAllToAll {
public:
    using BlockComm = BlockComm_;
    using BlockCommScale = BlockCommScale_;
    using CommScheduler = CommScheduler_;
    using ArchTag = typename BlockComm::ArchTag;
    using MatrixCoord = Catlass::MatrixCoord;
    using Element = typename BlockComm::ElementSrc;
    using ElementScale = typename BlockCommScale::ElementSrc;
    static_assert(std::is_same_v<ArchTag, typename BlockCommScale::ArchTag>);
    static_assert(ScaleGroup > 0);

#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    static constexpr auto READY_MODE = Mode;
#else
    static constexpr auto READY_MODE =
        Mode == tile::ReadyMode::PerWindowOrderedRdma ? tile::ReadyMode::PerWindow : Mode;
#endif
    using Ready = tile::TileHierarchicalReady<tile::TileHierarchicalSignals, READY_MODE>;

    struct Arguments {
        uint32_t m;
        uint32_t n;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commSizeM;
        uint32_t qpNum;
        uintptr_t ptrSrc;
        uintptr_t ptrDst;
        uintptr_t ptrScaleSrc;
        uintptr_t ptrScaleDst;
        uintptr_t ptrReady;
    };

    struct Params {
        uint32_t m;
        uint32_t n;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commSizeM;
        uint32_t qpNum;
        uint32_t scaleN;
        uint64_t peerStrideBytes;
        uint64_t scalePeerStrideBytes;
        __gm__ uint8_t* ptrSrc;
        __gm__ uint8_t* ptrDst;
        __gm__ uint8_t* ptrScaleSrc;
        __gm__ uint8_t* ptrScaleDst;
        __gm__ uint8_t* ptrReady;
    };

    template <class TensorElement>
    static uint64_t GetTensorBytes(uint32_t m, uint32_t n)
    {
        constexpr auto elementBits = Catlass::SizeOfBits<TensorElement>::value;
        static_assert(elementBits % Catlass::BytesToBits(1U) == 0, "Tensor elements must be byte aligned");
        return uint64_t(m) * n * Catlass::BitsToBytes(elementBits);
    }

    static Params ToUnderlyingArguments(Arguments const& args)
    {
        const uint32_t scaleN = args.n / ScaleGroup;
        return {
            args.m, args.n, args.rankIdx, args.rankSize, args.commSizeM, args.qpNum, scaleN,
            Align(GetTensorBytes<Element>(args.m, args.n)), Align(GetTensorBytes<ElementScale>(args.m, scaleN)),
            reinterpret_cast<__gm__ uint8_t*>(args.ptrSrc), reinterpret_cast<__gm__ uint8_t*>(args.ptrDst),
            reinterpret_cast<__gm__ uint8_t*>(args.ptrScaleSrc),
            reinterpret_cast<__gm__ uint8_t*>(args.ptrScaleDst), reinterpret_cast<__gm__ uint8_t*>(args.ptrReady)};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        const uint32_t commSizeM = params.commSizeM;
        const uint32_t commLoops = GetCommLoops(params.m, commSizeM);
        const uint32_t rankSize = params.rankSize;
        if (aicoreIdx > rankSize || commLoops == 0) {
            return;
        }

        const auto readyAddress = reinterpret_cast<uintptr_t>(params.ptrReady);
        if (aicoreIdx == rankSize) {
            tile::TileHierarchicalSignals signals(readyAddress, nullptr);
            Ready ready(signals, params.rankIdx, rankSize, params.qpNum);
            if constexpr (READY_MODE == tile::ReadyMode::FinalOnly) {
                ready.WaitAll(commLoops - 1);
            } else {
                for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
                    ready.WaitAll(commIdx);
                }
            }
            return;
        }

        Catlass::Arch::Resource<ArchTag> resource;
        auto ubReady = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
        tile::TileHierarchicalSignals signals(
            readyAddress, reinterpret_cast<__ubuf__ uint32_t*>(ubReady.GetPhyAddr()));
        Ready ready(signals, params.rankIdx, rankSize, params.qpNum);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            const uint32_t actualCommSizeM = GetActualCommSizeM(params.m, commSizeM, commIdx);
            const MatrixCoord commOffset{commIdx * commSizeM, 0};
            CommScheduler commScheduler({actualCommSizeM, params.n, rankSize}, params.rankIdx, aicoreIdx);
            CommScheduler scaleCommScheduler({actualCommSizeM, params.scaleN, rankSize}, params.rankIdx, aicoreIdx);
            CopyTensor<BlockComm>(
                resource, commScheduler, params, commOffset, params.n,
                params.ptrSrc, params.ptrDst, params.peerStrideBytes);
            CopyTensor<BlockCommScale>(
                resource, scaleCommScheduler, params, commOffset, params.scaleN,
                params.ptrScaleSrc, params.ptrScaleDst, params.scalePeerStrideBytes);
            const uint32_t peer = commScheduler.GetPeer();
            const auto transport = commScheduler.GetTransport();
            if constexpr (READY_MODE != tile::ReadyMode::FinalOnly) {
                ready.NotifyWindow(peer, transport, commIdx);
            }
            if (commIdx == commLoops - 1) {
                ready.FinishPeer(peer, transport, commIdx);
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const&) {}

private:
    template <class BlockCopy>
    CATLASS_DEVICE void CopyTensor(
        Catlass::Arch::Resource<ArchTag>& resource, CommScheduler const& commScheduler, Params const& params,
        MatrixCoord const& commOffset, uint32_t n, __gm__ uint8_t* ptrSrc, __gm__ uint8_t* ptrDst,
        uint64_t peerStrideBytes)
    {
        using ElementSrc = typename BlockCopy::ElementSrc;
        using ElementDst = typename BlockCopy::ElementDst;
        const uint32_t remoteRankIdx = commScheduler.GetPeer();
        typename BlockCopy::Params blockCommParams(commScheduler.GetActualBlockShape({0U, 0U}));
        BlockCopy blockRemoteCopy(resource, blockCommParams);

        AscendC::GlobalTensor<ElementSrc> gmSrc;
        gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSrc*>(ptrSrc + remoteRankIdx * peerStrideBytes));
        AscendC::GlobalTensor<ElementDst> gmDst;
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ ElementDst*>(ptrDst + params.rankIdx * peerStrideBytes));
        const Catlass::layout::RowMajor layout(params.m, n);
        const uint32_t commCoreLoops = commScheduler.GetCoreLoop();
        for (uint32_t commLoopIdx = 0; commLoopIdx < commCoreLoops; ++commLoopIdx) {
            const auto commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
            const MatrixCoord blockCoord{commBlockCoord.row(), commBlockCoord.column()};
            const auto blockOffsetInRank = commScheduler.GetBlockOffsetInRank(blockCoord);
            const auto actualCommBlockShape = commScheduler.GetActualBlockShape(blockCoord);
            const auto offsetInRank = layout.GetOffset(commOffset + blockOffsetInRank);
            const auto layoutBlock = layout.GetTileLayout(actualCommBlockShape);
            blockRemoteCopy(
                gmSrc[offsetInRank], layoutBlock, gmDst[offsetInRank], layoutBlock, actualCommBlockShape,
                remoteRankIdx, commScheduler.GetTransport(), params.qpNum, Ready::RDMA_COMPLETION);
        }
    }
};
#endif

} // namespace deepep::comm::kernel
#endif
