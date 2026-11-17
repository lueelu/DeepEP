// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_combine_send.hpp
 * \brief Return expert output tiles to their source ranks during combine.
 */

#ifndef MEGA_MOE_COMBINE_SEND_HPP
#define MEGA_MOE_COMBINE_SEND_HPP
#include "catlass/gemm_coord.hpp"
#include "kernel_operator.h"
#include "layout/mega_moe_context.hpp"
#include "routing/peer_row_view.hpp"
using namespace AscendC;

namespace MegaMoeCombineImpl {
template <typename ElementMMadOut2, typename GemmCoord>
__aicore__ inline void CombineTokens(uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t groupIdx,
                                     uint32_t expertRowOffset, uint32_t rankId,
                                     LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, GemmCoord& actualBlockShape,
                                     const Params& params, LocalTensor<int32_t>* cachedCumsum = nullptr,
                                     LocalTensor<int32_t>* cachedSourcePrefix = nullptr)
{
    int32_t lenTile = actualBlockShape.m();
    // mLoc is relative to the current GMM2 slice. Combine routing metadata is
    // expressed in full-expert row coordinates, so add the slice offset.
    int32_t stTile = expertRowOffset + mLoc;
    int32_t edTile = stTile + lenTile;
    if (params.peerRowsGroups != 0U) {
        using namespace MegaMoeBF16Training::Wave;
        const auto expert = GetPeerExpert(params, groupIdx);
        const PeerCounts<GmPeerPrefix> counts{
            GmPeerPrefix{reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM)},
            params.tilingData->expertPerRank, groupIdx, expert.rows};
        PeerRowCursor<PeerCounts<GmPeerPrefix>> cursor(counts, params.tilingData->epWorldSize, rankId,
                                                       params.peerRowsGroups, expert.rows, expert.firstSliceRows,
                                                       stTile, edTile);
        MegaMoeImpl::PeerTransfer transfer{};
        while (cursor.Next(transfer)) {
            const auto prefix = reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrSumBeforeRank);
            const uint32_t sourceRow = AscendC::ReadGmByPassDCache(
                                           prefix + transfer.sourceRank * params.tilingData->expertPerRank + groupIdx) +
                                       transfer.sourceOrdinalBegin;
            AscendC::GlobalTensor<ElementMMadOut2> target;
            target.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
                GetRankWinAddrWithOffset(transfer.sourceRank, params.peermemInfo.ptrD - params.peermemInfo.ptrBase)));
            const AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(transfer.rowCount),
                static_cast<uint32_t>(actualBlockShape.n() * sizeof(ElementMMadOut2)), 0U,
                static_cast<uint32_t>((n - actualBlockShape.n()) * sizeof(ElementMMadOut2)), 0U};
            AscendC::DataCopyPad(target[static_cast<uint64_t>(sourceRow) * n + nLoc],
                                 l0cOutUbGMM2[(transfer.destinationRow - stTile) * actualBlockShape.n()], copy);
        }
        return;
    }
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        int32_t lenRankInExpert;
        int32_t dstExpertOffset;
        if (cachedCumsum != nullptr && cachedSourcePrefix != nullptr) {
            const int32_t previous =
                dstEpIdx ? cachedCumsum->GetValue((dstEpIdx - 1U) * params.tilingData->expertPerRank + groupIdx) : 0;
            lenRankInExpert = cachedCumsum->GetValue(dstEpIdx * params.tilingData->expertPerRank + groupIdx) - previous;
            dstExpertOffset = cachedSourcePrefix->GetValue(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        } else {
            lenRankInExpert = tokenPerExpertWin(
                dstEpIdx * Ops::Base::CeilAlign(
                               static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
                               static_cast<uint32_t>(ALIGN_128)) +
                rankId * params.tilingData->expertPerRank + groupIdx);
            dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        }
        int32_t stRankInExpert = preSumRankInExpert;
        int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;

        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }

        int32_t stData = tla::max(stRankInExpert, stTile);
        int32_t edData = tla::min(edRankInExpert, edTile);
        int32_t lenData = edData - stData;
        if (lenData <= 0) {
            continue;
        }

        uint32_t dstOffsetInExpert = 0;
        if (stTile > stRankInExpert) {
            dstOffsetInExpert = stTile - stRankInExpert;
        }

        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
        uint64_t gmRemoteOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));

        // 远端偏移地址
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
        auto gmTileD = gmRemoteD[gmDstOffset];

        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
        ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);

        AscendC::DataCopyPad(gmTileD, l0cOutUbGMM2[tileOffset * actualBlockShape.n()], ub2GmParams);
        tileOffset += lenData;
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_HPP
