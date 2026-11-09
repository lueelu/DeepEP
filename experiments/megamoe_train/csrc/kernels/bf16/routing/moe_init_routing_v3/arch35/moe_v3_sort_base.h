// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*!
 * \file moe_v3_sort_base.h
 * \brief
 */
#ifndef MOE_V3_SORT_BASE_H_REGBASE
#define MOE_V3_SORT_BASE_H_REGBASE

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeSortBase {
public:
    __aicore__ inline MoeSortBase(){};
    __aicore__ inline int64_t GetSyncRound();

protected:
    __aicore__ inline void CleanWSCache();
    __aicore__ inline void SyncAll();

protected:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;

    int64_t tileLength;
    int64_t bufferNum = 1;
    int64_t totalLength;
    int64_t coreNum;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t n;
    int64_t k;
    int64_t rowIdxType_ = 0;

    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeSortBase::SyncAll()
{
    AscendC::SyncAll();
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_SORT_BASE_H_REGBASE
