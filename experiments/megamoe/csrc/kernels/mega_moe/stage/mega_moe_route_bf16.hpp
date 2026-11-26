// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "common/mega_moe_utils.hpp"
#include "mega_moe_route_compact.hpp"
#include "mc2_common/mc2_kernel_utils.hpp"

namespace MegaMoeImpl {
namespace Bf16Route {

using namespace AscendC;

constexpr uint32_t kIndexBatch = 64;

// Publish this call's route table, including when backward follows a newer microbatch.
__aicore__ inline void PublishDispatchInput(GM_ADDR input, GM_ADDR sourceTokens, GM_ADDR rows, GM_ADDR routeIndices,
                                            uint32_t m, uint32_t hidden, uint32_t topK, uint32_t core, uint32_t cores,
                                            LocalTensor<uint8_t> scratch)
{
    MegaMoeImpl::CopyContiguous(sourceTokens, input, uint64_t(m) * hidden * sizeof(bfloat16_t), core, cores, scratch);
    MegaMoeImpl::CopyContiguous(routeIndices, rows, uint64_t(m) * topK * sizeof(int32_t), core, cores, scratch);
}

template <class Visit>
__aicore__ inline void ForEachSourceToken(GlobalTensor<int32_t> routeIds, LocalTensor<int32_t> indices, uint32_t rows,
                                          uint32_t topK, Visit visit)
{
    for (uint32_t begin = 0; begin < rows; begin += kIndexBatch) {
        const uint32_t count = rows - begin < kIndexBatch ? rows - begin : kIndexBatch;
        DataCopyPad(indices, routeIds[begin], {1U, count * uint32_t(sizeof(int32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        MegaMoeImpl::SyncCopyStage<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < count; ++i) {
            visit(begin + i, static_cast<uint32_t>(indices.GetValue(i)) / topK);
        }
        MegaMoeImpl::SyncCopyStage<HardEvent::S_MTE2>();
    }
}

constexpr uint32_t kBatch = 4096;

struct CompactScratch {
    LocalTensor<int32_t> ids;
    LocalTensor<int32_t> indices;
    LocalTensor<int32_t> routes;
    LocalTensor<uint8_t> mask;
};

// The first pass publishes source counts. The second places each stable list in
// its exact expert interval without reserving a worst-case list per expert.
template <bool Store>
__aicore__ inline uint32_t ScanExpert(CompactScratch& ub, GlobalTensor<int32_t>& ids, GlobalTensor<int32_t>& rows,
                                      int32_t expert, uint32_t items, uint32_t firstRow)
{
    uint32_t count = 0;
    for (uint32_t first = 0; first < items; first += kBatch) {
        const uint32_t n = items - first < kBatch ? items - first : kBatch;
        DataCopyPad(ub.ids, ids[first], {1U, n * 4U, 0U, 0U, 0U}, DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        CreateVecIndex(ub.indices, static_cast<int32_t>(first), n);
        const uint32_t found = CompactExpertRoutes(ub.routes, ub.ids, ub.indices, ub.mask, expert, n);
        SyncFunc<HardEvent::V_S>();
        if constexpr (Store) {
            if (found != 0) {
                SyncFunc<HardEvent::V_MTE3>();
                DataCopyPad(rows[firstRow + count], ub.routes, {1U, found * 4U, 0U, 0U, 0U});
                SyncFunc<HardEvent::MTE3_V>();
            }
        }
        count += found;
        SyncFunc<HardEvent::V_MTE2>();
    }
    return count;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void InvertRows(__gm__ int32_t* rows, __gm__ int32_t* inverse,
                                                                uint32_t begin, uint32_t end)
{
    for (uint32_t i = begin + Simt::GetThreadIdx(); i < end; i += Simt::GetThreadNum()) {
        inverse[rows[i]] = i;
    }
}

}  // namespace Bf16Route

__aicore__ inline void InitBf16Routes(GM_ADDR idsAddr, GM_ADDR x, GM_ADDR sourceTokens, GM_ADDR routeIndices,
                                      GM_ADDR countsAddr, GM_ADDR rowsAddr, GM_ADDR inverseAddr, uint32_t items,
                                      uint32_t topK, uint32_t hidden, uint32_t experts, uint32_t core, uint32_t cores)
{
    using namespace AscendC;
    using namespace Bf16Route;
    const auto owned = SplitRouteExperts(experts, core, cores);
    auto* counts = reinterpret_cast<__gm__ int32_t*>(countsAddr);
    auto* rowPtr = reinterpret_cast<__gm__ int32_t*>(rowsAddr);
    auto* inverse = reinterpret_cast<__gm__ int32_t*>(inverseAddr);
    GlobalTensor<int32_t> ids, rows;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idsAddr));
    rows.SetGlobalBuffer(rowPtr);
    {
        TPipe pipe;
        TBuf<> storage;
        pipe.InitBuffer(storage, 3 * kBatch * sizeof(int32_t) + kBatch / 8);
        auto data = storage.Get<int32_t>();
        CompactScratch scratch{data, data[kBatch], data[2 * kBatch], data[3 * kBatch].ReinterpretCast<uint8_t>()};
        for (int32_t e = owned.begin; e < owned.begin + owned.count; ++e) {
            WriteGmByPassDCache(counts + e, static_cast<int32_t>(ScanExpert<false>(scratch, ids, rows, e, items, 0)));
        }
        SyncAll();
        uint32_t firstRow = 0;
        if (owned.count > 0) {
            for (int32_t e = 0; e < owned.begin; ++e) firstRow += ReadGmByPassDCache(counts + e);
        }
        for (int32_t e = owned.begin; e < owned.begin + owned.count; ++e) {
            firstRow += ScanExpert<true>(scratch, ids, rows, e, items, firstRow);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        pipe.Destroy();
    }
    const uint32_t begin = (static_cast<uint64_t>(items) * core) / cores;
    const uint32_t end = (static_cast<uint64_t>(items) * (core + 1)) / cores;
    if (begin < end) {
        Simt::VF_CALL<InvertRows>(Simt::Dim3{256, 1, 1}, rowPtr, inverse, begin, end);
    }
    TPipe pipe;
    TBuf<> storage;
    pipe.InitBuffer(storage, MegaMoeImpl::kCopyBufferBytes);
    PublishDispatchInput(x, sourceTokens, rowsAddr, routeIndices, items / topK, hidden, topK, core, cores,
                         storage.Get<uint8_t>());
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}

}  // namespace MegaMoeImpl
