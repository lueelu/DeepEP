#pragma once

#include "kernels/common/udma_shared_sender.h"
#include "common/types.h"

#ifndef ASCEND_DEEPEP_COMBINE_SIMT_WEIGHTS
#define ASCEND_DEEPEP_COMBINE_SIMT_WEIGHTS 0
#endif

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__ inline uint32_t CombineWeightFirst(const CombineTilingData& td)
{
    return td.udma_workers_per_peer *
           Workers(td.world_size, td.udma_peers_per_worker, td.udma_peer_order, td.udma_ranks_per_group);
}

__aicore__ inline bool CombineSimtWeightsEnabled(const CombineTilingData& td)
{
    return ASCEND_DEEPEP_COMBINE_SIMT_WEIGHTS && td.has_topk_weights && td.num_topk &&
           !td.udma_token_schedule && CombineWeightFirst(td) < td.block_dim;
}

// Every remote (peer, source token, k) has one writer: rank_prefix partitions
// rows by original source peer, workers partition rows, and lanes partition k.
// The combine-rank slot isolates contributions from different senders.
__simt_vf__ __launch_bounds__(128) inline void CombineWeightsVf(
    const __gm__ float* weights, const __gm__ int32_t* src_idx, __gm__ uint8_t* workspace,
    uint32_t row_begin, uint32_t row_end, uint32_t worker, uint32_t workers, uint32_t num_topk,
    uint32_t rank, uint32_t peer, uint32_t capacity, uint64_t offset, uint32_t stride)
{
    constexpr uint32_t lanes = 8;
    for (uint64_t row = uint64_t(row_begin) + worker + uint64_t(threadIdx.x / lanes) * workers;
         row < row_end; row += uint64_t(workers) * (128U / lanes)) {
        const int32_t token = src_idx[row];
        if (token < 0 || uint32_t(token) >= capacity) {
            __asc_simt_vf::__trap();
            return;
        }
        const uint64_t slot = uint64_t(rank) * capacity + uint32_t(token);
        auto* dst = reinterpret_cast<__gm__ float*>(workspace + offset + slot * stride);
        for (uint32_t k = threadIdx.x % lanes; k < num_topk; k += lanes)
            simt::aclshmem_float_p(dst + k, weights[row * num_topk + k], peer);
    }
    asc_threadfence();
}

__aicore__ inline void CombineRemoteWeights(
    GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR workspace,
    const CombineTilingData& td, uint32_t core)
{
    const uint32_t first = CombineWeightFirst(td);
    if (core < first) return;
    const uint32_t worker = core - first, workers = td.block_dim - first;
    AscendC::GlobalTensor<int32_t> ranks;
    ranks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    for (uint32_t peer = 0; peer < td.world_size; ++peer) {
        if (peer == td.rank) continue; // Existing local-copy path owns self.
        const uint32_t begin = peer ? ranks.GetValue((peer - 1U) * td.world_size + td.rank) : 0;
        const uint32_t end = ranks.GetValue(peer * td.world_size + td.rank);
        if (uint64_t(begin) + worker >= end) continue;
        asc_vf_call<CombineWeightsVf>(dim3(128), reinterpret_cast<__gm__ float*>(weights),
            reinterpret_cast<__gm__ int32_t*>(src_idx), workspace, begin, end, worker, workers,
            td.num_topk, td.rank, peer, td.token_capacity, td.topk_weights_buffer_offset,
            td.topk_weights_row_stride);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    }
}
#endif
} // namespace ascend_deepep::udma
