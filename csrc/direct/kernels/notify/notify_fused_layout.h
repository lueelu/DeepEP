#pragma once

#include "kernels/notify/notify_dispatch_plan.h"
#include "common/fused_layout.h"

namespace ascend_deepep {
// Counts/routes/chunks retain the plan helper's ABI. Top-k and private
// histograms occupy the remaining UB, then all regions are reused by notify.
static_assert(kFusedTopkOffset >= kPlanChunksOffset + kPlanMaxChunks * 4U);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__simt_vf__ __launch_bounds__(128) inline void ReduceFusedPartialsVf(const __ubuf__ int32_t* input,
                                                                     __ubuf__ int32_t* output, uint32_t rows,
                                                                     uint32_t columns)
{
    for (uint32_t c = threadIdx.x; c < columns; c += 128) {
        int32_t sum = 0;
        for (uint32_t r = 0; r < rows; ++r) sum += input[r * columns + c];
        output[c] = sum;
    }
}

template <class Resource>
__aicore__ inline void ReduceFusedLayoutPartials(Resource& resource, GM_ADDR input_ptr, GM_ADDR output_ptr,
                                                 uint32_t rows, uint32_t stride, uint32_t columns, uint32_t core,
                                                 uint32_t cores)
{
    auto input = resource.template GetBufferByByte<int32_t>(0);
    auto output = resource.template GetBufferByByte<int32_t>(65536);
    AscendC::GlobalTensor<int32_t> source, destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input_ptr));
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output_ptr));
    const uint32_t per_core = ((columns + cores - 1U) / cores + 7U) / 8U * 8U;
    const uint32_t begin = core * per_core;
    const uint32_t end = begin + per_core < columns ? begin + per_core : columns;
    const uint32_t tile = 16384U / rows / 8U * 8U;
    for (uint32_t c = begin; c < end; c += tile) {
        const uint32_t valid = end - c < tile ? end - c : tile;
        const uint32_t width = (valid + 7U) / 8U * 8U;
        const AscendC::DataCopyExtParams load(rows, width * 4U, (stride - width) * 4U, 0, 0);
        AscendC::DataCopyPad(input, source[c], load, AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        asc_vf_call<ReduceFusedPartialsVf>(dim3(128), reinterpret_cast<__ubuf__ int32_t*>(input.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int32_t*>(output.GetPhyAddr()), rows, width);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        const AscendC::DataCopyExtParams store(1, valid * 4U, 0, 0, 0);
        AscendC::DataCopyPad(destination[c], output, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    }
}

__simt_vf__ __launch_bounds__(128) inline void DecodeFusedRoutesVf(const __ubuf__ topk_idx_t* topk,
                                                                   __ubuf__ uint8_t* routes, uint32_t tokens,
                                                                   uint32_t k, uint32_t experts, uint32_t world)
{
    for (uint32_t t = threadIdx.x; t < tokens; t += 128) {
        for (uint32_t p = 0; p < world; ++p) routes[t * world + p] = 0;
        for (uint32_t j = 0; j < k; ++j) {
            const auto e = topk[t * k + j];
            if (e == -1) continue;
            if (e < 0 || uint64_t(e) >= experts) {
                __asc_simt_vf::__trap();
                return;
            }
            routes[t * world + uint32_t(e) / (experts / world)] = 1;
        }
    }
}

// Each active thread exclusively owns one histogram; no atomic updates and no
// dependence on route skew. Accumulate across tiles, reduce once per channel.
__simt_vf__ __launch_bounds__(128) inline void CountFusedExpertsVf(const __ubuf__ topk_idx_t* topk,
                                                                   __ubuf__ int32_t* hist, uint32_t tokens, uint32_t k,
                                                                   uint32_t experts, uint32_t stride, uint32_t shards)
{
    const uint32_t lane = threadIdx.x;
    if (lane >= shards) return;
    for (uint32_t t = lane; t < tokens; t += shards)
        for (uint32_t j = 0; j < k; ++j) {
            const auto e = topk[t * k + j];
            if (e == -1) continue;
            if (e < 0 || uint64_t(e) >= experts) {
                __asc_simt_vf::__trap();
                return;
            }
            ++hist[lane * stride + uint32_t(e)];
        }
}

__simt_vf__ __launch_bounds__(128) inline void StoreFusedExpertsVf(const __ubuf__ int32_t* hist, __gm__ int32_t* output,
                                                                   uint32_t stride, uint32_t shards)
{
    for (uint32_t e = threadIdx.x; e < stride; e += 128) {
        int32_t n = 0;
        for (uint32_t lane = 0; lane < shards; ++lane) n += hist[lane * stride + e];
        output[e] = n;
    }
    asc_threadfence();
}

template <class Resource>
__aicore__ inline void BuildFusedDispatchPlan(Resource& resource, GM_ADDR routes_gm, GM_ADDR plan, GM_ADDR head,
                                              const DispatchTilingData& td, uint32_t channel)
{
    auto counts = resource.template GetBufferByByte<int32_t>(0);
    auto routes = resource.template GetBufferByByte<uint8_t>(kPlanCountBytes);
    auto chunks = resource.template GetBufferByByte<int32_t>(kPlanChunksOffset);
    auto topk = resource.template GetBufferByByte<topk_idx_t>(kFusedTopkOffset);
    auto hist = resource.template GetBufferByByte<int32_t>(kFusedHistogramOffset);
    const uint32_t capacity = kFusedHistogramBytes / (td.expert_stride * 4U);
    const uint32_t shards = capacity < 32U ? capacity : 32U;
    const uint32_t by_topk = kFusedTopkBytes / (td.num_topk * sizeof(topk_idx_t));
    const uint32_t by_route = kPlanRouteBytes / td.world_size;
    uint32_t tile_tokens = by_topk < by_route ? by_topk : by_route;
    tile_tokens = tile_tokens < 128U ? tile_tokens : 128U;
    // Keep non-final tiles on the plan's eight-token chunk boundary.
    tile_tokens = tile_tokens / kPlanChunkTokens * kPlanChunkTokens;
    if (!shards || !tile_tokens) {
        trap();
        return;
    }
    const uint32_t length = (td.num_tokens + td.num_channels - 1U) / td.num_channels;
    const uint32_t stride = (length + 7U) / 8U * 8U;
    const uint32_t begin = channel * length < td.num_tokens ? channel * length : td.num_tokens;
    const uint32_t end = begin + length < td.num_tokens ? begin + length : td.num_tokens;
    AscendC::GlobalTensor<topk_idx_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ topk_idx_t*>(td.notify_topk_idx));
    AscendC::GlobalTensor<uint8_t> output;
    output.SetGlobalBuffer(routes_gm);
    AscendC::Duplicate(counts, int32_t(0), td.rank_stride);
    AscendC::Duplicate(hist, int32_t(0), shards * td.expert_stride);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t start = begin; start < end; start += tile_tokens) {
        const uint32_t size = end - start < tile_tokens ? end - start : tile_tokens;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        const AscendC::DataCopyExtParams copy(1, size * td.num_topk * sizeof(topk_idx_t), 0, 0, 0);
        AscendC::DataCopyPad(topk, input[uint64_t(start) * td.num_topk], copy,
                             AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        auto* idx = reinterpret_cast<__ubuf__ topk_idx_t*>(topk.GetPhyAddr());
        auto* mask = reinterpret_cast<__ubuf__ uint8_t*>(routes.GetPhyAddr());
        auto* chunk = reinterpret_cast<__ubuf__ int32_t*>(chunks.GetPhyAddr());
        asc_vf_call<DecodeFusedRoutesVf>(dim3(128), idx, mask, size, td.num_topk, td.num_experts, td.world_size);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<CountFusedExpertsVf>(dim3(128), idx, reinterpret_cast<__ubuf__ int32_t*>(hist.GetPhyAddr()), size,
                                         td.num_topk, td.num_experts, td.expert_stride, shards);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t per_peer = (size + 7U) / 8U;
        asc_vf_call<CountDispatchChunksVf>(dim3(128), mask, chunk, td.world_size, size, per_peer);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<PrefixDispatchChunksVf>(dim3(128), chunk, reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()),
                                            td.world_size, per_peer);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<ScatterDispatchPlanVf>(dim3(128), mask, chunk, reinterpret_cast<__gm__ int32_t*>(plan),
                                           reinterpret_cast<__gm__ int32_t*>(head), td.world_size, td.num_channels,
                                           channel, stride, start, size, per_peer);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        const AscendC::DataCopyExtParams store(1, size * td.world_size, 0, 0, 0);
        AscendC::DataCopyPad(output[uint64_t(start) * td.world_size], routes, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    asc_vf_call<StoreFusedExpertsVf>(
        dim3(128), reinterpret_cast<__ubuf__ int32_t*>(hist.GetPhyAddr()),
        reinterpret_cast<__gm__ int32_t*>(td.notify_expert_partials) + uint64_t(channel) * td.expert_stride,
        td.expert_stride, shards);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}
#endif
}  // namespace ascend_deepep
