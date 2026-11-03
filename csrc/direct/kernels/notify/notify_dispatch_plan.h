#pragma once

#include "common/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep {
constexpr uint32_t kPlanCountBytes = 4096;
constexpr uint32_t kPlanRouteBytes = 65536;
constexpr uint32_t kPlanChunkTokens = 8;
constexpr uint32_t kPlanChunksOffset = kPlanCountBytes + kPlanRouteBytes;
constexpr uint32_t kPlanMaxChunks = kPlanRouteBytes / kPlanChunkTokens + kPlanCountBytes / sizeof(int32_t);
static_assert(kPlanChunksOffset + kPlanMaxChunks * sizeof(int32_t) <= 216 * 1024);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Small independent token chunks expose parallelism even for EP2/4/8. Three
// ordered VFs preserve the original stable token order without atomic append.
__simt_vf__ __launch_bounds__(128) inline void CountDispatchChunksVf(
    const __ubuf__ uint8_t* routes, __ubuf__ int32_t* chunks, uint32_t world, uint32_t tokens, uint32_t per_peer)
{
    for (uint32_t task = threadIdx.x; task < world * per_peer; task += 128) {
        const uint32_t peer = task / per_peer;
        const uint32_t begin = (task % per_peer) * kPlanChunkTokens;
        int32_t count = 0;
        for (uint32_t t = begin; t < tokens && t < begin + kPlanChunkTokens; ++t)
            count += routes[t * world + peer] != 0;
        chunks[task] = count;
    }
}

__simt_vf__ __launch_bounds__(128) inline void PrefixDispatchChunksVf(
    __ubuf__ int32_t* chunks, __ubuf__ int32_t* counts, uint32_t world, uint32_t per_peer)
{
    for (uint32_t peer = threadIdx.x; peer < world; peer += 128) {
        int32_t count = counts[peer];
        for (uint32_t c = 0; c < per_peer; ++c) {
            const uint32_t i = peer * per_peer + c;
            const int32_t n = chunks[i];
            chunks[i] = count;
            count += n;
        }
        counts[peer] = count;
    }
}

__simt_vf__ __launch_bounds__(128) inline void ScatterDispatchPlanVf(
    const __ubuf__ uint8_t* routes, const __ubuf__ int32_t* chunks, __gm__ int32_t* plan, __gm__ int32_t* head,
    uint32_t world, uint32_t channels, uint32_t channel, uint32_t stride, uint32_t begin, uint32_t tokens,
    uint32_t per_peer)
{
    for (uint32_t task = threadIdx.x; task < world * per_peer; task += 128) {
        const uint32_t peer = task / per_peer;
        const uint32_t first = (task % per_peer) * kPlanChunkTokens;
        int32_t count = chunks[task];
        const uint64_t base = (uint64_t(peer) * channels + channel) * stride;
        for (uint32_t t = first; t < tokens && t < first + kPlanChunkTokens; ++t) {
            const uint64_t index = uint64_t(begin + t) * world + peer;
            if (routes[t * world + peer]) {
                if (head != nullptr)
                    head[index] = count;
                plan[base + count++] = static_cast<int32_t>(begin + t);
            } else if (head != nullptr) {
                head[index] = -1;
            }
        }
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(128) inline void PrefixChannelCountsVf(
    __ubuf__ int32_t* values, uint32_t rank_stride, uint32_t channels)
{
    for (uint32_t rank = threadIdx.x; rank < rank_stride; rank += 128) {
        int32_t sum = 0;
        for (uint32_t channel = 0; channel < channels; ++channel) {
            const uint32_t index = channel * rank_stride + rank;
            sum += values[index];
            values[index] = sum;
        }
    }
}

// On return counts at UB offset 0 are ready for the existing notify count
// publication. This phase overlays UB that notify later uses for prefixes.
template <class Resource>
__aicore__ inline void BuildDispatchPlan(
    Resource& resource, GM_ADDR routes, GM_ADDR plan, GM_ADDR head, const DispatchTilingData& td, uint32_t channel)
{
    auto counts = resource.template GetBufferByByte<int32_t>(0);
    auto tile = resource.template GetBufferByByte<uint8_t>(kPlanCountBytes);
    auto chunks = resource.template GetBufferByByte<int32_t>(kPlanChunksOffset);
    AscendC::GlobalTensor<uint8_t> input;
    input.SetGlobalBuffer(routes);
    const uint32_t length = (td.num_tokens + td.num_channels - 1) / td.num_channels;
    const uint32_t stride = (length + 7U) / 8U * 8U;
    const uint32_t begin = channel * length < td.num_tokens ? channel * length : td.num_tokens;
    const uint32_t end = begin + length < td.num_tokens ? begin + length : td.num_tokens;
    const uint32_t tile_tokens = kPlanRouteBytes / td.world_size;
    AscendC::Duplicate(counts, int32_t(0), td.rank_stride);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t start = begin; start < end; start += tile_tokens) {
        const uint32_t size = end - start < tile_tokens ? end - start : tile_tokens;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::DataCopyExtParams copy{1, size * td.world_size, 0, 0, 0};
        AscendC::DataCopyPad(
            tile, input[uint64_t(start) * td.world_size], copy, AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t per_peer = (size + kPlanChunkTokens - 1) / kPlanChunkTokens;
        asc_vf_call<CountDispatchChunksVf>(
            dim3(128), reinterpret_cast<__ubuf__ uint8_t*>(tile.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(chunks.GetPhyAddr()), td.world_size, size, per_peer);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<PrefixDispatchChunksVf>(
            dim3(128), reinterpret_cast<__ubuf__ int32_t*>(chunks.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()), td.world_size, per_peer);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<ScatterDispatchPlanVf>(
            dim3(128), reinterpret_cast<__ubuf__ uint8_t*>(tile.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(chunks.GetPhyAddr()), reinterpret_cast<__gm__ int32_t*>(plan),
            reinterpret_cast<__gm__ int32_t*>(head), td.world_size, td.num_channels, channel, stride, start, size,
            per_peer);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
}
#endif
} // namespace ascend_deepep
