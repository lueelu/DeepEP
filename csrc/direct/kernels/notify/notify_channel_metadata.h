#pragma once

#include "common/dispatch_channel_metadata.h"
#include "common/platform.h"
#include "common/types.h"

namespace ascend_deepep {
static_assert(kChannelMetadataUbBytes <= platform::kKernelUbBytes);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// AIV loads token-major heads/indices/weights into UB. Each thread owns one
// (token, top-k slot), and writes every routed peer's corresponding packed slot.
// Invalid expert slots are explicitly -1/0, preserving the sparse-scatter ABI.
__simt_vf__ __launch_bounds__(128) inline void PackChannelMetadataVf(
    const __ubuf__ int32_t* heads, const __ubuf__ topk_idx_t* indices, const __ubuf__ float* weights,
    __gm__ topk_idx_t* packed_indices, __gm__ float* packed_weights,
    uint32_t tokens, uint32_t topk, uint32_t world, uint32_t experts_per_rank,
    uint32_t channels, uint32_t channel, uint32_t stride, uint32_t has_weights)
{
    for (uint32_t item = threadIdx.x; item < tokens * topk; item += 128) {
        const uint32_t token = item / topk, k = item % topk;
        const topk_idx_t expert = indices[item];
        for (uint32_t peer = 0; peer < world; ++peer) {
            const int32_t ordinal = heads[token * world + peer];
            if (ordinal < 0) continue;
            const uint64_t row = (uint64_t(peer) * channels + channel) * stride + uint32_t(ordinal);
            const bool valid = expert >= topk_idx_t(peer * experts_per_rank) &&
                               expert < topk_idx_t((peer + 1U) * experts_per_rank);
            packed_indices[row * topk + k] = valid ? expert - topk_idx_t(peer * experts_per_rank) : topk_idx_t(-1);
            if (has_weights) packed_weights[row * topk + k] = valid ? weights[item] : 0.0f;
        }
    }
    asc_threadfence();
}

// Called at notify's tail, after this core's plan/head is complete and all
// earlier UB readers have drained. Same-stream kernel completion gates dispatch.
template <class Resource>
__aicore__ inline void PackChannelMetadata(Resource& resource, GM_ADDR head,
    const DispatchTilingData& td, uint32_t channel)
{
    auto heads = resource.template GetBufferByByte<int32_t>(0);
    auto indices = resource.template GetBufferByByte<topk_idx_t>(kChannelMetadataHeadBytes);
    auto weights = resource.template GetBufferByByte<float>(kChannelMetadataHeadBytes + kChannelMetadataIndexBytes);
    AscendC::GlobalTensor<int32_t> head_gm;
    AscendC::GlobalTensor<topk_idx_t> index_gm;
    AscendC::GlobalTensor<float> weight_gm;
    head_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(head));
    index_gm.SetGlobalBuffer(reinterpret_cast<__gm__ topk_idx_t*>(td.notify_topk_idx));
    weight_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(td.notify_topk_weights));
    const uint32_t length = (td.num_tokens + td.num_channels - 1U) / td.num_channels;
    const uint32_t stride = (length + 7U) / 8U * 8U;
    const uint32_t begin = channel * length < td.num_tokens ? channel * length : td.num_tokens;
    const uint32_t end = begin + length < td.num_tokens ? begin + length : td.num_tokens;
    for (uint32_t start = begin; start < end; start += kChannelMetadataTileTokens) {
        const uint32_t n = end - start < kChannelMetadataTileTokens ? end - start : kChannelMetadataTileTokens;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::DataCopyPad(heads, head_gm[uint64_t(start) * td.world_size],
            AscendC::DataCopyExtParams{1, n * td.world_size * 4U, 0, 0, 0},
            AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::DataCopyPad(indices, index_gm[uint64_t(start) * td.num_topk],
            AscendC::DataCopyExtParams{1, n * td.num_topk * uint32_t(sizeof(topk_idx_t)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
        if (td.has_topk_weights)
            AscendC::DataCopyPad(weights, weight_gm[uint64_t(start) * td.num_topk],
                AscendC::DataCopyExtParams{1, n * td.num_topk * 4U, 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        asc_vf_call<PackChannelMetadataVf>(dim3(128),
            reinterpret_cast<__ubuf__ int32_t*>(heads.GetPhyAddr()),
            reinterpret_cast<__ubuf__ topk_idx_t*>(indices.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr()),
            reinterpret_cast<__gm__ topk_idx_t*>(td.packed_channel_indices),
            reinterpret_cast<__gm__ float*>(td.packed_channel_weights), n, td.num_topk, td.world_size,
            td.num_experts / td.world_size, td.num_channels, channel, stride, td.has_topk_weights);
        // V readers must finish before the next MTE2 load overwrites these slots.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    }
}
#endif
} // namespace ascend_deepep
