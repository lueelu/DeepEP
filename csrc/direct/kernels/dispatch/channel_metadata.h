#pragma once

#include "common/dispatch_channel_metadata.h"
#include "common/platform.h"
#include "common/types.h"

namespace ascend_deepep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// The plan already deduplicates tokens per peer and preserves token order.
// All input reads are from AIV-loaded UB. A lane owns one output (row, k),
// including invalid slots; no peer scan, atomics or output initialization needed.
__simt_vf__ __launch_bounds__(128) inline void PackPlanMetadataVf(
    const __ubuf__ int32_t* tokens, const __ubuf__ topk_idx_t* indices, const __ubuf__ float* weights,
    __gm__ topk_idx_t* output_indices, __gm__ float* output_weights, __gm__ int32_t* output_src, uint32_t rows,
    uint32_t input_begin, uint32_t topk, uint32_t peer, uint32_t experts_per_rank, uint32_t has_weights)
{
    for (uint32_t item = threadIdx.x; item < rows * topk; item += 128U) {
        const uint32_t row = item / topk, k = item % topk;
        const uint32_t token = uint32_t(tokens[row]);
        const uint32_t input = (token - input_begin) * topk + k;
        const topk_idx_t expert = indices[input];
        const bool valid =
            expert >= topk_idx_t(peer * experts_per_rank) && expert < topk_idx_t((peer + 1U) * experts_per_rank);
        output_indices[item] = valid ? expert - topk_idx_t(peer * experts_per_rank) : topk_idx_t(-1);
        if (has_weights) output_weights[item] = valid ? weights[input] : 0.0f;
        if (output_src != nullptr && k == 0) output_src[row] = int32_t(token);
    }
    asc_threadfence();
}

struct PlanMetadataWindow {
    uint32_t rows;
    uint32_t input_begin;
};

// Batch the sender's peers in one VF and reuse the input token window. The
// Descriptor: {peer, valid_rows, first_ordinal, cache_cursor}; only valid plan
// rows are traversed. In particular, do not launch a VF per peer/token row.
__simt_vf__ __launch_bounds__(128) inline void PackRemoteMetadataVf(
    const __ubuf__ int32_t* plans, const __ubuf__ uint32_t* descriptors, const __ubuf__ topk_idx_t* indices,
    const __ubuf__ float* weights, __gm__ topk_idx_t* output_indices, __gm__ float* output_weights, uint32_t peers,
    uint32_t tile_tokens, uint32_t input_begin, uint32_t topk, uint32_t experts_per_rank, uint32_t channels,
    uint32_t channel, uint32_t stride, uint32_t has_weights)
{
    for (uint32_t p = 0; p < peers; ++p) {
        const uint32_t peer = descriptors[p * 4U], rows = descriptors[p * 4U + 1U];
        const uint32_t cursor = descriptors[p * 4U + 3U];
        const uint64_t base = ((uint64_t(peer) * channels + channel) * stride + descriptors[p * 4U + 2U]) * topk;
        for (uint32_t item = threadIdx.x; item < rows * topk; item += 128U) {
            const uint32_t token = uint32_t(plans[p * tile_tokens + cursor + item / topk]);
            const uint32_t input = (token - input_begin) * topk + item % topk;
            const topk_idx_t expert = indices[input];
            const bool valid =
                expert >= topk_idx_t(peer * experts_per_rank) && expert < topk_idx_t((peer + 1U) * experts_per_rank);
            output_indices[base + item] = valid ? expert - topk_idx_t(peer * experts_per_rank) : topk_idx_t(-1);
            if (has_weights) output_weights[base + item] = valid ? weights[input] : 0.0f;
        }
    }
    asc_threadfence();
}

template <uint32_t Capacity, class Resource, class Sender>
__aicore__ inline void LoadRemoteMetadataWindow(Resource& resource, Sender& sender, GM_ADDR plan,
                                                const DispatchTilingData& td, const ChannelMetadataTile& tile,
                                                uint32_t channel, uint32_t stride, uint32_t input_begin,
                                                uint32_t input_count, const uint32_t* totals, uint32_t* consumed,
                                                uint32_t offset, uint32_t slot, uint32_t plan_tokens,
                                                uint32_t* cache_begin, uint32_t* cache_rows)
{
    constexpr uint32_t descriptor_bytes = (Capacity * kChannelMetadataDescriptorWords * 4U + 31U) / 32U * 32U;
    const auto event = slot ? EVENT_ID7 : EVENT_ID6;
    auto descriptors = resource.template GetBufferByByte<uint32_t>(offset);
    auto plans = resource.template GetBufferByByte<int32_t>(offset + descriptor_bytes);
    auto indices = resource.template GetBufferByByte<topk_idx_t>(offset + descriptor_bytes + tile.index_offset);
    auto weights = resource.template GetBufferByByte<float>(offset + descriptor_bytes +
                                                            (td.has_topk_weights ? tile.weight_offset : 0));
    AscendC::GlobalTensor<int32_t> pg;
    AscendC::GlobalTensor<topk_idx_t> ig;
    AscendC::GlobalTensor<float> wg;
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
    ig.SetGlobalBuffer(reinterpret_cast<__gm__ topk_idx_t*>(td.notify_topk_idx));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(td.notify_topk_weights));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
    bool loaded = false;
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        const uint32_t left = totals[p] - consumed[p];
        const uint32_t n = left < plan_tokens ? left : plan_tokens;
        if (!n) continue;
        // The caller has retired this slot's VF before inspecting/reusing it.
        // Keep its unconsumed suffix if it covers the entire input window.
        const uint32_t cached_end = cache_begin[p] + cache_rows[p];
        if (consumed[p] >= cache_begin[p] && consumed[p] < cached_end &&
            (cached_end == totals[p] ||
             uint32_t(plans.GetValue(p * plan_tokens + cache_rows[p] - 1U)) >= input_begin + input_count - 1U))
            continue;
        const uint64_t first = (uint64_t(sender.Rank(p)) * td.num_channels + channel) * stride + consumed[p];
        AscendC::DataCopyPad(plans[p * plan_tokens], pg[first], AscendC::DataCopyExtParams{1, n * 4U, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        cache_begin[p] = consumed[p];
        cache_rows[p] = n;
        loaded = true;
    }
    if (loaded) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
    }
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        if (consumed[p] == totals[p]) {
            descriptors.SetValue(p * 4U, sender.Rank(p));
            descriptors.SetValue(p * 4U + 1U, 0);
            descriptors.SetValue(p * 4U + 2U, consumed[p]);
            descriptors.SetValue(p * 4U + 3U, 0);
            continue;
        }
        const uint32_t cursor = consumed[p] - cache_begin[p];
        uint32_t low = cursor, high = cache_rows[p];
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2U;
            if (uint32_t(plans.GetValue(p * plan_tokens + middle)) < input_begin + input_count)
                low = middle + 1U;
            else
                high = middle;
        }
        descriptors.SetValue(p * 4U, sender.Rank(p));
        descriptors.SetValue(p * 4U + 1U, low - cursor);
        descriptors.SetValue(p * 4U + 2U, consumed[p]);
        descriptors.SetValue(p * 4U + 3U, cursor);
        consumed[p] += low - cursor;
    }
    // Input DMA need only complete before VF reads it, not before scalar work.
    AscendC::DataCopyPad(
        indices, ig[uint64_t(input_begin) * td.num_topk],
        AscendC::DataCopyExtParams{1, input_count * td.num_topk * uint32_t(sizeof(topk_idx_t)), 0, 0, 0},
        AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
    if (td.has_topk_weights)
        AscendC::DataCopyPad(weights, wg[uint64_t(input_begin) * td.num_topk],
                             AscendC::DataCopyExtParams{1, input_count * td.num_topk * 4U, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
}

template <uint32_t Capacity, class Resource, class Sender, class PostChannel>
__aicore__ inline void PackRemoteChannelMetadata(Resource& resource, Sender& sender, GM_ADDR plan,
                                                 GM_ADDR channel_prefix, const DispatchTilingData& td, uint32_t core,
                                                 uint32_t scratch_offset, PostChannel& post_channel)
{
    if (!sender.Count()) return;
    constexpr uint32_t descriptor_bytes = (Capacity * kChannelMetadataDescriptorWords * 4U + 31U) / 32U * 32U;
    const uint32_t slots = td.channel_metadata_double_buffer ? 2U : 1U;
    const uint32_t budget = (platform::kKernelUbBytes - scratch_offset) / slots - descriptor_bytes;
    const auto selected = SelectChannelMetadataTile(td.channel_metadata_tile_tokens, Capacity, td.num_topk,
                                                    sizeof(topk_idx_t), td.has_topk_weights, budget);
    if (!selected.tokens) {
        trap();
        return;
    }
    const uint32_t length = (td.num_tokens + td.num_channels - 1U) / td.num_channels;
    const uint32_t stride = (length + 7U) / 8U * 8U;
    const auto cached = ExtendMetadataPlanCache(selected, sender.Count(), stride, budget);
    const auto tile = cached.input;
    const uint32_t slot_bytes = descriptor_bytes + tile.bytes;
    AscendC::GlobalTensor<int32_t> prefixes;
    prefixes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix));
    for (uint32_t ch = core % td.udma_workers_per_peer; ch < td.num_channels; ch += td.udma_workers_per_peer) {
        uint32_t totals[Capacity]{}, consumed[Capacity]{}, total = 0;
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            const uint32_t peer = sender.Rank(p);
            const uint32_t previous = ch ? prefixes.GetValue(peer * td.num_channels + ch - 1U) : 0;
            totals[p] = prefixes.GetValue(peer * td.num_channels + ch) - previous;
            total += totals[p];
        }
        if (!total) continue;
        const uint32_t begin = ch * length < td.num_tokens ? ch * length : td.num_tokens;
        const uint32_t end = begin + length < td.num_tokens ? begin + length : td.num_tokens;
        bool pending[2] = {false, false};
        uint32_t cache_begin[2][Capacity]{}, cache_rows[2][Capacity]{};
        uint32_t slot = 0;
        const uint32_t initial = end - begin < tile.tokens ? end - begin : tile.tokens;
        LoadRemoteMetadataWindow<Capacity>(resource, sender, plan, td, tile, ch, stride, begin, initial, totals,
                                           consumed, scratch_offset, 0, cached.plan_tokens, cache_begin[0],
                                           cache_rows[0]);
        for (uint32_t start = begin; start < end; start += tile.tokens) {
            const auto event = slot ? EVENT_ID7 : EVENT_ID6;
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            const uint32_t next = start + tile.tokens, other = slot ^ 1U;
            const uint32_t offset = scratch_offset + slot * slot_bytes;
            auto descriptors = resource.template GetBufferByByte<uint32_t>(offset);
            auto plans = resource.template GetBufferByByte<int32_t>(offset + descriptor_bytes);
            auto indices = resource.template GetBufferByByte<topk_idx_t>(offset + descriptor_bytes + tile.index_offset);
            auto weights = resource.template GetBufferByByte<float>(offset + descriptor_bytes +
                                                                    (td.has_topk_weights ? tile.weight_offset : 0));
            asc_vf_call<PackRemoteMetadataVf>(
                dim3(128), reinterpret_cast<__ubuf__ int32_t*>(plans.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(descriptors.GetPhyAddr()),
                reinterpret_cast<__ubuf__ topk_idx_t*>(indices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr()),
                reinterpret_cast<__gm__ topk_idx_t*>(td.packed_channel_indices),
                reinterpret_cast<__gm__ float*>(td.packed_channel_weights), sender.Count(), cached.plan_tokens, start,
                td.num_topk, td.num_experts / td.world_size, td.num_channels, ch, stride, td.has_topk_weights);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(event);
            pending[slot] = true;
            // Launch the current VF before blocking on the next slot's DMA.
            if (slots == 2U && next < end) {
                if (pending[other]) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(other ? EVENT_ID7 : EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(other ? EVENT_ID7 : EVENT_ID6);
                    pending[other] = false;
                }
                LoadRemoteMetadataWindow<Capacity>(resource, sender, plan, td, tile, ch, stride, next,
                                                   end - next < tile.tokens ? end - next : tile.tokens, totals,
                                                   consumed, scratch_offset + other * slot_bytes, other,
                                                   cached.plan_tokens, cache_begin[other], cache_rows[other]);
            }
            if (slots == 1U) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(event);
                pending[slot] = false;
                if (next < end)
                    LoadRemoteMetadataWindow<Capacity>(resource, sender, plan, td, tile, ch, stride, next,
                                                       end - next < tile.tokens ? end - next : tile.tokens, totals,
                                                       consumed, scratch_offset, 0, cached.plan_tokens, cache_begin[0],
                                                       cache_rows[0]);
            }
            if (slots == 2U) slot = other;
        }
        for (uint32_t i = 0; i < slots; ++i) {
            if (pending[i]) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i ? EVENT_ID7 : EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(i ? EVENT_ID7 : EVENT_ID6);
            }
        }
        // GM staging is immutable until final Quiet. Retire this channel's
        // SIMT stores before handing its addresses to the UDMA sender.
        post_channel(ch);
    }
}

// Load a bounded sorted plan prefix, then one contiguous input window covering
// its valid rows. Sparse gaps skip straight to the next plan token. Binary
// search bounds the window without a scalar loop over tokens or top-k slots.
template <class Resource>
__aicore__ inline PlanMetadataWindow LoadPlanMetadataWindow(Resource& resource, GM_ADDR plan,
                                                            const DispatchTilingData& td,
                                                            const ChannelMetadataTile& tile, uint64_t plan_row,
                                                            uint32_t remaining, uint32_t offset, uint32_t slot)
{
    const auto event = slot ? EVENT_ID7 : EVENT_ID6;
    auto tokens = resource.template GetBufferByByte<int32_t>(offset);
    auto indices = resource.template GetBufferByByte<topk_idx_t>(offset + tile.index_offset);
    auto weights = resource.template GetBufferByByte<float>(offset + (td.has_topk_weights ? tile.weight_offset : 0U));
    AscendC::GlobalTensor<int32_t> pg;
    AscendC::GlobalTensor<topk_idx_t> ig;
    AscendC::GlobalTensor<float> wg;
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
    ig.SetGlobalBuffer(reinterpret_cast<__gm__ topk_idx_t*>(td.notify_topk_idx));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(td.notify_topk_weights));
    const uint32_t candidates = remaining < tile.tokens ? remaining : tile.tokens;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
    AscendC::DataCopyPad(tokens, pg[plan_row], AscendC::DataCopyExtParams{1, candidates * 4U, 0, 0, 0},
                         AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
    const uint32_t begin = uint32_t(tokens.GetValue(0));
    uint32_t low = 1, high = candidates;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (uint32_t(tokens.GetValue(middle)) - begin < tile.tokens)
            low = middle + 1U;
        else
            high = middle;
    }
    const uint32_t span = uint32_t(tokens.GetValue(low - 1U)) - begin + 1U;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
    AscendC::DataCopyPad(indices, ig[uint64_t(begin) * td.num_topk],
                         AscendC::DataCopyExtParams{1, span * td.num_topk * uint32_t(sizeof(topk_idx_t)), 0, 0, 0},
                         AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
    if (td.has_topk_weights)
        AscendC::DataCopyPad(weights, wg[uint64_t(begin) * td.num_topk],
                             AscendC::DataCopyExtParams{1, span * td.num_topk * 4U, 0, 0, 0},
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    return {low, begin};
}

// Each peer/channel has exactly one existing owner. Remote destinations are
// immutable per-call GM staging, retained until sender Quiet; local destinations
// are final consumer regions. Exact stores never pad across another channel.
// UB ledger per slot: aligned plan[tile] + indices[tile,topk] + weights[tile,topk].
// Two independent input slots use events 6/7. Sender descriptors stay below
// scratch_offset; neither input slot aliases WQEs, headers or queue state.
template <class Resource>
__aicore__ inline void PackPeerChannelMetadata(Resource& resource, GM_ADDR plan, const DispatchTilingData& td,
                                               uint32_t peer, uint32_t channel, uint32_t count, uint32_t stride,
                                               uint64_t output_indices, uint64_t output_weights, uint64_t output_src,
                                               uint32_t scratch_offset)
{
    if (!count) return;
    const uint32_t slots = td.channel_metadata_double_buffer ? 2U : 1U;
    const auto tile =
        SelectChannelMetadataTile(td.channel_metadata_tile_tokens, 1U, td.num_topk, sizeof(topk_idx_t),
                                  td.has_topk_weights, (platform::kKernelUbBytes - scratch_offset) / slots);
    if (!tile.tokens) {
        trap();
        return;
    }  // Host checks the same live sender budget.
    const uint64_t base = (uint64_t(peer) * td.num_channels + channel) * stride;
    bool pending[2] = {false, false};
    auto window = LoadPlanMetadataWindow(resource, plan, td, tile, base, count, scratch_offset, 0);
    uint32_t slot = 0;
    for (uint32_t row = 0; row < count;) {
        const auto event = slot ? EVENT_ID7 : EVENT_ID6;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        const uint32_t next_row = row + window.rows;
        const uint32_t other = slot ^ 1U;
        PlanMetadataWindow next{};
        const uint32_t offset = scratch_offset + slot * tile.bytes;
        auto tokens = resource.template GetBufferByByte<int32_t>(offset);
        auto indices = resource.template GetBufferByByte<topk_idx_t>(offset + tile.index_offset);
        auto weights =
            resource.template GetBufferByByte<float>(offset + (td.has_topk_weights ? tile.weight_offset : 0U));
        asc_vf_call<PackPlanMetadataVf>(
            dim3(128), reinterpret_cast<__ubuf__ int32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ topk_idx_t*>(indices.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr()),
            reinterpret_cast<__gm__ topk_idx_t*>(output_indices) + uint64_t(row) * td.num_topk,
            td.has_topk_weights ? reinterpret_cast<__gm__ float*>(output_weights) + uint64_t(row) * td.num_topk
                                : nullptr,
            output_src ? reinterpret_cast<__gm__ int32_t*>(output_src) + row : nullptr, window.rows, window.input_begin,
            td.num_topk, peer, td.num_experts / td.world_size, td.has_topk_weights);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(event);
        pending[slot] = true;
        if (slots == 2U && next_row < count) {
            if (pending[other]) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(other ? EVENT_ID7 : EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(other ? EVENT_ID7 : EVENT_ID6);
                pending[other] = false;
            }
            next = LoadPlanMetadataWindow(resource, plan, td, tile, base + next_row, count - next_row,
                                          scratch_offset + other * tile.bytes, other);
        }
        if (slots == 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(event);
            pending[slot] = false;
            if (next_row < count)
                next = LoadPlanMetadataWindow(resource, plan, td, tile, base + next_row, count - next_row,
                                              scratch_offset, 0);
        }
        row = next_row;
        window = next;
        if (slots == 2U) slot = other;
    }
    for (uint32_t i = 0; i < slots; ++i) {
        if (pending[i]) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i ? EVENT_ID7 : EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(i ? EVENT_ID7 : EVENT_ID6);
        }
    }
    // Finish SIMT GM writes before UDMA reads staging or a consumer uses self.
}
#endif
}  // namespace ascend_deepep
