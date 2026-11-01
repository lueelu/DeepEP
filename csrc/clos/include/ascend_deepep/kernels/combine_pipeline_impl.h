#pragma once

#include "ascend_deepep/kernels/aggregate_impl.h"
#include "ascend_deepep/kernels/udma_combine_schedule.h"

namespace ascend_deepep::udma {

template <uint32_t ReservedUbBytes = 0, class Resource>
__aicore__ inline void CopyAggregateSelf(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR workspace,
    const CombineTilingData& td, uint32_t begin, uint32_t end)
{
    auto staging = resource.ubBuf.template GetBufferByByte<uint8_t>(ReservedUbBytes);
    AscendC::GlobalTensor<uint8_t> input, values, output;
    AscendC::GlobalTensor<int32_t> indices;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    values.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(weights));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    const uint32_t row_bytes = td.hidden * td.x_element_bytes;
    const uint32_t weight_bytes = td.num_topk * sizeof(float);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (uint32_t row = begin; row < end; ++row) {
        const int32_t original = indices.GetValue(row);
        if (original < 0 || static_cast<uint32_t>(original) >= td.token_capacity) trap();
        const uint64_t slot = uint64_t(td.rank) * td.token_capacity + static_cast<uint32_t>(original);
        for (uint32_t field = 0; field < (td.has_topk_weights ? 2U : 1U); ++field) {
            const uint32_t bytes = field ? weight_bytes : row_bytes;
            constexpr uint32_t capacity = aggregate::kAggregateUbBytes - ReservedUbBytes;
            for (uint32_t offset = 0; offset < bytes; offset += capacity) {
                const uint32_t count = bytes - offset < capacity ? bytes - offset : capacity;
                AscendC::DataCopyExtParams copy{1, count, 0, 0, 0};
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::DataCopyPad(staging, field ? values[uint64_t(row) * bytes + offset] : input[uint64_t(row) * bytes + offset],
                    copy, AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(output[(field ? td.topk_weights_buffer_offset : td.x_buffer_offset) +
                    slot * (field ? td.topk_weights_row_stride : td.x_row_stride) + offset], staging, copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// All AIVs own their aggregate rows and an independent QP. Persistent sender
// metadata occupies the UB prefix; aggregate/self-copy may only use its suffix.
template <typename T>
__aicore__ inline void CombineSelfPipelineImpl(
    GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix,
    GM_ADDR workspace, const CombineTilingData& td)
{
    const uint32_t core = get_block_idx() * get_subblockdim() + get_subblockid();
    const uint32_t cores = get_block_num() * get_subblockdim();
    if (cores != kQpCount || td.num_channels != cores || td.world_size > 8 ||
        (td.pipeline_chunk_tokens != 32 && td.pipeline_chunk_tokens != 64)) trap();
    constexpr uint32_t reserved = kCombineSenderBytes + kMaxPeers * kTileTokens * sizeof(int32_t);
    static_assert(reserved % 32 == 0);
    Catlass::Arch::Resource<ascend_deepep::KernelArch> resource;
    auto flag = resource.ubBuf.template GetBufferByByte<uint64_t>(reserved);
    Begin(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, cores, flag, cores);
    auto wq = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
    auto indices = resource.ubBuf.template GetBufferByByte<int32_t>(kCombineSenderBytes);
    CombineSender sender(td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()),
        workspace, x, weights, 0, core);
    AscendC::GlobalTensor<int32_t> prefix, original;
    prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    original.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    AggregateTilingData a{};
    a.num_recv_tokens = td.num_recv_tokens;
    a.num_expanded_tokens = td.num_expanded_tokens;
    a.hidden = td.hidden;
    a.num_topk = td.pipeline_topk;
    a.block_dim = td.block_dim;
    a.has_expanded_values = td.has_topk_weights;
    a.x_dtype = td.x_dtype;
    a.x_element_bytes = td.x_element_bytes;
    a.rank = td.rank;
    a.world_size = td.world_size;
    a.single_source_slots = td.single_source_slots;
    a.rank_prefix_matrix = reinterpret_cast<uint64_t>(rank_prefix);
    const uint32_t chunk = td.pipeline_chunk_tokens;
    const uint32_t fields = td.has_topk_weights ? 2U : 1U;
    uint32_t row[kMaxPeers]{}, end[kMaxPeers]{}, chunk_end[kMaxPeers]{}, cursor[kMaxPeers]{}, field[kMaxPeers]{};
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        const uint32_t peer = sender.Rank(p);
        const uint32_t begin = peer ? prefix.GetValue((peer - 1) * td.world_size + td.rank) : 0;
        row[p] = begin + core * chunk;
        end[p] = prefix.GetValue(peer * td.world_size + td.rank);
    }
    bool pending = true;
    while (pending) {
        pending = false;
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            uint32_t visit = td.udma_wqes_per_visit;
            while (visit && row[p] < end[p]) {
                if (!chunk_end[p]) {
                    // Finish reading pending index spans before this UB window
                    // changes. Payload stays in full GM storage until quiet.
                    sender.Flush();
                    chunk_end[p] = row[p] + chunk < end[p] ? row[p] + chunk : end[p];
                    aggregate::AggregateRange<T, reserved>(resource, reinterpret_cast<GM_ADDR>(td.expanded_x), nullptr,
                        reinterpret_cast<GM_ADDR>(td.pipeline_values), reinterpret_cast<GM_ADDR>(td.pipeline_mapping),
                        x, weights, a, static_cast<int32_t>(row[p]), static_cast<int32_t>(chunk_end[p]));
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
                    const uint32_t aligned = row[p] / 8 * 8;
                    cursor[p] = row[p] - aligned;
                    AscendC::DataCopyExtParams copy{1, (chunk_end[p] - aligned) * uint32_t(sizeof(int32_t)), 0, 0, 0};
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::DataCopyPad(indices[p * kTileTokens], original[aligned], copy,
                        AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                }
                const uint32_t available = (chunk_end[p] - row[p]) * fields - field[p];
                auto* window = reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()) + p * kTileTokens + cursor[p];
                const uint32_t count = sender.PutRows(p, window, row[p], field[p], available < visit ? available : visit);
                const uint32_t rows = (field[p] + count) / fields;
                field[p] = (field[p] + count) % fields;
                cursor[p] += rows;
                row[p] += rows;
                visit -= count;
                if (row[p] == chunk_end[p]) {
                    row[p] += (cores - 1U) * chunk;
                    chunk_end[p] = 0;
                }
            }
            // Publish this visit before doing more aggregate work on another
            // peer, so NIC progress overlaps the next local calculation.
            sender.Flush();
            pending |= row[p] < end[p];
        }
    }
    const uint32_t self_begin = td.rank ? prefix.GetValue((td.rank - 1) * td.world_size + td.rank) : 0;
    const uint32_t self_end = prefix.GetValue(td.rank * td.world_size + td.rank);
    for (uint32_t begin = self_begin + core * chunk; begin < self_end; begin += cores * chunk) {
        const uint32_t end = begin + chunk < self_end ? begin + chunk : self_end;
        aggregate::AggregateRange<T, reserved>(resource, reinterpret_cast<GM_ADDR>(td.expanded_x), nullptr,
            reinterpret_cast<GM_ADDR>(td.pipeline_values), reinterpret_cast<GM_ADDR>(td.pipeline_mapping),
            x, weights, a, static_cast<int32_t>(begin), static_cast<int32_t>(end));
        CopyAggregateSelf<reserved>(resource, x, weights, src_idx, workspace, td, begin, end);
    }
    // Every core, including empty row partitions, publishes its own marker.
    // Independent QPs require all markers; one writer's SO cannot cover others.
    sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core);
    sender.Quiet();
    WaitIncoming<false>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, cores, cores);
}

template <typename T>
__aicore__ inline void CombinePipelineImpl(
    GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix,
    GM_ADDR channel_prefix, GM_ADDR workspace, const CombineTilingData& td)
{
    const uint32_t core = get_block_idx() * get_subblockdim() + get_subblockid();
    const uint32_t cores = get_block_num() * get_subblockdim();
    if (td.udma_token_schedule) {
        CombineSelfPipelineImpl<T>(x, weights, src_idx, rank_prefix, workspace, td);
        return;
    }
    const uint32_t senders = td.udma_workers_per_peer * Workers(td.world_size, td.udma_peers_per_worker, td.udma_peer_order, td.udma_ranks_per_group);
    if (cores != td.num_channels || senders >= cores || td.udma_token_schedule ||
        (td.pipeline_chunk_tokens != 32 && td.pipeline_chunk_tokens != 64)) trap();
    const uint32_t chunks = (td.num_recv_tokens + td.pipeline_chunk_tokens - 1) / td.pipeline_chunk_tokens;
    Catlass::Arch::Resource<ascend_deepep::KernelArch> resource;
    auto flag = resource.ubBuf.template GetBufferByByte<uint64_t>(0);
    for (uint32_t chunk = core; chunk < chunks; chunk += cores)
        StoreAggregateReady(td.pipeline_ready, chunk, 0, flag);
    // The existing Begin collective also orders ready-slot reset. No new
    // cross-rank collective is inserted into the communication protocol.
    Begin(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, cores, flag, td.udma_workers_per_peer);

    if (core >= senders) {
        AscendC::GlobalTensor<int32_t> prefix;
        prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
        const uint32_t self_begin = td.rank ? prefix.GetValue((td.rank - 1) * td.world_size + td.rank) : 0;
        const uint32_t self_end = prefix.GetValue(td.rank * td.world_size + td.rank);
        AggregateTilingData a{};
        a.num_recv_tokens = td.num_recv_tokens;
        a.num_expanded_tokens = td.num_expanded_tokens;
        a.hidden = td.hidden;
        a.num_topk = td.pipeline_topk;
        a.block_dim = td.block_dim;
        a.has_expanded_values = td.has_topk_weights;
        a.x_dtype = td.x_dtype;
        a.x_element_bytes = td.x_element_bytes;
        a.rank = td.rank;
        a.world_size = td.world_size;
        a.single_source_slots = td.single_source_slots;
        a.rank_prefix_matrix = reinterpret_cast<uint64_t>(rank_prefix);
        for (uint32_t chunk = core - senders; chunk < chunks; chunk += cores - senders) {
            const uint32_t begin = chunk * td.pipeline_chunk_tokens;
            const uint32_t end = begin + td.pipeline_chunk_tokens < td.num_recv_tokens ?
                begin + td.pipeline_chunk_tokens : td.num_recv_tokens;
            aggregate::AggregateRange<T>(resource, reinterpret_cast<GM_ADDR>(td.expanded_x), nullptr,
                reinterpret_cast<GM_ADDR>(td.pipeline_values), reinterpret_cast<GM_ADDR>(td.pipeline_mapping),
                x, weights, a, static_cast<int32_t>(begin), static_cast<int32_t>(end));
            // Range drains every actual MTE3/V operation, including mapped
            // values, before the ordered ready store on the same MTE3 pipeline.
            const uint32_t local_begin = begin > self_begin ? begin : self_begin;
            const uint32_t local_end = end < self_end ? end : self_end;
            if (local_begin < local_end)
                CopyAggregateSelf(resource, x, weights, src_idx, workspace, td, local_begin, local_end);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
            StoreAggregateReady(td.pipeline_ready, chunk, 1, flag);
        }
    }
    // Producers have no QP peers, but must join all original completion waits.
    CombineRemote<3>(resource, x, weights, src_idx, rank_prefix, channel_prefix, workspace, td, core);
}
} // namespace ascend_deepep::udma
