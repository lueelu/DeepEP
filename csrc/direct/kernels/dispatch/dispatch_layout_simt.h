#pragma once

#include "common/simt_layout.h"
#include "common/platform.h"
#include "common/types.h"

namespace ascend_deepep {
// These VFs deliberately accept UB pointers only. All GM movement belongs to
// AIV DMA; even partial-count reduction never gathers GM from SIMT threads.
__simt_vf__ __launch_bounds__(128) inline void LayoutInitHistogramVf(__ubuf__ int32_t* histogram, uint32_t lanes,
                                                                     uint32_t columns)
{
    for (uint32_t i = threadIdx.x; i < lanes * columns; i += 128) histogram[i] = 0;
}

__simt_vf__ __launch_bounds__(128) inline void LayoutRoutesVf(const __ubuf__ topk_idx_t* topk, __ubuf__ uint8_t* mask,
                                                              __ubuf__ int32_t* histogram, uint32_t tokens, uint32_t k,
                                                              uint32_t world, uint32_t experts, uint32_t rank_stride,
                                                              uint32_t columns, uint32_t lanes)
{
    const uint32_t lane = threadIdx.x;
    if (lane >= lanes) return;
    auto* counts = histogram + lane * columns;
    for (uint32_t t = lane; t < tokens; t += lanes) {
        auto* row = mask + t * world;
        for (uint32_t p = 0; p < world; ++p) row[p] = 0;
        for (uint32_t j = 0; j < k; ++j) {
            const auto e = topk[t * k + j];
            if (e == -1) continue;
            if (e < 0 || uint64_t(e) >= experts) {
                __asc_simt_vf::__trap();
                return;
            }
            const uint32_t expert = uint32_t(e);
            ++counts[rank_stride + expert];
            const uint32_t peer = expert / (experts / world);
            if (!row[peer]) {
                row[peer] = 1;
                ++counts[peer];
            }
        }
    }
}

__simt_vf__ __launch_bounds__(128) inline void LayoutReduceUbVf(const __ubuf__ int32_t* input, __ubuf__ int32_t* output,
                                                                uint32_t rows, uint32_t columns)
{
    for (uint32_t c = threadIdx.x; c < columns; c += 128) {
        int32_t total = 0;
        for (uint32_t r = 0; r < rows; ++r) total += input[r * columns + c];
        output[c] = total;
    }
}

template <class Resource>
__aicore__ inline void BuildSimtLayoutPartials(Resource& resource, GM_ADDR topk_ptr, GM_ADDR mask_ptr,
                                               GM_ADDR scratch_ptr, const DispatchLayoutTilingData& td, uint32_t core,
                                               uint32_t cores)
{
    const auto plan =
        MakeSimtLayoutUbPlan(td.world_size, td.num_experts, td.num_topk, sizeof(topk_idx_t), platform::kKernelUbBytes);
    if (!plan.tile) {
        trap();
        return;
    }
    auto histogram = resource.template GetBufferByByte<int32_t>(plan.histogram_offset);
    auto partial = resource.template GetBufferByByte<int32_t>(plan.output_offset);
    auto* hist = reinterpret_cast<__ubuf__ int32_t*>(histogram.GetPhyAddr());
    asc_vf_call<LayoutInitHistogramVf>(dim3(128), hist, plan.lanes, plan.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::GlobalTensor<topk_idx_t> input;
    AscendC::GlobalTensor<uint8_t> output;
    AscendC::GlobalTensor<int32_t> scratch;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ topk_idx_t*>(topk_ptr));
    output.SetGlobalBuffer(mask_ptr);
    scratch.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_ptr));
    const uint32_t begin = uint64_t(td.num_tokens) * core / cores;
    const uint32_t end = uint64_t(td.num_tokens) * (core + 1U) / cores;
    bool input_credit[2] = {false, false};
    bool output_credit[2] = {false, false};
    // Event slots 0/1 belong to the independent ping/pong buffers. No event
    // waits are issued for empty token ranges or absent (K=0) input transfers.
    for (uint32_t start = begin, batch = 0; start < end; start += plan.tile, ++batch) {
        const uint32_t slot = batch % 2U;
        const uint32_t size = end - start < plan.tile ? end - start : plan.tile;
        auto indices = resource.template GetBufferByByte<topk_idx_t>(slot * plan.input_bytes);
        auto mask = resource.template GetBufferByByte<uint8_t>(2U * plan.input_bytes + slot * plan.mask_bytes);
        if (td.num_topk) {
            if (!batch) {
                const AscendC::DataCopyExtParams copy(1, size * td.num_topk * sizeof(topk_idx_t), 0, 0, 0);
                AscendC::DataCopyPad(indices, input[uint64_t(start) * td.num_topk], copy,
                                     AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
            const uint32_t next = start + size;
            if (next < end) {
                const uint32_t other = slot ^ 1U;
                if (input_credit[other]) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(other);
                    input_credit[other] = false;
                }
                const uint32_t n = end - next < plan.tile ? end - next : plan.tile;
                auto next_indices = resource.template GetBufferByByte<topk_idx_t>(other * plan.input_bytes);
                const AscendC::DataCopyExtParams copy(1, n * td.num_topk * sizeof(topk_idx_t), 0, 0, 0);
                AscendC::DataCopyPad(next_indices, input[uint64_t(next) * td.num_topk], copy,
                                     AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(other);
            }
        }
        if (output_credit[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
            output_credit[slot] = false;
        }
        asc_vf_call<LayoutRoutesVf>(dim3(128), reinterpret_cast<const __ubuf__ topk_idx_t*>(indices.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()), hist, size, td.num_topk,
                                    td.world_size, td.num_experts, td.rank_stride, plan.row_elems, plan.lanes);
        // Histogram updates must finish before the next VF accumulates to it.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        if (td.num_topk) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
            input_credit[slot] = true;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        const AscendC::DataCopyExtParams store(1, size * td.world_size, 0, 0, 0);
        AscendC::DataCopyPad(output[uint64_t(start) * td.world_size], mask, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        output_credit[slot] = true;
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (input_credit[slot]) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
        if (output_credit[slot]) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
    }
    asc_vf_call<LayoutReduceUbVf>(dim3(128), hist, reinterpret_cast<__ubuf__ int32_t*>(partial.GetPhyAddr()),
                                  plan.lanes, plan.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    const AscendC::DataCopyExtParams rank_copy(1, td.rank_stride * 4U, 0, 0, 0);
    const AscendC::DataCopyExtParams expert_copy(1, td.expert_stride * 4U, 0, 0, 0);
    AscendC::DataCopyPad(scratch[uint64_t(core) * td.rank_stride], partial, rank_copy);
    AscendC::DataCopyPad(scratch[uint64_t(cores) * td.rank_stride + uint64_t(core) * td.expert_stride],
                         partial[td.rank_stride], expert_copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}

template <class Resource>
__aicore__ inline void ReduceSimtLayoutPartials(Resource& resource, GM_ADDR scratch_ptr, GM_ADDR rank_ptr,
                                                GM_ADDR expert_ptr, const DispatchLayoutTilingData& td, uint32_t core,
                                                uint32_t cores)
{
    AscendC::GlobalTensor<int32_t> scratch, ranks, experts;
    scratch.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_ptr));
    ranks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_ptr));
    experts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_ptr));
    const uint32_t rank_blocks = td.rank_stride / 8U;
    const uint32_t blocks = rank_blocks + td.expert_stride / 8U;
    uint32_t block = uint64_t(blocks) * core / cores;
    const uint32_t end = uint64_t(blocks) * (core + 1U) / cores;
    const uint32_t capacity = platform::kKernelUbBytes / (cores + 1U) / 32U;
    while (block < end) {
        const bool is_rank = block < rank_blocks;
        const uint32_t field_end = is_rank ? rank_blocks : blocks;
        uint32_t count = (end < field_end ? end : field_end) - block;
        count = count < capacity ? count : capacity;
        const uint32_t width = count * 8U;
        const uint32_t column = (is_rank ? block : block - rank_blocks) * 8U;
        const uint32_t stride = is_rank ? td.rank_stride : td.expert_stride;
        const uint64_t offset = (is_rank ? 0U : uint64_t(cores) * td.rank_stride) + column;
        auto input = resource.template GetBufferByByte<int32_t>(0);
        auto output = resource.template GetBufferByByte<int32_t>(cores * width * 4U);
        const AscendC::DataCopyExtParams load(cores, width * 4U, (stride - width) * 4U, 0, 0);
        AscendC::DataCopyPad(input, scratch[offset], load, AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        asc_vf_call<LayoutReduceUbVf>(dim3(128), reinterpret_cast<const __ubuf__ int32_t*>(input.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ int32_t*>(output.GetPhyAddr()), cores, width);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        const AscendC::DataCopyExtParams store(1, width * 4U, 0, 0, 0);
        if (is_rank)
            AscendC::DataCopyPad(ranks[column], output, store);
        else
            AscendC::DataCopyPad(experts[column], output, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        block += count;
    }
}
}  // namespace ascend_deepep
