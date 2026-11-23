#pragma once

#include "kernel_operator.h"
#include "common/platform.h"
#include "kernels/common/v2_handle.h"

namespace ascend_deepep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
struct V2SourceTile {
    uint32_t rows = 0;
    uint32_t source_offset = 0;
    uint32_t topk_offset = 0;
    uint32_t output_offset = 0;
    uint32_t bytes = 0;
};

// All tile capacities are multiples of eight rows, so every UB region starts
// on a 32-byte boundary for either supported top-k index width. A logical tail
// may be shorter; only its valid bytes are transferred to/from compact GM.
__aicore__ inline V2SourceTile MakeV2SourceTile(uint32_t world, uint32_t k, uint32_t budget)
{
    const uint64_t prefix_bytes = (uint64_t(world) * sizeof(int32_t) + 31U) / 32U * 32U;
    const uint64_t row_bytes = sizeof(int32_t) + uint64_t(k) * sizeof(topk_idx_t) +
        (uint64_t(k) + 2U) * sizeof(int32_t);
    if (!world || prefix_bytes >= budget) return {};
    uint64_t rows = (budget - prefix_bytes) / row_bytes;
    rows = (rows < 256U ? rows : 256U) / 8U * 8U;
    if (!rows) return {};
    V2SourceTile tile;
    tile.rows = uint32_t(rows);
    tile.source_offset = uint32_t(prefix_bytes);
    tile.topk_offset = uint32_t(prefix_bytes + rows * sizeof(int32_t));
    tile.output_offset = uint32_t(tile.topk_offset + rows * uint64_t(k) * sizeof(topk_idx_t));
    tile.bytes = uint32_t(prefix_bytes + rows * row_bytes);
    return tile;
}

// Neighboring SIMT lanes write adjacent metadata elements in UB. Only the
// first two columns read source/top-k, and every output column is written once.
// The caller's interval belongs to one source rank, including repeated/empty
// prefixes. Padding intervals do not read either input pointer.
__simt_vf__ __launch_bounds__(128) inline void V2HandleSourceTileVf(
    const __ubuf__ int32_t* source, const __ubuf__ topk_idx_t* topk, __ubuf__ int32_t* output,
    uint32_t rows, uint32_t k, uint64_t source_base, uint32_t master_base, bool valid)
{
    const uint32_t columns = k + 2U;
    for (uint32_t item = threadIdx.x; item < rows * columns; item += 128U) {
        const uint32_t row = item / columns, col = item % columns;
        int32_t value = -1;
        if (valid && col == 0) value = int32_t(source_base + source[row]);
        if (valid && col == 1) {
            uint32_t master = 0;
            while (master < k && topk[uint64_t(row) * k + master] < 0) ++master;
            value = int32_t(master_base + master);
        }
        output[item] = value;
    }
}

// Called only after transport quiet/local completion and all output-copy UB
// credits have drained. Reuse the existing arena; no new tensor or collective.
// The original eight-row core ownership is preserved even when a rank interval
// ends on an unaligned row. Different cores never share an output cache line.
template <class Resource>
__aicore__ inline void BuildV2HandleSource(
    Resource& resource, const __gm__ int32_t* prefix, const __gm__ int32_t* ranks,
    const __gm__ int32_t* source, const __gm__ topk_idx_t* topk, __gm__ int32_t* output,
    uint32_t rank, uint32_t world, uint32_t capacity, uint32_t k,
    uint32_t rows, uint32_t core, uint32_t cores)
{
    const uint64_t groups = (uint64_t(rows) + 7U) / 8U;
    const uint64_t per_core = (groups + cores - 1U) / cores;
    const uint64_t first = uint64_t(core) * per_core * 8U;
    if (first >= rows) return;
    const uint64_t limit = uint64_t(core + 1U) * per_core * 8U;
    const uint32_t end = uint32_t(limit < rows ? limit : rows);
    const auto tile = MakeV2SourceTile(world, k, uint32_t(platform::kKernelUbBytes));
    if (!tile.rows) {
        // Preserve support for shapes whose input/output rows cannot fit UB.
        asc_vf_call<V2HandleSourceVf>(dim3(128), ranks, source, topk, output,
            rank, world, capacity, k, rows, core, cores);
        return;
    }
    auto prefix_ub = resource.template GetBufferByByte<int32_t>(0);
    auto source_ub = resource.template GetBufferByByte<int32_t>(tile.source_offset);
    auto topk_ub = resource.template GetBufferByByte<topk_idx_t>(tile.topk_offset);
    auto output_ub = resource.template GetBufferByByte<int32_t>(tile.output_offset);
    AscendC::GlobalTensor<int32_t> prefix_gm, source_gm, output_gm;
    AscendC::GlobalTensor<topk_idx_t> topk_gm;
    prefix_gm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(prefix));
    source_gm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(source));
    topk_gm.SetGlobalBuffer(const_cast<__gm__ topk_idx_t*>(topk));
    output_gm.SetGlobalBuffer(output);
    AscendC::DataCopyPad(prefix_ub, prefix_gm,
        AscendC::DataCopyExtParams{1, world * uint32_t(sizeof(int32_t)), 0, 0, 0},
        AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
    const uint32_t valid = uint32_t(prefix_ub.GetValue(world - 1U));
    uint32_t row = uint32_t(first), source_rank = 0;
    while (row < end) {
        uint32_t interval_end = end;
        const bool is_valid = row < valid;
        if (is_valid) {
            while (row >= uint32_t(prefix_ub.GetValue(source_rank))) ++source_rank;
            const uint32_t rank_end = uint32_t(prefix_ub.GetValue(source_rank));
            interval_end = rank_end < end ? rank_end : end;
        }
        const uint64_t source_base = uint64_t(source_rank) * capacity;
        const uint32_t master_base = source_rank * k;
        // No per-row rank lookup. Every tile in this interval shares both bases.
        while (row < interval_end) {
            const uint32_t count = interval_end - row < tile.rows ? interval_end - row : tile.rows;
            if (is_valid) {
                AscendC::DataCopyPad(source_ub, source_gm[row],
                    AscendC::DataCopyExtParams{1, count * uint32_t(sizeof(int32_t)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                if (k) {
                    AscendC::DataCopyPad(topk_ub, topk_gm[uint64_t(row) * k],
                        AscendC::DataCopyExtParams{1, count * k * uint32_t(sizeof(topk_idx_t)), 0, 0, 0},
                        AscendC::DataCopyPadExtParams<topk_idx_t>{false, 0, 0, 0});
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            }
            asc_vf_call<V2HandleSourceTileVf>(dim3(128),
                reinterpret_cast<const __ubuf__ int32_t*>(source_ub.GetPhyAddr()),
                reinterpret_cast<const __ubuf__ topk_idx_t*>(topk_ub.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(output_ub.GetPhyAddr()),
                count, k, source_base, master_base, is_valid);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::DataCopyPad(output_gm[uint64_t(row) * (k + 2U)], output_ub,
                AscendC::DataCopyExtParams{1, count * (k + 2U) * uint32_t(sizeof(int32_t)), 0, 0, 0});
            // Retire output reads before the next VF writes output UB, and
            // input reads before the next MTE2 refill. Neither is a peer wait.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            if (is_valid) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
            row += count;
        }
    }
    // Complete output stores before timing/arena reuse by the caller.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
}
#endif
} // namespace ascend_deepep
