#include "kernels/aggregate/aggregate_impl.h"

namespace {
using namespace ascend_deepep::aggregate;
template <typename T>
__aicore__ inline void AggregateImpl(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR expanded_values, GM_ADDR recv_to_expanded_idx,
    GM_ADDR aggregated_x, GM_ADDR aggregated_values, const ascend_deepep::AggregateTilingData& tiling)
{
    const int32_t core = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t cores = get_block_num() * get_subblockdim();
    const int32_t rows = static_cast<int32_t>(tiling.num_recv_tokens);
    const int32_t per_core = (rows + cores - 1) / cores;
    const int32_t begin = AggregateMin(core * per_core, rows);
    ascend_deepep::UbWorkspace resource;
    AggregateRange<T>(resource, x, topk_weights, expanded_values, recv_to_expanded_idx,
        aggregated_x, aggregated_values, tiling, begin, AggregateMin(begin + per_core, rows));
}
} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void aggregate_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR expanded_values, GM_ADDR recv_to_expanded_idx, GM_ADDR aggregated_x,
    GM_ADDR aggregated_values, ascend_deepep::AggregateTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kAggregateDtypeBfloat16) {
        AggregateImpl<bfloat16_t>(
            x, topk_weights, expanded_values, recv_to_expanded_idx, aggregated_x, aggregated_values, tiling);
    } else if (tiling.x_dtype == kAggregateDtypeFloat16) {
        AggregateImpl<half>(
            x, topk_weights, expanded_values, recv_to_expanded_idx, aggregated_x, aggregated_values, tiling);
    }
}
