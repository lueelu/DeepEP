#include "combine_udma_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void v1_combine_udma_kernel(
    GM_ADDR x, GM_ADDR topk_weights, GM_ADDR bias_0, GM_ADDR bias_1, GM_ADDR src_idx, GM_ADDR rank_prefix_matrix,
    GM_ADDR channel_prefix_matrix, GM_ADDR send_head, GM_ADDR symmetric_buffer, GM_ADDR recv_x,
    GM_ADDR recv_topk_weights, ascend_deepep::CombineTilingData tiling, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    if (tiling.x_dtype == kCombineDtypeBfloat16) {
        V1CombineUdmaImpl<bfloat16_t>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    } else if (tiling.x_dtype == kCombineDtypeFloat16) {
        V1CombineUdmaImpl<half>(
            x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
            symmetric_buffer, recv_x, recv_topk_weights, tiling);
    }
}
