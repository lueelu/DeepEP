#include "notify_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void notify_full_kernel(
    GM_ADDR input, GM_ADDR workspace, GM_ADDR meta, GM_ADDR hist, GM_ADDR bases, GM_ADDR prefix, GM_ADDR counts,
    GM_ADDR gateways, GM_ADDR local_dst, GM_ADDR mask, GM_ADDR dst, GM_ADDR forward, GM_ADDR forward_counts,
    GM_ADDR backward, GM_ADDR backward_counts, GM_ADDR ranges, GM_ADDR masks, GM_ADDR rank_rows, GM_ADDR gather_rows,
    GM_ADDR weight_return_meta, ascend_deepep::NotifyTiling t)
{
#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 3510
    util_set_ffts_config(0U);
    // 核数在入口选择编译期特化，SIMT 热循环中的除法仍可优化成常量移位。
    if (t.num_cores == 64U) {
        if (t.rank_dedup) {
            FullNotify<64U, false, true>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask,
                                         dst, forward, forward_counts, backward, backward_counts, ranges, masks,
                                         rank_rows, gather_rows, weight_return_meta, t);
        } else {
            FullNotify<64U>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask, dst,
                            forward, forward_counts, backward, backward_counts, ranges, masks, rank_rows, gather_rows,
                            weight_return_meta, t);
        }
    } else {
        if (t.rank_dedup) {
            FullNotify<32U, false, true>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask,
                                         dst, forward, forward_counts, backward, backward_counts, ranges, masks,
                                         rank_rows, gather_rows, weight_return_meta, t);
        } else {
            FullNotify<32U>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask, dst,
                            forward, forward_counts, backward, backward_counts, ranges, masks, rank_rows, gather_rows,
                            weight_return_meta, t);
        }
    }
#endif
}
