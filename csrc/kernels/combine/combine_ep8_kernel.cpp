#define DEEPEP_COMBINE_EP8 1
#define combine_kernel combine_ep8_kernel
#include "combine_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_kernel(
    GM_ADDR workspace, GM_ADDR input, GM_ADDR forward, GM_ADDR counts, GM_ADDR backward, GM_ADDR backward_counts,
    GM_ADDR mask, GM_ADDR ranges, GM_ADDR masks, GM_ADDR weights, GM_ADDR weight_meta,
    ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Combine<false>(workspace, input, forward, counts, backward, backward_counts, mask, ranges, masks, nullptr, weights,
                   weight_meta, tiling);
#endif
}
