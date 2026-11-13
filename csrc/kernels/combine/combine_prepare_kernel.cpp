#include "combine_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_prepare_kernel(
    GM_ADDR workspace, ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Prepare(workspace, tiling);
#endif
}
