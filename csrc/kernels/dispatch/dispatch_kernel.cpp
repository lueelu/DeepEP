#include "dispatch_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_kernel(GM_ADDR workspace,
                                                                                    GM_ADDR destinations,
                                                                                    GM_ADDR forward, GM_ADDR counts,
                                                                                    GM_ADDR weights, GM_ADDR scales,
                                                                                    ascend_deepep::DispatchTiling t)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Dispatch<false>(workspace, destinations, forward, counts, weights, scales, t);
#endif
}
