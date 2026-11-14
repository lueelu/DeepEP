#include "notify_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void notify_dispatch_barrier_kernel()
{
#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 3510
    util_set_ffts_config(0U);
    aclshmemx_barrier_all_vec();
#endif
}
