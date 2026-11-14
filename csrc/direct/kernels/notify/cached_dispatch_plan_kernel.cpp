#include "notify_dispatch_common.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void cached_dispatch_plan_kernel(
    GM_ADDR routes, GM_ADDR plan, GM_ADDR head, ascend_deepep::DispatchTilingData td, uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC { return; }
    ascend_deepep::UbWorkspace resource;
    const uint32_t channel = get_block_idx() * get_subblockdim() + get_subblockid();
    ascend_deepep::BuildDispatchPlan(resource, routes, plan, head, td, channel);
}
