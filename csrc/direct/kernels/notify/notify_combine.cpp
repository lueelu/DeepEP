#include "shmem.h"

#include "common/launch.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void cached_notify_combine_kernel(uint64_t ffts_addr)
{
    util_set_ffts_config(ffts_addr);
    aclshmemi_sync_npu_v4<true>(0);
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_combine_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr)
{
    cached_notify_combine_kernel<<<block_dim, nullptr, stream>>>(ffts_addr);
}
