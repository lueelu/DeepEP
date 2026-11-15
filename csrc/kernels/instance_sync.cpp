// Copyright (c) 2026, DeepEP contributors
#include "kernel_operator.h"
#include "shmem.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void deepep_instance_sync_kernel(uint32_t instance_id,
                                                                                                uint64_t ffts,
                                                                                                bool wait_handle)
{
    aclshmemx_instance_ctx_set(instance_id);
    AscendC::SetSyncBaseAddr(ffts);
    if (wait_handle)
        aclshmemi_handle(ACLSHMEM_TEAM_WORLD);
    else
        aclshmemx_barrier_all_vec();
}

extern "C" __attribute__((visibility("default"))) void deepep_instance_sync_do(void* stream, uint32_t instance_id,
                                                                               uint64_t ffts, bool wait_handle)
{
    deepep_instance_sync_kernel<<<1, nullptr, stream>>>(instance_id, ffts, wait_handle);
}
