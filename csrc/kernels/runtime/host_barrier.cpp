#include "kernel_operator.h"
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "../launch.hpp"
#include "host_barrier.h"

ACLSHMEM_GLOBAL void deepep_ndr_host_barrier_kernel_v2(uint64_t ffts_config, uint64_t wait_cycles, uint32_t trace)
{
    AscendC::SetSyncBaseAddr(ffts_config);
    deepep_host_barrier(ACLSHMEM_TEAM_WORLD, wait_cycles, trace != 0);
}

extern "C" DEEPEP_EXPORT void deepep_ndr_host_barrier_launch_v2(void* stream, uint64_t wait_cycles, uint32_t trace)
{
    deepep_ndr_host_barrier_kernel_v2<<<1, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(util_get_ffts_config(), wait_cycles, trace);
}
