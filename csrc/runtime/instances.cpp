// Copyright (c) 2026, DeepEP contributors
#include "instances.hpp"
#include <acl/acl.h>
#include <array>
#include <stdexcept>
#include <string>
#include "host/init/shmem_host_init.h"

namespace {
std::recursive_mutex mutex;
// 128 IDs work with both HBM-only and HBM+DRAM SDK builds.
std::array<bool, 128> occupied{};
std::string configuration;
}  // namespace

extern "C" std::recursive_mutex* deepep_instance_mutex()
{
    return &mutex;
}
extern "C" bool deepep_has_instances()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (bool used : occupied)
        if (used) return true;
    return false;
}
extern "C" uint32_t deepep_next_instance()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (uint32_t id = 0; id < occupied.size(); ++id)
        if (!occupied[id]) return id;
    throw std::runtime_error("All DeepEP SHMEM instance IDs are in use");
}
extern "C" uint32_t deepep_reserve_instance(const char* requested)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!requested) throw std::invalid_argument("Missing SHMEM configuration");
    if (deepep_has_instances()) {
        if (configuration != requested)
            throw std::runtime_error(
                "Live Buffers require the same device, EP membership and SHMEM transport configuration");
    } else {
        if (aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED)
            throw std::runtime_error("SHMEM is owned outside DeepEP");
        configuration = requested;
    }
    const auto id = deepep_next_instance();
    occupied[id] = true;
    return id;
}
extern "C" void deepep_release_instance(uint32_t id)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (id >= occupied.size() || !occupied[id]) throw std::runtime_error("Invalid SHMEM instance release");
    occupied[id] = false;
    if (!deepep_has_instances()) configuration.clear();
}
namespace deepep {
void select_instance(uint32_t id)
{
    if (aclshmemx_instance_ctx_set(id) != 0) throw std::runtime_error("Cannot select Buffer SHMEM instance");
}
}  // namespace deepep

extern "C" void deepep_instance_barrier(uint32_t id, uint64_t ffts)
{
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) throw std::runtime_error("Cannot create barrier stream");
    deepep_instance_sync_do(stream, id, ffts, false);
    const auto status = aclrtSynchronizeStream(stream);
    const auto cleanup = aclrtDestroyStream(stream);
    if (status != ACL_SUCCESS || cleanup != ACL_SUCCESS) throw std::runtime_error("Buffer instance barrier failed");
}
