// Copyright (c) 2026, DeepEP contributors
#pragma once
#include <cstdint>
#include <mutex>

// Both native modules link one registry. The C boundary also works when torch
// and the lightweight pybind module use different libstdc++ string ABIs.
extern "C" {
std::recursive_mutex* deepep_instance_mutex();
uint32_t deepep_next_instance();
uint32_t deepep_reserve_instance(const char* configuration);
void deepep_release_instance(uint32_t id);
bool deepep_has_instances();
}

namespace deepep {
using InstanceLock = std::lock_guard<std::recursive_mutex>;
inline std::recursive_mutex& instance_mutex()
{
    return *deepep_instance_mutex();
}
// Call under InstanceLock. Host malloc/free/ptr operate on the selected SDK
// context. Device launches instead carry an immutable instance ID.
void select_instance(uint32_t id);
}  // namespace deepep

extern "C" void deepep_instance_sync_do(void* stream, uint32_t instance_id, uint64_t ffts, bool wait_handle);
extern "C" void deepep_instance_barrier(uint32_t instance_id, uint64_t ffts);
