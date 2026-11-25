#pragma once
#include <algorithm>
#include <acl/acl.h>
#include <stdexcept>
#include "tiling/platform/platform_ascendc.h"
#include <deep_ep/common/compiled.hpp>
#include "runtime/jit.hpp"
namespace deepep {
struct LlPlatformCores { uint32_t platform, available, active; int32_t device; };
inline LlPlatformCores query_ll_core_config() {
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) throw std::runtime_error("Cannot query AscendC platform for LL");
    const uint32_t total = platform->GetCoreNumAiv();
    int32_t device = -1;
    uint32_t available = total;
    if (aclrtGetDevice(&device) == ACL_SUCCESS) {
        int64_t value = 0;
        if (aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &value) != ACL_SUCCESS || value <= 0)
            throw std::runtime_error("Cannot query current-device AIV availability for LL");
        available = static_cast<uint32_t>(value);
    } else {
        device = -1; // Host-only planning: no current device is bound.
    }
    const uint32_t active = std::min(total, available);
    if (!deep_ep::LlCoreCountValid(active) ||
        (device >= 0 && active > uint32_t(deep_ep::jit_num_vec_cores())))
        throw std::runtime_error("LL requires an even AIV count within the current device's vector cores");
    return {total, available, active, device};
}
}
