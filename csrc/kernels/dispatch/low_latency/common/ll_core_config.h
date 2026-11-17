#ifndef DEEPEP_LL_CORE_CONFIG_H
#define DEEPEP_LL_CORE_CONFIG_H
#include <cstdint>
#include "ll_compile_attrs.h"
namespace DispatchDedup {
// Count frames and the full Dispatch C API are one wire. The remote/local
// histogram split is gone, so this protocol is the only LL count identity.
constexpr uint32_t kLlCoreProtocolVersion = 36U; // multi-host direct counts, 512B DataAsFlag frames, raw C API
// Capacity is not participation. Pure AIV launches use the queried even count.
constexpr uint32_t kLlMaxCoreNum = 72U;
constexpr uint32_t kLlMaxRecvCoreNum = kLlMaxCoreNum / 2U;
constexpr uint32_t kLlDefaultPlanCoreNum = 64U; // CPU-only legacy helper default
constexpr uint32_t kLlNotifyCoreNum = 8U;
// Physical UB capacity. Host admission and every device layout read this.
constexpr uint32_t kLlUbBytes = 190U * 1024U;
constexpr __aicore__ __forceinline__ bool LlCoreCountValid(uint32_t cores) {
    return cores >= 16U && cores <= kLlMaxCoreNum && cores % 2U == 0U;
}
}

#endif
