#pragma once

#include <cstdint>

#include "kernels/common/ub_workspace.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "kernels/notify/notify_recv_prefix.h"
#include "kernels/notify/notify_dispatch_plan.h"
#include "kernels/common/v2_handle.h"
#include "common/launch.h"
#include "common/types.h"

using namespace AscendC;

namespace {

constexpr int32_t kNotifyInt32Bytes = 4;
constexpr int32_t kNotifyInt32AlignElems = 8;
constexpr int32_t kNotifyDispatchUbBytes = static_cast<int32_t>(ascend_deepep::platform::kKernelUbBytes);
constexpr int32_t kMaxVectorTileElems = 16 * 1024;
constexpr int32_t kRouteRowUbOffset = 2 * kMaxVectorTileElems * kNotifyInt32Bytes;
constexpr int32_t kRouteHalfUbOffset = kRouteRowUbOffset + kMaxVectorTileElems * sizeof(uint8_t);
constexpr int32_t kRouteHalfUbEnd = kRouteHalfUbOffset + kMaxVectorTileElems * sizeof(half);
static_assert(kRouteHalfUbEnd <= kNotifyDispatchUbBytes);

__aicore__ inline int32_t NotifyIntMin(int32_t lhs, int32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

} // namespace
