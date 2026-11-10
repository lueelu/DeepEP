#pragma once
#include <cstdint>
#include "kernels/mega_moe_urma/abi.hpp"
namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define SHARED_INLINE __aicore__ inline
#else
#define SHARED_INLINE inline
#endif
SHARED_INLINE constexpr uint32_t SharedTaskCount(uint32_t bs, uint32_t intermediate)
{
    return ((bs + 255U) / 256U) * (intermediate / 256U) * 2U;
}
SHARED_INLINE constexpr uint32_t SharedOwnedTasks(uint32_t total, uint32_t worker)
{
    return worker < total ? (total - 1U - worker) / 28U + 1U : 0U;
}
struct SharedTask {
    uint32_t group, tile, slice;
};
SHARED_INLINE constexpr SharedTask SharedTaskAt(uint32_t task, uint32_t intermediate)
{
    const uint32_t n = intermediate / 256U;
    return {task / (n * 2U), (task / 2U) % n, task % 2U};
}
constexpr uint32_t kSharedMaxOwnedTasks = SharedOwnedTasks(SharedTaskCount(kMaxTokensPerRank, 16384U), 0U);
constexpr uint32_t kSharedDoneWords = (kSharedMaxOwnedTasks + 63U) / 64U;
static_assert(kSharedMaxOwnedTasks == 293U && kSharedDoneWords == 5U);
#undef SHARED_INLINE
}  // namespace MegaMoeUrma
