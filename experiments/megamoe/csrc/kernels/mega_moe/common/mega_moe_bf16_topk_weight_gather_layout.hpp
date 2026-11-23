// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstdint>

#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_ROUTING_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_BF16_ROUTING_HOST_DEVICE
#endif

namespace MegaMoeBF16Routing {

// Keep every source window below the BF16 Gather byte-offset limit. The
// same 128 KiB budget bounds FP32 windows in the short-lived gather phase.
template <uint32_t ElementBytes>
struct TopkWeightGatherPlanFor {
    static_assert(ElementBytes == 2 || ElementBytes == 4, "TopK weights must use two or four bytes");
    static constexpr uint32_t kElementBytes = ElementBytes;
    static constexpr uint32_t kSourceWindowBytes = 128 * 1024;
    static constexpr uint32_t kMaxRoutes = kSourceWindowBytes / kElementBytes;
    static constexpr uint32_t kTwoWindowMaxRoutes = 2 * kMaxRoutes;
    static constexpr uint32_t kBatchRoutes = 2048;
    static constexpr uint32_t kIndexBytes = kBatchRoutes * sizeof(uint32_t);
    static constexpr uint32_t kOutputBytes = kBatchRoutes * kElementBytes;
    static constexpr uint32_t kMaxUbBytes = kSourceWindowBytes + kIndexBytes + kOutputBytes;
    static constexpr uint32_t kTwoWindowUbBytes = kMaxUbBytes + kIndexBytes + kOutputBytes;

    bool enabled = false;
    uint32_t sourceBytes = 0;
    uint32_t begin = 0;
    uint32_t count = 0;
    bool twoWindows = false;
    uint32_t windowCount = 0;

    MEGA_MOE_BF16_ROUTING_HOST_DEVICE constexpr TopkWeightGatherPlanFor(uint64_t totalRoutes, uint32_t coreIndex,
                                                                        uint32_t coreCount,
                                                                        bool allowMultipleWindows = false)
    {
        // Routing indices are signed int32. Larger inputs use the caller's
        // scalar fallback instead of truncating window coordinates.
        if (totalRoutes == 0 || totalRoutes > 0x7FFFFFFFULL || (!allowMultipleWindows && totalRoutes > kMaxRoutes) ||
            coreCount == 0 || coreIndex >= coreCount) {
            return;
        }
        enabled = true;
        windowCount = static_cast<uint32_t>((totalRoutes + kMaxRoutes - 1) / kMaxRoutes);
        twoWindows = windowCount == 2;
        sourceBytes =
            windowCount > 1 ? kSourceWindowBytes : static_cast<uint32_t>((totalRoutes * kElementBytes + 31) / 32 * 32);
        const uint64_t perCore = (totalRoutes + coreCount - 1) / coreCount;
        const uint64_t start = perCore * coreIndex;
        begin = static_cast<uint32_t>(start < totalRoutes ? start : totalRoutes);
        const uint64_t remaining = totalRoutes - begin;
        count = static_cast<uint32_t>(remaining < perCore ? remaining : perCore);
    }

    MEGA_MOE_BF16_ROUTING_HOST_DEVICE constexpr uint32_t BatchCount(uint32_t offset) const
    {
        if (offset >= count) {
            return 0;
        }
        return count - offset < kBatchRoutes ? count - offset : kBatchRoutes;
    }

    MEGA_MOE_BF16_ROUTING_HOST_DEVICE static constexpr uint32_t WindowRows(uint64_t totalRoutes, uint32_t window)
    {
        const uint64_t first = static_cast<uint64_t>(window) * kMaxRoutes;
        if (first >= totalRoutes) {
            return 0;
        }
        const uint64_t remaining = totalRoutes - first;
        return static_cast<uint32_t>(remaining < kMaxRoutes ? remaining : kMaxRoutes);
    }
};

using TopkWeightGatherPlan = TopkWeightGatherPlanFor<sizeof(uint16_t)>;
using Fp32TopkWeightGatherPlan = TopkWeightGatherPlanFor<sizeof(float)>;

static_assert(TopkWeightGatherPlan::kMaxUbBytes == 140 * 1024,
              "BF16 routed-weight gather must fit its independent A5 UB phase");
static_assert(TopkWeightGatherPlan::kTwoWindowUbBytes == 152 * 1024,
              "Two windows reuse one source table, with bounded offsets and merge buffers");
static_assert(Fp32TopkWeightGatherPlan::kMaxUbBytes == 144 * 1024,
              "FP32 single-window gather keeps its source and batch buffers bounded");
static_assert(Fp32TopkWeightGatherPlan::kTwoWindowUbBytes == 160 * 1024,
              "FP32 multi-window gather reuses one source window and two output batches");

}  // namespace MegaMoeBF16Routing

#undef MEGA_MOE_BF16_ROUTING_HOST_DEVICE
