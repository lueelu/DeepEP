// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstdint>

#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE
#endif

namespace MegaMoeBF16Routing {

struct UnpermutePlan {
    static constexpr uint32_t kIndexElements = 1024;
    static constexpr uint32_t kIndexBytes = kIndexElements * sizeof(int32_t);
    static constexpr uint32_t kBf16Rows = 4;
    static constexpr uint32_t kFp32Rows = 2;
    bool enabled = false;
    uint32_t rowBytes = 0;
    uint32_t batchTokens = 0;

    MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE constexpr UnpermutePlan(uint32_t hidden, uint32_t topK, uint32_t ubBytes)
    {
        const uint64_t bytes = uint64_t{hidden} * sizeof(uint16_t);
        const uint64_t scratchBytes = kBf16Rows * bytes + uint64_t{kFp32Rows} * hidden * sizeof(float);
        if (hidden == 0 || bytes % 32 != 0 || topK == 0 || topK > kIndexElements ||
            scratchBytes + kIndexBytes > ubBytes) {
            return;
        }
        enabled = true;
        rowBytes = static_cast<uint32_t>(bytes);
        batchTokens = kIndexElements / topK;
    }

    MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE constexpr uint32_t BatchCount(uint32_t remaining) const
    {
        return remaining < batchTokens ? remaining : batchTokens;
    }
};

}  // namespace MegaMoeBF16Routing

#undef MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE
