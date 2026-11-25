// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "comm/kernel/hierarchical_alltoall.hpp"
namespace deepep::hierarchical {
struct LaunchArgs {
    uint32_t m;
    uint32_t n;
    uint32_t rankIdx;
    uint32_t rankSize;
    uint32_t commSizeM;
    uint32_t qpNum;
    uintptr_t ptrSrc;
    uintptr_t ptrDst;
    uintptr_t ptrScaleSrc;
    uintptr_t ptrScaleDst;
    uintptr_t ptrReady;
};
void GetAllToAllTensorBytes(uint32_t m, uint32_t n, uint64_t& dataBytes, uint64_t& scaleBytes);
void LaunchAllToAll(const LaunchArgs& args, uint32_t blocks, void* stream, uint64_t ffts);
uint64_t BarrierBytes(uint32_t world);
void LaunchBarrier(uint32_t rank, uint32_t world, uint32_t epoch, void* workspace, void* stream, uint64_t ffts);

constexpr uint32_t kCoreCount = 32;
constexpr uint32_t kScaleGroup = 32;
constexpr uint8_t kPadding = 0xa5;

DEEPEP_COMM_HOST_DEVICE uint8_t Pattern(uint32_t source, uint32_t peer, uint64_t index, bool scale)
{
    const uint64_t key = index * 17 + (index / 251) * 13 + source * 29 + peer * 43;
    // Finite FP8 payload encodings; positive finite E8M0 scales, no NaN encoding.
    return scale ? uint8_t(120 + key % 15) : uint8_t(1 + key % 119);
}
} // namespace deepep::hierarchical
