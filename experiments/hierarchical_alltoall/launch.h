// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-2.0
#pragma once
#include "layout.h"
namespace deepep::hierarchical {
struct ChecksumArguments {
    uintptr_t checks;
    bool enabled;
};
using LaunchArgs = comm::kernel::AllToAllArguments<Layout, ChecksumArguments>;
void LaunchAllToAll(const LaunchArgs& args, uint32_t blocks, void* stream, uint64_t ffts);
uint64_t BarrierBytes(uint32_t world);
void LaunchBarrier(uint32_t rank, uint32_t world, uint32_t epoch, void* workspace, void* stream, uint64_t ffts);
}
