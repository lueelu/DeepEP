// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>
#define ASCEND_DEEPEP_EXPORT __attribute__((visibility("default")))
#define ASCEND_DEEPEP_KERNEL_LAUNCH_UB(bytes) (bytes)

namespace deepep {
void launch_notify_barrier(void* stream, uint32_t num_cores);
}
