// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>
#define ASCEND_DEEPEP_EXPORT __attribute__((visibility("default")))
#define ASCEND_DEEPEP_KERNEL_LAUNCH_UB(bytes) (bytes)
extern "C" {
ASCEND_DEEPEP_EXPORT void notify_full_kernel_do(void* stream, uint8_t* input, uint8_t* workspace, uint8_t* meta,
                                                uint8_t* hist, uint8_t* bases, uint8_t* prefix, uint8_t* counts,
                                                uint8_t* gateways, uint8_t* local_dst, uint8_t* mask, uint8_t* dst,
                                                uint8_t* forward, uint8_t* forward_counts, uint8_t* backward,
                                                uint8_t* backward_counts, uint8_t* ranges, uint8_t* masks,
                                                uint8_t* rank_rows, uint8_t* gather_rows, uint8_t* weight_return_meta,
                                                uint8_t* tiling);
ASCEND_DEEPEP_EXPORT void notify_dispatch_barrier_kernel_do(void* stream, uint32_t num_cores);
ASCEND_DEEPEP_EXPORT void notify_dispatch_peer_plan_kernel_do(void* stream, uint8_t* workspace, uint8_t* dst,
                                                              uint8_t* forward, uint8_t* counts, uint8_t* plan,
                                                              uint8_t* tiling, uint64_t stride_offset);
}
