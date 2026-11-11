#pragma once
#include "comm_args.h"
#include "ll_group_policy.h"
namespace LlRdma {
constexpr uint32_t kBackend = 1U;
constexpr uint32_t kCoreBytes = 128U * 1024U;
// ABI v3 tail: addresses[0]=unbiased symmetric workspace, [1]=per-core staging;
// config[0]=backend, [1]=QP count. Remote peerMems are NULL in RDMA mode.
}
