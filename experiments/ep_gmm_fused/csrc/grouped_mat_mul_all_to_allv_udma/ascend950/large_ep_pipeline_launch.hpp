/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef DEEPEP_LARGE_EP_PIPELINE_LAUNCH_HPP
#define DEEPEP_LARGE_EP_PIPELINE_LAUNCH_HPP
#include "grouped_mat_mul_all_to_allv_udma_device.hpp"
namespace deepep::ep_gmm_fused {
using LargeEpN128Config = GroupedMatMulAllToAllVUdmaM0_256Config;
struct LargeEpN128Params {
    LargeEpN128Config::Device::Params base;
    uint32_t chunks;
};
void LaunchLargeEpN128(const LargeEpN128Params& params, bool wholeRound, uint32_t blockDim, void* stream,
                       uint64_t fftsAddress);
}  // namespace deepep::ep_gmm_fused
#endif
