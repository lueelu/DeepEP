/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef DEEPEP_LARGE_EP_PIPELINE_LAUNCH_HPP
#define DEEPEP_LARGE_EP_PIPELINE_LAUNCH_HPP
#include "grouped_mat_mul_all_to_allv_udma_device.hpp"
namespace deepep::ep_gmm_fused {
using LargeEpN128Config = GroupedMatMulAllToAllVUdmaM0_256Config;
template <class Config>
struct LargeEpParams {
    typename Config::Device::Params base;
    uint32_t chunks;
    uint32_t ranksPerNode;
};
using LargeEpN128Params = LargeEpParams<LargeEpN128Config>;

template <class Config>
void LaunchLargeEp(const LargeEpParams<Config>& params, uint32_t blockDim, void* stream, uint64_t fftsAddress);
void LaunchLargeEpN128(const LargeEpN128Params& params, bool wholeRound, uint32_t blockDim, void* stream,
                       uint64_t fftsAddress);
}  // namespace deepep::ep_gmm_fused
#endif
