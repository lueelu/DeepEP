/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "large_ep_pipeline_kernel.hpp"
#include "catccos/dgemm/device/kernel_adapter.hpp"
namespace deepep::ep_gmm_fused {
template <class Kernel>
static void Launch(const typename Kernel::Params& params, uint32_t blockDim, void* stream, uint64_t fftsAddress)
{
    if (fftsAddress == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, fftsAddress, nullptr);
    }
}
void LaunchLargeEpN128(const LargeEpN128Params& params, bool wholeRound, uint32_t blockDim, void* stream,
                       uint64_t fftsAddress)
{
    if (wholeRound) {
        Launch<LargeEpPipeline::Kernel<true>>(params, blockDim, stream, fftsAddress);
    } else {
        Launch<LargeEpPipeline::Kernel<false>>(params, blockDim, stream, fftsAddress);
    }
}
template <class Config>
void LaunchLargeEp(const LargeEpParams<Config>& params, uint32_t blockDim, void* stream, uint64_t fftsAddress)
{
    Launch<LargeEpPipeline::Kernel<false, Config, TilingValidation::kMaxRanks>>(params, blockDim, stream, fftsAddress);
}

template void LaunchLargeEp(const LargeEpParams<GroupedMatMulAllToAllVUdmaDefaultConfig>&, uint32_t, void*, uint64_t);
template void LaunchLargeEp(const LargeEpN128Params&, uint32_t, void*, uint64_t);
template void LaunchLargeEp(const LargeEpParams<GroupedMatMulAllToAllVUdmaM0_256N0_256Config>&, uint32_t, void*,
                            uint64_t);
}  // namespace deepep::ep_gmm_fused
