/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef DEEPEP_LARGE_EP_PIPELINE_SYNC_HPP
#define DEEPEP_LARGE_EP_PIPELINE_SYNC_HPP
#include "large_ep_pipeline_launch.hpp"
namespace deepep::ep_gmm_fused::LargeEpPipeline {
// 400 MiB of data followed by the runtime's 4096-byte control reservation.
// Single writer per core line; stages occupy separate words in that line.
struct SyncState {
    static constexpr uint32_t kCoreCount = 32;
    static constexpr uint32_t kLineWords = 16;
    static constexpr uint32_t kControlBytes = (kCoreCount + 4) * 64;
    static_assert(kControlBytes <= 4096, "pipeline control reservation");
    __gm__ int32_t* control;
    CATLASS_DEVICE explicit SyncState(GM_ADDR syncMmadFinish)
        : control(reinterpret_cast<__gm__ int32_t*>(syncMmadFinish))
    {
    }
    CATLASS_DEVICE __gm__ int32_t* Core(uint32_t core, uint32_t stage) const
    {
        return control + core * kLineWords + stage;
    }
    CATLASS_DEVICE __gm__ int32_t* Ready(uint32_t stage) const
    {
        return control + (kCoreCount + stage) * kLineWords;
    }
    CATLASS_DEVICE __gm__ int32_t* Complete(uint32_t stage) const
    {
        return control + (kCoreCount + 2 + stage) * kLineWords;
    }
    // Mode4 4/5 target paired AIV0; stage releases use mode2 0/1.
    // CATLASS barriers8/9/10 and SyncAll<true> flag14 remain reserved.
    CATLASS_DEVICE static void PublishChunk(uint32_t stage)
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(4 + stage);
    }
    CATLASS_DEVICE static void WaitChunk(uint32_t stage)
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(4 + stage);
    }
    CATLASS_DEVICE static void WaitReleased(uint32_t stage)
    {
        Catlass::Arch::CrossCoreFlag flag(stage);
        Catlass::Arch::CrossCoreWaitFlag(flag);
    }
    CATLASS_DEVICE static void Release(uint32_t stage)
    {
        Catlass::Arch::CrossCoreFlag flag(stage);
        Catlass::Arch::CrossCoreSetFlag<2, PIPE_MTE3>(flag);
    }
};
}  // namespace deepep::ep_gmm_fused::LargeEpPipeline
#endif
