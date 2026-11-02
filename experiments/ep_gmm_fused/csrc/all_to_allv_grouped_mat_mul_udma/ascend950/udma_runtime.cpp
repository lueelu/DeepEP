/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#include "udma_runtime.hpp"

#include "shmem.h"

#include <ATen/ATen.h>

namespace deepep::ep_gmm_fused {
UdmaRuntime& UdmaRuntime::Instance()
{
    static UdmaRuntime runtime;
    return runtime;
}

void UdmaRuntime::Attach(int32_t rank, int32_t worldSize, uintptr_t workspace, uint64_t workspaceBytes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(rank >= 0 && rank < worldSize, "invalid UDMA rank ", rank, " for world size ", worldSize);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8, "UDMA world size must be in [1, 8]");
    TORCH_CHECK(!initialized_, "UDMA workspace is already attached");
    TORCH_CHECK(workspace != 0 && workspaceBytes >= kKernelWorkspaceBytes, "invalid public runtime workspace");

    int32_t device = -1;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "cannot query active NPU");
    workspace_ = reinterpret_cast<uint8_t*>(workspace);
    fftsAddress_ = shmemx_get_ffts_config();

    rank_ = rank;
    worldSize_ = worldSize;
    device_ = device;
    initialized_ = true;
}

void UdmaRuntime::EnsureInitialized() const
{
    TORCH_CHECK(initialized_, "UDMA runtime is not initialized; call init_udma_group() before the operator");
}

void UdmaRuntime::CheckDevice(int32_t device) const
{
    EnsureInitialized();
    TORCH_CHECK(device == device_, "tensor device differs from the initialized UDMA device");
}

void UdmaRuntime::RecordStream(aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    if (lastStream_ != nullptr && lastStream_ != stream) {
        const aclError syncStatus = aclrtSynchronizeStream(lastStream_);
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize the previous UDMA stream: ", syncStatus);
    }
    lastStream_ = stream;
}

void UdmaRuntime::Detach(aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    aclrtStream streamToWait = lastStream_ != nullptr ? lastStream_ : stream;
    if (streamToWait != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(streamToWait);
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize UDMA stream during finalization: ", syncStatus);
    }
    initialized_ = false;
    rank_ = -1;
    worldSize_ = 0;
    workspace_ = nullptr;
    lastStream_ = nullptr;
    fftsAddress_ = 0;
    device_ = -1;
}

uint8_t* UdmaRuntime::Workspace() const
{
    EnsureInitialized();
    return workspace_;
}

uint64_t UdmaRuntime::FftsAddress() const
{
    EnsureInitialized();
    return fftsAddress_;
}

int32_t UdmaRuntime::Rank() const
{
    EnsureInitialized();
    return rank_;
}

int32_t UdmaRuntime::WorldSize() const
{
    EnsureInitialized();
    return worldSize_;
}

}  // namespace deepep::ep_gmm_fused
