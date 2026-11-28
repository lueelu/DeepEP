/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026

#ifndef DEEPEP_EP_GMM_FUSED_UDMA_RUNTIME_HPP
#define DEEPEP_EP_GMM_FUSED_UDMA_RUNTIME_HPP

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace deepep::ep_gmm_fused {

class UdmaRuntime {
public:
    static constexpr size_t kKernelWorkspaceBytes = 200UL * 1024 * 1024 * sizeof(uint16_t) + 4096;

    static UdmaRuntime& Instance();

    // Borrow the public Session's workspace; this adapter never owns SHMEM.
    void Attach(int32_t rank, int32_t worldSize, uintptr_t workspace, uint64_t workspaceBytes, uint32_t ranksPerNode,
                int32_t pipelineChunks);
    void Detach(aclrtStream stream);
    void EnsureInitialized() const;
    void CheckDevice(int32_t device) const;

    void RecordStream(aclrtStream stream);
    uint8_t* Workspace() const;
    uint64_t FftsAddress() const;
    int32_t Rank() const;
    int32_t WorldSize() const;
    uint32_t RanksPerNode() const { return ranksPerNode_; }
    uint64_t SymmetricDataBytes() const { return symmetricDataBytes_; }
    int32_t PipelineChunks() const { return pipelineChunks_; }

private:
    UdmaRuntime() = default;
    ~UdmaRuntime() = default;
    UdmaRuntime(const UdmaRuntime&) = delete;
    UdmaRuntime& operator=(const UdmaRuntime&) = delete;

    std::mutex mutex_;
    int32_t device_{-1};
    bool initialized_{false};
    int32_t rank_{-1};
    int32_t worldSize_{0};
    uint32_t ranksPerNode_{0};
    uint64_t symmetricDataBytes_{0};
    int32_t pipelineChunks_{-1};
    uint8_t* workspace_{nullptr};
    aclrtStream lastStream_{nullptr};
    uint64_t fftsAddress_{0};
};

}  // namespace deepep::ep_gmm_fused

#endif  // DEEPEP_EP_GMM_FUSED_UDMA_RUNTIME_HPP
