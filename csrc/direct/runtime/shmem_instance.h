#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <torch/custom_class.h>

std::vector<int64_t> GetShmemUniqueId();
std::tuple<int64_t, std::vector<int64_t>> GetShmemBootstrap(const std::string& key);

class ShmemInstance : public torch::jit::CustomClassHolder {
public:
    ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port);
    ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port, int64_t data_op_engine_type,
                  bool use_1g_huge_pages = false, std::vector<int64_t> unique_id = {}, std::string bootstrap_key = {},
                  bool v1_shared_jetty = false, bool v1_token_schedule = false, int64_t udma_dispatch_peer_order = 0,
                  int64_t udma_combine_peer_order = 0, int64_t udma_ranks_per_group = 0,
                  int64_t udma_peers_per_group_per_worker = 0, int64_t udma_workers_per_peer = 1,
                  int64_t udma_qps_per_worker = 1);

    ~ShmemInstance();

    ShmemInstance(const ShmemInstance&) = delete;
    ShmemInstance& operator=(const ShmemInstance&) = delete;

    void Init();
    void Init(int64_t local_mem_size);
    void Init(int64_t local_mem_size, int64_t workspace_size);
    void Barrier();
    void WaitHandle(aclrtStream stream) const;
    void Synchronize(aclrtStream stream) const;
    void Destroy();

    uint64_t FftsAddr() const;
    bool Initialized() const;
    bool IsAvailable() const;
    int64_t Rank() const;
    int64_t WorldSize() const;
    void* WorkspacePtr() const;
    bool Use1gHugePages() const;
    bool SharedJetty() const
    {
        return shared_jetty_;
    }
    bool TokenSchedule() const
    {
        return token_schedule_;
    }
    uint32_t DispatchPeerOrder() const
    {
        return dispatch_peer_order_;
    }
    uint32_t CombinePeerOrder() const
    {
        return combine_peer_order_;
    }
    uint32_t RanksPerGroup() const
    {
        return ranks_per_group_;
    }
    uint32_t WorkersPerPeer() const
    {
        return workers_per_peer_;
    }
    uint32_t QpsPerWorker() const
    {
        return qps_per_worker_;
    }
    uint32_t PeersPerGroupPerWorker() const
    {
        return peers_per_group_per_worker_;
    }

private:
    void InitRuntime();
    void AllocateHugePageWorkspace();
    void ReleaseHugePageWorkspace();
    void DestroyImpl(bool synchronize);
    void DestroyNoThrow() noexcept;

    int rank_;
    int world_size_;
    uint64_t local_mem_size_;
    uint64_t workspace_size_;
    uint64_t workspace_allocation_size_;
    std::string ip_port_;
    std::vector<int64_t> unique_id_;
    std::string bootstrap_key_;
    int64_t data_op_engine_type_;
    bool use_1g_huge_pages_;
    bool shared_jetty_;
    bool token_schedule_;
    uint32_t dispatch_peer_order_ = 0;
    uint32_t combine_peer_order_ = 0;
    uint32_t ranks_per_group_ = 0;
    uint32_t peers_per_group_per_worker_ = 0;
    uint32_t workers_per_peer_ = 1;
    uint32_t qps_per_worker_ = 1;
    bool initialized_;
    bool acquired_global_runtime_;
    bool workspace_source_mapped_;

    aclrtDrvMemHandle workspace_handle_;
    void* workspace_source_ptr_;
    void* workspace_ptr_;

    uint64_t ffts_addr_;
};
