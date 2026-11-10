// Copyright (c) 2026, Lu Lu
#include "session.hpp"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include "tiling/platform/platform_ascendc.h"
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "kernels/ll_core_config.h"
#include "kernels/ll_rdma_layout.h"

namespace deepep::ubx {
namespace {
std::mutex mutex;
bool occupied = false;
void check(int status, const char* operation)
{
    if (status) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
}  // namespace
bool Session::busy()
{
    std::lock_guard<std::mutex> guard(mutex);
    return occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED;
}
Session::Session(int r, int w, const std::string& endpoint, unsigned timeout) : rank(r), world(w)
{
    const uint32_t host = UbxDispatch::kLlPhysicalRanksPerHost;
    if (world < int(host) || world > 256 || (world & (world - 1)) || world % host || rank < 0 || rank >= world ||
        UbxDispatch::LlGroupForWorld(world) > host || timeout < 1 || timeout > 3600 || endpoint.empty() ||
        endpoint.size() >= ACLSHMEM_MAX_IP_PORT_LEN || endpoint.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid UBX topology, timeout or endpoint");
    std::lock_guard<std::mutex> guard(mutex);
    if (occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED)
        throw std::runtime_error("A SHMEM session already exists; destroy it before selecting UBX");
    check(aclrtGetCurrentContext(&context_), "current NPU context");
    const char* soc = aclrtGetSocName();
    if (!context_ || !soc || std::string(soc).rfind("Ascend950", 0) != 0)
        throw std::runtime_error("UBX requires an initialized Ascend950 context");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t device = -1;
    int64_t available = 0;
    check(aclrtGetDevice(&device), "current NPU");
    check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &available), "available AIV count");
    if (!platform || available <= 0) throw std::runtime_error("Cannot query UBX AIV count");
    cores = std::min(platform->GetCoreNumAiv(), uint32_t(available));
    if (!UbxDispatch::LlCoreCountValid(cores) || (world > int(host) && cores > ACLSHMEM_MAX_QP_NUM))
        throw std::runtime_error("UBX AIV count exceeds the per-core RDMA QP capacity");
    constexpr uint64_t alignment = 2ULL << 20;
    const uint64_t stage_bytes =
        world > int(host) ? (uint64_t(cores) * UbxRdma::kCoreBytes + alignment - 1) / alignment * alignment : 0;
    const uint64_t workspace_bytes = capacity + alignment;
    const uint64_t allocation = workspace_bytes + stage_bytes;
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t uid{};
    attr.my_pe = rank;
    attr.n_pes = world;
    attr.local_mem_size = allocation;
    std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    attr.comm_args = &uid;
    attr.option_attr.data_op_engine_type =
        world > int(host) ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE)
                          : ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = timeout;
    attr.option_attr.shm_create_timeout = timeout;
    attr.option_attr.control_operation_timeout = timeout;
    // Keep a failed initialization reserved; another backend must not reuse partial SDK state.
    occupied = true;
    if (world > int(host)) check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, cores), "ROCE QP setup");
    check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "UBX SHMEM initialization");
    workspace_ = aclshmem_align(512U, allocation);
    if (!workspace_) throw std::runtime_error("UBX symmetric allocation failed");
    if (reinterpret_cast<uintptr_t>(workspace_) % 512U)
        throw std::runtime_error("UBX RDMA workspace requires 512-byte alignment");
    check(aclrtMemset(workspace_, allocation, 0, allocation), "UBX workspace initialization");
    stage_ = static_cast<uint8_t*>(workspace_) + workspace_bytes;
    closed_ = false;
}
void Session::check_context() const
{
    aclrtContext current = nullptr;
    check(aclrtGetCurrentContext(&current), "current NPU context");
    if (closed_ || current != context_) throw std::runtime_error("Use a live UBX session on its creating NPU");
}
UbxDispatch::CommArgs Session::comm_args() const
{
    check_context();
    UbxDispatch::CommArgs args{};
    args.rank = rank;
    args.rankSize = world;
    args.localRankSize = UbxDispatch::LlGroupForWorld(world);
    args.localRank = rank % args.localRankSize;
    args.reserved0 = UbxDispatch::kLlPhysicalRanksPerHost;
    args.peerMemBytes = capacity;
    args.fftsVal = shmemx_get_ffts_config();
    args.reservedAddresses[0] = static_cast<uint8_t*>(workspace_);
    args.reservedAddresses[1] = static_cast<uint8_t*>(stage_);
    args.reservedConfig[0] = UbxRdma::kBackend;
    args.reservedConfig[1] = cores;
    for (int peer = 0; peer < world; ++peer) {
        if (!UbxDispatch::LlSameHost(rank, peer)) continue;
        auto* base = static_cast<uint8_t*>(peer == rank ? workspace_ : aclshmem_ptr(workspace_, peer));
        if (!base || reinterpret_cast<uintptr_t>(base) % 32) throw std::runtime_error("Missing UBX local MTE mapping");
        args.peerMems[peer] = base + uint64_t(peer) * 1024U;
    }
    return args;
}
void Session::close()
{
    std::lock_guard<std::mutex> guard(mutex);
    if (closed_) return;
    check_context();
    check(aclrtSynchronizeDevice(), "UBX completion before destruction");
    aclshmem_free(workspace_);
    check(aclshmem_finalize(), "UBX SHMEM finalize");
    workspace_ = nullptr;
    closed_ = true;
    occupied = false;
}
void Session::reset()
{
    check_context();
    check(aclrtSynchronizeDevice(), "UBX reset completion");
    check(aclrtMemset(workspace_, capacity + (2ULL << 20), 0, capacity + (2ULL << 20)), "UBX workspace reset");
}
}  // namespace deepep::ubx
