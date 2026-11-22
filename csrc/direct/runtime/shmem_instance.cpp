#include "shmem_instance.h"
#include "runtime/instances.hpp"
#include "runtime/instance_endpoint.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <mutex>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "shmem.h"
#include "common/udma_schedule.h"
#include "common/check.h"
#include "common/types.h"

namespace {

constexpr uint64_t kDeviceLargePageBytes = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t kDeviceHugePage1gBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t kUdmaQpCount = 2;
constexpr uint64_t kMicrosecondsPerSecond = 1000000;
aclshmemx_uniqueid_t g_default_flag_uid{};
std::map<uint32_t, std::string> bootstrap_keys;

uint32_t ReadTimeoutSeconds(const char* name, uint32_t default_seconds,
                            uint32_t max_seconds = std::numeric_limits<uint32_t>::max())
{
    const char* value = std::getenv(name);
    if (value == nullptr) return default_seconds;

    uint32_t seconds = 0;
    const char* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, seconds);
    TORCH_CHECK(result.ec == std::errc{} && result.ptr == end && seconds > 0 && seconds <= max_seconds, name,
                " must be a positive integer number of seconds in [1, ", max_seconds, "]");
    return seconds;
}

void FillInitAttributes(int rank, int world_size, uint64_t local_mem_size, const std::string& ip_port,
                        int64_t data_op_engine_type, aclshmemx_init_attr_t* attributes)
{
    if (attributes == nullptr) {
        throw std::invalid_argument("attributes must not be null");
    }
    const std::size_t ip_len = std::min(ip_port.size(), static_cast<std::size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
    std::copy_n(ip_port.data(), ip_len, attributes->ip_port);
    attributes->ip_port[ip_len] = '\0';

    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->local_mem_size = local_mem_size;
    // Preserve the SDK's layout version so it reads runtime_config.
    attributes->option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(data_op_engine_type);
    attributes->comm_args = reinterpret_cast<void*>(&g_default_flag_uid);
}

}  // namespace

std::vector<int64_t> GetShmemUniqueId()
{
    deepep::InstanceLock lock(deepep::instance_mutex());
    aclshmemx_uniqueid_t uid{};
    ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_get_uniqueid(&uid));
    const auto* bytes = reinterpret_cast<const unsigned char*>(&uid);
    return {bytes, bytes + sizeof(uid)};
}

std::tuple<int64_t, std::vector<int64_t>> GetShmemBootstrap(const std::string& key)
{
    deepep::InstanceLock lock(deepep::instance_mutex());
    if (deepep_has_instances()) {
        if (bootstrap_keys.empty() || bootstrap_keys.begin()->second != key) return {-1, {}};
    } else if (aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
        return {-1, {}};
    }
    try {
        return {static_cast<int64_t>(deepep_next_instance()), {}};
    } catch (const std::runtime_error&) {
        return {-1, {}};
    }
}

ShmemInstance::ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port)
    : ShmemInstance(rank, world_size, std::move(ip_port), ACLSHMEM_DATA_OP_MTE, false)
{
}

ShmemInstance::ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port, int64_t data_op_engine_type,
                             bool use_1g_huge_pages, std::vector<int64_t> unique_id, std::string bootstrap_key,
                             bool v1_shared_jetty, bool v1_token_schedule, int64_t udma_dispatch_peer_order,
                             int64_t udma_combine_peer_order, int64_t udma_ranks_per_group,
                             int64_t udma_peers_per_group_per_worker, int64_t udma_workers_per_peer,
                             int64_t udma_qps_per_worker, bool auto_resources)
    : rank_(static_cast<int>(rank)),
      world_size_(static_cast<int>(world_size)),
      local_mem_size_(0),
      workspace_size_(0),
      workspace_allocation_size_(0),
      ip_port_(std::move(ip_port)),
      unique_id_(std::move(unique_id)),
      bootstrap_key_(std::move(bootstrap_key)),
      data_op_engine_type_(data_op_engine_type),
      use_1g_huge_pages_(use_1g_huge_pages),
      shared_jetty_(v1_shared_jetty && (data_op_engine_type & ACLSHMEM_DATA_OP_UDMA) != 0),
      token_schedule_(v1_token_schedule),
      initialized_(false),
      acquired_global_runtime_(false),
      workspace_source_mapped_(false),
      workspace_handle_(nullptr),
      workspace_source_ptr_(nullptr),
      workspace_ptr_(nullptr),
      ffts_addr_(0)
{
    if (udma_dispatch_peer_order < 0 || udma_dispatch_peer_order > 3 || udma_combine_peer_order < 0 ||
        udma_combine_peer_order > 3) {
        throw std::invalid_argument("invalid UDMA peer order");
    }
    const bool topology_order = udma_dispatch_peer_order != 0 || udma_combine_peer_order != 0;
    if (udma_ranks_per_group < 0 || udma_ranks_per_group > world_size ||
        (udma_ranks_per_group != 0 && world_size % udma_ranks_per_group != 0) ||
        (topology_order && udma_ranks_per_group == 0)) {
        throw std::invalid_argument("UDMA ranks_per_group must divide the EP group size");
    }
    if ((topology_order || udma_ranks_per_group != 0) && (!shared_jetty_ || token_schedule_)) {
        throw std::invalid_argument("UDMA peer order/group size requires explicit shared-Jetty UDMA peer schedule");
    }
    dispatch_peer_order_ = static_cast<uint32_t>(udma_dispatch_peer_order);
    combine_peer_order_ = static_cast<uint32_t>(udma_combine_peer_order);
    ranks_per_group_ = static_cast<uint32_t>(udma_ranks_per_group);
    if (udma_workers_per_peer != 1 && udma_workers_per_peer != 2)
        throw std::invalid_argument("udma_workers_per_peer must be 1 or 2");
    if (udma_workers_per_peer != 1 && (!shared_jetty_ || token_schedule_))
        throw std::invalid_argument("multiple workers per peer require shared-Jetty UDMA peer schedule");
    workers_per_peer_ = static_cast<uint32_t>(udma_workers_per_peer);
    if (udma_qps_per_worker != 1 && udma_qps_per_worker != 2)
        throw std::invalid_argument("udma_qps_per_worker must be 1 or 2");
    if (udma_qps_per_worker != 1 && (workers_per_peer_ != 1 || !shared_jetty_ || token_schedule_))
        throw std::invalid_argument(
            "two QPs per worker require shared-Jetty UDMA peer schedule and one worker per peer");
    qps_per_worker_ = static_cast<uint32_t>(udma_qps_per_worker);
    const uint32_t qp_lanes = workers_per_peer_ * qps_per_worker_;
    if (qp_lanes != 1 &&
        (dispatch_peer_order_ != ascend_deepep::udma::kGroupRoundRobinOrder ||
         combine_peer_order_ != ascend_deepep::udma::kGroupRoundRobinOrder) &&
        ascend_deepep::udma::Workers(world_size, ascend_deepep::udma::kMaxPeers) * qp_lanes >
            ascend_deepep::udma::kQpCount)
        throw std::invalid_argument("no peer configuration fits the 32 sending QP limit");
    const bool round_robin = dispatch_peer_order_ == ascend_deepep::udma::kGroupRoundRobinOrder ||
                             combine_peer_order_ == ascend_deepep::udma::kGroupRoundRobinOrder;
    if (round_robin) {
        if (udma_peers_per_group_per_worker != 1 && udma_peers_per_group_per_worker != 2 &&
            udma_peers_per_group_per_worker != 4)
            throw std::invalid_argument("group_round_robin requires peers per group per worker in 1,2,4");
        const uint64_t lanes =
            (udma_ranks_per_group + udma_peers_per_group_per_worker - 1) / udma_peers_per_group_per_worker;
        const uint64_t count = (udma_ranks_per_group + lanes - 1) / lanes * (world_size / udma_ranks_per_group);
        if (lanes * qp_lanes > ascend_deepep::udma::kQpCount || count > ascend_deepep::udma::kGroupPeerCapacity)
            throw std::invalid_argument("group_round_robin exceeds 32 workers or 64 peers per worker");
    } else if (udma_peers_per_group_per_worker != 0) {
        throw std::invalid_argument("peers per group per worker requires group_round_robin");
    }
    peers_per_group_per_worker_ = static_cast<uint32_t>(udma_peers_per_group_per_worker);
    if (token_schedule_ && !shared_jetty_) {
        throw std::invalid_argument("token schedule requires V1 shared-Jetty UDMA");
    }
    if (unique_id_.empty() == ip_port_.empty()) {
        throw std::invalid_argument("provide exactly one SHMEM endpoint or unique ID");
    }
    if (!unique_id_.empty() &&
        (unique_id_.size() != sizeof(aclshmemx_uniqueid_t) ||
         std::any_of(unique_id_.begin(), unique_id_.end(), [](int64_t b) { return b < 0 || b > 255; }))) {
        throw std::invalid_argument("invalid SHMEM unique ID bytes");
    }
    if (rank < 0 || rank > std::numeric_limits<int>::max() || world_size <= 0 ||
        world_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("rank/world_size is out of range");
    }
    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument("invalid rank/world_size");
    }
    constexpr int64_t kValidEngineMask = (static_cast<int64_t>(ACLSHMEM_DATA_OP_MAX) << 1) - 1;
    if (data_op_engine_type_ <= 0 || (data_op_engine_type_ & ~kValidEngineMask) != 0) {
        throw std::invalid_argument("invalid CANN SHMEM data operation engine type");
    }
    auto_resources_ = auto_resources;
    if (auto_resources_ && (!shared_jetty_ || token_schedule_ || ranks_per_group_ == 0))
        throw std::invalid_argument("automatic launch plans require shared-Jetty peer resources and a fixed group");
    const char* tuning = std::getenv("DEEPEP_CLOS_TUNING_SESSION");
    tuning_session_ = tuning && std::strcmp(tuning, "1") == 0;
    if (tuning_session_ && (!shared_jetty_ || token_schedule_ || ranks_per_group_ == 0))
        throw std::invalid_argument("tuning session requires shared-Jetty peer scheduling and a fixed group size");
}

std::vector<int64_t> ShmemInstance::ConfigureTuning(int64_t dispatch_order, int64_t combine_order, int64_t group_peers,
                                                    int64_t workers, int64_t qps)
{
    deepep::InstanceLock lock(deepep::instance_mutex());
    if (!initialized_ || !tuning_session_ || auto_resources_ || !deepep_is_only_instance(instance_id_))
        throw std::runtime_error("configure_tuning requires one active tuning-session Buffer per process");
    // Validate before any collective or mutation. The benchmark agrees the
    // complete plan across ranks before entering this private interface.
    ShmemInstance candidate(rank_, world_size_, ip_port_, data_op_engine_type_, use_1g_huge_pages_, unique_id_,
                            bootstrap_key_, true, false, dispatch_order, combine_order, ranks_per_group_, group_peers,
                            workers, qps);
    ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeDevice());
    Barrier();
    dispatch_peer_order_ = candidate.dispatch_peer_order_;
    combine_peer_order_ = candidate.combine_peer_order_;
    peers_per_group_per_worker_ = candidate.peers_per_group_per_worker_;
    workers_per_peer_ = candidate.workers_per_peer_;
    qps_per_worker_ = candidate.qps_per_worker_;
    return {dispatch_peer_order_, combine_peer_order_, peers_per_group_per_worker_, workers_per_peer_, qps_per_worker_};
}

ShmemInstance::LaunchPlan ShmemInstance::ResolveLaunch(const std::vector<int64_t>& values, bool combine, int64_t peers,
                                                       int64_t visit, int64_t cores) const
{
    using namespace ascend_deepep::udma;
    if (values.empty()) {
        const auto order = combine ? combine_peer_order_ : dispatch_peer_order_;
        return {order,
                ranks_per_group_,
                order == kGroupRoundRobinOrder ? peers_per_group_per_worker_ : uint32_t(peers),
                workers_per_peer_,
                qps_per_worker_,
                uint32_t(visit),
                -1,
                0};
    }
    // Validate before narrowing or any kernel submission. P/W/Q changes are
    // covered by the symmetric union reserved once in InitRuntime.
    if (!auto_resources_ || values.size() != 9 || values[0] != 1)
        throw std::invalid_argument("invalid or unreserved UDMA launch plan");
    const auto order = values[1], group = values[2], p = values[3], w = values[4], q = values[5];
    const auto v = values[6], copy = values[7], weight = values[8];
    // Dispatch plans also carry the normalized after-submit marker (2).
    // Spare-AIV copy (1) and weight transport overrides remain combine-only.
    if (order < 0 || order > 3 || group != ranks_per_group_ || p < 1 || p > 8 || !ValidPeers(p) || (w != 1 && w != 2) ||
        (q != 1 && q != 2) || w * q > 2 || v < 4 || v > 32 || !ValidVisit(v) || copy < 0 || copy > 2 || weight < 0 ||
        weight > 2 || (!combine && ((copy != 0 && copy != 2) || weight)))
        throw std::invalid_argument("UDMA launch plan is outside the reserved schedule set");
    const auto owners = Workers(world_size_, p, order, group);
    const auto aivs = cores == 0 ? 32 : cores;
    if (cores < 0 || cores % 2 || owners * w > aivs || owners * w * q > kQpCount)
        throw std::invalid_argument("UDMA launch plan exceeds communication AIV/QP budget");
    if (order == kGroupRoundRobinOrder &&
        (p > 4 || ((group + owners - 1) / owners) * (world_size_ / group) > kGroupPeerCapacity))
        throw std::invalid_argument("UDMA launch plan exceeds the round-robin peer table");
    return {uint32_t(order), uint32_t(group), uint32_t(p), uint32_t(w),
            uint32_t(q),     uint32_t(v),     copy,        uint32_t(weight)};
}

ShmemInstance::~ShmemInstance()
{
    DestroyNoThrow();
}

void ShmemInstance::Init()
{
    if (initialized_) {
        return;
    }
    if (local_mem_size_ == 0) {
        throw std::runtime_error("ShmemInstance local_mem_size is not configured; call init(local_mem_size)");
    }

    if ((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0) {
        constexpr uint32_t kMaxSyncTimeoutSeconds = ACLSHMEM_SYNC_TIMEOUT_MAX_US / kMicrosecondsPerSecond;
        const uint32_t seconds = ReadTimeoutSeconds("DEEPEP_NPU_TIMEOUT_SECONDS", 0, kMaxSyncTimeoutSeconds);
        // Zero is the SDK's inherit/default sentinel and is never a device-polling deadline.
        shmem_sync_timeout_us_ = static_cast<uint32_t>(seconds * kMicrosecondsPerSecond);
        npu_timeout_cycles_ = (seconds == 0 ? ascend_deepep::kDefaultNpuTimeoutSeconds : seconds) *
                              ascend_deepep::kNpuSystemCyclesPerSecond;
        notify_host_timeout_seconds_ = ReadTimeoutSeconds("DEEPEP_CPU_TIMEOUT_SECONDS", 120);
    }

    deepep::InstanceLock lock(deepep::instance_mutex());
    try {
        InitRuntime();
        deepep::select_instance(instance_id_);
        ffts_addr_ = shmemx_get_ffts_config();
        const int actual_rank = aclshmem_my_pe();
        const int actual_world_size = aclshmem_n_pes();
        if (actual_rank != rank_ || actual_world_size != world_size_) {
            throw std::runtime_error("CANN SHMEM PE information does not match the process group");
        }

        deepep_instance_barrier(instance_id_, ffts_addr_);
        initialized_ = true;
    } catch (...) {
        DestroyNoThrow();
        throw;
    }
}

void ShmemInstance::Init(int64_t local_mem_size)
{
    if (local_mem_size <= 0 || static_cast<uint64_t>(local_mem_size) > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("local_mem_size must be greater than zero");
    }
    const auto requested_local_mem_size = static_cast<uint64_t>(local_mem_size);
    if ((requested_local_mem_size % kDeviceLargePageBytes) != 0) {
        throw std::invalid_argument("local_mem_size must be aligned to DEVICE_LARGE_PAGE_SIZE (2MB)");
    }
    if (local_mem_size_ != 0 && local_mem_size_ != requested_local_mem_size) {
        throw std::runtime_error("ShmemInstance local_mem_size has already been configured differently");
    }
    local_mem_size_ = requested_local_mem_size;
    Init();
}

void ShmemInstance::Init(int64_t local_mem_size, int64_t workspace_size)
{
    if (!use_1g_huge_pages_) {
        throw std::invalid_argument("workspace-backed initialization requires use_1g_huge_pages=True");
    }
    if (workspace_size <= 0) {
        throw std::invalid_argument("workspace_size must be greater than zero when using 1 GiB huge pages");
    }
    const auto requested_workspace_size = static_cast<uint64_t>(workspace_size);
    if (workspace_size_ != 0 && workspace_size_ != requested_workspace_size) {
        throw std::runtime_error("ShmemInstance workspace_size has already been configured differently");
    }
    workspace_size_ = requested_workspace_size;
    Init(local_mem_size);
}

void ShmemInstance::AllocateHugePageWorkspace()
{
    if (!use_1g_huge_pages_ || workspace_size_ == 0) {
        throw std::runtime_error("1 GiB huge-page workspace is not configured");
    }

    int32_t device_id = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&device_id));
    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = ACL_HBM_MEM_HUGE1G;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device_id;
    property.reserve = 0;

    size_t granularity = 0;
    ASCEND_DEEPEP_CHECK_ACL(
        aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity));
    if (granularity != kDeviceHugePage1gBytes) {
        throw std::runtime_error("ACL_HBM_MEM_HUGE1G returned an unexpected allocation granularity: " +
                                 std::to_string(granularity));
    }
    if (workspace_size_ > std::numeric_limits<uint64_t>::max() - (granularity - 1)) {
        throw std::overflow_error("1 GiB huge-page workspace size overflows during alignment");
    }
    workspace_allocation_size_ = ((workspace_size_ + granularity - 1) / granularity) * granularity;

    try {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMallocPhysical(&workspace_handle_, static_cast<size_t>(workspace_allocation_size_), &property, 0));
        ASCEND_DEEPEP_CHECK_ACL(aclrtReserveMemAddress(&workspace_source_ptr_,
                                                       static_cast<size_t>(workspace_allocation_size_), 0, nullptr, 0));
        ASCEND_DEEPEP_CHECK_ACL(aclrtMapMem(workspace_source_ptr_, static_cast<size_t>(workspace_allocation_size_), 0,
                                            workspace_handle_, 0));
        workspace_source_mapped_ = true;

        aclrtMemAccessDesc access{};
        access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemSetAccess(workspace_source_ptr_, static_cast<size_t>(workspace_allocation_size_), &access, 1));
    } catch (...) {
        const auto original_error = std::current_exception();
        try {
            ReleaseHugePageWorkspace();
        } catch (...) {
        }
        std::rethrow_exception(original_error);
    }
}

void ShmemInstance::ReleaseHugePageWorkspace()
{
    std::ostringstream errors;
    if (workspace_source_mapped_) {
        const auto status = aclrtUnmapMem(workspace_source_ptr_);
        if (status != ACL_SUCCESS) {
            errors << "aclrtUnmapMem status=" << status << "; ";
        }
        workspace_source_mapped_ = false;
    }
    if (workspace_source_ptr_ != nullptr) {
        const auto status = aclrtReleaseMemAddress(workspace_source_ptr_);
        if (status != ACL_SUCCESS) {
            errors << "aclrtReleaseMemAddress status=" << status << "; ";
        }
        workspace_source_ptr_ = nullptr;
    }
    if (workspace_handle_ != nullptr) {
        const auto status = aclrtFreePhysical(workspace_handle_);
        if (status != ACL_SUCCESS) {
            errors << "aclrtFreePhysical status=" << status << "; ";
        }
        workspace_handle_ = nullptr;
    }
    workspace_ptr_ = nullptr;
    workspace_allocation_size_ = 0;
    if (!errors.str().empty()) {
        throw std::runtime_error("failed to release 1 GiB huge-page workspace: " + errors.str());
    }
}

void ShmemInstance::InitRuntime()
{
    if (acquired_global_runtime_) {
        return;
    }

    deepep::InstanceLock lock(deepep::instance_mutex());
    int32_t device = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&device));
    std::ostringstream config;
    config << "direct:" << device << ':' << rank_ << ':' << world_size_ << ':' << data_op_engine_type_ << ':'
           << shared_jetty_ << ':' << token_schedule_ << ':' << dispatch_peer_order_ << ':' << combine_peer_order_
           << ':' << ranks_per_group_ << ':' << peers_per_group_per_worker_ << ':' << workers_per_peer_ << ':'
           << qps_per_worker_ << ':' << tuning_session_ << ':' << auto_resources_ << ':' << bootstrap_key_;
    const bool first = !deepep_has_instances();
    instance_id_ = deepep_reserve_instance(config.str().c_str());
    bool runtime_started = false;
    try {
        if (first) {
            if ((data_op_engine_type_ & ACLSHMEM_DATA_OP_UDMA) != 0) {
                if (shared_jetty_) {
                    if (world_size_ > static_cast<int>(ascend_deepep::udma::kQpCount * 8 + 1))
                        throw std::invalid_argument("V1 shared Jetty supports at most 257 ranks");
                    // Refuse the SDK's independent-queue fallback on older HCOMM.
                    void* hcomm = dlopen("libhcomm.so", RTLD_NOW | RTLD_LOCAL);
                    if (!hcomm) throw std::runtime_error("V1 shared Jetty requires libhcomm.so");
                    const char* missing = nullptr;
                    for (const char* symbol : {"HcommChannelConfigCreate", "HcommChannelConfigDestroy",
                                               "HcommChannelConfigSetInt", "HcommChannelCreateWithConfig"}) {
                        if (!dlsym(hcomm, symbol)) missing = symbol;
                    }
                    dlclose(hcomm);
                    if (missing)
                        throw std::runtime_error(std::string("V1 shared Jetty requires HCOMM symbol ") + missing);
                }
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_qp_num(
                    ACLSHMEM_DATA_OP_UDMA, shared_jetty_ ? ascend_deepep::udma::kQpCount : kUdmaQpCount));
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, shared_jetty_));
                if (shared_jetty_) {
                    std::vector<uint64_t> masks(world_size_);
                    for (int peer = 0; peer < world_size_; ++peer)
                        masks[peer] =
                            (tuning_session_ || auto_resources_)
                                ? ascend_deepep::udma::TuningPeerMask(rank_, peer, world_size_, ranks_per_group_)
                            : token_schedule_ ? ascend_deepep::udma::TokenPeerMask(rank_, peer)
                                              : ascend_deepep::udma::PeerMask(
                                                    rank_, peer, world_size_, dispatch_peer_order_, combine_peer_order_,
                                                    ranks_per_group_, peers_per_group_per_worker_,
                                                    workers_per_peer_ * qps_per_worker_);
                    ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_udma_peer_qp_masks(masks.data(), masks.size()));
                } else {
                    ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_udma_peer_qp_masks(nullptr, 0));
                }
            }
        }
        aclshmemx_init_attr_t attributes{};
        FillInitAttributes(rank_, world_size_, local_mem_size_, ip_port_, data_op_engine_type_, &attributes);
        // Keep an aligned, rank-local copy alive throughout both initialization paths.
        aclshmemx_uniqueid_t uid{};
        auto init_flags = ACLSHMEMX_INIT_WITH_DEFAULT;
        if (!unique_id_.empty()) {
            auto* bytes = reinterpret_cast<unsigned char*>(&uid);
            std::copy(unique_id_.begin(), unique_id_.end(), bytes);
            ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_attr_uniqueid_args(
                rank_, world_size_, static_cast<int64_t>(local_mem_size_), &uid, &attributes));
            init_flags = ACLSHMEMX_INIT_WITH_UNIQUEID;
        }

        attributes.runtime_config.sync_timeout_us = shmem_sync_timeout_us_;
        attributes.instance_id = instance_id_;
        deepep::InstanceEndpoint endpoint(attributes);
        {
            if (use_1g_huge_pages_) {
                AllocateHugePageWorkspace();
                aclshmemx_buffer_optional_attr_t optional_attributes{};
                optional_attributes.mem_handle = workspace_handle_;
                aclshmemx_buffer_desc_t descriptor{};
                descriptor.addr = workspace_source_ptr_;
                descriptor.size = workspace_allocation_size_;
                descriptor.optional_attr = &optional_attributes;
                // One descriptor keeps every existing kernel offset unchanged: workspace byte 0 is
                // symmetric-heap byte 0, and padding exists only beyond the public workspace capacity.
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_init_attr_with_buffers(init_flags, &attributes, &descriptor, 1));
                runtime_started = true;
                workspace_ptr_ = aclshmemx_get_buffer_ptr(workspace_source_ptr_);
                if (workspace_ptr_ == nullptr) {
                    throw std::runtime_error("CANN SHMEM failed to translate the 1 GiB huge-page workspace");
                }
            } else {
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_init_attr(init_flags, &attributes));
                runtime_started = true;
            }
        }
        bootstrap_keys.emplace(instance_id_, bootstrap_key_);
        acquired_global_runtime_ = true;
    } catch (...) {
        const auto original_error = std::current_exception();
        // The selected SDK rolls back a failed init itself. Never finalize a
        // different live instance. A failed finalize retains its reservation.
        if (!runtime_started || aclshmemx_finalize(instance_id_) == 0) {
            deepep_release_instance(instance_id_);
        }
        try {
            ReleaseHugePageWorkspace();
        } catch (...) {
        }
        std::rethrow_exception(original_error);
    }
}

void ShmemInstance::Barrier()
{
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    deepep_instance_barrier(instance_id_, ffts_addr_);
}

void ShmemInstance::WaitHandle(aclrtStream stream) const
{
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    deepep_instance_sync_do(stream, instance_id_, ffts_addr_, true);
}

void ShmemInstance::Synchronize(aclrtStream stream) const
{
    WaitHandle(stream);
    ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream));
}

void ShmemInstance::Destroy()
{
    DestroyImpl(true);
}

void ShmemInstance::DestroyImpl(bool synchronize)
{
    if (!initialized_ && !acquired_global_runtime_ && workspace_source_ptr_ == nullptr &&
        workspace_handle_ == nullptr) {
        return;
    }

    if (initialized_ && synchronize) {
        deepep_instance_barrier(instance_id_, ffts_addr_);
    }

    if (notify_host_counts_ != nullptr) {
        // A failed polling call may leave a writer running. Never unmap its target.
        ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeDevice());
        ASCEND_DEEPEP_CHECK_ACL(aclrtHostUnregister(notify_host_counts_));
        ASCEND_DEEPEP_CHECK_ACL(aclrtFreeHost(notify_host_counts_));
        notify_host_counts_ = nullptr;
        notify_device_counts_ = nullptr;
        notify_count_capacity_ = 0;
        notify_host_pending_ = false;
    }

    deepep::InstanceLock lock(deepep::instance_mutex());
    if (acquired_global_runtime_) {
        ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_finalize(instance_id_));
        ReleaseHugePageWorkspace();
        bootstrap_keys.erase(instance_id_);
        deepep_release_instance(instance_id_);
        acquired_global_runtime_ = false;
    }
    initialized_ = false;
    ffts_addr_ = 0;
    ReleaseHugePageWorkspace();
}

void ShmemInstance::DestroyNoThrow() noexcept
{
    try {
        DestroyImpl(false);
    } catch (...) {
    }
}

uint64_t ShmemInstance::FftsAddr() const
{
    return ffts_addr_;
}

bool ShmemInstance::Initialized() const
{
    return initialized_;
}

bool ShmemInstance::IsAvailable() const
{
    return initialized_;
}

int64_t ShmemInstance::Rank() const
{
    return rank_;
}

int64_t ShmemInstance::WorldSize() const
{
    return world_size_;
}

void* ShmemInstance::WorkspacePtr() const
{
    if (!initialized_ || !use_1g_huge_pages_ || workspace_ptr_ == nullptr) {
        throw std::runtime_error("1 GiB huge-page workspace is not available");
    }
    return workspace_ptr_;
}

bool ShmemInstance::Use1gHugePages() const
{
    return use_1g_huge_pages_;
}

void* ShmemInstance::PrepareNotifyHostCounts(std::size_t count)
{
    TORCH_CHECK(initialized_, "Notify host counts require an initialized Buffer");
    TORCH_CHECK(!notify_host_pending_, "previous Notify host-count publication did not complete");
    if (notify_host_unavailable_) return nullptr;
    TORCH_CHECK(count > 0 && count <= std::numeric_limits<std::size_t>::max() / sizeof(uint32_t),
                "invalid Notify host-count capacity");
    if (count > notify_count_capacity_) {
        if (notify_host_counts_ != nullptr) {
            ASCEND_DEEPEP_CHECK_ACL(aclrtHostUnregister(notify_host_counts_));
            ASCEND_DEEPEP_CHECK_ACL(aclrtFreeHost(notify_host_counts_));
            notify_host_counts_ = nullptr;
            notify_device_counts_ = nullptr;
            notify_count_capacity_ = 0;
        }
        ASCEND_DEEPEP_CHECK_ACL(aclrtMallocHost(&notify_host_counts_, count * sizeof(uint32_t)));
        const auto status = aclrtHostRegister(notify_host_counts_, count * sizeof(uint32_t), ACL_HOST_REGISTER_MAPPED,
                                              &notify_device_counts_);
        if (status != ACL_SUCCESS) {
            ASCEND_DEEPEP_CHECK_ACL(aclrtFreeHost(notify_host_counts_));
            notify_host_counts_ = nullptr;
            notify_device_counts_ = nullptr;
            notify_host_unavailable_ = true;
            TORCH_WARN("Notify host mapping unavailable (ACL status ", status, "); using count readback");
            return nullptr;
        }
        notify_count_capacity_ = count;
    }
    // Zero is not ready; count+1 makes an empty receive distinguishable from it.
    // The preceding wait consumed every sole-writer store before this reuse.
    auto* slots = static_cast<uint32_t*>(notify_host_counts_);
    for (std::size_t i = 0; i < count; ++i) __atomic_store_n(slots + i, 0U, __ATOMIC_RELEASE);
    notify_host_pending_ = true;
    return notify_device_counts_;
}

std::vector<int32_t> ShmemInstance::WaitNotifyHostCounts(std::size_t count, int64_t* total_observed_ns)
{
    TORCH_CHECK(notify_host_pending_ && count <= notify_count_capacity_, "Notify host counts were not prepared");
    const auto start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(notify_host_timeout_seconds_);
    const auto* slots = static_cast<const uint32_t*>(notify_host_counts_);
    std::vector<int32_t> counts(count);
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t value;
        while ((value = __atomic_load_n(slots + i, __ATOMIC_ACQUIRE)) == 0) {
            const auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed >= timeout) {
                const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
                std::fprintf(stderr,
                             "[DEEPEP_CPU_TIMEOUT] op=notify_dispatch stage=host_counts rank=%d instance_id=%u "
                             "count_slot=%zu count_slots=%zu observed=%u elapsed_us=%lld timeout_s=%u\n",
                             rank_, instance_id_, i, count, value, static_cast<long long>(elapsed_us),
                             notify_host_timeout_seconds_);
                std::fflush(stderr);
                TORCH_CHECK(false, "timed out waiting for Notify host counts on rank ", rank_, " at slot ", i,
                            " after ", notify_host_timeout_seconds_, " seconds");
            }
        }
        TORCH_CHECK(value - 1U <= uint32_t(std::numeric_limits<int32_t>::max()), "invalid Notify host count");
        counts[i] = static_cast<int32_t>(value - 1U);
        if (i == 0 && total_observed_ns != nullptr) {
            *total_observed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
        }
    }
    notify_host_pending_ = false;
    return counts;
}
