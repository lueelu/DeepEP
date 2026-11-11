#include "shmem_instance.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "shmem.h"
#include "common/udma_schedule.h"
#include "common/check.h"

namespace {

constexpr uint64_t kDeviceLargePageBytes = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t kDeviceHugePage1gBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t kUdmaQpCount = 2;
bool g_runtime_shared_jetty = false;

std::mutex g_runtime_mutex;
int g_runtime_refcount = 0;
bool g_runtime_initialized = false;
int g_runtime_rank = -1;
int g_runtime_world_size = -1;
uint64_t g_runtime_local_mem_size = 0;
uint64_t g_runtime_workspace_size = 0;
std::string g_runtime_ip_port;
std::vector<int64_t> g_runtime_unique_id;
std::string g_runtime_bootstrap_key;
int64_t g_runtime_data_op_engine_type = 0;
bool g_runtime_use_1g_huge_pages = false;
aclshmemx_uniqueid_t g_default_flag_uid{};

void FillInitAttributes(int rank, int world_size, uint64_t local_mem_size, const std::string& ip_port,
                        int64_t data_op_engine_type, aclshmemx_init_attr_t* attributes)
{
    if (attributes == nullptr) {
        throw std::invalid_argument("attributes must not be null");
    }
    std::memset(attributes, 0, sizeof(*attributes));

    const std::size_t ip_len = std::min(ip_port.size(), static_cast<std::size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
    std::copy_n(ip_port.data(), ip_len, attributes->ip_port);
    attributes->ip_port[ip_len] = '\0';

    const int attr_version = (1 << 16) + static_cast<int>(sizeof(aclshmemx_init_attr_t));

    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, static_cast<data_op_engine_type_t>(data_op_engine_type), DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void*>(&g_default_flag_uid);
}

}  // namespace

std::vector<int64_t> GetShmemUniqueId()
{
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_runtime_initialized || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
        throw std::runtime_error("cannot create a new SHMEM UID while a runtime is active");
    }
    aclshmemx_uniqueid_t uid{};
    ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_get_uniqueid(&uid));
    const auto* bytes = reinterpret_cast<const unsigned char*>(&uid);
    return {bytes, bytes + sizeof(uid)};
}

std::tuple<int64_t, std::vector<int64_t>> GetShmemBootstrap(const std::string& key)
{
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (!g_runtime_initialized) {
        return {0, {}};
    }
    return {key == g_runtime_bootstrap_key ? 1 : -1, g_runtime_unique_id};
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
                             int64_t udma_qps_per_worker)
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
    const char* tuning = std::getenv("DEEPEP_CLOS_TUNING_SESSION");
    tuning_session_ = tuning && std::strcmp(tuning, "1") == 0;
    if (tuning_session_ && (!shared_jetty_ || token_schedule_ || ranks_per_group_ == 0))
        throw std::invalid_argument("tuning session requires shared-Jetty peer scheduling and a fixed group size");
}

std::vector<int64_t> ShmemInstance::ConfigureTuning(int64_t dispatch_order, int64_t combine_order, int64_t group_peers,
                                                    int64_t workers, int64_t qps)
{
    if (!initialized_ || !tuning_session_ || g_runtime_refcount != 1)
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

    try {
        InitRuntime();
        ffts_addr_ = shmemx_get_ffts_config();
        const int actual_rank = aclshmem_my_pe();
        const int actual_world_size = aclshmem_n_pes();
        if (actual_rank != rank_ || actual_world_size != world_size_) {
            throw std::runtime_error("CANN SHMEM PE information does not match the process group");
        }

        aclshmem_barrier_all();
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

    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (!g_runtime_initialized) {
        if (aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
            throw std::runtime_error("Another backend already owns the process SHMEM runtime");
        }
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
                if (missing) throw std::runtime_error(std::string("V1 shared Jetty requires HCOMM symbol ") + missing);
            }
            ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_qp_num(
                ACLSHMEM_DATA_OP_UDMA, shared_jetty_ ? ascend_deepep::udma::kQpCount : kUdmaQpCount));
            ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, shared_jetty_));
            if (shared_jetty_) {
                std::vector<uint64_t> masks(world_size_);
                for (int peer = 0; peer < world_size_; ++peer)
                    masks[peer] = tuning_session_
                                      ? ascend_deepep::udma::TuningPeerMask(rank_, peer, world_size_, ranks_per_group_)
                                  : token_schedule_
                                      ? ascend_deepep::udma::TokenPeerMask(rank_, peer)
                                      : ascend_deepep::udma::PeerMask(rank_, peer, world_size_, dispatch_peer_order_,
                                                                      combine_peer_order_, ranks_per_group_,
                                                                      peers_per_group_per_worker_,
                                                                      workers_per_peer_ * qps_per_worker_);
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_udma_peer_qp_masks(masks.data(), masks.size()));
            } else {
                ASCEND_DEEPEP_CHECK_SHMEM(aclshmemx_set_udma_peer_qp_masks(nullptr, 0));
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

        bool runtime_started = false;
        try {
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
        } catch (...) {
            const auto original_error = std::current_exception();
            if (runtime_started) {
                aclshmem_finalize();
            }
            try {
                ReleaseHugePageWorkspace();
            } catch (...) {
            }
            std::rethrow_exception(original_error);
        }

        g_runtime_shared_jetty = shared_jetty_;
        g_runtime_initialized = true;
        g_runtime_rank = rank_;
        g_runtime_world_size = world_size_;
        g_runtime_local_mem_size = local_mem_size_;
        g_runtime_workspace_size = workspace_size_;
        g_runtime_ip_port = ip_port_;
        g_runtime_unique_id = unique_id_;
        g_runtime_bootstrap_key = bootstrap_key_;
        g_runtime_data_op_engine_type = data_op_engine_type_;
        g_runtime_use_1g_huge_pages = use_1g_huge_pages_;
    } else if (g_runtime_shared_jetty || shared_jetty_) {
        throw std::runtime_error(
            "V1 shared Jetty supports one active Buffer per process; destroy it before rebuilding");
    } else if (g_runtime_use_1g_huge_pages || use_1g_huge_pages_) {
        throw std::runtime_error("1 GiB huge-page mode supports only one active Buffer runtime per process");
    } else if (g_runtime_rank != rank_ || g_runtime_world_size != world_size_ ||
               g_runtime_local_mem_size != local_mem_size_ || g_runtime_workspace_size != workspace_size_ ||
               g_runtime_ip_port != ip_port_ || g_runtime_unique_id != unique_id_ ||
               g_runtime_bootstrap_key != bootstrap_key_ || g_runtime_data_op_engine_type != data_op_engine_type_ ||
               g_runtime_shared_jetty != shared_jetty_ || g_runtime_use_1g_huge_pages != use_1g_huge_pages_) {
        throw std::runtime_error(
            "CANN SHMEM has already been initialized with incompatible "
            "rank/world/local_mem_size/workspace_size/ip_port/transport/huge-page mode");
    }
    ++g_runtime_refcount;
    acquired_global_runtime_ = true;
}

void ShmemInstance::Barrier()
{
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    aclshmem_barrier_all();
}

void ShmemInstance::WaitHandle(aclrtStream stream) const
{
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    aclshmem_handle_t handle{};
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
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
        aclshmem_barrier_all();
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
    initialized_ = false;
    ffts_addr_ = 0;

    if (acquired_global_runtime_) {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        --g_runtime_refcount;
        acquired_global_runtime_ = false;
        if (g_runtime_refcount == 0 && g_runtime_initialized) {
            aclshmem_finalize();
            g_runtime_initialized = false;
            g_runtime_rank = -1;
            g_runtime_world_size = -1;
            g_runtime_local_mem_size = 0;
            g_runtime_workspace_size = 0;
            g_runtime_ip_port.clear();
            g_runtime_unique_id.clear();
            g_runtime_bootstrap_key.clear();
            g_runtime_data_op_engine_type = 0;
            g_runtime_shared_jetty = false;
            g_runtime_use_1g_huge_pages = false;
            std::memset(&g_default_flag_uid, 0, sizeof(g_default_flag_uid));
        }
    }
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

std::vector<int32_t> ShmemInstance::WaitNotifyHostCounts(std::size_t count)
{
    TORCH_CHECK(notify_host_pending_ && count <= notify_count_capacity_, "Notify host counts were not prepared");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    const auto* slots = static_cast<const uint32_t*>(notify_host_counts_);
    std::vector<int32_t> counts(count);
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t value;
        while ((value = __atomic_load_n(slots + i, __ATOMIC_ACQUIRE)) == 0) {
            TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "timed out waiting for Notify host counts");
        }
        TORCH_CHECK(value - 1U <= uint32_t(std::numeric_limits<int32_t>::max()), "invalid Notify host count");
        counts[i] = static_cast<int32_t>(value - 1U);
    }
    notify_host_pending_ = false;
    return counts;
}
