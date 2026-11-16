// Copyright (c) 2026, Lu Lu

#include "runtime/session.hpp"
#include <cstring>
#include <cstdlib>
#include <exception>
#include <dlfcn.h>
#include <vector>
#include "kernels/udma_layout.hpp"
#include "kernels/common.hpp"
#include "runtime/udma_peer_masks.hpp"
#include <mutex>
#include <stdexcept>
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_cc.h"
#include <limits>

namespace deepep {
namespace {
std::mutex runtime_mutex;
bool occupied = false;
void check(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed (status=" + std::to_string(status) + ")");
    }
}
}  // namespace

bool Session::busy()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    return occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED;
}

Session::Session(int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout, bool udma,
                 bool megamoe, bool mpi, bool combine_only, bool use_1g_huge_pages, bool megamoe_train)
{
    if (megamoe_train && (udma || megamoe || mpi || combine_only || use_1g_huge_pages))
        throw std::invalid_argument("Training requires a dedicated MTE Session");
    if (combine_only && (!udma || megamoe)) throw std::invalid_argument("Combine-only requires UDMA");
    if (use_1g_huge_pages && (!udma || megamoe))
        throw std::invalid_argument("1 GiB huge pages require the Elastic UDMA runtime");
    use_1g_huge_pages_ = use_1g_huge_pages;
    const char* dense_env = std::getenv("DEEPEP_SHMEM_DENSE_QPS");
    const std::string dense = dense_env == nullptr ? "0" : dense_env;
    if (dense != "0" && dense != "1") throw std::invalid_argument("DEEPEP_SHMEM_DENSE_QPS must be 0 or 1");
    combine_only_ = combine_only;
    constexpr uint64_t alignment = 2ULL * 1024 * 1024;
    if (size < 2 || size > (megamoe ? 128 : (megamoe_train ? 1024 : 512)) ||
        (!megamoe && !megamoe_train && (size & (size - 1))) || rank < 0 || rank >= size || bytes < alignment ||
        bytes > (megamoe ? (1ULL << 30) : ((1ULL << 63) - 1U)) || bytes % alignment != 0 || timeout < 1 ||
        timeout > 3600 || (megamoe && (udma || mpi)) ||
        (!mpi && (endpoint.empty() || endpoint.size() >= ACLSHMEM_MAX_IP_PORT_LEN ||
                  endpoint.find('\0') != std::string::npos))) {
        throw std::invalid_argument("Invalid rank, capacity, timeout or bootstrap endpoint");
    }
    // BF16 and FP8 use the same peer-table transport at every supported EP size.
    peer_table_ = udma;
    if (peer_table_ && (size < 2 || size > 512 || (size & (size - 1)))) {
        throw std::invalid_argument("peer-table requires EP2/4/8/16/32/64/128/256/512");
    }
    std::lock_guard<std::mutex> guard(runtime_mutex);
    if (occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
        throw std::runtime_error("A SHMEM runtime already exists in this process");
    }
    check(aclrtGetCurrentContext(&context_), "aclrtGetCurrentContext");
    if (context_ == nullptr) {
        throw std::runtime_error("Initialize the framework NPU context before creating a runtime");
    }
    const char* soc = aclrtGetSocName();
    if (soc == nullptr || std::string(soc).rfind("Ascend950", 0) != 0) {
        throw std::runtime_error("This runtime build is qualified only for Ascend950");
    }
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t uid{};
    attr.my_pe = rank;
    attr.n_pes = size;
    const uint64_t allocation_bytes = bytes + ((bytes == (8ULL << 30)) ? (2ULL << 20) : 0ULL);
    attr.local_mem_size = allocation_bytes;
    if (!mpi) {
        std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    }
    // A null communicator selects MPI_COMM_WORLD. The SHMEM MPI plugin owns
    // MPI_Init/Finalize when the application has not initialized MPI itself.
    attr.comm_args = mpi ? nullptr : &uid;
    attr.option_attr.data_op_engine_type =
        udma ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA) : ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = timeout;
    attr.option_attr.shm_create_timeout = timeout;
    attr.option_attr.control_operation_timeout = timeout;
    using SetPeerQpMasks = int (*)(const uint64_t*, uint32_t);
    auto set_masks =
        udma ? reinterpret_cast<SetPeerQpMasks>(dlsym(RTLD_DEFAULT, "aclshmemx_set_udma_peer_qp_masks")) : nullptr;
    if (udma && dense != "1" && !set_masks) {
        throw std::runtime_error(
            "SHMEM lacks aclshmemx_set_udma_peer_qp_masks; use the SDK pinned in dependencies.lock.json, "
            "or explicitly select dense channels with DEEPEP_SHMEM_DENSE_QPS=1");
    }
    // A failed initialization can leave SDK state behind. Keep the process
    // reserved until a new process is launched; do not finalize foreign state.
    occupied = true;
    bool init_attempted = false;
    bool runtime_started = false;
    try {
        if (udma) {
            // Combine owns pairs 0..15; peer-table Dispatch uses pairs 0..7.
            // Serialized operators reuse the bank, not concurrent SQ writers.
            // The pinned SDK shares Clos SQs across peers; MESH routes are separate.
            check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, ascend_deepep::udma_layout::kQpCount),
                  "aclshmemx_set_qp_num");
            check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true), "aclshmemx_set_shared_jetty");
            std::vector<uint64_t> masks;
            if (dense != "1") {
                masks.resize(size);
                for (int peer = 0; peer < size; ++peer)
                    masks[peer] = combine_only ? first_hit_schedule::CombineQpMask(rank, peer, size)
                                               : udma_admission::PeerMask(rank, peer, size);
            }
            if (set_masks) {
                check(set_masks(masks.empty() ? nullptr : masks.data(), masks.size()),
                      "aclshmemx_set_udma_peer_qp_masks");
                sparse_qp_enabled_ = !masks.empty();
            }
        }
        const auto mode = mpi ? ACLSHMEMX_INIT_WITH_MPI : ACLSHMEMX_INIT_WITH_DEFAULT;
        if (use_1g_huge_pages_) {
            allocate_huge_workspace(allocation_bytes);
            aclshmemx_buffer_optional_attr_t optional{};
            optional.mem_handle = huge_page_handle_;
            aclshmemx_buffer_desc_t descriptor{};
            descriptor.addr = huge_page_source_;
            descriptor.size = huge_page_bytes_;
            descriptor.optional_attr = &optional;
            // The fixed prefix is the entire workspace; only SDK metadata uses the allocator.
            attr.local_mem_size = 0;
            init_attempted = true;
            check(aclshmemx_init_attr_with_buffers(mode, &attr, &descriptor, 1), "aclshmemx_init_attr_with_buffers");
            runtime_started = true;
            workspace_ = aclshmemx_get_buffer_ptr(huge_page_source_);
            if (reinterpret_cast<uintptr_t>(workspace_) % 512U != 0)
                throw std::runtime_error("SHMEM returned an unaligned 1 GiB workspace");
        } else {
            init_attempted = true;
            check(aclshmemx_init_attr(mode, &attr), "aclshmemx_init_attr");
            runtime_started = true;
            // Peer-table aggregate packets require physical 512B boundaries.
            workspace_ = udma ? aclshmem_align(512U, allocation_bytes) : aclshmem_malloc(allocation_bytes);
        }
        if (workspace_ == nullptr) {
            throw std::runtime_error("SHMEM workspace allocation or translation failed");
        }
        check(aclrtMemset(workspace_, allocation_bytes, 0, allocation_bytes), "aclrtMemset");
        bytes_ = bytes;
        allocation_bytes_ = allocation_bytes;
        rank_ = rank;
        world_ = size;
        udma_ = udma;
        closed_ = false;
    } catch (...) {
        const auto original_error = std::current_exception();
        if (runtime_started && !use_1g_huge_pages_ && workspace_ != nullptr) {
            aclshmem_free(workspace_);
            workspace_ = nullptr;
        }
        if (!init_attempted || (runtime_started && aclshmem_finalize() == 0)) {
            try {
                release_huge_workspace();
                occupied = false;
            } catch (...) {
                // Retain the process reservation if a caller-owned mapping cannot be released.
            }
        }
        std::rethrow_exception(original_error);
    }
}

void Session::allocate_huge_workspace(uint64_t bytes)
{
    constexpr uint64_t page_bytes = 1ULL << 30;
    if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - (page_bytes - 1))
        throw std::overflow_error("1 GiB workspace size overflows during alignment");
    int32_t device = -1;
    check(aclrtGetDevice(&device), "aclrtGetDevice");
    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = ACL_HBM_MEM_HUGE1G;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;
    size_t granularity = 0;
    check(aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity),
          "aclrtMemGetAllocationGranularity(ACL_HBM_MEM_HUGE1G)");
    if (granularity != page_bytes) throw std::runtime_error("ACL_HBM_MEM_HUGE1G requires 1 GiB allocation granularity");
    huge_page_bytes_ = ((bytes + page_bytes - 1) / page_bytes) * page_bytes;
    check(aclrtMallocPhysical(&huge_page_handle_, huge_page_bytes_, &property, 0), "aclrtMallocPhysical(HUGE1G)");
    check(aclrtReserveMemAddress(&huge_page_source_, huge_page_bytes_, 0, nullptr, 0), "aclrtReserveMemAddress");
    check(aclrtMapMem(huge_page_source_, huge_page_bytes_, 0, huge_page_handle_, 0), "aclrtMapMem");
    huge_page_mapped_ = true;
    aclrtMemAccessDesc access{};
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    access.location = property.location;
    check(aclrtMemSetAccess(huge_page_source_, huge_page_bytes_, &access, 1), "aclrtMemSetAccess");
}

void Session::release_huge_workspace()
{
    if (huge_page_mapped_) {
        check(aclrtUnmapMem(huge_page_source_), "aclrtUnmapMem");
        huge_page_mapped_ = false;
    }
    if (huge_page_source_ != nullptr) {
        check(aclrtReleaseMemAddress(huge_page_source_), "aclrtReleaseMemAddress");
        huge_page_source_ = nullptr;
    }
    if (huge_page_handle_ != nullptr) {
        check(aclrtFreePhysical(huge_page_handle_), "aclrtFreePhysical");
        huge_page_handle_ = nullptr;
    }
    huge_page_bytes_ = 0;
}

#ifdef DEEPEP_BUILD_EP_GMM_FUSED
size_t Session::udma_unique_id_size()
{
    return sizeof(aclshmemx_uniqueid_t);
}

std::vector<uint8_t> Session::udma_unique_id()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    aclshmemx_uniqueid_t uid{};
    check(aclshmemx_get_uniqueid(&uid), "aclshmemx_get_uniqueid");
    std::vector<uint8_t> bytes(sizeof(uid));
    std::memcpy(bytes.data(), &uid, sizeof(uid));
    return bytes;
}

std::unique_ptr<Session> Session::create_udma(int rank, int size, uint64_t heap_bytes, uint64_t workspace_bytes,
                                              const std::string& endpoint, const std::vector<uint8_t>& uid_bytes)
{
    if (size < 1 || size > 8 || rank < 0 || rank >= size ||
        heap_bytes > uint64_t(std::numeric_limits<int64_t>::max()) || workspace_bytes == 0 ||
        workspace_bytes > heap_bytes || (!uid_bytes.empty() && uid_bytes.size() != sizeof(aclshmemx_uniqueid_t)) ||
        (uid_bytes.empty() && (endpoint.empty() || endpoint.size() >= ACLSHMEM_MAX_IP_PORT_LEN ||
                               endpoint.find('\0') != std::string::npos))) {
        throw std::invalid_argument("Invalid UDMA rank, heap, workspace, bootstrap endpoint or unique ID");
    }
    std::lock_guard<std::mutex> guard(runtime_mutex);
    if (occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
        throw std::runtime_error("A SHMEM runtime already exists in this process");
    }
    std::unique_ptr<Session> session(new Session());
    check(aclrtGetCurrentContext(&session->context_), "aclrtGetCurrentContext");
    if (session->context_ == nullptr) {
        throw std::runtime_error("Initialize the framework NPU context before creating a runtime");
    }
    const char* soc = aclrtGetSocName();
    if (soc == nullptr || std::string(soc).rfind("Ascend950", 0) != 0) {
        throw std::runtime_error("This runtime build is qualified only for Ascend950");
    }
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t uid{};
    auto mode = ACLSHMEMX_INIT_WITH_DEFAULT;
    if (uid_bytes.empty()) {
        attr.my_pe = rank;
        attr.n_pes = size;
        attr.local_mem_size = heap_bytes;
        std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
        uid.version = ACLSHMEM_UNIQUEID_VERSION;
        uid.my_pe = rank;
        uid.n_pes = size;
        attr.comm_args = &uid;
    } else {
        std::memcpy(&uid, uid_bytes.data(), sizeof(uid));
        check(aclshmemx_set_attr_uniqueid_args(rank, size, static_cast<int64_t>(heap_bytes), &uid, &attr),
              "aclshmemx_set_attr_uniqueid_args");
        mode = ACLSHMEMX_INIT_WITH_UNIQUEID;
    }
    // Preserve the locked SDK's optional-attribute version and timeout defaults.
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    occupied = true;
    check(aclshmemx_init_attr(mode, &attr), "aclshmemx_init_attr");
    try {
        session->workspace_ = aclshmem_malloc(workspace_bytes);
        if (session->workspace_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
        check(aclrtMemset(session->workspace_, workspace_bytes, 0, workspace_bytes), "aclrtMemset");
        session->bytes_ = workspace_bytes;
        session->allocation_bytes_ = workspace_bytes;
        session->rank_ = rank;
        session->world_ = size;
        session->udma_ = true;
        session->closed_ = false;
    } catch (...) {
        if (session->workspace_ != nullptr) {
            aclshmem_free(session->workspace_);
            session->workspace_ = nullptr;
        }
        if (aclshmem_finalize() == 0) {
            occupied = false;
        }
        throw;
    }
    return session;
}

uintptr_t Session::udma_workspace_address() const
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    check_context();
    if (!udma_) {
        throw std::runtime_error("This runtime does not own a UDMA workspace");
    }
    return reinterpret_cast<uintptr_t>(workspace_);
}
#endif

void Session::check_context() const
{
    if (poisoned_) {
        throw std::runtime_error("Runtime cleanup failed; restart the distributed job");
    }
    if (closed_) {
        throw std::runtime_error("Runtime is closed");
    }
    aclrtContext current = nullptr;
    check(aclrtGetCurrentContext(&current), "aclrtGetCurrentContext");
    if (current != context_) {
        throw std::runtime_error("Use the runtime on its creating NPU context");
    }
}

uintptr_t Session::workspace_address() const
{
    check_context();
    return reinterpret_cast<uintptr_t>(workspace_);
}

void Session::barrier()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    check_context();
    aclshmem_barrier_all();
}

void Session::close()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    if (closed_) {
        return;
    }
    check_context();
    aclshmem_barrier_all();
    if (workspace_ != nullptr && !use_1g_huge_pages_) {
        aclshmem_free(workspace_);
        workspace_ = nullptr;
    }
    // On failure retain the reservation and never claim successful cleanup.
    const int status = aclshmem_finalize();
    if (status != 0) {
        poisoned_ = true;
        check(status, "aclshmem_finalize (restart the distributed job)");
    }
    workspace_ = nullptr;
    try {
        release_huge_workspace();
    } catch (...) {
        poisoned_ = true;
        throw;
    }
    occupied = false;
    closed_ = true;
}
}  // namespace deepep
