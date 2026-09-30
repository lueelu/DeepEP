// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#include "runtime/session.hpp"
#include <cstring>
#include <cstdlib>
#include <vector>
#include "kernels/udma_layout.hpp"
#include <mutex>
#include <stdexcept>
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_cc.h"
#ifdef DEEPEP_BUILD_EP_GMM_FUSED
#include <limits>
#endif

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
                 bool megamoe)
{
    constexpr uint64_t alignment = 2ULL * 1024 * 1024;
    if (size < 2 || size > (megamoe ? 128 : 256) || (!megamoe && (size & (size - 1))) || rank < 0 || rank >= size ||
        bytes < alignment || bytes > (megamoe ? 1ULL : 32ULL) * 1024 * 1024 * 1024 || bytes % alignment != 0 ||
        timeout < 1 || timeout > 300 || (megamoe && udma) || endpoint.empty() ||
        endpoint.size() >= ACLSHMEM_MAX_IP_PORT_LEN || endpoint.find('\0') != std::string::npos) {
        throw std::invalid_argument("Invalid rank, capacity, timeout or bootstrap endpoint");
    }
    // BF16 and FP8 use the same peer-table transport at every supported EP size.
    peer_table_ = udma;
    if (peer_table_ && (size < 2 || size > 256 || (size & (size - 1)))) {
        throw std::invalid_argument("peer-table requires EP2/4/8/16/32/64/128/256");
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
    attr.local_mem_size = bytes;
    std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    attr.comm_args = &uid;
    attr.option_attr.data_op_engine_type =
        udma ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA) : ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = timeout;
    attr.option_attr.shm_create_timeout = timeout;
    attr.option_attr.control_operation_timeout = timeout;
    // A failed initialization can leave SDK state behind. Keep the process
    // reserved until a new process is launched; do not finalize foreign state.
    occupied = true;
    if (udma) {
        // Combine owns pairs 0..15; peer-table Dispatch uses pairs 0..7.
        // Serialized operators reuse the bank, not concurrent SQ writers.
        // The pinned SDK shares Clos SQs across peers; MESH routes are separate.
        check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, ascend_deepep::udma_layout::kQpCount),
              "aclshmemx_set_qp_num");
        check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true), "aclshmemx_set_shared_jetty");
    }
    check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
    try {
        // Peer-table aggregate packets require physical 512B boundaries, not
        // merely 512B-relative offsets within a possibly 16B-aligned allocation.
        workspace_ = udma ? aclshmem_align(512U, bytes) : aclshmem_malloc(bytes);
        if (workspace_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
        check(aclrtMemset(workspace_, bytes, 0, bytes), "aclrtMemset");
        bytes_ = bytes;
        rank_ = rank;
        world_ = size;
        udma_ = udma;
        closed_ = false;
    } catch (...) {
        if (workspace_ != nullptr) {
            aclshmem_free(workspace_);
            workspace_ = nullptr;
        }
        if (aclshmem_finalize() == 0) {
            occupied = false;
        }
        throw;
    }
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
    if (workspace_ != nullptr) {
        aclshmem_free(workspace_);
        workspace_ = nullptr;
    }
    // On failure retain the reservation and never claim successful cleanup.
    const int status = aclshmem_finalize();
    if (status != 0) {
        poisoned_ = true;
        check(status, "aclshmem_finalize (restart the distributed job)");
    }
    occupied = false;
    closed_ = true;
}
}  // namespace deepep
