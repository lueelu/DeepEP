// Copyright (c) 2026, Lu Lu

// Link-time SDK doubles test ownership decisions, never ABI/device support.
#include "runtime/session.hpp"
#include "host/init/shmem_host_init.h"
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int initialized = 0, allocations = 0, frees = 0, finalized = 0, barriers = 0, memsets = 0;
int init_status = 0, memset_status = 0, finalize_status = 0;
bool allocation_failure = false;
int qp_count = 0;
int expected_world = 2;
uint64_t expected_bytes = 2 << 20;
int storage = 0, context_token = 0, another_context = 0;
void* current_context = &context_token;
void* source_address = reinterpret_cast<void*>(0x100000000ULL);
void* symmetric_address = reinterpret_cast<void*>(0x200000000ULL);
int physical_token = 0;
uint64_t physical_bytes = 0;
size_t page_granularity = 1ULL << 30;
std::string fault;
std::vector<std::string> physical_calls;
bool huge_registered = false;
template <class Action>
void fails(Action action)
{
    bool failed = false;
    try {
        action();
    } catch (const std::exception&) {
        failed = true;
    }
    assert(failed);
}
}  // namespace
int aclrtGetCurrentContext(aclrtContext* context)
{
    *context = current_context;
    return 0;
}
const char* aclrtGetSocName()
{
    return "Ascend950";
}
int aclrtMemset(void* destination, size_t maximum, int value, size_t count)
{
    assert(destination == (huge_registered ? symmetric_address : &storage));
    assert(maximum == expected_bytes && count == expected_bytes && value == 0);
    ++memsets;
    return memset_status;
}
int aclrtGetDevice(int32_t* device)
{
    *device = 3;
    return 0;
}
int aclrtMemGetAllocationGranularity(const aclrtPhysicalMemProp* p, int mode, size_t* result)
{
    assert(p->handleType == ACL_MEM_HANDLE_TYPE_NONE && p->allocationType == ACL_MEM_ALLOCATION_TYPE_PINNED);
    assert(p->memAttr == ACL_HBM_MEM_HUGE1G && p->location.id == 3 && p->location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    assert(p->reserve == 0 && mode == ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM);
    *result = page_granularity;
    return fault == "granularity" ? -1 : 0;
}
int aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t bytes, const aclrtPhysicalMemProp* p, uint64_t flags)
{
    assert(p->memAttr == ACL_HBM_MEM_HUGE1G && flags == 0);
    assert(bytes == (((expected_bytes + (1ULL << 30) - 1) >> 30) << 30));
    physical_calls.push_back("allocate");
    if (fault == "allocate") return -1;
    *handle = &physical_token;
    physical_bytes = bytes;
    return 0;
}
int aclrtReserveMemAddress(void** addr, size_t bytes, size_t alignment, void* requested, uint64_t flags)
{
    assert(bytes == physical_bytes && alignment == 0 && requested == nullptr && flags == 0);
    physical_calls.push_back("reserve");
    if (fault == "reserve") return -1;
    *addr = source_address;
    return 0;
}
int aclrtMapMem(void* addr, size_t bytes, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    assert(addr == source_address && bytes == physical_bytes && offset == 0 && handle == &physical_token && flags == 0);
    physical_calls.push_back("map");
    return fault == "map" ? -1 : 0;
}
int aclrtMemSetAccess(void* addr, size_t bytes, const aclrtMemAccessDesc* access, size_t count)
{
    assert(addr == source_address && bytes == physical_bytes && count == 1);
    assert(access->flags == ACL_RT_MEM_ACCESS_FLAGS_READWRITE && access->location.id == 3);
    assert(access->location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    physical_calls.push_back("access");
    return fault == "access" ? -1 : 0;
}
int aclrtUnmapMem(void* addr)
{
    assert(addr == source_address && initialized == 0);
    physical_calls.push_back("unmap");
    return fault == "unmap" ? -1 : 0;
}
int aclrtReleaseMemAddress(void* addr)
{
    assert(addr == source_address && initialized == 0);
    physical_calls.push_back("release-address");
    return fault == "release-address" ? -1 : 0;
}
int aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    assert(handle == &physical_token && initialized == 0);
    physical_calls.push_back("free-physical");
    return fault == "free-physical" ? -1 : 0;
}
int aclshmemx_init_attr_with_buffers(int mode, aclshmemx_init_attr_t* attr, const aclshmemx_buffer_desc_t* descriptors,
                                     size_t count)
{
    assert(mode == ACLSHMEMX_INIT_WITH_DEFAULT || mode == ACLSHMEMX_INIT_WITH_MPI);
    assert(attr->n_pes == expected_world && attr->local_mem_size == 0 && count == 1);
    assert(attr->option_attr.data_op_engine_type == (ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA));
    assert(descriptors->addr == source_address && descriptors->size == physical_bytes);
    assert(descriptors->optional_attr->mem_handle == &physical_token);
    assert(descriptors->optional_attr->fabric_handle == nullptr);
    for (auto value : descriptors->reserved) assert(value == 0);
    for (auto value : descriptors->optional_attr->reserved) assert(value == 0);
    physical_calls.push_back("register");
    if (init_status == 0) initialized = 2;
    huge_registered = true;
    return init_status;
}
void* aclshmemx_get_buffer_ptr(const void* addr)
{
    assert(addr == source_address && initialized == 2);
    physical_calls.push_back("translate");
    return fault == "translate"   ? nullptr
           : fault == "unaligned" ? static_cast<char*>(symmetric_address) + 16
                                  : symmetric_address;
}
int aclshmemx_init_status()
{
    return initialized;
}
int aclshmemx_init_attr(int, aclshmemx_init_attr_t* attr)
{
    assert(attr->n_pes == expected_world && attr->local_mem_size == expected_bytes);
    assert(attr->option_attr.data_op_engine_type == (ACLSHMEM_DATA_OP_MTE | (qp_count ? ACLSHMEM_DATA_OP_UDMA : 0)));
    if (init_status == 0) {
        initialized = 2;
    }
    return init_status;
}
int aclshmemx_set_shared_jetty(data_op_engine_type_t engine, bool enabled)
{
    assert(engine == ACLSHMEM_DATA_OP_UDMA && enabled);
    return 0;
}
int aclshmemx_set_qp_num(data_op_engine_type_t engine, uint32_t count)
{
    assert(engine == ACLSHMEM_DATA_OP_UDMA && count == 32 && initialized == 0);
    qp_count = count;
    return 0;
}
int aclshmem_finalize()
{
    if (huge_registered) physical_calls.push_back("finalize");
    ++finalized;
    if (finalize_status == 0) {
        initialized = 0;
    }
    return finalize_status;
}
void* aclshmem_malloc(size_t bytes)
{
    assert(bytes == expected_bytes);
    ++allocations;
    return allocation_failure ? nullptr : &storage;
}
// Match the aligned allocation API used by Session; reuse the existing fault-injection stub.
void* aclshmem_align(size_t, size_t bytes)
{
    return aclshmem_malloc(bytes);
}
void aclshmem_free(void* value)
{
    assert(value == &storage);
    ++frees;
}
void aclshmem_barrier_all()
{
    ++barriers;
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    // These SDK doubles intentionally exercise the explicit dense compatibility path.
    setenv("DEEPEP_SHMEM_DENSE_QPS", "1", 1);
    const std::string mode = argv[1];
    auto create = []() { return deepep::Session(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30); };
    if (mode.rfind("huge-", 0) == 0) {
        fault = mode.substr(5);
        if (fault == "rounded") expected_bytes = (1ULL << 30) + (2 << 20);
        if (fault == "exact") expected_bytes = 4ULL << 30;
        if (fault == "ll-guard") expected_bytes = (8ULL << 30) + (2ULL << 20);
        if (fault == "overflow") expected_bytes = (1ULL << 63) - (2 << 20);
        if (fault == "wrong-granularity") page_granularity = 2 << 20;
        if (fault == "init") init_status = -1;
        if (fault == "memset") memset_status = -1;
        if (fault == "finalize") finalize_status = -1;
        const uint64_t logical_bytes = fault == "ll-guard" ? 8ULL << 30 : expected_bytes;
        auto create_huge = [&]() {
            return deepep::Session(0, 2, logical_bytes, "tcp://127.0.0.1:19091", 30, true, false, fault == "mpi", true,
                                   true);
        };
        const std::vector<std::string> setup{"allocate", "reserve", "map", "access", "register", "translate"};
        if (fault == "normal" || fault == "rounded" || fault == "exact" || fault == "ll-guard" || fault == "mpi" ||
            fault == "abandoned" || fault == "finalize" || fault == "unmap" || fault == "release-address" ||
            fault == "free-physical") {
            auto session = create_huge();
            assert(session.use_1g_huge_pages() && session.workspace() == symmetric_address);
            assert(session.workspace_allocation_bytes() == physical_bytes && session.capacity() == logical_bytes);
            assert(allocations == 0 && frees == 0 && physical_calls == setup && memsets == 1);
            if (fault == "abandoned") return 0;
            if (fault == "finalize" || fault == "unmap" || fault == "release-address" || fault == "free-physical") {
                fails([&]() { session.close(); });
                const auto calls = physical_calls;
                fails([&]() { session.close(); });
                fails([&]() { session.barrier(); });
                assert(deepep::Session::busy() && !session.closed() && calls == physical_calls);
                assert(physical_calls.back() == fault && finalized == 1 && frees == 0);
                return 0;
            }
            session.close();
            session.close();
            auto expected = setup;
            expected.insert(expected.end(), {"finalize", "unmap", "release-address", "free-physical"});
            assert(physical_calls == expected && frees == 0 && finalized == 1 && !deepep::Session::busy());
        } else {
            fails(create_huge);
            assert(frees == 0 && allocations == 0 && deepep::Session::busy() == (fault == "init"));
            std::vector<std::string> expected;
            if (fault == "allocate") expected = {"allocate"};
            if (fault == "reserve") expected = {"allocate", "reserve", "free-physical"};
            if (fault == "map") expected = {"allocate", "reserve", "map", "release-address", "free-physical"};
            if (fault == "access")
                expected = {"allocate", "reserve", "map", "access", "unmap", "release-address", "free-physical"};
            if (fault == "init") expected = {"allocate", "reserve", "map", "access", "register"};
            if (fault == "translate" || fault == "unaligned" || fault == "memset") {
                expected = setup;
                expected.insert(expected.end(), {"finalize", "unmap", "release-address", "free-physical"});
            }
            assert(physical_calls == expected);
        }
    } else if (mode == "low-latency" || mode.rfind("ll-rank-bias-", 0) == 0) {
        expected_world = mode == "low-latency" ? 16 : std::stoi(mode.substr(13));
        const uint64_t logical_bytes = 8ULL << 30;
        expected_bytes = logical_bytes + (2ULL << 20);
        // Exercise the highest rank, full guard allocation, original-pointer
        // ownership and logical capacity independently of the EP size.
        deepep::Session session(expected_world - 1, expected_world, logical_bytes, "tcp://xxx.xxx.xxx.xxx:xxxxx", 30,
                                true);
        assert(session.capacity() == logical_bytes && session.workspace() == &storage && memsets == 1);
        assert(session.workspace_allocation_bytes() == expected_bytes);
        assert(uint64_t(expected_world - 1) * 1024U < expected_bytes - logical_bytes);
        session.close();
        assert(frees == 1 && finalized == 1 && !deepep::Session::busy());
    } else if (mode == "ep16-small-workspace") {
        expected_world = 16;
        deepep::Session session(0, 16, expected_bytes, "tcp://xxx.xxx.xxx.xxx:xxxxx", 30, true);
        assert(session.capacity() == (2 << 20));
        session.close();
        assert(frees == 1 && finalized == 1 && !deepep::Session::busy());
    } else if (mode == "udma") {
        deepep::Session session(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30, true);
        assert(session.udma() && qp_count == 32);
        session.close();
        assert(!deepep::Session::busy());
    } else if (mode == "megamoe-train") {
        // Training has its own range/window contract and shares process exclusion.
        expected_world = 768;
        expected_bytes = 3ULL << 30;
        fails([]() {
            deepep::Session s(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30, true, false, false, false, false, true);
        });
        assert(!deepep::Session::busy());
        deepep::Session session(767, expected_world, expected_bytes, "tcp://127.0.0.1:19091", 30, false, false, false,
                                false, false, true);
        assert(session.capacity() == expected_bytes && !session.udma());
        assert(session.workspace_address() == reinterpret_cast<uintptr_t>(session.workspace()));
        fails(create);
        session.close();
        session.close();
        assert(frees == 1 && finalized == 1 && !deepep::Session::busy());
    } else if (mode == "megamoe") {
        fails([]() { deepep::Session s(0, 256, 2 << 20, "tcp://127.0.0.1:19091", 30, false, true); });
        fails([]() { deepep::Session s(0, 2, 2ULL << 30, "tcp://127.0.0.1:19091", 30, false, true); });
        fails([]() { deepep::Session s(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30, true, true); });
        assert(!deepep::Session::busy() && allocations == 0);
        deepep::Session session(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30, false, true);
        assert(!session.udma() && qp_count == 0);
        assert(session.workspace_address() == reinterpret_cast<uintptr_t>(session.workspace()));
        assert(session.rank() == 0 && session.world() == 2);
        session.close();
        fails([&]() { session.workspace_address(); });
        assert(!deepep::Session::busy());
    } else if (mode == "invalid") {
        fails([]() { deepep::Session s(0, 2, 3, "x", 0); });
        assert(!deepep::Session::busy() && allocations == 0);
    } else if (mode == "init-fault") {
        init_status = -1;
        fails(create);
        assert(deepep::Session::busy() && finalized == 0);
    } else if (mode == "allocation-fault" || mode == "memset-fault") {
        allocation_failure = mode == "allocation-fault";
        memset_status = mode == "memset-fault" ? -1 : 0;
        fails(create);
        assert(!deepep::Session::busy() && finalized == 1);
        assert(frees == (allocation_failure ? 0 : 1));
    } else if (mode == "foreign") {
        initialized = 2;
        fails(create);
        assert(finalized == 0 && allocations == 0);
    } else if (mode == "abandoned") {
        {
            auto session = create();
            assert(!session.closed());
        }
        assert(deepep::Session::busy() && finalized == 0 && frees == 0);
    } else {
        auto session = create();
        fails(create);
        current_context = &another_context;
        fails([&]() { session.close(); });
        assert(finalized == 0 && frees == 0);
        current_context = &context_token;
        if (mode == "close-fault") {
            finalize_status = -1;
            fails([&]() { session.close(); });
            assert(!session.closed() && deepep::Session::busy() && frees == 1);
            fails([&]() { session.close(); });
            fails([&]() { session.barrier(); });
            assert(finalized == 1 && frees == 1);
            return 0;
        }
        session.barrier();
        session.close();
        session.close();
        assert(session.closed() && !deepep::Session::busy() && frees == 1);
        fails([&]() { session.barrier(); });
        auto next = create();
        next.close();
        assert(frees == 2);
    }
}
