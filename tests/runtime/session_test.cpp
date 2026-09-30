// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

// Link-time SDK doubles test ownership decisions, never ABI/device support.
#include "runtime/session.hpp"
#include "host/init/shmem_host_init.h"
#include <cassert>
#include <stdexcept>
#include <string>

namespace {
int initialized = 0, allocations = 0, frees = 0, finalized = 0, barriers = 0, memsets = 0;
int init_status = 0, memset_status = 0, finalize_status = 0;
bool allocation_failure = false;
int qp_count = 0;
int expected_world = 2;
uint64_t expected_bytes = 2 << 20;
int storage = 0, context_token = 0, another_context = 0;
void* current_context = &context_token;
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
    assert(destination == &storage && maximum == expected_bytes && count == expected_bytes && value == 0);
    ++memsets;
    return memset_status;
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
    const std::string mode = argv[1];
    auto create = []() { return deepep::Session(0, 2, 2 << 20, "tcp://127.0.0.1:19091", 30); };
    if (mode == "low-latency") {
        expected_world = 16;
        expected_bytes = 8ULL << 30;
        deepep::Session session(0, 16, expected_bytes, "tcp://xxx.xxx.xxx.xxx:xxxxx", 30, true);
        assert(session.capacity() == expected_bytes && session.workspace() == &storage && memsets == 1);
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
