// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstddef>
#include <cstdint>
constexpr int ACLSHMEM_MAX_IP_PORT_LEN = 64;
constexpr int ACLSHMEM_STATUS_NOT_INITIALIZED = 0;
constexpr int ACLSHMEM_DATA_OP_MTE = 1;
constexpr int ACLSHMEM_DATA_OP_UDMA = 8;
using data_op_engine_type_t = int;
constexpr int ACLSHMEMX_INIT_WITH_DEFAULT = 1;
constexpr int ACLSHMEMX_INIT_WITH_MPI = 2;
constexpr int ACLSHMEMX_INIT_WITH_UNIQUEID = 4;
struct aclshmemx_uniqueid_t {
    char bytes[128]{};
};
struct aclshmemx_init_attr_t {
    uint64_t instance_id = 0;
    int my_pe = 0;
    int n_pes = 0;
    uint64_t local_mem_size = 0;
    char ip_port[64]{};
    void* comm_args = nullptr;
    struct {
        int data_op_engine_type = 0;
        unsigned shm_init_timeout = 120, shm_create_timeout = 120, control_operation_timeout = 120;
    } option_attr;
};
int aclshmemx_init_status();
int aclshmemx_init_attr(int, aclshmemx_init_attr_t*);
struct aclshmemx_buffer_optional_attr_t {
    void* mem_handle = nullptr;
    const void* fabric_handle = nullptr;
    uint64_t reserved[6]{};
};
struct aclshmemx_buffer_desc_t {
    void* addr = nullptr;
    uint64_t size = 0;
    const aclshmemx_buffer_optional_attr_t* optional_attr = nullptr;
    uint64_t reserved[5]{};
};
int aclshmemx_init_attr_with_buffers(int, aclshmemx_init_attr_t*, const aclshmemx_buffer_desc_t*, size_t);
void* aclshmemx_get_buffer_ptr(const void*);
int aclshmem_finalize();
int aclshmemx_set_qp_num(data_op_engine_type_t, uint32_t);
int aclshmemx_set_shared_jetty(data_op_engine_type_t, bool);

int aclshmemx_instance_ctx_set(uint64_t);
int aclshmemx_finalize(uint64_t);
uint64_t shmemx_get_ffts_config();
