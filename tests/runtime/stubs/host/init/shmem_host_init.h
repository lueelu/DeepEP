// Copyright (c) 2026, Lu Lu
// Modified by zhu-mingzhe71 2026

#pragma once
#include <cstddef>
#include <cstdint>
constexpr int ACLSHMEM_MAX_IP_PORT_LEN = 64;
constexpr int ACLSHMEM_STATUS_NOT_INITIALIZED = 0;
constexpr int ACLSHMEM_DATA_OP_MTE = 1;
constexpr int ACLSHMEM_DATA_OP_UDMA = 8;
using data_op_engine_type_t = int;
constexpr int ACLSHMEMX_INIT_WITH_DEFAULT = 1;
struct aclshmemx_uniqueid_t {
    char bytes[128]{};
};
struct aclshmemx_init_attr_t {
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
int aclshmem_finalize();
int aclshmemx_set_qp_num(data_op_engine_type_t, uint32_t);
int aclshmemx_set_shared_jetty(data_op_engine_type_t, bool);
