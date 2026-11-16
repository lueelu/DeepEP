// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once
#include <cstdint>

// Test-only topology view for Session's optional peer-access check. It does
// not model the real SDK layout and must never be linked against SHMEM.
struct aclshmem_device_host_state_t {
    int mype;
    int npes;
    uint8_t topo_list[512];
};
