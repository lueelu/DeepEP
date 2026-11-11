// Copyright (c) 2026, Lu Lu
// Modified by nino888 2026

#pragma once
#include <cstddef>
#include <cstdint>
using aclrtContext = void*;
using aclrtStream = void*;
int aclrtCreateStream(aclrtStream*);
int aclrtSynchronizeStream(aclrtStream);
int aclrtDestroyStream(aclrtStream);
using aclrtDrvMemHandle = void*;
constexpr int ACL_MEM_HANDLE_TYPE_NONE = 0;
constexpr int ACL_MEM_ALLOCATION_TYPE_PINNED = 1;
constexpr int ACL_HBM_MEM_HUGE1G = 2;
constexpr int ACL_MEM_LOCATION_TYPE_DEVICE = 1;
constexpr int ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM = 0;
constexpr int ACL_RT_MEM_ACCESS_FLAGS_READWRITE = 3;
struct aclrtMemLocation {
    int type = 0;
    int id = -1;
};
struct aclrtPhysicalMemProp {
    int handleType = 0;
    int allocationType = 0;
    int memAttr = 0;
    aclrtMemLocation location{};
    uint64_t reserve = 0;
};
struct aclrtMemAccessDesc {
    int flags = 0;
    aclrtMemLocation location{};
};
int aclrtGetDevice(int32_t*);
int aclrtMemGetAllocationGranularity(const aclrtPhysicalMemProp*, int, size_t*);
int aclrtMallocPhysical(aclrtDrvMemHandle*, size_t, const aclrtPhysicalMemProp*, uint64_t);
int aclrtReserveMemAddress(void**, size_t, size_t, void*, uint64_t);
int aclrtMapMem(void*, size_t, size_t, aclrtDrvMemHandle, uint64_t);
int aclrtMemSetAccess(void*, size_t, const aclrtMemAccessDesc*, size_t);
int aclrtUnmapMem(void*);
int aclrtReleaseMemAddress(void*);
int aclrtFreePhysical(aclrtDrvMemHandle);
int aclrtGetCurrentContext(aclrtContext*);
const char* aclrtGetSocName();
int aclrtMemset(void*, size_t, int, size_t);
