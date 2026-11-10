// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026
#ifndef DISPATCH_DEDUP_TYPES_H
#define DISPATCH_DEDUP_TYPES_H

#include <cstdint>

#ifndef __CCE__
#include <map>
#include <string>
#endif

namespace UbxDispatch {
constexpr int DISPATCH_DEDUP_SUCCESS = 0;
constexpr int DISPATCH_DEDUP_ERROR_NOT_INITIALIZED = -1;
constexpr int DISPATCH_DEDUP_ERROR_MKIRT = -2;
constexpr int DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL = -3;
constexpr int DISPATCH_DEDUP_ERROR_INTERNAL = -4;
constexpr int DISPATCH_DEDUP_ERROR_TIMEOUT = -5;
constexpr int DISPATCH_DEDUP_ERROR_NOT_SUPPORT = -6;
constexpr int DISPATCH_DEDUP_ERROR_NOT_FOUND = -7;
constexpr int64_t DISPATCH_DEDUP_INVALID_VALUE = -1;

// shared buffer size，这里要和collectives.cce文件中的常量联动修改！！！
constexpr int64_t DISPATCH_DEDUP_BUFF_BYTES = 2048LL * 1024 * 1024;
constexpr int DISPATCH_DEDUP_FLAG_BUFF_BYTES = 4 * 1024 * 1024;
constexpr int DISPATCH_DEDUP_COMM_BUFFER_SIZE = 2044;        // 单位MB
constexpr int DISPATCH_DEDUP_MEMORY_DOMAIN_BUFFER_SIZE = 8;  // 单位MB

enum class ChipName {
    CHIP_310P3 = 0,
    CHIP_910B1,
    CHIP_910B2,
    CHIP_910B3,
    CHIP_910B4,
    CHIP_910B41,
    CHIP_910B2C,
    CHIP_910_9391,
    CHIP_910_9381,
    CHIP_910_9392,
    CHIP_910_9382,
    CHIP_910_9372,
    CHIP_910_9361,
    CHIP_910_9362,
    CHIP_910A5,
    CHIP_950PR,
    CHIP_950,
    RESERVED,
};

enum class PhysicalLink {
    HCCS = 0,
    PCIE = 1,
    RESERVED,
};

enum UbxDispatchDataType {
    DISPATCH_DEDUP_DATA_TYPE_INT8 = 0,
    DISPATCH_DEDUP_DATA_TYPE_INT16 = 1,
    DISPATCH_DEDUP_DATA_TYPE_INT32 = 2,
    DISPATCH_DEDUP_DATA_TYPE_FP16 = 3,
    DISPATCH_DEDUP_DATA_TYPE_FP32 = 4,
    DISPATCH_DEDUP_DATA_TYPE_INT64 = 5,
    DISPATCH_DEDUP_DATA_TYPE_UINT64 = 6,
    DISPATCH_DEDUP_DATA_TYPE_UINT8 = 7,
    DISPATCH_DEDUP_DATA_TYPE_UINT16 = 8,
    DISPATCH_DEDUP_DATA_TYPE_UINT32 = 9,
    DISPATCH_DEDUP_DATA_TYPE_FP64 = 10,
    DISPATCH_DEDUP_DATA_TYPE_BFP16 = 11,
    DISPATCH_DEDUP_DATA_TYPE_INT128 = 12,
    DISPATCH_DEDUP_DATA_TYPE_HIF8 = 14,
    DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 = 15,
    DISPATCH_DEDUP_DATA_TYPE_FP8E5M2 = 16,
    DISPATCH_DEDUP_DATA_TYPE_FP8E8M0 = 17,
    DISPATCH_DEDUP_DATA_TYPE_RESERVED = 255
};

enum UbxDispatchReduceOp {
    DISPATCH_DEDUP_REDUCE_SUM = 0,
    DISPATCH_DEDUP_REDUCE_PROD = 1,
    DISPATCH_DEDUP_REDUCE_MAX = 2,
    DISPATCH_DEDUP_REDUCE_MIN = 3,
    DISPATCH_DEDUP_REDUCE_RESERVED = 255
};

// 包含 物理链路、芯片名称 信息。
struct PhysicalInfo {
    ChipName chipName = ChipName::RESERVED;
    PhysicalLink physicalLink = PhysicalLink::RESERVED;
    uint32_t coreNum = 0;
};

enum class UbxDispatchType {
    ALL_REDUCE = 1,
    REDUCE_SCATTER = 2,
    ALL_GATHER = 3,
    BROADCAST = 4,
    ALL2ALL = 5,
    ALL2ALL_V_C = 6,
    GATHER = 7,
    LOCAL_REDUCE = 8,
    SEND = 9,
    RECV = 10,
    PURE_MATMUL = 101,
    MATMUL_ALL_REDUCE = 102,
    MATMUL_REDUCE_SCATTER = 103,
    ALL_GATHER_MATMUL = 104,
    ALL_GATHER_MATMUL_V2 = 105,
    ALL2ALL_MATMUL = 106,
    MATMUL_ALL2ALL = 107,
    MTE2_TEST = 108,
    ALL_GATHER_MATMUL_REDUCE_SCATTER = 111,
    DISPATCH_DEDUP_TYPE_MAX = 310,
    BANDWIDTH = 201,

    ALLTOALLV_ALLGATHER_MATMUL = 305,
    MATMUL_REDUCESCATTER_ALLTOALLV = 306,
    ALLTOALLVC_ALLGATHER_MATMUL = 307,
    MATMUL_REDUCESCATTER_ALLTOALLVC = 308,
    ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN = 309,
    MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN = 310,
    LCAL_TYPE_MAX = 311
};

#ifndef __CCE__
const std::map<UbxDispatchType, std::string> DISPATCH_DEDUP_TYPE2NAME = {
    {UbxDispatchType::ALL_REDUCE, "UbxDispatchAllReduce"},
    {UbxDispatchType::REDUCE_SCATTER, "UbxDispatchReduceScatter"},
    {UbxDispatchType::ALL_GATHER, "UbxDispatchAllGather"},
    {UbxDispatchType::BROADCAST, "UbxDispatchBroadcast"},
    {UbxDispatchType::PURE_MATMUL, "UbxDispatchPureMatmul"},
    {UbxDispatchType::MATMUL_ALL_REDUCE, "UbxDispatchMatmulAllReduce"},
    {UbxDispatchType::MATMUL_REDUCE_SCATTER, "UbxDispatchMatmulReduceScatter"},
    {UbxDispatchType::ALL_GATHER_MATMUL, "UbxDispatchAllGatherMatmul"},
    {UbxDispatchType::ALL_GATHER_MATMUL_V2, "UbxDispatchAllGatherMatmulV2"},
    {UbxDispatchType::ALL2ALL_MATMUL, "UbxDispatchAll2AllMatmul"},
    {UbxDispatchType::MATMUL_ALL2ALL, "UbxDispatchMatmulAll2All"},
    {UbxDispatchType::MTE2_TEST, "UbxDispatchMTE2Test"},
    {UbxDispatchType::ALL2ALL, "UbxDispatchAll2All"},
    {UbxDispatchType::ALL2ALL_V_C, "UbxDispatchAll2AllVC"},
    {UbxDispatchType::ALL_GATHER_MATMUL_REDUCE_SCATTER, "UbxDispatchAllGatherMatmulReduceScatter"},
    {UbxDispatchType::BANDWIDTH, "UbxDispatchBandwidthTest"},
    {UbxDispatchType::LOCAL_REDUCE, "UbxDispatchLocalReduce"},
    {UbxDispatchType::GATHER, "UbxDispatchGather"},
    {UbxDispatchType::SEND, "UbxDispatchSend"},
    {UbxDispatchType::RECV, "UbxDispatchRecv"},
    {UbxDispatchType::ALLTOALLV_ALLGATHER_MATMUL, "UbxDispatchAllToAllVAllGatherMatmul"},
    {UbxDispatchType::MATMUL_REDUCESCATTER_ALLTOALLV, "UbxDispatchMatmulReduceScatterAllToAllV"},

    {UbxDispatchType::ALLTOALLVC_ALLGATHER_MATMUL, "UbxDispatchAllToAllVAllGatherMatmul"},
    {UbxDispatchType::MATMUL_REDUCESCATTER_ALLTOALLVC, "UbxDispatchMatmulReduceScatterAllToAllV"}};
#endif

}  // namespace UbxDispatch
#endif  // DISPATCH_DEDUP_TYPES_H
