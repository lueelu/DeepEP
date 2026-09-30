/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
// Modified by zhu-mingzhe71 2026
#ifndef DEEPEP_EP_GMM_FUSED_TILING_VALIDATION_HPP
#define DEEPEP_EP_GMM_FUSED_TILING_VALIDATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

// Host-only validation, independent of Torch/CANN so boundary cases can also
// be checked on a CPU. No function here modifies the device synchronization.
namespace deepep::ep_gmm_fused::TilingValidation {

constexpr uint64_t kSymmetricDataBytes = 400ULL * 1024 * 1024;
// CATLASS Ascend950 LocalTensorBuffer allocates ArchTag::UB_SIZE, not the
// physical 256 KiB. Device configuration headers assert this budget matches.
constexpr uint64_t kCompiledUbBytes = 248ULL * 1024;
constexpr uint64_t kWqeScratchBytes = 256;
constexpr uint64_t kIntMax = std::numeric_limits<int32_t>::max();
constexpr uint64_t kUintMax = std::numeric_limits<uint32_t>::max();

struct AllToAllV {
    uint32_t m0, n0, k0, commBlockM, stageRows, workspaceStages;
    uint32_t localCopyTileRows, mSplit, exportCoreCount, selfCopyCores;
};

struct GroupedMatMul {
    uint32_t m0, n0, k0, commInterval, localCopyTileRows;
};

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

inline uint32_t LocalCopyRowCapacity(uint32_t columns, uint64_t ubBytes)
{
    const uint64_t capacity = std::min(ubBytes, kCompiledUbBytes);
    const uint64_t alignedColumns = CeilDiv(columns, 16) * 16;
    // Two BF16 tiles plus the UDMA WQE scratch area.
    return columns == 0 || capacity <= kWqeScratchBytes
               ? 0
               : static_cast<uint32_t>((capacity - kWqeScratchBytes) / (4 * alignedColumns));
}

inline const char* CheckValues(const int64_t* values, size_t count, size_t expected, size_t zeroAllowed)
{
    if (count != expected) {
        return "tiling has an invalid field count";
    }
    for (size_t i = 0; i < count; ++i) {
        if (values[i] < (i == zeroAllowed ? 0 : 1) || values[i] > static_cast<int64_t>(kIntMax)) {
            return "tiling values must be positive int32 (export_core_count may be zero)";
        }
    }
    return nullptr;
}

inline bool IsCompiledShape(uint32_t m0, uint32_t n0, uint32_t k0, bool allowSquare)
{
    return k0 == 256 &&
           ((m0 == 128 && n0 == 256) || (m0 == 256 && n0 == 128) || (allowSquare && m0 == 256 && n0 == 256));
}

inline const char* ParseAllToAllV(const int64_t* values, size_t count, bool exportOutput, AllToAllV& result)
{
    if (const char* error = CheckValues(values, count, 10, 8)) {
        return error;
    }
    result = {static_cast<uint32_t>(values[0]), static_cast<uint32_t>(values[1]), static_cast<uint32_t>(values[2]),
              static_cast<uint32_t>(values[3]), static_cast<uint32_t>(values[4]), static_cast<uint32_t>(values[5]),
              static_cast<uint32_t>(values[6]), static_cast<uint32_t>(values[7]), static_cast<uint32_t>(values[8]),
              static_cast<uint32_t>(values[9])};
    if (!IsCompiledShape(result.m0, result.n0, result.k0, true)) {
        return "tiling MMAD shape has no compiled AllToAllV kernel instance";
    }
    if (result.stageRows % result.m0 != 0) {
        return "stage_rows must be divisible by M0";
    }
    if (result.workspaceStages != 2 && result.workspaceStages != 3) {
        return "workspace_stages must be 2 or 3";
    }
    if (result.mSplit != 1 && result.mSplit != 2 && result.mSplit != 4) {
        return "m_split must be 1, 2 or 4";
    }
    if (!exportOutput && (result.workspaceStages != 2 || result.mSplit != 1 || result.exportCoreCount != 0)) {
        return "non-export tiling requires workspace_stages=2, m_split=1, export_core_count=0";
    }
    return nullptr;
}

inline const char* ParseGroupedMatMul(const int64_t* values, size_t count, GroupedMatMul& result)
{
    if (const char* error = CheckValues(values, count, 5, 5)) {
        return error;
    }
    result = {static_cast<uint32_t>(values[0]), static_cast<uint32_t>(values[1]), static_cast<uint32_t>(values[2]),
              static_cast<uint32_t>(values[3]), static_cast<uint32_t>(values[4])};
    if (!IsCompiledShape(result.m0, result.n0, result.k0, false)) {
        return "tiling MMAD shape has no compiled GroupedMatMul kernel instance";
    }
    if (result.localCopyTileRows > kIntMax / 2) {
        return "local_copy_tile_rows * 2 exceeds int32 range";
    }
    return nullptr;
}

inline const char* CheckHardware(uint32_t worldSize, uint32_t coreCount, uint64_t ubBytes)
{
    if (worldSize == 0 || worldSize > 8 || coreCount < worldSize) {
        return "tiling requires world_size in [1, 8] and at least one AI core per rank";
    }
    if (ubBytes <= kWqeScratchBytes) {
        return "failed to query a usable device UB capacity";
    }
    return nullptr;
}

inline const char* ValidateAllToAllV(const AllToAllV& tiling, uint32_t m, uint32_t k, uint32_t worldSize,
                                     uint32_t coreCount, uint64_t ubBytes)
{
    if (const char* error = CheckHardware(worldSize, coreCount, ubBytes)) {
        return error;
    }
    if (tiling.exportCoreCount > coreCount || static_cast<uint64_t>(worldSize) + tiling.selfCopyCores - 1 > coreCount) {
        return "tiling worker count exceeds the available AI cores";
    }
    // Export and communication run on different AIV subcores. Each uses two
    // tiles from UB offset zero; the communication block reserves the UB tail.
    if (4ULL * tiling.localCopyTileRows * tiling.k0 + kWqeScratchBytes > std::min(ubBytes, kCompiledUbBytes)) {
        return "local_copy_tile_rows exceeds the two-buffer UB capacity";
    }
    // Division checks keep the capacity calculation safe even for hostile input.
    const uint64_t stageElements = static_cast<uint64_t>(tiling.stageRows) * worldSize;
    // DistRowMajor aligns BF16 staging rows to 512 bytes (256 elements).
    const uint64_t alignedK = CeilDiv(k, 256) * 256;
    if (k == 0 || stageElements > kSymmetricDataBytes / 2 / tiling.workspaceStages / alignedK) {
        return "tiling exceeds the symmetric workspace capacity";
    }
    const uint64_t blockCount = CeilDiv(tiling.stageRows, tiling.commBlockM);
    const uint64_t roundCount = CeilDiv(m, tiling.stageRows);
    if (m == 0 || m > kUintMax - tiling.stageRows + 1 || blockCount * worldSize > kIntMax ||
        roundCount > kIntMax / (blockCount * worldSize)) {
        return "tiling communication schedule exceeds the int32 completion counter";
    }
    return nullptr;
}

inline const char* ValidateGroupedMatMul(const GroupedMatMul& tiling, uint32_t n, uint32_t worldSize,
                                         uint32_t coreCount, uint64_t ubBytes)
{
    if (const char* error = CheckHardware(worldSize, coreCount, ubBytes)) {
        return error;
    }
    const uint64_t blocks = static_cast<uint64_t>(coreCount) * tiling.commInterval;
    if (blocks > kUintMax || blocks % worldSize != 0) {
        return "AI-core count * comm_interval must fit uint32 and be divisible by world_size";
    }
    const uint64_t nLoops = CeilDiv(n, tiling.n0);
    if (nLoops == 0 || n > kUintMax - tiling.n0 + 1 || blocks / worldSize < nLoops) {
        return "tiling requires AI-core count * comm_interval / world_size >= ceil(N / N0)";
    }
    const uint64_t stageRows = (blocks / worldSize / nLoops) * tiling.m0;
    if (stageRows > kUintMax / worldSize || stageRows > kSymmetricDataBytes / 4 / worldSize / n) {
        return "tiling exceeds the target-order symmetric workspace capacity";
    }
    const uint64_t alignedColumns = CeilDiv(n, 16) * 16;
    if (4ULL * tiling.localCopyTileRows * alignedColumns + kWqeScratchBytes > std::min(ubBytes, kCompiledUbBytes)) {
        return "local_copy_tile_rows exceeds the two-buffer UB capacity; explicit tiling is not clamped";
    }
    return nullptr;
}

inline const char* ValidateAllToAllVGrid(const AllToAllV& tiling, uint32_t n, uint32_t worldSize, uint32_t coreCount)
{
    const uint64_t nTiles = CeilDiv(n, tiling.n0);
    const uint64_t maxTasks = static_cast<uint64_t>(tiling.stageRows / tiling.m0) * worldSize * nTiles;
    if (n == 0 || n > kUintMax - tiling.n0 + 1 || maxTasks > kUintMax - coreCount) {
        return "tiling MMAD task grid exceeds the uint32 scheduler range";
    }
    return nullptr;
}

}  // namespace deepep::ep_gmm_fused::TilingValidation
#endif
