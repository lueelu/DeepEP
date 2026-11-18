// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_COMM_ARGS_H
#define DISPATCH_DEDUP_COMM_ARGS_H
#include <cstdint>
#include <cstddef>
#include "ll_compile_attrs.h"
#include "ll_core_config.h"

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#include "kernel_operator.h"
#endif

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace DispatchDedup {
constexpr int DISPATCH_DEDUP_MAX_RANK_SIZE = 256;  // 最大支持的npu卡数
constexpr int DFX_COUNT = 50;
// Flag tail reserved at the end of every symmetric window.
constexpr int DISPATCH_DEDUP_FLAG_BUFF_BYTES = 4 * 1024 * 1024;

// Tiling dtype codes. The values are the Host/device wire ABI and are pinned
// against kLlDispatch* by the static_asserts in ops/low_latency.cpp.
enum DispatchDedupDataType {
    DISPATCH_DEDUP_DATA_TYPE_FP16 = 3,
    DISPATCH_DEDUP_DATA_TYPE_BFP16 = 11,
    DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 = 15,
    DISPATCH_DEDUP_DATA_TYPE_FP8E5M2 = 16,
    DISPATCH_DEDUP_DATA_TYPE_RESERVED = 255
};

struct CommArgs {
    int rank = 0;  // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0;        // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0;  // legacy collective flag map; unused by LL
    GM_ADDR peerMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // 传入初始化获得的buff，所有allreduce都是同一个参数

    // Legacy collective fields. LL never writes or reads them, but the
    // 256-rank CommArgs size and offsets are a frozen Host/device ABI.
    GM_ADDR creditMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};
    int64_t sendCountMatrix[DISPATCH_DEDUP_MAX_RANK_SIZE * DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // for all2allv
    int64_t dfx[DFX_COUNT] = {};
    GM_ADDR dumpAddr = nullptr;
    int32_t magics[DISPATCH_DEDUP_MAX_RANK_SIZE] = {0};

    uint64_t fftsVal = 0;
    GM_ADDR udmaInfoPtr = nullptr;
    GM_ADDR udmaRegistryPtr = nullptr;
    GM_ADDR sdmaWorkspacePtr = nullptr;
    GM_ADDR udmaFullmeshPtr = nullptr;
    uint64_t udmaRegistrationGeneration = 0U;
    uint64_t peerMemBytes = 0U;
    int32_t commDomain = 0;
    uint32_t reserved0 = 0U;  // LL v4: physical ranks per host (production sets 16).
    // LL ABI v4: RDMA uses addresses[0:2] for symmetric workspace/staging,
    // config[0:2] for backend/QP count. MTE and other paths keep this tail zero.
    GM_ADDR reservedAddresses[4] = {};
    uint32_t reservedConfig[2] = {};
};

static_assert(sizeof(CommArgs) == 529944U, "256-rank communication ABI; rebuild Host and kernel together");
static_assert(offsetof(CommArgs, reservedAddresses) == 529904U, "reserved ABI tail offset");

}  // namespace DispatchDedup
#endif  // DISPATCH_DEDUP_COMM_ARGS_H
