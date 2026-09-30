// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_COMM_ARGS_H
#define DISPATCH_DEDUP_COMM_ARGS_H
#include <cstdint>
#include <cstddef>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE 1
#else
#define DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE 0
#endif

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#include "kernel_operator.h"
#endif

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE && !defined(FORCE_INLINE_AICORE)
#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#endif

namespace DispatchDedup {
constexpr int DISPATCH_DEDUP_MAX_RANK_SIZE = 128;  // 最大支持的npu卡数
constexpr int RANK_SIZE_TWO = 2;  // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模
constexpr int64_t IPC_BUFF_MAX_SIZE = 1022LL * 1024 * 1024;
constexpr int64_t CREDIT_IPC_STRIDE = 512;
constexpr int64_t CREDIT_IPC_SLOT_BYTES = static_cast<int64_t>(DISPATCH_DEDUP_MAX_RANK_SIZE) * CREDIT_IPC_STRIDE;
constexpr int64_t DISPATCH_CREDIT_IPC_BYTES = 2 * CREDIT_IPC_SLOT_BYTES;
constexpr int64_t COMBINE_CREDIT_IPC_SIGNAL_BYTES = 64;
constexpr int64_t COMBINE_CREDIT_IPC_TRANSITION_COUNT = 7;
constexpr int64_t COMBINE_CREDIT_IPC_CORE_COUNT = 16;
constexpr int64_t COMBINE_CREDIT_IPC_PLANE_BYTES =
    COMBINE_CREDIT_IPC_TRANSITION_COUNT * COMBINE_CREDIT_IPC_CORE_COUNT * COMBINE_CREDIT_IPC_SIGNAL_BYTES;
constexpr int64_t COMBINE_CREDIT_IPC_EPOCH_COUNT = 2;
constexpr int64_t COMBINE_CREDIT_IPC_BYTES = COMBINE_CREDIT_IPC_EPOCH_COUNT * COMBINE_CREDIT_IPC_PLANE_BYTES;
constexpr int64_t COMBINE_CREDIT_IPC_BASE = DISPATCH_CREDIT_IPC_BYTES;
constexpr int64_t CREDIT_IPC_BYTES = COMBINE_CREDIT_IPC_BASE + COMBINE_CREDIT_IPC_BYTES;
static_assert(COMBINE_CREDIT_IPC_BASE % COMBINE_CREDIT_IPC_SIGNAL_BYTES == 0,
              "Combine Credit IPC base must be cache-line aligned");
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024;  // 每个ping-pong分区前2MB作为flag，之后1022MB存储数据
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;             // cce 算子在用
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

// 2step算法中，2个aiv真正用作数据预处理
constexpr int64_t PRE_CORE_REAL_NUM = 2;

constexpr int64_t ALLREDUCE_4_STEP_PARALLELISM_910A5 = 2;

constexpr int64_t AIV_PER_AICORE = 2;

constexpr int DFX_COUNT = 50;

constexpr int64_t HALF_NUM = 2;

constexpr int64_t THREE_NUM = 3;

constexpr int64_t FOUR_NUM = 4;

constexpr int64_t VADD_MAX_REPEAT = 255;
constexpr int64_t VADD_UNIT_BYTE = 256;

// vadd单位粒度是256B，vadd最大repeat次数为255，两个相乘的结果
constexpr int64_t MAX_VADD_SIZE = VADD_MAX_REPEAT * VADD_UNIT_BYTE;
constexpr int64_t BLOCK_UNIT_BYTE = 32;
constexpr int64_t VADD_UNIT_TO_BLOCK_UNIT_RATIO = VADD_UNIT_BYTE / BLOCK_UNIT_BYTE;  // 8

constexpr int32_t LCCL_DUMP_UINT_SIZE = 1 * 1024 * 1024;

enum Op : int { COPYONLY = -1, ADD = 0, MUL = 1, MAX = 2, MIN = 3 };

struct ExtraFlag {
    static constexpr uint32_t RDMA = 1;
    static constexpr uint32_t TOPO_910B2C = 1 << 1;
    static constexpr uint32_t TOPO_910_93 = 1 << 2;
    static constexpr uint32_t DETERMINISTIC = 1 << 3;
    static constexpr uint32_t QUANT_FP16 = 1 << 4;
    static constexpr uint32_t QUANT_FP32 = 1 << 5;
    static constexpr uint32_t TOPO_910A5 = 1 << 6;
    static constexpr uint32_t QUANT_DELAY = 1 << 7;
    static constexpr uint32_t QUANT_CURRENT = 1 << 8;
    static constexpr uint32_t TOPO_PCIE = 1 << 9;
    static constexpr uint32_t UDMA = 1 << 10;
    static constexpr uint32_t SDMA = 1 << 11;
    static constexpr uint32_t UDMA_SHARED_QP = 1 << 12;
    static constexpr uint32_t UDMA_FULLMESH = 1 << 13;
    static constexpr uint32_t ATOMIC_ENABLE = 1 << 15;  // 表示在910A5算子中启用atomic实现
    static constexpr uint32_t IS_GREATER_THAN_40_AIV = 1 << 16;
    static constexpr uint32_t MEMORY_ONLY = 1 << 18;
};

struct CommArgs {
    int rank = 0;  // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0;        // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0;  // 32 bit map，具体每一位的含义就在此文件正上方
    GM_ADDR peerMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // 传入初始化获得的buff，所有allreduce都是同一个参数
    GM_ADDR creditMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};
    /**
     * @param sendCountMatrix 大小是rankSize*rankSize的一维数组
     * eg: sendCountMatrix[1] 的数值，对应二维数组的[0][1]，表示 卡0 要给 卡1 发送的数据个数
     */
    int64_t sendCountMatrix[DISPATCH_DEDUP_MAX_RANK_SIZE * DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // for all2allv
    int64_t dfx[DFX_COUNT] = {};
    GM_ADDR dumpAddr = nullptr;
    int32_t magics[DISPATCH_DEDUP_MAX_RANK_SIZE] = {0};
    uint64_t fftsVal = 0;
    GM_ADDR udmaInfoPtr = nullptr;       // device-side DispatchDedup::UDMAInfo*; nullptr 表示 UDMA 不可用
    GM_ADDR udmaRegistryPtr = nullptr;   // device-side DispatchDedupUDMARegistry* for user-registered UDMA memory
    GM_ADDR sdmaWorkspacePtr = nullptr;  // device-side SDMA workspace; nullptr 表示 SDMA 不可用
    GM_ADDR udmaFullmeshPtr = nullptr;   // device-side DispatchDedupUDMAFullmeshDeviceView*
    uint64_t udmaRegistrationGeneration = 0U;
    uint64_t peerMemBytes = 0U;
    int32_t commDomain = 0;
    uint32_t reserved0 = 0U;
    // Unused ABI tail; kept zero by the Host.
    GM_ADDR reservedAddresses[4] = {};
    uint32_t reservedConfig[2] = {};
};

static_assert(sizeof(CommArgs) == 134168U, "communication ABI size unchanged");
static_assert(offsetof(CommArgs, reservedAddresses) == 134128U, "reserved ABI tail offset");

struct LcclDumpBlockInfo {
    uint32_t len = 0;
    uint32_t core = 0;        // current core id
    uint32_t blockNum = 0;    // total core num
    uint32_t dumpOffset = 0;  // size used by current core
    uint32_t magic = 0;       // magic number
    uint32_t rsv = 0;
    uint64_t dumpAddr = 0;  // start addr of dump
};

struct LcclDumpLogInfo {
    uint32_t logId = 0;
    uint32_t blockId = 0;
    uint64_t syscyc = 0;
    uint64_t curPc = 0;
    uint32_t operationType = 0;
    uint32_t rsv = 0;
};

// 定义联合体
union LcclDumpUnion {
    LcclDumpBlockInfo blockInfo;
    LcclDumpLogInfo logInfo;
};

enum LogId : int { OVERALL = 0, INIT, PROCESS };

}  // namespace DispatchDedup
#endif  // DISPATCH_DEDUP_COMM_ARGS_H
