// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

/*!
 * \file tiling_impl.hpp
 * \brief Host-side tiling and buffer size calculation for MegaMoE BF16 training.
 */

#include "csrc/kernels/bf16/layout/mega_moe_layout.hpp"
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <map>
#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "mega_moe_quant_constants.hpp"

template <typename T, typename U>
constexpr auto HostCeilDiv(T value, U divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T, typename U>
constexpr auto HostCeilAlign(T value, U alignment)
{
    return HostCeilDiv(value, alignment) * alignment;
}

constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int64_t PEERMEM_DATA_OFFSET = 60 * 1024;
constexpr int64_t INT_CACHELINE = 16;
constexpr int64_t SWIGLU_N_HALF = 2;
constexpr int64_t SIZE_INT_8 = 1;
constexpr int64_t SIZE_BF_16 = 2;
constexpr int64_t SIZE_INT_32 = 4;

using namespace AscendC;

void OpLogDImpl(const char* nodeName, const char* fmt, ...)
{
    constexpr size_t BUF_SIZE = 4096;
    char buf[BUF_SIZE];

    va_list args;
    va_start(args, fmt);

    vsnprintf(buf, BUF_SIZE, fmt, args);

    va_end(args);

    printf("[DEBUG][%s] %s\n", nodeName, buf);
}

#define OP_LOGD(nodeName, fmt, ...) OpLogDImpl(nodeName, fmt, ##__VA_ARGS__)

namespace optiling {

namespace {
// init routing
const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;
const static int64_t SORT_API_MAX_ELEM = 32 * 255LL;
const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t MRG_LIST_NUM = 4LL;
const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
const static int64_t UB_BLOCK_SIZE = 32LL;
const static size_t DIM_TWO = 2ULL;
const static int64_t KV_FACTOR = 2LL;
const static int64_t EXPERT_IDX_MAX = 10240LL;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

const static int64_t ROW_IDX_GATHER = 0LL;
const static int64_t ROW_IDX_SCATTER = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;
const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_BS = 1LL;
const static int64_t MAX_BS = 16384LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 16LL;
const static int64_t H_BASE = 1024LL;
const static int64_t N_BASE = 1024LL;
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_EP_WORLD_SIZE = 768LL;
const static int64_t MAX_MOE_EXPERT_NUM = 1024LL;
const static int64_t DISABLE_EXPERT_CAPACITY = -1LL;
const static int64_t DISPATCH_QUANT_OUT_TYPE_OFFSET = 20LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
}  // namespace

struct MegaMoeAuxTilingContext {
    // Platform
    int64_t aivCoreNum;
    int64_t totalUbSize;
    int64_t availUbSize;

    // Critical Variable
    int64_t sortLoopMaxElement;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols;
    int64_t inputXDtypeSize;
    int64_t isInputScale;
    int64_t isInputOffset;
    int64_t sortMode;

    // Op Attr
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensNumType;
    int64_t quantMode;
    int64_t expertStart;
    int64_t expertEnd;
    int64_t rowIdxType;

    bool expertTokensNumFlag;
};

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 1) return 0;
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

void PrintMoeV3Arch35VBSComputeTilingData(const MoeV3Arch35VBSComputeTilingData& data, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VBSComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35VMSMiddleComputeTilingData(const MoeV3Arch35VMSMiddleComputeTilingData& data, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VMSMiddleComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
}

void PrintMoeV3Arch35SortOutComputeTilingData(const MoeV3Arch35SortOutComputeTilingData& data, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35SortOutComputeTilingData ==========");
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35ExpertTokensCountTilingData(const MoeV3Arch35ExpertTokensCountTilingData& data,
                                                 const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35ExpertTokensCountTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
}

void PrintMoeV3Arch35GatherOutComputeTilingData(const MoeV3Arch35GatherOutComputeTilingData& data, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35GatherOutComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreIndicesElements is %ld", data.perCoreIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesElements is %ld", data.lastCoreIndicesElements);
    OP_LOGD(nodeName, "perCoreIndicesLoops is %ld", data.perCoreIndicesLoops);
    OP_LOGD(nodeName, "perCorePerLoopIndicesElements is %ld", data.perCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "perCoreLastLoopIndicesElements is %ld", data.perCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesLoops is %ld", data.lastCoreIndicesLoops);
    OP_LOGD(nodeName, "lastCorePerLoopIndicesElements is %ld", data.lastCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreLastLoopIndicesElements is %ld", data.lastCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "colsLoops is %ld", data.colsLoops);
    OP_LOGD(nodeName, "perLoopCols is %ld", data.perLoopCols);
    OP_LOGD(nodeName, "lastLoopCols is %ld", data.lastLoopCols);
}

void PrintMoeInitRoutingV3Arch35TilingData(const MoeInitRoutingV3Arch35TilingData& data, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MoeInitRoutingV3Arch35TilingData ==========");
    OP_LOGD(nodeName, "coreNum is %ld", data.coreNum);
    OP_LOGD(nodeName, "BS is %ld", data.n);
    OP_LOGD(nodeName, "H is %ld", data.cols);
    OP_LOGD(nodeName, "topK is %ld", data.k);
    OP_LOGD(nodeName, "expertStart is %ld", data.expertStart);
    OP_LOGD(nodeName, "expertEnd is %ld", data.expertEnd);
    OP_LOGD(nodeName, "actualExpertNum is %ld", data.actualExpertNum);
    OP_LOGD(nodeName, "quantMode is %ld", data.quantMode);
    OP_LOGD(nodeName, "rowIdxType is %ld", data.rowIdxType);
    OP_LOGD(nodeName, "isInputScale is %s", data.isInputScale ? "True" : "False");
    OP_LOGD(nodeName, "isInputOffset is %s", data.isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "expertNum is %ld", data.expertNum);
    OP_LOGD(nodeName, "expertTokensNumType is %ld", data.expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %ld", data.expertTokensNumFlag);
    OP_LOGD(nodeName, "gatherFirstFullload is %ld", data.gatherFirstFullload);
    OP_LOGD(nodeName, "epFullload is %ld", data.epFullload);
    OP_LOGD(nodeName, "activeNum is %ld", data.activeNum);
    OP_LOGD(nodeName, "dropPadMode is %ld", data.dropPadMode);
    OP_LOGD(nodeName, "smoothType is %ld", data.smoothType);
    OP_LOGD(nodeName, "InitRouting TilingKey is %ld", data.tilingKey);

    PrintMoeV3Arch35VBSComputeTilingData(data.vbsComputeParamsOp, nodeName);
    PrintMoeV3Arch35VMSMiddleComputeTilingData(data.vmsMiddleComputeParamsOp, nodeName);
    PrintMoeV3Arch35SortOutComputeTilingData(data.sortOutComputeParamsOp, nodeName);
    PrintMoeV3Arch35ExpertTokensCountTilingData(data.expertTokensCountTilingDataOp, nodeName);
    PrintMoeV3Arch35GatherOutComputeTilingData(data.gatherOutComputeParamsOp, nodeName);
}

void PrintMegaMoeTilingData(const MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");

    PrintMoeInitRoutingV3Arch35TilingData(tilingData->moeInitRoutingTilingData, nodeName);

    OP_LOGD(nodeName, "BS is %u", tilingData->m);
    OP_LOGD(nodeName, "H is %u", tilingData->k);
    OP_LOGD(nodeName, "N is %u", tilingData->n);

    OP_LOGD(nodeName, "topK is %u", tilingData->topK);
    OP_LOGD(nodeName, "expertPerRank is %u", tilingData->expertPerRank);
    OP_LOGD(nodeName, "groupListType is %u", tilingData->groupListType);

    OP_LOGD(nodeName, "epWorldSize is %u", tilingData->epWorldSize);
    OP_LOGD(nodeName, "maxOutputSize is %u", tilingData->maxOutputSize);

    OP_LOGD(nodeName, "transX is %s", (tilingData->transX ? "True" : "False"));
    OP_LOGD(nodeName, "transW is %s", (tilingData->transW ? "True" : "False"));
    OP_LOGD(nodeName, "transW2 is %s", (tilingData->transW2 ? "True" : "False"));
}

void printAuxiliaryTilingCtx(const MegaMoeAuxTilingContext* ctx, const uint32_t expertPerRank,
                             const uint32_t epWorldSize, const std::vector<int64_t>& expertIdxShape,
                             const char* nodeName)
{
    OP_LOGD(nodeName, "========== AuxiliaryTilingCtx ==========");
    OP_LOGD(nodeName, "BS is %ld", ctx->n);
    OP_LOGD(nodeName, "H is %ld", ctx->cols);
    OP_LOGD(nodeName, "topK is %ld", ctx->k);
    OP_LOGD(nodeName, "expertPerRank is %ld", expertPerRank);
    OP_LOGD(nodeName, "epWorldSize is %ld", epWorldSize);

    OP_LOGD(nodeName, "availUbSize is %ld", ctx->availUbSize);
    OP_LOGD(nodeName, "sortLoopMaxElement is %ld", ctx->sortLoopMaxElement);
    OP_LOGD(nodeName, "totalLength is %ld", ctx->totalLength);
    OP_LOGD(nodeName, "activeNum is %ld", ctx->activeNum);
    OP_LOGD(nodeName, "expertCapacity is %ld", ctx->expertCapacity);
    OP_LOGD(nodeName, "expertNum is %ld", ctx->expertNum);

    OP_LOGD(nodeName, "isInputScale is %s, isInputOffset is %s", ctx->isInputScale ? "True" : "False",
            ctx->isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "dropPadMode is %ld, expertTokensNumType is %ld", ctx->dropPadMode, ctx->expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %s, quantMode is %ld", ctx->expertTokensNumFlag ? "True" : "False",
            ctx->quantMode);

    OP_LOGD(nodeName, "expertIdxShape is [%ld, %ld]", expertIdxShape[0], expertIdxShape[1]);
    OP_LOGD(nodeName, "expertRange is [%ld, %ld]", ctx->expertStart, ctx->expertEnd);
    OP_LOGD(nodeName, "rowIdxType is %ld", ctx->rowIdxType);
}

static int64_t OpQuantModeToInitRoutingQuantMode(const int64_t opQuantMode)
{
    switch (opQuantMode) {
        case DISPATCH_QUANT_MODE_NO_QUANT:
            return 0LL;
        case DISPATCH_QUANT_OUT_TYPE_E5M2:
            return 2LL;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN:
            return 3LL;
        default:
            return -1LL;
    }
    return -1LL;
}

template <typename xDtype>
static bool SetAuxiliaryTilingCtx(MoeInitRoutingV3Arch35TilingData& initTilingData, MegaMoeAuxTilingContext& ctx,
                                  const uint32_t aivNum, const uint64_t ubSize, const char* nodeName)
{
    int64_t inputXDtypeSize = sizeof(xDtype);

    ctx.aivCoreNum = aivNum;
    ctx.totalUbSize = ubSize;
    ctx.availUbSize = ubSize - SIMT_DCACHE_SIZE;

    ctx.n = initTilingData.n;
    ctx.cols = initTilingData.cols;
    ctx.k = initTilingData.k;
    ctx.inputXDtypeSize = inputXDtypeSize;
    ctx.isInputScale = initTilingData.isInputScale;
    ctx.isInputOffset = initTilingData.isInputOffset;

    ctx.activeNum = initTilingData.activeNum;
    ctx.expertCapacity = DISABLE_EXPERT_CAPACITY;
    ctx.expertNum = initTilingData.expertNum;
    ctx.dropPadMode = initTilingData.dropPadMode;
    ctx.expertTokensNumType = initTilingData.expertTokensNumType;
    ctx.expertTokensNumFlag = initTilingData.expertTokensNumFlag;
    ctx.quantMode = initTilingData.quantMode;

    ctx.expertStart = initTilingData.expertStart;
    ctx.expertEnd = initTilingData.expertEnd;
    ctx.rowIdxType = initTilingData.rowIdxType;
    ctx.totalLength = initTilingData.n * initTilingData.k;

    ctx.sortLoopMaxElement =
        ctx.availUbSize / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    ctx.sortLoopMaxElement = std::min(ctx.sortLoopMaxElement, SORT_API_MAX_ELEM);

    return true;
}

static void Tiling4VBSOneCore(MoeV3Arch35VBSComputeTilingData* vbsTiling, const MegaMoeAuxTilingContext& ctx)
{
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = ctx.totalLength;
    vbsTiling->perCoreLoops = HostCeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

static void Tiling4VBSMultiCore(MoeV3Arch35VBSComputeTilingData* vbsTiling, MegaMoeAuxTilingContext& ctx)
{
    int64_t needCoreNum = HostCeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    needCoreNum = std::min(needCoreNum, ctx.aivCoreNum);

    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (ctx.totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = ctx.totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = HostCeilDiv(ctx.totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = HostCeilDiv(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = ctx.totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            HostCeilAlign(static_cast<int64_t>(HostCeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops)),
                          SORT32_ALIGN_ELEMENT);
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
}

static void Tiling4VBSCompute(MoeInitRoutingV3Arch35TilingData* tilingData, MegaMoeAuxTilingContext& ctx)
{
    auto* vbsTiling = &(tilingData->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = ctx.sortLoopMaxElement;

    if (ctx.totalLength <= ctx.sortLoopMaxElement) {
        ctx.sortMode = 0;
        Tiling4VBSOneCore(vbsTiling, ctx);
    } else {
        ctx.sortMode = 1;
        Tiling4VBSMultiCore(vbsTiling, ctx);
    }
    // Keep the existing sort/scatter pipeline, but select the plain gather
    // key for the BF16/no-quant path.
    const int64_t gatherKey = ctx.quantMode == 0 ? 1001000 : 1031000;
    tilingData->tilingKey = ctx.sortMode * SORT_CORE_TILINGKEY_BASE + gatherKey;
}

static void Tiling4VMSMiddleCompute(MoeInitRoutingV3Arch35TilingData* tilingData)
{
    auto* vbsTiling = &(tilingData->vbsComputeParamsOp);
    auto* vmsMiddleTiling = &(tilingData->vmsMiddleComputeParamsOp);

    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) {
        vmsMiddleTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = HostCeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum;
}

static void Tiling4SortOutCompute(MoeInitRoutingV3Arch35TilingData* tilingData)
{
    auto* sortOutTiling = &(tilingData->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

static void Tiling4ExpertTokensCountCompute(MoeInitRoutingV3Arch35TilingData* tilingData,
                                            const MegaMoeAuxTilingContext& ctx)
{
    auto* tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = HostCeilDiv(totalElements, ctx.aivCoreNum);
    int64_t needCoreNum = HostCeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;

    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingData->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE)
                                   ? tilingData->actualExpertNum
                                   : (tilingData->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (ctx.availUbSize -
         HostCeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));

    int64_t perCoreLoops = HostCeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = HostCeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = HostCeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = HostCeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

static int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols, int64_t inputXDtypeSize, int64_t availUbSize)
{
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize) + AlignBytes(perLoopCols, sizeof(int8_t));
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    return (availUbSize - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

static void Tiling4GatherOutMxQuant(MoeInitRoutingV3Arch35TilingData* tilingData, const MegaMoeAuxTilingContext& ctx)
{
    auto* gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = HostCeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = HostCeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = HostCeilAlign(tilingData->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = HostCeilAlign(static_cast<int64_t>(HostCeilDiv(perLoopCols, NUM_TWO)), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    }
    int64_t colsLoops = HostCeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = HostCeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = HostCeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

void ComputeMoeInitRoutingV3Tiling(MoeInitRoutingV3Arch35TilingData& tilingData, MegaMoeAuxTilingContext& ctx)
{
    Tiling4VBSCompute(&tilingData, ctx);
    Tiling4VMSMiddleCompute(&tilingData);
    Tiling4SortOutCompute(&tilingData);
    Tiling4ExpertTokensCountCompute(&tilingData, ctx);
    Tiling4GatherOutMxQuant(&tilingData, ctx);  // support fp8_e4m3fn, fp8_e5m2 currently
    return;
}

template <typename xDtype, int QUANT_TYPE>
static bool SetInitTilingData(MegaMoeTilingData& td, uint32_t m, uint32_t k, uint32_t n, uint32_t topK,
                              uint32_t epWorldSize, uint32_t expertPerRank, uint32_t blockNumPerEP,
                              uint32_t maxOutputSize, uint32_t rankId, int32_t dispatchRows, uint8_t groupListType,
                              uint32_t aivNum, uint64_t ubSize)
{
    td.m = m;
    td.k = k;  // hiddensize
    td.n = n;  // moe_mid_hiddensize * 2
    td.topK = topK;
    td.epWorldSize = epWorldSize;
    td.expertPerRank = expertPerRank;
    td.blockNumPerEP = blockNumPerEP;
    td.maxOutputSize = maxOutputSize;
    td.rankId = rankId;
    td.dispatchRows = dispatchRows;
    td.groupListType = groupListType;
    td.transX = false;
    td.transW = true;
    td.transW2 = true;
    td.swigluLimit = kMegaMoeSwigluLimitDisabled;
    td.savedGateUpMode = 2;  // BF16 saving defaults to raw BF16.

    int64_t expertNum = epWorldSize * expertPerRank;
    td.moeInitRoutingTilingData.coreNum = aivNum;
    td.moeInitRoutingTilingData.n = m;                                                      //
    td.moeInitRoutingTilingData.cols = k;                                                   //
    td.moeInitRoutingTilingData.k = topK;                                                   //
    td.moeInitRoutingTilingData.expertStart = 0;                                            //
    td.moeInitRoutingTilingData.expertEnd = expertNum;                                      //
    td.moeInitRoutingTilingData.actualExpertNum = expertNum;                                //
    td.moeInitRoutingTilingData.quantMode = OpQuantModeToInitRoutingQuantMode(QUANT_TYPE);  // E4M3:3 E5M2:2
    td.moeInitRoutingTilingData.rowIdxType = ROW_IDX_SCATTER;                               //
    td.moeInitRoutingTilingData.isInputScale = 0;                                           //
    td.moeInitRoutingTilingData.isInputOffset = 0;                                          //
    td.moeInitRoutingTilingData.expertNum = expertNum;                                      //
    td.moeInitRoutingTilingData.expertTokensNumType = EXPERT_TOKENS_TYPE_COUNT;             //
    td.moeInitRoutingTilingData.expertTokensNumFlag = 1;                                    //
    td.moeInitRoutingTilingData.gatherFirstFullload = 0;
    td.moeInitRoutingTilingData.epFullload = 0;                        // unsupported
    td.moeInitRoutingTilingData.activeNum = m * topK;                  //
    td.moeInitRoutingTilingData.dropPadMode = DROP_PAD_MODE_DROPLESS;  //
    td.moeInitRoutingTilingData.smoothType = 0;                        // unsupported

    std::string nodeName = "rank:" + std::to_string(rankId);
    MegaMoeAuxTilingContext ctx;
    SetAuxiliaryTilingCtx<xDtype>(td.moeInitRoutingTilingData, ctx, aivNum, ubSize, nodeName.c_str());
    ComputeMoeInitRoutingV3Tiling(td.moeInitRoutingTilingData, ctx);

    return true;
}

// 调试
enum WorkspaceIdx {
    PTR_A2 = 0,
    PTR_A2_SCALE,
    COMM0,
    COMM1,
    COMM2,
    COMM3,
    COMM4,
    EXPANDED_ROW_IDX,
    EXPANDED_ROW_IDX_GATHER,
    FLAG_SWIGLU_TO_GMM2,

    WORKSPACE_NUM
};

struct WorkspaceInfo {
    size_t totalSize = 0;
    size_t offset[WORKSPACE_NUM] = {0};
};

WorkspaceInfo CalcWorkspaceSize(const MegaMoeTilingData& td)
{
    WorkspaceInfo ws;
    const bool isBf16 = td.moeInitRoutingTilingData.quantMode == 0;
    const int64_t elementBytes = isBf16 ? SIZE_BF_16 : SIZE_INT_8;
    ws.totalSize += HostCeilAlign(elementBytes * static_cast<int64_t>(td.maxOutputSize) * td.k, ALIGN_512);
    if (!isBf16) {
        ws.totalSize += HostCeilAlign(static_cast<int64_t>(td.maxOutputSize) * td.k / MXFP_SCALE_GROUP_NUM, ALIGN_512);
    }
    ws.totalSize +=
        HostCeilAlign(elementBytes * static_cast<int64_t>(td.maxOutputSize) * td.n / SWIGLU_N_HALF, ALIGN_512);
    if (!isBf16) {
        ws.totalSize += HostCeilAlign(
            static_cast<int64_t>(td.maxOutputSize) * td.n / SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);
    }
    ws.totalSize += HostCeilAlign(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank * SIZE_INT_32, ALIGN_512);
    ws.totalSize += SIZE_INT_32 * HostCeilAlign(static_cast<int64_t>(td.m), ALIGN_256) * td.topK * 2;
    ws.totalSize += SIZE_INT_32 * HostCeilAlign(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128);
    ws.totalSize += 2 * HostCeilAlign(static_cast<int64_t>(SIZE_INT_32) * td.expertPerRank * INT_CACHELINE, ALIGN_128);
    return ws;
}

inline size_t CalcBf16EarlyTopkWeightWorkspaceSize(const MegaMoeTilingData& td)
{
    return HostCeilAlign(
        static_cast<int64_t>(td.maxOutputSize) * MegaMoeBF16Training::TopkWeightBytes(td.topkWeightType), ALIGN_512);
}

enum PeerMemIdx {
    PEER_TOKEN_PER_EXPERT = 0,
    PEER_COMM_FLAG,
    PEER_PTR_A0,
    PEER_PTR_A1,
    PEER_PTR_D,

    PEER_MEM_NUM
};

struct PeerMemInfo {
    size_t totalSize = 0;
    size_t offset[PEER_MEM_NUM] = {0};
};

PeerMemInfo CalcPeermemSize(const MegaMoeTilingData& td)
{
    PeerMemInfo pm;
    // Keep this host calculation consistent with device-side PeermemInfo.
    pm.totalSize = PEERMEM_DATA_OFFSET;
    pm.totalSize += td.epWorldSize * HostCeilAlign(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128) *
                    SIZE_INT_32;
    const bool isBf16 = td.moeInitRoutingTilingData.quantMode == 0;
    if (isBf16) {
        pm.totalSize += static_cast<int64_t>(td.m) * td.topK * td.k * SIZE_BF_16;
    } else {
        pm.totalSize += static_cast<int64_t>(td.m) * td.topK * (td.k + td.k / MXFP_SCALE_GROUP_NUM);
    }
    pm.totalSize += static_cast<int64_t>(td.m) * td.topK * td.k * SIZE_BF_16;

    return pm;
}

inline size_t CalcBf16EarlyTopkWeightPeermemSize(const MegaMoeTilingData& td)
{
    return HostCeilAlign(static_cast<int64_t>(td.m) * td.topK * MegaMoeBF16Training::TopkWeightBytes(td.topkWeightType),
                         ALIGN_512);
}

template <typename T>
void PrintBuffer(void* buf, size_t rows, size_t cols, size_t stride, size_t offset = 0, size_t rankId = 0)
{
    if (buf == nullptr) {
        std::cout << "buffer is nullptr" << std::endl;
        return;
    }

    T* data = reinterpret_cast<T*>(static_cast<uint8_t*>(buf) + offset);

    size_t printRows = rows;

    std::cout << "===== Buffer Dump =====\n"
              << "rows=" << rows << ", cols=" << cols << ", stride=" << stride << ", offset=" << offset << "\n";

    for (size_t r = 0; r < printRows; ++r) {
        std::cout << "[rankId:" << rankId << "] " << "[" << r << "] ";

        for (size_t c = 0; c < cols; ++c) {
            if constexpr (std::is_integral_v<T>) {
                std::cout << std::setw(6) << static_cast<int32_t>(data[r * stride + c]) << " ";
            } else {
                std::cout << std::setw(6) << static_cast<float>(data[r * stride + c]) << " ";
            }
        }

        std::cout << "\n";
    }

    std::cout << "=======================" << std::endl;
}

template <typename T>
bool CompareTensor(const T* output, const T* golden, size_t tokenNum, size_t hiddenSize, float atol = 1e-3,
                   float rtol = 0.1f,
                   size_t maxTokenPrint = 100,  // 最多打印多少个错误token
                   size_t maxDetailPrint = 10,  // 每个token最多打印几个错误位置
                   bool printDetail = false)    // 是否打印具体错误值
{
    struct ErrorInfo {
        size_t pos;
        double golden;
        double output;
        double absErr;
        double relErr;
    };

    std::map<size_t, std::vector<ErrorInfo>> tokenErrors;

    size_t totalErrorCount = 0;

    // 收集错误
    for (size_t token = 0; token < tokenNum; token++) {
        for (size_t pos = 0; pos < hiddenSize; pos++) {
            size_t idx = token * hiddenSize + pos;

            double outVal = static_cast<double>(output[idx]);
            double goldVal = static_cast<double>(golden[idx]);

            double absErr = std::abs(outVal - goldVal);

            double relErr = absErr / (std::abs(goldVal) + 1e-12);

            bool pass;
            if constexpr (std::is_integral_v<T>) {
                // 整型：允许 ±1
                pass = (absErr <= 1.0);
            } else {
                // 浮点：原逻辑
                pass = (absErr <= atol) || (relErr <= rtol);
            }

            if (!pass) {
                totalErrorCount++;

                tokenErrors[token].push_back({pos, goldVal, outVal, absErr, relErr});
            }
        }
    }

    std::cout << "\n========== Compare Summary ==========\n";

    std::cout << "Total Error Count: " << totalErrorCount << "/" << tokenNum * hiddenSize << "\n\n";

    if (tokenErrors.empty()) {
        std::cout << "PASS\n";
        return true;
    }

    // 第一阶段：打印token错误统计
    std::cout << "Token Error Summary:\n";

    size_t tokenPrintCount = 0;

    for (auto& [token, errors] : tokenErrors) {
        if (tokenPrintCount >= maxTokenPrint) break;

        std::cout << "Token " << token << " : " << errors.size() << " errors\n";

        tokenPrintCount++;
    }

    // 第二阶段：打印详细错误
    if (printDetail) {
        std::cout << "\n========== Error Detail ==========\n";

        tokenPrintCount = 0;

        for (auto& [token, errors] : tokenErrors) {
            if (tokenPrintCount >= maxTokenPrint) break;

            std::cout << "\nToken " << token << ":\n";

            size_t detailCount = std::min(errors.size(), maxDetailPrint);

            for (size_t i = 0; i < detailCount; i++) {
                auto& e = errors[i];

                std::cout << "[ERROR]"
                          << " pos=" << e.pos << " golden=" << std::setprecision(10) << e.golden
                          << " output=" << e.output << " abs_err=" << e.absErr << " rel_err=" << e.relErr << "\n";
            }

            if (errors.size() > maxDetailPrint) {
                std::cout << "... " << errors.size() - maxDetailPrint << " more errors\n";
            }

            tokenPrintCount++;
        }
    }

    return false;
}

}  // namespace optiling
