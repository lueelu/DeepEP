// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#include "kernel_operator.h"
#include "simt_api/device_functions.h"
#include "shmem.h"
#include "../common.hpp"
#include "launch.hpp"
#include "tiling.hpp"
#include "../dispatch/tiling.hpp"
#include "../dispatch/peer_order.hpp"

using namespace AscendC;
namespace {
#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 3510
constexpr uint32_t kSimtThreads = 128U;
constexpr uint32_t kSimtCores = 32U;
constexpr uint32_t kTileTokens = ascend_deepep::kNotifyLocalTopkTileTokens;
constexpr uint32_t kMetaFields = ascend_deepep::kNotifyLocalTopkRouteMetaFields;
// Wire fields pack rank <= 255 and top-k positions <= 15.
// Forward column 0 stores token_end; source rank is the partition index.
// Weight return IDs encode (src_rank * tokens + token_id) * topk + topk_id.
constexpr uint32_t kWireRankMask = 0xffU;
constexpr uint32_t kWireTopkMask = 0xfU;
constexpr uint32_t kWirePrimaryTopkShift = 8U;
constexpr uint32_t kWireSecondaryTopkShift = 12U;

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void ParseLocalTopkTokensVf(
    __gm__ ascend_deepep::topk_idx_t* topk_idx, __gm__ int64_t* server_mask, __gm__ int32_t* route_meta, uint32_t world,
    uint32_t tokens, uint32_t topk, uint32_t experts, uint32_t chunk_tokens, uint32_t core, __ubuf__ int32_t* routes_ub)
{
    const uint32_t local_experts = experts / world;
    const uint32_t tiles = (uint64_t(tokens) + kTileTokens - 1U) / kTileTokens;
    const uint32_t local_tiles = core < tiles ? (tiles - core + CoreCount - 1U) / CoreCount : 0U;
    // 每个线程独占一个 token，沿用原 tile/core 归属，方便后续本核有序计数。
    for (uint32_t local_token = threadIdx.x; local_token < local_tiles * kTileTokens; local_token += kSimtThreads) {
        const uint32_t token = (core + local_token / kTileTokens * CoreCount) * kTileTokens + local_token % kTileTokens;
        if (token >= tokens) {
            continue;
        }
        int32_t expert_ids[16];
        int32_t first_slot[32];
        for (uint32_t server = 0; server < 32U; ++server) {
            first_slot[server] = -1;
        }
        const uint64_t base = uint64_t(token) * topk;
        int64_t mask = 0;
        for (uint32_t k = 0; k < topk; ++k) {
            const int64_t value = asc_ldcg(topk_idx + base + k);
            const int32_t expert = value >= 0 && uint64_t(value) < experts ? int32_t(value) : -1;
            expert_ids[k] = expert;
            if (expert >= 0) {
                const uint32_t server = uint32_t(expert) / local_experts / 8U;
                mask |= int64_t(1) << server;
                if (first_slot[server] < 0) {
                    first_slot[server] = int32_t(k);
                }
            }
        }
        asc_stcg(server_mask + token, mask);
        for (uint32_t k = 0; k < topk; ++k) {
            const int32_t expert = expert_ids[k];
            int32_t primary = -1;
            int32_t gateway = -1;
            int32_t flags = 0;
            if (expert >= 0) {
                primary = first_slot[uint32_t(expert) / local_experts / 8U];
                gateway = expert_ids[primary] / int32_t(local_experts);
                flags = int32_t(k) == primary ? 7 : 9;
            }
            auto* cached = routes_ub + (uint64_t(local_token) * topk + k) * 2U;
            cached[0] = expert;
            cached[1] = expert >= 0 ? gateway * 16 + primary : -1;
            auto* row = route_meta + (base + k) * kMetaFields;
            asc_stcg(row, expert);
            asc_stcg(row + 1U, gateway);
            asc_stcg(row + 2U, primary);
            asc_stcg(row + 3U, int32_t(token / chunk_tokens));
            // 索引 4 的 ordinal 留给下一步：不能用并行原子分配改变接收顺序。
            asc_stcg(row + 5U, flags);
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void CountLocalTopkTilesVf(__gm__ int32_t* route_meta, uint32_t world,
                                                                     uint32_t tokens, uint32_t topk, uint32_t experts,
                                                                     uint32_t core, __ubuf__ int32_t* hist_ub,
                                                                     __ubuf__ int32_t* routes_ub)
{
    const uint32_t tiles = (uint64_t(tokens) + kTileTokens - 1U) / kTileTokens;
    const uint32_t hist_stride = (experts + world + 15U) / 16U * 16U;
    // 每个 tile 唯一写者，严格按 token -> slot 累加，重复专家的槽位也分别计数。
    for (uint32_t tile = threadIdx.x * CoreCount + core; tile < tiles; tile += (CoreCount * kSimtThreads)) {
        auto* counts = hist_ub + uint64_t(tile / CoreCount) * hist_stride;
        const uint32_t begin = tile * kTileTokens;
        const uint32_t count = tokens - begin < kTileTokens ? tokens - begin : kTileTokens;
        auto* cached = routes_ub + uint64_t(tile / CoreCount) * kTileTokens * topk * 2U;
        for (uint32_t route = 0; route < count * topk; ++route) {
            const int32_t expert = cached[route * 2U];
            int32_t ordinal = -1;
            if (expert >= 0) {
                ordinal = counts[expert];
                counts[expert] = ordinal + 1;
                const uint32_t packed = uint32_t(cached[route * 2U + 1U]);
                if (route % topk == packed % 16U) {
                    ++counts[experts + packed / 16U];
                }
            }
            asc_stcg(route_meta + (uint64_t(begin) * topk + route) * kMetaFields + 4U, ordinal);
        }
    }
    asc_threadfence();
}

template <bool CachedHist = false, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void ParseLocalTopkVf(__gm__ ascend_deepep::topk_idx_t* topk_idx,
                                                                __gm__ int64_t* server_mask, __gm__ int32_t* route_meta,
                                                                __gm__ int32_t* hist, uint32_t world, uint32_t tokens,
                                                                uint32_t topk, uint32_t experts, uint32_t chunk_tokens,
                                                                uint32_t core, __ubuf__ int32_t* hist_ub = nullptr)
{
    const uint32_t local_experts = experts / world;
    const uint32_t columns = experts + world;
    const uint32_t hist_stride = (columns + 15U) / 16U * 16U;
    const uint32_t tiles = (tokens + kTileTokens - 1U) / kTileTokens;
    // tile 按线程与核交错分配；同一 tile 的计数只有一个线程更新，无需原子操作。
    for (uint32_t tile = threadIdx.x * CoreCount + core; tile < tiles; tile += (CoreCount * kSimtThreads)) {
        auto* counts = hist + uint64_t(tile) * hist_stride;
        // 同核 tile 按 core、core+CoreCount... 排列，各线程使用互不重叠的 UB 行。
        __ubuf__ int32_t* local_counts = hist_ub;
        if constexpr (CachedHist) {
            local_counts += uint64_t(tile / CoreCount) * hist_stride;
        }
        // UB 路径由调用方先用 SIMD Duplicate 清零，各线程只更新自己的计数行。
        if constexpr (!CachedHist) {
            for (uint32_t col = 0; col < hist_stride; ++col) {
                asc_stcg(counts + col, int32_t(0));
            }
        }
        const uint32_t begin = tile * kTileTokens;
        const uint32_t end = tokens - begin < kTileTokens ? tokens : begin + kTileTokens;
        for (uint32_t token = begin; token < end; ++token) {
            int32_t expert_ids[16];
            int32_t first_slot_per_server[32];
            for (uint32_t server = 0; server < 32U; ++server) {
                first_slot_per_server[server] = -1;
            }
            const uint64_t base = uint64_t(token) * topk;
            int64_t mask = 0;
            // 按原始 top-k 顺序选 gateway：每个 server 第一次出现的专家所在 rank。
            // 非法专家跳过；server_mask 的第 s 位表示该 token 是否需要发往 server s。
            for (uint32_t k = 0; k < topk; ++k) {
                const int64_t value = asc_ldcg(topk_idx + base + k);
                const int32_t expert = value >= 0 && uint64_t(value) < experts ? int32_t(value) : -1;
                expert_ids[k] = expert;
                if (expert < 0) {
                    continue;
                }
                const uint32_t server = uint32_t(expert) / local_experts / 8U;
                mask |= int64_t(1) << server;
                if (first_slot_per_server[server] < 0) {
                    first_slot_per_server[server] = int32_t(k);
                }
            }
            asc_stcg(server_mask + token, mask);
            // 每条有效路由都有专家内序号，重复专家的不同 top-k 槽也分别计数。
            // 此时 ordinal 只在 tile 内有效，第三阶段再加上前面所有 tile 的数量。
            for (uint32_t k = 0; k < topk; ++k) {
                auto* row = route_meta + (base + k) * kMetaFields;
                const int32_t expert = expert_ids[k];
                int32_t gateway = -1;
                int32_t primary = -1;
                int32_t ordinal = -1;
                int32_t flags = 0;
                if (expert >= 0) {
                    primary = first_slot_per_server[uint32_t(expert) / local_experts / 8U];
                    gateway = expert_ids[primary] / int32_t(local_experts);
                    if constexpr (CachedHist) {
                        ordinal = local_counts[expert];
                        local_counts[expert] = ordinal + 1;
                    } else {
                        ordinal = asc_ldcg(counts + expert);
                        asc_stcg(counts + expert, ordinal + 1);
                    }
                    // 每个 token/server 只直发一份数据，其余路由由 gateway 转发。
                    // 7 = valid | direct | primary；9 = valid | relay。
                    flags = int32_t(k) == primary ? 7 : 9;
                    if (int32_t(k) == primary) {
                        if constexpr (CachedHist) {
                            ++local_counts[experts + uint32_t(gateway)];
                        } else {
                            auto* count = counts + experts + uint32_t(gateway);
                            asc_stcg(count, asc_ldcg(count) + 1);
                        }
                    }
                }
                // 六列依次为 expert、gateway、首槽、原始 chunk、专家内序号、标记。
                asc_stcg(row, expert);
                asc_stcg(row + 1, gateway);
                asc_stcg(row + 2, primary);
                asc_stcg(row + 3, int32_t(token / chunk_tokens));
                asc_stcg(row + 4, ordinal);
                asc_stcg(row + 5, flags);
            }
        }
    }
    asc_threadfence();
}

// 一个任务独占 (segment, column)，先计算段内前缀和及段总量。
// 段最多 16 个 tile；不同任务没有交叉写，不依赖线程执行顺序。
constexpr uint32_t kPrefixSegmentTiles = 16U;
template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void PrefixLocalTopkSegmentsVf(__gm__ int32_t* hist,
                                                                         __gm__ int32_t* segment_prefix, uint32_t world,
                                                                         uint32_t tokens, uint32_t experts,
                                                                         uint32_t core)
{
    const uint32_t columns = experts + world;
    const uint32_t stride = (columns + 15U) / 16U * 16U;
    const uint32_t tiles = (uint64_t(tokens) + kTileTokens - 1U) / kTileTokens;
    const uint32_t segments = (tiles + kPrefixSegmentTiles - 1U) / kPrefixSegmentTiles;
    for (uint64_t task = threadIdx.x * CoreCount + core; task < uint64_t(segments) * columns;
         task += (CoreCount * kSimtThreads)) {
        const uint32_t segment = task / columns;
        const uint32_t col = task % columns;
        const uint32_t begin = segment * kPrefixSegmentTiles;
        const uint32_t end = tiles - begin < kPrefixSegmentTiles ? tiles : begin + kPrefixSegmentTiles;
        int32_t sum = 0;
        for (uint32_t tile = begin; tile < end; ++tile) {
            auto* value = hist + uint64_t(tile) * stride + col;
            const int32_t count = asc_ldcg(value);
            asc_stcg(value, sum);
            sum += count;
        }
        asc_stcg(segment_prefix + uint64_t(segment) * stride + col, sum);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void PrefixLocalTopkSegmentTotalsVf(__gm__ int32_t* segment_prefix,
                                                                              __gm__ int32_t* expert_counts,
                                                                              __gm__ int32_t* gateway_counts,
                                                                              uint32_t world, uint32_t tokens,
                                                                              uint32_t experts, uint32_t core)
{
    const uint32_t columns = experts + world;
    const uint32_t stride = (columns + 15U) / 16U * 16U;
    const uint32_t tiles = (uint64_t(tokens) + kTileTokens - 1U) / kTileTokens;
    const uint32_t segments = (tiles + kPrefixSegmentTiles - 1U) / kPrefixSegmentTiles;
    for (uint32_t col = threadIdx.x * CoreCount + core; col < columns; col += (CoreCount * kSimtThreads)) {
        int32_t sum = 0;
        for (uint32_t segment = 0; segment < segments; ++segment) {
            auto* value = segment_prefix + uint64_t(segment) * stride + col;
            const int32_t count = asc_ldcg(value);
            asc_stcg(value, sum);
            sum += count;
        }
        // 空输入也完整覆盖计数；finalize 直接合并段内、段间两级偏移。
        if (col < experts) {
            asc_stcg(expert_counts + col, sum);
        } else {
            asc_stcg(gateway_counts + col - experts, sum);
        }
    }
    asc_threadfence();
}

// 每个任务独占块中的一列。相邻线程读取相邻列，在 UB 中按行有序扫描，
// 不再对每一个前缀元素执行 GM 标量读写。GM 矩形块由外层 MTE 统一搬入/写回。
constexpr uint32_t kPrefixBlockColumns = 16U;
template <bool Totals, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void PrefixLocalTopkBlocksVf(
    __ubuf__ int32_t* cache, __gm__ int32_t* segment_prefix, __gm__ int32_t* expert_counts,
    __gm__ int32_t* gateway_counts, uint32_t rows, uint32_t block_rows, uint32_t world, uint32_t experts, uint32_t core)
{
    const uint32_t columns = experts + world;
    const uint32_t column_blocks = (columns + kPrefixBlockColumns - 1U) / kPrefixBlockColumns;
    const uint32_t stride = column_blocks * kPrefixBlockColumns;
    const uint64_t blocks = (uint64_t(rows) + block_rows - 1U) / block_rows * column_blocks;
    const uint64_t local_blocks = core < blocks ? (blocks - core + CoreCount - 1U) / CoreCount : 0U;
    for (uint64_t task = threadIdx.x; task < local_blocks * kPrefixBlockColumns; task += kSimtThreads) {
        const uint64_t local_block = task / kPrefixBlockColumns;
        const uint32_t lane = task % kPrefixBlockColumns;
        const uint64_t block = core + local_block * CoreCount;
        const uint32_t col = block % column_blocks * kPrefixBlockColumns + lane;
        if (col >= columns) {
            continue;
        }
        const uint32_t begin = block / column_blocks * block_rows;
        const uint32_t count = rows - begin < block_rows ? rows - begin : block_rows;
        auto* values = cache + local_block * block_rows * kPrefixBlockColumns + lane;
        int32_t sum = 0;
        for (uint32_t row = 0; row < count; ++row) {
            const int32_t value = values[row * kPrefixBlockColumns];
            values[row * kPrefixBlockColumns] = sum;
            sum += value;
        }
        if constexpr (Totals) {
            if (col < experts) {
                asc_stcg(expert_counts + col, sum);
            } else {
                asc_stcg(gateway_counts + col - experts, sum);
            }
        } else {
            asc_stcg(segment_prefix + (block / column_blocks) * stride + col, sum);
        }
    }
    asc_threadfence();
}

template <bool Segmented = false, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void FinalizeLocalTopkVf(__gm__ int32_t* route_meta,
                                                                   __gm__ int32_t* local_dst, __gm__ int32_t* hist,
                                                                   uint32_t world, uint32_t tokens, uint32_t topk,
                                                                   uint32_t experts, uint32_t core,
                                                                   __gm__ int32_t* segment_prefix = nullptr)
{
    const uint32_t hist_stride = (experts + world + 15U) / 16U * 16U;
    const uint64_t routes = uint64_t(tokens) * topk;
    // prefix 已完成，route 之间没有依赖；每个线程独占 route，避免串行处理整个 tile。
    for (uint64_t route = threadIdx.x * CoreCount + core; route < routes; route += (CoreCount * kSimtThreads)) {
        const uint64_t tile = route / topk / kTileTokens;
        auto* prefix = hist + uint64_t(tile) * hist_stride;
        auto* row = route_meta + route * kMetaFields;
        const int32_t expert = asc_ldcg(row);
        int32_t destination = -1;
        if (expert >= 0) {
            int32_t ordinal = asc_ldcg(row + 4) + asc_ldcg(prefix + expert);
            if constexpr (Segmented) {
                ordinal += asc_ldcg(segment_prefix + (tile / kPrefixSegmentTiles) * hist_stride + expert);
            }
            asc_stcg(row + 4, ordinal);
            // 仅直发槽写 local_dst；转发槽和非法槽保持 -1。
            // 该值仍是本地专家流内序号，尚不是交换路由后的最终接收偏移。
            if ((asc_ldcg(row + 5) & 2) != 0) {
                destination = ordinal;
            }
        }
        asc_stcg(local_dst + route, destination);
    }
    asc_threadfence();
}

using NotifyTiling = ascend_deepep::NotifyTiling;

// 所有 GM 区间由主机 MakeNotifyTiling 以 512B 对齐划分。
// R1/R2 接收区由 source rank 独占一行；R3 由本 server gateway lane 独占一行。
// F 发包按 token tile 划分，接收端逐行写回；B 行按 lane/source/chunk 划分。
// 前缀和确定独占写区间，各级本地偏移 scratch 不与通信接收区重叠。
// UB：8 KiB 通信暂存+32 KiB meta 缓存+8 KiB bases 缓存。
// forward 阶段 meta/bases 由 MTE2 写；计划和 uint16 token 前缀矩阵位于 meta 后方。
// count、核内 token prefix、row writer 之间显式 V 同步；跨 tile 的 prefix 使用原有全核同步。
// parse 在更早阶段暂借 meta UB 更新 histogram，经 V_MTE3 搬出，排空后才供 MTE2 复用。
// 每轮由 peer%CoreCount 核发送，MTE 完成后才复用 UB 或进入 collective。
constexpr uint32_t kNotifySimtThreads = 128U;
template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyPackCountsVf(__gm__ int32_t* counts, __gm__ int32_t* send,
                                                                  uint32_t experts, uint32_t core)
{
    for (uint32_t e = threadIdx.x * CoreCount + core; e < experts; e += (CoreCount * kNotifySimtThreads)) {
        asc_stcg(send + e, asc_ldcg(counts + e));
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyExpertTotalsVf(__gm__ int32_t* inbox, __gm__ int32_t* bases,
                                                                    __gm__ int32_t* expert_totals, uint32_t rank,
                                                                    uint32_t world, uint32_t experts, uint64_t stride,
                                                                    uint32_t core)
{
    // 每个 expert 独占一个任务；跨 source 累加总量和当前 source 之前的数量。
    // bases 暂存 source 前缀，下一阶段才补上同 rank 中前面 expert 的总量。
    for (uint32_t e = threadIdx.x * CoreCount + core; e < experts; e += (CoreCount * kNotifySimtThreads)) {
        int32_t total = 0;
        int32_t source_prefix = 0;
        for (uint32_t source = 0; source < world; ++source) {
            const int32_t count = asc_ldcg(inbox + uint64_t(source) * stride + e);
            if (source < rank) {
                source_prefix += count;
            }
            total += count;
        }
        asc_stcg(expert_totals + e, total);
        asc_stcg(bases + e, source_prefix);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyBasesVf(__gm__ int32_t* expert_totals, __gm__ int32_t* bases,
                                                             __gm__ int32_t* rank_rows, uint32_t world,
                                                             uint32_t experts, uint32_t core)
{
    // 每个 peer 独占一段 expert；保持 expert/source/token/slot 的接收顺序。
    for (uint32_t peer = threadIdx.x * CoreCount + core; peer < world; peer += (CoreCount * kNotifySimtThreads)) {
        int32_t total = 0;
        for (uint32_t e = peer * (experts / world); e < (peer + 1U) * (experts / world); ++e) {
            asc_stcg(bases + e, total + asc_ldcg(bases + e));
            total += asc_ldcg(expert_totals + e);
        }
        asc_stcg(rank_rows + peer, total);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyAddressStrideVf(__gm__ int32_t* rank_rows,
                                                                     __gm__ int32_t* address_stride, uint32_t world,
                                                                     uint32_t core)
{
    // 全部 route 使用同一个编码步长，仅 core 0/thread 0 计算一次。
    // 全空路由仍写入 1，覆盖上一轮的值。
    if (core == 0U && threadIdx.x == 0U) {
        int32_t stride = 1;
        for (uint32_t rank = 0; rank < world; ++rank) {
            const int32_t count = asc_ldcg(rank_rows + rank);
            stride = count > stride ? count : stride;
        }
        asc_stcg(address_stride, stride);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyResolveDstVf(__gm__ int32_t* meta, __gm__ int32_t* bases,
                                                                  __gm__ int32_t* address_stride, __gm__ int32_t* dst,
                                                                  uint32_t world, uint32_t experts, uint64_t routes,
                                                                  uint32_t core)
{
    const int32_t stride = asc_ldcg(address_stride);
    for (uint64_t route = threadIdx.x * CoreCount + core; route < routes; route += (CoreCount * kNotifySimtThreads)) {
        auto* row = meta + route * 6U;
        const int32_t expert = asc_ldcg(row);
        int32_t encoded = (-2147483647 - 1);
        if (expert >= 0) {
            encoded = (expert / int32_t(experts / world)) * stride + asc_ldcg(bases + expert) + asc_ldcg(row + 4U);
            if ((asc_ldcg(row + 5U) & 2) == 0) {
                encoded = ~encoded;
            }
        }
        asc_stcg(dst + route, encoded);
    }
    asc_threadfence();
}

// 发送端按 32 token 分块；接收端按 32 行统计 gather，最终 forward 逐行写回。
// 与 NotifyTiling 的 scratch 大小保持一致；每个 tile 只写自己分配的区间。
constexpr uint32_t kForwardTile = 32U;

template <bool Write, bool Cached = false, bool SavePlan = false, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyForwardTilesVf(
    __gm__ int32_t* meta, __gm__ int32_t* bases, __gm__ int32_t* offsets, __gm__ int32_t* packets, uint32_t rank,
    uint32_t world, uint32_t experts, uint32_t tokens, uint32_t topk, uint64_t packet_words, uint32_t core,
    __ubuf__ int32_t* meta_ub, __ubuf__ int32_t* bases_ub, __ubuf__ int32_t* plan_ub = nullptr)
{
    static_assert(!SavePlan || (Cached && !Write));
    const uint32_t tiles = (uint64_t(tokens) + kForwardTile - 1U) / kForwardTile;
    const uint32_t worker = threadIdx.x * CoreCount + core;
    for (uint32_t tile = worker; tile < tiles; tile += (CoreCount * kNotifySimtThreads)) {
        // Each tile owns its GM cursor; Write consumes the prefix in place.
        if constexpr (!Write) {
            for (uint32_t gateway = 0; gateway < world; ++gateway) {
                asc_stcg(offsets + uint64_t(gateway) * tiles + tile, int32_t(0));
            }
        }
        const uint32_t begin = tile * kForwardTile;
        const uint32_t end = tokens - begin < kForwardTile ? tokens : begin + kForwardTile;
        for (uint32_t token = begin; token < end; ++token) {
            // 本核的 tile 为 core、core+CoreCount……，UB 紧凑存放，保留完整 tile 步长。
            const uint64_t ub_token = (uint64_t(tile / CoreCount) * kForwardTile + token - begin) * topk * 6U;
            int32_t gateways[16];
            int32_t primaries[16];
            int32_t hits[16];
            for (uint32_t k = 0; k < topk; ++k) {
                hits[k] = 0;
            }
            for (uint32_t k = 0; k < topk; ++k) {
                if constexpr (Cached) {
                    gateways[k] = meta_ub[ub_token + k * 6U + 1U];
                    primaries[k] = meta_ub[ub_token + k * 6U + 2U];
                } else {
                    auto* row = meta + (uint64_t(token) * topk + k) * 6U;
                    gateways[k] = asc_ldcg(row + 1U);
                    primaries[k] = asc_ldcg(row + 2U);
                }
                if (gateways[k] >= 0) {
                    ++hits[primaries[k]];
                }
            }
            // 单贡献保留首槽；多贡献只输出非首槽。重复专家仍保留独立行。
            for (uint32_t k = 0; k < topk; ++k) {
                const uint64_t plan_index = (ub_token / 6U + k) * 2U;
                if constexpr (SavePlan) {
                    // 每次覆盖所有真实 route，包括非法槽及多贡献的首槽。
                    plan_ub[plan_index] = -1;
                    plan_ub[plan_index + 1U] = 0;
                }
                const int32_t gateway = gateways[k];
                if (gateway < 0) {
                    continue;
                }
                const uint32_t primary = uint32_t(primaries[k]);
                const bool direct = hits[primary] == 1;
                if (!direct && k == primary) {
                    continue;
                }
                auto* cursor = offsets + uint64_t(gateway) * tiles + tile;
                const uint32_t index = uint32_t(asc_ldcg(cursor));
                asc_stcg(cursor, int32_t(index + 1U));
                if constexpr (SavePlan) {
                    // 沿 token/slot 原顺序确定行号，后续写线程的调度不改变排列。
                    plan_ub[plan_index] = int32_t(index);
                    plan_ub[plan_index + 1U] = direct ? 1 : 0;
                }
                if constexpr (Write) {
                    int32_t expert;
                    int32_t primary_row;
                    int32_t peer_row = -1;
                    if constexpr (Cached) {
                        auto* row = meta_ub + ub_token + k * 6U;
                        auto* first = meta_ub + ub_token + primary * 6U;
                        expert = row[0];
                        primary_row = bases_ub[first[0]] + first[4];
                        if (!direct) {
                            peer_row = bases_ub[expert] + row[4];
                        }
                    } else {
                        auto* row = meta + (uint64_t(token) * topk + k) * 6U;
                        auto* first = meta + (uint64_t(token) * topk + primary) * 6U;
                        expert = asc_ldcg(row);
                        primary_row = asc_ldcg(bases + asc_ldcg(first)) + asc_ldcg(first + 4U);
                        if (!direct) {
                            peer_row = asc_ldcg(bases + expert) + asc_ldcg(row + 4U);
                        }
                    }
                    auto* out = packets + uint64_t(gateway) * packet_words + 16U + uint64_t(index) * 6U;
                    asc_stcg(out, int32_t(rank | (primary << kWirePrimaryTopkShift) | (k << kWireSecondaryTopkShift)));
                    asc_stcg(out + 1U, int32_t(token));
                    asc_stcg(out + 2U, primary_row);
                    asc_stcg(out + 3U, direct ? -1 : expert / int32_t(experts / world));
                    asc_stcg(out + 4U, peer_row);
                    asc_stcg(out + 5U, int32_t(-1));
                }
            }
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyForwardTokensVf(uint32_t tokens, uint32_t topk, uint32_t world,
                                                                     uint32_t core, __ubuf__ int32_t* meta_ub,
                                                                     __ubuf__ int32_t* plan_ub,
                                                                     __ubuf__ uint16_t* token_prefix_ub)
{
    const uint32_t tiles = (uint64_t(tokens) + kForwardTile - 1U) / kForwardTile;
    const uint32_t local_tiles = core < tiles ? (tiles - core + CoreCount - 1U) / CoreCount : 0U;
    // 每线程独占一个 token 的全部 top-k 槽，仍按 slot 顺序分配 token 内行号。
    // 矩阵由调用方 Duplicate 清零；尾 tile 的无效 token 保持零计数。
    for (uint32_t local_token = threadIdx.x; local_token < local_tiles * kForwardTile;
         local_token += kNotifySimtThreads) {
        const uint32_t local_tile = local_token / kForwardTile;
        const uint32_t token_in_tile = local_token % kForwardTile;
        const uint32_t token = (core + local_tile * CoreCount) * kForwardTile + token_in_tile;
        if (token >= tokens) {
            continue;
        }
        int32_t hits[16];
        for (uint32_t k = 0; k < topk; ++k) {
            hits[k] = 0;
        }
        const uint64_t first_route = uint64_t(local_token) * topk;
        for (uint32_t k = 0; k < topk; ++k) {
            auto* row = meta_ub + (first_route + k) * 6U;
            if (row[1] >= 0) {
                ++hits[uint32_t(row[2])];
            }
        }
        for (uint32_t k = 0; k < topk; ++k) {
            auto* row = meta_ub + (first_route + k) * 6U;
            auto* plan = plan_ub + (first_route + k) * 2U;
            plan[0] = -1;
            plan[1] = 0;
            if (row[1] < 0) {
                continue;
            }
            const uint32_t primary = uint32_t(row[2]);
            const bool direct = hits[primary] == 1;
            if (!direct && k == primary) {
                continue;
            }
            auto* count =
                token_prefix_ub + (uint64_t(local_tile) * world + uint32_t(row[1])) * kForwardTile + token_in_tile;
            plan[0] = int32_t(*count);
            plan[1] = direct ? 1 : 0;
            *count = uint16_t(*count + 1U);
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyForwardTokenPrefixVf(__gm__ int32_t* offsets, uint32_t tokens,
                                                                          uint32_t world, uint32_t core,
                                                                          __ubuf__ uint16_t* token_prefix_ub)
{
    const uint32_t tiles = (uint64_t(tokens) + kForwardTile - 1U) / kForwardTile;
    const uint32_t local_tiles = core < tiles ? (tiles - core + CoreCount - 1U) / CoreCount : 0U;
    // 每个任务独占 (local_tile, gateway) 的 32 个 token，顺序不依赖调度。
    for (uint32_t task = threadIdx.x; task < local_tiles * world; task += kNotifySimtThreads) {
        uint32_t sum = 0;
        auto* row = token_prefix_ub + uint64_t(task) * kForwardTile;
        for (uint32_t token = 0; token < kForwardTile; ++token) {
            const uint32_t count = row[token];
            row[token] = uint16_t(sum);
            sum += count;
        }
        const uint32_t tile = core + (task / world) * CoreCount;
        asc_stcg(offsets + uint64_t(task % world) * tiles + tile, int32_t(sum));
    }
    asc_threadfence();
}

template <bool TokenPrefix = false, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyForwardRowsVf(
    __gm__ int32_t* offsets, __gm__ int32_t* packets, uint32_t rank, uint32_t world, uint32_t experts, uint32_t tokens,
    uint32_t topk, uint64_t packet_words, uint32_t core, __ubuf__ int32_t* meta_ub, __ubuf__ int32_t* bases_ub,
    __ubuf__ int32_t* plan_ub, __ubuf__ uint16_t* token_prefix_ub = nullptr)
{
    const uint32_t tiles = (uint64_t(tokens) + kForwardTile - 1U) / kForwardTile;
    const uint32_t local_tiles = core < tiles ? (tiles - core + CoreCount - 1U) / CoreCount : 0U;
    // 一核的全部 route 分给 128 个线程，每个有效计划独占一个最终输出行。
    const uint64_t routes = uint64_t(local_tiles) * kForwardTile * topk;
    for (uint64_t route = threadIdx.x; route < routes; route += kNotifySimtThreads) {
        const uint64_t local_token = route / topk;
        const uint32_t tile = core + uint32_t(local_token / kForwardTile) * CoreCount;
        const uint32_t token = tile * kForwardTile + uint32_t(local_token % kForwardTile);
        if (token >= tokens) {
            continue;
        }
        const int32_t local_row = plan_ub[route * 2U];
        if (local_row < 0) {
            continue;
        }
        const bool direct = plan_ub[route * 2U + 1U] != 0;
        auto* row = meta_ub + route * 6U;
        const uint32_t gateway = uint32_t(row[1]);
        auto* first = meta_ub + (route - route % topk + uint32_t(row[2])) * 6U;
        const int32_t primary_row = bases_ub[first[0]] + first[4];
        const int32_t expert = row[0];
        const int32_t peer_row = direct ? -1 : bases_ub[expert] + row[4];
        uint32_t index = uint32_t(asc_ldcg(offsets + uint64_t(gateway) * tiles + tile)) + uint32_t(local_row);
        if constexpr (TokenPrefix) {
            index += token_prefix_ub[(local_token / kForwardTile * world + gateway) * kForwardTile +
                                     local_token % kForwardTile];
        }
        auto* out = packets + uint64_t(gateway) * packet_words + 16U + uint64_t(index) * 6U;
        asc_stcg(out, int32_t(rank | (uint32_t(row[2]) << kWirePrimaryTopkShift) |
                              (uint32_t(route % topk) << kWireSecondaryTopkShift)));
        asc_stcg(out + 1U, int32_t(token));
        asc_stcg(out + 2U, primary_row);
        asc_stcg(out + 3U, direct ? -1 : expert / int32_t(experts / world));
        asc_stcg(out + 4U, peer_row);
        asc_stcg(out + 5U, int32_t(-1));
    }
    asc_threadfence();
}

template <bool Cached = false, uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyForwardPrefixVf(__gm__ int32_t* offsets, __gm__ int32_t* packets,
                                                                     uint32_t world, uint32_t tokens,
                                                                     uint64_t packet_words, uint32_t core,
                                                                     __ubuf__ int32_t* offsets_ub = nullptr)
{
    const uint32_t tiles = (uint64_t(tokens) + kForwardTile - 1U) / kForwardTile;
    for (uint32_t gateway = threadIdx.x * CoreCount + core; gateway < world;
         gateway += (CoreCount * kNotifySimtThreads)) {
        int32_t total = 0;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            int32_t count;
            if constexpr (Cached) {
                // 每个 gateway 只有一个线程写，串行 UB 扫描保持原来的 tile 顺序。
                const uint32_t stride = (tiles + 7U) / 8U * 8U;
                auto* slot = offsets_ub + uint64_t(gateway / CoreCount) * stride + tile;
                count = *slot;
                *slot = total;
            } else {
                auto* slot = offsets + uint64_t(gateway) * tiles + tile;
                count = asc_ldcg(slot);
                asc_stcg(slot, total);
            }
            total += count;
        }
        // 零 token 时也写 header=0，不能沿用上轮 packet 数量。
        asc_stcg(packets + uint64_t(gateway) * packet_words, total);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyGatherCountTilesVf(__gm__ int32_t* packets,
                                                                        __gm__ int32_t* offsets,
                                                                        __gm__ int32_t* forward, uint32_t rank,
                                                                        uint32_t world, uint32_t capacity,
                                                                        uint64_t packet_words, uint32_t core)
{
    const uint32_t tiles = (uint64_t(capacity) + kForwardTile - 1U) / kForwardTile;
    // source 放在任务低位，各 source 分到相同数量的线程，只遍历实际收到的行。
    const uint32_t worker = threadIdx.x * CoreCount + core;
    const uint32_t source = worker % world;
    const uint32_t source_worker = worker / world;
    // 主机只接受 2/4/8/16/32/64/128/256 卡，均可整除总线程数。
    const uint32_t source_workers = (CoreCount * kNotifySimtThreads) / world;
    auto* packet = packets + uint64_t(source) * packet_words;
    const uint32_t count = uint32_t(asc_ldcg(packet));
    const uint32_t used_tiles = (uint64_t(count) + kForwardTile - 1U) / kForwardTile;
    for (uint32_t tile = source_worker; tile < used_tiles; tile += source_workers) {
        const uint64_t task = uint64_t(source) * tiles + tile;
        const uint32_t begin = tile * kForwardTile;
        const uint32_t end = begin + kForwardTile < count ? begin + kForwardTile : count;
        int32_t gather_count = 0;
        for (uint32_t i = begin; i < end; ++i) {
            const int32_t peer = asc_ldcg(packet + 16U + uint64_t(i) * 6U + 3U);
            // 每行唯一写者；将 tile 内序号暂存到最终输出的第六列，供逐行写回读取。
            auto* slot = forward + (uint64_t(source) * capacity + i) * 6U + 5U;
            if (peer >= 0 && peer != int32_t(rank)) {
                asc_stcg(slot, gather_count++);
            } else {
                asc_stcg(slot, int32_t(-1));
            }
        }
        asc_stcg(offsets + task, gather_count);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyGatherSourcePrefixVf(__gm__ int32_t* packets,
                                                                          __gm__ int32_t* offsets,
                                                                          __gm__ int32_t* prefix, uint32_t world,
                                                                          uint32_t capacity, uint64_t packet_words,
                                                                          uint32_t core)
{
    const uint32_t tiles = (uint64_t(capacity) + kForwardTile - 1U) / kForwardTile;
    for (uint32_t source = threadIdx.x * CoreCount + core; source < world; source += (CoreCount * kNotifySimtThreads)) {
        const uint32_t count = uint32_t(asc_ldcg(packets + uint64_t(source) * packet_words));
        const uint32_t used_tiles = (uint64_t(count) + kForwardTile - 1U) / kForwardTile;
        int32_t total = 0;
        for (uint32_t tile = 0; tile < used_tiles; ++tile) {
            auto* slot = offsets + uint64_t(source) * tiles + tile;
            const int32_t value = asc_ldcg(slot);
            asc_stcg(slot, total);
            total += value;
        }
        asc_stcg(prefix + source, total);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyGatherPrefixVf(__gm__ int32_t* prefix, __gm__ int32_t* gather_send,
                                                                    uint32_t rank, uint32_t world, uint32_t core)
{
    // 此线程只扫描 world 个计数，不再串行扫描全部 forward 行。
    if (core == 0U && threadIdx.x == 0U) {
        int32_t total = 0;
        for (uint32_t source = 0; source < world; ++source) {
            const int32_t count = asc_ldcg(prefix + source);
            asc_stcg(prefix + source, total);
            total += count;
        }
        asc_stcg(prefix + world, total);
        asc_stcg(gather_send + rank, total);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyBuildForwardRowsVf(
    __gm__ int32_t* packets, __gm__ int32_t* offsets, __gm__ int32_t* prefix, __gm__ int32_t* forward,
    __gm__ int32_t* counts, __gm__ int32_t* weight_return_meta, uint32_t rank, uint32_t world, uint32_t capacity,
    uint64_t packet_words, uint32_t core, uint32_t tokens, uint32_t topk)
{
    const uint32_t tiles = (uint64_t(capacity) + kForwardTile - 1U) / kForwardTile;
    const uint32_t worker = threadIdx.x * CoreCount + core;
    const uint32_t source = worker % world;
    const uint32_t source_worker = worker / world;
    const uint32_t source_workers = (CoreCount * kNotifySimtThreads) / world;
    auto* packet = packets + uint64_t(source) * packet_words;
    const uint32_t count = uint32_t(asc_ldcg(packet));
    if (source_worker == 0U) {
        asc_stcg(counts + source, int32_t(count));
    }
    const int32_t source_base = asc_ldcg(prefix + source);
    // 一个线程每次写一行。EP32 每个 source 有 128 个线程，不再让一个线程串行写 32 行。
    for (uint32_t i = source_worker; i < count; i += source_workers) {
        auto* row = packet + 16U + uint64_t(i) * 6U;
        auto* out = forward + (uint64_t(source) * capacity + i) * 6U;
        const int32_t local_gather = asc_ldcg(out + 5U);
        const uint32_t wire_source = uint32_t(asc_ldcg(row));
        const int32_t source_rank = int32_t(wire_source & kWireRankMask);
        const int32_t token = asc_ldcg(row + 1U);
        // Column 0 is filled with token_end by NotifyChunkRangesVf.
        for (uint32_t field = 1; field < 5U; ++field) {
            asc_stcg(out + field, asc_ldcg(row + field));
        }
        const uint64_t task = uint64_t(source) * tiles + i / kForwardTile;
        asc_stcg(out + 5U, local_gather >= 0 ? source_base + asc_ldcg(offsets + task) + local_gather : -1);
        // A primary is repeated in its token's secondary records. Only the
        // first record owns its metadata, including across thread/tile splits.
        if (i == 0U || asc_ldcg(row - 6U + 1U) != token) {
            const uint32_t slot = (wire_source >> kWirePrimaryTopkShift) & kWireTopkMask;
            asc_stcg(weight_return_meta + asc_ldcg(row + 2U),
                     int32_t((uint64_t(source_rank) * tokens + token) * topk + slot));
        }
        // Local secondaries have no backward record. Remote secondaries are
        // written by NotifyCollectBackwardVf on their expert rank instead.
        if (asc_ldcg(row + 3U) == int32_t(rank)) {
            const uint32_t slot = (wire_source >> kWireSecondaryTopkShift) & kWireTopkMask;
            asc_stcg(weight_return_meta + asc_ldcg(row + 4U),
                     int32_t((uint64_t(source_rank) * tokens + token) * topk + slot));
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyChunkRangesVf(__gm__ int32_t* packets, __gm__ int32_t* forward,
                                                                   uint32_t capacity, __gm__ int32_t* ranges,
                                                                   __gm__ int32_t* masks, uint32_t rank, uint32_t world,
                                                                   uint32_t chunks, uint32_t chunk_tokens,
                                                                   uint32_t tokens, uint64_t packet_words,
                                                                   uint32_t core)
{
    for (uint64_t task = threadIdx.x * CoreCount + core; task < uint64_t(world) * chunks;
         task += (CoreCount * kNotifySimtThreads)) {
        const uint32_t source = task / chunks;
        const uint32_t chunk = task % chunks;
        const uint32_t first = tokens < chunk_tokens ? tokens : chunk_tokens;
        const uint32_t splits = first < 4U ? first : 4U;
        const uint32_t token_begin_bound =
            chunk < splits ? (chunk * first + splits - 1U) / splits : (chunk + 1U - splits) * chunk_tokens;
        const uint32_t token_end_bound =
            chunk < splits ? ((chunk + 1U) * first + splits - 1U) / splits : (chunk + 2U - splits) * chunk_tokens;
        auto* packet = packets + uint64_t(source) * packet_words;
        const uint32_t count = uint32_t(asc_ldcg(packet));
        uint32_t begin = 0;
        uint32_t end = count;
        // 二分找到本 chunk 起点，避免每个 chunk 从 source 第一行重新扫描。
        while (begin < end) {
            const uint32_t mid = begin + (end - begin) / 2U;
            const uint32_t token = uint32_t(asc_ldcg(packet + 16U + uint64_t(mid) * 6U + 1U));
            if (uint32_t(token) < token_begin_bound) {
                begin = mid + 1U;
            } else {
                end = mid;
            }
        }
        end = begin;
        int32_t mask = 0;
        uint32_t token_begin = begin;
        int32_t previous_token = -1;
        while (end < count) {
            auto* row = packet + 16U + uint64_t(end) * 6U;
            const int32_t token = asc_ldcg(row + 1U);
            if (uint32_t(token) >= token_end_bound) {
                break;
            }
            if (token != previous_token) {
                for (uint32_t i = token_begin; i < end; ++i) {
                    asc_stcg(forward + (uint64_t(source) * capacity + i) * 6U, int32_t(end));
                }
                token_begin = end;
                previous_token = token;
            }
            const int32_t peer = asc_ldcg(row + 3U);
            if (peer >= 0 && peer != int32_t(rank)) {
                mask |= 1 << (peer % 8);
            }
            ++end;
        }
        // A token never crosses raw-token chunks. Each column-0 cell has
        // exactly one writer; the other columns are built from packets above.
        for (uint32_t i = token_begin; i < end; ++i) {
            asc_stcg(forward + (uint64_t(source) * capacity + i) * 6U, int32_t(end));
        }
        asc_stcg(ranges + task * 2U, int32_t(begin));
        asc_stcg(ranges + task * 2U + 1U, int32_t(end));
        asc_stcg(masks + task, mask);
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyCountBackwardTasksVf(__gm__ int32_t* forward,
                                                                          __gm__ int32_t* ranges,
                                                                          __gm__ int32_t* offsets, uint32_t rank,
                                                                          uint32_t world, uint32_t chunks,
                                                                          uint32_t packet_capacity, uint32_t core)
{
    // 一个线程统计一个 (lane, source, chunk)，只扫描该 chunk 的 forward 区间。
    // 所有任务都会写入计数，包括空任务，重复调用不依赖上轮 scratch 内容。
    const uint32_t lanes = world < 8U ? world : 8U;
    const uint32_t worker = threadIdx.x * CoreCount + core;
    const uint64_t task_count = uint64_t(lanes) * world * chunks;
    for (uint64_t task = worker; task < task_count; task += (CoreCount * kNotifySimtThreads)) {
        const uint32_t lane = task / (uint64_t(world) * chunks);
        const uint64_t remainder = task % (uint64_t(world) * chunks);
        const uint32_t source = remainder / chunks;
        const uint32_t chunk = remainder % chunks;
        const uint64_t range_index = uint64_t(source) * chunks + chunk;
        const uint32_t begin = uint32_t(asc_ldcg(ranges + range_index * 2U));
        const uint32_t end = uint32_t(asc_ldcg(ranges + range_index * 2U + 1U));
        const uint32_t peer = rank / 8U * 8U + lane;
        uint32_t count = 0U;
        for (uint32_t i = begin; i < end; ++i) {
            auto* row = forward + (uint64_t(source) * packet_capacity + i) * 6U;
            if (asc_ldcg(row + 3U) == int32_t(peer) && asc_ldcg(row + 5U) >= 0) {
                ++count;
            }
        }
        asc_stcg(offsets + task, int32_t(count));
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyPrefixBackwardChunksVf(
    __gm__ int32_t* offsets, __gm__ int32_t* group_offsets, __gm__ int32_t* source_masks, __gm__ int32_t* group_masks,
    uint32_t rank, uint32_t world, uint32_t chunks, uint32_t core)
{
    const uint32_t lanes = world < 8U ? world : 8U;
    const uint32_t groups = world / (world < 64U ? world : 64U);
    // One owner per (lane, chunk, group). Sources stay in ascending rank order.
    for (uint32_t task = threadIdx.x * CoreCount + core; task < lanes * chunks * groups;
         task += CoreCount * kNotifySimtThreads) {
        const uint32_t lane = task / (chunks * groups);
        const uint32_t chunk = task / groups % chunks, group = task % groups;
        int32_t total = 0, mask = 0;
        for (uint32_t source = 0; source < world; ++source) {
            if (first_hit_schedule::CombineGroup(rank, world, source) != group) {
                continue;
            }
            auto* slot = offsets + (uint64_t(lane) * world + source) * chunks + chunk;
            const int32_t count = asc_ldcg(slot);
            asc_stcg(slot, total);
            total += count;
            if (lane == 0U) {
                mask |= asc_ldcg(source_masks + uint64_t(source) * chunks + chunk);
            }
        }
        asc_stcg(group_offsets + uint64_t(task) * 2U, total);
        if (lane == 0U) {
            asc_stcg(group_masks + group * chunks + chunk, mask);
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyPrefixBackwardTasksVf(__gm__ int32_t* group_offsets,
                                                                           __gm__ int64_t* packets, uint32_t world,
                                                                           uint32_t chunks, uint64_t packet_words,
                                                                           uint32_t core)
{
    const uint32_t lanes = world < 8U ? world : 8U;
    const uint32_t groups = world / (world < 64U ? world : 64U);
    const uint32_t lane = threadIdx.x * CoreCount + core;
    if (lane < lanes) {
        uint32_t total = 0U;
        for (uint32_t task = 0; task < chunks * groups; ++task) {
            auto* slot = group_offsets + (uint64_t(lane) * chunks * groups + task) * 2U;
            const uint32_t count = uint32_t(asc_ldcg(slot));
            asc_stcg(slot, int32_t(total));
            total += count;
            asc_stcg(slot + 1U, int32_t(total));
        }
        asc_stcg(packets + uint64_t(lane) * packet_words, int64_t(total));
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyWriteBackwardTasksVf(
    __gm__ int32_t* forward, __gm__ int32_t* ranges, __gm__ int32_t* offsets, __gm__ int32_t* group_offsets,
    __gm__ int64_t* packets, __gm__ int32_t* forward_packets, uint32_t rank, uint32_t world, uint32_t chunks,
    uint32_t topk, uint32_t packet_capacity, uint64_t packet_words, uint64_t forward_packet_words, uint32_t core)
{
    // 各任务独占 prefix 分配的区间。chunk 尾标记也由同一个任务写入。
    const uint32_t lanes = world < 8U ? world : 8U;
    const uint32_t worker = threadIdx.x * CoreCount + core;
    const uint64_t task_count = uint64_t(lanes) * world * chunks;
    for (uint64_t task = worker; task < task_count; task += (CoreCount * kNotifySimtThreads)) {
        const uint32_t lane = task / (uint64_t(world) * chunks);
        const uint64_t remainder = task % (uint64_t(world) * chunks);
        const uint32_t source = remainder / chunks;
        const uint32_t chunk = remainder % chunks;
        const uint64_t range_index = uint64_t(source) * chunks + chunk;
        const uint32_t begin = uint32_t(asc_ldcg(ranges + range_index * 2U));
        const uint32_t end = uint32_t(asc_ldcg(ranges + range_index * 2U + 1U));
        const uint32_t peer = rank / 8U * 8U + lane;
        const uint32_t groups = world / (world < 64U ? world : 64U);
        const uint32_t group = first_hit_schedule::CombineGroup(rank, world, source);
        auto* bounds = group_offsets + ((uint64_t(lane) * chunks + chunk) * groups + group) * 2U;
        uint32_t output = uint32_t(asc_ldcg(offsets + task)) + uint32_t(asc_ldcg(bounds));
        __gm__ int64_t* last = nullptr;
        auto* packet = packets + uint64_t(lane) * packet_words;
        for (uint32_t i = begin; i < end; ++i) {
            auto* row = forward + (uint64_t(source) * packet_capacity + i) * 6U;
            if (asc_ldcg(row + 3U) != int32_t(peer) || asc_ldcg(row + 5U) < 0) {
                continue;
            }
            auto* out = packet + 8U + uint64_t(output++) * 3U;
            // forward_recv remains intact until this exchange has finished.
            // Its record order matches the restored, persisted forward list.
            const uint32_t wire_source =
                uint32_t(asc_ldcg(forward_packets + uint64_t(source) * forward_packet_words + 16U + uint64_t(i) * 6U));
            const uint32_t topk_id = (wire_source >> kWireSecondaryTopkShift) & kWireTopkMask;
            const uint32_t weight_offset = uint32_t(asc_ldcg(row + 1U)) * topk + topk_id;
            asc_stcg(out, int64_t(uint64_t(source) | (uint64_t(weight_offset) << 32U)));
            asc_stcg(out + 1U, int64_t(asc_ldcg(row + 4U)));
            asc_stcg(out + 2U, int64_t(asc_ldcg(row + 5U)));
            last = out + 2U;
        }
        if (last != nullptr && output == uint32_t(asc_ldcg(bounds + 1U))) {
            const uint64_t flag = uint64_t(group * chunks + chunk + 1U) << 32U;
            asc_stcg(last, int64_t(uint64_t(asc_ldcg(last)) | flag));
        }
    }
    asc_threadfence();
}

template <uint32_t CoreCount = kSimtCores>
__simt_vf__ __launch_bounds__(128) inline void NotifyCollectBackwardVf(
    __gm__ int64_t* packets, __gm__ int64_t* backward, __gm__ int32_t* counts, __gm__ int32_t* gather_recv,
    __gm__ int32_t* gather_rows, __gm__ int32_t* weight_return_meta, uint32_t world, uint32_t topk, uint32_t capacity,
    uint64_t packet_words, uint32_t core, uint32_t tokens)
{
    const uint32_t worker = threadIdx.x * CoreCount + core;
    for (uint32_t lane = 0; lane < (world < 8U ? world : 8U); ++lane) {
        auto* packet = packets + uint64_t(lane) * packet_words;
        const uint32_t count = asc_ldcg(packet);
        if (worker == lane) {
            asc_stcg(counts + lane, int32_t(count));
        }
        for (uint64_t word = worker; word < uint64_t(count) * 3U; word += (CoreCount * kNotifySimtThreads)) {
            int64_t value = asc_ldcg(packet + 8U + word);
            if (word % 3U == 0U) {
                const uint32_t source = uint32_t(value);
                const uint32_t weight_offset = uint32_t(uint64_t(value) >> 32U);
                // Read the published packet, not another thread's new output.
                asc_stcg(weight_return_meta + asc_ldcg(packet + 8U + word + 1U),
                         int32_t(uint64_t(source) * tokens * topk + weight_offset));
                value = int64_t(source);
            }
            asc_stcg(backward + uint64_t(lane) * capacity * 3U + word, value);
        }
    }
    for (uint32_t peer = worker; peer < world; peer += (CoreCount * kNotifySimtThreads)) {
        asc_stcg(gather_rows + peer, asc_ldcg(gather_recv + peer));
    }
    asc_threadfence();
}

__aicore__ inline void NotifyTransfer(GM_ADDR destination, GM_ADDR source, uint64_t bytes, uint32_t peer,
                                      const NotifyTiling& t, LocalTensor<uint8_t> scratch)
{
    auto* mapped = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(destination, peer));
    GlobalTensor<uint8_t> src;
    GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(source);
    if (peer == t.rank) {
        dst.SetGlobalBuffer(destination);
    } else {
        dst.SetGlobalBuffer(mapped);
    }
    for (uint64_t offset = 0; offset < bytes; offset += ascend_deepep::kNotifyCommunicationUbBytes) {
        const uint32_t count = bytes - offset < ascend_deepep::kNotifyCommunicationUbBytes
                                   ? bytes - offset
                                   : ascend_deepep::kNotifyCommunicationUbBytes;
        const DataCopyExtParams copy(1, count, 0, 0, 0);
        DataCopyPad(scratch, src[offset], copy, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(dst[offset], scratch, copy);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (peer != t.rank) {
        // Flush remote MTE stores before the next rank-wide barrier.
        aclshmemx_mte_quiet();
    }
}

__aicore__ inline uint32_t NotifyPacketCount(GM_ADDR packet)
{
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packet));
    dcci_cacheline(packet);
    return gm.GetValue(0);
}

// histogram / segment_prefix 的 GM 行本身按 16 个 int32 对齐，包含尾部 padding。
// 每个矩形块只有一个核负责；UB 行固定 64B，不发生尾行自动补齐导致的地址偏移。
template <bool Totals, uint32_t CoreCount>
__aicore__ inline void PrefixLocalTopkCached(__gm__ int32_t* matrix, __gm__ int32_t* segment_prefix,
                                             __gm__ int32_t* counts, __gm__ int32_t* gateways, uint32_t rows,
                                             uint32_t block_rows, uint32_t world, uint32_t experts, uint32_t core,
                                             LocalTensor<int32_t> cache)
{
    static_assert(kPrefixBlockColumns == ascend_deepep::kNotifyPrefixBlockColumns);
    const uint32_t column_blocks = (experts + world + kPrefixBlockColumns - 1U) / kPrefixBlockColumns;
    const uint32_t stride = column_blocks * kPrefixBlockColumns;
    const uint64_t blocks = (uint64_t(rows) + block_rows - 1U) / block_rows * column_blocks;
    GlobalTensor<int32_t> matrix_gm;
    matrix_gm.SetGlobalBuffer(matrix);
    // 上一阶段的 VF 可能读写这块 UB，先排空，再由 MTE2 覆盖。
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    for (uint64_t block = core; block < blocks; block += CoreCount) {
        const uint32_t begin = block / column_blocks * block_rows;
        const uint32_t count = rows - begin < block_rows ? rows - begin : block_rows;
        const uint64_t gm_offset = uint64_t(begin) * stride + block % column_blocks * kPrefixBlockColumns;
        const uint64_t ub_offset = block / CoreCount * block_rows * kPrefixBlockColumns;
        const DataCopyExtParams copy(uint16_t(count), kPrefixBlockColumns * sizeof(int32_t),
                                     (stride - kPrefixBlockColumns) * sizeof(int32_t), 0, 0);
        DataCopyPad(cache[ub_offset], matrix_gm[gm_offset], copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    asc_vf_call<PrefixLocalTopkBlocksVf<Totals, CoreCount>>(
        dim3(128), reinterpret_cast<__ubuf__ int32_t*>(cache.GetPhyAddr()), segment_prefix, counts, gateways, rows,
        block_rows, world, experts, core);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
    for (uint64_t block = core; block < blocks; block += CoreCount) {
        const uint32_t begin = block / column_blocks * block_rows;
        const uint32_t count = rows - begin < block_rows ? rows - begin : block_rows;
        const uint64_t gm_offset = uint64_t(begin) * stride + block % column_blocks * kPrefixBlockColumns;
        const uint64_t ub_offset = block / CoreCount * block_rows * kPrefixBlockColumns;
        const DataCopyExtParams copy(uint16_t(count), kPrefixBlockColumns * sizeof(int32_t), 0,
                                     (stride - kPrefixBlockColumns) * sizeof(int32_t), 0);
        DataCopyPad(matrix_gm[gm_offset], cache[ub_offset], copy);
    }
    // 所有 GM 前缀必须先写回，才能到达外层 SyncAll 并交给下一阶段读取。
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
}

template <uint32_t CoreCount>
__aicore__ inline void FullNotify(GM_ADDR input, GM_ADDR workspace, GM_ADDR meta_address, GM_ADDR hist_address,
                                  GM_ADDR bases_address, GM_ADDR prefix_address, GM_ADDR counts_address,
                                  GM_ADDR gateways_address, GM_ADDR local_dst, GM_ADDR mask, GM_ADDR dst,
                                  GM_ADDR forward, GM_ADDR forward_counts, GM_ADDR backward, GM_ADDR backward_counts,
                                  GM_ADDR ranges, GM_ADDR masks, GM_ADDR rank_rows, GM_ADDR gather_rows,
                                  GM_ADDR weight_return_meta, const NotifyTiling& t)
{
    const uint32_t core = GetBlockIdx();
    TPipe pipe;
    TBuf<TPosition::VECOUT> scratch_buffer;
    pipe.InitBuffer(scratch_buffer, ascend_deepep::kNotifyCommunicationUbBytes);
    auto scratch = scratch_buffer.Get<uint8_t>();
    TBuf<TPosition::VECIN> forward_meta_buffer;
    TBuf<TPosition::VECIN> forward_bases_buffer;
    static_assert(ascend_deepep::kNotifyDynamicUbBytes <= 216U * 1024U);
    pipe.InitBuffer(forward_meta_buffer, ascend_deepep::kNotifyForwardMetaUbBytes);
    pipe.InitBuffer(forward_bases_buffer, ascend_deepep::kNotifyForwardBasesUbBytes);
    auto forward_meta_ub = forward_meta_buffer.Get<int32_t>();
    auto forward_bases_ub = forward_bases_buffer.Get<int32_t>();
    const bool cache_forward = t.cache_forward_inputs != 0U;
    auto* forward_plan =
        reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()) + t.forward_plan_offset_words;
    auto* forward_token_prefix =
        reinterpret_cast<__ubuf__ uint16_t*>(forward_meta_ub.GetPhyAddr()) + t.forward_token_prefix_offset_words * 2U;
    auto* meta = reinterpret_cast<__gm__ int32_t*>(meta_address);
    auto* hist = reinterpret_cast<__gm__ int32_t*>(hist_address);
    auto* counts = reinterpret_cast<__gm__ int32_t*>(counts_address);
    auto* gateways = reinterpret_cast<__gm__ int32_t*>(gateways_address);
    auto* bases = reinterpret_cast<__gm__ int32_t*>(bases_address);
    auto* rank_counts = reinterpret_cast<__gm__ int32_t*>(rank_rows);
    auto* prefix = reinterpret_cast<__gm__ int32_t*>(prefix_address);
    // 独立的本地 scratch，避免快卡提前发送 gather 数据覆盖慢卡的任务偏移。
    auto* backward_task_offsets = reinterpret_cast<__gm__ int32_t*>(workspace + t.backward_task_offsets);
    auto* forward_offsets = reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_tile_offsets);
    auto* gather_offsets = reinterpret_cast<__gm__ int32_t*>(workspace + t.gather_tile_offsets);
    auto* backward_group_offsets = reinterpret_cast<__gm__ int32_t*>(workspace + t.backward_group_offsets);
    auto* expert_totals = reinterpret_cast<__gm__ int32_t*>(workspace + t.expert_totals);
    auto* address_stride = reinterpret_cast<__gm__ int32_t*>(workspace + t.address_stride);
    auto* local_segments = reinterpret_cast<__gm__ int32_t*>(workspace + t.local_prefix_segments);
    static_assert(kPrefixSegmentTiles == ascend_deepep::kNotifyPrefixSegmentTiles);
    static_assert(kForwardTile == ascend_deepep::kNotifyForwardTileRows);
    if (t.cache_parse_hist != 0U) {
        // 保持 tile/token/slot 顺序，UB 只替代计数的存储位置，不使用原子分配序号。
        const uint32_t hist_stride = (t.experts + t.world + 15U) / 16U * 16U;
        const uint32_t tiles = (uint64_t(t.tokens) + kTileTokens - 1U) / kTileTokens;
        if (core < tiles) {
            const uint32_t local_tiles = (tiles - core + CoreCount - 1U) / CoreCount;
            Duplicate(forward_meta_ub, int32_t(0), local_tiles * hist_stride);
            PipeBarrier<PIPE_V>();
        }
        if (t.cache_parse_routes != 0U) {
            // bases UB is available for cached routes until the forward phase.
            auto* routes_ub = reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr());
            asc_vf_call<ParseLocalTopkTokensVf<CoreCount>>(
                dim3(128), reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(input),
                reinterpret_cast<__gm__ int64_t*>(mask), meta, t.world, t.tokens, t.topk, t.experts, t.chunk_tokens,
                core, routes_ub);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            // 下一 VF 的 tile 线程读取多个 token 线程的结果，显式保护 V 到 V 依赖。
            PipeBarrier<PIPE_V>();
            asc_vf_call<CountLocalTopkTilesVf<CoreCount>>(
                dim3(128), meta, t.world, t.tokens, t.topk, t.experts, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()), routes_ub);
            // 排空最后一个 route UB 读者，之后才允许 MTE2 预取 bases 到同一缓冲。
            SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            asc_vf_call<ParseLocalTopkVf<true, CoreCount>>(
                dim3(128), reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(input),
                reinterpret_cast<__gm__ int64_t*>(mask), meta, hist, t.world, t.tokens, t.topk, t.experts,
                t.chunk_tokens, core, reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()));
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        GlobalTensor<int32_t> hist_gm;
        hist_gm.SetGlobalBuffer(hist);
        for (uint32_t tile = core; tile < tiles; tile += CoreCount) {
            DataCopy(hist_gm[uint64_t(tile) * hist_stride], forward_meta_ub[uint64_t(tile / CoreCount) * hist_stride],
                     hist_stride);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        // 排空全部 histogram 存储，再允许后面的 forward 预取覆盖这块 UB。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    } else {
        asc_vf_call<ParseLocalTopkVf<false, CoreCount>>(
            dim3(128), reinterpret_cast<__gm__ ascend_deepep::topk_idx_t*>(input),
            reinterpret_cast<__gm__ int64_t*>(mask), meta, hist, t.world, t.tokens, t.topk, t.experts, t.chunk_tokens,
            core, reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    const uint32_t prefix_tiles = (uint64_t(t.tokens) + kTileTokens - 1U) / kTileTokens;
    const uint32_t prefix_segments = (prefix_tiles + kPrefixSegmentTiles - 1U) / kPrefixSegmentTiles;
    if (t.cache_local_prefix != 0U) {
        PrefixLocalTopkCached<false, CoreCount>(hist, local_segments, counts, gateways, prefix_tiles,
                                                kPrefixSegmentTiles, t.world, t.experts, core, forward_meta_ub);
    } else {
        asc_vf_call<PrefixLocalTopkSegmentsVf<CoreCount>>(dim3(128), hist, local_segments, t.world, t.tokens, t.experts,
                                                          core);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    if (t.cache_local_prefix_totals != 0U) {
        PrefixLocalTopkCached<true, CoreCount>(local_segments, local_segments, counts, gateways, prefix_segments,
                                               prefix_segments, t.world, t.experts, core, forward_meta_ub);
    } else {
        asc_vf_call<PrefixLocalTopkSegmentTotalsVf<CoreCount>>(dim3(128), local_segments, counts, gateways, t.world,
                                                               t.tokens, t.experts, core);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<FinalizeLocalTopkVf<true, CoreCount>>(dim3(128), meta, reinterpret_cast<__gm__ int32_t*>(local_dst),
                                                      hist, t.world, t.tokens, t.topk, t.experts, core, local_segments);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    asc_vf_call<NotifyPackCountsVf<CoreCount>>(
        dim3(128), counts, reinterpret_cast<__gm__ int32_t*>(workspace + t.count_send), t.experts, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    for (uint32_t peer = core; peer < t.world; peer += CoreCount) {
        NotifyTransfer(workspace + t.count_recv + uint64_t(t.rank) * t.count_stride, workspace + t.count_send,
                       uint64_t(t.experts) * 4U, peer, t, scratch);
    }
    aclshmemx_barrier_all_vec();
    asc_vf_call<NotifyExpertTotalsVf<CoreCount>>(dim3(128), reinterpret_cast<__gm__ int32_t*>(workspace + t.count_recv),
                                                 bases, expert_totals, t.rank, t.world, t.experts, t.count_stride / 4U,
                                                 core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyBasesVf<CoreCount>>(dim3(128), expert_totals, bases, rank_counts, t.world, t.experts, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyAddressStrideVf<CoreCount>>(dim3(128), rank_counts, address_stride, t.world, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyResolveDstVf<CoreCount>>(dim3(128), meta, bases, address_stride,
                                               reinterpret_cast<__gm__ int32_t*>(dst), t.world, t.experts,
                                               uint64_t(t.tokens) * t.topk, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    // meta 缓存和 prefix 搬运均使用此 tile 数，prefix 不依赖 cache_forward 分支。
    const uint32_t tiles = (uint64_t(t.tokens) + kForwardTile - 1U) / kForwardTile;
    if (cache_forward) {
        // bases 已经经过 VF 完成及全核同步；meta 在更早的 count 交换前已完成。
        // 预取计入 forward_count，forward_write 复用缓存，比较性能需合看这两个阶段。
        GlobalTensor<int32_t> bases_gm;
        GlobalTensor<int32_t> meta_gm;
        bases_gm.SetGlobalBuffer(bases);
        meta_gm.SetGlobalBuffer(meta);
        const DataCopyExtParams bases_copy(1, t.experts * sizeof(int32_t), 0, 0, 0);
        DataCopyPad(forward_bases_ub, bases_gm, bases_copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        const uint32_t tile_words = kForwardTile * t.topk * 6U;
        for (uint32_t tile = core; tile < tiles; tile += CoreCount) {
            const uint32_t token_begin = tile * kForwardTile;
            const uint32_t token_count = t.tokens - token_begin < kForwardTile ? t.tokens - token_begin : kForwardTile;
            const uint32_t local_tile = tile / CoreCount;
            // 每个 UB tile 起点均 32B 对齐，尾 tile 只读真实 token，不越过 GM 尾部。
            const DataCopyExtParams meta_copy(1, token_count * t.topk * 6U * sizeof(int32_t), 0, 0, 0);
            DataCopyPad(forward_meta_ub[local_tile * tile_words], meta_gm[uint64_t(token_begin) * t.topk * 6U],
                        meta_copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        if (t.forward_token_prefix_offset_words != 0U) {
            const uint32_t local_tiles = core < tiles ? (tiles - core + CoreCount - 1U) / CoreCount : 0U;
            if (local_tiles != 0U) {
                Duplicate(forward_meta_ub[t.forward_token_prefix_offset_words], int32_t(0),
                          local_tiles * t.world * kForwardTile / 2U);
                PipeBarrier<PIPE_V>();
            }
            asc_vf_call<NotifyForwardTokensVf<CoreCount>>(
                dim3(128), t.tokens, t.topk, t.world, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()), forward_plan, forward_token_prefix);
        } else if (t.forward_plan_offset_words != 0U) {
            asc_vf_call<NotifyForwardTilesVf<false, true, true, CoreCount>>(
                dim3(128), meta, bases, forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send),
                t.rank, t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan);
        } else {
            asc_vf_call<NotifyForwardTilesVf<false, true, false, CoreCount>>(
                dim3(128), meta, bases, forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send),
                t.rank, t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan);
        }
    } else {
        asc_vf_call<NotifyForwardTilesVf<false, false, false, CoreCount>>(
            dim3(128), meta, bases, forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send),
            t.rank, t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
            reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    if (t.forward_token_prefix_offset_words != 0U) {
        // token 计数和前缀只在本核 UB 内交接，无需增加全核同步。
        PipeBarrier<PIPE_V>();
        asc_vf_call<NotifyForwardTokenPrefixVf<CoreCount>>(dim3(128), forward_offsets, t.tokens, t.world, core,
                                                           forward_token_prefix);
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
    }
    SyncAll<true>();
    // count 交换已排空，此处复用通信 scratch，不覆盖仍供 row writer 使用的 meta/plan。
    // 每核拥有 gateway=core,core+CoreCount,...，每个 UB 行起点按 32B 对齐。
    const uint32_t prefix_stride = t.forward_prefix_ub_stride;
    if (prefix_stride != 0U) {
        auto prefix_ub = scratch.ReinterpretCast<int32_t>();
        GlobalTensor<int32_t> offsets_gm;
        offsets_gm.SetGlobalBuffer(forward_offsets);
        const DataCopyExtParams prefix_copy(1, tiles * sizeof(int32_t), 0, 0, 0);
        for (uint32_t gateway = core; gateway < t.world; gateway += CoreCount) {
            DataCopyPad(prefix_ub[(gateway / CoreCount) * prefix_stride], offsets_gm[uint64_t(gateway) * tiles],
                        prefix_copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        asc_vf_call<NotifyForwardPrefixVf<true, CoreCount>>(
            dim3(128), forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send), t.world,
            t.tokens, t.forward_stride / 4U, core, reinterpret_cast<__ubuf__ int32_t*>(prefix_ub.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t gateway = core; gateway < t.world; gateway += CoreCount) {
            // 只回写有效 tile，不将 UB 行尾 padding 写入相邻 gateway 的 GM 区域。
            DataCopyPad(offsets_gm[uint64_t(gateway) * tiles], prefix_ub[(gateway / CoreCount) * prefix_stride],
                        prefix_copy);
        }
        // 先排空 prefix 写回，再允许后续发送用 MTE2 覆盖同一 scratch。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    } else {
        asc_vf_call<NotifyForwardPrefixVf<false, CoreCount>>(
            dim3(128), forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send), t.world,
            t.tokens, t.forward_stride / 4U, core, nullptr);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    if (t.forward_plan_offset_words != 0U) {
        // count VF 写出的 UB 计划将由同核不同线程读取，显式保证 V 到 V 的依赖。
        PipeBarrier<PIPE_V>();
        if (t.forward_token_prefix_offset_words != 0U) {
            asc_vf_call<NotifyForwardRowsVf<true, CoreCount>>(
                dim3(128), forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send), t.rank,
                t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan, forward_token_prefix);
        } else {
            asc_vf_call<NotifyForwardRowsVf<false, CoreCount>>(
                dim3(128), forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send), t.rank,
                t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
                reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan, forward_token_prefix);
        }
    } else if (cache_forward) {
        asc_vf_call<NotifyForwardTilesVf<true, true, false, CoreCount>>(
            dim3(128), meta, bases, forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send),
            t.rank, t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
            reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan);
    } else {
        asc_vf_call<NotifyForwardTilesVf<true, false, false, CoreCount>>(
            dim3(128), meta, bases, forward_offsets, reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_send),
            t.rank, t.world, t.experts, t.tokens, t.topk, t.forward_stride / 4U, core,
            reinterpret_cast<__ubuf__ int32_t*>(forward_meta_ub.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(forward_bases_ub.GetPhyAddr()), forward_plan);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    for (uint32_t peer = core; peer < t.world; peer += CoreCount) {
        auto* packet = workspace + t.forward_send + uint64_t(peer) * t.forward_stride;
        NotifyTransfer(workspace + t.forward_recv + uint64_t(t.rank) * t.forward_stride, packet,
                       64U + uint64_t(NotifyPacketCount(packet)) * 24U, peer, t, scratch);
    }
    aclshmemx_barrier_all_vec();
    asc_vf_call<NotifyGatherCountTilesVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_recv), gather_offsets,
        reinterpret_cast<__gm__ int32_t*>(forward), t.rank, t.world, t.forward_capacity, t.forward_stride / 4U, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyGatherSourcePrefixVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_recv), gather_offsets, prefix, t.world,
        t.forward_capacity, t.forward_stride / 4U, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyGatherPrefixVf<CoreCount>>(
        dim3(128), prefix, reinterpret_cast<__gm__ int32_t*>(workspace + t.gather_send), t.rank, t.world, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyBuildForwardRowsVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_recv), gather_offsets, prefix,
        reinterpret_cast<__gm__ int32_t*>(forward), reinterpret_cast<__gm__ int32_t*>(forward_counts),
        reinterpret_cast<__gm__ int32_t*>(weight_return_meta), t.rank, t.world, t.forward_capacity,
        t.forward_stride / 4U, core, t.tokens, t.topk);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    // chunk 信息直接读取接收包，不依赖其他核写完 forward 表。
    asc_vf_call<NotifyChunkRangesVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_recv),
        reinterpret_cast<__gm__ int32_t*>(forward), t.forward_capacity, reinterpret_cast<__gm__ int32_t*>(ranges),
        reinterpret_cast<__gm__ int32_t*>(workspace + t.source_chunk_masks), t.rank, t.world, t.chunks, t.chunk_tokens,
        t.tokens, t.forward_stride / 4U, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyCountBackwardTasksVf<CoreCount>>(dim3(128), reinterpret_cast<__gm__ int32_t*>(forward),
                                                       reinterpret_cast<__gm__ int32_t*>(ranges), backward_task_offsets,
                                                       t.rank, t.world, t.chunks, t.forward_capacity, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyPrefixBackwardChunksVf<CoreCount>>(
        dim3(128), backward_task_offsets, backward_group_offsets,
        reinterpret_cast<__gm__ int32_t*>(workspace + t.source_chunk_masks), reinterpret_cast<__gm__ int32_t*>(masks),
        t.rank, t.world, t.chunks, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyPrefixBackwardTasksVf<CoreCount>>(dim3(128), backward_group_offsets,
                                                        reinterpret_cast<__gm__ int64_t*>(workspace + t.backward_send),
                                                        t.world, t.chunks, t.backward_stride / 8U, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    asc_vf_call<NotifyWriteBackwardTasksVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int32_t*>(forward), reinterpret_cast<__gm__ int32_t*>(ranges),
        backward_task_offsets, backward_group_offsets, reinterpret_cast<__gm__ int64_t*>(workspace + t.backward_send),
        reinterpret_cast<__gm__ int32_t*>(workspace + t.forward_recv), t.rank, t.world, t.chunks, t.topk,
        t.forward_capacity, t.backward_stride / 8U, t.forward_stride / 4U, core);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    SyncAll<true>();
    for (uint32_t peer = core; peer < t.world; peer += CoreCount) {
        if (peer / 8U == t.rank / 8U) {
            auto* packet = workspace + t.backward_send + uint64_t(peer % 8U) * t.backward_stride;
            NotifyTransfer(workspace + t.backward_recv + uint64_t(t.rank % 8U) * t.backward_stride, packet,
                           64U + uint64_t(NotifyPacketCount(packet)) * 24U, peer, t, scratch);
        }
        NotifyTransfer(workspace + t.gather_recv + uint64_t(t.rank) * sizeof(int32_t),
                       workspace + t.gather_send + uint64_t(t.rank) * sizeof(int32_t), sizeof(int32_t), peer, t,
                       scratch);
    }
    aclshmemx_barrier_all_vec();
    asc_vf_call<NotifyCollectBackwardVf<CoreCount>>(
        dim3(128), reinterpret_cast<__gm__ int64_t*>(workspace + t.backward_recv),
        reinterpret_cast<__gm__ int64_t*>(backward), reinterpret_cast<__gm__ int32_t*>(backward_counts),
        reinterpret_cast<__gm__ int32_t*>(workspace + t.gather_recv), reinterpret_cast<__gm__ int32_t*>(gather_rows),
        reinterpret_cast<__gm__ int32_t*>(weight_return_meta), t.world, t.topk, t.backward_capacity,
        t.backward_stride / 8U, core, t.tokens);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
}
#endif
}  // namespace

// Notify 的输出会交给后续 dispatch/combine 使用。这个 barrier kernel 负责
// 跨 rank 的阶段交接，确保元数据就绪后再进入 payload 通信。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void notify_dispatch_barrier_kernel()
{
#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 3510
    util_set_ffts_config(0U);
    aclshmemx_barrier_all_vec();
#endif
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void notify_full_kernel(
    GM_ADDR input, GM_ADDR workspace, GM_ADDR meta, GM_ADDR hist, GM_ADDR bases, GM_ADDR prefix, GM_ADDR counts,
    GM_ADDR gateways, GM_ADDR local_dst, GM_ADDR mask, GM_ADDR dst, GM_ADDR forward, GM_ADDR forward_counts,
    GM_ADDR backward, GM_ADDR backward_counts, GM_ADDR ranges, GM_ADDR masks, GM_ADDR rank_rows, GM_ADDR gather_rows,
    GM_ADDR weight_return_meta, ascend_deepep::NotifyTiling t)
{
#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 3510
    util_set_ffts_config(0U);
    // 核数在入口选择编译期特化，SIMT 热循环中的除法仍可优化成常量移位。
    if (t.num_cores == 64U) {
        FullNotify<64U>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask, dst, forward,
                        forward_counts, backward, backward_counts, ranges, masks, rank_rows, gather_rows,
                        weight_return_meta, t);
    } else {
        FullNotify<32U>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask, dst, forward,
                        forward_counts, backward, backward_counts, ranges, masks, rank_rows, gather_rows,
                        weight_return_meta, t);
    }
#endif
}

extern "C" ASCEND_DEEPEP_EXPORT void notify_full_kernel_do(
    void* stream, uint8_t* input, uint8_t* workspace, uint8_t* meta, uint8_t* hist, uint8_t* bases, uint8_t* prefix,
    uint8_t* counts, uint8_t* gateways, uint8_t* local_dst, uint8_t* mask, uint8_t* dst, uint8_t* forward,
    uint8_t* forward_counts, uint8_t* backward, uint8_t* backward_counts, uint8_t* ranges, uint8_t* masks,
    uint8_t* rank_rows, uint8_t* gather_rows, uint8_t* weight_return_meta, uint8_t* tiling)
{
    // SIMD 主核调用的 SIMT VF 也访问这些 TPipe buffer，必须在启动时声明动态 UB 范围。
    const auto& config = *reinterpret_cast<ascend_deepep::NotifyTiling*>(tiling);
    notify_full_kernel<<<config.num_cores, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(ascend_deepep::kNotifyDynamicUbBytes),
                         stream>>>(input, workspace, meta, hist, bases, prefix, counts, gateways, local_dst, mask, dst,
                                   forward, forward_counts, backward, backward_counts, ranges, masks, rank_rows,
                                   gather_rows, weight_return_meta, config);
}

extern "C" ASCEND_DEEPEP_EXPORT void notify_dispatch_barrier_kernel_do(void* stream, uint32_t num_cores)
{
    notify_dispatch_barrier_kernel<<<num_cores, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>();
}

namespace {
#include "../dispatch/peer_io.hpp"
#include "../dispatch/peer_plan.hpp"
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void notify_dispatch_peer_plan_kernel(
    GM_ADDR workspace, GM_ADDR dst, GM_ADDR forward, GM_ADDR counts, GM_ADDR plan, ascend_deepep::DispatchTiling t,
    uint64_t stride_offset)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    // Decode Notify's actual row stride; the generated table stores unencoded rows.
    GlobalTensor<int32_t> stride;
    stride.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + stride_offset));
    stride.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    t.rows = stride.GetValue(0);
    t.address_stride = t.rows;
    PreparePeerPlan(workspace, dst, forward, counts, plan, t);
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT void notify_dispatch_peer_plan_kernel_do(void* stream, uint8_t* workspace, uint8_t* dst,
                                                                         uint8_t* forward, uint8_t* counts,
                                                                         uint8_t* plan, uint8_t* tiling,
                                                                         uint64_t stride_offset)
{
    const auto t = *reinterpret_cast<ascend_deepep::DispatchTiling*>(tiling);
    notify_dispatch_peer_plan_kernel<<<32, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(4096), stream>>>(
        workspace, dst, forward, counts, plan, t, stride_offset);
}
