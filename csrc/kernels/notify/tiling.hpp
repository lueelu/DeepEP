// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#pragma once
#include <cstdint>

namespace ascend_deepep {
using topk_idx_t = int64_t;
inline constexpr uint32_t kNotifyLocalTopkRouteMetaFields = 6U;
inline constexpr uint32_t kNotifyLocalTopkTileTokens = 16U;
inline constexpr uint32_t kNotifyCores = 32U;
inline constexpr uint32_t kNotifyForwardTileRows = 32U;
inline constexpr uint32_t kNotifyForwardMetaUbBytes = 32U * 1024U;
inline constexpr uint32_t kNotifyForwardBasesUbBytes = 8U * 1024U;
inline constexpr uint32_t kNotifyCommunicationUbBytes = 8U * 1024U;
inline constexpr uint32_t kNotifyPrefixSegmentTiles = 16U;
inline constexpr uint32_t kNotifyPrefixBlockColumns = 16U;
// 按连续 16 列搬运矩形块；每个核缓存自己负责的全部块，尾块仍预留完整空间。
inline constexpr bool NotifyLocalPrefixFitsUb(uint32_t rows, uint32_t block_rows, uint32_t columns,
                                              uint32_t num_cores = kNotifyCores)
{
    if (rows == 0U || block_rows == 0U) {
        return false;
    }
    const uint64_t column_blocks = (uint64_t(columns) + kNotifyPrefixBlockColumns - 1U) / kNotifyPrefixBlockColumns;
    const uint64_t blocks = (uint64_t(rows) + block_rows - 1U) / block_rows * column_blocks;
    const uint64_t local_blocks = (blocks + num_cores - 1U) / num_cores;
    return local_blocks * block_rows * kNotifyPrefixBlockColumns * sizeof(int32_t) <= kNotifyForwardMetaUbBytes;
}
// TPipe::InitBuffer 使用动态 UB；kernel 启动必须申请同样大小，不能只检查物理容量。
inline constexpr uint32_t kNotifyDynamicUbBytes =
    kNotifyCommunicationUbBytes + kNotifyForwardMetaUbBytes + kNotifyForwardBasesUbBytes;
static_assert(kNotifyDynamicUbBytes % 32U == 0U);

// prefix 暂借通信 UB，每个 gateway 的计数行按 32B 对齐，GM 仍保持紧凑布局。
// 返回每行的 int32 步长；零 token 或容量不足时使用原来的 GM 路径。
inline constexpr uint32_t NotifyForwardPrefixUbStride(uint32_t tokens, uint32_t world,
                                                      uint32_t num_cores = kNotifyCores)
{
    const uint64_t tiles = (uint64_t(tokens) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    const uint64_t stride = (tiles + 7U) / 8U * 8U;
    const uint64_t gateways = (uint64_t(world) + num_cores - 1U) / num_cores;
    return tiles != 0U && stride * gateways * sizeof(int32_t) <= kNotifyCommunicationUbBytes ? uint32_t(stride) : 0U;
}

// 缓存每个核的全部交错 tile，count/write 之间保留，无需反复搬运。
inline constexpr bool NotifyForwardInputsFitUb(uint32_t tokens, uint32_t topk, uint32_t experts,
                                               uint32_t num_cores = kNotifyCores)
{
    const uint64_t tiles = (uint64_t(tokens) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    const uint64_t local_tiles = (tiles + num_cores - 1U) / num_cores;
    return tokens > 0U && uint64_t(experts) * 4U <= kNotifyForwardBasesUbBytes &&
           local_tiles * kNotifyForwardTileRows * topk * 6U * 4U <= kNotifyForwardMetaUbBytes;
}
// meta 每 route 六个 int32，写入计划另占两个 int32：tile 内行号、direct 标记。
// 计划紧跟全部 meta tile，两个区域的起点及 tile 步长均为 32B 对齐。
inline constexpr uint32_t NotifyForwardPlanOffsetWords(uint32_t tokens, uint32_t topk, uint32_t experts,
                                                       uint32_t num_cores = kNotifyCores)
{
    const uint64_t tiles = (uint64_t(tokens) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    const uint64_t local_tiles = (tiles + num_cores - 1U) / num_cores;
    const uint64_t routes = local_tiles * kNotifyForwardTileRows * topk;
    return NotifyForwardInputsFitUb(tokens, topk, experts, num_cores) &&
                   routes * 8U * sizeof(int32_t) <= kNotifyForwardMetaUbBytes
               ? uint32_t(routes * 6U)
               : 0U;
}
// uint16 矩阵 [local_tile, gateway, token_in_tile] 紧跟写入计划。
// 每个 tile 最多 32*16 条路由，计数和排他前缀均可由 uint16 表示。
inline constexpr uint32_t NotifyForwardTokenPrefixOffsetWords(uint32_t tokens, uint32_t topk, uint32_t experts,
                                                              uint32_t world, uint32_t num_cores = kNotifyCores)
{
    const uint32_t plan = NotifyForwardPlanOffsetWords(tokens, topk, experts, num_cores);
    const uint64_t tiles = (uint64_t(tokens) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    const uint64_t local_tiles = (tiles + num_cores - 1U) / num_cores;
    const uint64_t matrix_start = uint64_t(plan) / 6U * 8U;
    const uint64_t matrix_bytes = local_tiles * world * kNotifyForwardTileRows * sizeof(uint16_t);
    return plan != 0U && matrix_start * 4U + matrix_bytes <= kNotifyForwardMetaUbBytes ? uint32_t(matrix_start) : 0U;
}
// parse 暂借 forward meta 的 UB，每个 tile 独占一行，行步长按 64B 对齐。
inline constexpr bool NotifyParseHistFitsUb(uint32_t tokens, uint32_t world, uint32_t experts,
                                            uint32_t num_cores = kNotifyCores)
{
    const uint64_t tiles = (uint64_t(tokens) + kNotifyLocalTopkTileTokens - 1U) / kNotifyLocalTopkTileTokens;
    const uint64_t local_tiles = (tiles + num_cores - 1U) / num_cores;
    const uint64_t stride = (uint64_t(experts) + world + 15U) / 16U * 16U;
    return tokens > 0U && local_tiles * stride * sizeof(int32_t) <= kNotifyForwardMetaUbBytes;
}
// parse 阶段暂借 bases UB：每条 route 两个 int32，分别为 expert 和 gateway*16+primary。
inline constexpr bool NotifyParseRoutesFitUb(uint32_t tokens, uint32_t topk, uint32_t world, uint32_t experts,
                                             uint32_t num_cores = kNotifyCores)
{
    const uint64_t tiles = (uint64_t(tokens) + kNotifyLocalTopkTileTokens - 1U) / kNotifyLocalTopkTileTokens;
    const uint64_t local_tiles = (tiles + num_cores - 1U) / num_cores;
    return NotifyParseHistFitsUb(tokens, world, experts, num_cores) &&
           local_tiles * kNotifyLocalTopkTileTokens * topk * 2U * sizeof(int32_t) <= kNotifyForwardBasesUbBytes;
}
inline constexpr uint32_t kNotifyPacketHeader = 64U;

struct NotifyTiling {
    uint32_t num_cores;
    uint32_t rank;
    uint32_t world;
    uint32_t tokens;
    uint32_t topk;
    uint32_t experts;
    uint32_t chunk_tokens;
    uint32_t hidden;
    uint32_t chunks;
    uint32_t forward_capacity;
    uint32_t backward_capacity;
    uint32_t cache_forward_inputs;
    uint32_t cache_parse_hist;
    uint32_t cache_parse_routes;
    uint32_t cache_local_prefix;
    uint32_t cache_local_prefix_totals;
    uint32_t forward_plan_offset_words;
    uint32_t forward_token_prefix_offset_words;
    uint32_t forward_prefix_ub_stride;
    uint64_t count_stride;
    uint64_t forward_stride;
    uint64_t backward_stride;
    uint64_t count_send;
    uint64_t count_recv;
    uint64_t forward_send;
    uint64_t forward_recv;
    uint64_t backward_send;
    uint64_t backward_recv;
    uint64_t gather_send;
    uint64_t gather_recv;
    uint64_t backward_task_offsets;
    uint64_t forward_tile_offsets;
    uint64_t gather_tile_offsets;
    uint64_t backward_group_offsets;
    uint64_t source_chunk_masks;
    uint64_t expert_totals;
    uint64_t address_stride;
    uint64_t local_prefix_segments;
    uint64_t workspace_bytes;
};

// 主机在检查 shape 和乘法上界后调用。设备直接消费偏移，不重新布局。
inline NotifyTiling MakeNotifyTiling(uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk, uint32_t experts,
                                     uint32_t chunk_tokens, uint32_t hidden, uint32_t num_cores = kNotifyCores)
{
    NotifyTiling t{};
    t.num_cores = num_cores;
    t.rank = rank;
    t.world = world;
    t.tokens = tokens;
    t.topk = topk;
    t.experts = experts;
    t.chunk_tokens = chunk_tokens;
    t.hidden = hidden;
    const uint32_t first = tokens < chunk_tokens ? tokens : chunk_tokens;
    const uint32_t splits = first < 4U ? first : 4U;
    t.chunks = tokens ? (uint64_t(tokens) + chunk_tokens - 1U) / chunk_tokens + splits - 1U : 0U;
    t.forward_capacity = tokens ? uint64_t(tokens) * topk : 1U;
    t.backward_capacity = uint64_t(world) * t.forward_capacity;
    t.cache_forward_inputs = NotifyForwardInputsFitUb(tokens, topk, experts, num_cores) ? 1U : 0U;
    t.cache_parse_hist = NotifyParseHistFitsUb(tokens, world, experts, num_cores) ? 1U : 0U;
    t.cache_parse_routes = NotifyParseRoutesFitUb(tokens, topk, world, experts, num_cores) ? 1U : 0U;
    t.forward_plan_offset_words = NotifyForwardPlanOffsetWords(tokens, topk, experts, num_cores);
    t.forward_token_prefix_offset_words = NotifyForwardTokenPrefixOffsetWords(tokens, topk, experts, world, num_cores);
    t.forward_prefix_ub_stride = NotifyForwardPrefixUbStride(tokens, world, num_cores);
    const auto align512 = [](uint64_t n) { return (n + 511U) / 512U * 512U; };
    const uint32_t lanes = world < 8U ? world : 8U;
    t.count_stride = align512(uint64_t(experts) * 4U);
    t.forward_stride = align512(kNotifyPacketHeader + uint64_t(t.forward_capacity) * 24U);
    t.backward_stride = align512(kNotifyPacketHeader + uint64_t(t.backward_capacity) * 24U);
    t.count_send = 0U;
    t.count_recv = t.count_send + t.count_stride;
    t.forward_send = t.count_recv + world * t.count_stride;
    t.forward_recv = t.forward_send + world * t.forward_stride;
    t.backward_send = t.forward_recv + world * t.forward_stride;
    t.backward_recv = t.backward_send + lanes * t.backward_stride;
    t.gather_send = t.backward_recv + lanes * t.backward_stride;
    t.gather_recv = t.gather_send + align512(uint64_t(world) * 4U);
    // 本地任务 scratch 独占区域，不能与其他 rank 可写的接收区重叠。
    t.backward_task_offsets = t.gather_recv + world * 512U;
    t.forward_tile_offsets = t.backward_task_offsets + align512(uint64_t(lanes) * world * t.chunks * 4U);
    const uint64_t send_tiles = (uint64_t(tokens) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    const uint64_t recv_tiles = (uint64_t(t.forward_capacity) + kNotifyForwardTileRows - 1U) / kNotifyForwardTileRows;
    t.gather_tile_offsets = t.forward_tile_offsets + align512(uint64_t(world) * send_tiles * 4U);
    t.backward_group_offsets = t.gather_tile_offsets + align512(uint64_t(world) * recv_tiles * 4U);
    const uint32_t groups = world / (world < 64U ? world : 64U);
    t.source_chunk_masks = t.backward_group_offsets + align512(uint64_t(lanes) * groups * t.chunks * 2U * 4U);
    t.expert_totals = t.source_chunk_masks + align512(uint64_t(world) * t.chunks * 4U);
    t.address_stride = t.expert_totals + align512(uint64_t(experts) * 4U);
    t.local_prefix_segments = t.address_stride + 512U;
    const uint64_t local_tiles = (uint64_t(tokens) + kNotifyLocalTopkTileTokens - 1U) / kNotifyLocalTopkTileTokens;
    const uint64_t segments = (local_tiles + kNotifyPrefixSegmentTiles - 1U) / kNotifyPrefixSegmentTiles;
    t.cache_local_prefix =
        NotifyLocalPrefixFitsUb(local_tiles, kNotifyPrefixSegmentTiles, experts + world, num_cores) ? 1U : 0U;
    t.cache_local_prefix_totals = NotifyLocalPrefixFitsUb(segments, segments, experts + world, num_cores) ? 1U : 0U;
    const uint64_t hist_stride = (uint64_t(experts) + world + 15U) / 16U * 16U;
    t.workspace_bytes = t.local_prefix_segments + align512(segments * hist_stride * 4U);
    return t;
}
}  // namespace ascend_deepep
