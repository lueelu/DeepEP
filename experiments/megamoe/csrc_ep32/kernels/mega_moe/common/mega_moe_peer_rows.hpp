// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_PEER_ROWS_H
#define MEGA_MOE_PEER_ROWS_H

#include <cstdint>

// Prefill 独立副本默认启用。改为 0 并重编 kernel 可对照原来的逐 slice 扫描路径。
// 本开关不影响 host/device 布局、tiling ABI、通信顺序或配额语义。
#ifndef MEGAMOE_PREFILL_PEER_FAST_PATH
#define MEGAMOE_PREFILL_PEER_FAST_PATH 1
#endif

namespace MegaMoeImpl {

// 按卡内序号交错不同机器，先远端后本机；完整 peer 段仍连续落位，不新增 owner/ready 协议。
// 同一专家的所有 slice 必须使用同一排列，否则 cut 余数会导致源行重复或遗漏。
__aicore__ inline uint32_t GetDispatchPeer(uint32_t ordinal, uint32_t rank, uint32_t peers, uint32_t ranksPerNode)
{
    if (ranksPerNode == 0U || ranksPerNode >= peers) {
        return (rank + ordinal) % peers;
    }
    // Host 保证每机卡数整除 EP，且同机 rank 连续编号。
    uint32_t nodes = peers / ranksPerNode;
    uint32_t nodeOffset = ordinal % nodes;
    if (nodeOffset < 2U) {
        nodeOffset = 1U - nodeOffset;
    }
    uint32_t node = (rank / ranksPerNode + nodeOffset) % nodes;
    uint32_t localRank = (rank % ranksPerNode + ordinal / nodes) % ranksPerNode;
    return node * ranksPerNode + localRank;
}

struct PeerRowLimits {
    uint64_t total = 0U;
    uint32_t minimum = 0xFFFFFFFFU;
    uint32_t maximum = 0U;
};

/*
 * 功能：缓存当前专家的 peer 统计，输入为本次 launch 的真实路由计数。
 * 输出：供不同 slice 复用的 total/minimum/maximum。
 * 设计：每个 AIV1 的普通 C++ 状态，不放在 SwiGLU 会覆盖的显式 UB Tensor 内。
 * 调度按专家单调推进，因此只保留一个专家；PrepareMoeExpertTokenCountTable 每次重置。
 */
struct PeerRowLimitsCache {
    bool valid = false;
    uint32_t expert = 0U;
    uint32_t peers = 0U;
    PeerRowLimits limits{};
};

// 把各 peer 的路由队列逐轮各取一行，耗尽者跳过；cut 表示此前消费的完整轮及余数。
// 两个 cut 之差决定 slice 配额，避免独立四舍五入导致行数不守恒或跨 slice 重复。
struct PeerRowCut {
    uint32_t rounds;
    uint32_t extraPeers;
};

template <typename PeerCounts>
__aicore__ inline PeerRowLimits GetPeerRowLimits(const PeerCounts& counts, uint32_t peers)
{
    PeerRowLimits limits;
    for (uint32_t peer = 0U; peer < peers; ++peer) {
        uint32_t count = counts.Get(peer);
        limits.total += count;
        limits.minimum = count < limits.minimum ? count : limits.minimum;
        limits.maximum = count > limits.maximum ? count : limits.maximum;
    }
    return limits;
}

// 同专家只在本核第一次实际参与 Dispatch 时扫描一次。换专家/EP 或 reset 后重新计算。
template <typename PeerCounts>
__aicore__ inline const PeerRowLimits& GetCachedPeerRowLimits(PeerRowLimitsCache& cache, const PeerCounts& counts,
                                                              uint32_t expert, uint32_t peers)
{
    if (!cache.valid || cache.expert != expert || cache.peers != peers) {
        cache.limits = GetPeerRowLimits(counts, peers);
        cache.expert = expert;
        cache.peers = peers;
        cache.valid = true;
    }
    return cache.limits;
}

// 只由真实最小行数证明“当前 slice 结束前没有 peer 提前耗尽”，不根据 BS 或均值猜测。
// 若 minimum=0、发生偏斜耗尽或范围为空，交给原通用路径。
__aicore__ inline bool CanUseRegularPeerSlice(const PeerRowLimits& limits, uint32_t peers, uint32_t begin, uint32_t end)
{
    return peers != 0U && begin < end && static_cast<uint64_t>(end) <= static_cast<uint64_t>(limits.minimum) * peers;
}

/*
 * 无 peer 提前耗尽时的 slice 配额描述，只保存两个边界的商/余数，不生成累计表。
 * 第 j 个交错 peer 的源区间为 [qb+(j<rb), qe+(j<re))。
 * 前 j 个 peer 的目标行数为 j*(qe-qb)+min(j,re)-min(j,rb)，自然支持零配额和余数跨轮。
 */
struct RegularPeerSlice {
    uint32_t beginRounds = 0U;
    uint32_t beginExtra = 0U;
    uint32_t endRounds = 0U;
    uint32_t endExtra = 0U;

    // 返回相对 slice 起点的累计行数；中间乘法用 uint64，防止大 EP/行数时溢出。
    __aicore__ inline uint32_t Prefix(uint32_t ordinal) const
    {
        uint32_t beforeExtra = ordinal < beginExtra ? ordinal : beginExtra;
        uint32_t afterExtra = ordinal < endExtra ? ordinal : endExtra;
        return static_cast<uint32_t>(static_cast<uint64_t>(ordinal) * (endRounds - beginRounds) + afterExtra -
                                     beforeExtra);
    }

    // 源队列边界是 token ordinal，不是目标 row；保持原 token/top-k 身份。
    __aicore__ inline uint32_t SourceBegin(uint32_t ordinal) const
    {
        return beginRounds + static_cast<uint32_t>(ordinal < beginExtra);
    }

    // 配套的源队列结束位置，不读取 peer count，也不修改余数状态。
    __aicore__ inline uint32_t SourceEnd(uint32_t ordinal) const
    {
        return endRounds + static_cast<uint32_t>(ordinal < endExtra);
    }
};

// 调用者已验证 peers>0 及规则配额条件，任意 slice 边界均可转换，无需整除 EP。
__aicore__ inline RegularPeerSlice CreateRegularPeerSlice(uint32_t peers, uint32_t begin, uint32_t end)
{
    return {begin / peers, begin % peers, end / peers, end % peers};
}

// 每 peer 配额相同时用除法 O(1) 定位；其余规则配额对公式二分，无需线性扫描或共享表同步。
__aicore__ inline uint32_t FindRegularPeerOrdinal(const RegularPeerSlice& slice, uint32_t peers, uint32_t row)
{
    uint32_t rowsPerPeer = slice.endRounds - slice.beginRounds;
    if (slice.beginExtra == slice.endExtra && rowsPerPeer != 0U) {
        return row / rowsPerPeer;
    }
    uint32_t begin = 0U;
    uint32_t end = peers;
    while (begin < end) {
        uint32_t middle = begin + (end - begin) / 2U;
        if (slice.Prefix(middle + 1U) <= row) {
            begin = middle + 1U;
        } else {
            end = middle;
        }
    }
    return begin;
}

template <typename PeerCounts>
__aicore__ inline PeerRowCut GetPeerRowCut(const PeerCounts& counts, uint32_t peers, const PeerRowLimits& limits,
                                           uint32_t prefixRows)
{
    if (prefixRows == 0U || peers == 0U) {
        return {0U, 0U};
    }
    if (prefixRows >= limits.total) {
        return {limits.maximum, 0U};
    }
    // 均衡负载或尚无 peer 耗尽时直接求商；不引入逐 token 枚举或额外 workspace。
    if (prefixRows <= static_cast<uint64_t>(limits.minimum) * peers) {
        return {prefixRows / peers, prefixRows % peers};
    }
    uint32_t low = prefixRows / peers;
    uint32_t high = limits.maximum;
    while (low + 1U < high) {
        uint32_t middle = low + (high - low) / 2U;
        uint64_t rows = 0U;
        for (uint32_t peer = 0U; peer < peers; ++peer) {
            uint32_t count = counts.Get(peer);
            rows += count < middle ? count : middle;
        }
        if (rows <= prefixRows) {
            low = middle;
        } else {
            high = middle;
        }
    }
    uint64_t rows = 0U;
    for (uint32_t peer = 0U; peer < peers; ++peer) {
        uint32_t count = counts.Get(peer);
        rows += count < low ? count : low;
    }
    return {low, static_cast<uint32_t>(prefixRows - rows)};
}

// 必须按同一 peer 排列依次调用；只消耗未耗尽 peer 的余数名额。
__aicore__ inline uint32_t TakePeerRowPrefix(uint32_t count, PeerRowCut& cut)
{
    uint32_t rows = count < cut.rounds ? count : cut.rounds;
    if (count > cut.rounds && cut.extraPeers != 0U) {
        ++rows;
        --cut.extraPeers;
    }
    return rows;
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_PEER_ROWS_H
