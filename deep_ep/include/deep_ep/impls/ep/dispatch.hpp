// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>
#include <cstddef>

#include <kernel_operator.h>
#ifdef DEEPEP_LL_RDMA
#include <shmem.h>
#endif

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/ub.hpp>
#include <deep_ep/comm/transport.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/ep/diagnostics.hpp>
#include <deep_ep/comm/device_profile.hpp>
#include <kernels/common/ep_packet_receive_layout.h>
#include <kernels/common/core_context.h>
#include <kernels/common/ep_server_order.h>

namespace deep_ep {
// Shared packet layout still lives in the common header for combine.
namespace PacketReceiveLayout = DispatchDedup::PacketReceiveLayout;
}

// Public SHMEM headers: MTE performs UB -> peer symmetric GM on PIPE_MTE3.
// Completion is tracked by the caller's existing MTE3 events before UB reuse.
// No SHMEM global barrier is inserted into per-rank, uneven packet loops.
#include <device/gm2gm/engine/shmem_device_mte.h>
#include <device/ub2gm/engine/shmem_device_mte.h>

namespace deep_ep {
using comm::TransportView;
// Each core owns one local 512B magic slot. Retire its 32B scratch before Counts uses UB0.
__aicore__ inline uint32_t RotateDispatchServerDedupMagic(const TransportView& view, uint32_t core)
{
    auto* state = reinterpret_cast<__ubuf__ uint32_t*>(uint64_t(0U));
    auto* slot = view.local_ptr() + uint64_t(core) * kEpServerDedupMagicStride;
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(state), reinterpret_cast<__gm__ uint8_t*>(slot), 1U, 32U, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 32U + 0U, math::align(32U, 32U) + 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
    const uint32_t magic = state[0] ? 0U : 1U;
    state[0] = magic;
    asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(slot), reinterpret_cast<__ubuf__ uint8_t*>(state), 1U, 32U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 32U + 0U, math::align(32U, 32U) + 0U);
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    return magic;
}
}  // namespace DispatchDedup

namespace deep_ep {
__aicore__ inline uint32_t ServerDedupWireOffset(uint32_t logical)
{
    return logical + logical / 480U * 32U;
}
class ServerDedupPacketTail {
public:
    __aicore__ inline ServerDedupPacketTail(__ubuf__ uint8_t* wire, uint32_t header, uint32_t index)
        : header_(reinterpret_cast<__ubuf__ uint32_t*>(wire + header)),
          index_(reinterpret_cast<__ubuf__ uint32_t*>(wire + index)) {}
    __aicore__ inline __ubuf__ uint32_t* Header() { return header_; }
    __aicore__ inline __ubuf__ uint32_t* ReadIndex() { return index_; }
    __aicore__ inline __ubuf__ uint32_t* Entry(uint32_t index) { return index_ + (index + 1U) * 2U; }
private:
    __ubuf__ uint32_t *header_, *index_;
};
__aicore__ inline void ReadServerDedupFlags(__gm__ uint8_t* slot, uint32_t blocks, __ubuf__ float* flags,
                                           bool uncached = false)
{
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(flags), reinterpret_cast<__gm__ uint8_t*>(slot + 480U), blocks, 32U, 0U, 0U, false, (uncached ? asc_load_l2_cache_mode::NOTALLOC_CLEAN : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM), 32U + 480U, math::align(32U, 32U) + 0U);
}
__aicore__ inline void WriteServerDedupPayloadRange(__gm__ uint8_t* output, __ubuf__ uint8_t* wire,
                                                   uint32_t logical, uint32_t bytes)
{
    uint32_t written = 0U;
    const uint32_t within = logical % 480U;
    if (within && bytes) {
        const uint32_t available = 480U - within, head = bytes < available ? bytes : available;
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(output), reinterpret_cast<__ubuf__ uint8_t*>(wire + ServerDedupWireOffset(logical)), 1U, head, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, head + 0U, math::align(head, 32U) + 0U);
        logical += head; bytes -= head; written += head;
    }
    const uint32_t full = bytes / 480U;
    if (full) {
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(output + written), reinterpret_cast<__ubuf__ uint8_t*>(wire + ServerDedupWireOffset(logical)), full, 480U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 480U + 0U, math::align(480U, 32U) + 32U);
        logical += full * 480U; bytes -= full * 480U; written += full * 480U;
    }
    if (bytes) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(output + written), reinterpret_cast<__ubuf__ uint8_t*>(wire + ServerDedupWireOffset(logical)), 1U, bytes, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, bytes + 0U, math::align(bytes, 32U) + 0U);
}
__aicore__ inline uint32_t BuildServerDedupMetadata(__ubuf__ uint32_t* type, __ubuf__ uint32_t* destination,
                                                  ServerDedupPacketTail tail, uint32_t slot)
{
    auto* header = tail.Header();
    const uint32_t role = header[3U];
    type[0] = role;
    if (role == uint32_t(EpServerDedupTokenType::Terminal)) {
        destination[0] = slot; destination[1] = UINT32_MAX; destination[2] = UINT32_MAX;
    } else {
        for (uint32_t i = 0; i < 3U; ++i) destination[i] = header[i];
    }
    return role;
}
__aicore__ inline void PrepareServerDedupForward(ServerDedupPacketTail tail)
{
    tail.Header()[3U] = uint32_t(EpServerDedupTokenType::Terminal);
}

// Static UB maps live in the layout structs below; no runtime bump allocator.
}  // namespace DispatchDedup

namespace deep_ep {

// R05-01/R05-02: stable physical slot positions and a compact pending list.
// The caller supplies the complete, ready prefix in localExpert/source order.
// Both tables remain in UB until this core has finished clearing its slots.
class ServerDedupReceiveTasks {
public:
    // The caller supplies the complete, ready prefix in localExpert/source order
    // plus the two static capacity-sized tables from the receiver UB section.
    __aicore__ inline void Build(__ubuf__ int32_t* prefix, uint32_t rank, uint32_t ranks, uint32_t experts,
                                 uint32_t bs, uint32_t receiver, __ubuf__ uint32_t* tokenPos,
                                 __ubuf__ uint32_t* pendingIdx, uint32_t recvCores)
    {
        tokenPos_ = tokenPos;
        pendingIdx_ = pendingIdx;
        const uint32_t total = uint32_t(prefix[experts - 1U]);
        // 交错分配：receiver 核拥有全局行号 {receiver, receiver+recvCores, ...}。
        // 这样每个核同时拿到早到源与晚到源的 token，relay 转发量也随 token 摊平，
        // 不会再整段继承单个源的到达时刻；输出行由首行和步长直接还原。
        begin_ = receiver;
        stride_ = recvCores;
        count_ = begin_ < total ? (total - 1U - begin_) / stride_ + 1U : 0U;
        pendingCount_ = count_;
        if (count_ == 0U) return;

        const uint32_t localExperts = experts / ranks;
        uint32_t record = 0U;
        // 行号递增，二分只需在剩余 record 区间内进行；小批量下约 1µs。
        for (uint32_t j = 0; j < count_; ++j) {
            const uint32_t row = begin_ + j * stride_;
            uint32_t low = record, high = experts;
            while (low < high) {
                const uint32_t middle = low + (high - low) / 2U;
                if (uint32_t(prefix[middle]) <= row)
                    low = middle + 1U;
                else
                    high = middle;
            }
            record = low;
            const uint32_t previous = record == 0U ? 0U : uint32_t(prefix[record - 1U]);
            const uint32_t source = record % ranks;
            const uint32_t localExpert = record / ranks;
            const uint32_t region = ((source + rank) % ranks) * localExperts + localExpert;
            tokenPos_[j] = region * bs + row - previous;
            pendingIdx_[j] = j;
        }
    }

    __aicore__ inline uint32_t Begin() const
    {
        return begin_;
    }
    // 本核拥有的全局行号步长；输出行 = Begin() + j * Stride()。
    __aicore__ inline uint32_t Stride() const
    {
        return stride_;
    }
    __aicore__ inline uint32_t Count() const
    {
        return count_;
    }
    __aicore__ inline uint32_t PendingCount() const
    {
        return pendingCount_;
    }
    __aicore__ inline uint32_t Position(uint32_t j)
    {
        return tokenPos_[j];
    }
    __aicore__ inline uint32_t PendingIndex(uint32_t p)
    {
        return pendingIdx_[p];
    }

    // Call only after every required write for PendingIndex(p) has been issued.
    // Recheck p after removal; output row and cleanup ownership still use j.
    __aicore__ inline void RemovePending(uint32_t p)
    {
        --pendingCount_;
        pendingIdx_[p] = pendingIdx_[pendingCount_];
    }

private:
    uint32_t begin_ = 0U, stride_ = 1U, count_ = 0U, pendingCount_ = 0U;
    __ubuf__ uint32_t *tokenPos_ = nullptr, *pendingIdx_ = nullptr;
};
}  // namespace DispatchDedup

namespace deep_ep {

/**
 *  for (int i = 0; i < num_i32_elems; ++ i)
 *      if (offset <= data[i] and data[i] < offset + num_bins)
 *          acc[data[i] - offset + 1] ++;
 */
__simd_vf__ void histogram_accumulate_u16(__ubuf__ int32_t* data, __ubuf__ uint16_t* acc,
                                          const int num_i32_elems, const int offset, const int num_bins) {
    static constexpr int kNumIdsPerVector = sizeof(vector_int32_t) / sizeof(int32_t);
    vector_bool mask_b16 = asc_create_mask_b16(PAT_ALL);
    vector_int32_t value_vec;
    vector_uint16_t count_vec;
    asc_duplicate_scalar(count_vec, static_cast<uint16_t>(0), mask_b16);

    uint32_t remaining = static_cast<uint32_t>(num_i32_elems);
    for (int elem_offset = 0; elem_offset < num_i32_elems; elem_offset += kNumIdsPerVector) {
        uint32_t active_bytes = remaining * static_cast<uint32_t>(sizeof(int32_t));
        const vector_bool mask_b32 = asc_update_mask_b32(remaining);
        asc_loadalign(value_vec, data + elem_offset);
        asc_max_scalar(value_vec, value_vec, offset - 1, mask_b32);
        asc_min_scalar(value_vec, value_vec, offset + num_bins, mask_b32);
        asc_add_scalar(value_vec, value_vec, 1 - offset, mask_b32);

        const vector_bool mask_b8 = asc_update_mask_b8(active_bytes);
        asc_frequency_histogram_bin0(count_vec, (vector_uint8_t&)value_vec, mask_b8);
    }
    asc_storealign(acc, count_vec, mask_b16);
}

}  // namespace DispatchDedup

namespace deep_ep {
// At most ceil(1024 / 8)=128 logical records per receiver and world>=32.
// Direct counts can change layout at each 16-source host boundary. A
// 128-record partition starting mid-host needs at most nine contiguous runs.
constexpr uint32_t kLlCountReadMaxRuns = 9U;
struct LlCountReadRun {
    uint64_t recordOffset = 0, commitOffset = 0;
    uint32_t first = 0, records = 0; // first is relative to this receiver's UB
};
struct LlCountReadPlan {
    LlCountReadRun runs[kLlCountReadMaxRuns]{};
    uint32_t size = 0;
    uint32_t sourceStrideBlocks = 0; // gap in 32B units, excluding the copied block
    uint32_t expectedFlagSum = 0;
};
__aicore__ __forceinline__ bool BuildCountReadPlan(uint32_t experts, uint32_t world, uint32_t countCores,
    uint32_t destination, uint32_t first, uint32_t count, LlCountReadPlan& out,
    bool directRanks = false)
{
    out.size = 0;
    out.expectedFlagSum = 0;
    if (world < 32U || world > 256U || experts < world || experts > 1024U || experts % world ||
        !countCores || countCores > kEpServerDedupMaxVecCores || destination >= world || first > experts ||
        count > experts - first || count > 128U) return false;
    out.sourceStrideBlocks = (experts / world) * 16U - 1U;
    for (uint32_t consumed = 0; consumed < count;) {
        if (out.size == kLlCountReadMaxRuns) return false;
        const uint32_t logical = first + consumed;
        const uint32_t source = logical % world;
        const uint32_t available = directRanks
            ? kNumRanksPerHost - source % kNumRanksPerHost : world - source;
        const uint32_t length = count - consumed < available ? count - consumed : available;
        const auto frame = CountFrame(experts, world, countCores, destination, source, logical / world, directRanks);
        out.runs[out.size++] = {frame.recordOffset, frame.commitOffset, consumed, length};
        out.expectedFlagSum += length * frame.flagWords;
        consumed += length;
    }
    return true;
}
}

namespace deep_ep {
// Issue only; caller retains its existing MTE2->Vector->Scalar wait chain.
// Source-major frames are strided across source ranks. Each expert row becomes
// one DMA while the UB snapshot remains in the original expert-major order.
__aicore__ inline void ReadCountFrames(__ubuf__ uint32_t* snapshot,
                                      __gm__ uint8_t* state,
                                      const LlCountReadPlan& plan, bool body)
{
    for (uint32_t i = 0; i < plan.size; ++i) {
        const auto& run = plan.runs[i];
        const uint64_t offset = body ? run.recordOffset : run.commitOffset;
        count_ub::Load(snapshot + run.first * 8U, state + offset, 32U,
                       run.records, uint64_t(plan.sourceStrideBlocks) * 32U);
    }
}
}

namespace deep_ep {
struct LlCountAssignment {
    uint32_t first, end, remoteRank, alignedRecords, remoteRecords;
};
__aicore__ __forceinline__ uint32_t CountRemotePeer(uint32_t rank, uint32_t world, uint32_t task)
{
    // Compact remote-rank enumeration skips this source's physical host.
    const uint32_t hostFirst = rank / kNumRanksPerHost * kNumRanksPerHost;
    return task < world - kNumRanksPerHost
        ? (task < hostFirst ? task : task + kNumRanksPerHost) : world;
}
__aicore__ __forceinline__ LlCountAssignment CountAssignment(uint32_t rank, uint32_t world, uint32_t experts,
    uint32_t cores, uint32_t core, bool allCore, bool directRanks)
{
    const uint32_t countCores = allCore ? cores : cores / 2U;
    uint32_t first = 0U, end = 0U;
    if (allCore || core >= cores / 2U) {
        const uint32_t worker = allCore ? core : core - cores / 2U;
        const uint32_t base = experts / countCores, extra = experts % countCores;
        first = worker * base + (worker < extra ? worker : extra);
        end = first + base + (worker < extra ? 1U : 0U);
    }
    if (directRanks) {
        const uint32_t hostExperts = kNumRanksPerHost * (experts / world);
        const uint32_t hostFirst = (rank / kNumRanksPerHost) * hostExperts;
        const uint32_t hostEnd = hostFirst + hostExperts;
        first = first < hostFirst ? hostFirst : first;
        end = end > hostEnd ? hostEnd : end;
        if (end < first) end = first;
    }
    const uint32_t records = end > first ? (experts + countCores - 1U) / countCores : 0U;
    const uint32_t remote = directRanks ? CountRemotePeer(rank, world, core) : world;
    const uint32_t remoteRecords = remote < world ? (experts / world + 1U + 7U) / 8U * 8U : 0U;
    return {first, end, remote, (records + 7U) / 8U * 8U, remoteRecords};
}
}

namespace deep_ep {
// A non-dedup core can own flat routes without owning a token-weight slice.
// If total routes are fewer than sender cores, only the first routeCount cores
// own routes; otherwise every sender owns at least one.
__aicore__ __forceinline__ bool LlSenderHasWork(bool dedup, uint32_t tokenCount,
    uint32_t routeCount, uint32_t core)
{
    return tokenCount != 0U || (!dedup && core < routeCount);
}

template <class Words>
class LlSplitRouteWords {
    Words first_, second_;
    uint32_t half_;
public:
    __aicore__ __forceinline__ LlSplitRouteWords(Words first, Words second, uint32_t half)
        : first_(first), second_(second), half_(half) {}
    __aicore__ __forceinline__ uint32_t GetValue(uint32_t index) const {
        return index < half_ ? first_.GetValue(index) : second_.GetValue(index - half_);
    }
    __aicore__ __forceinline__ void SetValue(uint32_t index, uint32_t value) {
        if (index < half_) first_.SetValue(index, value);
        else second_.SetValue(index - half_, value);
    }
};

// Two old q-reduction scratch arrays provide at least 2*routeCount slots. Use
// a direct histogram when experts fit, otherwise a <=50%-loaded hash table.
// A packed entry holds expert+1 in its low 16 bits and count in its high 16:
// the existing admission bounds E<=1024 and B*K<2^14 make both fields exact.
// Q remains immutable and independent of packet order after this forward pass.
template <class Ids, class Words>
__aicore__ __forceinline__ void LlBuildRouteQ(Ids ids, uint32_t routeCount, uint32_t topK, uint32_t experts,
    Words firstScratch, Words secondScratch, Words q, uint32_t halfSlots)
{
    if (routeCount <= topK) {
        // Every original q call returns zero for the first token. No histogram
        // or hash state is needed when the complete input is that one token.
        for (uint32_t index = 0; index < routeCount; ++index) q.SetValue(index, 0U);
        return;
    }
    LlSplitRouteWords<Words> table(firstScratch, secondScratch, halfSlots);
    const uint32_t slots = halfSlots * 2U;
    const bool dense = experts <= slots;
    const uint32_t initialized = dense ? experts : slots;
    for (uint32_t slot = 0; slot < initialized; ++slot) table.SetValue(slot, 0U);
    for (uint32_t index = 0; index < routeCount; ++index) {
        const uint32_t expert = uint32_t(ids.GetValue(index));
        if (dense) {
            const uint32_t previous = table.GetValue(expert);
            q.SetValue(index, index < topK ? 0U : previous);
            table.SetValue(expert, previous + 1U);
            continue;
        }
        uint32_t hash = expert * 0x9e3779b9U;
        hash ^= hash >> 16U;
        uint32_t slot = hash % slots;
        const uint32_t key = expert + 1U;
        uint32_t entry = table.GetValue(slot);
        while (entry != 0U && (entry & 0xffffU) != key) {
            if (++slot == slots) slot = 0U;
            entry = table.GetValue(slot);
        }
        const uint32_t previous = entry >> 16U;
        // Preserve the original first-token special case even for repeated IDs.
        q.SetValue(index, index < topK ? 0U : previous);
        table.SetValue(slot, key | ((previous + 1U) << 16U));
    }
}
}

namespace deep_ep {
// Cache one token in the existing input queue (plain or prequantized).
// Packet tails are mutable, so a cache hit must retire the prior Vector pack
// read before Scalar changes that tail. A miss retains the original queue
// dependencies. No packet lane is used as cached storage.
class LlSenderInputReuse {
    uint32_t token_ = 0;
    bool valid_ = false;
public:
    template <class Operations>
    __aicore__ __forceinline__ void Prepare(uint32_t token, Operations& operations) {
        if (valid_ && token == token_) {
            operations.WaitPackedRead();
            return;
        }
        if (valid_) operations.ReleaseInput();
        operations.LoadInput(token);
        token_ = token;
        valid_ = true;
    }
    template <class Operations>
    __aicore__ __forceinline__ void Finish(Operations& operations) {
        if (valid_) operations.ReleaseInput();
        valid_ = false;
    }
};

// Prequantized padding plan moved to layout/ep/token.hpp.
}

namespace deep_ep {
// Rotate an existing permutation; never change route ownership or destination.
// For a scheduled entry ordinal<count. count==0 has no entries; handle it
// defensively as well. Only one modulo is needed because ordinal+offset<2*count.
__aicore__ __forceinline__ uint32_t LlSendStaggerOrdinal(uint32_t count, uint32_t phase, uint32_t ordinal)
{
    if (count <= 1U) return 0U;
    const uint32_t index = ordinal + phase % count;
    return index < count ? index : index - count;
}

// Only rotate the current token's first-hop tasks. Expert selection is owned
// by the sender's topology policy; this function never changes that selection.
__aicore__ __forceinline__ uint32_t LlFirstHopOrder(uint32_t count, uint32_t rankInGroup,
    uint32_t core, uint32_t token, uint32_t ordinal)
{
    return LlSendStaggerOrdinal(count, rankInGroup + core + token, ordinal);
}

// Each source card/core enters the existing target-group permutation at a
// different point. Preserve the large-domain half-interleaving policy.
__aicore__ __forceinline__ uint32_t LlDirectGroupOrder(uint32_t groups, uint32_t ownGroup,
    uint32_t rankInGroup, uint32_t core, uint32_t ordinal,
    bool remoteFirst = false)
{
    if (remoteFirst && groups > 1U) {
        // Post RDMA before local copies so a singleton remote DIRECT is not
        // delayed by a long local group. Rotate only the remote permutation.
        // Remove ownGroup from the original order; keep its half-interleaving.
        if (ordinal == groups - 1U) return ownGroup;
        const uint32_t split = groups / 2U;
        const uint32_t ownLength = ownGroup < split ? split : groups - split;
        const uint32_t ownOrdinal = groups < 16U ? 0U :
            (ownLength > split ? groups - 1U : 2U * (ownLength - 1U));
        uint32_t remoteOrdinal = LlSendStaggerOrdinal(groups - 1U, rankInGroup + core, ordinal);
        if (remoteOrdinal >= ownOrdinal) ++remoteOrdinal;
        return EpServerOrder::ServerDedupDispatchDirectOrder(groups, ownGroup, remoteOrdinal);
    }
    return EpServerOrder::ServerDedupDispatchDirectOrder(
        groups, ownGroup, LlSendStaggerOrdinal(groups, rankInGroup + core, ordinal));
}
}  // namespace DispatchDedup

namespace deep_ep {
// Each target physical host contains four consecutive
// full-mesh groups of four ranks. Count actual expert deliveries, not distinct
// cards: each relay carries one equal-sized token for its target expert.
// Two scalar words suffice for K<=32; no UB array, route rescan or cross-core work.
class LlFullMeshFirstHop {
    uint32_t counts_ = 0U;
    uint32_t first_ = 0U;
public:
    __aicore__ __forceinline__ void Add(uint32_t destinationRank, uint32_t k) {
        const uint32_t shift = ((destinationRank & 15U) >> 2U) * 8U;
        counts_ += 1U << shift;
        if (((first_ >> shift) & 255U) == 0U) first_ |= (k + 1U) << shift;
    }
    __aicore__ __forceinline__ uint32_t FirstK() const {
        uint32_t bestCount = 0U, bestK = 32U;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t shift = group * 8U;
            const uint32_t count = (counts_ >> shift) & 255U;
            if (!count) continue;
            const uint32_t k = ((first_ >> shift) & 255U) - 1U;
            if (count > bestCount || (count == bestCount && k < bestK)) {
                bestCount = count;
                bestK = k;
            }
        }
        return bestCount ? bestK : 0U;
    }
};

// Classify supplies a mask for exactly one target host. Select independently
// for each mask; a single token-wide firstK is invalid with multiple hosts.
// Masks are disjoint, so each grouped route is loaded at most once here.
__aicore__ __forceinline__ uint32_t LlRelayMemberMask(uint32_t mask, uint32_t firstK) {
    return mask & ~(1U << firstK);
}
}


namespace deep_ep {
using namespace count_ub;

// -------------------------------------------------------------------------------
// Static UB maps (DeepEP-Ascend shape):
//   DispatchUBLayout               pure state/scratch, typed arrays, no getters;
//   DispatchUBLayoutWithBuffer     adds the count-frame prefix and the buffers.
// Every region is sized from the per-rank capacity, so lanes, stages and the
// admitted footprint are compile-time facts of this specialization.
// ---------------------------------------------------------------------------
template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenElems, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kNumRanksPerGroup, bool kDedup, bool kRdma,
          bool kPrequantized>
struct alignas(kNumUbAlignmentBytes) DispatchUBLayout {
    static constexpr bool kDedupMode = kDedup;
    // Shape-derived axes. Only num_tokens stays a runtime argument.
    static constexpr EpServerDedupInputMode kInputMode =
        kPrequantized ? EpServerDedupInputMode::PrequantizedFp8Packs : EpServerDedupInputMode::Plain16;
    static constexpr EpServerDedupDataLayout kData = DispatchServerDedupDataLayout(
        kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk, kNumExperts, kInputMode);
    static constexpr EpServerDedupCountLayout kCountLayout =
        DispatchServerDedupCountLayout(kNumExperts, kNumVecCores);

    static constexpr int kNumMaxRoutes = int(kNumMaxTokensPerRank * kNumTopk);
    static constexpr int kNumSendCores = int(kNumVecCores / 2U);
    static constexpr int kNumRecvCores = int(kNumVecCores / 2U);
    static constexpr int kNumLocalExperts = int(kNumExperts / kNumRanks);
    static constexpr int kNumGroups = int(kNumRanks / kNumRanksPerGroup);
    static constexpr int kNumPerCoreTokens = math::ceil_div(int(kNumMaxTokensPerRank), kNumSendCores);
    static constexpr int kNumCapacityRows =
        int(kNumRanks) * int(kNumMaxTokensPerRank) * math::min(int(kNumTopk), kNumLocalExperts);
    // Count participation is shape-derived: small batches use every core.
    static constexpr bool kUseAllCoreCount =
        kNumMaxTokensPerRank < kEpServerDedupAllCoreCountBsThreshold;
    static constexpr int kNumCountCores = kUseAllCoreCount ? int(kNumVecCores) : int(kNumVecCores / 2U);
    static constexpr int kNumCountRecords = math::ceil_div(int(kNumExperts), kNumCountCores);
    static constexpr bool kPackedFrames = kRdma && int(kNumRanks) > int(kNumRanksPerHost);
    static constexpr bool kDirectRanks = kRdma && int(kNumRanks) > int(kNumRanksPerHost);
    static constexpr int kNumRemoteRecords = kDirectRanks ? math::align(kNumLocalExperts + 1, 8) : 0;
    // Local count records are the only region MTE3 still reads into payload.
    static constexpr int kCountTxBytes = kPackedFrames
        ? int(CountWireBytes(uint32_t(kNumCountRecords)))
        : math::align(kNumCountRecords, 8) * kNumUbAlignmentBytes;
    static constexpr int kNumRouteBytes = math::align(kNumMaxRoutes * 4, 256);
    static constexpr int kNumCountWorkBytes =
        math::max(kNumRouteBytes, kNumRemoteRecords * kNumUbAlignmentBytes);
    static constexpr int kNumServerBytes = math::max(math::align(kNumGroups * 4, 32), 64);
    static constexpr int kNumFirstHopMasks = math::max(0, math::min(kNumGroups - 1, int(kNumTopk) / 2));
    static constexpr int kNumBlocks = int(kData.blockCount);
    // Receiver tables follow the admitted row capacity, not the arrival count.
    static constexpr int kNumTaskRows = math::ceil_div(kNumCapacityRows, kNumRecvCores);
    static constexpr int kNumPrefixBytes = math::align(int(kNumExperts) * 4, 32);
    static constexpr int kNumTaskBytes = math::align(kNumTaskRows * 4, 32);
    static constexpr int kNumCountSnapshotBytes = math::align(kNumCountRecords * 32, 256) + 256;
    static constexpr int kNumSumSlotBytes = math::align(kNumRecvCores * 32, 256);
    static constexpr int kNumReadyBytes = math::align(kNumRecvCores * 32 + 256, 256);
    static constexpr int kNumProbeBatch = 16;
    static constexpr int kNumFlagsBytes = kNumProbeBatch * kNumBlocks * 32 + 256;
    static constexpr int kNumSumsBytes = kNumProbeBatch * 32;

    // Count stage: routes stay shared with the sender's route image. Every
    // sub-buffer starts on a UB alignment boundary so MTE and vector accesses
    // keep their 32B granularity.
    struct alignas(kNumUbAlignmentBytes) CountState {
        alignas(kNumUbAlignmentBytes) uint16_t histogram[kNumRouteBytes / 2];
        alignas(kNumUbAlignmentBytes) uint8_t  work[kNumCountWorkBytes];
        alignas(kNumUbAlignmentBytes) uint32_t scalar[32];
        alignas(kNumUbAlignmentBytes) uint32_t snapshot[kNumCountSnapshotBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t dense[int(kNumExperts)];
        alignas(kNumUbAlignmentBytes) uint32_t sum_slot[kNumSumSlotBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t sum_snapshot[kNumSumSlotBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t sum_dense[kNumRecvCores];
        alignas(kNumUbAlignmentBytes) uint32_t ready[kNumReadyBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t clean[kNumRecvCores * 8];
        alignas(kNumUbAlignmentBytes) int32_t  prefix[int(kNumExperts)];
        alignas(kNumUbAlignmentBytes) int64_t  expert_counts[kNumLocalExperts];
    };
    // Sender stage. Token-slice arrays are capacity-sized (per-core maximum).
    struct alignas(kNumUbAlignmentBytes) SenderState {
        alignas(kNumUbAlignmentBytes) uint32_t qdst[kNumMaxRoutes];
        alignas(kNumUbAlignmentBytes) float    route_weights[kNumPerCoreTokens * int(kNumTopk)];
        alignas(kNumUbAlignmentBytes) float    source_weights[kNumPerCoreTokens * int(kNumTopk)];
        alignas(kNumUbAlignmentBytes) uint32_t source_masks[kNumPerCoreTokens];
        alignas(kNumUbAlignmentBytes) uint32_t direct_tasks[kNumPerCoreTokens * int(kNumTopk) * 4];
        alignas(kNumUbAlignmentBytes) uint32_t server_state[kNumServerBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t firsthop_masks[kNumFirstHopMasks];
    };
    // Receiver stage. Probe tables live in the receiver buffer overlay so a
    // sender core never reserves them.
    struct alignas(kNumUbAlignmentBytes) ReceiverState {
        alignas(kNumUbAlignmentBytes) int32_t  prefix[kNumPrefixBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t task_pos[kNumTaskBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t pending_idx[kNumTaskBytes / 4];
    };
    struct alignas(kNumUbAlignmentBytes) ReceiverProbe {
        alignas(kNumUbAlignmentBytes) float    flags[kNumFlagsBytes / 4];
        alignas(kNumUbAlignmentBytes) float    sums[kNumSumsBytes / 4];
        alignas(kNumUbAlignmentBytes) uint32_t snapshot[kNumProbeBatch];
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        alignas(kNumUbAlignmentBytes) uint8_t  trace[profile::kPacketTraceCapacity * 32U];
#endif
    };

    int32_t routes[kNumMaxRoutes];
    // Sender and receiver run on disjoint cores; overlay their scratch, and the
    // count scratch, so the admitted footprint keeps the previous lane count.
    alignas(kNumUbAlignmentBytes) union {
        CountState count;
        SenderState sender;
        ReceiverState receiver;
    } phases;
};

template <uint32_t kRanks, uint32_t kTokens, uint32_t kHidden, uint32_t kTopk, uint32_t kExperts,
          uint32_t kVecCores, uint32_t kUbBytes, uint32_t kRanksPerGroup, bool kPrequantized, bool kDedup,
          bool kRdma>
struct alignas(kNumUbAlignmentBytes) DispatchUBLayoutWithBuffer {
    // Axes re-exported for the phase classes, which template on this map.
    static constexpr uint32_t kNumRanks = kRanks;
    static constexpr uint32_t kNumMaxTokensPerRank = kTokens;
    static constexpr uint32_t kNumHiddenElems = kHidden;
    static constexpr uint32_t kNumTopk = kTopk;
    static constexpr uint32_t kNumExperts = kExperts;
    static constexpr uint32_t kNumVecCores = kVecCores;
    static constexpr uint32_t kNumUbBytes = kUbBytes;
    static constexpr uint32_t kNumRanksPerGroup = kRanksPerGroup;
    static constexpr bool kPrequantizedMode = kPrequantized;
    static constexpr bool kDedupMode = kDedup;
    static constexpr bool kRdmaMode = kRdma;
    using layout_t = DispatchUBLayout<kRanks, kTokens, kHidden, kTopk, kExperts, kVecCores, kRanksPerGroup,
                                      kDedup, kRdma, kPrequantized>;
    static constexpr int kCountTxBytes = layout_t::kCountTxBytes;
    static constexpr int kPayloadBytes = int(layout_t::kData.packedPayloadBytes);
    static constexpr int kTokenStride = int(layout_t::kData.tokenStride);
    // Receiver lanes carry the metadata tail right after the packet.
    static constexpr int kLaneStride = kTokenStride + 64;
    // Hidden prefetch depth. The optional N-stage ring (plan P5) can raise this.
    static constexpr int kNumInputStages = 2;
    static constexpr int kRdmaStageSlots = kRdma ? int(rdma::kCoreBytes) / kTokenStride : 0;
    static constexpr int kRdmaWorkspaceBytes = kRdma
        ? (kRdmaStageSlots + 1) * int(comm::DispatchRdmaBatch::kWorkspaceBytes) : 0;
    // Role-exclusive buffer overlays: sender cores own the payload stages and
    // RDMA workspace, receiver cores own the probe tables. Keeping them apart
    // preserves the sender lane count admitted by the previous implementation.
    static constexpr int kProbeBytes = layout_t::kNumFlagsBytes + layout_t::kNumSumsBytes +
        layout_t::kNumProbeBatch * int(sizeof(uint32_t))
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        + int(profile::kPacketTraceCapacity) * 32
#endif
        ;
    static constexpr int kSenderStageBytes = kNumInputStages * kPayloadBytes + kRdmaWorkspaceBytes;
    static constexpr int kSenderFixedBytes = kCountTxBytes + int(sizeof(layout_t)) + kSenderStageBytes;
    static constexpr int kReceiverFixedBytes = kCountTxBytes + int(sizeof(layout_t)) + kProbeBytes;
    static constexpr int kSenderLanes =
        math::min<int>(8, (int(kNumUbBytes) - kSenderFixedBytes) / kTokenStride);
    static constexpr int kReceiverLanes =
        math::min<int>(8, (int(kNumUbBytes) - kReceiverFixedBytes) / kLaneStride);
    EP_STATIC_ASSERT(kSenderLanes > 0, "Insufficient UB for dispatch sender lanes");
    EP_STATIC_ASSERT(kReceiverLanes > 0, "Insufficient UB for dispatch receiver lanes");
    EP_STATIC_ASSERT(!kRdma || kTokenStride <= int(rdma::kCoreBytes), "Packet exceeds RDMA staging core bytes");
    EP_STATIC_ASSERT(kSenderFixedBytes + kSenderLanes * kTokenStride <= int(kNumUbBytes),
                     "Dispatch sender UB exceeds capacity");
    EP_STATIC_ASSERT(kReceiverFixedBytes + kReceiverLanes * kLaneStride <= int(kNumUbBytes),
                     "Dispatch receiver UB exceeds capacity");

    alignas(kNumUbAlignmentBytes) uint8_t count_tx[kCountTxBytes];
    alignas(kNumUbAlignmentBytes) layout_t workspace;
    // Both roles never run on the same core, so their wire and scratch overlay.
    alignas(kNumUbAlignmentBytes) union {
        struct {
            uint8_t wire[kSenderLanes * kTokenStride];
            uint8_t hidden[kNumInputStages][kPayloadBytes];
            uint8_t rdma_ws[kRdmaWorkspaceBytes];
        } sender;
        struct {
            uint8_t wire[kReceiverLanes * kLaneStride];
            alignas(kNumUbAlignmentBytes) typename layout_t::ReceiverProbe probe;
        } receiver;
    } buffers;
};

// 中文审计：这里包含跨rank计数交换，不属于four_metadata测得的本地路由阶段。
// 修改计数协议需联查dispatch_server_dedup_kernel.h中的阶段顺序及Receiver使用的前缀布局。
// Adapted from Memory's CalAndSendCntByRankAndExpert / WaitDispatch / cumsum
// chain. Counts cover the complete original routing tensor, including relay routes.
// P6: one DispatchContext, three stateless phases. No Init/Process lifetime:
//   count → send → recv run in dispatch_body order on the shared UB map.
struct DispatchContext {
    TransportView view{};
    __gm__ uint8_t* x = nullptr;
    __gm__ uint8_t* expert_ids = nullptr;
    __gm__ uint8_t* expert_scales = nullptr;
    __gm__ uint8_t* expand_x = nullptr;
    __gm__ uint8_t* dynamic_scales = nullptr;
    __gm__ uint8_t* expert_token_nums = nullptr;
    __gm__ uint8_t* send_counts = nullptr;
    __gm__ uint8_t* token_type = nullptr;
    __gm__ uint8_t* destination_index = nullptr;
    __gm__ uint8_t* relay_read_index = nullptr;
    __gm__ uint8_t* source_mask = nullptr;
    __gm__ uint8_t* source_weights = nullptr;
    __gm__ uint8_t* input_scales = nullptr;
    profile::DeviceProfile* profile = nullptr;
    comm::CountRetirement retirement;
    uint32_t rank_idx = 0U, num_tokens = 0U;
    uint64_t window_offset = 0U;
    CoreContext core{};
};

// Count-only runtime state: per-core split plus the local control bases.
// All count arrays are members of the static UB map (phases.count).
struct CountPhaseState {
    uint64_t state_offset = 0U;
    uint64_t original_overflow_mode = 0U;
    uint32_t start_status_index = 0U, end_status_index = 0U, status_per_core = 0U;
    uint32_t active_cum_sum_cores = 0U;
    __gm__ uint8_t* state_gm = nullptr;
    __gm__ uint8_t* recv_cnt_workspace_gm = nullptr;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    bool count_remote = false;
#endif
};

// 摘要：绑定本轮控制区，保存并调整 CTRL[60]，供整数位模式的 FP32 归约使用。
template <class Layout>
__aicore__ inline void CountStageBegin(DispatchContext& ctx, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    ctx.retirement.Init(ctx.view);
    st.state_offset = ctx.window_offset + layout_t::kCountLayout.cntOffset;
    // 计数记录与 cumsum 槽位都由本机 L2 可达的写者更新（远端 RDMA 与同机 MTE3），
    // 读侧保持默认缓存视图才能立即看到新值；绕过 L2 会读到尚未回写的 HBM，
    // 表现为 ready 轮询多等十几微秒。
    __gm__ uint8_t* local = ctx.view.peer_ptr(ctx.rank_idx) + st.state_offset;
    st.state_gm = local;
    st.recv_cnt_workspace_gm = local + layout_t::kCountLayout.prefixOffset;
    // Memory's count reductions add integer bit patterns as FP32 subnormals.
    st.original_overflow_mode = uint64_t(asc_get_ctrl()) & (1ULL << 60U);
    asc_set_ctrl(uint64_t(asc_get_ctrl()) & ~(1ULL << 60U));
}

// 摘要：将连续区间均分到各核，余数优先分配给前面的核，返回半开区间及长度。
__aicore__ inline void CountSplitToCore(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start,
                                        uint32_t& end, uint32_t& count)
{
    const uint32_t base = total / cores;
    const uint32_t remainder = total % cores;
    count = base + (core < remainder ? 1U : 0U);
    start = core * base + (core < remainder ? core : remainder);
    end = start + count;
}

// 本机专家区间与跨机远端 rank 共用：清零对齐槽、批量打 ready，再用一次硬件直方图回填 count。
// 跨机帧同时初始化尾 ready 记录；本机 wire 初始化所有数据及 Flag 块。
template <class Layout>
__aicore__ inline void CountSlots(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st,
                                  __ubuf__ uint32_t* records, uint32_t first, uint32_t bins,
                                  bool wire, bool remoteTail)
{
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    profile::DeviceProfile* const profile_ = ctx.profile;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (st.count_remote) profile_->CountBoundary<12>();
        else profile_->CountBoundary<13>();
        profile_->CompleteCountCall(st.count_remote, bins);
    }
    if (profile_ && profile_->CountNeedsCalibration()) {
        const uint64_t c0 = profile_->CountClock(), c1 = profile_->CountClock();
        uint64_t emptyWait = 0U;
        if constexpr (profile::kCountComputeMode == 2U) {
            // MTE2_V submits a Vector dependency. Retire it before measuring an
            // idle V_S so the calibration cannot include the initial route DMA.
            Sync<PIPE_V, PIPE_S>();
            const uint64_t e0 = profile_->Clock();
            Sync<PIPE_V, PIPE_S>();
            emptyWait = profile_->Clock() - e0;
        }
        profile_->CountCalibration(c1 - c0, emptyWait);
    }
    const uint64_t t0 = profile_ ? profile_->CountClock() : 0U;
#endif
    const uint32_t slots = (bins + (remoteTail ? 1U : 0U) + 7U) / 8U * 8U;
    Fill(records, 0U, wire ? CountWireBytes(bins) / sizeof(int32_t) : slots * UB_ALIGN_DATA_COUNT);
    asc_sync_pipe(PIPE_V);
    RecordReady(records, 0x3f800000U, slots);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if constexpr (profile::kCountComputeMode == 2U) {
        if (profile_ && profile_->Enabled()) Sync<PIPE_V, PIPE_S>();
    }
    const uint64_t t1 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (st.count_remote) profile_->CountBoundary<3>();
        else profile_->CountBoundary<7>();
    }
#endif
    histogram_accumulate_u16(ub->workspace.routes, ub->workspace.phases.count.histogram,
                             static_cast<int>(ctx.num_tokens * uint32_t(Layout::kNumTopk)), static_cast<int>(first),
                             static_cast<int>(bins));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (st.count_remote) profile_->CountBoundary<4>();
        else profile_->CountBoundary<8>();
    }
#endif
    Sync<PIPE_V, PIPE_S>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t3 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (st.count_remote) profile_->CountBoundary<5>();
        else profile_->CountBoundary<9>();
    }
#endif
    for (uint32_t i = 0; i < bins; ++i)
        records[i * UB_ALIGN_DATA_COUNT + 1U] = uint32_t(ub->workspace.phases.count.histogram[i + 1U]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (st.count_remote) profile_->CountBoundary<6>();
        else profile_->CountBoundary<10>();
        profile_->CountSections(st.count_remote, bins, t0, t1, t2, t3, profile_->CountClock());
    }
#endif
    (void)st;
}

template <class Layout>
__aicore__ inline void CountPackLocalCountWire(DispatchContext& ctx, __ubuf__ Layout* ub,
                                               __ubuf__ uint32_t* records, uint32_t count)
{
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    profile::DeviceProfile* const profile_ = ctx.profile;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t wirePackBegin = profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    // Reverse order preserves dense sources while inserting one 32B flag per
    // fifteen records. Every destination reads this same image without edits.
    for (uint32_t i = count; i > kLlCountWireRecords;) {
        --i;
        const uint32_t value = records[i * UB_ALIGN_DATA_COUNT + 1U];
        const uint32_t target = CountWireRecordOffset(i) / sizeof(int32_t);
        records[target] = 0x3f800000U;
        records[target + 1U] = value;
    }
    for (uint32_t block = 0; block < CountWireBytes(count); block += kLlCountWireBlockBytes)
        for (uint32_t word = 0; word < UB_ALIGN_DATA_COUNT; ++word)
            records[(block + kLlCountWireFlagOffset) / sizeof(int32_t) + word] = 0x3f800000U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        profile_->CountBoundary<16, true>();
        profile_->CompletePublishAdd<17>(1U);
        profile_->CountWirePack(profile_->PublishClock() - wirePackBegin);
    }
#endif
    (void)ctx; (void)ub;
}

// 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
template <class Layout>
__aicore__ inline void CountSendExpertCntRange(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st,
                                               uint32_t start, uint32_t end)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t epWorldSize_ = Layout::kNumRanks;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t moeExpertNumPerRank_ = uint32_t(layout_t::kNumLocalExperts);
    constexpr uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    constexpr bool useAllCoreCount_ = layout_t::kUseAllCoreCount;
    constexpr bool packedFrames_ = layout_t::kPackedFrames;
    constexpr bool directRanks_ = layout_t::kDirectRanks;
    const TransportView& view_ = ctx.view;
    profile::DeviceProfile* const profile_ = ctx.profile;
    const uint32_t epRankId_ = view_.rank;
    const uint32_t coreNum_ = ctx.core.count;
    const uint64_t stateOffset_ = st.state_offset;
    __ubuf__ uint32_t* const records_ = reinterpret_cast<__ubuf__ uint32_t*>(ub->count_tx);
    __ubuf__ uint32_t* const work_ = reinterpret_cast<__ubuf__ uint32_t*>(ub->workspace.phases.count.work);

    for (uint32_t expert = start; expert < end;) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t planBegin = profile_ ? profile_->PublishClock() : 0U;
        if (profile_) profile_->CountBoundary<13>();
#endif
        const uint32_t dst = expert / moeExpertNumPerRank_;
        const uint32_t localExpert = expert % moeExpertNumPerRank_;
        const uint32_t count =
            end - expert < moeExpertNumPerRank_ - localExpert ? end - expert : moeExpertNumPerRank_ - localExpert;
        const uint64_t offset = (uint64_t(localExpert) * epWorldSize_ + epRankId_) * kEpServerDedupCountGmStride;
        if (packedFrames_) {
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, dst, epRankId_, localExpert, directRanks_);
            if (frame.firstExpert != localExpert || frame.records != count) { trap(); return; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t planEnd = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<13, true>();
#endif
            comm::CountFrame(view_, dst, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(records_) + frame.sourceOffset, frame.wireBytes, count
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->PublishAdd<13>(planEnd-planBegin);
#endif
            expert += count;
            continue;
        }
        comm::Records(view_, dst, stateOffset_ + offset,
            reinterpret_cast<__ubuf__ uint8_t*>(records_) + (expert - start) * kNumUbAlignmentBytes,
            count, epWorldSize_ * kEpServerDedupCountGmStride, true,
            reinterpret_cast<__ubuf__ uint8_t*>(work_));
        expert += count;
    }
}

// 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
template <class Layout>
__aicore__ inline void CountSendCnt(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t epWorldSize_ = Layout::kNumRanks;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t moeExpertNumPerRank_ = uint32_t(layout_t::kNumLocalExperts);
    constexpr uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    constexpr bool useAllCoreCount_ = layout_t::kUseAllCoreCount;
    constexpr bool packedFrames_ = layout_t::kPackedFrames;
    constexpr bool directRanks_ = layout_t::kDirectRanks;
    const TransportView& view_ = ctx.view;
    profile::DeviceProfile* const profile_ = ctx.profile;
    const uint32_t epRankId_ = view_.rank;
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t coreNum_ = ctx.core.count;
    const uint32_t expertIdsCnt_ = ctx.num_tokens * uint32_t(Layout::kNumTopk);
    const uint64_t stateOffset_ = st.state_offset;
    const uint64_t countWindow_ = ctx.window_offset;
    comm::CountRetirement& retirement_ = ctx.retirement;
    __ubuf__ uint32_t* const records_ = reinterpret_cast<__ubuf__ uint32_t*>(ub->count_tx);
    __ubuf__ int32_t* const routes_ = ub->workspace.routes;
    __ubuf__ uint16_t* const histogram_ = ub->workspace.phases.count.histogram;
    __ubuf__ uint32_t* const work_ = reinterpret_cast<__ubuf__ uint32_t*>(ub->workspace.phases.count.work);

#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t routeBegin = profile_ ? profile_->CountClock() : 0U;
#endif
    // Same-host work keeps the original expert split. Each physical core owns
    // remote rank tasks core, core+coreNum, ... independent of the BS threshold.
    const uint32_t cores = useAllCoreCount_ ? coreNum_ : aivUsedCumSum_;
    const auto assignment = CountAssignment(epRankId_, epWorldSize_, rscvStatusNum_,
                                            coreNum_, aivId_, useAllCoreCount_, directRanks_);
    const uint32_t start = assignment.first, end = assignment.end, count = end - start;
    const bool remoteOwner = assignment.remoteRank < epWorldSize_;
    // Only local records stay reserved into payload. MTE3_S retires the remote
    // UB frame copy; its isolated GM source survives independently until CQ retirement.
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<1>();
#endif
    if (count == 0U && !remoteOwner) return;

    // 3. 完整 bs×topK 路由搬入静态 routes，直方图/工作区来自静态 map。
    Load(routes_, ctx.expert_ids, expertIdsCnt_ * sizeof(int32_t));
    Sync<PIPE_MTE2, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<2>();
    if (profile_) profile_->CountRoute(profile_->CountClock() - routeBegin);
#endif
    if (remoteOwner) {
        auto remoteFrame = work_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint32_t firstFrameIssued = 0U;
#endif
        for (uint32_t task = aivId_; task < epWorldSize_ - kNumRanksPerHost; task += coreNum_) {
            const uint32_t destination = CountRemotePeer(epRankId_, epWorldSize_, task);
            const uint32_t first = destination * moeExpertNumPerRank_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            st.count_remote = true;
#endif
            CountSlots(ctx, ub, st, remoteFrame, first, moeExpertNumPerRank_, false, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->CountBoundary<12>();
            const uint64_t publishBegin = profile_ ? profile_->CountClock() : 0U;
            const uint64_t p0 = profile_ ? profile_->PublishClock() : 0U;
#endif
            Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p1 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<2, true>();
#endif
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_, cores,
                                          destination, epRankId_, 0U, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p2 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<3, true>();
#endif
            comm::DirectCountFrame(view_, destination, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(remoteFrame), moeExpertNumPerRank_,
                reinterpret_cast<__ubuf__ uint8_t*>(histogram_), countWindow_
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p3 = profile_ ? profile_->PublishClock() : 0U;
#endif
            retirement_.Posted(task);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<9, true>();
                profile_->CompletePublishAdd<1>(1U);
                const uint64_t p4 = profile_->PublishClock();
                profile_->PublishAdd<1>(1U); profile_->PublishAdd<2>(p1-p0);
                profile_->PublishAdd<3>(p2-p1); profile_->PublishAdd<9>(p4-p3);
                profile_->CountPublish(true, profile_->CountClock() - publishBegin);
                if (!firstFrameIssued) { profile_->Mark<44>(); firstFrameIssued = 1U; }  // w44: send_count_first_frame_issued
                // Only the last frame of this core's stride gets a clock so the
                // per-frame post cost is not inflated by repeated scalar stores.
                if (task + coreNum_ >= epWorldSize_ - kNumRanksPerHost) profile_->Mark<45>();  // w45: send_count_last_frame_issued
            }
#endif
            // Source GM slots are disjoint. Acquire the SDK's retired UB scratch
            // before the next Vector pass; no network completion wait here.
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t postSVBegin = profile_ ? profile_->CountClock() : 0U;
            if (profile_) profile_->CountBoundary<12>();
#endif
            Sync<PIPE_S, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<11>();
                profile_->CountPostSV(profile_->CountClock() - postSVBegin);
            }
#endif
        }
    }

    // 4. 本机专家区间统一走硬件直方图；零计数专家也发布 ready=1 和 count=0，避免对端永久等待。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    st.count_remote = false;
#endif
    if (count) {
        CountSlots(ctx, ub, st, records_, start, count, packedFrames_, false);
        if (packedFrames_) CountPackLocalCountWire(ctx, ub, records_, count);
    }

    // 5. Scalar 填好的记录通过 MTE3 写到目标专家所属 rank。
    // 此处仅下发，不全量等待：records_ 保持不变，而筛选临时 UB 可以交给后续阶段。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t localPublishBegin = profile_ ? profile_->CountClock() : 0U;
    const uint64_t l0 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t l1 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<12, true>();
#endif
    if (count) CountSendExpertCntRange(ctx, ub, st, start, end);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (count && profile_) {
        profile_->CompletePublishAdd<10>(1U);
        profile_->PublishAdd<10>(1U); profile_->PublishAdd<12>(l1-l0);
        profile_->CountPublish(false, profile_->CountClock() - localPublishBegin);
    }
#endif
    // Only records_ is still read by MTE3. V_S retired the shared scratch.
    // Keep CTRL[60] cleared for the subsequent sender phase.
}

// 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
template <class Layout>
__aicore__ inline void CountWaitDispatchClearStatus(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t epWorldSize_ = Layout::kNumRanks;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    constexpr bool useAllCoreCount_ = layout_t::kUseAllCoreCount;
    constexpr bool packedFrames_ = layout_t::kPackedFrames;
    constexpr bool directRanks_ = layout_t::kDirectRanks;
    const uint32_t epRankId_ = ctx.view.rank;
    const uint32_t coreNum_ = ctx.core.count;
    const uint32_t startStatusIndex_ = st.start_status_index, endStatusIndex_ = st.end_status_index;
    const uint32_t recStatusNumPerCore_ = st.status_per_core;
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __ubuf__ uint32_t* const snapshot_ = ub->workspace.phases.count.snapshot;

    if (packedFrames_) {
        // Called only after GetCumSum has observed every receiver's snapshot.
        // Exactly the owner of a frame's first logical record clears its wire.
        auto zero = snapshot_;
        Fill(zero, 0U, kLlCountWireBlockBytes / sizeof(float));
        Sync<PIPE_V, PIPE_MTE3>();
        for (uint32_t logical = startStatusIndex_; logical < endStatusIndex_; ++logical) {
            const uint32_t expert = logical / epWorldSize_, source = logical % epWorldSize_;
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_, source, expert, directRanks_);
            // Remote frames retain dense-record cleanup. Local copies include
            // neighbouring source records, so reclaim their entire wire too:
            // a later shape may put a remote commit at any old record boundary.
            if (frame.flagWords == 1U)
                Store(stateGM_ + frame.recordOffset, zero, 32U);
            if (expert == frame.firstExpert) {
                if (frame.flagWords == 1U)
                    Store(stateGM_ + frame.commitOffset, zero, 32U);
                else
                    for (uint32_t block = 0; block < frame.wireBytes; block += kLlCountWireBlockBytes)
                        Store(stateGM_ + frame.frameOffset + block, zero, kLlCountWireBlockBytes);
            }
        }
        return;
    }
    Sync<PIPE_S, PIPE_V>();
    RecordReady(snapshot_, 0U, math::align(recStatusNumPerCore_, 8U));
    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride, snapshot_,
          32U, recStatusNumPerCore_, kEpServerDedupCountGmStride - 32U);
    // The snapshot is no longer modified after this issue. Its DMA can overlap
    // cumsum; the caller retires all count-stage writes before receiver Reset.
}

// 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
template <class Layout>
__aicore__ inline void CountGatherSumRecvCnt(DispatchContext&, __ubuf__ Layout* ub, CountPhaseState& st);

// 摘要：将本核片段总数和就绪标记写入本 rank 的紧凑记分板；空片段也必须发布。
template <class Layout>
__aicore__ inline void CountPublishRecvCntSum(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st,
                                              uint32_t sumOfRecvCnt)
{
    constexpr uint32_t UB_ALIGN = uint32_t(kNumUbAlignmentBytes);
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    using layout_t = typename Layout::layout_t;
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t aivUsedAllToAll_ = uint32_t(layout_t::kNumRecvCores);
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __ubuf__ uint32_t* const sumSlot_ = ub->workspace.phases.count.sum_slot;

    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    // Compact scoreboard: one writer-owned 32B slot per receiver core, so
    // consumers read a single contiguous burst instead of a transposed row.
    for (uint32_t i = 0; i < UB_ALIGN_DATA_COUNT; ++i) sumSlot_[i] = 0U;
    sumSlot_[0] = sumOfRecvCnt;
    sumSlot_[1] = 0x3f800000U;
    Sync<PIPE_S, PIPE_MTE3>();
    Store(stateGM_ + layout_t::kCountLayout.cumsumOffset + newAivId * UB_ALIGN, sumSlot_, UB_ALIGN);
    // sumSlot_ stays immutable until CalRecvAndSetFlag reuses it. Polling
    // incoming sums must not wait here for this core's outgoing cnt.
}

// 摘要：等待所有接收核总数就绪，求本核之前各片段之和；槽位回收在后续屏障后执行。
template <class Layout>
__aicore__ inline uint32_t CountGetCumSum(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st,
                                         uint32_t newAivId)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t UB_ALIGN = uint32_t(kNumUbAlignmentBytes);
    const uint32_t activeCumSumCores_ = st.active_cum_sum_cores;
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __ubuf__ uint32_t* const scalar_ = ub->workspace.phases.count.scalar;
    __ubuf__ uint32_t* const sumSnapshot_ = ub->workspace.phases.count.sum_snapshot;
    __ubuf__ uint32_t* const sumDense_ = ub->workspace.phases.count.sum_dense;

    uint32_t offset = 0;
    // Subtotals are contiguous 32B slots, so one burst covers every active core.
    Sync<PIPE_S, PIPE_V>();

    while (true) {
        Load(sumSnapshot_, stateGM_ + layout_t::kCountLayout.cumsumOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, activeCumSumCores_, 1);
        asc_sync_pipe(PIPE_V);
        Sum<true>(scalar_ + 8U, sumDense_, activeCumSumCores_);
        Sync<PIPE_V, PIPE_S>();
        if (*reinterpret_cast<__ubuf__ float*>(scalar_ + 8U) == float(activeCumSumCores_)) break;
    }
    if (newAivId != 0U) {
        Sync<PIPE_S, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, newAivId, 0);
        asc_sync_pipe(PIPE_V);
        Sum<false>(scalar_ + 8U, sumDense_, newAivId);
        Sync<PIPE_V, PIPE_S>();
        offset = scalar_[8U];
    }

    // The subtotal slots are reclaimed after WaitCumSumFlag, when every receiver
    // has already consumed them; no per-consumer column clear remains here.
    return offset;
}

// 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
template <class Layout>
__aicore__ inline void CountWaitCumSumFlag(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t UB_ALIGN = uint32_t(kNumUbAlignmentBytes);
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t aivUsedAllToAll_ = uint32_t(layout_t::kNumRecvCores);
    const uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    const uint32_t activeCumSumCores_ = st.active_cum_sum_cores;
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __ubuf__ uint32_t* const scalar_ = ub->workspace.phases.count.scalar;
    __ubuf__ uint32_t* const ready_ = ub->workspace.phases.count.ready;
    __ubuf__ uint32_t* const clean_ = ub->workspace.phases.count.clean;

    const uint32_t targetFlag = activeCumSumCores_ * UB_ALIGN_DATA_COUNT;
    const uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    uint32_t cumSumFlagOffset =
        layout_t::kCountLayout.cumsumFlagOffset + newAivId * aivUsedCumSum_ * UB_ALIGN;

    while (true) {
        Load(ready_, stateGM_ + cumSumFlagOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        Sum<false>(scalar_, ready_, targetFlag);
        Sync<PIPE_V, PIPE_S>();
        if (scalar_[0] == targetFlag) break;
    }
    Fill(clean_, 0U, targetFlag);
    Sync<PIPE_S, PIPE_MTE3>();

    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + cumSumFlagOffset, clean_, activeCumSumCores_ * UB_ALIGN);

    // This barrier opens only after every receiver has completed GetCumSum, so
    // each publisher may reclaim its own subtotal slot. The next use of this
    // bank then starts from zeros without an extra epoch tag.
    if (newAivId < activeCumSumCores_) {
        Store(stateGM_ + layout_t::kCountLayout.cumsumOffset + newAivId * UB_ALIGN,
              clean_ + newAivId * UB_ALIGN_DATA_COUNT, UB_ALIGN);
    }
}

// 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
template <class Layout>
__aicore__ inline void CountWaitDispatch(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t UB_ALIGN = uint32_t(kNumUbAlignmentBytes);
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    constexpr uint32_t epWorldSize_ = Layout::kNumRanks;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    constexpr bool useAllCoreCount_ = layout_t::kUseAllCoreCount;
    constexpr bool packedFrames_ = layout_t::kPackedFrames;
    constexpr bool directRanks_ = layout_t::kDirectRanks;
    const uint32_t epRankId_ = ctx.view.rank;
    const uint32_t coreNum_ = ctx.core.count;
    const uint32_t startStatusIndex_ = st.start_status_index;
    const uint32_t recStatusNumPerCore_ = st.status_per_core;
    profile::DeviceProfile* const profile_ = ctx.profile;
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __ubuf__ uint32_t* const scalar_ = ub->workspace.phases.count.scalar;
    __ubuf__ uint32_t* const snapshot_ = ub->workspace.phases.count.snapshot;
    __ubuf__ uint32_t* const dense_ = ub->workspace.phases.count.dense;

    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    LlCountReadPlan readPlan;
    if (packedFrames_ && !BuildCountReadPlan(rscvStatusNum_, epWorldSize_,
            useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_,
            startStatusIndex_, recStatusNumPerCore_, readPlan, directRanks_)) { trap(); return; }
    if (packedFrames_) {
        compareTarget = static_cast<float>(readPlan.expectedFlagSum);
    }

    Sync<PIPE_S, PIPE_V>();
    if (profile_) profile_->Mark<24>();  // w24: count_ready_poll_begin
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    uint32_t firstReadySeen = 0U, firstIterDone = 0U;
#endif
    while (sumOfFlag != compareTarget) {
        if (packedFrames_) {
            ReadCountFrames(snapshot_, stateGM_, readPlan, false);
        } else Load(snapshot_, stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride,
                    UB_ALIGN, recStatusNumPerCore_, kEpServerDedupCountGmStride - UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        if (packedFrames_) {
            Sum<true>(scalar_ + 8U, snapshot_, recStatusNumPerCore_ * UB_ALIGN_DATA_COUNT);
        } else {
            GatherWords(dense_, snapshot_, recStatusNumPerCore_, 0);
            asc_sync_pipe(PIPE_V);
            Sum<true>(scalar_ + 8U, dense_, recStatusNumPerCore_);
        }
        Sync<PIPE_V, PIPE_S>();
        sumOfFlag = *reinterpret_cast<__ubuf__ float*>(scalar_ + 8U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) {
            // Partial arrival plus the first full poll iteration separate "count
            // data is late on the fabric" from "one scan is expensive".
            if (!firstReadySeen && sumOfFlag != 0.0f) { profile_->Mark<46>(); firstReadySeen = 1U; }  // w46: recv_count_first_ready
            if (!firstIterDone) { profile_->Mark<47>(); firstIterDone = 1U; }  // w47: recv_count_poll_first_iter_end
        }
#endif
    }
    if (profile_) profile_->Mark<25>();  // w25: count_ready_poll_end_after_V_S

    if (packedFrames_) {
        // Commit acquired: load the actual dense records, preserving the old
        // logical expert/source ordering for prefix sums and Receiver metadata.
        ReadCountFrames(snapshot_, stateGM_, readPlan, true);
        Sync<PIPE_MTE2, PIPE_V>();
    }
    CountGatherSumRecvCnt(ctx, ub, st);
    if (!packedFrames_) CountWaitDispatchClearStatus(ctx, ub, st);
}

template <class Layout>
__aicore__ inline void CountGatherSumRecvCnt(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    const uint32_t recStatusNumPerCore_ = st.status_per_core;
    __ubuf__ uint32_t* const dense_ = ub->workspace.phases.count.dense;
    __ubuf__ uint32_t* const snapshot_ = ub->workspace.phases.count.snapshot;

    Sync<PIPE_S, PIPE_V>();
    GatherWords(dense_, snapshot_, recStatusNumPerCore_, 1);
    Sync<PIPE_V, PIPE_S>();

    // Compute the relative prefix while outgoing cnt may still be in flight.
    // Gather preserves integer count bits; no floating-point count arithmetic.
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
        cumulative += dense_[i];
        dense_[i] = cumulative;
    }
    // The final prefix is already the subtotal; keep its original wire bits.
    CountPublishRecvCntSum(ctx, ub, st, cumulative);
}

// 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与共享前缀，再发布完成标记。
template <class Layout>
__aicore__ inline void CountCalRecvAndSetFlag(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t UB_ALIGN = uint32_t(kNumUbAlignmentBytes);
    constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t aivUsedAllToAll_ = uint32_t(layout_t::kNumRecvCores);
    const uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    const uint32_t recStatusNumPerCore_ = st.status_per_core;
    const uint32_t startStatusIndex_ = st.start_status_index;
    profile::DeviceProfile* const profile_ = ctx.profile;
    __gm__ uint8_t* const stateGM_ = st.state_gm;
    __gm__ uint8_t* const recvCntWorkspaceGM_ = st.recv_cnt_workspace_gm;
    __gm__ uint8_t* const sendCountsOutGM_ = ctx.send_counts;
    __ubuf__ uint32_t* const dense_ = ub->workspace.phases.count.dense;
    __ubuf__ uint32_t* const sumSlot_ = ub->workspace.phases.count.sum_slot;

    if (recStatusNumPerCore_ == 0U) return;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 当前 UB 中是片段内前缀；先等待所有片段总数，再累加编号更小接收核的总数。
    // 这样得到全 rank 接收表中的基址，各核可独立修正自己负责的片段。
    if (profile_) profile_->Mark<26>();  // w26: cumsum_peer_join_begin
    const uint32_t offset = CountGetCumSum(ctx, ub, st, newAivId);
    if (profile_) profile_->Mark<27>();  // w27: cumsum_peer_join_clear_issue_end
    if (layout_t::kPackedFrames) CountWaitDispatchClearStatus(ctx, ub, st);

    if (recStatusNumPerCore_ != 0U) {
        for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
            dense_[i] += offset;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        // sendCountsOut 写的是累计前缀，而非单条记录的原始发送数量。
        Store(sendCountsOutGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));

        // Producers own disjoint fragments of one read-only GM prefix. All
        // receivers acquire the existing ready row before reading this copy.
        Store(recvCntWorkspaceGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));
    }

    // 发布前缀完成标记前，先等待旧片段总数发送不再读取 sumSlot_，
    // 再复用它构造标记；前缀写入与完成标记之间保留原 MTE3 顺序约束。
    // This is the first overwrite of the asynchronous local-sum publication.
    Sync<PIPE_MTE3, PIPE_V>();
    Fill(sumSlot_, 1U, math::align(aivUsedCumSum_, 8U) * UB_ALIGN_DATA_COUNT);
    Sync<PIPE_V, PIPE_MTE3>();
    asc_sync_pipe(PIPE_MTE3);

    Store(stateGM_ + layout_t::kCountLayout.cumsumFlagOffset + newAivId * UB_ALIGN,
          sumSlot_, UB_ALIGN, aivUsedCumSum_, (aivUsedCumSum_ - 1U) * UB_ALIGN);
}

// 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
template <class Layout>
__aicore__ inline void CountRecvCnt(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t aivUsedCumSum_ = uint32_t(layout_t::kNumRecvCores);
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t aivUsedAllToAll_ = aivUsedCumSum_;

    // 接收端控制区共有 localExperts×worldSize（等于 experts）条记录，
    // 按“本地专家 → 来源 rank”排列；每个接收核负责一段连续记录。
    // Each count owns a 512B GM slot, so contiguous ownership can split at any
    // record. Keep all nonempty receivers active and their loads within one record.
    st.active_cum_sum_cores = rscvStatusNum_ < aivUsedCumSum_ ? rscvStatusNum_ : aivUsedCumSum_;
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // Inactive receivers still join prefix readiness in CompleteCumsum.
    if (receiver >= st.active_cum_sum_cores) return;
    CountSplitToCore(rscvStatusNum_, aivUsedCumSum_, receiver, st.start_status_index, st.end_status_index,
                     st.status_per_core);
    // Vector publication/cleanup touches complete groups of eight 32B records.
    // Padding is UB-only; matrix strides and participation use actual core counts.
    // 等待本片 ready 到齐后提取 count，并计算局部累计值、发布本片总数。
    // 本阶段不会等待 token payload；即使分到空片段也要发布总数 0，参与后续跨核汇总。
    CountWaitDispatch(ctx, ub, st);
}

// 摘要：汇合接收核片段总数，补齐跨核前缀偏移并发布完整前缀表的就绪标记。
template <class Layout>
__aicore__ inline void CountCalcCumsum(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    CountCalRecvAndSetFlag(ctx, ub, st);
}

// 摘要：等待所有前缀片段就绪，仅首个接收核加载前缀并输出各专家计数。
template <class Layout>
__aicore__ inline void CountCompleteCumsum(DispatchContext& ctx, __ubuf__ Layout* ub, CountPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t rscvStatusNum_ = Layout::kNumExperts;
    constexpr uint32_t moeExpertNumPerRank_ = uint32_t(layout_t::kNumLocalExperts);
    constexpr uint32_t epWorldSize_ = Layout::kNumRanks;
    const uint32_t aivId_ = ctx.core.index;
    const uint32_t aivUsedAllToAll_ = uint32_t(layout_t::kNumRecvCores);
    profile::DeviceProfile* const profile_ = ctx.profile;
    __gm__ uint8_t* const recvCntWorkspaceGM_ = st.recv_cnt_workspace_gm;
    __gm__ uint8_t* const expertTokenNumsGM_ = ctx.expert_token_nums;

    // Convert the physical AIV index to its receive-group index.
    // All receivers join readiness; only receiver0 needs a count-stage prefix.
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // 各核先写前缀再发完成标记；全部标记到齐后，读取才不会混入未更新片段。
    if (profile_) profile_->Mark<28>();  // w28: prefix_ready_wait_begin
    CountWaitCumSumFlag(ctx, ub, st);
    if (profile_) profile_->Mark<29>();  // w29: prefix_ready_wait_clear_issue_end
    // 仅接收组的第 0 核写公开专家计数，避免多核重复写同一输出。
    // 每个专家最后一个来源 rank 的前缀即该专家结束位置，直接输出累计数。
    if (receiver == 0U) {
        auto prefix = ub->workspace.phases.count.prefix;
        Load(prefix, recvCntWorkspaceGM_, rscvStatusNum_ * sizeof(int32_t));
        Sync<PIPE_MTE2, PIPE_S>();
        auto expertCounts = ub->workspace.phases.count.expert_counts;
        // 公开输出是逐专家接收条数（AOT 契约），由相邻专家累计前缀之差得到；
        // 内部前缀与 send_counts 仍保持累计形式。
        int64_t base = 0;
        for (uint32_t expert = 0; expert < moeExpertNumPerRank_; ++expert) {
            const int64_t end = int64_t(prefix[(expert + 1U) * epWorldSize_ - 1U]);
            expertCounts[expert] = end - base;
            base = end;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        Store(expertTokenNumsGM_, expertCounts, moeExpertNumPerRank_ * sizeof(int64_t));
    }
    // No listening table or payload receive/copy is constructed in this increment.
}

// 摘要：等待本核全部流水收尾，并恢复初始化前的 CTRL[60] 设置。
template <class Layout>
__aicore__ inline void CountFinish(DispatchContext&, CountPhaseState& st)
{
    asc_sync_pipe(PIPE_ALL);
    asc_set_ctrl((uint64_t(asc_get_ctrl()) & ~(1ULL << 60U)) | st.original_overflow_mode);
}

// -------------------------------------------------------------------------------
// Sender phase. Runtime state is only lane bookkeeping; every array is a member
// of the static UB map.
// -------------------------------------------------------------------------------
struct SenderPhaseState {
    __ubuf__ uint8_t* input[2]{};
    __ubuf__ uint8_t* cached_payload = nullptr;
    __ubuf__ uint8_t* out = nullptr;
    __ubuf__ uint8_t* rdma_workspace = nullptr;
    LlSenderInputReuse input_reuse;
    comm::DispatchRdmaBatch rdma_batch;
    uint32_t input_sequence = 0U, cached_tail_entries = 0U;
    bool has_scratch = false, full_mesh_first_hop = false;
    uint32_t work_count = 0U, start = 0U, count = 0U, packets = 0U, direct_count = 0U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    profile::DeviceProfile* packet_profile = nullptr;
#endif
};

template <class Layout, class XType>
struct SenderInputOperations {
    DispatchContext& ctx;
    __ubuf__ Layout* ub;
    SenderPhaseState& st;
    __aicore__ inline void LoadInput(uint32_t token);
    // The cache is read by V packing. Retire it before either input slot is reused.
    __aicore__ inline void ReleaseInput() { asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0); }
    __aicore__ inline void WaitPackedRead() { asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0); }
};

template <class Layout>
__aicore__ inline uint32_t SenderCalTokenSendExpertCnt(__ubuf__ Layout* ub, uint32_t expert, uint32_t index)
{
    constexpr uint32_t k_ = Layout::kNumTopk;
    if (index < k_) return 0U;
    ub::RoutePrefixCount(ub->workspace.phases.sender.qdst, ub->workspace.routes, index, int32_t(expert));
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    return ub->workspace.phases.sender.qdst[0];
}

template <class Layout, class XType>
__aicore__ inline void SenderLoadTokenInput(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st,
                                            uint32_t token)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t h_ = Layout::kNumHiddenElems;
    constexpr bool prequantized_ = Layout::kPrequantizedMode;
    constexpr uint32_t inputRowBytes_ = prequantized_ ? h_ : h_ * uint32_t(sizeof(XType));
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;

    auto* input = st.input[st.input_sequence++ % 2U];
    if (prequantized_) {
        const auto padding = LlPlanPrequantPadding(h_, uint32_t(data_.quantCount),
            uint32_t(data_.scaleBytes), uint32_t(data_.packedPayloadBytes));
        ub::FillPrequantPadding(input, padding);
        asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
        st.cached_tail_entries = 0U;
    }
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(input), reinterpret_cast<__gm__ uint8_t*>(ctx.x + uint64_t(token) * inputRowBytes_), 1U, inputRowBytes_, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, inputRowBytes_ + 0U, math::align(inputRowBytes_, 32U) + 0U);
    if (prequantized_) asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(input + data_.quantCount), reinterpret_cast<__gm__ uint8_t*>(ctx.input_scales + uint64_t(token) * data_.scaleBytes), 1U, uint32_t(data_.scaleBytes), 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, uint32_t(data_.scaleBytes) + 0U, math::align(uint32_t(data_.scaleBytes), 32U) + 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
    st.cached_payload = input;
}

template <class Layout, class XType>
__aicore__ inline void SenderInputOperations<Layout, XType>::LoadInput(uint32_t token)
{
    SenderLoadTokenInput<Layout, XType>(ctx, ub, st, token);
}

template <class Layout, class XType>
__aicore__ inline void SenderFillTail(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st,
                                     __ubuf__ uint8_t* payload, uint32_t token, uint32_t firstK, uint32_t mask)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr bool prequantized_ = Layout::kPrequantizedMode;
    constexpr bool enableDedup_ = Layout::kDedupMode;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    const uint32_t start_ = st.start;

    auto* header = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.headerPayloadOffset);
    auto* index = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.readIndexPayloadOffset);
    header[0] = uint32_t(ctx.view.rank); header[1] = token; header[2] = firstK;
    header[3] = uint32_t(mask ? EpServerDedupTokenType::FirstHop : EpServerDedupTokenType::Direct);
    if (!enableDedup_) { index[0] = 0U; return; }
    auto* weights = reinterpret_cast<__ubuf__ float*>(index);
    token -= start_;
    auto* routeWeights_ = ub->workspace.phases.sender.route_weights;
    weights[1] = routeWeights_[token * k_ + firstK];
    uint32_t entries = 0U;
    const uint32_t members = LlRelayMemberMask(mask, firstK);
    for (uint32_t k = 0; members && k < k_; ++k) if (members & (1U << k)) {
        const uint32_t routeIndex = (start_ + token) * k_ + k, expert = uint32_t(ub->workspace.routes[routeIndex]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const bool traceQ = st.packet_profile && st.packet_profile->SecondRemote();
        const uint64_t qBegin = traceQ ? st.packet_profile->Clock() : 0U;
#endif
        const uint32_t q = SenderCalTokenSendExpertCnt(ub, expert, routeIndex);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (traceQ) st.packet_profile->MemberQ(st.packet_profile->Clock() - qBegin);
#endif
        ++entries;
        index[entries * 2U] = expert | (q << 16U);
        weights[entries * 2U + 1U] = routeWeights_[token * k_ + k];
    }
    if (prequantized_) {
        for (uint32_t stale = entries + 1U; stale <= st.cached_tail_entries; ++stale) {
            index[stale * 2U] = 0U; index[stale * 2U + 1U] = 0U;
        }
        st.cached_tail_entries = entries;
    }
    index[0] = entries;
}

template <class Layout, class XType>
__aicore__ inline uint32_t SenderFirstHop(SenderPhaseState& st, __ubuf__ Layout* ub, uint32_t base,
                                         uint32_t mask)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr uint32_t localExperts_ = uint32_t(layout_t::kNumLocalExperts);

    LlFullMeshFirstHop mesh;
    for (uint32_t k = 0; k < k_; ++k) {
        if (!(mask & (1U << k))) continue;
        if (!st.full_mesh_first_hop) return k;
        mesh.Add(uint32_t(ub->workspace.routes[base + k]) / localExperts_, k);
    }
    return mesh.FirstK();
}

template <class Layout, class XType>
__aicore__ inline void SenderSendPacket(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st,
                                        uint32_t token, uint32_t firstK, uint32_t expert, uint32_t q,
                                        uint32_t mask)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t kMaxTokens = Layout::kNumMaxTokensPerRank;
    constexpr uint32_t Group = Layout::kNumRanksPerGroup;
    constexpr uint32_t localExperts_ = uint32_t(layout_t::kNumLocalExperts);
    constexpr uint32_t lanes_ = uint32_t(Layout::kSenderLanes);
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;

    const uint32_t destination = expert / localExperts_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (st.packet_profile) st.packet_profile->SendBegin(ctx.view.remote_write(destination),
        destination / Group == uint32_t(ctx.view.rank) / Group, token, firstK, expert, destination);
#endif
    SenderInputOperations<Layout, XType> operations{ctx, ub, st};
    st.input_reuse.Prepare(token, operations);
    SenderFillTail<Layout, XType>(ctx, ub, st, st.cached_payload, token, firstK, mask);
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
    const uint32_t lane = st.packets % lanes_;
    asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
    auto* out = st.out + lane * data_.tokenStride;
    ub::PackPayload(out, st.cached_payload, uint32_t(data_.blockCount));
    // Publish the layout marker through V too; avoid a simultaneous Scalar store.
    ub::SetPacketMarker(out, mask ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact);
    asc_sync_notify(PIPE_V, PIPE_MTE3, event_t(lane));
    asc_sync_wait(PIPE_V, PIPE_MTE3, event_t(lane));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const bool tracePrevious = st.packet_profile && st.packet_profile->SecondRemote() && st.rdma_batch.HasStaged();
    const uint64_t previousBegin = tracePrevious ? st.packet_profile->Clock() : 0U;
#endif
    st.rdma_batch.SubmitPending();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (tracePrevious) st.packet_profile->PreviousSubmit(st.packet_profile->Clock() - previousBegin);
#endif
    const uint64_t offset = DispatchServerDedupDispatchOffset(data_, ctx.view.rank, destination,
        expert % localExperts_, q, ctx.view.num_ranks, localExperts_, kMaxTokens) + ctx.window_offset;
#ifndef DEEPEP_LL_NO_COMM
    if (ctx.view.remote_write(destination)) st.rdma_batch.Enqueue(destination, offset, out, lanes_ > 1U);
    else
#endif
        comm::Packet(ctx.view, destination, offset, out, uint32_t(data_.tokenStride), 7U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (st.packet_profile) st.packet_profile->GroupSubmitted();
#endif
    asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
    ++st.packets;
}

template <class Layout>
__aicore__ inline void SenderClassify(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st,
                                     uint32_t token, uint32_t& directMask, uint32_t& groups)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr uint32_t Group = Layout::kNumRanksPerGroup;
    constexpr uint32_t localExperts_ = uint32_t(layout_t::kNumLocalExperts);
    const uint32_t start_ = st.start;
    __ubuf__ uint32_t* const serverState_ = ub->workspace.phases.sender.server_state;
    __ubuf__ uint32_t* const firstHopMasks_ = ub->workspace.phases.sender.firsthop_masks;

    uint32_t seen = 0U, grouped = 0U;
    for (uint32_t k = 0; k < k_; ++k) {
        const uint32_t expert = uint32_t(ub->workspace.routes[(start_ + token) * k_ + k]);
        const uint32_t server = expert / localExperts_ / Group, bit = 1U << k, serverBit = 1U << server;
        if (server == uint32_t(ctx.view.rank) / Group) directMask |= bit;
        else if (!(seen & serverBit)) {
            seen |= serverBit; serverState_[server] = k; directMask |= bit;
        } else if (!(grouped & serverBit)) {
            const uint32_t firstBit = 1U << serverState_[server];
            directMask &= ~firstBit;
            firstHopMasks_[groups] = firstBit | bit;
            serverState_[server] = groups++; grouped |= serverBit;
        } else {
            const uint32_t index = serverState_[server];
            firstHopMasks_[index] |= bit;
        }
    }
}

template <class Layout, class XType>
__aicore__ inline void SenderSendWithDedup(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr uint32_t Group = Layout::kNumRanksPerGroup;
    constexpr uint32_t localExperts_ = uint32_t(layout_t::kNumLocalExperts);
    const uint32_t start_ = st.start, work_count_ = st.work_count;

    const uint32_t rankInGroup = uint32_t(ctx.view.rank) % Group, sendCore = ctx.core.index;
    __ubuf__ uint32_t* const directTasks_ = ub->workspace.phases.sender.direct_tasks;
    __ubuf__ uint32_t* const sourceMasks_ = ub->workspace.phases.sender.source_masks;
    __ubuf__ uint32_t* const serverState_ = ub->workspace.phases.sender.server_state;
    __ubuf__ uint32_t* const firstHopMasks_ = ub->workspace.phases.sender.firsthop_masks;
    __ubuf__ float* const routeWeights_ = ub->workspace.phases.sender.route_weights;
    __ubuf__ float* const sourceWeights_ = ub->workspace.phases.sender.source_weights;

    for (uint32_t token = 0; token < work_count_; ++token) {
        uint32_t directMask = 0U, firstHopCount = 0U;
        SenderClassify(ctx, ub, st, token, directMask, firstHopCount);
        uint32_t sourceMask = directMask;
        for (uint32_t group = 0; group < firstHopCount; ++group) {
            const uint32_t scheduled = LlFirstHopOrder(firstHopCount, rankInGroup, sendCore, start_ + token, group);
            const uint32_t mask = firstHopMasks_[scheduled];
            const uint32_t firstK = SenderFirstHop<Layout, XType>(st, ub, (start_ + token) * k_, mask);
            const uint32_t index = (start_ + token) * k_ + firstK, expert = uint32_t(ub->workspace.routes[index]);
            SenderSendPacket<Layout, XType>(ctx, ub, st, start_ + token, firstK, expert,
                                            SenderCalTokenSendExpertCnt(ub, expert, index), mask);
            sourceMask |= 1U << firstK;
            sourceWeights_[token * k_ + firstK] = 1.0f;
            // Relay members travel inside this packet. Publish their real
            // weights so the destination's source_weights row has no stale
            // UB left in mask-active slots.
            const uint32_t members = LlRelayMemberMask(mask, firstK);
            for (uint32_t k = 0; members && k < k_; ++k)
                if (members & (1U << k)) sourceWeights_[token * k_ + k] = routeWeights_[token * k_ + k];
        }
        for (uint32_t k = 0; k < k_; ++k) if (directMask & (1U << k)) {
            directTasks_[st.direct_count * 4U] = token;
            directTasks_[st.direct_count * 4U + 1U] = k;
            directTasks_[st.direct_count * 4U + 2U] = uint32_t(ub->workspace.routes[(start_ + token) * k_ + k]);
            ++st.direct_count;
        }
        sourceMasks_[token] = sourceMask;
    }
    const uint32_t serverNum = ctx.view.num_ranks / Group, expertsPerServer = localExperts_ * Group;
    const uint32_t ownServer = ctx.view.rank / Group;
    for (uint32_t server = 0; server < serverNum; ++server) serverState_[server] = UINT32_MAX;
    for (uint32_t next = st.direct_count; next; --next) {
        const uint32_t task = next - 1U, server = directTasks_[task * 4U + 2U] / expertsPerServer;
        directTasks_[task * 4U + 3U] = serverState_[server];
        serverState_[server] = task;
    }
    for (uint32_t step = 0; step < serverNum; ++step) {
        const uint32_t server = LlDirectGroupOrder(
            serverNum, ownServer, rankInGroup, sendCore, step, st.full_mesh_first_hop);
        for (uint32_t task = serverState_[server]; task != UINT32_MAX; task = directTasks_[task * 4U + 3U]) {
            const uint32_t expert = directTasks_[task * 4U + 2U], token = directTasks_[task * 4U];
            const uint32_t k = directTasks_[task * 4U + 1U];
            SenderSendPacket<Layout, XType>(ctx, ub, st, start_ + token, k, expert,
                                            SenderCalTokenSendExpertCnt(ub, expert, (start_ + token) * k_ + k), 0U);
            sourceWeights_[token * k_ + k] = routeWeights_[token * k_ + k];
        }
    }
}

template <class Layout, class XType>
__aicore__ inline void SenderSendWithoutDedup(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t sendCores_ = uint32_t(Layout::kNumVecCores / 2U);
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr uint32_t Group = Layout::kNumRanksPerGroup;
    constexpr uint32_t localExperts_ = uint32_t(layout_t::kNumLocalExperts);
    const uint32_t num_tokens_ = ctx.num_tokens;

    const uint32_t total = num_tokens_ * k_, core = ctx.core.index, base = total / sendCores_;
    const uint32_t remainder = total % sendCores_, begin = core * base + (core < remainder ? core : remainder);
    const uint32_t count = base + (core < remainder ? 1U : 0U), serverNum = ctx.view.num_ranks / Group;
    if (serverNum < 16U) {
        for (uint32_t i = begin; i < begin + count; ++i) {
            const uint32_t expert = uint32_t(ub->workspace.routes[i]);
            SenderSendPacket<Layout, XType>(ctx, ub, st, i / k_, i % k_, expert,
                                            SenderCalTokenSendExpertCnt(ub, expert, i), 0U);
        }
        return;
    }
    const uint32_t expertsPerServer = localExperts_ * Group, ownServer = ctx.view.rank / Group;
    for (uint32_t step = 0; step < serverNum; ++step) {
        const uint32_t server = EpServerOrder::ServerDedupDispatchDirectOrder(serverNum, ownServer, step);
        const uint32_t firstExpert = server * expertsPerServer;
        for (uint32_t i = begin; i < begin + count; ++i) {
            const uint32_t expert = uint32_t(ub->workspace.routes[i]);
            if (expert < firstExpert || expert >= firstExpert + expertsPerServer) continue;
            SenderSendPacket<Layout, XType>(ctx, ub, st, i / k_, i % k_, expert,
                                            SenderCalTokenSendExpertCnt(ub, expert, i), 0U);
        }
    }
}

template <class Layout, class XType>
__aicore__ inline void SendStage(DispatchContext& ctx, __ubuf__ Layout* ub, SenderPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t sendCores_ = uint32_t(Layout::kNumVecCores / 2U);
    constexpr uint32_t Group = Layout::kNumRanksPerGroup;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr bool enableDedup_ = Layout::kDedupMode;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    constexpr uint32_t lanes_ = uint32_t(Layout::kSenderLanes);
    const TransportView& view_ = ctx.view;
    profile::DeviceProfile* profile = ctx.profile;

    // 发送工作按运行时 B 分配：静态布局仍按 M，活跃核数恢复为 sendCores。
    const uint32_t tokens = ctx.num_tokens;
    const uint32_t index = ctx.core.index, base = tokens / sendCores_,
        remainder = tokens % sendCores_;
    st.count = base + (index < remainder ? 1U : 0U);
    st.start = index * base + (index < remainder ? index : remainder);
    st.work_count = st.count;
    st.full_mesh_first_hop = Group == kNumRanksPerHost && view_.is_rdma();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    st.packet_profile = profile;
    st.rdma_batch.SetProfile(profile);
#endif
    if (profile && profile->Enabled()) {
        const uint32_t routes = ctx.num_tokens * k_;
        profile->Value<43>(enableDedup_ ? st.count * k_ :  // w43: send_route_count
            routes / sendCores_ + (ctx.core.index < routes % sendCores_ ? 1U : 0U));
    }
    if (!LlSenderHasWork(enableDedup_, st.work_count, ctx.num_tokens * k_, ctx.core.index)) return;
    // The count image at UB0 remains owned by MTE3 until count retirement.
    if (view_.is_rdma()) st.rdma_batch.SetCountRetirement(&ctx.retirement);
    st.input[0] = ub->buffers.sender.hidden[0];
    st.input[1] = ub->buffers.sender.hidden[1];
    if (view_.is_rdma()) {
        st.rdma_workspace = ub->buffers.sender.rdma_ws;
        st.rdma_batch.SetWorkspace(st.rdma_workspace);
        st.rdma_batch.Init(view_, uint32_t(data_.tokenStride));
    }
    st.out = ub->buffers.sender.wire;
    st.has_scratch = true;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (st.packet_profile) st.packet_profile->SendLanes(lanes_);
#endif
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(ub->workspace.routes), reinterpret_cast<__gm__ uint8_t*>(ctx.expert_ids), 1U, ctx.num_tokens * k_ * 4U, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, ctx.num_tokens * k_ * 4U + 0U, math::align(ctx.num_tokens * k_ * 4U, 32U) + 0U);
    if (st.work_count) asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(ub->workspace.phases.sender.route_weights), reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<__gm__ float*>(ctx.expert_scales) + uint64_t(st.start) * k_), 1U, st.work_count * k_ * 4U, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, st.work_count * k_ * 4U + 0U, math::align(st.work_count * k_ * 4U, 32U) + 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (uint32_t lane = 0; lane < lanes_; ++lane) {
        ub::FillPacketFlags(st.out + lane * data_.tokenStride, uint32_t(data_.blockCount));
        asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
    }
    asc_sync_pipe(PIPE_V);
    if (profile) profile->Mark<38>();  // w38: send_buffers_routes_ready
    if (enableDedup_) SenderSendWithDedup<Layout, XType>(ctx, ub, st);
    else SenderSendWithoutDedup<Layout, XType>(ctx, ub, st);
    st.rdma_batch.SubmitPending();
    if (profile) profile->Mark<39>();  // w39: send_packets_issue_end
    SenderInputOperations<Layout, XType> operations{ctx, ub, st};
    st.input_reuse.Finish(operations);
    st.rdma_batch.Flush();
    for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
    asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0);
    if (enableDedup_) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<__gm__ uint32_t*>(ctx.source_mask) + st.start), reinterpret_cast<__ubuf__ uint8_t*>(ub->workspace.phases.sender.source_masks), 1U, st.work_count * 4U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, st.work_count * 4U + 0U, math::align(st.work_count * 4U, 32U) + 0U);
    if (st.work_count) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<__gm__ float*>(ctx.source_weights) + uint64_t(st.start) * k_), reinterpret_cast<__ubuf__ uint8_t*>(enableDedup_ ? ub->workspace.phases.sender.source_weights : ub->workspace.phases.sender.route_weights), 1U, st.work_count * k_ * 4U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, st.work_count * k_ * 4U + 0U, math::align(st.work_count * k_ * 4U, 32U) + 0U);
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if (profile) profile->Mark<40>();  // w40: send_metadata_end_after_MTE3_S
}

// -------------------------------------------------------------------------------
// Receiver phase. Task tables and probe buffers come from the UB map.
// -------------------------------------------------------------------------------
struct ReceiverPhaseState {
    ServerDedupReceiveTasks tasks;
    uint32_t receiver = 0U, lanes = 0U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __ubuf__ uint32_t* trace = nullptr;
    bool trace_enabled = false;
    uint32_t trace_capacity = 0U, trace_ready = 0U, trace_processed = 0U, trace_batches = 0U;
    uint32_t first_hop_tokens = 0U, relay_members = 0U, relay_distinct_ranks = 0U;
#endif
};

struct ReceiverProbeResult {
    uint32_t readyMask = 0U, keepFlagsMask = 0U;
};

template <class Layout>
__aicore__ inline __gm__ uint8_t* ReceiverSlot(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st,
                                              uint32_t j)
{
    using layout_t = typename Layout::layout_t;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    return ctx.view.local_ptr() + ctx.window_offset + data_.dispatchOffset +
        uint64_t(st.tasks.Position(j)) * data_.tokenStride;
}

template <class Layout>
__aicore__ inline ReceiverProbeResult ReceiverProbe(DispatchContext& ctx, __ubuf__ Layout* ub,
                                                    ReceiverPhaseState& st, uint32_t start, uint32_t count)
{
    using layout_t = typename Layout::layout_t;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    __ubuf__ float* const flags_ = ub->buffers.receiver.probe.flags;
    __ubuf__ float* const sums_ = ub->buffers.receiver.probe.sums;
    __ubuf__ uint32_t* const snapshot_ = ub->buffers.receiver.probe.snapshot;

    const uint32_t blocks = uint32_t(data_.blockCount);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t j = st.tasks.PendingIndex(start + i);
        snapshot_[i] = j;
        ReadServerDedupFlags(ReceiverSlot(ctx, ub, st, j), blocks, flags_ + i * blocks * 8U, ctx.view.is_rdma());
    }
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    ub::SumPacketFlags(sums_, flags_, count, blocks * 8U);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    ReceiverProbeResult result;
    for (uint32_t i = 0; i < count; ++i) {
        const auto state = PacketReceiveLayout::Decode(sums_[i * 8U], blocks);
        if (state != PacketReceiveLayout::State::Pending) result.readyMask |= 1U << i;
        if (state == PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask |= 1U << i;
    }
    return result;
}

template <class Layout>
__aicore__ inline void ReceiverIssueLoad(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st,
                                         uint32_t j, uint32_t lane, bool keepFlags)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t laneStride_ = uint32_t(Layout::kLaneStride);
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    __ubuf__ uint8_t* const laneBytes_ = ub->buffers.receiver.wire;

    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
    auto* wire = laneBytes_ + lane * laneStride_;
    if (keepFlags) asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(wire), reinterpret_cast<__gm__ uint8_t*>(ReceiverSlot(ctx, ub, st, j)), 1U, uint32_t(data_.tokenStride), 0U, 0U, false, (ctx.view.is_rdma() ? asc_load_l2_cache_mode::NOTALLOC_CLEAN : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM), uint32_t(data_.tokenStride) + 0U, math::align(uint32_t(data_.tokenStride), 32U) + 0U);
    else asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(wire), reinterpret_cast<__gm__ uint8_t*>(ReceiverSlot(ctx, ub, st, j)), uint32_t(data_.blockCount), 480U, 0U, 0U, false, (ctx.view.is_rdma() ? asc_load_l2_cache_mode::NOTALLOC_CLEAN : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM), 480U + 32U, math::align(480U, 32U) + 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, event_t(lane));
    asc_sync_notify(PIPE_MTE2, PIPE_S, event_t(lane));
}

#ifdef DEEPEP_LL_REFERENCE_PROFILE
__aicore__ inline void ReceiverRecordReady(ReceiverPhaseState& st, uint32_t mask, uint32_t count,
                                           __ubuf__ uint32_t* snapshot)
{
    ++st.trace_batches;
    for (uint32_t i = 0; i < count; ++i) if (mask & (1U << i)) {
        const uint32_t j = snapshot[i]; ++st.trace_ready;
        if (j < st.trace_capacity) { st.trace[j * 8U] = st.trace_ready; st.trace[j * 8U + 1U] = st.trace_batches; }
    }
}
#endif

#ifdef DEEPEP_LL_REFERENCE_PROFILE
template <class Layout>
__aicore__ inline void ReceiverRecordProcessed(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st,
                                             uint32_t j, __ubuf__ uint8_t* wire, uint32_t headerOffset,
                                             bool keepFlags)
{
    using layout_t = typename Layout::layout_t;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    if (st.trace_enabled) ++st.trace_processed;
    if (!st.trace_enabled || j >= st.trace_capacity) return;
    auto* record = st.trace + j * 8U;
    auto* header = reinterpret_cast<__ubuf__ uint32_t*>(wire + headerOffset);
    const uint32_t role = header[3U], members = role == uint32_t(EpServerDedupTokenType::FirstHop) ?
        reinterpret_cast<__ubuf__ uint32_t*>(wire +
            (keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset))[0U] : 0U;
    record[2U] = st.trace_processed; record[3U] = header[0U]; record[4U] = header[1U];
    record[5U] = header[2U]; record[6U] = role | (members << 8U); record[7U] = st.tasks.Position(j);
}
#endif

template <class Layout>
__aicore__ inline void ReceiverConsumeLoaded(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st,
                                             uint32_t j, uint32_t lane, bool keepFlags)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t k_ = Layout::kNumTopk;
    constexpr bool wireFp8_ = Layout::kPrequantizedMode;
    constexpr uint32_t hiddenBytes_ = wireFp8_ ? Layout::kNumHiddenElems : 2U * Layout::kNumHiddenElems;
    constexpr uint32_t laneStride_ = uint32_t(Layout::kLaneStride);
    constexpr bool dedupMode_ = Layout::kDedupMode;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    __ubuf__ uint8_t* const laneBytes_ = ub->buffers.receiver.wire;
    __gm__ uint8_t* const expandX_ = ctx.expand_x;
    __gm__ uint8_t* const dynamicScales_ = ctx.dynamic_scales;
    __gm__ uint32_t* const tokenType_ = reinterpret_cast<__gm__ uint32_t*>(ctx.token_type);
    __gm__ uint32_t* const destinationIndex_ = reinterpret_cast<__gm__ uint32_t*>(ctx.destination_index);
    __gm__ uint32_t* const relayReadIndex_ = reinterpret_cast<__gm__ uint32_t*>(ctx.relay_read_index);

    asc_sync_wait(PIPE_MTE2, PIPE_S, event_t(lane));
    auto* wire = laneBytes_ + lane * laneStride_;
    const uint32_t row = st.tasks.Begin() + j * st.tasks.Stride();
    const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    ReceiverRecordProcessed(ctx, ub, st, j, wire, headerOffset, keepFlags);
#endif
    if (!dedupMode_ && !keepFlags) {
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(expandX_ + uint64_t(row) * hiddenBytes_), reinterpret_cast<__ubuf__ uint8_t*>(wire), 1U, hiddenBytes_, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, hiddenBytes_ + 0U, math::align(hiddenBytes_, 32U) + 0U);
        if (wireFp8_) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dynamicScales_ + uint64_t(row) * data_.scaleBytes), reinterpret_cast<__ubuf__ uint8_t*>(wire + data_.quantCount), 1U, uint32_t(data_.scaleBytes), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, uint32_t(data_.scaleBytes) + 0U, math::align(uint32_t(data_.scaleBytes), 32U) + 0U);
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(destinationIndex_ + uint64_t(row) * 3U), reinterpret_cast<__ubuf__ uint8_t*>(wire + headerOffset), 1U, 12U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 12U + 0U, math::align(12U, 32U) + 0U);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        return;
    }
    auto* type = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride);
    auto* destination = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride + 32U);
    ServerDedupPacketTail tail(wire, headerOffset,
        uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
    bool firstHop = false;
    if (dedupMode_) firstHop = BuildServerDedupMetadata(type, destination, tail, st.tasks.Position(j)) ==
        uint32_t(EpServerDedupTokenType::FirstHop);
    else for (uint32_t i = 0; i < 3U; ++i) destination[i] = tail.Header()[i];
    if (firstHop) {
        PrepareServerDedupForward(tail);
        reinterpret_cast<__ubuf__ float*>(wire)[PacketReceiveLayout::kMarkerOffset / 4U] =
            PacketReceiveLayout::kCompact;
    }
    asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
    if (keepFlags) {
        WriteServerDedupPayloadRange(expandX_ + uint64_t(row) * hiddenBytes_, wire, 0U, hiddenBytes_);
        if (wireFp8_) WriteServerDedupPayloadRange(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
            wire, uint32_t(data_.quantCount), uint32_t(data_.scaleBytes));
    } else {
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(expandX_ + uint64_t(row) * hiddenBytes_), reinterpret_cast<__ubuf__ uint8_t*>(wire), 1U, hiddenBytes_, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, hiddenBytes_ + 0U, math::align(hiddenBytes_, 32U) + 0U);
        if (wireFp8_) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dynamicScales_ + uint64_t(row) * data_.scaleBytes), reinterpret_cast<__ubuf__ uint8_t*>(wire + data_.quantCount), 1U, uint32_t(data_.scaleBytes), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, uint32_t(data_.scaleBytes) + 0U, math::align(uint32_t(data_.scaleBytes), 32U) + 0U);
    }
    if (dedupMode_) asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(tokenType_ + row), reinterpret_cast<__ubuf__ uint8_t*>(type), 1U, 4U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 4U + 0U, math::align(4U, 32U) + 0U);
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(destinationIndex_ + uint64_t(row) * 3U), reinterpret_cast<__ubuf__ uint8_t*>(destination), 1U, 12U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 12U + 0U, math::align(12U, 32U) + 0U);
    if (firstHop) {
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(relayReadIndex_ + uint64_t(row) * k_ * 2U), reinterpret_cast<__ubuf__ uint8_t*>(tail.ReadIndex()), 1U, k_ * 8U, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, k_ * 8U + 0U, math::align(k_ * 8U, 32U) + 0U);
        const uint32_t source = tail.Header()[0U], ranks = uint32_t(ctx.view.num_ranks), members = tail.ReadIndex()[0U];
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        ++st.first_hop_tokens; st.relay_members += members;
        uint64_t seen[4]{};
        for (uint32_t i = 0; i < members; ++i) {
            const uint32_t peer = (tail.Entry(i)[0U] & 0xffffU) / uint32_t(layout_t::kNumLocalExperts), slot = peer >> 6U;
            const uint64_t bit = uint64_t(1) << (peer & 63U);
            if (slot < 4U && !(seen[slot] & bit)) { seen[slot] |= bit; ++st.relay_distinct_ranks; }
        }
#endif
        for (uint32_t i = 0; i < members; ++i) {
            const uint32_t index = tail.Entry(i)[0U], expert = index & 0xffffU, q = index >> 16U;
            const uint32_t destinationRank = expert / uint32_t(layout_t::kNumLocalExperts);
            comm::Packet(ctx.view, destinationRank, DispatchServerDedupDispatchOffset(data_, source,
                destinationRank, expert % uint32_t(layout_t::kNumLocalExperts), q, ranks,
                uint32_t(layout_t::kNumLocalExperts), Layout::kNumMaxTokensPerRank) + ctx.window_offset,
                wire, uint32_t(data_.tokenStride), lane);
        }
    }
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
}

template <class Layout>
__aicore__ inline void ReceiverClearFlags(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    __ubuf__ float* const flags_ = ub->buffers.receiver.probe.flags;

    ub::FillWords(reinterpret_cast<__ubuf__ uint32_t*>(flags_), 0U, uint32_t(data_.blockCount) * 8U);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (uint32_t j = 0; j < st.tasks.Count(); ++j)
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(ReceiverSlot(ctx, ub, st, j) + 480U), reinterpret_cast<__ubuf__ uint8_t*>(flags_), uint32_t(data_.blockCount), 32U, (ctx.view.is_rdma() ? asc_store_l2_cache_mode::NOTALLOC_CLEAN : asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM), 32U + 480U, math::align(32U, 32U) + 0U);
}

#ifdef DEEPEP_LL_REFERENCE_PROFILE
template <class Layout>
__aicore__ inline void ReceiverExportPacketTrace(DispatchContext&, __ubuf__ Layout*, ReceiverPhaseState& st,
                                                profile::DeviceProfile* profile)
{
    using layout_t = typename Layout::layout_t;
    if (!profile || !profile->Enabled()) return;
    profile->Detail(96U, st.trace_capacity); profile->Detail(97U, st.trace_capacity);
    profile->Detail(98U, st.tasks.Count()); profile->Detail(99U, st.trace_ready);
    profile->Detail(100U, st.trace_processed); profile->Detail(101U, st.lanes);
    profile->Detail(104U, st.tasks.Begin()); profile->Detail(105U, Layout::kNumMaxTokensPerRank);
    profile->Detail(106U, uint32_t(layout_t::kNumLocalExperts));
    profile->Detail(107U, st.tasks.Count() > st.trace_capacity ? 1U : 0U);
    profile->Detail(102U, profile->Clock());
    if (st.trace_capacity) {
        asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(profile->PacketLogAddress()), reinterpret_cast<__ubuf__ uint8_t*>(st.trace), 1U, st.trace_capacity * 32U, asc_store_l2_cache_mode::NOTALLOC_CLEAN, st.trace_capacity * 32U + 0U, math::align(st.trace_capacity * 32U, 32U) + 0U);
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
    profile->Detail(103U, profile->Clock());
}
#endif

template <class Layout>
__aicore__ inline void RecvStage(DispatchContext& ctx, __ubuf__ Layout* ub, ReceiverPhaseState& st)
{
    using layout_t = typename Layout::layout_t;
    constexpr uint32_t recvCoreNum_ = uint32_t(Layout::kNumVecCores / 2U);
    constexpr uint32_t experts_ = Layout::kNumExperts;
    constexpr uint32_t kProbeBatch = uint32_t(layout_t::kNumProbeBatch);
    constexpr uint32_t kReceiverLanes = uint32_t(Layout::kReceiverLanes);
    constexpr EpServerDedupDataLayout data_ = layout_t::kData;
    const TransportView& view_ = ctx.view;
    profile::DeviceProfile* profile = ctx.profile;
    __ubuf__ int32_t* const prefix_ = ub->workspace.phases.receiver.prefix;
    __ubuf__ float* const flags_ = ub->buffers.receiver.probe.flags;

    // Count UB users are drained; the receiver section reuses that space.
    st.receiver = ctx.core.index - recvCoreNum_;
    const uint32_t prefixBytes = math::align(experts_ * 4U, 32U);
    const auto& counts = layout_t::kCountLayout;
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(prefix_), reinterpret_cast<__gm__ uint8_t*>(ctx.view.local_ptr() + ctx.window_offset +
        counts.cntOffset + counts.prefixOffset), 1U, prefixBytes, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, prefixBytes + 0U, math::align(prefixBytes, 32U) + 0U);
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
    if (profile) profile->Mark<30>();  // w30: receive_prefix_loaded_after_MTE2_S
    st.tasks.Build(prefix_, uint32_t(view_.rank), uint32_t(view_.num_ranks), experts_,
                   Layout::kNumMaxTokensPerRank, st.receiver,
                   ub->workspace.phases.receiver.task_pos,
                   ub->workspace.phases.receiver.pending_idx, recvCoreNum_);
    if (profile) {
        profile->Mark<31>();  // w31: receive_tasks_built
        profile->Value<37>(st.tasks.Count());  // w37: receive_task_count
    }
    if (!st.tasks.Count()) {
        if (profile) {
            profile->Value<33>(0U);  // w33: receive_packet_poll_accum_cycles
            profile->Value<34>(0U);  // w34: receive_load_consume_submit_accum_cycles
            profile->Value<41>(0U);  // w41: receive_probe_batch_count
            profile->Value<48>(0U);  // w48: receive_first_hop_tokens
            profile->Value<49>(0U);  // w49: receive_relay_members
        }
        return;
    }
    const uint32_t blocks = uint32_t(data_.blockCount);
    const uint32_t flagsBytes = kProbeBatch * blocks * 32U + 256U;
    ub::FillBytes(reinterpret_cast<__ubuf__ uint8_t*>(flags_), 0U, flagsBytes);
    asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
    __ubuf__ uint32_t* const snapshot_ = ub->buffers.receiver.probe.snapshot;
    st.lanes = kReceiverLanes;
    if (st.lanes > st.tasks.Count()) st.lanes = st.tasks.Count();
    if (!st.lanes) { trap(); return; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile && profile->Enabled()) {
        st.trace_enabled = true;
        st.trace_capacity = st.tasks.Count();
        if (st.trace_capacity > profile::kPacketTraceCapacity)
            st.trace_capacity = profile::kPacketTraceCapacity;
        st.trace = reinterpret_cast<__ubuf__ uint32_t*>(ub->buffers.receiver.probe.trace);
    }
#endif
    for (uint32_t lane = 0; lane < st.lanes; ++lane) asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
    uint32_t scan = 0U, issued = 0U, consumed = 0U;
    if (profile) profile->Mark<32>();  // w32: receive_buffers_ready
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    uint64_t pollCycles = 0U, consumeCycles = 0U, batches = 0U;
    const bool referenceEnabled = profile && profile->Enabled();
#endif
    while (st.tasks.PendingCount()) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t pollBegin = referenceEnabled ? profile->Clock() : 0U;
#endif
        ReceiverProbeResult probe;
        uint32_t count;
        do {
            if (scan >= st.tasks.PendingCount()) scan = 0U;
            const uint32_t remaining = st.tasks.PendingCount() - scan;
            count = remaining < kProbeBatch ? remaining : kProbeBatch;
            probe = ReceiverProbe(ctx, ub, st, scan, count);
            if (!probe.readyMask) scan += count;
        } while (!probe.readyMask);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t pollEnd = referenceEnabled ? profile->Clock() : 0U;
        if (referenceEnabled) {
            if (!batches) profile->Value<42>(pollEnd);  // w42: first_packet_ready
            pollCycles += pollEnd - pollBegin; ++batches;
            ReceiverRecordReady(st, probe.readyMask, count, snapshot_);
        }
#endif
        uint32_t nextLoad = count;
        for (uint32_t i = count; i; --i) {
            const uint32_t index = i - 1U;
            if (!(probe.readyMask & (1U << index))) continue;
            while (issued - consumed < st.lanes && nextLoad) {
                --nextLoad;
                if (!(probe.readyMask & (1U << nextLoad))) continue;
                ReceiverIssueLoad(ctx, ub, st, snapshot_[nextLoad], issued % st.lanes,
                                  (probe.keepFlagsMask & (1U << nextLoad)) != 0U);
                ++issued;
            }
            ReceiverConsumeLoaded(ctx, ub, st, snapshot_[index], consumed % st.lanes,
                                  (probe.keepFlagsMask & (1U << index)) != 0U);
            st.tasks.RemovePending(scan + index);
            ++consumed;
        }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (referenceEnabled) consumeCycles += profile->Clock() - pollEnd;
#endif
    }
    for (uint32_t lane = 0; lane < st.lanes; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
    ReceiverClearFlags(ctx, ub, st);
    if (profile) profile->Mark<35>();  // w35: receive_lane_wait_clear_issue_end
    asc_sync_pipe(PIPE_ALL);
    if (profile) profile->Mark<36>();  // w36: receive_final_PIPE_ALL_end
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (referenceEnabled) {
        profile->Value<33>(pollCycles);  // w33: receive_packet_poll_accum_cycles
        profile->Value<34>(consumeCycles);  // w34: receive_load_consume_submit_accum_cycles
        profile->Value<41>(batches);  // w41: receive_probe_batch_count
        profile->Value<48>(st.first_hop_tokens);  // w48: receive_first_hop_tokens
        profile->Value<49>(st.relay_members);  // w49: receive_relay_members
        profile->Value<62>(st.relay_distinct_ranks);  // w62: receive_relay_distinct_ranks
    }
#endif
}
}  // namespace DispatchDedup

namespace deep_ep {

// Runtime argument block of the SuperKernel sub-entry. One field per explicit
// kernel parameter and in the same order, so the bind record and the ordinary
// launch share one ABI.
struct DispatchArgs {
    __gm__ uint8_t* x;
    __gm__ uint8_t* expert_ids;
    __gm__ uint8_t* expert_scales;
    __gm__ uint8_t* expand_x_out;
    __gm__ uint8_t* dynamic_scales_out;
    __gm__ uint8_t* expert_token_nums_out;
    __gm__ uint8_t* send_counts_out;
    __gm__ uint8_t* token_type_out;
    __gm__ uint8_t* destination_index_out;
    __gm__ uint8_t* relay_read_index_out;
    __gm__ uint8_t* source_mask_out;
    __gm__ uint8_t* source_weights_out;
    __gm__ uint8_t* input_scales;
    __gm__ uint8_t* peer_ptrs;
    __gm__ uint8_t* rdma_remote_base;
    __gm__ uint8_t* rdma_stage_base;
    __gm__ uint8_t* profile_out;
    int32_t rank_idx;
    int32_t num_tokens;
    int32_t physical_ranks_per_host;
    int32_t backend;
    int32_t stage_block_num;
    int64_t peer_mem_bytes;
    int64_t ffts_val;
};
EP_STATIC_ASSERT(sizeof(DispatchArgs) == 176U, "SuperKernel argument view must match the global ABI");
EP_STATIC_ASSERT(offsetof(DispatchArgs, rank_idx) == 136U, "rank slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, num_tokens) == 140U, "token count slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, physical_ranks_per_host) == 144U, "ranks-per-host slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, peer_mem_bytes) == 160U, "peer memory slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, ffts_val) == 168U, "ffts slot offset");

// The static UB maps (DispatchUBLayout / DispatchUBLayoutWithBuffer) are
// defined next to the phase classes above; dispatch_body only instantiates them.

template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kNumUbBytes,
          uint32_t kNumRanksPerGroup,
          bool kPrequantized, bool kDedup, bool kRdma>
__aicore__ inline void dispatch_body(
    __gm__ uint8_t* x, __gm__ uint8_t* expert_ids, __gm__ uint8_t* expert_scales,
    __gm__ uint8_t* expand_x_out, __gm__ uint8_t* dynamic_scales_out, __gm__ uint8_t* expert_token_nums_out,
    __gm__ uint8_t* send_counts_out, __gm__ uint8_t* token_type_out, __gm__ uint8_t* destination_index_out,
    __gm__ uint8_t* relay_read_index_out, __gm__ uint8_t* source_mask_out, __gm__ uint8_t* source_weights_out,
    __gm__ uint8_t* input_scales, __gm__ uint8_t* peer_ptrs, __gm__ uint8_t* rdma_remote_base, __gm__ uint8_t* rdma_stage_base,
    __gm__ uint8_t* profile_out,
    const int32_t rank_idx, const int32_t num_tokens, const int32_t physical_ranks_per_host,
    const int32_t backend, const int32_t stage_block_num, const int64_t peer_mem_bytes,
    const int64_t ffts_val, CoreContext core)
{
    (void)rdma_stage_base;
    TransportView view{};
    view.peer_words = reinterpret_cast<__gm__ const uint64_t*>(peer_ptrs);
    view.rdma_remote_base = rdma_remote_base;
    view.rdma_stage_base = rdma_stage_base;
    view.rank = uint32_t(rank_idx);
    view.num_ranks = kNumRanks;
    view.ranks_per_host = uint32_t(physical_ranks_per_host);
    view.stage_block_num = uint32_t(stage_block_num);
    view.backend = uint32_t(backend);
    view.peer_mem_bytes = uint64_t(peer_mem_bytes);

    EP_DEVICE_ASSERT(view.num_ranks == kNumRanks && view.rank < kNumRanks);
    EP_DEVICE_ASSERT(core.count == kNumVecCores && core.index < kNumVecCores);
    EP_DEVICE_ASSERT(num_tokens >= 0 && uint32_t(num_tokens) <= kNumMaxTokensPerRank);
    EP_DEVICE_ASSERT(view.is_rdma() == kRdma);

    util_set_ffts_config(uint64_t(ffts_val));

    // The admitted input element is BF16; prequantized rows are opaque FP8
    // packs, so this type only sizes the plain path.
    using XType = bfloat16_t;
    // Hidden counts are element counts in the device layout; the template
    // carries the byte count, matching DeepEP-Ascend's kNumHiddenBytes.
    constexpr uint32_t kNumHiddenElems = kPrequantized ? kNumHiddenBytes
                                                       : kNumHiddenBytes / sizeof(XType);
    using ub_layout_t = DispatchUBLayoutWithBuffer<kNumRanks, kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk,
                                                   kNumExperts, kNumVecCores, kNumUbBytes, kNumRanksPerGroup,
                                                   kPrequantized, kDedup, kRdma>;
    EP_STATIC_ASSERT(sizeof(ub_layout_t) <= kNumUbBytes, "Dispatch UB capacity");
    // Lane count is a compile-time fact; the map asserts at least one lane.
    // UB address space starts at zero; the map's member order is the layout.
    auto* ub = reinterpret_cast<__ubuf__ ub_layout_t*>(uint64_t(0U));

    profile::DeviceProfile profile_state;
    if constexpr (true) {
        profile_state.Init(profile_out);
        profile_state.Value<0>(profile::DeviceProfile::EntryClock(profile_out));
        profile_state.Mark<1>();
    }

    const uint32_t magic = RotateDispatchServerDedupMagic(view, core.index);
    profile_state.Mark<2>();
    profile_state.Mark<3>();

    // P6: unpack every runtime pointer/scalar once, then run the three phases.
    DispatchContext context{};
    context.view = view;
    context.x = x;
    context.expert_ids = expert_ids;
    context.expert_scales = expert_scales;
    context.expand_x = expand_x_out;
    context.dynamic_scales = dynamic_scales_out;
    context.expert_token_nums = expert_token_nums_out;
    context.send_counts = send_counts_out;
    context.token_type = token_type_out;
    context.destination_index = destination_index_out;
    context.relay_read_index = relay_read_index_out;
    context.source_mask = source_mask_out;
    context.source_weights = source_weights_out;
    context.input_scales = input_scales;
    context.profile = &profile_state;
    context.rank_idx = uint32_t(rank_idx);
    context.num_tokens = uint32_t(num_tokens);
    context.window_offset = DispatchServerDedupWindowOffset(int64_t(magic));
    context.core = core;

    CountPhaseState count_state;
    CountStageBegin<ub_layout_t>(context, count_state);
    profile_state.Mark<4>();

    SenderPhaseState sender_state;
    ReceiverPhaseState receiver_state;
    const uint32_t send_cores = core.count / 2U;
    profile_state.Mark<5>();

    profile_state.Mark<6>();
    CountSendCnt<ub_layout_t>(context, ub, count_state);
    profile_state.Mark<7>();

    if (core.index < send_cores) {
        profile_state.Mark<8>();
        SendStage<ub_layout_t, XType>(context, ub, sender_state);
        profile_state.Mark<9>();
    } else {
        profile_state.Mark<10>();
        CountRecvCnt<ub_layout_t>(context, ub, count_state);
        profile_state.Mark<11>();
        profile_state.Mark<12>();
        CountCalcCumsum<ub_layout_t>(context, ub, count_state);
        profile_state.Mark<13>();
        profile_state.Mark<14>();
        CountCompleteCumsum<ub_layout_t>(context, ub, count_state);
        profile_state.Mark<15>();
        profile_state.Mark<16>();
        asc_sync_pipe(PIPE_ALL);
        profile_state.Mark<17>();
        profile_state.Mark<18>();
        RecvStage<ub_layout_t>(context, ub, receiver_state);
        profile_state.Mark<19>();
    }

    profile_state.Mark<20>();
    CountFinish<ub_layout_t>(context, count_state);
    if (context.retirement.Pending()) {
        if (core.index >= send_cores)
            context.retirement.Finish(reinterpret_cast<__ubuf__ uint8_t*>(ub->workspace.phases.receiver.prefix));
        else if (sender_state.has_scratch) context.retirement.Finish(ub->buffers.sender.wire);
        else context.retirement.Finish(
            reinterpret_cast<__ubuf__ uint8_t*>(ub->workspace.phases.count.histogram));
    }
    profile_state.Mark<21>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (core.index >= send_cores) ReceiverExportPacketTrace<ub_layout_t>(context, ub, receiver_state, &profile_state);
#endif
    profile_state.Finish(uint32_t(rank_idx), kNumRanks, uint32_t(num_tokens));
}

template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kNumUbBytes,
          uint32_t kNumRanksPerGroup,
          bool kPrequantized, bool kDedup, bool kRdma>
__global__ __aicore__ void dispatch_impl(
    __gm__ uint8_t* x, __gm__ uint8_t* expert_ids, __gm__ uint8_t* expert_scales,
    __gm__ uint8_t* expand_x_out, __gm__ uint8_t* dynamic_scales_out, __gm__ uint8_t* expert_token_nums_out,
    __gm__ uint8_t* send_counts_out, __gm__ uint8_t* token_type_out, __gm__ uint8_t* destination_index_out,
    __gm__ uint8_t* relay_read_index_out, __gm__ uint8_t* source_mask_out, __gm__ uint8_t* source_weights_out,
    __gm__ uint8_t* input_scales, __gm__ uint8_t* peer_ptrs, __gm__ uint8_t* rdma_remote_base, __gm__ uint8_t* rdma_stage_base,
    __gm__ uint8_t* profile_out,
    const int32_t rank_idx, const int32_t num_tokens, const int32_t physical_ranks_per_host,
    const int32_t backend, const int32_t stage_block_num, const int64_t peer_mem_bytes,
    const int64_t ffts_val)
{
    EP_STATIC_ASSERT(kNumVecCores > 0 && kNumMaxTokensPerRank > 0 && kNumHiddenBytes > 0);
    dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                  kNumUbBytes, kNumRanksPerGroup, kPrequantized, kDedup, kRdma>(
        x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out, expert_token_nums_out, send_counts_out,
        token_type_out, destination_index_out, relay_read_index_out, source_mask_out, source_weights_out,
        input_scales, peer_ptrs, rdma_remote_base, rdma_stage_base, profile_out, rank_idx,
        num_tokens, physical_ranks_per_host, backend, stage_block_num, peer_mem_bytes, ffts_val,
        OrdinaryCoreContext());
}

// One SuperKernel sub-entry template. `kSkSlot` only distinguishes the four
// symbols the bind table requires; the logic is identical.
template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kNumUbBytes,
          uint32_t kNumRanksPerGroup,
          bool kPrequantized, bool kDedup, bool kRdma, uint32_t kSkSlot>
__aicore__ inline void dispatch_impl_sk(const DispatchArgs* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    const CoreContext core{uint32_t(system_args->SkGetBlockIdx()), uint32_t(system_args->SkGetNumBlocks())};
    dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                  kNumUbBytes, kNumRanksPerGroup, kPrequantized, kDedup, kRdma>(
        args->x, args->expert_ids, args->expert_scales, args->expand_x_out, args->dynamic_scales_out,
        args->expert_token_nums_out, args->send_counts_out, args->token_type_out, args->destination_index_out,
        args->relay_read_index_out, args->source_mask_out, args->source_weights_out, args->input_scales,
        args->peer_ptrs, args->rdma_remote_base, args->rdma_stage_base, args->profile_out,
        args->rank_idx, args->num_tokens, args->physical_ranks_per_host, args->backend, args->stage_block_num,
        args->peer_mem_bytes, args->ffts_val, core);
}

}  // namespace deep_ep
