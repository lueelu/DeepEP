// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/ub.hpp>
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include <deep_ep/comm/device_profile.hpp>
#endif

#ifdef DEEPEP_LL_RDMA
#include <device/gm2gm/engine/shmem_device_rdma.h>
#endif

namespace deep_ep {
namespace comm {

struct TransportView {
    __gm__ const uint64_t* peer_words = nullptr;  // GM table, one pointer-sized entry per rank
    __gm__ uint8_t* rdma_remote_base = nullptr;
    __gm__ uint8_t* rdma_stage_base = nullptr;
    uint32_t rank = 0U;
    uint32_t num_ranks = 0U;
    uint32_t ranks_per_host = kNumRanksPerHost;
    uint32_t stage_block_num = 0U;
    uint32_t backend = 0U;
    uint64_t peer_mem_bytes = 0U;

    __aicore__ inline __gm__ uint8_t* local_ptr() const { return peer_ptr(rank); }
    __aicore__ inline __gm__ uint8_t* peer_ptr(uint32_t peer) const
    {
        return reinterpret_cast<__gm__ uint8_t*>(peer_words[peer]);
    }
    __aicore__ inline bool is_rdma() const { return backend == rdma::kBackend; }
    __aicore__ inline bool same_host(uint32_t peer) const
    {
        return LlSameHost(rank, peer, ranks_per_host);
    }
    __aicore__ inline bool remote_write(uint32_t peer) const { return is_rdma() && !same_host(peer); }
};
__aicore__ inline bool Rdma(const TransportView& a) { return a.backend == rdma::kBackend; }
__aicore__ inline bool SameHost(const TransportView& a, uint32_t peer) {
    return LlSameHost(uint32_t(a.rank), peer,
        a.ranks_per_host ? a.ranks_per_host : kNumRanksPerHost);
}
__aicore__ inline bool RemoteWrite(const TransportView& a, uint32_t peer) { return Rdma(a) && !SameHost(a, peer); }
__aicore__ inline void Check(const TransportView& a, uint32_t peer, uint64_t offset, uint64_t bytes) {
    if (peer >= uint32_t(a.num_ranks) || offset > a.peer_mem_bytes || bytes > a.peer_mem_bytes - offset) trap();
}
__aicore__ inline __gm__ uint8_t* RemoteAddress(const TransportView& a, uint32_t peer, uint64_t offset) {
    return a.rdma_remote_base + uint64_t(peer) * 1024U + offset;
}
__aicore__ inline __gm__ uint8_t* Address(const TransportView& a, uint32_t peer, uint64_t offset) {
    return RemoteWrite(a, peer) ? RemoteAddress(a, peer, offset) : a.peer_ptr(peer) + offset;
}
__aicore__ inline __gm__ uint8_t* Stage(const TransportView& a, uint32_t bytes) {
    if (asc_get_block_num() != a.stage_block_num || asc_get_block_idx() >= a.stage_block_num ||
        bytes > rdma::kCoreBytes) trap();
    return a.rdma_stage_base + uint64_t(asc_get_block_idx()) * rdma::kCoreBytes;
}
__aicore__ inline void Put(const TransportView& a, uint32_t peer, __gm__ uint8_t* dst, __gm__ uint8_t* src,
                           uint32_t bytes, __ubuf__ uint8_t* scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_put_nbi(dst, src, scratch, bytes, peer, asc_get_block_idx(), syncId);
#else
    trap();
#endif
}
__aicore__ inline void Quiet(uint32_t peer, __ubuf__ uint8_t* scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_quiet(peer, asc_get_block_idx(), scratch, syncId);
#else
    trap();
#endif
}
// Raw dispatch path. The SDK borrows the first 128B after the staged DMA completes.
__aicore__ inline void Packet(const TransportView& a, uint32_t peer, uint64_t offset,
                              __ubuf__ uint8_t* packet, uint32_t bytes, uint32_t syncId = 7U,
                              bool preservePacket = true) {
    Check(a, peer, offset, bytes);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst), reinterpret_cast<__ubuf__ uint8_t*>(packet), 1U, bytes, (Rdma(a) ? asc_store_l2_cache_mode::NOTALLOC_CLEAN : asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM), bytes + 0U, math::align(bytes, 32U) + 0U);
        return;
    }
    if (!bytes || bytes % 512U || syncId >= 8U) { trap(); return; }
    auto stage = Stage(a, bytes);
    asc_sync_pipe(PIPE_ALL);
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(stage), reinterpret_cast<__ubuf__ uint8_t*>(packet), 1U, bytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN, bytes + 0U, math::align(bytes, 32U) + 0U);
    asc_sync_notify(PIPE_MTE3, PIPE_S, event_t(syncId));
    asc_sync_wait(PIPE_MTE3, PIPE_S, event_t(syncId));
    Put(a, peer, dst, stage, bytes, packet, syncId);
    Quiet(peer, packet, syncId);
    if (preservePacket) {
        asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(packet), reinterpret_cast<__gm__ uint8_t*>(stage), 1U, 128U, 0U, 0U, false, asc_load_l2_cache_mode::NOTALLOC_CLEAN, 128U + 0U, math::align(128U, 32U) + 0U);
        asc_sync_notify(PIPE_MTE2, PIPE_S, event_t(syncId));
        asc_sync_wait(PIPE_MTE2, PIPE_S, event_t(syncId));
        asc_sync_pipe(PIPE_ALL);
    }
}
// Count-only raw path, preserving the body-before-ready fallback for RDMA.
__aicore__ inline void Records(const TransportView& a, uint32_t peer, uint64_t offset,
                               __ubuf__ uint8_t* records, uint32_t count, uint32_t stride,
                               bool counts, __ubuf__ uint8_t* scratch) {
    if (!count) return;
    Check(a, peer, offset, uint64_t(count - 1U) * stride + 32U);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {
        count_ub::Store(dst, records, 32U, count, stride - 32U, Rdma(a));
        return;
    }
    auto stage = Stage(a, count * 32U);
    asc_sync_pipe(PIPE_ALL);
    count_ub::Store(stage, records, count * 32U, 1U, 0U, true);
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
    if (counts) {
        for (uint32_t i = 0; i < count; ++i)
            Put(a, peer, dst + uint64_t(i)*stride + 4U, stage + i*32U + 4U, 28U, scratch);
        Quiet(peer, scratch);
    }
    for (uint32_t i = 0; i < count; ++i)
        Put(a, peer, dst + uint64_t(i)*stride, stage + i*32U, counts ? 4U : 32U, scratch);
    Quiet(peer, scratch);
}
}

#ifdef DEEPEP_LL_REFERENCE_PROFILE
#endif
namespace comm {
// Per-core CQ bookkeeping, independent of receiver publication. A payload
// quiet on the same peer/QP also retires the earlier count request.
class CountRetirement {
    TransportView args_{};
    uint32_t pending_ = 0U;
public:
    __aicore__ inline void Init(const TransportView& args) { args_ = args; pending_ = 0U; }
    __aicore__ inline void Posted(uint32_t task) { pending_ |= 1U << (task / asc_get_block_num()); }
    __aicore__ inline bool Pending() const { return pending_ != 0U; }
    __aicore__ inline void Completed(uint32_t peer) {
        const uint32_t host = uint32_t(args_.rank) / kNumRanksPerHost * kNumRanksPerHost;
        if (peer >= host && peer < host + kNumRanksPerHost) return;
        const uint32_t task = peer < host ? peer : peer - kNumRanksPerHost;
        if (task % asc_get_block_num() == asc_get_block_idx()) pending_ &= ~(1U << (task / asc_get_block_num()));
    }
    __aicore__ inline void Finish(__ubuf__ uint8_t* scratch) {
        const uint32_t host = uint32_t(args_.rank) / kNumRanksPerHost * kNumRanksPerHost;
        for (uint32_t task = asc_get_block_idx(), bit = 1U; task < uint32_t(args_.num_ranks) - kNumRanksPerHost;
             task += asc_get_block_num(), bit <<= 1U) {
            if (!(pending_ & bit)) continue;
            const uint32_t peer = task < host ? task : task + kNumRanksPerHost;
            Quiet(peer, scratch);
        }
        pending_ = 0U;
    }
};

// Count TX is isolated in each symmetric bank, outside count/prefix, combine
// controls and the payload region. One disjoint slot per destination rank.
constexpr uint64_t kCountTxBase = 1280ULL * 1024U;
static_assert(kCountTxBase >= 1024U * 1024U + 18432U + 36U * 512U,
              "count TX above existing combine controls");
static_assert(kCountTxBase + (1024U + 256U) * 32U < 2U * 1024U * 1024U, "count TX below payload");
// Remote rank-owned counts only. records includes a fully initialized tail
// record with ready=1 even for zeros. CQ retirement is deferred to the role's
// existing payload quiet or final core completion, never required by SendCnt.
__aicore__ inline void DirectCountFrame(const TransportView& args, uint32_t peer,
                                       uint64_t frameOffset, __ubuf__ uint8_t* records,
                                       uint32_t count, __ubuf__ uint8_t* scratch, uint64_t window = 0U
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                                       , profile::DeviceProfile* trace = nullptr
#endif
                                       ) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t0 = trace ? trace->PublishClock() : 0U;
#endif
    const uint32_t bytes = (count + 1U) * 32U;
    Check(args, peer, frameOffset, bytes);
    auto stage = args.local_ptr() + window + kCountTxBase + uint64_t(peer) * bytes;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t1 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<4, true>();
#endif
    count_ub::Store(stage, records, bytes, 1U, 0U, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<5, true>();
#endif
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t3 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<6, true>();
    if constexpr (profile::kCountComputeMode >= 3U) {
        auto dst = Address(args,peer,frameOffset);
        const uint64_t t4 = trace ? trace->PublishClock() : 0U;
        if (trace) trace->CountBoundary<7, true>();
        Put(args,peer,dst,stage,bytes,scratch);
        const uint64_t t5 = trace ? trace->PublishClock() : 0U;
        if (trace) {
            trace->CountBoundary<8, true>();
            trace->PublishAdd<4>(t1-t0); trace->PublishAdd<5>(t2-t1);
            trace->PublishAdd<6>(t3-t2); trace->PublishAdd<7>(t4-t3);
            trace->PublishAdd<8>(t5-t4);
        }
    } else
#endif
    Put(args,peer,Address(args,peer,frameOffset),stage,bytes,scratch);
}
// Same DataAsFlag blocks as payload: 480B data followed by eight FP32 ones.
// One copy publishes every intersecting block; the source stays immutable
// until the existing role completion barrier. No body/commit completion wait.
__aicore__ inline void CountFrame(const TransportView& args, uint32_t peer,
                                 uint64_t frameOffset, __ubuf__ uint8_t* records,
                                 uint32_t wireBytes, uint32_t count
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                                 , profile::DeviceProfile* trace = nullptr
#endif
                                 ) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t0 = trace ? trace->PublishClock() : 0U;
#endif
    Check(args, peer, frameOffset, wireBytes);
    auto dst=Address(args,peer,frameOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t1 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<14, true>();
#endif
    count_ub::Store(dst, records, wireBytes, 1U, 0U, Rdma(args));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = trace ? trace->PublishClock() : 0U;
    if (trace) {
        trace->CountBoundary<15, true>();
        trace->CompletePublishAdd<11>(1U); trace->CompletePublishAdd<18>(count);
        trace->PublishAdd<11>(1U); trace->PublishAdd<18>(count);
        trace->PublishAdd<14>(t1-t0); trace->PublishAdd<15>(t2-t1);
        // Retired local wait/commit sections remain zero in profile mode 3.
    }
#endif
}
}

namespace comm {
// One staged-but-not-posted packet. Reuse the existing GM batch and output
// lanes; the next packet must use a different UB lane until SubmitPending.
// Each GM stage slot also owns a dedicated RDMA workspace, so a posted Put
// never aliases the packet lane it was copied from.
class DispatchRdmaBatch {
public:
    // ACLSHMEM_RDMA_MTE_STAGING_UB_SIZE: one in-flight Put owns one slot until
    // the matching Quiet returns.
    static constexpr uint32_t kWorkspaceBytes = 128U;
private:
    static constexpr uint32_t kPeerWords = (DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    TransportView args_{};
    uint32_t bytes_ = 0, capacity_ = 0, count_ = 0;
    __gm__ uint8_t* stageBase_ = nullptr;
    __gm__ uint8_t* pendingStage_ = nullptr;
    uint32_t pendingPeer_ = 0U, pendingSlot_ = 0U;
    uint64_t pendingOffset_ = 0U;
    uint32_t peerMask_[kPeerWords]{};
    __ubuf__ uint8_t* workspaceBase_ = nullptr;
    __ubuf__ uint8_t* quietScratch_ = nullptr;
    bool staged_ = false;
    CountRetirement* countRetirement_ = nullptr;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    profile::DeviceProfile* profile_ = nullptr;
    bool pendingSecond_ = false;
#endif
    __aicore__ inline void Retire(bool secondTriggered) {
        if (!count_) return;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t quietBegin = profile_ ? profile_->Clock() : 0U;
#else
        (void)secondTriggered;
#endif
        for (uint32_t word = 0; word < kPeerWords; ++word) {
            const uint32_t mask = peerMask_[word];
            if (!mask) continue;
            for (uint32_t bit = 0; bit < 32U; ++bit) if (mask & (1U << bit)) {
                const uint32_t peer = word * 32U + bit;
                Quiet(peer, quietScratch_);
                if (countRetirement_) countRetirement_->Completed(peer);
            }
            peerMask_[word] = 0U;
        }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) profile_->FlushDone(quietBegin, secondTriggered);
#endif
        count_ = 0;
    }
public:
    __aicore__ inline void SetCountRetirement(CountRetirement* p) { countRetirement_ = p; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void SetProfile(profile::DeviceProfile* p) { profile_ = p; }
#endif
    // One workspace per GM stage slot plus one dedicated Quiet workspace.
    // Call before Init so capacity_ can place the Quiet slot after all slots.
    __aicore__ inline void SetWorkspace(__ubuf__ uint8_t* base) { workspaceBase_ = base; }
    __aicore__ inline void Init(const TransportView& args, uint32_t bytes) {
        if (count_) { trap(); return; }
        args_ = args; bytes_ = bytes; count_ = 0;
        if (!bytes || bytes % 512U || bytes > rdma::kCoreBytes || !workspaceBase_) { trap(); return; }
        const uint32_t slots = rdma::kCoreBytes / bytes_;
        capacity_ = slots; // use every complete packet slot in the 128KiB GM stage
        quietScratch_ = workspaceBase_ + uint64_t(capacity_) * kWorkspaceBytes;
        if (uint32_t(args_.num_ranks) > kNumRanksPerHost)
            stageBase_ = Stage(args_, capacity_ * bytes_);
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }
    __aicore__ inline bool HasStaged() const { return staged_; }
    // Caller submits the previous packet before issuing either the next local
    // copy or next remote staging. It may prepare the next packet first, using
    // a DIFFERENT lane. Single-lane callers select defer=false.
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset,
                                   __ubuf__ uint8_t* packet, bool defer = false) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        pendingSecond_ = profile_ && profile_->SecondRemote();
        if (profile_) profile_->SecondEnqueue();
#endif
        Check(args_, peer, offset, bytes_);
        pendingStage_ = stageBase_ + uint64_t(count_) * bytes_;
        pendingSlot_ = count_;
        pendingPeer_ = peer; pendingOffset_ = offset;
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(pendingStage_), reinterpret_cast<__ubuf__ uint8_t*>(packet), 1U, bytes_, asc_store_l2_cache_mode::NOTALLOC_CLEAN, bytes_ + 0U, math::align(bytes_, 32U) + 0U);
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
        peerMask_[peer / 32U] |= 1U << (peer % 32U);
        ++count_; staged_ = true;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) profile_->StageQueued(pendingSecond_, defer);
#endif
        if (!defer) SubmitPending();
    }
    __aicore__ inline void SubmitPending() {
        if (!staged_) return;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) profile_->StageWaitBegin(pendingSecond_);
#endif
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) { profile_->StageWaitEnd(pendingSecond_); profile_->SecondPutBegin(pendingSecond_); }
#endif
        // Only after the wait may NIC read GM; the SDK borrows this slot's
        // workspace until the batch Quiet below returns.
        Put(args_, pendingPeer_, RemoteAddress(args_, pendingPeer_, pendingOffset_),
            pendingStage_, bytes_, workspaceBase_ + uint64_t(pendingSlot_) * kWorkspaceBytes);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) profile_->PutReturned(pendingSecond_);
#endif
        staged_ = false;
        if (count_ == capacity_) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            Retire(pendingSecond_);
#else
            Retire(false);
#endif
        }
    }
    __aicore__ inline void Flush() {
        SubmitPending();
        Retire(false);
    }
    // Compatibility entry points for the separate, unchanged Combine tensor implementation.
    template <class Buffer>
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, Buffer packet, bool defer = false) {
        Enqueue(peer, offset, reinterpret_cast<__ubuf__ uint8_t*>(packet.GetPhyAddr()), defer);
    }
};
}

}  // namespace deep_ep
