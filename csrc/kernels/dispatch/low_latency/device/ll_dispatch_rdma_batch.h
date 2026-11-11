#pragma once
#include "ll_transport.h"
#include "ll_count_frame_transport.h"
#include "../common/ll_reference_profile_device.h"

namespace LlTransport {
// One staged-but-not-posted packet. Reuse the existing GM batch and output
// lanes; the next packet must use a different UB lane until SubmitPending.
class DispatchRdmaBatch {
    static constexpr uint32_t kPeerWords = (DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    const __gm__ Args* args_ = nullptr;
    uint32_t bytes_ = 0, capacity_ = 0, count_ = 0;
    GM_ADDR stageBase_ = nullptr;
    GM_ADDR pendingStage_ = nullptr;
    uint32_t pendingPeer_ = 0U;
    uint64_t pendingOffset_ = 0U;
    uint32_t peerMask_[kPeerWords]{};
    LocalTensor<uint8_t> pendingScratch_;
    bool staged_ = false;
    CountRetirement* countRetirement_ = nullptr;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    bool pendingSecond_ = false;
#endif
    __aicore__ inline void Retire(LocalTensor<uint8_t> scratch, bool secondTriggered) {
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
                Quiet(peer, scratch);
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
    __aicore__ inline void SetProfile(LlReferenceProfile::DeviceProfile* p) { profile_ = p; }
#endif
    __aicore__ inline void Init(const __gm__ Args* args, uint32_t bytes) {
        if (count_) { Trap(); return; }
        args_ = args; bytes_ = bytes; count_ = 0;
        if (!bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { Trap(); return; }
        const uint32_t slots = LlRdma::kCoreBytes / bytes_;
        capacity_ = slots; // use every complete packet slot in the 128KiB GM stage
        if (uint32_t(args_->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stageBase_ = Stage(args_, capacity_ * bytes_);
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }
    __aicore__ inline bool HasStaged() const { return staged_; }
    // Caller submits the previous packet before issuing either the next local
    // copy or next remote staging. It may prepare the next packet first, using
    // a DIFFERENT lane. Single-lane callers select defer=false.
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset,
                                   LocalTensor<uint8_t> packet, bool defer = false) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        pendingSecond_ = profile_ && profile_->SecondRemote();
        if (profile_) profile_->SecondEnqueue();
#endif
        Check(args_, peer, offset, bytes_);
        pendingStage_ = stageBase_ + uint64_t(count_) * bytes_;
        pendingPeer_ = peer; pendingOffset_ = offset; pendingScratch_ = packet;
        DataCopy(Tensor(pendingStage_, true), packet, bytes_);
        SetFlag<HardEvent::MTE3_S>(7U); // same event as the old immediate Sync
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
        WaitFlag<HardEvent::MTE3_S>(7U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) { profile_->StageWaitEnd(pendingSecond_); profile_->SecondPutBegin(pendingSecond_); }
#endif
        // Only after the wait may NIC read GM and SDK overwrite the old UB lane.
        Put(args_, pendingPeer_, RemoteAddress(args_, pendingPeer_, pendingOffset_),
            pendingStage_, bytes_, pendingScratch_);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) profile_->PutReturned(pendingSecond_);
#endif
        staged_ = false;
        if (count_ == capacity_) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            Retire(pendingScratch_, pendingSecond_);
#else
            Retire(pendingScratch_, false);
#endif
        }
    }
    __aicore__ inline void Flush(LocalTensor<uint8_t> scratch) {
        SubmitPending();
        Retire(scratch, false);
    }
};
}
