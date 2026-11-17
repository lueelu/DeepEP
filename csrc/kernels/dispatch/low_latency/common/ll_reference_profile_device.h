#pragma once
#include "kernel_operator.h"
#include "ll_reference_profile_layout.h"
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include "shmem.h"
#endif
#if defined(DEEPEP_LL_STAGE_TIMER)
#error Legacy LL stage/expert instrumentation has been removed
#endif

namespace LlReferenceProfile {
// Count-boundary micro profiling is independent of the schema-4 reference
// record. Mode 1 keeps the shared record limited to the parsed 29 stages; the
// count call sites still compile, so the reference-profile identity and the
// runtime wheel stay one variant.
constexpr uint32_t kCountComputeMode = 1U;
// Reference dispatch style: one clock and one scalar GM store per fixed point.
// Packet timing stays in local Scalar state. Receiver identities use spare UB
// and are copied only after business completion; no live lane is removed.
class DeviceProfile {
public:
    __aicore__ static inline uint64_t EntryClock(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return address ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Init(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        record_ = address ? reinterpret_cast<__gm__ uint64_t*>(address) +
                            AscendC::GetBlockIdx() * kWordsPerCore : nullptr;
        valid_ = 0U;
        for (uint32_t i = 0; i < 32U; ++i) send_[i] = 0U;
        sendRemote_ = sendGroup_ = false;
#endif
    }
    __aicore__ inline bool Enabled() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ != nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Clock() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Value(uint64_t value)
    {
        static_assert(Word <= 21U || (Word >= 24U && Word <= 49U) || Word == 57U || Word == 62U,
                      "reference point/counter index");
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        record_[Word] = value;
        valid_ |= uint64_t(1) << Word;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Mark()
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) Value<Word>(AscendC::GetSystemCycle());
#endif
    }
    // Count-call boundaries. The validated schema-4 record has no count
    // sub-phase storage (its 29 stages come from Marks and the packet detail
    // words), so mode 1 keeps these as typed no-ops. Keeping the names on the
    // profile object is what lets the Count C API read the same source in both
    // wheel identities.
    __aicore__ inline uint64_t CountClock() const { return Clock(); }
    __aicore__ inline uint64_t PublishClock() const { return Clock(); }
    template <uint32_t Word, bool Publish = false> __aicore__ inline void CountBoundary() {}
    template <uint32_t Word> __aicore__ inline void CompletePublishAdd(uint64_t) {}
    template <uint32_t Word> __aicore__ inline void PublishAdd(uint64_t) {}
    __aicore__ inline void CompleteCountCall(bool, uint32_t) {}
    __aicore__ inline bool CountNeedsCalibration() const { return false; }
    __aicore__ inline void CountCalibration(uint64_t, uint64_t) {}
    __aicore__ inline void CountSections(bool, uint32_t, uint64_t, uint64_t, uint64_t,
                                         uint64_t, uint64_t) {}
    __aicore__ inline void CountWirePack(uint64_t) {}
    __aicore__ inline void CountPublish(bool, uint64_t) {}
    __aicore__ inline void CountPostSV(uint64_t) {}
    __aicore__ inline void CountRoute(uint64_t) {}
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void SendBegin(bool remote, bool group, uint32_t token, uint32_t k,
                                    uint32_t expert, uint32_t peer) {
        if (!record_) return;
        sendRemote_ = remote; sendGroup_ = group;
        const uint64_t order = ++send_[24];
        if (remote) {
            const uint64_t n = ++send_[1];
            if (n == 1U) { send_[3] = Clock(); send_[20] = order; }
            send_[21] = order;
            if (n == 2U) {
                send_[7] = Clock(); send_[12] = token; send_[13] = k;
                send_[14] = expert; send_[15] = peer;
            }
        }
        if (group) {
            if (++send_[2] == 1U) { send_[5] = Clock(); send_[22] = order; }
            send_[23] = order;
        }
    }
    __aicore__ inline bool SecondRemote() const {
        return record_ && sendRemote_ && send_[1] == 2U;
    }
    __aicore__ inline void SecondEnqueue() { if (SecondRemote()) send_[8] = Clock(); }
    __aicore__ inline void SecondPutBegin(bool selected) { if (record_ && selected) send_[9] = Clock(); }
    __aicore__ inline void SecondPutBegin() { SecondPutBegin(SecondRemote()); }
    __aicore__ inline void PutReturned(bool selected) {
        if (!record_) return;
        const uint64_t now = Clock();
        send_[4] = now;
        if (selected) send_[10] = now;
    }
    __aicore__ inline void PutReturned() { PutReturned(SecondRemote()); }
    __aicore__ inline void StageQueued(bool selected, bool deferred) {
        if (!record_) return;
        if (deferred) ++send_[29];
        if (selected) { send_[26] = Clock(); send_[31] = deferred ? 1U : 0U; }
    }
    __aicore__ inline void StageWaitBegin(bool selected) { if (record_ && selected) send_[27] = Clock(); }
    __aicore__ inline void StageWaitEnd(bool selected) { if (record_ && selected) send_[28] = Clock(); }
    __aicore__ inline void PreviousSubmit(uint64_t ticks) { if (SecondRemote()) send_[30] += ticks; }
    __aicore__ inline void GroupSubmitted() { if (record_ && sendGroup_) send_[6] = Clock(); }
    __aicore__ inline void MemberQ(uint64_t ticks) { if (SecondRemote()) send_[11] += ticks; }
    __aicore__ inline void FlushDone(uint64_t begin, bool secondTriggered) {
        if (!record_) return;
        const uint64_t end = Clock();
        send_[16] += end - begin; ++send_[17]; send_[18] = end;
        if (secondTriggered) send_[19] = 1U;
    }
    __aicore__ inline void SendLanes(uint32_t lanes) { if (record_) send_[25] = lanes; }
    __aicore__ inline void Detail(uint32_t word, uint64_t value) { if (record_) record_[word] = value; }
    __aicore__ inline GM_ADDR PacketLogAddress() const {
        return record_ ? reinterpret_cast<GM_ADDR>(record_ + kPacketLogWord) : nullptr;
    }
#endif
    __aicore__ inline void Finish(uint32_t rank, uint32_t world, uint32_t bs)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        Mark<57>();  // w57: before_finalize; includes post-business packet-log export
        send_[0] = 1U;
        for (uint32_t i = 0; i < 32U; ++i) record_[64U + i] = send_[i];
        record_[50] = rank;                         // w50: rank (metadata, not a clock)
        record_[51] = AscendC::GetBlockIdx();        // w51: AIV_core
        record_[52] = world;                        // w52: EP
        record_[53] = AscendC::GetBlockNum();        // w53: active_aiv
        record_[54] = bs;                           // w54: BS
        record_[55] = kSchema;                      // w55: schema
        record_[56] = 1U;                           // w56: operation, dispatch only
        record_[58] = valid_;                       // w58: valid_mask, absent != zero
        record_[59] = kMagic;                       // w59: magic
        record_[60] = 1U;                           // w60: boundary_mode, observe
        record_[61] = 1U;                           // w61: complete
        // Python supplies a fresh zeroed tensor; reserved/absent words remain
        // zero. Publish Scalar header/meta after business finish and log DMA.
        // This matches the reference dispatch cache publication, not PIPE_ALL.
        // Pinned SHMEM's helper includes the cacheline containing addr+length.
        // Use the last byte so an aligned record never flushes its neighbor.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_), kPacketLogWord * sizeof(uint64_t) - 1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    }
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __gm__ uint64_t* record_ = nullptr;
    uint64_t valid_ = 0U;
    uint64_t send_[32]{};
    bool sendRemote_ = false, sendGroup_ = false;
#endif
};
}  // namespace LlReferenceProfile
