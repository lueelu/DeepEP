// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026
#pragma once
#include "ll_dispatch_c_api.h"
#include "ll_transport.h"
#include "../common/ll_reference_profile_device.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../../../common/ep_packet_receive_layout.h"
#include "dispatch_server_dedup_receive_tasks.h"
#include "dispatch_server_dedup_receive_packet.h"

namespace DispatchDedup {
class ServerDedupReceiver {
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, GM_ADDR expandX, GM_ADDR dynamicScales,
        GM_ADDR tokenType, GM_ADDR destinationIndex, GM_ADDR relayReadIndex, uint32_t bs,
        uint32_t h, uint32_t k, uint32_t experts, EpServerDedupInputMode inputMode,
        int64_t magic = 0, bool enableDedup = true)
    {
        args_ = args; wireFp8_ = inputMode != EpServerDedupInputMode::Plain16;
        coreNum_ = asc_get_block_num(); recvCoreNum_ = coreNum_ / 2U;
        enableDedup_ = enableDedup; bs_ = bs; k_ = k; experts_ = experts;
        localExperts_ = experts / uint32_t(args_->rankSize);
        receiver_ = asc_get_block_idx() - recvCoreNum_;
        hiddenBytes_ = wireFp8_ ? h : h * 2U;
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, inputMode);
        windowOffset_ = DispatchServerDedupWindowOffset(magic);
        data_.dispatchOffset += windowOffset_;
        expandX_ = expandX; dynamicScales_ = dynamicScales;
        tokenType_ = reinterpret_cast<__gm__ uint32_t*>(tokenType);
        destinationIndex_ = reinterpret_cast<__gm__ uint32_t*>(destinationIndex);
        relayReadIndex_ = reinterpret_cast<__gm__ uint32_t*>(relayReadIndex);
    }
    __aicore__ inline void Process(LlReferenceProfile::DeviceProfile* profile = nullptr)
    {
        // Caller has drained all count UB users. Reuse physical UB from zero.
        LlDispatchC::UbLayout layout;
        const uint32_t prefixBytes = LlDispatchC::Align(experts_ * 4U);
        prefix_ = layout.Allocate<int32_t>(prefixBytes);
        const auto counts = DispatchServerDedupCountLayout(experts_, coreNum_);
        LlDispatchC::Load(prefix_, args_->peerMems[args_->rank] + windowOffset_ +
            counts.cntOffset + counts.prefixOffset, prefixBytes);
        LlDispatchC::Sync<PIPE_MTE2, PIPE_S>();
        if (profile) profile->Mark<30>();  // w30: receive_prefix_loaded_after_MTE2_S
        tasks_.Build(prefix_, uint32_t(args_->rank), uint32_t(args_->rankSize), experts_, bs_, receiver_,
                     layout, recvCoreNum_);
        if (profile) {
            profile->Mark<31>();  // w31: receive_tasks_built
            profile->Value<37>(tasks_.Count());  // w37: receive_task_count
        }
        if (!tasks_.Count()) {
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
        const uint32_t workBytes = kProbeBatch * uint32_t(DispatchDedupAlign((blocks + 7U) / 8U, 32U)) * 4U + 256U;
        flags_ = layout.Allocate<float>(flagsBytes);
        sums_ = layout.Allocate<float>(kProbeBatch * 32U);
        // Preserve the old reduction workspace budget: changing APIs does not grow lane count.
        layout.Allocate(workBytes);
        LlDispatchC::FillBytes(reinterpret_cast<__ubuf__ uint8_t*>(flags_), 0U, flagsBytes);
        LlDispatchC::Sync<PIPE_V, PIPE_MTE2>();
        snapshot_ = layout.Allocate<uint32_t>(kProbeBatch * 4U);
        laneStride_ = uint32_t(data_.tokenStride) + 64U;
        const uint32_t fixed = layout.UsedBytes(), available = (LlDispatchC::kUbBytes - fixed) / laneStride_;
        lanes_ = available < 8U ? available : 8U;
        if (lanes_ > tasks_.Count()) lanes_ = tasks_.Count();
        if (!lanes_) { trap(); return; }
        laneBytes_ = layout.Allocate(lanes_ * laneStride_);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile && profile->Enabled()) {
            traceEnabled_ = true;
            traceCapacity_ = (LlDispatchC::kUbBytes - layout.UsedBytes()) / 32U;
            if (traceCapacity_ > LlReferenceProfile::kPacketTraceCapacity)
                traceCapacity_ = LlReferenceProfile::kPacketTraceCapacity;
            if (traceCapacity_ > tasks_.Count()) traceCapacity_ = tasks_.Count();
            if (traceCapacity_) trace_ = layout.Allocate<uint32_t>(traceCapacity_ * 32U);
        }
#endif
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        uint32_t scan = 0U, issued = 0U, consumed = 0U;
        if (profile) profile->Mark<32>();  // w32: receive_buffers_ready
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint64_t pollCycles = 0U, consumeCycles = 0U, batches = 0U;
        const bool referenceEnabled = profile && profile->Enabled();
#endif
        while (tasks_.PendingCount()) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t pollBegin = referenceEnabled ? profile->Clock() : 0U;
#endif
            ProbeResult probe;
            uint32_t count;
            do {
                if (scan >= tasks_.PendingCount()) scan = 0U;
                const uint32_t remaining = tasks_.PendingCount() - scan;
                count = remaining < kProbeBatch ? remaining : kProbeBatch;
                probe = Probe(scan, count);
                if (!probe.readyMask) scan += count;
            } while (!probe.readyMask);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t pollEnd = referenceEnabled ? profile->Clock() : 0U;
            if (referenceEnabled) {
                if (!batches) profile->Value<42>(pollEnd);  // w42: first_packet_ready
                pollCycles += pollEnd - pollBegin; ++batches; RecordReady(probe.readyMask, count);
            }
#endif
            uint32_t nextLoad = count;
            for (uint32_t i = count; i; --i) {
                const uint32_t index = i - 1U;
                if (!(probe.readyMask & (1U << index))) continue;
                while (issued - consumed < lanes_ && nextLoad) {
                    --nextLoad;
                    if (!(probe.readyMask & (1U << nextLoad))) continue;
                    IssueLoad(snapshot_[nextLoad], issued % lanes_, (probe.keepFlagsMask & (1U << nextLoad)) != 0U);
                    ++issued;
                }
                ConsumeLoaded(snapshot_[index], consumed % lanes_, (probe.keepFlagsMask & (1U << index)) != 0U);
                tasks_.RemovePending(scan + index);
                ++consumed;
            }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (referenceEnabled) consumeCycles += profile->Clock() - pollEnd;
#endif
        }
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        ClearFlags();
        if (profile) profile->Mark<35>();  // w35: receive_lane_wait_clear_issue_end
        asc_sync_pipe(PIPE_ALL);
        if (profile) profile->Mark<36>();  // w36: receive_final_PIPE_ALL_end
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (referenceEnabled) {
            profile->Value<33>(pollCycles);  // w33: receive_packet_poll_accum_cycles
            profile->Value<34>(consumeCycles);  // w34: receive_load_consume_submit_accum_cycles
            profile->Value<41>(batches);  // w41: receive_probe_batch_count
            profile->Value<48>(firstHopTokens_);  // w48: receive_first_hop_tokens
            profile->Value<49>(relayMembers_);  // w49: receive_relay_members
            profile->Value<62>(relayDistinctRanks_);  // w62: receive_relay_distinct_ranks
        }
#endif
    }
    __aicore__ inline __ubuf__ uint8_t* FinalScratch() { return reinterpret_cast<__ubuf__ uint8_t*>(prefix_); }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void ExportPacketTrace(LlReferenceProfile::DeviceProfile* profile)
    {
        if (!profile || !profile->Enabled()) return;
        profile->Detail(96U, traceCapacity_); profile->Detail(97U, traceCapacity_);
        profile->Detail(98U, tasks_.Count()); profile->Detail(99U, traceReady_);
        profile->Detail(100U, traceProcessed_); profile->Detail(101U, lanes_);
        profile->Detail(104U, tasks_.Begin()); profile->Detail(105U, bs_);
        profile->Detail(106U, localExperts_); profile->Detail(107U, tasks_.Count() > traceCapacity_ ? 1U : 0U);
        profile->Detail(102U, profile->Clock());
        if (traceCapacity_) {
            LlDispatchC::Sync<PIPE_S, PIPE_MTE3>();
            LlDispatchC::Store(profile->PacketLogAddress(), trace_, traceCapacity_ * 32U, 1U, 0U, 0U, true);
            LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
        }
        profile->Detail(103U, profile->Clock());
    }
#endif
private:
    static constexpr uint32_t kProbeBatch = 16U;
    struct ProbeResult { uint32_t readyMask = 0U, keepFlagsMask = 0U; };
    __aicore__ inline GM_ADDR Slot(uint32_t j)
    {
        return args_->peerMems[args_->rank] + data_.dispatchOffset + uint64_t(tasks_.Position(j)) * data_.tokenStride;
    }
    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count)
    {
        const uint32_t blocks = uint32_t(data_.blockCount);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t j = tasks_.PendingIndex(start + i);
            snapshot_[i] = j;
            ReadServerDedupFlags(Slot(j), blocks, flags_ + i * blocks * 8U, LlTransport::Rdma(args_));
        }
        LlDispatchC::Sync<PIPE_MTE2, PIPE_V>();
        LlDispatchC::SumPacketFlags(sums_, flags_, count, blocks * 8U);
        LlDispatchC::Sync<PIPE_V, PIPE_S>();
        ProbeResult result;
        for (uint32_t i = 0; i < count; ++i) {
            const auto state = PacketReceiveLayout::Decode(sums_[i * 8U], blocks);
            if (state != PacketReceiveLayout::State::Pending) result.readyMask |= 1U << i;
            if (state == PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask |= 1U << i;
        }
        return result;
    }
    __aicore__ inline void IssueLoad(uint32_t j, uint32_t lane, bool keepFlags)
    {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        auto* wire = laneBytes_ + lane * laneStride_;
        if (keepFlags) LlDispatchC::Load(wire, Slot(j), uint32_t(data_.tokenStride), 1U, 0U, 0U, LlTransport::Rdma(args_));
        else LlDispatchC::Load(wire, Slot(j), 480U, uint32_t(data_.blockCount), 32U, 0U, LlTransport::Rdma(args_));
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, event_t(lane));
        asc_sync_notify(PIPE_MTE2, PIPE_S, event_t(lane));
    }
    __aicore__ inline void ConsumeLoaded(uint32_t j, uint32_t lane, bool keepFlags)
    {
        asc_sync_wait(PIPE_MTE2, PIPE_S, event_t(lane));
        auto* wire = laneBytes_ + lane * laneStride_;
        const uint32_t row = tasks_.Begin() + j * tasks_.Stride();
        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        RecordProcessed(j, wire, headerOffset, keepFlags);
#endif
        if (!enableDedup_ && !keepFlags) {
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
            LlDispatchC::Store(expandX_ + uint64_t(row) * hiddenBytes_, wire, hiddenBytes_);
            if (wireFp8_) LlDispatchC::Store(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire + data_.quantCount, uint32_t(data_.scaleBytes));
            LlDispatchC::Store(destinationIndex_ + uint64_t(row) * 3U, wire + headerOffset, 12U);
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
            return;
        }
        auto* type = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride);
        auto* destination = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride + 32U);
        ServerDedupPacketTail tail(wire, headerOffset,
            uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
        bool firstHop = false;
        if (enableDedup_) firstHop = BuildServerDedupMetadata(type, destination, tail, tasks_.Position(j)) ==
            uint32_t(EpServerDedupTokenType::FirstHop);
        else for (uint32_t i = 0; i < 3U; ++i) destination[i] = tail.Header()[i];
        if (firstHop) {
            PrepareServerDedupForward(tail);
            reinterpret_cast<__ubuf__ float*>(wire)[PacketReceiveLayout::kMarkerOffset / 4U] =
                PacketReceiveLayout::kCompact;
        }
        LlDispatchC::Sync<PIPE_S, PIPE_MTE3>();
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
        if (keepFlags) {
            WriteServerDedupPayloadRange(expandX_ + uint64_t(row) * hiddenBytes_, wire, 0U, hiddenBytes_);
            if (wireFp8_) WriteServerDedupPayloadRange(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire, uint32_t(data_.quantCount), uint32_t(data_.scaleBytes));
        } else {
            LlDispatchC::Store(expandX_ + uint64_t(row) * hiddenBytes_, wire, hiddenBytes_);
            if (wireFp8_) LlDispatchC::Store(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire + data_.quantCount, uint32_t(data_.scaleBytes));
        }
        if (enableDedup_) LlDispatchC::Store(tokenType_ + row, type, 4U);
        LlDispatchC::Store(destinationIndex_ + uint64_t(row) * 3U, destination, 12U);
        if (firstHop) {
            LlDispatchC::Store(relayReadIndex_ + uint64_t(row) * k_ * 2U, tail.ReadIndex(), k_ * 8U);
            const uint32_t source = tail.Header()[0U], ranks = uint32_t(args_->rankSize), members = tail.ReadIndex()[0U];
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            ++firstHopTokens_; relayMembers_ += members;
            uint64_t seen[4]{};
            for (uint32_t i = 0; i < members; ++i) {
                const uint32_t peer = (tail.Entry(i)[0U] & 0xffffU) / localExperts_, slot = peer >> 6U;
                const uint64_t bit = uint64_t(1) << (peer & 63U);
                if (slot < 4U && !(seen[slot] & bit)) { seen[slot] |= bit; ++relayDistinctRanks_; }
            }
#endif
            for (uint32_t i = 0; i < members; ++i) {
                const uint32_t index = tail.Entry(i)[0U], expert = index & 0xffffU, q = index >> 16U;
                const uint32_t destinationRank = expert / localExperts_;
                LlTransport::Packet(args_, destinationRank, DispatchServerDedupDispatchOffset(data_, source,
                    destinationRank, expert % localExperts_, q, ranks, localExperts_, bs_),
                    wire, uint32_t(data_.tokenStride), lane);
            }
        }
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
    }
    __aicore__ inline void ClearFlags()
    {
        LlDispatchC::FillWords(reinterpret_cast<__ubuf__ uint32_t*>(flags_), 0U, uint32_t(data_.blockCount) * 8U);
        LlDispatchC::Sync<PIPE_V, PIPE_MTE3>();
        for (uint32_t j = 0; j < tasks_.Count(); ++j)
            LlDispatchC::Store(Slot(j) + 480U, flags_, 32U, uint32_t(data_.blockCount), 480U, 0U, LlTransport::Rdma(args_));
    }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void RecordReady(uint32_t mask, uint32_t count)
    {
        ++traceBatches_;
        for (uint32_t i = 0; i < count; ++i) if (mask & (1U << i)) {
            const uint32_t j = snapshot_[i]; ++traceReady_;
            if (j < traceCapacity_) { trace_[j * 8U] = traceReady_; trace_[j * 8U + 1U] = traceBatches_; }
        }
    }
    __aicore__ inline void RecordProcessed(uint32_t j, __ubuf__ uint8_t* wire, uint32_t headerOffset, bool keepFlags)
    {
        if (traceEnabled_) ++traceProcessed_;
        if (!traceEnabled_ || j >= traceCapacity_) return;
        auto* record = trace_ + j * 8U;
        auto* header = reinterpret_cast<__ubuf__ uint32_t*>(wire + headerOffset);
        const uint32_t role = header[3U], members = role == uint32_t(EpServerDedupTokenType::FirstHop) ?
            reinterpret_cast<__ubuf__ uint32_t*>(wire +
                (keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset))[0U] : 0U;
        record[2U] = traceProcessed_; record[3U] = header[0U]; record[4U] = header[1U];
        record[5U] = header[2U]; record[6U] = role | (members << 8U); record[7U] = tasks_.Position(j);
    }
    __ubuf__ uint32_t* trace_ = nullptr;
    bool traceEnabled_ = false;
    uint32_t traceCapacity_ = 0U, traceReady_ = 0U, traceProcessed_ = 0U, traceBatches_ = 0U;
    uint32_t firstHopTokens_ = 0U, relayMembers_ = 0U, relayDistinctRanks_ = 0U;
#endif
    const __gm__ CommArgs* args_ = nullptr;
    bool enableDedup_ = false, wireFp8_ = false;
    uint64_t windowOffset_ = 0U;
    uint32_t recvCoreNum_ = 0U, coreNum_ = 0U, bs_ = 0U, k_ = 0U, experts_ = 0U;
    uint32_t localExperts_ = 0U, receiver_ = 0U, hiddenBytes_ = 0U, laneStride_ = 0U, lanes_ = 0U;
    EpServerDedupDataLayout data_{};
    ServerDedupReceiveTasks tasks_;
    __ubuf__ int32_t* prefix_ = nullptr;
    __ubuf__ float *flags_ = nullptr, *sums_ = nullptr;
    __ubuf__ uint32_t* snapshot_ = nullptr;
    __ubuf__ uint8_t* laneBytes_ = nullptr;
    __gm__ uint8_t *expandX_ = nullptr, *dynamicScales_ = nullptr;
    __gm__ uint32_t *tokenType_ = nullptr, *destinationIndex_ = nullptr, *relayReadIndex_ = nullptr;
};
}  // namespace DispatchDedup
