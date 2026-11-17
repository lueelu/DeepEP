// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026
#ifndef DISPATCH_SERVER_DEDUP_SEND_H
#define DISPATCH_SERVER_DEDUP_SEND_H
#include "ll_dispatch_c_api.h"
#include "ll_transport.h"
#include "ll_dispatch_rdma_batch.h"
#include "dispatch_dedup_mxfp8_quant.h"
#include "../common/ll_reference_profile_device.h"
#include "../common/ep_memory_server_dedup_data_layout.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../common/ll_send_order.h"
#include "../common/ll_fullmesh_first_hop.h"
#include "../common/ll_sender_route_plan.h"
#include "../common/ll_sender_input_reuse.h"
#include "../../../common/ep_packet_receive_layout.h"

namespace DispatchDedup {
template <class XType, uint32_t Group = 16U>
class ServerDedupSender {
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, GM_ADDR x, GM_ADDR ids, GM_ADDR weights,
        GM_ADDR sourceMask, GM_ADDR sourceWeights, uint32_t bs, uint32_t h, uint32_t k,
        uint32_t experts, EpServerDedupInputMode inputMode, bool e4m3, int64_t magic = 0,
        GM_ADDR inputScales = nullptr, bool enableDedup = true, uint32_t coreNum = 0U)
    {
        args_ = args;
        prequantized_ = inputMode == EpServerDedupInputMode::PrequantizedFp8Packs;
        needQuantize_ = inputMode == EpServerDedupInputMode::Quantize16;
        e4m3_ = e4m3;
        inputRowBytes_ = prequantized_ ? h : h * sizeof(XType);
        sendCores_ = (coreNum ? coreNum : asc_get_block_num()) / 2U;
        enableDedup_ = enableDedup;
        fullMeshFirstHop_ = Group == kLlPhysicalRanksPerHost && LlTransport::Rdma(args);
        bs_ = bs; h_ = h; k_ = k; experts_ = experts;
        localExperts_ = experts / args->rankSize;
        const uint32_t core = asc_get_block_idx(), base = bs / sendCores_, remainder = bs % sendCores_;
        count_ = base + (core < remainder ? 1U : 0U);
        start_ = core * base + (core < remainder ? core : remainder);
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, inputMode);
        data_.dispatchOffset += DispatchServerDedupWindowOffset(magic);
        x_ = x; inputScales_ = inputScales;
        ids_ = reinterpret_cast<__gm__ int32_t*>(ids);
        weights_ = reinterpret_cast<__gm__ float*>(weights);
        maskOut_ = reinterpret_cast<__gm__ uint32_t*>(sourceMask);
        weightOut_ = reinterpret_cast<__gm__ float*>(sourceWeights);
    }

    __aicore__ inline void Process(uint32_t reservedBytes,
        LlReferenceProfile::DeviceProfile* profile = nullptr, LlTransport::CountRetirement* retirement = nullptr)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        packetProfile_ = profile;
        rdmaBatch_.SetProfile(profile);
#endif
        if (profile && profile->Enabled()) {
            const uint32_t routes = bs_ * k_;
            profile->Value<43>(enableDedup_ ? count_ * k_ :  // w43: send_route_count
                routes / sendCores_ + (asc_get_block_idx() < routes % sendCores_ ? 1U : 0U));
        }
        if (!LlSenderHasWork(enableDedup_, count_, bs_ * k_, asc_get_block_idx())) return;
        // The count image at UB0 remains owned by MTE3 until count retirement.
        LlDispatchC::UbLayout layout(reservedBytes);
        if (LlTransport::Rdma(args_)) rdmaBatch_.SetCountRetirement(retirement);
        inputBytes_ = needQuantize_ ? uint32_t(data_.quantCount * sizeof(XType)) :
                                    uint32_t(data_.packedPayloadBytes);
        input_[0] = layout.Allocate(inputBytes_);
        input_[1] = layout.Allocate(inputBytes_);
        if (needQuantize_) {
            LlDispatchC::FillBytes(input_[0], 0U, 2U * inputBytes_);
            LlDispatchC::Sync<PIPE_V, PIPE_MTE2>();
        }
        const uint32_t routeBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 256U));
        const uint32_t qBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 32U));
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        const uint32_t quantWorkBytes = uint32_t(DispatchDedupAlign(
            DispatchDedupAlign(scaleCount, 32U) * 4U + uint64_t(scaleCount) * 2U, 32U));
        const uint32_t scratchBytes = needQuantize_ && quantWorkBytes > qBytes ? quantWorkBytes : qBytes;
        routeIds_ = layout.Allocate<int32_t>(routeBytes);
        qDst_ = layout.Allocate<uint32_t>(scratchBytes);
        // Preserve the admitted UB footprint and hence lane count.
        layout.Allocate(scratchBytes);
        layout.Allocate(qBytes);
        if (needQuantize_) quant_ = layout.Allocate(uint32_t(data_.packedPayloadBytes));
        routeWeights_ = layout.Allocate<float>(count_ * k_ * 4U);
        sourceWeights_ = layout.Allocate<float>(count_ * k_ * 4U);
        sourceMasks_ = layout.Allocate<uint32_t>(count_ * 4U);
        directTasks_ = layout.Allocate<uint32_t>(count_ * k_ * 16U);
        const uint32_t serverBytes = LlDispatchC::Align(args_->rankSize / Group * 4U);
        // Shared phase-wise: Classify uses [server] as per-token scratch, then
        // the direct-task list reuses it as head links. The two phases never
        // overlap, so one array keeps the admitted UB footprint.
        serverState_ = layout.Allocate<uint32_t>(serverBytes < 64U ? 64U : serverBytes);
        const uint32_t remoteServers = args_->rankSize / Group - 1U;
        const uint32_t maxGroups = remoteServers < k_ / 2U ? remoteServers : k_ / 2U;
        if (maxGroups) firstHopMasks_ = layout.Allocate<uint32_t>(maxGroups * 4U);
        if (LlTransport::Rdma(args_)) {
            const uint32_t tokenStride = uint32_t(data_.tokenStride);
            if (!tokenStride || tokenStride % 512U || tokenStride > LlRdma::kCoreBytes) { trap(); return; }
            const uint32_t stageSlots = LlRdma::kCoreBytes / tokenStride;
            rdmaWorkspace_ = layout.Allocate<uint8_t>(
                (stageSlots + 1U) * LlTransport::DispatchRdmaBatch::kWorkspaceBytes);
            rdmaBatch_.SetWorkspace(rdmaWorkspace_);
            rdmaBatch_.Init(args_, tokenStride);
        }
        const uint32_t available = (LlDispatchC::kUbBytes - layout.UsedBytes()) / uint32_t(data_.tokenStride);
        lanes_ = available < 8U ? available : 8U;
        if (!lanes_) { trap(); return; }
        out_ = layout.Allocate(lanes_ * uint32_t(data_.tokenStride));
        hasScratch_ = true;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->SendLanes(lanes_);
#endif
        LlDispatchC::Load(routeIds_, ids_, bs_ * k_ * 4U);
        if (count_) LlDispatchC::Load(routeWeights_, weights_ + uint64_t(start_) * k_, count_ * k_ * 4U);
        LlDispatchC::Sync<PIPE_MTE2, PIPE_S>();
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            LlDispatchC::FillPacketFlags(out_ + lane * data_.tokenStride, uint32_t(data_.blockCount));
            asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
        }
        asc_sync_pipe(PIPE_V);
        if (profile) profile->Mark<38>();  // w38: send_buffers_routes_ready
        if (enableDedup_) SendWithDedup();
        else SendWithoutDedup();
        rdmaBatch_.SubmitPending();
        if (profile) profile->Mark<39>();  // w39: send_packets_issue_end
        SenderInputOperations operations{*this};
        inputReuse_.Finish(operations);
        rdmaBatch_.Flush();
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
        LlDispatchC::Sync<PIPE_S, PIPE_MTE3>();
        if (enableDedup_) LlDispatchC::Store(maskOut_ + start_, sourceMasks_, count_ * 4U);
        if (count_) LlDispatchC::Store(weightOut_ + uint64_t(start_) * k_,
            enableDedup_ ? sourceWeights_ : routeWeights_, count_ * k_ * 4U);
        LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
        if (profile) profile->Mark<40>();  // w40: send_metadata_end_after_MTE3_S
    }
    __aicore__ inline bool HasFinalScratch() const { return hasScratch_; }
    __aicore__ inline __ubuf__ uint8_t* FinalScratch() { return out_; }

private:
    __aicore__ inline uint32_t FirstHop(uint32_t base, uint32_t mask)
    {
        LlFullMeshFirstHop mesh;
        for (uint32_t k = 0; k < k_; ++k) {
            if (!(mask & (1U << k))) continue;
            if (!fullMeshFirstHop_) return k;
            mesh.Add(uint32_t(routeIds_[base + k]) / localExperts_, k);
        }
        return mesh.FirstK();
    }
    __aicore__ inline void SendWithDedup()
    {
        const uint32_t rankInGroup = uint32_t(args_->rank) % Group, sendCore = asc_get_block_idx();
        for (uint32_t token = 0; token < count_; ++token) {
            uint32_t directMask = 0U, firstHopCount = 0U;
            Classify(token, directMask, firstHopCount);
            uint32_t sourceMask = directMask;
            for (uint32_t group = 0; group < firstHopCount; ++group) {
                const uint32_t scheduled = LlFirstHopOrder(firstHopCount, rankInGroup, sendCore, start_ + token, group);
                const uint32_t mask = firstHopMasks_[scheduled];
                const uint32_t firstK = FirstHop((start_ + token) * k_, mask);
                const uint32_t index = (start_ + token) * k_ + firstK, expert = uint32_t(routeIds_[index]);
                SendPacket(start_ + token, firstK, expert, CalTokenSendExpertCnt(expert, index), mask);
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
                directTasks_[directCount_ * 4U] = token;
                directTasks_[directCount_ * 4U + 1U] = k;
                directTasks_[directCount_ * 4U + 2U] = uint32_t(routeIds_[(start_ + token) * k_ + k]);
                ++directCount_;
            }
            sourceMasks_[token] = sourceMask;
        }
        const uint32_t serverNum = args_->rankSize / Group, expertsPerServer = localExperts_ * Group;
        const uint32_t ownServer = args_->rank / Group;
        for (uint32_t server = 0; server < serverNum; ++server) serverState_[server] = UINT32_MAX;
        for (uint32_t next = directCount_; next; --next) {
            const uint32_t task = next - 1U, server = directTasks_[task * 4U + 2U] / expertsPerServer;
            directTasks_[task * 4U + 3U] = serverState_[server];
            serverState_[server] = task;
        }
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = LlDirectGroupOrder(
                serverNum, ownServer, rankInGroup, sendCore, step, fullMeshFirstHop_);
            for (uint32_t task = serverState_[server]; task != UINT32_MAX; task = directTasks_[task * 4U + 3U]) {
                const uint32_t expert = directTasks_[task * 4U + 2U], token = directTasks_[task * 4U];
                const uint32_t k = directTasks_[task * 4U + 1U];
                SendPacket(start_ + token, k, expert, CalTokenSendExpertCnt(expert, (start_ + token) * k_ + k), 0U);
                sourceWeights_[token * k_ + k] = routeWeights_[token * k_ + k];
            }
        }
    }
    __aicore__ inline void SendWithoutDedup()
    {
        const uint32_t total = bs_ * k_, core = asc_get_block_idx(), base = total / sendCores_;
        const uint32_t remainder = total % sendCores_, begin = core * base + (core < remainder ? core : remainder);
        const uint32_t count = base + (core < remainder ? 1U : 0U), serverNum = args_->rankSize / Group;
        if (serverNum < 16U) {
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_[i]);
                SendPacket(i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
            return;
        }
        const uint32_t expertsPerServer = localExperts_ * Group, ownServer = args_->rank / Group;
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = EpServerOrder::ServerDedupDispatchDirectOrder(serverNum, ownServer, step);
            const uint32_t firstExpert = server * expertsPerServer;
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_[i]);
                if (expert < firstExpert || expert >= firstExpert + expertsPerServer) continue;
                SendPacket(i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
        }
    }
    __aicore__ inline void Classify(uint32_t token, uint32_t& directMask, uint32_t& groups)
    {
        uint32_t seen = 0U, grouped = 0U;
        for (uint32_t k = 0; k < k_; ++k) {
            const uint32_t expert = uint32_t(routeIds_[(start_ + token) * k_ + k]);
            const uint32_t server = expert / localExperts_ / Group, bit = 1U << k, serverBit = 1U << server;
            if (server == uint32_t(args_->rank) / Group) directMask |= bit;
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
    __aicore__ inline uint32_t CalTokenSendExpertCnt(uint32_t expert, uint32_t index)
    {
        if (index < k_) return 0U;
        LlDispatchC::RoutePrefixCount(qDst_, routeIds_, index, int32_t(expert));
        LlDispatchC::Sync<PIPE_V, PIPE_S>();
        return qDst_[0];
    }
    template <bool E4M3>
    __aicore__ inline void QuantMxfp8(__ubuf__ uint8_t* output, __ubuf__ XType* input)
    {
        const uint32_t scales = uint32_t(data_.quantCount / 32U);
        auto* maxExp = reinterpret_cast<__ubuf__ uint16_t*>(qDst_);
        auto* halfScale = reinterpret_cast<__ubuf__ uint16_t*>(qDst_ + DispatchDedupAlign(scales, 32U));
        DispatchDedupMxfp8Quant::ComputeMaxExp(input, maxExp, uint32_t(data_.quantCount));
        DispatchDedupMxfp8Quant::ComputeScale<E4M3>(maxExp,
            reinterpret_cast<__ubuf__ uint16_t*>(output + data_.quantCount), halfScale, scales);
        DispatchDedupMxfp8Quant::ComputeFp8Data<XType, E4M3>(
            input, halfScale, reinterpret_cast<__ubuf__ int8_t*>(output), uint32_t(data_.quantCount));
    }
    __aicore__ inline void FillTail(__ubuf__ uint8_t* payload, uint32_t token, uint32_t firstK, uint32_t mask)
    {
        auto* header = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.headerPayloadOffset);
        auto* index = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.readIndexPayloadOffset);
        header[0] = uint32_t(args_->rank); header[1] = token; header[2] = firstK;
        header[3] = uint32_t(mask ? EpServerDedupTokenType::FirstHop : EpServerDedupTokenType::Direct);
        if (!enableDedup_) { index[0] = 0U; return; }
        auto* weights = reinterpret_cast<__ubuf__ float*>(index);
        token -= start_;
        weights[1] = routeWeights_[token * k_ + firstK];
        uint32_t entries = 0U;
        const uint32_t members = LlRelayMemberMask(mask, firstK);
        for (uint32_t k = 0; members && k < k_; ++k) if (members & (1U << k)) {
            const uint32_t routeIndex = (start_ + token) * k_ + k, expert = uint32_t(routeIds_[routeIndex]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const bool traceQ = packetProfile_ && packetProfile_->SecondRemote();
            const uint64_t qBegin = traceQ ? packetProfile_->Clock() : 0U;
#endif
            const uint32_t q = CalTokenSendExpertCnt(expert, routeIndex);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (traceQ) packetProfile_->MemberQ(packetProfile_->Clock() - qBegin);
#endif
            ++entries;
            index[entries * 2U] = expert | (q << 16U);
            weights[entries * 2U + 1U] = routeWeights_[token * k_ + k];
        }
        if (prequantized_) {
            for (uint32_t stale = entries + 1U; stale <= cachedTailEntries_; ++stale) {
                index[stale * 2U] = 0U; index[stale * 2U + 1U] = 0U;
            }
            cachedTailEntries_ = entries;
        }
        index[0] = entries;
    }
    struct SenderInputOperations {
        ServerDedupSender& sender;
        __aicore__ inline void LoadInput(uint32_t token) { sender.LoadTokenInput(token); }
        // The cache is read by V packing. Retire it before either slot or quant_ is overwritten.
        __aicore__ inline void ReleaseInput() { LlDispatchC::Sync<PIPE_V, PIPE_MTE2>(); }
        __aicore__ inline void WaitPackedRead() { LlDispatchC::Sync<PIPE_V, PIPE_S>(); }
    };
    __aicore__ inline void LoadTokenInput(uint32_t token)
    {
        auto* input = input_[inputSequence_++ % 2U];
        if (prequantized_) {
            const auto padding = LlPlanPrequantPadding(h_, uint32_t(data_.quantCount),
                uint32_t(data_.scaleBytes), uint32_t(data_.packedPayloadBytes));
            LlDispatchC::FillPrequantPadding(input, padding);
            LlDispatchC::Sync<PIPE_V, PIPE_MTE2>();
            cachedTailEntries_ = 0U;
        }
        LlDispatchC::Load(input, x_ + uint64_t(token) * inputRowBytes_, inputRowBytes_);
        if (prequantized_) LlDispatchC::Load(input + data_.quantCount,
            inputScales_ + uint64_t(token) * data_.scaleBytes, uint32_t(data_.scaleBytes));
        if (needQuantize_) {
            LlDispatchC::Sync<PIPE_MTE2, PIPE_V>();
            if (e4m3_) QuantMxfp8<true>(quant_, reinterpret_cast<__ubuf__ XType*>(input));
            else QuantMxfp8<false>(quant_, reinterpret_cast<__ubuf__ XType*>(input));
            LlDispatchC::Sync<PIPE_V, PIPE_S>();
            cachedPayload_ = quant_;
        } else {
            LlDispatchC::Sync<PIPE_MTE2, PIPE_S>();
            cachedPayload_ = input;
        }
    }
    __aicore__ inline void SendPacket(uint32_t token, uint32_t firstK, uint32_t expert, uint32_t q, uint32_t mask)
    {
        const uint32_t destination = expert / localExperts_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->SendBegin(LlTransport::RemoteWrite(args_, destination),
            destination / Group == uint32_t(args_->rank) / Group, token, firstK, expert, destination);
#endif
        SenderInputOperations operations{*this};
        inputReuse_.Prepare(token, operations);
        FillTail(cachedPayload_, token, firstK, mask);
        LlDispatchC::Sync<PIPE_S, PIPE_V>();
        const uint32_t lane = packets_ % lanes_;
        asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
        auto* out = out_ + lane * data_.tokenStride;
        LlDispatchC::PackPayload(out, cachedPayload_, uint32_t(data_.blockCount));
        // Publish the layout marker through V too; avoid a simultaneous Scalar store.
        LlDispatchC::SetPacketMarker(out, mask ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact);
        LlDispatchC::Sync<PIPE_V, PIPE_MTE3>(event_t(lane));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const bool tracePrevious = packetProfile_ && packetProfile_->SecondRemote() && rdmaBatch_.HasStaged();
        const uint64_t previousBegin = tracePrevious ? packetProfile_->Clock() : 0U;
#endif
        rdmaBatch_.SubmitPending();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (tracePrevious) packetProfile_->PreviousSubmit(packetProfile_->Clock() - previousBegin);
#endif
        const uint64_t offset = DispatchServerDedupDispatchOffset(data_, args_->rank, destination,
            expert % localExperts_, q, args_->rankSize, localExperts_, bs_);
#ifndef DEEPEP_LL_NO_COMM
        if (LlTransport::RemoteWrite(args_, destination)) rdmaBatch_.Enqueue(destination, offset, out, lanes_ > 1U);
        else
#endif
            LlTransport::Packet(args_, destination, offset, out, uint32_t(data_.tokenStride), 7U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->GroupSubmitted();
#endif
        asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
        ++packets_;
    }
    const __gm__ CommArgs* args_ = nullptr;
    LlTransport::DispatchRdmaBatch rdmaBatch_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    LlReferenceProfile::DeviceProfile* packetProfile_ = nullptr;
#endif
    LlSenderInputReuse inputReuse_;
    __ubuf__ uint8_t* input_[2]{};
    __ubuf__ uint8_t *cachedPayload_ = nullptr, *quant_ = nullptr, *out_ = nullptr;
    __ubuf__ uint8_t* rdmaWorkspace_ = nullptr;
    __ubuf__ int32_t* routeIds_ = nullptr;
    __ubuf__ uint32_t *qDst_ = nullptr, *sourceMasks_ = nullptr, *directTasks_ = nullptr;
    __ubuf__ uint32_t *serverState_ = nullptr, *firstHopMasks_ = nullptr;
    __ubuf__ float *routeWeights_ = nullptr, *sourceWeights_ = nullptr;
    __gm__ uint8_t *x_ = nullptr, *inputScales_ = nullptr;
    __gm__ int32_t* ids_ = nullptr;
    __gm__ float *weights_ = nullptr, *weightOut_ = nullptr;
    __gm__ uint32_t* maskOut_ = nullptr;
    uint32_t inputSequence_ = 0U, inputBytes_ = 0U, cachedTailEntries_ = 0U;
    bool hasScratch_ = false, enableDedup_ = false, fullMeshFirstHop_ = false;
    bool prequantized_ = false, needQuantize_ = false, e4m3_ = true;
    uint32_t sendCores_ = 0U, bs_ = 0U, h_ = 0U, k_ = 0U, experts_ = 0U, localExperts_ = 0U;
    uint32_t start_ = 0U, count_ = 0U, inputRowBytes_ = 0U, lanes_ = 0U, packets_ = 0U, directCount_ = 0U;
    EpServerDedupDataLayout data_{};
};
}  // namespace DispatchDedup
#endif
