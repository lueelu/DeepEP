// Copyright (c) 2026, Lu Lu

// Execute the production Dispatch, transport completion, and memory layout.
// Only device primitives and the metadata pack/relay entry points are
// doubled. This validates byte ranges and protocol order, not NIC visibility.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>
#include "kernel_operator.h"
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_roce.hpp"

#define MEGA_MOE_SEND_MASK_H
constexpr int AIC = 0;
constexpr int AIV = 1;
constexpr int g_coreType = AIV;

namespace DispatchTest {
inline uint32_t payloadReadyBase = 0U;
inline uint32_t ownerSlot = 0U;
inline uint32_t ddrBarriers = 0U;
inline std::array<uint32_t, 16> broadcasts{};
inline bool notificationOpen = false;
inline const uint32_t* helperMask = nullptr;
inline std::vector<int32_t*> noticeDestinations;
inline const int32_t* noticeSource = nullptr;
inline uint32_t noticeGroups = 0;
}  // namespace DispatchTest
namespace AscendC {
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
inline void DataCopy(int32_t* dst, const LocalTensor<int32_t>& src, uint32_t size)
{
    using namespace DispatchTest;
    Check(notificationOpen && size == 8 && noticeDestinations.size() < 8,
          "notification submitted before scalar release");
    if (!noticeSource) noticeSource = src.data.data();
    Check(noticeSource == src.data.data(), "notification staging changed before retirement");
    noticeDestinations.push_back(dst);
}
#endif
enum class MemDsbT { DDR };
template <MemDsbT Kind>
void DataSyncBarrier()
{
    static_assert(Kind == MemDsbT::DDR);
    DispatchTest::Check(RoceTest::mteCanRead, "broadcast preceded MTE acquire");
    for (auto tail : DispatchTest::tails)
        DispatchTest::Check(DispatchTest::ReadWord(DispatchTest::local, tail) == DispatchTest::sequence,
                            "broadcast preceded current payload markers");
    ++DispatchTest::ddrBarriers;
}
}  // namespace AscendC

namespace MegaMoeImpl {
using namespace AscendC;
struct MoeStageCommonConfig {
    uint32_t rankId;
    uint32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
    uint32_t moeExpertPerRank;
    uint32_t worldSize = 32;
};
struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};
struct Params {
    MegaMoeTilingData* tilingData;
    PeermemInfo peermemInfo;
    GM_ADDR probsGmAddr;
};
struct SendMaskConfig {
    uint64_t railTopkIdsSlotBytes;
};
struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor{TPosition::VECCALC, 0U, 1024U}, topkIndexTensor{TPosition::VECCALC, 0U, 1024U};
};
inline void StoreCompactPlan(const MoeStageCommonConfig&, const Params&, LocalTensor<int32_t>&)
{
    throw std::logic_error("compact vector path requires the route fixture");
}
inline void AcquireRoutePackets(GM_ADDR data, uint32_t count, uint32_t seq, LocalTensor<int32_t>&) {
    for (uint32_t i = 1; i <= count; ++i) {
        uint32_t tail;
        std::memcpy(&tail, data + i * RAIL_PACKET_BYTES - 4, 4);
        DispatchTest::Check(tail == seq, "stale MTE packet");
    }
}
inline GM_ADDR g_winRankAddr_[32]{};

inline int32_t* HierarchicalSignalAddress(const PeermemInfo& info, HierarchicalSignal signal)
{
    return reinterpret_cast<int32_t*>(info.hierarchicalControlPtr + HierarchicalSignalOffset(signal));
}
inline void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    using namespace DispatchTest;
    const auto offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(value));
    if (offset >= payloadReadyBase && offset < payloadReadyBase + 16U * 16U * 512U) {
        const uint32_t window = (offset - payloadReadyBase) / (16U * 512U);
        const uint32_t slotBytes = (offset - payloadReadyBase) % (16U * 512U);
        Check(window < 16U && slotBytes == ownerSlot * 512U, "broadcast wrote another gateway slot or host");
        Check((RoceTest::blockIndex == 53 || RoceTest::blockIndex == 55) && ddrBarriers % 2U == 1U &&
                  RoceTest::mteCanRead,
              "broadcast preceded owner receive acquire/DDR boundary");
        Check(static_cast<uint32_t>(value) == sequence && broadcasts[window]++ == 0U,
              "broadcast cleared, truncated or duplicated an epoch");
    }
    Check(offset != relayReady, "payload readiness must come directly from the gateway's network commit");
    if (offset == metaReady) {
        Check(RoceTest::blockIndex == static_cast<int>(receiveLeaderJob),
              "readiness was not published by the receive leader");
        Check(metadataWaits >= 1U && RoceTest::mteCanRead,
              "metadata readiness preceded the independent receive acquire");
    }
    std::memcpy(address, &value, sizeof(value));
}
inline int32_t ReadGmByPassDCache(const int32_t* address)
{
    int32_t value;
    std::memcpy(&value, address, sizeof(value));
    return value;
}
inline void WaitUntilGmFlagEquals(int32_t* address, int32_t value)
{
    if (DispatchTest::completePacking) {
        auto pack = std::move(DispatchTest::completePacking);
        DispatchTest::completePacking = nullptr;
        pack();
    }
    DispatchTest::Check(ReadGmByPassDCache(address) == value, "Dispatch skipped local rail readiness");
    RoceTest::signalObserved = true;
}
template <HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    AscendC::SetFlag<Event>(Id);
    AscendC::WaitFlag<Event>(Id);
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
    using namespace DispatchTest;
    if constexpr (Event == HardEvent::S_MTE3) {
        Check(!notificationOpen && !noticeSource, "notification buffer still live");
        for (auto tail : tails) Check(ReadWord(local, tail) == sequence, "HNS notification before lane completion");
        notificationOpen = true;
    }
    if constexpr (Event == HardEvent::MTE3_S) {
        if (notificationOpen) {
            Check(helperMask && *helperMask != 255U && noticeDestinations.size() == 8,
                  "published before all writes retired");
            for (auto* dst : noticeDestinations) {
                for (uint32_t i = 0; i < 8; ++i)
                    Check(static_cast<uint32_t>(noticeSource[i]) == sequence, "staging reused early");
                std::copy_n(noticeSource, 8, dst);
            }
            noticeSource = nullptr;
            noticeDestinations.clear();
            notificationOpen = false;
            ++noticeGroups;
        }
    }
#endif
}

// Rail bytes are prepared by the fixture; validate the actual packer's identity,
// destination and publication order when production Dispatch invokes it.
inline void SendTopkIdsToRailServer(const MoeStageCommonConfig& common, GM_ADDR*, const SendMaskConfig& config,
                                    const Params& params, SendMaskScratch&, uint32_t targetServerIdx, uint32_t jobIdx)
{
    using namespace DispatchTest;
    Check(RoceTest::blockIndex == 0 && jobIdx == 0U, "metadata packer/epoch owner is not physical AIV0");
    Check(targetServerIdx == HierarchicalHost(common.rankId), "packer used its AIV index as the host rail slot");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence))) == sequence,
          "packer started before the current sequence was published");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::LocalRailPacked))) != sequence,
          "rail readiness was published before packing finished");
    auto* rail = params.peermemInfo.railTopkIdsRecvPtr + targetServerIdx * config.railTopkIdsSlotBytes;
    const uint32_t bytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK));
    const int32_t marker = RailPacketMarker(params.tilingData, 0x81U, common.tokenNum * common.topK);
    for (uint32_t end = RAIL_PACKET_BYTES; end <= bytes; end += RAIL_PACKET_BYTES) {
        std::memcpy(rail + end - sizeof(marker), &marker, sizeof(marker));
    }
    ++packCalls;
}
inline void CalcRailRelayExpertRange(uint32_t experts, uint32_t index, uint32_t workers, uint32_t hosts,
                                     uint32_t source, uint32_t& first, uint32_t& count)
{
    DispatchTest::Check(RoceTest::blockIndex >= 2 && RoceTest::subBlockIndex == 0 &&
                            index == static_cast<uint32_t>(RoceTest::blockIndex / 2 - 1) && workers == 27U &&
                            hosts == 1U && source == 0U,
                        "Relay lost its original group partition");
    first = index * (experts / workers) + std::min(index, experts % workers);
    count = experts / workers + (index < experts % workers ? 1U : 0U);
}
inline std::vector<uint32_t> relaySources;
inline void RelayRailTopkIdsToExperts(const MoeStageCommonConfig& common, GM_ADDR*, const SendMaskConfig&,
                                      const Params&, SendMaskScratch&, uint32_t sourceHost, uint32_t, int32_t, int32_t,
                                      AscendTimerDevice*, GM_ADDR = nullptr, int32_t = -1, bool* = nullptr)
{
    using namespace DispatchTest;
    Check(RoceTest::blockIndex >= 2 && RoceTest::subBlockIndex == 0 && RoceTest::mteCanRead,
          "Relay must run on g1..last AIV0 after local acquire");
    if (sourceHost != HierarchicalHost(common.rankId)) {
        Check(ReadWord(local, metaReady) == sequence, "remote Relay skipped MetaReady");
        Check(!relaySources.empty() && relaySources.back() == HierarchicalHost(common.rankId),
              "remote Relay ran before useful local work");
    }
    Check(ReadWord(local, payloadCommit) != sequence, "Relay must not wait for payload");
    relaySources.push_back(sourceHost);
    ++relayCalls;
}
inline void GmSignalWaitBarrier(int32_t*, int32_t)
{
    throw std::runtime_error("Dispatch unexpectedly entered final host barrier");
}
template <bool>
void SyncAll()
{
    throw std::runtime_error("Dispatch unexpectedly synchronized all cores");
}
}  // namespace MegaMoeImpl

#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_hierarchical_dispatch.hpp"

using namespace MegaMoeImpl;

void CheckDispatch(uint32_t rank, uint32_t bs, uint32_t capacity, uint32_t hidden, uint32_t topk,
                   std::string_view pattern, uint32_t invocation = 257U)
{
    using namespace DispatchTest;
    MegaMoeTilingData t{};
    t.bs = bs;
    t.numMaxTokensPerRank = capacity;
    t.h = hidden;
    t.topK = topk;
    t.moeExpertPerRank = 12U;
    t.epWorldSize = 32U;
    t.dispatchRanksPerNode = 16U;
    t.blockAivNum = 56U;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.roceScratchUbOffset = kScratchOffset;
    t.hierarchicalSequence = invocation;
    Check(UseDecodeRailRouting(&t), "fixture requires the production decode rail layout");
    const PeermemSizeParams sizes{capacity,           topk, hidden, 12U, 32U, 2U, 1U, false, false,
                                  TOPO_TYPE_MTE_ROCE, true, 16U};
    const uint32_t rowBytes = DispatchPacketCount(CalcQuantTokenScaleBytes(hidden, 1U, topk, false)) * RAIL_PACKET_BYTES;
    const size_t oldHeap = CalcPeermemLeastSize(sizes);
    const size_t dataBase = oldHeap + 16U * 16U * 512U;
    const auto heapBytes = dataBase + 2U * capacity * rowBytes + 512U + (1U + 2U * bs) * 4U;
    // Append compact mapped ready regions for the other fifteen peers.
    // Only their ready slots are accessed by this fixture.
    local.assign(heapBytes + 15U * 16U * 512U, 0x7BU);
    remote.assign(heapBytes, 0xA5U);
    posts.clear();
    queued.clear();
    quiets = waits = metadataWaits = packCalls = relayCalls = progressCalls = 0U;
    RoceTest::cycleStep = 1;
    peer = HierarchicalGateway(rank);
    sequence = invocation;
    receiveLeaderJob = 2U;
    handoff = false;
    relaySources.clear();
    completeIncomingPayload = nullptr;
    completePacking = nullptr;
    ddrBarriers = 0U;
    broadcasts.fill(0U);
    RoceTest::operations.clear();
    RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    std::vector<float> weights(bs * topk);
    for (uint32_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int32_t>(i % 19) - 7) / 8.0F;
    Params params{&t, PeermemInfo(local.data(), &t), reinterpret_cast<uint8_t*>(weights.data())};
    params.peermemInfo.dispatchPacketPtr = local.data() + dataBase;
    params.peermemInfo.relayPayloadPtr = params.peermemInfo.dispatchPacketPtr + capacity * rowBytes;
    params.peermemInfo.compactPtr = params.peermemInfo.relayPayloadPtr + capacity * rowBytes;
    MoeStageCommonConfig common{rank, bs, topk, hidden, 12U};
    payloadReadyBase = Offset(params.peermemInfo.gatewayPayloadReadyPtr, 16U * 512U);
    ownerSlot = rank % 16U;
    std::fill_n(g_winRankAddr_, 32U, nullptr);
    uint32_t nextWindow = 1U;
    for (uint32_t slot = 0; slot < 16; ++slot) {
        const uint32_t window = slot == ownerSlot ? 0U : nextWindow++;
        g_winRankAddr_[HierarchicalHost(rank) * 16U + slot] = local.data() + window * 16U * 512U;
    }
    SendMaskConfig config{static_cast<uint64_t>(CalcRailTopkIdsSlotBytes(&t))};
    SendMaskScratch scratch;
    const auto flagOffset = [&](HierarchicalSignal signal) {
        return Offset(reinterpret_cast<uint8_t*>(HierarchicalSignalAddress(params.peermemInfo, signal)), 4U);
    };
    metaCommit = flagOffset(HierarchicalSignal::NetMetaCommit);
    payloadCommit = flagOffset(HierarchicalSignal::NetPayloadCommit);
    metaReady = flagOffset(HierarchicalSignal::MetaReady);
    relayReady = flagOffset(HierarchicalSignal::RelayPayloadReady);
    const auto writeWord = [&](uint32_t offset, uint32_t value) {
        std::memcpy(local.data() + offset, &value, sizeof(value));
    };
    for (uint32_t lane = 0; lane < 8; ++lane) writeWord(payloadCommit + lane * 512U, sequence - 1U);
    writeWord(flagOffset(HierarchicalSignal::Sequence), sequence);
    writeWord(metaCommit, sequence - 1U);

    tails.clear();
    const uint32_t activeBytes = bs * rowBytes;
    const uint32_t source = Offset(params.peermemInfo.dispatchPacketPtr, activeBytes);
    const uint32_t destination = Offset(params.peermemInfo.relayPayloadPtr, activeBytes);
    Check(params.peermemInfo.dispatchPacketPtr + activeBytes <= params.peermemInfo.relayPayloadPtr &&
              params.peermemInfo.relayPayloadPtr + activeBytes <= params.peermemInfo.compactPtr,
          "actual batch exceeds its production allocation");
    for (uint32_t byte = 0U; byte < activeBytes; ++byte) {
        local[source + byte] = static_cast<uint8_t>((byte * 13U + rank) % 251U);
    }
    for (uint32_t end = 512; end <= activeBytes; end += 512) writeWord(source + end - 4, sequence);
    std::array<uint32_t, 8> marker{};
    for (uint32_t lane = 0; lane < 8; ++lane) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        const uint32_t end = bs * (lane + 1) / 8;
        if (end == bs * lane / 8) continue;
        marker[lane] = destination + end * rowBytes - 4;
#else
        marker[lane] = payloadCommit + lane * 512U;
#endif
        tails.push_back(marker[lane]);
        writeWord(marker[lane], sequence - 1);
    }
    const auto payload = std::vector<uint8_t>(local.begin() + source, local.begin() + source + activeBytes);
    const uint32_t railBytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(bs, topk));
    const uint32_t rail =
        Offset(params.peermemInfo.railTopkIdsRecvPtr + HierarchicalHost(rank) * config.railTopkIdsSlotBytes, railBytes);
    const uint32_t incoming = Offset(
        params.peermemInfo.railTopkIdsRecvPtr + (1U - HierarchicalHost(rank)) * config.railTopkIdsSlotBytes, railBytes);
    metadataMarker = incoming + railBytes - 4U;
    const uint32_t prefix = HierarchicalMetadataPrefixBytes(&t);
    const uint32_t messageStart = rail - prefix;
    const uint32_t messageBytes = railBytes + prefix;
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t slot = 0U; slot < topk; ++slot) {
            const bool other = pattern == "remote" || (pattern == "mixed" && (token + slot) % 3U == 0U) ||
                               (pattern == "last" && token + 1U == bs && slot + 1U == topk);
            const uint32_t targetHost = HierarchicalHost(rank) ^ static_cast<uint32_t>(other);
            const uint32_t expert = (targetHost * 16U + (token + slot) % 16U) * 12U + slot % 12U;
            const uint32_t item = token * topk + slot;
            writeWord(rail + ((item / 127U) * 128U + item % 127U) * 4U, expert);
        }
    }
    auto expected = remote;
    const auto selectJob = [](uint32_t job) {
        RoceTest::blockIndex = static_cast<int>(job);
        RoceTest::subBlockIndex = static_cast<int>(job % 2U);
    };
    // Other senders can finish before metadata packing. Every empty lane must
    // still post exactly its own commit, with no access to another QP.
    for (uint32_t lane = 1U; lane < 8U; ++lane) {
        selectJob(2U * lane + 1U);
        GatherAndPostHierarchicalDispatch({2U * lane + 1U, 56U}, common, params, nullptr, config, scratch);
    }
    Check(packCalls == 0U, "payload lanes unexpectedly depend on metadata packing");
    selectJob(1U);
    completePacking = [&] {
        selectJob(0U);
        PackHierarchicalCombineWeights(common, params);
        SendTopkIdsToRailServer(common, nullptr, config, params, scratch, HierarchicalHost(rank), 0U);
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::LocalRailPacked);
        selectJob(1U);
    };
    PostHierarchicalDispatch(common, params, config);
    GatherAndPostHierarchicalDispatch({1U, 56U}, common, params, nullptr, config, scratch);
    Check(packCalls == 1U, "missing metadata pack");
    Check(std::memcmp(local.data() + messageStart, weights.data(), weights.size() * sizeof(float)) == 0,
          "packed weights differ from source");
    std::copy_n(local.data() + messageStart, messageBytes, expected.data() + messageStart);
    std::copy(payload.begin(), payload.end(), expected.begin() + destination);
    uint32_t covered = 0U;
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        const uint32_t begin = bs * lane / 8U, end = bs * (lane + 1U) / 8U;
        std::vector<Write> lanePosts;
        for (const auto& post : posts)
            if (post.qp == lane + 1U) lanePosts.push_back(post);
        const uint32_t metadataPosts = lane == 0U ?
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
                                                  1U
                                                  : 0U;
#else
                                                  2U
                                                  : 0U;
#endif
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        constexpr unsigned commits = 0;
#else
        constexpr unsigned commits = 1;
#endif
        Check(lanePosts.size() == metadataPosts + (end > begin ? 1U : 0U) + commits, "wrong per-lane POST count");
        if (lane == 0U) {
            Check(lanePosts.front().source == messageStart && lanePosts.front().destination == messageStart &&
                      lanePosts.front().bytes == messageBytes,
                  "metadata must precede lane0 payload");
        }
        if (end > begin) {
            const auto& data = lanePosts[metadataPosts];
            Check(data.source == source + begin * rowBytes && data.destination == destination + begin * rowBytes &&
                      data.bytes == (end - begin) * rowBytes,
                  "payload range is missing, overlapping or compacted");
            covered += data.bytes;
        }
        if (commits) {
        Check(lanePosts.back().destination == payloadCommit + lane * 512U && lanePosts.back().bytes == 4U,
              "missing or misplaced terminal lane commit");
        std::memcpy(expected.data() + payloadCommit + lane * 512U, &sequence, 4U);
        }
    }
    Check(covered == activeBytes, "lanes did not cover actual batch exactly once");
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    std::memcpy(expected.data() + metaCommit, &sequence, 4U);
#endif
    // Wrong actual-tail epoch must block only the remote pass: local Relay
    // already completed. A current capacity-tail marker cannot release it.
    writeWord(metadataMarker, sequence - 128U);
    if (config.railTopkIdsSlotBytes > messageBytes)
        writeWord(incoming - prefix + static_cast<uint32_t>(config.railTopkIdsSlotBytes) - 4U, sequence);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    writeWord(metaCommit, sequence);
#endif
    selectJob(2U);
    RoceTest::cycleStep = static_cast<int64_t>(MEGAMOE_ROCE_WAIT_TIMEOUT_CYCLES);
    bool staleRejected = false;
    try {
        GatherAndPostHierarchicalDispatch({2U, 56U}, common, params, nullptr, config, scratch);
    } catch (const std::runtime_error&) {
        staleRejected = true;
    }
    RoceTest::cycleStep = 1;
    Check(staleRejected && relaySources == std::vector<uint32_t>{HierarchicalHost(rank)} &&
              ReadWord(local, metaReady) != sequence,
          "stale metadata released remote Relay or blocked local work");
    std::copy_n(local.data() + messageStart, messageBytes, local.data() + incoming - prefix);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    writeWord(metaCommit, sequence);
#endif
    relaySources.clear();
    for (uint32_t group = 1U; group < 28U; ++group) {
        selectJob(2U * group);
        GatherAndPostHierarchicalDispatch({2U * group, 56U}, common, params, nullptr, config, scratch);
    }
    Check(relaySources.size() == 54U && progressCalls == 0U, "Relay missed a source or borrowed sender CQ");
    for (uint32_t group = 8U; group < 28U; ++group) {
        selectJob(2U * group + 1U);
        GatherAndPostHierarchicalDispatch({2U * group + 1U, 56U}, common, params, nullptr, config, scratch);
    }
    Check(relaySources.size() == 54U, "AIV1 unexpectedly relayed");
    for (uint32_t job = 0U; job < 56U; ++job) {
        if (job == 53U || job == 55U) continue;
        selectJob(job);
        FinishHierarchicalDispatch({job, 56U}, common, params);
    }
    // Exercise arrival permutation; a stale final lane must never allow an
    // early broadcast. Arrival/receive completion do not retire outgoing CQEs.
    const std::array<uint32_t, 8> arrival{7, 3, 0, 6, 2, 5, 1, 4};
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    selectJob(53U);
    AscendC::laneSnapshots.clear();
    uint32_t remembered = 0U;
    for (const auto lane : arrival) {
        auto before = AscendC::laneSnapshots.size();
        const uint32_t previous = remembered;
        FinishHierarchicalDispatch({53U, 56U}, common, params, nullptr, &remembered);
        for (uint32_t i = 0; i < 8U; ++i) {
            if (marker[i] && !(previous & (1U << i))) {
                Check(AscendC::laneSnapshots.at(before++) ==
                          reinterpret_cast<const uint32_t*>(local.data() + marker[i] - 28U),
                      "helper did not sample the actual lane trailer");
            }
        }
        Check(before == AscendC::laneSnapshots.size(), "helper reread a cached or empty lane");
        Check((remembered & (1U << lane)) == 0U || !marker[lane], "stale lane remembered as ready");
        Check(std::all_of(broadcasts.begin(), broadcasts.end(), [](uint32_t n) { return n == 0; }),
              "bounded probe broadcast early");
        if (marker[lane]) writeWord(marker[lane], sequence);
        // Empty lanes can appear last in this permutation.
        if (std::all_of(marker.begin(), marker.end(), [&](auto m) { return !m || ReadWord(local, m) == sequence; }))
            break;
    }
    FinishHierarchicalDispatch({53U, 56U}, common, params, nullptr, &remembered);
    FinishHierarchicalDispatch({53U, 56U}, common, params, nullptr, &remembered);
    Check(remembered == 255U, "helper did not remember all lanes");
#else
    uint32_t arrivals = 0U;
    completeIncomingPayload = [&] {
        Check(std::all_of(broadcasts.begin(), broadcasts.end(), [](uint32_t n) { return n == 0; }),
              "broadcast preceded all eight arrivals");
        Check(arrivals < 8U, "receiver polled an already completed lane");
        while (!marker[arrival[arrivals]]) ++arrivals;
        writeWord(marker[arrival[arrivals++]], sequence);
    };
    selectJob(53U);
    FinishHierarchicalDispatch({53U, 56U}, common, params);
    completeIncomingPayload = nullptr;
#endif
    selectJob(55U);
    FinishHierarchicalDispatch({55U, 56U}, common, params);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    Check(waits == 2 * tails.size(), "missing lane commit acquire");
#endif
    Check(ddrBarriers == 4U, "missing collective payload acquire");
    for (uint32_t window = 0; window < 16; ++window) {
        Check(broadcasts[window] == 1U, "owner omitted a ready notification");
        for (uint32_t slot = 0; slot < 16; ++slot) {
            const uint32_t expectedReady = slot == ownerSlot ? sequence : 0x7B7B7B7BU;
            Check(ReadWord(local, payloadReadyBase + (window * 16U + slot) * 512U) == expectedReady,
                  "broadcast damaged another gateway's line");
        }
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // Deliver different QPs in reverse order; each terminal commit remains ordered
    // after its own data. Transfer ownership only at the modeled input barrier.
    for (uint32_t qp = 8U; qp > 0U; --qp) DeliverPosted(qp);
    Check(!queued.empty() && quiets == 0U, "incoming commits incorrectly completed local QPs");
#endif
    handoff = true;
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        selectJob(2U * lane);
        RoceFinishOutstanding(peer, HierarchicalRoceScratch(params), nullptr, AscendTimer::RdmaPhase::Combine,
                              lane + 1U);
    }
    Check(remote == expected && queued.empty(), "payload or tail completion corrupted neighboring regions");
    Check(payload == std::vector<uint8_t>(local.begin() + source, local.begin() + source + activeBytes),
          "source changed before final completion");
    std::fill_n(local.data() + source, activeBytes, uint8_t{0xDDU});
    Check(remote == expected, "completed transfers still referenced reusable sources");
    Check(std::none_of(AscendC::packEvents.begin(), AscendC::packEvents.end(), [](bool v) { return v; }),
          "packer left an event outstanding");
}

#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
void CheckGatewayDma(uint32_t rank, uint32_t epoch, uint32_t lateLane)
{
    using namespace DispatchTest;
    MegaMoeTilingData t{};
    t.bs = 32;
    t.topK = 8;
    t.h = 3584;
    t.moeExpertPerRank = 12;
    t.blockAivNum = 56;
    t.epWorldSize = 32;
    t.dispatchRanksPerNode = 16;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.hierarchicalSequence = sequence = epoch;
    t.roceScratchUbOffset = kScratchOffset;
    Check(UseRouteAllGather(&t), "target gate disabled");
    const uint32_t bytes = DispatchPacketCount(CalcQuantTokenScaleBytes(t.h, 1U, t.topK, false)) * RAIL_PACKET_BYTES;
    constexpr uint32_t window = 16U * HIERARCHICAL_FLAG_BYTES;
    const uint32_t payload = 16U * window;
    local.assign(payload + bytes * t.bs, 0x7B);
    Params params{&t, {}, nullptr};
    const uint32_t slot = rank % 16;
    for (uint32_t i = 0; i < 16; ++i) g_winRankAddr_[rank - slot + i] = local.data() + i * window;
    params.peermemInfo.rankSyncInWorldPtr = g_winRankAddr_[rank];
    params.peermemInfo.gatewayPayloadReadyPtr = g_winRankAddr_[rank];
    params.peermemInfo.relayPayloadPtr = local.data() + payload;
    MoeStageCommonConfig common{rank, t.bs, t.topK, t.h, t.moeExpertPerRank};
    tails.clear();
    for (uint32_t lane = 0; lane < 8; ++lane) {
        tails.push_back(payload + HierarchicalPayloadTokenBegin(t.bs, lane + 1U) * bytes - 4U);
        auto value = lane == lateLane ? sequence - 256 : sequence;
        std::memcpy(local.data() + tails.back(), &value, 4);
    }
    uint32_t masks[2]{};
    noticeGroups = 0;
    ddrBarriers = 0;
    for (uint32_t round = 0; round < 3; ++round) {
        if (round == 1) std::memcpy(local.data() + tails[lateLane], &sequence, 4);
        for (uint32_t half = 0; half < 2; ++half) {
            uint32_t job = 53 + 2 * half;
            RoceTest::blockIndex = job;
            helperMask = &masks[half];
            FinishHierarchicalDispatch({job, 56U}, common, params, nullptr, &masks[half]);
            Check(!notificationOpen && masks[half] == (round ? 255U : (255U ^ (1U << lateLane))), "partial or unretired gateway");
            for (uint32_t dest = 0; dest < 16; ++dest) {
                bool notified = round && (round == 2 || dest / 8 <= half);
                const uint32_t address = dest * window + slot * HIERARCHICAL_FLAG_BYTES;
                for (uint32_t i = 0; i < 32; i += 4)
                    Check(ReadWord(local, address + i) == (notified ? sequence : 0x7b7b7b7bU),
                          "wrong destination or partial write");
                Check(ReadWord(local, address + 32) == 0x7b7b7b7bU, "notification overran its line");
            }
        }
    }
    Check(noticeGroups == 2U && ddrBarriers == 0U, "expected one retired DMA batch per helper without a DDR barrier");
    helperMask = nullptr;
}
#endif

int main()
{
    try {
        for (uint32_t rank : {5U, 21U}) {
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
            CheckGatewayDma(rank, 0x80000101U, 7U);
            CheckGatewayDma(rank, 0x80000201U, 0U);
#endif
            for (const auto shape : std::array<std::array<uint32_t, 2>, 4>{{{1, 32}, {7, 64}, {31, 32}, {32, 64}}}) {
                for (const auto pattern : {"local", "remote", "mixed", "last"}) {
                    CheckDispatch(rank, shape[0], shape[1], 3456U, 8U, pattern);
                }
            }
            for (uint32_t bs : {72U, 73U, 96U, 127U, 128U}) CheckDispatch(rank, bs, 128U, 3456U, 8U, "mixed");
            CheckDispatch(rank, 256U, 256U, 3584U, 8U, "mixed");
            CheckDispatch(rank, 256U, 256U, 1024U, 32U, "last");
            CheckDispatch(rank, 31U, 64U, 1024U, 1U, "last");
            CheckDispatch(rank, 32U, 32U, 8192U, 32U, "remote");
            CheckDispatch(rank, 1U, 64U, 3584U, 32U, "local");
            CheckDispatch(rank, 32U, 64U, 3456U, 8U, "mixed", 0x80000101U);
            CheckDispatch(rank, 7U, 64U, 3584U, 8U, "last", 0xffffffffU);
            {
                CheckDispatch(rank, 32U, 64U, 3456U, 8U, "mixed", 257U);
                CheckDispatch(rank, 7U, 32U, 1024U, 1U, "remote", 0xffffffffU);
                CheckDispatch(rank, 1U, 64U, 3584U, 32U, "local", 257U);
                CheckDispatch(rank, 256U, 256U, 1024U, 32U, "last", 257U);
            }
        }
        RoceTest::blockIndex = RoceTest::subBlockIndex = 0;
        // Negative control: the asynchronous double must detect early source
        // reuse instead of accidentally behaving as a synchronous memcpy.
        auto* ub = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(DispatchTest::kScratchOffset));
        MegaMoeImpl::RocePutBytes(DispatchTest::local.data() + 64U, DispatchTest::local.data() + 128U, 16U,
                                  DispatchTest::peer, ub);
        DispatchTest::local[128U] ^= 1U;
        bool rejected = false;
        try {
            MegaMoeImpl::RoceQuiet(DispatchTest::peer, ub);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        DispatchTest::Check(rejected, "asynchronous test double failed to detect an invalid source lifetime");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
