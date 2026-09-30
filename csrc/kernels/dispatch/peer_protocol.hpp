// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

#include <cstdint>

namespace ascend_deepep {
// Notify-derived peer table (uint32 words, NOT mailbox uint64 words below):
// peer0 words 4/5/6/7 are mode/direct-offset/mailbox-offset/ABI(7).
// Mode3 (direct-only): direct-offset=0; suffix is owner-only profile storage.
// There is no helper publication/ACK or ready fanout in this mode.
// Options live in peer1's reserved word4 (absolute word12); never overwrite
// peer0 word7, which EVERY AIV validates before the entry barrier.
constexpr uint32_t kClosPeerTableOptionsWord = 12U;
constexpr uint32_t kClosPeerTableOptionsLoadWords = 16U;  // Two aligned 32B headers.
constexpr uint32_t kClosPeerTablePairedVf = 1U;
constexpr uint32_t kClosPeerTablePodInterleave = 2U;
// Independent A/B controls: a scale transport change must NOT silently reorder hidden.
constexpr uint32_t kClosPeerTableBalancedLayout = 4U;
constexpr uint32_t kClosPeerTableGroupOffset64 = 64U;
constexpr uint32_t kClosPeerTableSplitQp = 128U;
// One 32B descriptor: absolute hidden/weight/scale GM virtual addresses, then ready generation.
// Uses the private C++ path's target-platform 32B visibility assumption.
constexpr uint32_t kClosPeerTableAddressReady = 256U;
constexpr uint32_t kPeerAddressRowBytes = 512U;
constexpr uint32_t kPeerAddressRowWords = kPeerAddressRowBytes / sizeof(uint64_t);
constexpr uint32_t kPeerAddressRecordWords = 4U;
constexpr uint32_t kPeerAddressGenerationWord = 3U;
// Immutable local VF source, separate from every inbound descriptor/cache line.
constexpr uint32_t kPeerAddressOutgoingBytes = 128U;
static_assert(kPeerAddressRecordWords * sizeof(uint64_t) == 32U);
static_assert(kPeerAddressOutgoingBytes >= 64U && kPeerAddressOutgoingBytes + 32U <= kPeerAddressRowBytes);
constexpr uint32_t kClosPeerTablePodStaged = 512U;
constexpr uint32_t kClosPeerTableQp1EightWorkers = 1024U;
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint32_t
    DefaultClosPeerTableOptions(uint32_t world)
{
    // EP32/64: balanced+window. EP128/256: balanced+pod-staged.
    return kClosPeerTableSplitQp | kClosPeerTableAddressReady | kClosPeerTableQp1EightWorkers |
           kClosPeerTableBalancedLayout | (world >= 128U ? kClosPeerTablePodStaged : 0U);
}
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint32_t
    DefaultClosPeerTableOwner(uint32_t peer)
{
    return peer % 8U;
}
constexpr uint32_t kClosScaleReadySchedule = 8U;
constexpr uint32_t kClosScaleBalancedQp = 16U;
constexpr uint32_t kClosScaleDedicatedJetty = 32U;
static_assert(kClosPeerTableOptionsWord % 8U >= 4U && kClosPeerTableOptionsWord / 8U != 0U);
static_assert(kClosPeerTableOptionsWord < kClosPeerTableOptionsLoadWords);
// Handle-local mailbox appended to the two Notify tables. This is NOT
// SHMEM transport/control storage. Each cache line has exactly one writer.
constexpr uint32_t kClosRelayAssistOwners = 8U;
constexpr uint32_t kClosRelayAssistLineWords = 64U;  // 512B, conservative isolation
constexpr uint32_t kClosRelayAssistQueues = 64U;     // EP256, two QPs / actual EID
constexpr uint32_t kClosRelayAssistPeers = 32U;
constexpr uint32_t kClosRelayAssistQueueWord = 2U * kClosRelayAssistLineWords;
constexpr uint32_t kClosRelayAssistPeerWord = kClosRelayAssistQueueWord + kClosRelayAssistQueues * 4U;
constexpr uint32_t kClosRelayAssistOwnerWords = kClosRelayAssistPeerWord + kClosRelayAssistPeers * 4U;
constexpr uint32_t kClosRelayAssistBytes = kClosRelayAssistOwners * kClosRelayAssistOwnerWords * 8U;
// Opt-in split-QP: one cache-line-isolated publisher per owner/QP. QP0 owns
// tasks 0..7, QP1 owns tasks 8..15. No shared-writer task or queue state.
constexpr uint32_t kClosSplitLinkWords = 16U * kClosRelayAssistPeers * 16U;
constexpr uint32_t kClosSplitMailboxBytes = 2U * kClosRelayAssistBytes + kClosSplitLinkWords * 8U;
static_assert(kClosRelayAssistOwnerWords * 8U % 512U == 0U);
// Header word 0: generation, published LAST by Clos. Words 1/2: queue/peer count.
// After helper ACK, owner-only EXTRA profiles use words 3..9 for phase clocks;
// 10/11=last final-SO submit QP0/1, 12/13=all final-SO CQEs first observed QP0/1,
// 14/15=observation scan numbers, 16=QP profile version 1. No layout growth.
// split-qp uses version 2, word17=physical AIV, word18=logical QP; only that
// QP's clocks are populated. Link profiles live in the separate suffix above.
// BEFORE helper ACK, word3=EP64 group-offset membership, in generation's
// cache line. AFTER ACK the owner may reuse word3 for its phase clock.
// Queue record: {CQE address, old owner bit, physical QP, SO1 ticket}.
// Peer record: {destination rank, queue index 0, queue index 1, reserved}.
// Second line: generation ack (written LAST by AIV7), followed by profile clocks.
}  // namespace ascend_deepep
