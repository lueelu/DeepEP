// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"

namespace MegaMoeUrma {
inline constexpr aclshmemx_udma_op_config_t kUrmaRO{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};

// Defaults/capacity only. Kernel calls pass the active Config::world.
constexpr uint32_t kWorldSize = 16U;
constexpr uint32_t kSignalSlotBytes = 128U;
constexpr uint32_t kSignalCopyBytes = 32U;
constexpr uint32_t kMaxPutBytes = 256U * 1024U * 1024U;
constexpr uint32_t kCommunicationQp = 0U;

// One AIV1 owns QP0 for each remote peer. Dispatch submissions precede Combine.
// The SO atomic API uses QP0; all payloads must use that same ordered queue.
__aicore__ inline uint32_t PeerForOwner(uint32_t rank, uint32_t owner)
{
    return owner < rank ? owner : owner + 1U;
}

// dstOffset names a LOCAL symmetric-window address; SHMEM translates it for
// peer. Do not pass an already translated MTE peer pointer to this function.
// The caller completes local GM producers before submitting and retains src
// unchanged until Quiet(peer, qp) returns. This transport deliberately has no GET.
__aicore__ inline void Put(__gm__ uint8_t* symmetricBase, uint64_t dstOffset, __gm__ uint8_t* src, uint32_t bytes,
                           uint32_t peer)
{
    if (bytes == 0U) return;
    if (bytes > kMaxPutBytes) {
        aclshmemi_kernel_abort("MegaMoE URMA PUT exceeds 256 MiB: bytes=%u peer=%u\n", bytes, peer);
        return;
    }
    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S, kUrmaRO>(symmetricBase + dstOffset, src, nullptr, bytes, peer,
                                                        kCommunicationQp, 0U);
}

// The receiver resets this invocation's slots before publishing its start.
// SO is ordered after every preceding RO/SO on this PE/QP. Ordinary NO PUTs
// would not provide this guarantee. This is async and has no per-CV quiet.
__aicore__ inline void PublishOrdered(__gm__ uint8_t* symmetricBase, uint64_t signalOffset, uint32_t peer,
                                      uint64_t epoch)
{
    aclshmemx_udma_atomic_add<uint64_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
        reinterpret_cast<__gm__ uint64_t*>(symmetricBase + signalOffset), epoch, peer);
}

// Blocking completion of all local submissions, including the final SO flag.
// The one-owner rule also applies here: no second worker may poll the same CQ.
__aicore__ inline void Quiet(uint32_t peer)
{
    aclshmemx_udma_qp_quiet(peer, kCommunicationQp);
}

// MTE publication is retained only for the invocation start/barrier handshake.
// Each (source peer, CV chunk, phase) owns a separate 128B destination slot.
// The caller supplies 32B of UB and a free event ID; no hidden TPipe/UB state
// is introduced. MTE3_S protects both publication and immediate UB reuse.
__aicore__ inline void PublishDone(__gm__ uint8_t* symmetricBase, uint64_t signalOffset, uint32_t peer,
                                   uint64_t sequence, AscendC::LocalTensor<uint64_t> signalUb,
                                   AscendC::TEventID eventId)
{
    signalUb.SetValue(0, sequence);
    signalUb.SetValue(1, 0U);
    signalUb.SetValue(2, 0U);
    signalUb.SetValue(3, 0U);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    auto* remoteSlot = reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(symmetricBase + signalOffset, peer));
    AscendC::GlobalTensor<uint64_t> done;
    done.SetGlobalBuffer(remoteSlot);
    AscendC::DataCopy(done, signalUb, kSignalCopyBytes / sizeof(uint64_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Local polling only: this is not a remote read/GET and may be used by AIC or
// AIV. ReadGmByPassDCache avoids a stale locally cached signal. Payload consumers
// must likewise use the stage's coherent/cache-bypassing input access policy.
__aicore__ inline void WaitLocalDone(__gm__ uint8_t* localSlot, uint64_t sequence)
{
    auto* value = reinterpret_cast<__gm__ uint64_t*>(localSlot);
    while (AscendC::ReadGmByPassDCache(value) != sequence) {
    }
}

// Empty peer edges may advance to the next invocation before the slow owner
// observes the old start. Accept a later start only here; CV data readiness
// still requires the exact epoch. Any nonempty edge prevents such advancement
// until its old payload has been consumed and locally reduced.
__aicore__ inline void WaitStartAtLeast(__gm__ uint8_t* localSlot, uint64_t sequence)
{
    auto* value = reinterpret_cast<__gm__ uint64_t*>(localSlot);
    while (AscendC::ReadGmByPassDCache(value) < sequence) {
    }
}

// Only one local leader enters this function AFTER joining all local workers
// and their per-peer Quiet calls. All remote reads are of this rank's own flag
// slots. Each source rank owns one cache line in every destination rank.
// sequence must increase across phases and invocations; >= tolerates a fast
// peer reaching the next barrier before a slow peer observes this barrier.
// Protect returnY reuse after reduce with a final barrier or a next-invocation
// peer handshake whose publication follows completion of every local reducer.
__aicore__ inline void RankBarrier(__gm__ uint8_t* symmetricBase, uint64_t barrierOffset, uint32_t rank,
                                   uint64_t sequence, AscendC::LocalTensor<uint64_t> signalUb,
                                   AscendC::TEventID eventId, uint32_t world = kWorldSize)
{
    for (uint32_t peer = 0; peer < world; ++peer) {
        PublishDone(symmetricBase, barrierOffset + static_cast<uint64_t>(rank) * kSignalSlotBytes, peer, sequence,
                    signalUb, eventId);
    }
    for (uint32_t peer = 0; peer < world; ++peer) {
        auto* localSlot = reinterpret_cast<__gm__ uint64_t*>(symmetricBase + barrierOffset +
                                                             static_cast<uint64_t>(peer) * kSignalSlotBytes);
        while (AscendC::ReadGmByPassDCache(localSlot) < sequence) {
        }
    }
}

}  // namespace MegaMoeUrma
