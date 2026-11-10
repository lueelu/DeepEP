// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "parallel_route_plan.hpp"
#include "route_runtime.hpp"
#include "route_expert_vf.hpp"
#include "transport.hpp"
#include "timer.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kRouteInfoUb = 96U * 1024U;
struct ExpertRouteUb {
    static constexpr uint32_t ids = 0U, raw = 16U * 1024U, counts = 48U * 1024U;
    static constexpr uint32_t mask = 50U * 1024U, totals = 52U * 1024U, routes = 64U * 1024U;
    static constexpr uint32_t end = routes + kExpertRouteStorageItems * 4U;
};
static_assert(ExpertRouteUb::end <= kRequiredUbBytes);
static_assert(kRouteInfoUb + sizeof(DeviceRouteInfo) <= kRequiredUbBytes);

template <HardEvent E>
__aicore__ inline void RouteFence()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
__aicore__ inline void RouteLoad(uint64_t gm, uint32_t ub, uint32_t bytes)
{
    GlobalTensor<uint8_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> dst(TPosition::VECCALC, ub, bytes);
    DataCopyPad(dst, src, {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    RouteFence<HardEvent::MTE2_S>();
}
__aicore__ inline void RouteSave(uint64_t gm, uint32_t ub, uint32_t bytes)
{
    if (!bytes) return;
    GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> src(TPosition::VECCALC, ub, bytes);
    RouteFence<HardEvent::S_MTE3>();
    DataCopyPad(dst, src, {1U, bytes, 0U, 0U, 0U});
    RouteFence<HardEvent::MTE3_S>();
}
// Startup-only scratch at UB offset 0 is retired before RouteOwnedExperts
// reuses it. Assign whole 128B slots so AIV workers never share a cache line.
__aicore__ inline void ResetFlagRegion(uint64_t address, uint64_t bytes, uint32_t worker)
{
    const uint64_t slots = bytes / kFlagBytes;
    uint64_t offset = (slots * worker / kLocalWorkers) * kFlagBytes;
    const uint64_t end = (slots * (worker + 1U) / kLocalWorkers) * kFlagBytes;
    while (offset < end) {
        const uint32_t size = static_cast<uint32_t>(end - offset < 4096U ? end - offset : 4096U);
        RouteSave(address + offset, 0U, size);
        offset += size;
    }
}

__aicore__ inline void ResetInvocationFlags(const Arguments& a, uint32_t worker)
{
    LocalTensor<int32_t> zeros(TPosition::VECCALC, 0U, 1024U);
    Duplicate<int32_t>(zeros, 0, 1024U);
    RouteFence<HardEvent::V_MTE3>();
    const auto& c = a.config;
    const uint64_t chunks = uint64_t(c.world) * c.maxPeerChunks;
    const uint64_t groups = RouteGroupCapacity(uint64_t(c.world) * c.sourceRowCapacity, c.localExperts);
    // Match the Host allocation's minimum size, including its unused padding.
    const uint64_t groupBytes = groups * kFlagBytes < 512U ? 512U : groups * kFlagBytes;
    ResetFlagRegion(a.symmetric + c.dispatchReadyOffset, chunks * kFlagBytes, worker);
    ResetFlagRegion(a.symmetric + c.combineReadyOffset, chunks * kFlagBytes, worker);
    ResetFlagRegion(a.symmetric + c.barrierOffset + uint64_t(c.world) * kFlagBytes, uint64_t(c.world) * kFlagBytes,
                    worker);
    ResetFlagRegion(a.inputScaleDone, groupBytes, worker);
    ResetFlagRegion(a.activationDone, groupBytes, worker);
    ResetFlagRegion(a.gmm2Done, groupBytes, worker);
    ResetFlagRegion(a.gmmIssued, kAic * kFlagBytes, worker);
    ResetFlagRegion(a.localFlags, (a.sharedExperts ? kSharedLocalFlagCount : 16U) * kFlagBytes, worker);
    ResetFlagRegion(a.packReady, chunks * kFlagBytes, worker);
    if (a.sharedExperts) {
        const uint64_t maxTokens = c.sourceRowCapacity / c.topk;
        const uint64_t sharedGroups = (maxTokens + 255U) / 256U;
        const uint64_t tileBytes = sharedGroups * (a.sharedIntermediate / 256U) * kFlagBytes;
        const uint64_t activationBytes = sharedGroups * kFlagBytes;
        ResetFlagRegion(a.sharedGmm1Done, tileBytes < 512U ? 512U : tileBytes, worker);
        ResetFlagRegion(a.sharedActivationDone, activationBytes < 512U ? 512U : activationBytes, worker);
    }
    // RouteSave drains each MTE3 write. The existing first route SyncAll joins
    // every reset worker before any Start, route-ready or compute publication.
    // Persistent Start and route epoch slots deliberately remain untouched.
}

// Caller provides one S_MTE3 edge before the batch and one MTE3_S edge after
// it. Every UB source remains immutable until that final completion edge.
__aicore__ inline void RouteEnqueueSave(uint64_t gm, uint32_t ub, uint32_t bytes)
{
    if (bytes == 0U) return;
    GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> src(TPosition::VECCALC, ub, bytes);
    DataCopyPad(dst, src, {1U, bytes, 0U, 0U, 0U});
}
// Descriptor writes are contiguous metadata, never scalar token payload I/O.
// Each array has one writer; flush complete cache lines before the local join.
template <class T>
__aicore__ inline void RouteStore(uint64_t address, uint32_t index, const T& value)
{
    auto* dst = reinterpret_cast<__gm__ uint32_t*>(address + uint64_t(index) * sizeof(T));
    const auto* src = reinterpret_cast<const uint32_t*>(&value);
    for (uint32_t i = 0; i < sizeof(T) / 4U; ++i) dst[i] = src[i];
}
__aicore__ inline void RouteFlush(uint64_t address, uint64_t bytes)
{
    // DCache lines are 64B on Ascend 950; a 128B signal/mailbox slot is NOT
    // the cache-line size. Missing alternate lines leaves dirty count stores
    // alive across the DMA rewrite to prefixes and corrupts route addresses.
    if (bytes == 0U) return;
    constexpr uint64_t lineBytes = 64U;
    for (uint64_t p = address & ~(lineBytes - 1U); p < address + bytes; p += lineBytes) {
        GlobalTensor<uint32_t> line;
        line.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(p));
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(line);
    }
}
// Compile has already copied the inbox into RouteInfo before constructing
// this writer. The low 40KiB is dead scratch until emit; RouteInfo at 96KiB
// and the live CSR at 128KiB+ are untouched.
struct RouteDescriptorUb {
    static constexpr uint32_t stride = 4096U;
    // 3840 is divisible by 32 and every ordinary descriptor size (4/8/24/40).
    // Flushes never split a typed record, and every full DMA is 32B aligned.
    static constexpr uint32_t batchBytes = 3840U;
    // PackChunk x16 + PackRows = one 648B PackRound. Four complete rounds are
    // 2592B, divisible by 32; the mixed 40B/8B append stream cannot overflow.
    static constexpr uint32_t packBatchBytes = 4U * sizeof(PackRound);
    static constexpr uint32_t end = 10U * stride;
};
static_assert(RouteDescriptorUb::batchBytes % 32U == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(Fragment) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(RowSegment) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(Chunk) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(ComputeBatch) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(ComputeBatchTask) == 0U);
static_assert(RouteDescriptorUb::packBatchBytes % 32U == 0U);
static_assert(RouteDescriptorUb::packBatchBytes <= RouteDescriptorUb::stride);
static_assert(RouteDescriptorUb::end <= kRouteInfoUb);
#if MEGAMOE_URMA_ROUTE_REORDER
static_assert(RouteDescriptorUb::batchBytes % sizeof(PeerReorderSpan) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(PeerReorderRange) == 0U);
#endif

template <uint32_t UbOffset, uint32_t CapacityBytes = RouteDescriptorUb::batchBytes>
class RouteDescriptorDmaStream {
public:
    __aicore__ explicit RouteDescriptorDmaStream(uint64_t address) : address_(address) {}

    // CompileDeviceRoutes writes each table in increasing, contiguous byte
    // order. Keep the supplied offset for a buffer's first record; no hidden
    // producer-side read, atomic append, lookup or descriptor reorder.
    template <class T>
    __aicore__ inline void Append(uint64_t byteOffset, const T& value)
    {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0U);
        static_assert(UbOffset % 32U == 0U && CapacityBytes % 32U == 0U);
        if (usedBytes_ == 0U) firstByte_ = byteOffset;
        LocalTensor<uint32_t> buffer(TPosition::VECCALC, UbOffset, CapacityBytes / 4U);
        const auto* words = reinterpret_cast<const uint32_t*>(&value);
        for (uint32_t word = 0U; word < sizeof(T) / 4U; ++word) buffer.SetValue(usedBytes_ / 4U + word, words[word]);
        usedBytes_ += sizeof(T);
        if (usedBytes_ == CapacityBytes) Flush();
    }

    __aicore__ inline void Flush()
    {
        if (usedBytes_ == 0U) return;
        // RouteSave does S_MTE3 -> DataCopyPad -> MTE3_S. The buffer is only
        // reused after that completion edge; at most one DMA is outstanding.
        RouteSave(address_ + firstByte_, UbOffset, usedBytes_);
        usedBytes_ = 0U;
    }

private:
    uint64_t address_;
    uint64_t firstByte_ = 0U;
    uint32_t usedBytes_ = 0U;
};

// Prefix data becomes immutable after the final local join. Invalidating at
// entry avoids retaining last launch's summary in consumers' scalar caches.
__aicore__ inline void InvalidateRouteSummary(const Arguments& a)
{
    RouteFlush(a.routeInfo, sizeof(DeviceRouteInfo));
    RouteFlush(a.routeConfig, sizeof(Config));
}

__aicore__ inline void RouteOwnedExperts(const Arguments& a, uint32_t worker)
{
    const uint32_t experts = a.config.world * a.config.localExperts;
    const uint32_t each = experts / kLocalWorkers, extra = experts % kLocalWorkers;
    const uint32_t first = worker * each + (worker < extra ? worker : extra);
    const uint32_t owned = each + (worker < extra ? 1U : 0U);
    if (owned == 0U) return;  // empty owners still participate in the enclosing joins
    const uint32_t total = a.config.bs * a.config.topk;
    const uint32_t batchItems =
        owned <= 7U ? kExpertRouteBatch
                    : (owned <= 10U ? kExpertWideBatch : (kExpertRouteStorageItems / owned / 64U) * 64U);
    LocalTensor<int32_t> ids(TPosition::VECCALC, ExpertRouteUb::ids, kExpertRouteBatch);
    LocalTensor<int32_t> routes(TPosition::VECCALC, ExpertRouteUb::routes, kExpertRouteStorageItems);
    LocalTensor<uint32_t> counts(TPosition::VECCALC, ExpertRouteUb::counts, 64U);
    LocalTensor<uint32_t> totals(TPosition::VECCALC, ExpertRouteUb::totals, kExpertRoutesPerWorker);
    for (uint32_t e = 0; e < owned; ++e) totals.SetValue(e, 0U);
    for (uint32_t start = 0; start < total; start += batchItems) {
        const uint32_t valid = RouteMin(batchItems, total - start);
        if (a.topkIdBytes == 4U)
            RouteLoad(a.topkIds + uint64_t(start) * 4U, ExpertRouteUb::ids, valid * 4U);
        else {
            RouteLoad(a.topkIds + uint64_t(start) * 8U, ExpertRouteUb::raw, valid * 8U);
            LocalTensor<int32_t> raw(TPosition::VECCALC, ExpertRouteUb::raw, 2U * kExpertRouteBatch);
            LocalTensor<uint32_t> mask(TPosition::VECCALC, ExpertRouteUb::mask, kExpertRouteBatch / 16U);
            for (uint32_t i = 0; i < (2U * valid + 31U) / 32U; ++i) mask.SetValue(i, 0x55555555U);
            RouteFence<HardEvent::S_V>();
            uint64_t selected = 0U;
            GatherMask(ids, raw, mask, true, 2U * valid, {1, 1, 0, 0}, selected);
            RouteFence<HardEvent::V_S>();
        }
        RouteFence<HardEvent::S_V>();
        asc_vf_call<CompactOwnedExpertsVF>(reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int32_t*>(routes.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()), valid, start,
                                           first, owned, batchItems);
        RouteFence<HardEvent::V_S>();
        RouteFence<HardEvent::S_MTE3>();
        for (uint32_t e = 0; e < owned; ++e) {
            const uint32_t n = counts.GetValue(e), written = totals.GetValue(e);
            RouteEnqueueSave(a.expertRoutes + (uint64_t(first + e) * a.config.sourceRowCapacity + written) * 4U,
                             ExpertRouteUb::routes + e * batchItems * 4U, n * 4U);
            totals.SetValue(e, written + n);
        }
        // All bucket sources stay untouched until this one batch completion.
        RouteFence<HardEvent::MTE3_S>();
    }
    // Different owners may share a GM line: publish exact ranges with MTE,
    // never scalar cached writes or per-line DCCI from concurrent writers.
    RouteSave(a.routeHistogram + uint64_t(first) * 4U, ExpertRouteUb::totals, owned * 4U);
}

__aicore__ inline void RouteExpertPrefix(const Arguments& a)
{
    const uint32_t experts = a.config.world * a.config.localExperts;
    RouteLoad(a.routeHistogram, 0U, experts * 4U);
    LocalTensor<uint32_t> counts(TPosition::VECCALC, 0U, experts);
    LocalTensor<uint64_t> exportCounts(TPosition::VECCALC, 8192U, experts);
    LocalTensor<uint64_t> exportOffsets(TPosition::VECCALC, 24576U, experts + 1U);
    auto* r = reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    uint32_t total = 0U;
    for (uint32_t e = 0; e < experts; ++e) {
        const uint32_t n = counts.GetValue(e);
        r->outgoing[e] = total;
        exportOffsets.SetValue(e, uint64_t(total));
        exportCounts.SetValue(e, uint64_t(n));
        total += n;
    }
    r->outgoing[experts] = total;
    exportOffsets.SetValue(experts, uint64_t(total));
    RouteSave(a.routeCounts, 8192U, experts * 8U);
    RouteSave(a.routeOffsets, 24576U, (experts + 1U) * 8U);
    RouteSave(a.routeInfo, kRouteInfoUb, (experts + 1U) * 4U);
}

__aicore__ inline void RouteExchange(const Arguments& a, uint32_t worker)
{
    if (worker >= a.config.world) return;
    const uint32_t le = a.config.localExperts;
    // The leader may build RouteInfo while a slower local sender is still
    // publishing. Read the immutable, separately stored outgoing prefix here;
    // no sender races a DMA overwrite of the shared RouteInfo snapshot.
    RouteLoad(a.routeOffsets + uint64_t(worker) * le * 8U, 2048U, (le + 1U) * 8U);
    LocalTensor<uint64_t> prefix(TPosition::VECCALC, 2048U, le + 1U);
    const uint32_t base = static_cast<uint32_t>(prefix.GetValue(0U));
    LocalTensor<uint64_t> ret(TPosition::VECCALC, 0U, 4U);
    ret.SetValue(0, base);
    LocalTensor<uint32_t> counts(TPosition::VECCALC, 8U, le + 1U);
    for (uint32_t e = 0; e <= le; ++e) counts.SetValue(e, static_cast<uint32_t>(prefix.GetValue(e)) - base);
    auto* sym = reinterpret_cast<__gm__ uint8_t*>(a.symmetric);
    const uint64_t target = reinterpret_cast<uint64_t>(
        aclshmem_ptr(sym + a.config.routeInboxOffset + uint64_t(a.config.rank) * kRouteMailboxBytes, worker));
    RouteSave(target, 0U, 8U + (le + 1U) * 4U);
    LocalTensor<uint64_t> flag(TPosition::VECCALC, 1024U, 4U);
    // RouteSave has completed the payload MTE before publishing its exact
    // epoch into the single ready slot. Do not use an additive SO flag.
    PublishDone(sym, a.config.routeReadyOffset + uint64_t(a.config.rank) * kFlagBytes, worker, a.config.epoch, flag,
                EVENT_ID0);
}

// The small stage order remains explicit to preserve the existing lookahead
// and EP4 wave12 policies. Per-row and per-CV descriptors are not generated.
struct RouteScheduleOutput {
    RouteDescriptorDmaStream<0U> batches;
    RouteDescriptorDmaStream<4096U> tasks;
    __aicore__ explicit RouteScheduleOutput(const Arguments& a) : batches(a.computeBatches), tasks(a.computeTasks) {}
    __aicore__ inline void Batch(uint32_t i, const ComputeBatch& x)
    {
        batches.Append(uint64_t(i) * sizeof(x), x);
    }
    __aicore__ inline void Task(uint32_t i, const ComputeBatchTask& x)
    {
        tasks.Append(uint64_t(i) * sizeof(x), x);
    }
    __aicore__ inline void Finish()
    {
        batches.Flush();
        tasks.Flush();
    }
};

__aicore__ inline void RouteSummary(Arguments& a)
{
    RouteLoad(a.routeInfo, kRouteInfoUb, (a.config.world * a.config.localExperts + 1U) * 4U);
    RouteLoad(a.symmetric + a.config.routeInboxOffset, 0U, a.config.world * kRouteMailboxBytes);
    auto* r = reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    LocalTensor<uint32_t> inbox(TPosition::VECCALC, 0U, a.config.world * kRouteMailboxBytes / 4U);
    for (uint32_t s = 0; s < a.config.world; ++s) {
        const uint32_t at = s * kRouteMailboxBytes / 4U;
        r->returnBase[s] = uint64_t(inbox.GetValue(at)) | (uint64_t(inbox.GetValue(at + 1U)) << 32U);
        for (uint32_t e = 0; e <= a.config.localExperts; ++e) r->incoming[s][e] = inbox.GetValue(at + 2U + e);
    }
    const bool wave12 = a.config.world == 4U && a.sharedExperts != 0U;
    BuildRouteSummary(a.config, *r, wave12);
    // There is no materialized Segment/Link table in this protocol.
    a.config.routeSegmentCount = 0U;
    a.config.routeLinkCount = 0U;
    // Never rewrite the prefix/cache lines used by concurrent Pack/Dispatch.
    constexpr uint32_t receiveOffset = offsetof(DeviceRouteInfo, incoming);
    RouteSave(a.routeInfo + receiveOffset, kRouteInfoUb + receiveOffset, sizeof(DeviceRouteInfo) - receiveOffset);
    RouteScheduleOutput out(a);
    EmitRouteTasks(a.config, *r, out, wave12);
    out.Finish();
    RouteStore(a.routeConfig, 0U, a.config);
    RouteFlush(a.routeConfig, sizeof(Config));
}

__aicore__ inline void RouteReloadConfig(Arguments& a)
{
    RouteFlush(a.routeConfig, sizeof(Config));
    auto* dst = reinterpret_cast<uint32_t*>(&a.config);
    const volatile __gm__ uint32_t* src = reinterpret_cast<const volatile __gm__ uint32_t*>(a.routeConfig);
    for (uint32_t i = 0; i < sizeof(Config) / sizeof(uint32_t); ++i) dst[i] = src[i];
}

// Only local counts are required to issue Pack/Dispatch, including self rows.
// Keep these values in each engine's Arguments until it acquires the receive
// summary. RouteConfig GM is published only once, by the summary owner.
__aicore__ inline void RouteLocalConfig(Arguments& a)
{
    const auto* info = RouteInfo(a);
    a.config.packRoundCount = 0U;
    a.config.sendChunkCount = 0U;
    for (uint32_t peer = 0; peer < a.config.world; ++peer) {
        const uint32_t begin = peer * a.config.localExperts;
        const uint32_t rows = info->outgoing[begin + a.config.localExperts] - info->outgoing[begin];
        const uint32_t cvs = (rows + a.config.cvRows - 1U) / a.config.cvRows;
        a.config.packRoundCount = RouteMax(a.config.packRoundCount, cvs);
        a.config.sendChunkCount += cvs;
    }
    const uint32_t own = a.config.rank * a.config.localExperts;
    a.config.selfSendRow = info->outgoing[own];
    a.config.selfRows = info->outgoing[own + a.config.localExperts] - info->outgoing[own];
    a.config.selfRecvRow = uint64_t(a.config.rank) * a.config.sourceRowCapacity;
    a.config.fragmentCount = a.config.recvChunkCount = a.config.computeTaskCount = 0U;
    a.config.receiveRows = a.config.gmm1Tiles = a.config.gmm2Tiles = 0U;
}

__aicore__ inline void PrepareDeviceRoutes(Arguments& a, Timer& timer)
{
    timer.CoarseBegin(CoarsePhase::Route);
    const uint64_t start = timer.TraceBegin();
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    InvalidateRouteSummary(a);
    // No entry rank barrier: before this owner's previous invocation returned,
    // every peer owner observed that peer's finished epoch, including empty
    // edges. A peer publishes finished only AFTER RouteSummary copied all
    // inboxes to its private RouteInfo. The next serialized invocation can
    // therefore reuse even a single remote mailbox; no benchmark collective
    // or double buffer is part of this argument. Keep the tail observers/join.
    if constexpr (g_coreType == AIV) {
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        ResetInvocationFlags(a, worker);
        RouteOwnedExperts(a, worker);
    }
    // #1: all flag resets, count DMA stores and consumers' cache invalidations precede
    // the leader's first RouteInfo/Config update.
    SyncAll<false>();
    if constexpr (g_coreType == AIV)
        if (block == 0U && GetSubBlockIdx() == 0U) RouteExpertPrefix(a);
    // #2: count prefixes are immutable and ready for the exchange senders.
    SyncAll<false>();
    if constexpr (g_coreType == AIV) RouteExchange(a, LocalWorkerIndex(block, GetSubBlockIdx()));
    // The first activation helper will build the incoming summary while other
    // engines progress independently. No receiver wait or full-core join here.
    RouteLocalConfig(a);
    PipeBarrier<PIPE_ALL>();
    // Incoming metadata is acquired later, per role. This fence only releases
    // this core's route-exchange UB/events before its Pack/compute path.
    timer.TraceEnd(TraceEvent::RoutePrepare, start);
    timer.CoarseEnd(CoarsePhase::Route);
}
}  // namespace MegaMoeUrma
