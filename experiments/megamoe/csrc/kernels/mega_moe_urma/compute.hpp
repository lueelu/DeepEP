// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <cfloat>

#include "kernels/mega_moe_urma/abi.hpp"
#include "kernels/mega_moe_urma/compute_contract.hpp"
#include "kernels/mega_moe_urma/lazy_compute_routes.hpp"
#include "kernels/mega_moe_urma/route_handoff.hpp"
#include "kernels/mega_moe_urma/continuous_matmul.hpp"
#include "kernels/mega_moe_urma/deferred_output.hpp"
#include "kernels/mega_moe_urma/consumer_completion.hpp"
#include "kernels/mega_moe_urma/pack_round.hpp"
#include "kernels/mega_moe_urma/tail_completion.hpp"
#include "kernels/mega_moe_urma/transport.hpp"
#include "kernels/mega_moe_urma/timer.hpp"
#include "kernels/mega_moe_common/stage/mega_moe_gmm1_activation.hpp"

namespace MegaMoeUrma {

using namespace AscendC;
using UbxMegaMoeImpl::Get;

// A fragment is a compute M256 group of one expert. Row segments preserve its
// input/output correspondence with the original source-major communication CVs.
__aicore__ inline Fragment LoadFragment(const Arguments& args, uint32_t index)
{
    return RuntimeFragment(args, index);
}

__aicore__ inline __gm__ int32_t* ComputeFlag(uint64_t base, uint64_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(base + index * kFlagBytes);
}

__aicore__ inline void WaitComputeFlag(uint64_t base, uint64_t index, uint32_t expected)
{
    auto* flag = ComputeFlag(base, index);
    while (ReadGmByPassDCache(flag) < static_cast<int32_t>(expected)) {
    }
}

__aicore__ inline uint32_t FragmentGmm1Tiles(const Arguments& args, const Fragment& fragment)
{
    return Gmm1ComputeTileCount(fragment.rows, args.config);
}

__aicore__ inline uint32_t FragmentGmm2Tiles(const Arguments& args, const Fragment& fragment)
{
    return ComputeTileCount(fragment.rows, args.config.hidden);
}

__aicore__ inline UbxMegaMoeImpl::GMMAddrInfo Gmm1Addresses(const Arguments& args, const Fragment& fragment)
{
    const uint64_t h = args.config.hidden;
    const uint64_t n = static_cast<uint64_t>(args.config.intermediate) * 2U;
    const uint64_t scaleH = h / 32U;
    const uint64_t expert = fragment.expert;
    UbxMegaMoeImpl::GMMAddrInfo addr{};
    // X is gathered directly from source-major receive rows. MX scale rows
    // have been packed once by AIV1 to preserve the normal aligned L1 loader.
    addr.aGlobal = reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveXOffset);
    addr.aScaleGlobal = reinterpret_cast<GM_ADDR>(args.gatheredInputScale + fragment.rowBegin * scaleH);
    addr.bGlobal = reinterpret_cast<GM_ADDR>(args.weight1 + expert * h * n / 2U);
    addr.bScaleGlobal = reinterpret_cast<GM_ADDR>(args.weightScale1 + expert * scaleH * n);
    addr.gmm1OutGlobal = reinterpret_cast<GM_ADDR>((args.gmm1Output ? args.gmm1Output : args.activation) +
                                                   fragment.rowBegin * n * sizeof(bfloat16_t));
    return addr;
}

__aicore__ inline UbxMegaMoeImpl::GMMAddrInfo Gmm2Addresses(const Arguments& args, const Fragment& fragment)
{
    const uint64_t h = args.config.hidden;
    const uint64_t i = args.config.intermediate;
    const uint64_t scaleI = i / 32U;
    const uint64_t expert = fragment.expert;
    UbxMegaMoeImpl::GMMAddrInfo addr{};
    addr.aGlobal = reinterpret_cast<GM_ADDR>(args.activation + fragment.rowBegin * i);
    addr.aScaleGlobal = reinterpret_cast<GM_ADDR>(args.activationScale + fragment.rowBegin * scaleI);
    addr.bGlobal = reinterpret_cast<GM_ADDR>(args.weight2 + expert * i * h / 2U);
    addr.bScaleGlobal = reinterpret_cast<GM_ADDR>(args.weightScale2 + expert * scaleI * h);
    // The peer's communication AIV1 copies full rows after all group N tiles finish.
    addr.gmm2OutGlobal = reinterpret_cast<GM_ADDR>(args.gmm2Output + fragment.rowBegin * h * sizeof(bfloat16_t));
    return addr;
}

using UrmaGmmConfig = UbxMegaMoeImpl::GmmKernel::Config<true, 0, fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, fp8_e8m0_t,
                                                        fp8_e8m0_t, false, false, false, false, false, false, false>;
using UrmaProblemConfig = UrmaGmmConfig::ProblemConfig;
using UrmaRoutedGmmConfig =
    UbxMegaMoeImpl::GmmKernel::Config<true, 0, fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, fp8_e8m0_t, fp8_e8m0_t, false,
                                      false, false, false, false, false, true>;
using UrmaInputRows = UbxMegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false, RuntimeGroupSegments>;
using UrmaOutputRows = UbxMegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, false, true, RuntimeGroupSegments>;
// Only URMA uses a separate FP32 sigmoid, matching the NPU cascade's operation order.
using UrmaActivation = UbxMegaMoeImpl::BlockEpilogueActivationMxQuant<fp8_e4m3fn_t, bfloat16_t, kActivationTileM,
                                                                      kComputeTileN, false, false, true>;

// Tie the host-testable memory contract to the shared arithmetic's real UB
// layout. Top-k weights are consumed by Reduce, so no prefetch buffer exists.
constexpr auto kActualActivationUb =
    UbxMegaMoeImpl::Detail::BuildActivationMxQuantUbOffsets<bfloat16_t, kActivationTileM, kComputeTileN, false>();
static_assert(kActualActivationUb.firstInputOffsetBytes == ActivationUbContract::gateOffset &&
                  kActualActivationUb.secondInputOffsetBytes == ActivationUbContract::upOffset &&
                  kActualActivationUb.activationOutputOffsetBytes == ActivationUbContract::activationOffset &&
                  kActualActivationUb.quantOutputOffsetBytes == ActivationUbContract::quantOffset &&
                  kActualActivationUb.quantScaleOffsetBytes == ActivationUbContract::scaleOffset &&
                  kActualActivationUb.maxExpOffsetBytes == ActivationUbContract::maxExpOffset &&
                  kActualActivationUb.reciprocalScaleOffsetBytes == ActivationUbContract::reciprocalOffset &&
                  kActualActivationUb.scaleElementCapacity == ActivationUbContract::scaleElements,
              "URMA activation UB contract must match the shared epilogue");

template <typename RowIO>
__aicore__ inline RowIO FragmentRowIO(const Arguments& args, const Fragment& fragment)
{
    return {nullptr,
            fragment.segmentCount,
            args.config.rank,
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveXOffset),
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveScaleOffset),
            args.config.hidden,
            args.config.hidden / 32U,
            reinterpret_cast<GM_ADDR>(args.gmm2Output),
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.returnYOffset),
            args.config.hidden,
            RuntimeSegments(args, fragment)};
}

__aicore__ inline UbxMegaMoeImpl::ProblemShape FragmentShape(const Arguments& args, const Fragment& fragment)
{
    return {fragment.rows, static_cast<int64_t>(args.config.intermediate) * 2, args.config.hidden, 0};
}

__aicore__ inline TraceMeta FragmentTraceMeta(const Fragment& fragment, uint32_t index)
{
    TraceMeta meta;
    // A compute group may span several peers/CVs. Its aggregate input-ready
    // wait uses the fragment as the stable compute-group ID.
    meta.fragment = index;
    meta.expert = fragment.expert;
    meta.row = fragment.rowBegin;
    meta.rows = fragment.rows;
    return meta;
}

__aicore__ inline TraceMeta ComputeTileTraceMeta(const TraceMeta& fragmentMeta, uint32_t m, uint32_t n, uint32_t rows,
                                                 uint32_t columns)
{
    TraceMeta meta = fragmentMeta;
    meta.row += m;
    meta.rows = rows;
    // These are actual scheduler element coordinates, not linear tile IDs.
    meta.tileM = m;
    meta.tileN = n;
    meta.columns = columns;
    return meta;
}

// A new context must not republish the initial 8/9 free-slot credits until
// AIV0 has consumed the old context's final credits. The acknowledgement is
// ordered on those same MTE pipelines; it needs neither a scalar fence nor a
// full-core barrier. Only this AIC and its paired AIV0 use the context-end flag beyond the four B slots.
__aicore__ inline void CompleteComputeContext()
{
    constexpr uint16_t kContextEndFlag = UbxMegaMoeImpl::GmmKernel::A8W4BDeepResources::ContextEnd;
    if constexpr (g_coreType == AIC) {
        CrossCoreWaitFlag<4, PIPE_MTE1>(kContextEndFlag);
    } else {
        CrossCoreSetFlag<4, PIPE_MTE3>(kContextEndFlag);
    }
}

// Routed FIX completion is owned by paired AIV1 consumers. Shared GMMs keep
// their independent completion backend and do not consume these two channels.
using UrmaContinuousGmm = ContinuousMatmul<UrmaProblemConfig>;

template <class TimerT>
__aicore__ inline void RunComputeFragment(const Arguments& args, uint32_t block, uint32_t fragmentIndex,
                                          uint32_t firstTile, bool gmm1, UrmaContinuousGmm& pipeline,
                                          RoutedCompletionProducer& completion, TimerT& timer)
{
    const UbxMegaMoeImpl::BlockJobContext job{block, kAic};
    const Fragment fragment = LoadFragment(args, fragmentIndex);
    const auto fragmentMeta = FragmentTraceMeta(fragment, fragmentIndex);
    const auto shape = FragmentShape(args, fragment);
    auto config = gmm1 ? UrmaRoutedGmmConfig::BuildGmm1ProblemConfig(shape, job, kComputeTileM)
                       : UrmaRoutedGmmConfig::BuildGmm2ProblemConfig(shape, job, kComputeTileM);
    UbxMegaMoeImpl::GmmKernel::BlockScheduler schedule(
        {config.m, config.schedulerN, config.k},
        UbxMegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(config.tileN))});
    if constexpr (g_coreType == AIC) {
        const uint64_t flags = gmm1 ? args.inputScaleDone : args.activationDone;
        const uint32_t expected = gmm1 ? 1U : FragmentGmm1Tiles(args, fragment);
        const bool ready = ReadGmByPassDCache(ComputeFlag(flags, fragmentIndex)) >= static_cast<int32_t>(expected);
        // In particular, never retain the last GMM1 flag while blocking on
        // its activation: that would prevent the helper from making progress.
        // Earlier ready publications are progressed independently by paired AIV1.
        const uint64_t wait = 0U;
        if (!ready) WaitComputeFlag(flags, fragmentIndex, expected);
    }
    const Phase phase = gmm1 ? Phase::Gmm1 : Phase::Gmm2;
    timer.Begin(phase);
    const auto addresses = gmm1 ? Gmm1Addresses(args, fragment) : Gmm2Addresses(args, fragment);
    pipeline.UpdateProblem(config, addresses, gmm1);
    ContinuousComputeRows rowIO;
    static_cast<UrmaInputRows&>(rowIO) = FragmentRowIO<UrmaInputRows>(args, fragment);
    rowIO.gather = gmm1;
    rowIO.storeToUb = true;
    if (!gmm1) {
        rowIO.inputX = addresses.aGlobal;
        rowIO.inputStride = args.config.intermediate;
    }
    pipeline.SetRowIO(rowIO);
    pipeline.Begin();  // Idempotent; only the invocation's first owned tile opens events.
    const uint32_t count = schedule.GetTileNum();
    for (uint32_t tile = firstTile; tile < count; tile += kAic) {
        const auto coord = schedule.GetBlockCoord(tile);
        const auto tileShape = schedule.GetBlockShape(coord);
        const uint32_t m = Get<UbxMegaMoeImpl::M_VALUE>(coord);
        const uint32_t n = Get<UbxMegaMoeImpl::N_VALUE>(coord);
        const uint32_t rows = Get<UbxMegaMoeImpl::M_VALUE>(tileShape);
        const uint32_t columns = Get<UbxMegaMoeImpl::N_VALUE>(tileShape);
        const auto tileMeta = ComputeTileTraceMeta(fragmentMeta, m, n, rows, columns);
        if constexpr (g_coreType == AIC) {
            const uint64_t start = timer.TraceBegin();
            completion.Acquire();
            pipeline.Compute(m, n, rows, columns);
            timer.TraceEnd(gmm1 ? TraceEvent::Gmm1Tile : TraceEvent::Gmm2Tile, start, tileMeta);
            // Enqueue actual FIX completion, then announce issue. Scalar does
            // not wait for the output and cannot publish compute-ready here.
            completion.Commit();
        } else {
            const uint64_t start = timer.TraceBegin();
            pipeline.Dequantize(n, rows, columns);
            timer.TraceEnd(gmm1 ? TraceEvent::Gmm1Dequant : TraceEvent::Gmm2Dequant, start, tileMeta);
        }
    }
    timer.End(phase);
}

__aicore__ inline ComputeBatchRange LoadComputeBatch(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ ComputeBatch*>(args.computeBatches) + index;
    const ComputeBatch batch{gm->fragmentBegin, gm->fragmentCount};
    return ComputeBatchFragments(batch);
}

__aicore__ inline ComputeBatchTask LoadComputeTask(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ ComputeBatchTask*>(args.computeTasks) + index;
    return {gm->stage, gm->batchIndex};
}

// Only an AIC and its paired AIV0 enter this routine. They use the identical
// host-planned expert schedule and global tile ownership. SwiGLU remains
// independent; the GMM1 M-group window covers earlier activation latency.
template <class TimerT>
__aicore__ inline void RunCubeAndPrologue(const Arguments& args, uint32_t block, TimerT& timer)
{
    UrmaContinuousGmm pipeline;
    RoutedCompletionProducer completion(args, block);
    bool hasWork = false;
    for (uint32_t step = 0U; step < args.config.computeTaskCount; ++step) {
        const auto task = LoadComputeTask(args, step);
        const auto range = LoadComputeBatch(args, task.batchIndex);
        if (range.begin == range.end) continue;
        const auto first = LoadFragment(args, range.begin);
        const bool isGmm1 = task.stage == ComputeBatchStage::Gmm1;
        const uint64_t tileBegin = isGmm1 ? first.gmm1TileBegin : first.gmm2TileBegin;
        const uint32_t tilesPerGroup = isGmm1 ? FragmentGmm1Tiles(args, first) : FragmentGmm2Tiles(args, first);
        const uint64_t tileLimit = static_cast<uint64_t>(range.end - range.begin) * tilesPerGroup;
        // Jump directly between this core's groups before loading their GM
        // descriptors or building matmul configs. AIC/AIV0 keep identical
        // tile order and close the context on their last owned group.
        for (uint64_t tile = FirstOwnedComputeTile(tileBegin, RoutedOwnerResidue(isGmm1, block), kAic);
             tile < tileLimit;) {
            const auto owned = OwnedComputeGroupAt(tile, tilesPerGroup, kAic);
            const uint64_t nextTile = tile + static_cast<uint64_t>(owned.tileCount) * kAic;
            const uint32_t fragmentIndex = range.begin + owned.group;
            RunComputeFragment(args, block, fragmentIndex, owned.firstTile, isGmm1, pipeline, completion, timer);
            hasWork = true;
            tile = nextTile;
        }
    }
    if constexpr (g_coreType == AIC) completion.Finish(timer);
    if (hasWork) {
        pipeline.End(g_coreType == AIC);
        CompleteComputeContext();
    }
}

__aicore__ inline bool AreGroupInputsReady(const Arguments& args, const Fragment& fragment)
{
    auto cursor = RuntimeSegments(args, fragment).MakeCursor();
    RowSegment segment{};
    while (cursor.Next(segment)) {
        if (segment.sourceRank == args.config.rank) {
            const uint64_t begin = static_cast<uint64_t>(segment.cv) * args.config.cvRows;
            if (begin >= args.config.selfRows) return false;
            const uint64_t remaining = args.config.selfRows - begin;
            const uint32_t rows = remaining < args.config.cvRows ? remaining : args.config.cvRows;
            if (ReadGmByPassDCache(PackReadyFlag(args, args.config.rank, segment.cv)) < static_cast<int32_t>(rows))
                return false;
            continue;
        }
        auto* ready = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + args.config.dispatchReadyOffset +
                                                         segment.inputReadyIndex * kFlagBytes);
        if (ReadGmByPassDCache(ready) != args.config.epoch) return false;
    }
    return true;
}

// A single worker owns a whole group's scales. Arbitrary segment row offsets
// cannot be represented safely by the MX zZ L1 copy, so gather just this small
// plane in aligned UB rows, then store compact GM rows for the original loader.
template <uint32_t FixedBytes>
__aicore__ inline void GatherInputScaleImpl(const Arguments& args, const Fragment& fragment, uint32_t index,
                                            Timer& timer)
{
    const uint32_t scaleBytes = FixedBytes ? FixedBytes : args.config.hidden / 32U;
    const uint32_t scaleStride = (scaleBytes + 31U) / 32U * 32U;
    GlobalTensor<uint8_t> source, destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveScaleOffset));
    source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.gatheredInputScale));
    LocalTensor<uint8_t> scratch(TPosition::VECCALC, 128U * 1024U, kComputeTileM * scaleStride);
    // H<=8192 is the public dimension contract; the largest gathered plane
    // ends at192KiB, below the persistent counters at240KiB.
    static_assert(128U * 1024U + kComputeTileM * (8192U / 32U) <= kAiv1PersistentUbBase);
    const uint64_t start = timer.TraceBegin();
    auto cursor = RuntimeSegments(args, fragment).MakeCursor();
    RowSegment segment{};
    while (cursor.Next(segment)) {
        DataCopyPad(scratch[segment.tileRow * scaleStride], source[segment.receiveRow * scaleBytes],
                    {static_cast<uint16_t>(segment.rows), scaleBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    }
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyPad(destination[fragment.rowBegin * scaleBytes], scratch,
                {static_cast<uint16_t>(fragment.rows), scaleBytes, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WriteGmByPassDCache(ComputeFlag(args.inputScaleDone, index), 1);
    auto meta = FragmentTraceMeta(fragment, index);
    meta.bytes = static_cast<uint64_t>(fragment.rows) * scaleBytes;
    timer.TraceEnd(TraceEvent::InputScaleGather, start, meta);
}

__aicore__ inline void GatherInputScale(const Arguments& a, const Fragment& f, uint32_t i, Timer& t)
{
    if (a.config.hidden == 3584U)
        GatherInputScaleImpl<112U>(a, f, i, t);
    else
        GatherInputScaleImpl<0U>(a, f, i, t);
}

__aicore__ inline void RunReadyActivationTile(const Arguments& args, uint32_t worker, const Fragment& fragment,
                                              uint32_t fragmentIndex, uint32_t tile, UrmaActivation& epilogue,
                                              Timer& timer, uint32_t workers = 0U)
{
    const UbxMegaMoeImpl::BlockJobContext job{worker, workers ? workers : ActivationWorkers(args.config.world)};
    const auto fragmentMeta = FragmentTraceMeta(fragment, fragmentIndex);
    auto addr = Gmm1Addresses(args, fragment);
    addr.activationToGmm2Flag = nullptr;
    epilogue.UpdateGlobalAddr({static_cast<int64_t>(fragment.rowBegin * args.config.intermediate),
                               static_cast<int64_t>(fragment.rowBegin * (args.config.intermediate / 32U)), 0, 0, 0, 0});
    auto config = UrmaRoutedGmmConfig::BuildGmm1ProblemConfig(FragmentShape(args, fragment), job, kComputeTileM);
    const auto layouts = UrmaRoutedGmmConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t*>(addr.gmm1OutGlobal)), layouts.c);
    UbxMegaMoeImpl::GmmKernel::BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        UbxMegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(kComputeTileM), static_cast<int64_t>(kGmm1LogicalTileN))});
    epilogue.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    const auto coord = scheduler.GetBlockCoord(tile);
    const auto tileShape = scheduler.GetBlockShape(coord);
    const uint32_t m = Get<UbxMegaMoeImpl::M_VALUE>(coord);
    const uint32_t n = Get<UbxMegaMoeImpl::N_VALUE>(coord);
    const uint32_t rows = Get<UbxMegaMoeImpl::M_VALUE>(tileShape);
    const uint32_t columns = Get<UbxMegaMoeImpl::N_VALUE>(tileShape);
    // Only the staging copy changes to M128; ready counters stay M256.
    config.tileM = kActivationTileM;
    timer.Begin(Phase::Activation);
    for (uint32_t offset = 0U; offset < rows; offset += kActivationTileM) {
        const auto slice = ActivationSlice(m, rows, offset);
        const UbxMegaMoeImpl::GmmKernel::BlockScheduler::BlockShape sliceShape{slice.rows, columns, config.k};
        const auto sliceMeta = ComputeTileTraceMeta(fragmentMeta, slice.rowBegin, n, slice.rows, columns);
        const uint64_t sliceStart = timer.TraceBegin();
        UbxMegaMoeImpl::GmmKernel::Gmm1Aiv1EpilogueTileA8W4<bfloat16_t, UrmaRoutedGmmConfig::MakeLayoutC, true>(
            epilogue, gmC, addr, config, sliceShape, slice.rowBegin, n);
        timer.TraceEnd(TraceEvent::ActivationSlice, sliceStart, sliceMeta);
    }
    // The epilogue has drained all output DMA before the gather task can reuse UB.
    AtomicAdd(ComputeFlag(args.activationDone, fragmentIndex), 1);
    timer.End(Phase::Activation);
}

// Noncommunication AIV1 workers own scale gather, activation and static Pack,
// strictly in that order. The gather moves only a small aligned scale plane,
// but it publishes inputScaleDone, which gates the AIC's GMM1 for the whole
// group; leaving it behind an activation tile stalls GMM1 longer than the
// gather itself costs. Activation keeps tile granularity (one tile per turn),
// and Pack only fills the bubbles the two stages above leave unused.
// Reorder is exclusively performed by peer communication owners.
__aicore__ inline void RunActivation(Arguments& args, uint32_t worker, Timer& timer)
{
    const uint32_t activationWorkers = ActivationWorkers(args.config.world);
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    uint32_t nextPackRound = kPackStartupRounds;
    PackBatch earlyPack(args, timer);
    PrepareActivationRoutes(args, worker, nextPackRound, earlyPack, timer);
    earlyPack.Close();  // keep the existing blocking steady Pack/scale/activation path
    timer.CoarseBegin(CoarsePhase::Activation);
    UrmaActivation epilogue;
    // Match main MegaMoE's unclamped SwiGLU. The epilogue default is zero,
    // which clamps every up value to zero and makes the entire FFN zero.
    epilogue.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(args.activation),
                   .yScaleGmAddr = reinterpret_cast<GM_ADDR>(args.activationScale),
                   .clampLimit = FLT_MAX});
    uint32_t nextGather = worker;
    uint32_t nextActivation = 0U;
    uint32_t activationTile = 0U;

    bool packFinished = false;
    bool activationPrepared = false;
    bool idle = false;
    uint64_t idleStart = 0U;
    TraceEvent idleEvent = TraceEvent::WaitGmm1;
    TraceMeta idleMeta;
    while (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount ||
           nextPackRound < args.config.packRoundCount) {
        bool progress = false;
        // Scale gather first: it is short, but inputScaleDone is what the AIC
        // waits on once per group before issuing GMM1. Finishing activation
        // tiles ahead of it parks GMM1 behind the activation queue.
        if (nextGather < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextGather);
            if (AreGroupInputsReady(args, fragment)) {
                if (idle) {
                    (void)idleStart;
                    idle = false;
                }
                GatherInputScale(args, fragment, nextGather, timer);
                nextGather += activationWorkers;
                progress = true;
            }
        }
        // Activation second: it keeps tile granularity, one tile per turn, so a
        // freshly published GMM1 tile waits at most one gather or pack unit.
        // Skip groups without this worker's activation tile, but never skip a
        // scale gather merely because its owner has no activation in that group.
        if (!progress) {
            while (nextActivation < args.config.fragmentCount) {
                const auto fragment = LoadFragment(args, nextActivation);
                const uint32_t tileCount = FragmentGmm1Tiles(args, fragment);
                if (!activationPrepared) {
                    activationTile = FirstOwnedComputeTile(fragment.gmm1TileBegin, worker, activationWorkers);
                    activationPrepared = true;
                }
                if (activationTile >= tileCount) {
                    ++nextActivation;
                    activationPrepared = false;
                    continue;
                }
                if (ReadGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + activationTile)) != 0) {
                    if (idle) {
                        (void)idleStart;
                        idle = false;
                    }
                    RunReadyActivationTile(args, worker, fragment, nextActivation, activationTile, epilogue, timer);
                    activationTile += activationWorkers;
                    progress = true;
                }
                break;
            }
        }
        // Pack is the last resort: it runs only when neither of the two stages
        // above made progress in this turn.
        if (!progress && nextPackRound < args.config.packRoundCount) {
            if (idle) {
                (void)idleStart;
                idle = false;
            }
            PackOneRound(args, nextPackRound, worker + communicationWorkers, kAic, timer);
            ++nextPackRound;
            progress = true;
        }
        if (!packFinished && nextPackRound >= args.config.packRoundCount) {
            MarkPackWorkerComplete(args);
            packFinished = true;
        }
        if (!progress && (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount)) {
            if (!idle) {
                idleStart = 0U;
                idle = true;
                const bool waitingGather = nextGather < args.config.fragmentCount;
                const uint32_t index = waitingGather ? nextGather : nextActivation;
                idleEvent = waitingGather ? TraceEvent::WaitDispatch : TraceEvent::WaitGmm1;
                idleMeta = FragmentTraceMeta(LoadFragment(args, index), index);
            }
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
        }
    }
    if (idle) (void)idleStart;
    // Empty compute workloads and <=2 Pack rounds can skip the loop entirely.
    if (!packFinished) MarkPackWorkerComplete(args);
    timer.CoarseEnd(CoarsePhase::Activation);
}

}  // namespace MegaMoeUrma
