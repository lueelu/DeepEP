// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "reduce_route_metadata.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// H3584/K8, at most BS8192 over at least 41 reducers. Every worker keeps all of its
// metadata resident so a late token cannot block another metadata batch.
struct ReduceUbContract {
    static constexpr uint32_t hidden = 3584U;
    static constexpr uint32_t topk = 8U;
    static constexpr uint32_t inputs = 6U;
    static constexpr uint32_t outputs = 2U;
    static constexpr uint32_t readyGateEvent = 6U;
    static constexpr uint32_t maxTokens = (8192U + kReduceWorkers - 1U) / kReduceWorkers;
    static constexpr uint32_t maxRoutes = maxTokens * topk;
    static constexpr uint32_t maxRemoteDependencies = kWorld * 512U;
    // Conservative bound across EP4/8/16: at most 16*8192*8 received rows
    // and at most 96 local experts, each contributing one partial M256 group.
    static constexpr uint32_t maxSelfDependencies = kWorld * 8192U * topk / kComputeTileM + kMaxLocalExperts;
    static constexpr uint32_t bf16RowBytes = hidden * sizeof(uint16_t);
    static constexpr uint32_t fp32RowBytes = hidden * sizeof(float);
    static constexpr uint32_t inputBf16Offset = 0U;
    static constexpr uint32_t inputFp32Offset = inputs * bf16RowBytes;
    static constexpr uint32_t accumulatorOffset = inputFp32Offset + inputs * fp32RowBytes;
    static constexpr uint32_t outputOffset = accumulatorOffset + fp32RowBytes;
    static constexpr uint32_t inverseOffset = outputOffset + outputs * bf16RowBytes;
    static constexpr uint32_t weightsOffset = inverseOffset + maxRoutes * sizeof(int64_t);
    static constexpr uint32_t dependenciesOffset = weightsOffset + maxRoutes * sizeof(float);
    static constexpr uint32_t remoteCacheOffset = dependenciesOffset + maxRoutes * sizeof(uint64_t);
    static constexpr uint32_t remoteCacheBytes = (maxRemoteDependencies + 255U) / 256U * 32U;
    static constexpr uint32_t selfCacheOffset = remoteCacheOffset + remoteCacheBytes;
    static constexpr uint32_t selfCacheBytes = (maxSelfDependencies + 255U) / 256U * 32U;
    static constexpr uint32_t issuedOffset = selfCacheOffset + selfCacheBytes;
    static constexpr uint32_t issuedBytes = (maxTokens + 255U) / 256U * 32U;
    static constexpr uint32_t routeRingOffset = issuedOffset + issuedBytes;
    static constexpr uint32_t routeRingBytes = (inputs * sizeof(uint32_t) + 31U) / 32U * 32U;
    static constexpr uint32_t end = routeRingOffset + routeRingBytes;
};
static_assert(ReduceUbContract::maxTokens == 200U && ReduceUbContract::end == 191328U);
static_assert(ReduceUbContract::end <= kRequiredUbBytes);
static_assert(ReduceUbContract::inputFp32Offset % 32U == 0U && ReduceUbContract::accumulatorOffset % 32U == 0U &&
              ReduceUbContract::outputOffset % 32U == 0U && ReduceUbContract::inverseOffset % 32U == 0U &&
              ReduceUbContract::weightsOffset % 32U == 0U && ReduceUbContract::dependenciesOffset % 32U == 0U &&
              ReduceUbContract::remoteCacheOffset % 32U == 0U && ReduceUbContract::selfCacheOffset % 32U == 0U &&
              ReduceUbContract::issuedOffset % 32U == 0U && ReduceUbContract::routeRingOffset % 32U == 0U);

__aicore__ inline bool AllReduceInputsReady(const Arguments& args)
{
    auto* flag =
        reinterpret_cast<__gm__ int32_t*>(args.localFlags + static_cast<uint64_t>(kLocalFlagIncomingDone) * kFlagBytes);
    return AscendC::ReadGmByPassDCache(flag) == static_cast<int32_t>(args.config.world);
}

__aicore__ inline bool ReduceBitSet(AscendC::LocalTensor<uint32_t>& bits, uint32_t index)
{
    return (bits.GetValue(index / 32U) & (uint32_t{1} << (index % 32U))) != 0U;
}

__aicore__ inline void SetReduceBit(AscendC::LocalTensor<uint32_t>& bits, uint32_t index)
{
    const uint32_t word = index / 32U;
    bits.SetValue(word, bits.GetValue(word) | (uint32_t{1} << (index % 32U)));
}

// Never wait inside a token probe. A false dependency immediately returns to
// the scanner. Only positive observations are cached, for this invocation.
__aicore__ inline bool TryReduceToken(const Arguments& args, AscendC::LocalTensor<uint64_t>& dependencies,
                                      AscendC::LocalTensor<uint32_t>& remoteCache,
                                      AscendC::LocalTensor<uint32_t>& selfCache, uint32_t firstRoute)
{
    using namespace AscendC;
    constexpr uint32_t expectedSelfTiles = (ReduceUbContract::hidden + kComputeTileN - 1U) / kComputeTileN;
    for (uint32_t slot = 0U; slot < ReduceUbContract::topk; ++slot) {
        const uint64_t dependency = dependencies.GetValue(firstRoute + slot);
        const bool self = (dependency & kReduceSelfDependency) != 0U;
        const uint32_t index = static_cast<uint32_t>(dependency & ~kReduceSelfDependency);
        auto& cache = self ? selfCache : remoteCache;
        if (ReduceBitSet(cache, index)) continue;
        bool ready;
        if (self) {
            auto* flag = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + static_cast<uint64_t>(index) * kFlagBytes);
            ready = ReadGmByPassDCache(flag) >= static_cast<int32_t>(expectedSelfTiles);
        } else {
            auto* flag = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + args.config.combineReadyOffset +
                                                            static_cast<uint64_t>(index) * kFlagBytes);
            ready = ReadGmByPassDCache(flag) == args.config.epoch;
        }
        if (!ready) return false;
        SetReduceBit(cache, index);
    }
    return true;
}

__aicore__ inline void GateReduceInputs()
{
    using namespace AscendC;
    // Acquire newly observed readiness before any payload read. The credit
    // returns before the reads, protecting gate reuse without draining DMA.
    const auto gate = static_cast<TEventID>(ReduceUbContract::readyGateEvent);
    WaitFlag<HardEvent::MTE2_S>(gate);
    SetFlag<HardEvent::S_MTE2>(gate);
    WaitFlag<HardEvent::S_MTE2>(gate);
    SetFlag<HardEvent::MTE2_S>(gate);
}

// One local stream of routes in actual admission order. The six-item ring is
// slot metadata, not a queue waiting for a batch of ready tokens to accumulate.
// Consumer commands are submitted before reusing a full slot; no token switch
// resets events or drains any hardware engine.
template <typename Element>
class ReduceRoutePipeline {
public:
    __aicore__ inline void Init(const Arguments& args, uint64_t firstToken)
    {
        using namespace AscendC;
        firstToken_ = firstToken;
        returned_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(args.symmetric + args.config.returnYOffset));
        returned_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(args.output));
        inputs_ = LocalTensor<Element>(TPosition::VECCALC, Ub::inputBf16Offset, Ub::inputs * Ub::hidden);
        terms_ = LocalTensor<float>(TPosition::VECCALC, Ub::inputFp32Offset, Ub::inputs * Ub::hidden);
        accumulator_ = LocalTensor<float>(TPosition::VECCALC, Ub::accumulatorOffset, Ub::hidden);
        outputs_ = LocalTensor<Element>(TPosition::VECCALC, Ub::outputOffset, Ub::outputs * Ub::hidden);
        inverse_ = LocalTensor<int64_t>(TPosition::VECCALC, Ub::inverseOffset, Ub::maxRoutes);
        weights_ = LocalTensor<float>(TPosition::VECCALC, Ub::weightsOffset, Ub::maxRoutes);
        routes_ = LocalTensor<uint32_t>(TPosition::VECCALC, Ub::routeRingOffset, Ub::inputs);
        for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
            SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
        }
        for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
            SetFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
        }
        SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(Ub::readyGateEvent));
    }

    __aicore__ inline bool HasPending() const
    {
        return consumed_ != submitted_;
    }

    // Called immediately when a token passes the readiness probe.
    __aicore__ inline void SubmitToken(uint32_t token)
    {
        using namespace AscendC;
        for (uint32_t topk = 0U; topk < Ub::topk; ++topk) {
            if (submitted_ - consumed_ == Ub::inputs) ConsumeOne();
            const uint32_t route = token * Ub::topk + topk;
            const uint32_t slot = submitted_ % Ub::inputs;
            const auto event = static_cast<TEventID>(slot);
            const uint64_t row = static_cast<uint64_t>(inverse_.GetValue(route));
            routes_.SetValue(slot, route);
            WaitFlag<HardEvent::V_MTE2>(event);
            DataCopy(inputs_[slot * Ub::hidden], returned_[row * Ub::hidden], Ub::hidden);
            SetFlag<HardEvent::MTE2_V>(event);
            ++submitted_;
        }
    }

    __aicore__ inline void ConsumeOne()
    {
        using namespace AscendC;
        const uint32_t slot = consumed_ % Ub::inputs;
        const auto event = static_cast<TEventID>(slot);
        const uint32_t route = routes_.GetValue(slot);
        const uint32_t topk = route % Ub::topk;
        const float weight = weights_.GetValue(route);
        LocalTensor<Element> input = inputs_[slot * Ub::hidden];
        LocalTensor<float> term = terms_[slot * Ub::hidden];
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(term, input, RoundMode::CAST_NONE, Ub::hidden);
        PipeBarrier<PIPE_V>();
        if (topk == 0U) {
            Muls(accumulator_, term, weight, Ub::hidden);
        } else {
            Muls(term, term, weight, Ub::hidden);
            PipeBarrier<PIPE_V>();
            Add(accumulator_, accumulator_, term, Ub::hidden);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        if (topk == Ub::topk - 1U) {
            const uint32_t outputSlot = (consumed_ / Ub::topk) % Ub::outputs;
            const auto outputEvent = static_cast<TEventID>(outputSlot);
            LocalTensor<Element> result = outputs_[outputSlot * Ub::hidden];
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            PipeBarrier<PIPE_V>();
            Cast(result, accumulator_, RoundMode::CAST_RINT, Ub::hidden);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopy(output_[(firstToken_ + route / Ub::topk) * Ub::hidden], result, Ub::hidden);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        ++consumed_;
    }

    __aicore__ inline void SubmitPendingCompute()
    {
        while (HasPending()) ConsumeOne();
    }

    __aicore__ inline void Finish()
    {
        using namespace AscendC;
        SubmitPendingCompute();
        WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(Ub::readyGateEvent));
        for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
        }
        for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    using Ub = ReduceUbContract;
    uint64_t firstToken_ = 0U;
    uint32_t submitted_ = 0U;
    uint32_t consumed_ = 0U;
    AscendC::GlobalTensor<Element> returned_, output_;
    AscendC::LocalTensor<Element> inputs_, outputs_;
    AscendC::LocalTensor<float> terms_, accumulator_, weights_;
    AscendC::LocalTensor<int64_t> inverse_;
    AscendC::LocalTensor<uint32_t> routes_;
};

template <typename TraceSink>
__aicore__ inline void Reduce(const Arguments& args, uint32_t worker, TraceSink& timer)
{
    using namespace AscendC;
    using Ub = ReduceUbContract;
    const uint32_t owned = static_cast<uint32_t>(ReduceOwnedRows(args.config.bs, worker, args.config.world));
    if (owned == 0U) return;
    const uint64_t reduceStart = timer.TraceBegin();
    const uint64_t firstToken = ReduceRowBegin(args.config.bs, worker, args.config.world);
    const uint64_t firstRoute = firstToken * Ub::topk;
    GlobalTensor<int64_t> inverse;
    GlobalTensor<float> weights;
    inverse.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.reduceIndex));
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args.weights));
    LocalTensor<int64_t> inverseUb(TPosition::VECCALC, Ub::inverseOffset, Ub::maxRoutes);
    LocalTensor<float> weightsUb(TPosition::VECCALC, Ub::weightsOffset, Ub::maxRoutes);
    LocalTensor<uint64_t> dependenciesUb(TPosition::VECCALC, Ub::dependenciesOffset, Ub::maxRoutes);
    LocalTensor<uint32_t> remoteCache(TPosition::VECCALC, Ub::remoteCacheOffset, Ub::remoteCacheBytes / 4U);
    LocalTensor<uint32_t> selfCache(TPosition::VECCALC, Ub::selfCacheOffset, Ub::selfCacheBytes / 4U);
    LocalTensor<uint32_t> issued(TPosition::VECCALC, Ub::issuedOffset, Ub::issuedBytes / 4U);
    ReduceRoutePipeline<bfloat16_t> pipeline;
    pipeline.Init(args, firstToken);
    bool allInputsReady = AllReduceInputsReady(args);
    if (allInputsReady) GateReduceInputs();
    WaitPackInverseReady(args);
    DataCopy(inverseUb, inverse[firstRoute], owned * Ub::topk);
    DataCopy(weightsUb, weights[firstRoute], owned * Ub::topk);

    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    if (!allInputsReady) BuildReduceDependencies(args, firstRoute, owned * Ub::topk, inverseUb, dependenciesUb);
    for (uint32_t word = 0U; word < Ub::issuedBytes / 4U; ++word) issued.SetValue(word, 0U);
    if (!allInputsReady) {
        for (uint32_t word = 0U; word < Ub::remoteCacheBytes / 4U; ++word) remoteCache.SetValue(word, 0U);
        for (uint32_t word = 0U; word < Ub::selfCacheBytes / 4U; ++word) selfCache.SetValue(word, 0U);
    }

    uint32_t admitted = 0U;
    uint32_t probes = 0U;
    while (admitted != owned) {
        bool progress = false;
        for (uint32_t token = 0U; token < owned; ++token) {
            if (ReduceBitSet(issued, token)) continue;
            if (!allInputsReady) {
                if (probes != 0U && (probes & 7U) == 0U) {
                    allInputsReady = AllReduceInputsReady(args);
                    if (allInputsReady) GateReduceInputs();
                }
                ++probes;
            }
            if (!allInputsReady && !TryReduceToken(args, dependenciesUb, remoteCache, selfCache, token * Ub::topk)) {
                // Keep prior ready tokens moving while inspecting other work.
                if (pipeline.HasPending()) pipeline.ConsumeOne();
                continue;
            }
            if (!allInputsReady) GateReduceInputs();
            SetReduceBit(issued, token);
            pipeline.SubmitToken(token);
            ++admitted;
            progress = true;
        }
        // Submit the remaining V/output commands even without another ready
        // token. This is not an engine drain and does not reset any slot.
        pipeline.SubmitPendingCompute();
        if (!progress && admitted != owned) {
            const uint64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 200U) {
            }
        }
    }
    pipeline.Finish();
    TraceMeta meta;
    meta.row = firstToken;
    meta.rows = owned;
    meta.bytes = static_cast<uint64_t>(owned) * Ub::bf16RowBytes;
    timer.TraceEnd(TraceEvent::Reduce, reduceStart, meta);
}

}  // namespace MegaMoeUrma
