// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "reduce_pipeline.hpp"

#include "kernel_operator.h"
#include <type_traits>

namespace MegaMoeUrma::GenericReduce {

// H sliced at 2048 columns; TopK 1..8; BS16384 over at least 41 reducers. Every worker keeps all of its
// metadata resident so a late token cannot block another metadata batch.
struct ReduceUbContract {
    static constexpr uint32_t hidden = 2048U;
    static constexpr uint32_t topk = 8U;  // capacity, NOT the active token stride
    static constexpr uint32_t inputs = 6U;
    static constexpr uint32_t outputs = 2U;
    static constexpr uint32_t readyGateEvent = 6U;
    static constexpr uint32_t maxTokens = (kMaxTokensPerRank + kReduceWorkers - 1U) / kReduceWorkers;
    static constexpr uint32_t maxRoutes = maxTokens * topk;
    static constexpr uint32_t maxRemoteDependencies = kWorld * kMaxPeerChunks;
    // Conservative bound across EP4/8/16: at most 16*16384*8 received rows
    // and kMaxLocalExperts experts, each contributing one partial M256 group.
    static constexpr uint32_t maxSelfDependencies =
        kWorld * kMaxTokensPerRank * topk / kComputeTileM + kMaxLocalExperts;
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
static_assert(ReduceUbContract::maxTokens == 400U);
static_assert(ReduceUbContract::end <= kRequiredUbBytes);
static_assert(ReduceUbContract::inputFp32Offset % 32U == 0U && ReduceUbContract::accumulatorOffset % 32U == 0U &&
              ReduceUbContract::outputOffset % 32U == 0U && ReduceUbContract::inverseOffset % 32U == 0U &&
              ReduceUbContract::weightsOffset % 32U == 0U && ReduceUbContract::dependenciesOffset % 32U == 0U &&
              ReduceUbContract::remoteCacheOffset % 32U == 0U && ReduceUbContract::selfCacheOffset % 32U == 0U &&
              ReduceUbContract::issuedOffset % 32U == 0U && ReduceUbContract::routeRingOffset % 32U == 0U);

__aicore__ inline bool GenericAllReduceInputsReady(const Arguments& args)
{
    auto* flag =
        reinterpret_cast<__gm__ int32_t*>(args.localFlags + static_cast<uint64_t>(kLocalFlagIncomingDone) * kFlagBytes);
    return AscendC::ReadGmByPassDCache(flag) == static_cast<int32_t>(args.config.world);
}

__aicore__ inline bool GenericReduceBitSet(AscendC::LocalTensor<uint32_t>& bits, uint32_t index)
{
    return (bits.GetValue(index / 32U) & (uint32_t{1} << (index % 32U))) != 0U;
}

__aicore__ inline void GenericSetReduceBit(AscendC::LocalTensor<uint32_t>& bits, uint32_t index)
{
    const uint32_t word = index / 32U;
    bits.SetValue(word, bits.GetValue(word) | (uint32_t{1} << (index % 32U)));
}

// Never wait inside a token probe. A false dependency immediately returns to
// the scanner. Only positive observations are cached, for this invocation.
template <uint32_t TopK>
__aicore__ inline bool GenericTryReduceToken(const Arguments& args, AscendC::LocalTensor<uint64_t>& dependencies,
                                             AscendC::LocalTensor<uint32_t>& remoteCache,
                                             AscendC::LocalTensor<uint32_t>& selfCache, uint32_t firstRoute)
{
    using namespace AscendC;
    const uint32_t expectedSelfTiles = (args.config.hidden + kComputeTileN - 1U) / kComputeTileN;
    for (uint32_t slot = 0U; slot < TopK; ++slot) {
        const uint64_t dependency = dependencies.GetValue(firstRoute + slot);
        const bool self = (dependency & kReduceSelfDependency) != 0U;
        const uint32_t index = static_cast<uint32_t>(dependency & ~kReduceSelfDependency);
        auto& cache = self ? selfCache : remoteCache;
        if (GenericReduceBitSet(cache, index)) continue;
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
        GenericSetReduceBit(cache, index);
    }
    return true;
}

__aicore__ inline void GenericGateReduceInputs()
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
template <typename Element, bool KeepFp32 = false, uint32_t TopK = 8U>
class ReduceRoutePipeline {
public:
    __aicore__ inline void Init(const Arguments& args, uint64_t firstToken, uint32_t column, uint32_t columns)
    {
        using namespace AscendC;
        firstToken_ = firstToken;
        stride_ = args.config.hidden;
        column_ = column;
        columns_ = columns;
        returned_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(args.symmetric + args.config.returnYOffset));
        returned_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ Output*>(KeepFp32 ? args.sharedRoutedOutput : args.output));
        inputs_ = LocalTensor<Element>(TPosition::VECCALC, Ub::inputBf16Offset, Ub::inputs * Ub::hidden);
        terms_ = LocalTensor<float>(TPosition::VECCALC, Ub::inputFp32Offset, Ub::inputs * Ub::hidden);
        accumulator_ = LocalTensor<float>(TPosition::VECCALC, Ub::accumulatorOffset, Ub::hidden);
        outputs_ =
            LocalTensor<Output>(TPosition::VECCALC, KeepFp32 ? Ub::end : Ub::outputOffset, Ub::outputs * Ub::hidden);
        static_assert(Ub::end + Ub::outputs * Ub::fp32RowBytes <= kRequiredUbBytes);
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
        for (uint32_t topk = 0U; topk < TopK; ++topk) {
            if (submitted_ - consumed_ == Ub::inputs) ConsumeOne();
            const uint32_t route = token * TopK + topk;
            const uint32_t slot = submitted_ % Ub::inputs;
            const auto event = static_cast<TEventID>(slot);
            const uint64_t row = static_cast<uint64_t>(inverse_.GetValue(route));
            routes_.SetValue(slot, route);
            WaitFlag<HardEvent::V_MTE2>(event);
            DataCopy(inputs_[slot * Ub::hidden], returned_[row * stride_ + column_], columns_);
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
        const uint32_t topk = route % TopK;
        const float weight = weights_.GetValue(route);
        LocalTensor<Element> input = inputs_[slot * Ub::hidden];
        LocalTensor<float> term = terms_[slot * Ub::hidden];
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(term, input, RoundMode::CAST_NONE, columns_);
        PipeBarrier<PIPE_V>();
        if (topk == 0U) {
            Muls(accumulator_, term, weight, columns_);
        } else {
            Muls(term, term, weight, columns_);
            PipeBarrier<PIPE_V>();
            Add(accumulator_, accumulator_, term, columns_);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        if (topk == TopK - 1U) {
            const uint32_t outputSlot = (consumed_ / TopK) % Ub::outputs;
            const auto outputEvent = static_cast<TEventID>(outputSlot);
            LocalTensor<Output> result = outputs_[outputSlot * Ub::hidden];
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            PipeBarrier<PIPE_V>();
            if constexpr (KeepFp32)
                DataCopy(result, accumulator_, columns_);
            else
                Cast(result, accumulator_, RoundMode::CAST_RINT, columns_);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopy(output_[(firstToken_ + route / TopK) * stride_ + column_], result, columns_);
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
    using Output = std::conditional_t<KeepFp32, float, Element>;
    uint64_t firstToken_ = 0U;
    uint32_t stride_ = 0, column_ = 0, columns_ = 0;
    uint32_t submitted_ = 0U;
    uint32_t consumed_ = 0U;
    AscendC::GlobalTensor<Element> returned_;
    AscendC::GlobalTensor<Output> output_;
    AscendC::LocalTensor<Element> inputs_;
    AscendC::LocalTensor<Output> outputs_;
    AscendC::LocalTensor<float> terms_, accumulator_, weights_;
    AscendC::LocalTensor<int64_t> inverse_;
    AscendC::LocalTensor<uint32_t> routes_;
};

template <bool KeepFp32 = false, uint32_t TopK = 8U, typename TraceSink, class Observer>
__aicore__ inline void ReduceColumns(const Arguments& args, uint32_t worker, uint32_t workers, uint32_t column,
                                     TraceSink& timer, Observer& observer)
{
    using namespace AscendC;
    using Ub = ReduceUbContract;
    const uint32_t owned =
        static_cast<uint32_t>((args.config.bs / workers + (worker < args.config.bs % workers ? 1U : 0U)));
    if (owned == 0U) return;
    const uint64_t reduceStart = timer.TraceBegin();
    const uint64_t firstToken = (uint64_t(worker) * (args.config.bs / workers) +
                                 (worker < args.config.bs % workers ? worker : args.config.bs % workers));
    const uint64_t firstRoute = firstToken * TopK;
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
    ReduceRoutePipeline<bfloat16_t, KeepFp32, TopK> pipeline;
    const uint32_t columns = args.config.hidden - column < Ub::hidden ? args.config.hidden - column : Ub::hidden;
    pipeline.Init(args, firstToken, column, columns);
    bool allInputsReady = GenericAllReduceInputsReady(args);
    if (allInputsReady) GenericGateReduceInputs();
    static_assert(TopK >= 1U && TopK <= ReduceUbContract::topk);
    if (column == 0U) {
        // These three arrays are outside the column payload scratch and remain
        // resident across all H slices. Build readiness exactly once per entry,
        // never once per polling pass or once per 2048-column slice.
        WaitPackInverseReady(args);
        if constexpr (TopK == 8U) {
            DataCopy(inverseUb, inverse[firstRoute], owned * TopK);
            DataCopy(weightsUb, weights[firstRoute], owned * TopK);
        } else {
            const uint32_t routes = owned * TopK;
            DataCopyPad(inverseUb, inverse[firstRoute], {1U, routes * 8U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            DataCopyPad(weightsUb, weights[firstRoute], {1U, routes * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0.0F});
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (!allInputsReady) BuildReduceDependencies(args, firstRoute, owned * TopK, inverseUb, dependenciesUb);
    }
    for (uint32_t word = 0U; word < Ub::issuedBytes / 4U; ++word) issued.SetValue(word, 0U);
    if (!allInputsReady) {
        for (uint32_t word = 0U; word < Ub::remoteCacheBytes / 4U; ++word) remoteCache.SetValue(word, 0U);
        for (uint32_t word = 0U; word < Ub::selfCacheBytes / 4U; ++word) selfCache.SetValue(word, 0U);
    }

    uint32_t admitted = 0U;
    uint32_t probes = 0U;
    while (admitted != owned) {
        observer.Poll();
        bool progress = false;
        for (uint32_t token = 0U; token < owned; ++token) {
            if (GenericReduceBitSet(issued, token)) continue;
            if (!allInputsReady) {
                if (probes != 0U && (probes & 7U) == 0U) {
                    allInputsReady = GenericAllReduceInputsReady(args);
                    if (allInputsReady) GenericGateReduceInputs();
                }
                ++probes;
            }
            if (!allInputsReady &&
                !GenericTryReduceToken<TopK>(args, dependenciesUb, remoteCache, selfCache, token * TopK)) {
                // Keep prior ready tokens moving while inspecting other work.
                if (pipeline.HasPending()) pipeline.ConsumeOne();
                continue;
            }
            if (!allInputsReady) GenericGateReduceInputs();
            GenericSetReduceBit(issued, token);
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
    meta.bytes = static_cast<uint64_t>(owned) * columns * (KeepFp32 ? 4U : 2U);
    timer.TraceEnd(TraceEvent::Reduce, reduceStart, meta);
}

}  // namespace MegaMoeUrma::GenericReduce

namespace MegaMoeUrma {
struct NoReduceObserver {
    __aicore__ inline void Poll() {}
};
template <bool KeepFp32 = false, class T, class O>
__aicore__ inline void ReduceGeneric(const Arguments& a, uint32_t worker, uint32_t workers, T& timer, O& observer)
{
    for (uint32_t col = 0; col < a.config.hidden; col += GenericReduce::ReduceUbContract::hidden) {
        // Native/API validation guarantees [1,8]. Compile-time strides retain
        // constant division and loop bounds; no per-route runtime TopK division.
        switch (a.config.topk) {
            case 1:
                GenericReduce::ReduceColumns<KeepFp32, 1>(a, worker, workers, col, timer, observer);
                break;
            case 2:
                GenericReduce::ReduceColumns<KeepFp32, 2>(a, worker, workers, col, timer, observer);
                break;
            case 3:
                GenericReduce::ReduceColumns<KeepFp32, 3>(a, worker, workers, col, timer, observer);
                break;
            case 4:
                GenericReduce::ReduceColumns<KeepFp32, 4>(a, worker, workers, col, timer, observer);
                break;
            case 5:
                GenericReduce::ReduceColumns<KeepFp32, 5>(a, worker, workers, col, timer, observer);
                break;
            case 6:
                GenericReduce::ReduceColumns<KeepFp32, 6>(a, worker, workers, col, timer, observer);
                break;
            case 7:
                GenericReduce::ReduceColumns<KeepFp32, 7>(a, worker, workers, col, timer, observer);
                break;
            case 8:
                GenericReduce::ReduceColumns<KeepFp32, 8>(a, worker, workers, col, timer, observer);
                break;
        }
    }
}
template <class T>
__aicore__ inline void ReduceGeneric(const Arguments& a, uint32_t worker, uint32_t workers, T& timer)
{
    NoReduceObserver observer;
    ReduceGeneric(a, worker, workers, timer, observer);
}
}  // namespace MegaMoeUrma
