// Copyright (c) 2026, Lu Lu

// Actual task, row dispatch and ready publication, with deferred data copies.
// The planner/device doubles supply routing and deliberately clobber UB at each yield.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#define __gm__
constexpr int AIC = 0, AIV = 1, g_coreType = AIV;
constexpr uint32_t COMBINE_NO_QUANT = 0, CombineQuantMode = 0;
constexpr uint32_t GMM1_TILE_M = 256, W4A8_SMALL_BATCH_MAX_TOKENS = 256;
constexpr int64_t INT_CACHELINE = 16;
constexpr int SYNC_EVENT_ID5 = 5;
constexpr bool TopkWeightsPrefetch = true;
using ActivationType = int;
using QuantScaleOutType = int;
using GM_ADDR = uintptr_t;
uint32_t GetSubBlockIdx()
{
    return 1U;
}
namespace AscendTimer {
constexpr int AIV_DISPATCH = 1;
}
struct WorkRange {
    uint32_t start = 0, count = 0;
};
struct Job {
    uint32_t jobIndex = 0, totalJobs = 28;
};
struct ExpertTokenPosition {
    uint32_t expertIdx = 0, tokenIndexInExpert = 0;
    uint64_t globalTokenIndex = 0;
};
struct ExpertTokenRange {
    ExpertTokenPosition begin{}, end{};
};
struct ExpertDispatchCoreRange {
    uint32_t expertGlobalRowBegin, localExpertRowBegin, localExpertRowEnd;
};
struct TokenDispatchConfig {
    uint32_t maxOutputSize = 4096;
};
struct MoeStageCommonConfig {
    uint32_t moeExpertPerRank = 0, worldSize = 4, tokenNum = 32;
};
struct MoeSyncWorkspaceLayout {
    uint64_t dispatchFlagSlotCountPerExpert = 4 * INT_CACHELINE;
};
struct Tiling {
    bool hierarchical = true;
    uint32_t bs = 32, h = 3584, hiddenDim = 2560, moeExpertPerRank = 12, topK = 8, sharedExpertNum = 0;
    uint32_t epWorldSize = 32, combineQuantMode = 0;
};
constexpr uint32_t HIERARCHICAL_DISPATCH_WORKERS = 14U;
bool UseHierarchicalRoce(const Tiling* t)
{
    return t->hierarchical;
}
struct Params {
    Tiling* tilingData;
    struct {
        const std::vector<uint32_t>* expertRevTokenNumsPtr;
        uintptr_t flagDispatchToGmm1Ptr;
    } workspaceInfo;
};
struct Prefix {
    bool valid = false;
    std::vector<uint32_t> values;
    uint32_t GetValue(uint32_t index) const
    {
        assert(valid);
        return values.at(index);
    }
};
template <typename T>
struct TokenDispatchScratch {
    Prefix cumsumInfoTensor;
    uint32_t reloads = 0, worker = 0;
    std::vector<uint32_t> pending;
};
struct DispatchRouteBatch {};
struct DispatchCopyProgress {
    uint32_t deferredCount = 0;
};
template <class, class, bool, class... Args>
void ProbeDeferredDispatch(Args&&...)
{
    assert(false);
}
std::vector<uint32_t> data, owners;
std::vector<int32_t> readyCounts;
bool drained = true;
namespace AscendC {
enum class HardEvent { MTE3_S };
}
template <AscendC::HardEvent, int>
void SyncFuncStatic()
{
    assert(drained);
}
void AtomicAdd(int32_t* address, int32_t amount)
{
    assert(drained);
    *address += amount;
}
WorkRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount)
{ /* BALANCED_RANGE */
}
WorkRange GetRotatedBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount,
                                       uint64_t globalRowPrefix)
{ /* ROTATED_RANGE */
}
uint32_t GetDispatchSubsetJobIndex(uint32_t workerIdx, uint32_t workerCount, uint32_t selectedWorkers,
                                   uint32_t expertIdx)
{ /* DISPATCH_SUBSET */
}

// Controlled ranges include empty experts, multiple waves and M256 boundary tails.
template <uint32_t M>
ExpertTokenRange PlanNextExpertTokenRangeInWave(const std::vector<uint32_t>* counts, int, uint32_t experts,
                                                uint32_t budget, uint32_t& groups, ExpertTokenPosition position)
{
    assert(position.expertIdx < experts && groups < budget);
    auto end = position;
    uint32_t remaining = counts->at(position.expertIdx) - position.tokenIndexInExpert;
    uint32_t rows = std::min(remaining, (budget - groups) * M);
    end.globalTokenIndex += rows;
    end.tokenIndexInExpert += rows;
    groups += (rows + M - 1) / M;
    if (end.tokenIndexInExpert == counts->at(position.expertIdx)) {
        ++end.expertIdx;
        end.tokenIndexInExpert = 0;
    }
    return {position, end};
}
template <typename T>
void ReloadDispatchCumsumRange(const MoeStageCommonConfig&, TokenDispatchScratch<T>& scratch, uint32_t, uint32_t)
{
    scratch.cumsumInfoTensor.valid = true;
    ++scratch.reloads;
}
template <typename T>
uint32_t FindDispatchSourceRank(const MoeStageCommonConfig& common, const TokenDispatchScratch<T>& scratch,
                                uint32_t expert, uint32_t firstRow)
{
    uint32_t rank = 0;
    while (rank < common.worldSize && scratch.cumsumInfoTensor.GetValue(expert * common.worldSize + rank) <= firstRow) {
        ++rank;
    }
    return rank;
}
template <typename A, typename Q, bool Prefetch>
int32_t DispatchRankTokens(const TokenDispatchConfig&, const MoeStageCommonConfig&, const Params&, GM_ADDR*,
                           TokenDispatchScratch<A>& scratch, uint32_t, int32_t base, uint32_t, int32_t localBegin,
                           int32_t sourceBegin, int32_t sourceEnd, DispatchRouteBatch&, DispatchCopyProgress&)
{
    assert(sourceEnd > sourceBegin);
    for (int32_t i = 0; i < sourceEnd - sourceBegin; ++i) {
        scratch.pending.push_back(base + localBegin + i);
    }
    drained = false;
    // Multiple source ranks contribute; no bytes become visible before Drain.
    return sourceEnd - sourceBegin;
}
template <typename A, typename Q, bool P>
void FinishDispatchSources(const TokenDispatchConfig&, const MoeStageCommonConfig&, const Params&, GM_ADDR*,
                           TokenDispatchScratch<A>& scratch, uint32_t, DispatchRouteBatch&, DispatchCopyProgress&)
{
    for (auto row : scratch.pending) {
        assert(data.at(row) == 0);
        data[row] = row + 1;
        owners[row] = scratch.worker;
    }
    scratch.pending.clear();
    drained = true;
}
void PublishGmm1TileReady(const MoeSyncWorkspaceLayout& syncLayout, const Params& params, uint32_t localExpertId,
                          int32_t gmm1TileRowCount, int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{ /* PUBLISH_READY */
}
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
void DispatchOwnedExpertRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                             const MoeSyncWorkspaceLayout& syncLayout, const Params& params, GM_ADDR* winRankAddr,
                             TokenDispatchScratch<ActivationType>& scratch, uint32_t localExpertId,
                             const ExpertDispatchCoreRange& coreRange)
{ /* DISPATCH_OWNED */
}
template <typename A, typename Q, uint32_t M, bool P>
void DispatchPeerBalancedRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                              const MoeSyncWorkspaceLayout& layout, const Params& params, GM_ADDR* addresses,
                              TokenDispatchScratch<A>& scratch, uint32_t expert, const ExpertDispatchCoreRange& range,
                              uint32_t, uint32_t)
{
    DispatchOwnedExpertRows<A, Q, M, P>(context, common, layout, params, addresses, scratch, expert, range);
}
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch,
          bool PeerBalanced = false>
void DispatchTokenRange(const TokenDispatchConfig& context, const MoeStageCommonConfig& common, const Job& blockJob,
                        const MoeSyncWorkspaceLayout& syncLayout, const Params& params, GM_ADDR* winRankAddr,
                        TokenDispatchScratch<ActivationType>& scratch, const ExpertTokenRange& range)
{ /* DISPATCH_RANGE */
}
struct Fixture {
    void ProgressGatewayHint() {}
    MoeStageCommonConfig commonConfig_;
    MoeSyncWorkspaceLayout syncWorkspaceLayout_;
    TokenDispatchConfig tokenDispatchConfig_;
    Tiling tiling;
    Params params_{&tiling, {nullptr, 0}};
    struct {
        Job blockJob;
    } gmmExecutionConfig_;
    int countWorkspace_ = 0;
    uint32_t mGroupsPerWave_ = 2;
    TokenDispatchScratch<int> tokenDispatchScratch_;
    GM_ADDR g_winRankAddr_[4]{};
    struct {
        bool active = false;
        void SetWave(int64_t) {}
        void StartSpan(int)
        {
            assert(!active);
            active = true;
        }
        void EndSpan(int)
        {
            assert(active);
            active = false;
        }
    } timer_;
    void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& waveMGroupCount, uint64_t waveId)
    { /* DISPATCH_STEP */
    }
    ExpertTokenPosition position;
    uint32_t groups = 0, calls = 0;
    uint64_t wave = 0;
};

struct Result {
    std::vector<uint32_t> ownership;
    uint32_t reloads;
};
Result Run(bool hierarchical, uint32_t bs, uint32_t capacity)
{
    const std::vector<uint32_t> counts{0, 22, 1, 0, 35, 129, 257, 0, 3};
    std::vector<uint32_t> prefix;
    uint32_t total = 0;
    for (auto count : counts) {
        // Includes zero peer spans, so chunk resumption crosses real source ranges.
        for (uint32_t rank = 0; rank < 4; ++rank) {
            prefix.push_back(total + count * rank / 3);
        }
        total += count;
    }
    data.assign(total, 0);
    owners.assign(total, std::numeric_limits<uint32_t>::max());
    readyCounts.assign(counts.size() * 4 * INT_CACHELINE, 0);
    std::vector<std::unique_ptr<Fixture>> fixtures;
    for (uint32_t worker = 0; worker < 28; ++worker) {
        auto f = std::make_unique<Fixture>();
        f->commonConfig_ = {static_cast<uint32_t>(counts.size()), 4, bs};
        f->tiling = {hierarchical};
        f->tiling.bs = bs;
        f->params_.workspaceInfo = {&counts, reinterpret_cast<uintptr_t>(readyCounts.data())};
        f->gmmExecutionConfig_.blockJob = {worker, 28};
        f->tokenDispatchConfig_.maxOutputSize = capacity;
        f->tokenDispatchScratch_.worker = worker;
        f->tokenDispatchScratch_.cumsumInfoTensor.values = prefix;
        fixtures.push_back(std::move(f));
    }
    bool more = true;
    uint32_t rounds = 0;
    while (more) {
        assert(++rounds < 1000);
        more = false;
        for (auto& ptr : fixtures) {
            auto& f = *ptr;
            if (f.position.expertIdx == counts.size()) {
                continue;
            }
            more = true;
            if (f.groups >= f.mGroupsPerWave_) {
                f.groups = 0;
                ++f.wave;
            }
            f.DispatchNextExpertSlice(f.position, f.groups, f.wave);
            assert(drained && !f.timer_.active);

            // Activation/Combine can overwrite prefix and all transient stage UB.
            f.tokenDispatchScratch_.cumsumInfoTensor.valid = false;
            ++f.calls;
        }
    }
    uint32_t rowBase = 0;
    for (uint32_t expert = 0; expert < counts.size(); ++expert) {
        for (uint32_t m = 0; m < 4; ++m) {
            uint32_t begin = rowBase + m * 256;
            uint32_t end = rowBase + std::min((m + 1) * 256, counts[expert]);
            uint32_t expected = begin < std::min(end, capacity) ? std::min(end, capacity) - begin : 0;
            assert(readyCounts[expert * 4 * INT_CACHELINE + m * INT_CACHELINE] == static_cast<int32_t>(expected));
        }
        rowBase += counts[expert];
    }
    for (uint32_t row = 0; row < total; ++row) {
        assert(data[row] == (row < capacity ? row + 1 : 0));
    }
    uint32_t reloads = 0;
    for (const auto& f : fixtures) {
        reloads += f->tokenDispatchScratch_.reloads;
    }
    return {owners, reloads};
}
int main()
{
    for (bool hierarchical : {false, true})
        for (uint32_t bs : {1U, 32U, 256U})
            for (uint32_t capacity : {0U, 1U, 24U, 300U, 4096U}) Run(hierarchical, bs, capacity);
}
