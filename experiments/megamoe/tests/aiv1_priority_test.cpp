// Copyright (c) 2026, Lu Lu

// CPU stage doubles around the extracted production scheduler. Readiness changes
// during local Combine or polling; no scheduler is reimplemented.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 4;
namespace AscendTimer {
constexpr int64_t NO_WAVE = -1;
}
namespace AscendC {
std::function<int64_t()> cycle;
int64_t GetSystemCycle()
{
    return cycle();
}
}  // namespace AscendC

struct Fixture {
    struct Tiling { bool bf16Bs96 = false; } tiling;
    struct { void* inputScaleGmAddr; Tiling* tilingData; } params_{nullptr, &tiling};
    struct { uint32_t blockIdx = 0; } countWorkspace_;
    int hierarchicalCombine_ = 0;
    void ProgressGatewayHint() {}
    bool UseInputEarlyClosure(Tiling*, bool) const { return false; }
    bool UseAiv0CombineOwners(Tiling*) const { return true; }
    bool IsHierarchicalCombineOwner(Tiling*, uint32_t) const { return false; }
    int HierarchicalCombineLane(Tiling*, uint32_t) const { return 0; }
    template <class... T> int HierarchicalRoceScratch(T&&...) { return 0; }
    template <class... T> void PublishCoreClosure(T&&...) { assert(false); }
    template <class... T> bool TryProgressHierarchicalReducedCombine(T&&...) { assert(false); return false; }
    struct AivTaskCursor {
        uint32_t nextTile = 0;
        uint32_t tileCount = 0;
        uint32_t sliceIndex = 0;
        uint32_t nextSlice = 0;
        uint64_t wave = 0U;
    };
    struct SharedActivationCursor {
        bool done = false;
        uint32_t tile = 0;
    };
    struct ExpertTokenPosition {
        uint32_t expertIdx = 0;
    };
    struct Job {
        uint32_t jobIndex = 1;
        uint32_t totalJobs = 2;
    } aivJob_;
    struct {
        Job blockJob;
    } gmmExecutionConfig_;
    struct {
        uint32_t moeExpertPerRank = 0;
        uint32_t tokenNum = 32U;
    } commonConfig_;
    struct {
        bool Enabled() const { return false; }
        uint32_t clears = 0;
        int64_t finalWave = 999;
        void ClearSlice()
        {
            ++clears;
        }
        void SetWave(int64_t wave)
        {
            finalWave = wave;
        }
    } timer_;
    uint32_t sharedExpertNum_ = 0;
    uint32_t mGroupsPerWave_ = 2;
    uint32_t WaveGroups(uint64_t wave) const /* WAVE_GROUPS */

    // Each activation plan is {first physical tile, exclusive tile bound}.
    std::vector<std::pair<uint32_t, uint32_t>> activationPlans;
    std::set<std::pair<uint32_t, uint32_t>> readyTiles;
    std::set<std::pair<uint32_t, uint32_t>> consumedTiles;
    uint32_t combineSlices = 0;
    std::set<uint32_t> readyCombines;
    std::set<uint32_t> consumedCombines;
    bool sharedReady = false;
    uint32_t sharedTiles = 2;
    uint32_t consumedShared = 0;
    uint32_t activationAdvances = 0;
    uint32_t combineAdvances = 0;
    uint32_t operationCount = 0;
    uint32_t blockJobs = 2;
    int64_t clock = 0;
    std::vector<std::string> events;
    std::function<void(Fixture&, uint32_t)> afterCombine;
    std::function<void(Fixture&, uint32_t, uint32_t)> afterActivation;
    std::function<void(Fixture&)> afterDispatch;
    std::function<void(Fixture&)> onClock;

    void Guard()
    {
        assert(++operationCount < 1000);
    }
    void ReadyAllActivationTiles()
    {
        for (uint32_t slice = 0; slice < activationPlans.size(); ++slice) {
            for (uint32_t tile = activationPlans[slice].first; tile < activationPlans[slice].second;
                 tile += gmmExecutionConfig_.blockJob.totalJobs) {
                readyTiles.emplace(slice, tile);
            }
        }
    }
    void ReadyAllCombines()
    {
        for (uint32_t slice = 0; slice < combineSlices; ++slice) {
            readyCombines.insert(slice);
        }
    }
    bool NextAivSlice(AivTaskCursor& cursor, bool activation)
    {
        Guard();
        if (activation) {
            ++activationAdvances;
            if (cursor.nextSlice != 0) {
                assert(cursor.nextTile >= cursor.tileCount);
            }
            if (cursor.nextSlice == activationPlans.size()) {
                return false;
            }
            cursor.sliceIndex = cursor.nextSlice++;
            cursor.nextTile = activationPlans[cursor.sliceIndex].first;
            cursor.tileCount = activationPlans[cursor.sliceIndex].second;
        } else {
            ++combineAdvances;
            if (cursor.nextSlice != 0) {
                assert(consumedCombines.count(cursor.sliceIndex) == 1);
            }
            if (cursor.nextSlice == combineSlices) {
                return false;
            }
            cursor.sliceIndex = cursor.nextSlice++;
        }
        cursor.wave = cursor.sliceIndex / mGroupsPerWave_;
        return true;
    }
    struct Pipeline {
        bool enabled = true;
        uint32_t pending = 0U;
        std::vector<uint32_t> drains;
        template <typename Timer>
        void Drain(Timer&)
        {
            if (pending) drains.push_back(pending);
            pending = 0U;
        }
    } combinePipeline_;
    bool IsActivationReady(const AivTaskCursor& cursor, int32_t)
    {
        return readyTiles.count({cursor.sliceIndex, cursor.nextTile}) != 0U;
    }
    bool RunReadyActivation(const AivTaskCursor& cursor, int32_t& readySequence)
    {
        Guard();
        auto key = std::make_pair(cursor.sliceIndex, cursor.nextTile);
        assert(cursor.nextTile < cursor.tileCount);
        if (readyTiles.count(key) == 0) {
            return false;
        }
        assert(combinePipeline_.pending == 0U);
        assert(consumedTiles.insert(key).second);
        events.push_back("A" + std::to_string(key.first) + ":" + std::to_string(key.second));
        ++readySequence;
        if (afterActivation) {
            afterActivation(*this, key.first, key.second);
        }
        return true;
    }
    bool TryRunReadyCombine(const AivTaskCursor& cursor)
    {
        Guard();
        if (readyCombines.count(cursor.sliceIndex) == 0) {
            return false;
        }
        assert(consumedCombines.insert(cursor.sliceIndex).second);
        if (combinePipeline_.enabled) ++combinePipeline_.pending;
        events.push_back("C" + std::to_string(cursor.sliceIndex));
        if (afterCombine) {
            afterCombine(*this, cursor.sliceIndex);
        }
        return true;
    }
    void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& groups, uint64_t wave)
    {
        Guard();
        if (position.expertIdx >= commonConfig_.moeExpertPerRank) return;
        assert(combinePipeline_.pending == 0U);
        events.push_back("D" + std::to_string(position.expertIdx) + ":" + std::to_string(wave));
        ++position.expertIdx;
        ++groups;
        if (afterDispatch) {
            afterDispatch(*this);
        }
    }
    bool TryRunSharedExpertActivation(SharedActivationCursor& cursor, Pipeline* pipeline)
    {
        Guard();
        if (!sharedReady) {
            return false;
        }
        pipeline->Drain(timer_);
        assert(cursor.tile == consumedShared);
        events.push_back("S" + std::to_string(cursor.tile));
        ++consumedShared;
        cursor.done = ++cursor.tile == sharedTiles;
        return true;
    }

    void ProcessAiv1PriorityStages(int32_t& readySequence)
        /* PRODUCTION_SCHEDULER_BODY */

        void Run(int32_t expectedActivations)
    {
        AscendC::cycle = [this]() {
            Guard();
            ++clock;
            if (onClock) {
                onClock(*this);
            }
            return clock;
        };
        int32_t sequence = 40;
        ProcessAiv1PriorityStages(sequence);
        AscendC::cycle = {};
        assert(sequence == 40 + expectedActivations);
        assert(consumedTiles.size() == static_cast<std::size_t>(expectedActivations));
        assert(consumedCombines.size() == combineSlices);
        assert(activationAdvances == activationPlans.size() + 1);
        assert(combineAdvances == combineSlices + 1);
        assert(consumedShared == (sharedExpertNum_ == 0 ? 0 : sharedTiles));
        assert(timer_.clears == 1);
        assert(timer_.finalWave == AscendTimer::NO_WAVE);
        assert(gmmExecutionConfig_.blockJob.totalJobs == blockJobs);
    }
    void Expect(std::initializer_list<std::string> expected) const
    {
        assert(events == std::vector<std::string>(expected));
    }
};

void CombineReleasesMultipleTilesAndSlices()
{
    Fixture f;
    f.activationPlans = {{1, 6}, {0, 3}};
    f.combineSlices = 2;
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t slice) {
        if (slice == 0) {
            fixture.ReadyAllActivationTiles();
        }
    };
    f.Run(5);
    f.Expect({"C0", "A0:1", "A0:3", "A0:5", "A1:0", "A1:2", "C1"});
    assert(f.clock == 0);
}

void SharedActivationDrainsCombineBeforeRunning()
{
    Fixture f;
    f.sharedExpertNum_ = 1;
    f.combineSlices = 2;
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t) { fixture.sharedReady = true; };
    f.Run(0);
    f.Expect({"C0", "S0", "S1", "C1"});
}

void ExistingActivationDispatchSharedPriorityIsPreserved()
{
    Fixture f;
    f.activationPlans = {{1, 4}};
    f.ReadyAllActivationTiles();
    f.commonConfig_.moeExpertPerRank = 3;
    f.sharedExpertNum_ = 1;
    f.sharedReady = true;
    f.combineSlices = 1;
    f.ReadyAllCombines();
    f.Run(2);
    f.Expect({"D0:0", "A0:1", "A0:3", "D1:0", "D2:1", "S0", "S1", "C0"});
}

void BackoffCanReleaseActivationAndCombine()
{
    Fixture f;
    f.activationPlans = {{1, 2}, {0, 1}};
    f.combineSlices = 1;
    f.onClock = [](Fixture& fixture) {
        if (fixture.clock == 3) {
            fixture.readyTiles.emplace(0, 1);
            fixture.ReadyAllCombines();
        }
    };
    f.afterActivation = [](Fixture& fixture, uint32_t slice, uint32_t) {
        if (slice == 0) {
            fixture.readyTiles.emplace(1, 0);
        }
    };
    f.Run(2);
    f.Expect({"A0:1", "A1:0", "C0"});
    assert(f.clock >= GM_FLAG_POLL_BACKOFF_CYCLES);
}

void UnreadyCombineDoesNotStopDispatch()
{
    Fixture f;
    f.aivJob_.jobIndex = 3U;
    f.commonConfig_.moeExpertPerRank = 4;
    f.combineSlices = 2;
    uint32_t dispatches = 0;
    f.afterDispatch = [&dispatches](Fixture& fixture) {
        if (++dispatches == 4U) {
            fixture.ReadyAllCombines();
        }
    };
    f.Run(0);
    f.Expect({"D0:0", "D1:0", "D2:1", "D3:1", "C0", "C1"});
}

void EarlyCombineImmediatelyYieldsToNewActivation()
{
    Fixture f;
    f.aivJob_.jobIndex = 3U;
    f.commonConfig_.moeExpertPerRank = 4;
    f.combineSlices = 1;
    f.activationPlans = {{0, 1}};
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t) { fixture.ReadyAllActivationTiles(); };
    f.Run(1);
    f.Expect({"D0:0", "D1:0", "D2:1", "D3:1", "C0", "A0:0"});
}

void ConsecutiveRowsSurviveFailedProbesButYieldToActivation()
{
    Fixture f;
    f.combinePipeline_.enabled = true;
    f.aivJob_.jobIndex = 3U;
    f.combineSlices = 4U;
    f.activationPlans = {{0, 1}};
    f.ReadyAllCombines();
    // Also cover an activation becoming ready after wave 1 Combine starts.
    f.afterCombine = [](Fixture& fixture, uint32_t slice) {
        if (slice == 2U) fixture.ReadyAllActivationTiles();
    };
    f.Run(1);
    assert(f.combinePipeline_.drains == std::vector<uint32_t>({3U, 1U}));
    assert(f.combinePipeline_.pending == 0U);
    f.Expect({"C0", "C1", "C2", "A0:0", "C3"});
}

int main()
{
    ConsecutiveRowsSurviveFailedProbesButYieldToActivation();
    CombineReleasesMultipleTilesAndSlices();
    SharedActivationDrainsCombineBeforeRunning();
    ExistingActivationDispatchSharedPriorityIsPreserved();
    BackoffCanReleaseActivationAndCombine();
    UnreadyCombineDoesNotStopDispatch();
    EarlyCombineImmediatelyYieldsToNewActivation();
}
