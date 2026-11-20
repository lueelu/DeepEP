// Copyright (c) 2026, Lu Lu
// Actual ProcessWave output orchestration and host barrier.
// Stage doubles enforce prologue/UB lifetime and per-QP ownership; numerical
// readiness and asynchronous transport are covered by their production tests.
#include <array>
#include <cassert>
#include <cstdint>
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_peermem.hpp"

#define __gm__
#define __ubuf__
constexpr int AIC = 0, AIV = 1, g_coreType = AIV, PIPE_ALL = 0;

namespace AscendTimer {
enum Event { AIV_EXPORT_SYNC, AIV_FINAL_CROSS_RANK_WAIT, AIV_HOST_COMBINE_BARRIER_WAIT, AIV_UNPERMUTE };
enum class RdmaPhase { Combine };
}  // namespace AscendTimer

namespace MegaMoeImpl {
struct State {
    uint32_t physical = 0, preDrains = 0, payloads = 0, commits = 0, waits = 0, quiets = 0;
    uint32_t hostCqCalls = 0, unpermutes = 0, expectedRows = 0;
    bool routedDone = false, sharedDone = false, drained = false, hostDone = false, exported = false;
    bool timer = false, early = false, oddOwners = false, readersAcquired = false, producersAcquired = false;
} current;
alignas(512) std::array<uint8_t, 64U * 1024U> window{};
std::array<uint8_t, 512> scratch{};
GM_ADDR g_winRankAddr_[32]{};
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
};
struct MoeStageCommonConfig {
    uint32_t rankId = 0, topK = 8;
};
struct Params {
    MegaMoeTilingData* tilingData;
    struct {
        GM_ADDR rankSyncInWorldPtr = window.data();
        GM_ADDR combineReduceCommitPtr = window.data() + 4096;
    } peermemInfo;
    GM_ADDR inputScaleGmAddr = nullptr;
};
struct Progress {
    uint32_t sentRows = 0;
    bool initialized = false;
};
struct AscendTimerDevice {
    bool Enabled() const
    {
        return current.timer;
    }
    void Start(AscendTimer::Event event)
    {
        if (event == AscendTimer::AIV_EXPORT_SYNC) current.exported = true;
    }
    void End(AscendTimer::Event) {}
};
struct AscendTimerFixedScope {
    AscendTimerFixedScope(AscendTimerDevice*, AscendTimer::Event) {}
};
enum class Stage { SHARED_EXPERT_GMM2, CROSS_RANK_SYNC_OUTPUT, UNPERMUTE };
struct Dump {
    void UpdateStage(Stage) {}
};
struct MegaMoeUnpermuteBufferConfig {};
enum class RoceWaitKind { HostProducerBarrier };

bool ExpectedOwner()
{
    return current.physical % 2U == static_cast<uint32_t>(current.oddOwners) && current.physical / 2U < 16U;
}
void CheckOwner(uint32_t qp)
{
    assert(ExpectedOwner());
    assert(qp == current.physical / 2U + 1U);
    assert(current.routedDone && current.sharedDone);
}
template <int>
void PipeBarrier()
{
    current.drained = true;
}
template <bool>
void SyncAll()
{
    assert(current.routedDone && current.sharedDone);
    assert(current.exported);  // No extra all-core barrier before the AIV0 drain.
}
int32_t ReadGmByPassDCache(int32_t* address)
{
    return *address;
}
void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    *address = value;
}
uint8_t* HierarchicalRoceScratch(const Params&)
{
    return scratch.data();
}
int32_t* HierarchicalSignalAddress(const decltype(Params::peermemInfo)&, HierarchicalSignal signal)
{
    return reinterpret_cast<int32_t*>(window.data() + 8192 + static_cast<uint32_t>(signal) * 512U);
}
void RoceWaitFlagEquals(int32_t*, uint32_t, uint32_t, uint8_t* ub, RoceWaitKind)
{
    if (ub != nullptr) {
        CheckOwner(1U);
        assert(current.physical == static_cast<uint32_t>(current.oddOwners));
        ++current.hostCqCalls;
    }
}
void GmSignalWaitBarrier(int32_t*, int32_t) {}
void ProductionHostBarrier(const AivJobContext& job, const MoeStageCommonConfig& common, const Params& params)
{
    /* PRODUCTION_HOST_BARRIER */
}
void CrossRankSyncInHierarchicalHost(const AivJobContext& job, const MoeStageCommonConfig& common, const Params& params)
{
    ProductionHostBarrier(job, common, params);
    current.hostDone = true;
}

void DrainHierarchicalReducedCombine(const MoeStageCommonConfig&, const Params&, Progress& progress, uint32_t lane,
                                     uint8_t*, AscendTimerDevice*)
{
    CheckOwner(lane + 1U);
    assert(current.drained);
    if (!current.exported) {
        assert(!current.hostDone);
        ++current.preDrains;
    } else {
        assert(current.hostDone);
    }
    current.payloads += current.expectedRows - progress.sentRows;
    progress.sentRows = current.expectedRows;
}
void TraceHierarchicalCombineBoundary(const MoeStageCommonConfig&, const Params&, Progress&, uint32_t,
                                      AscendTimerDevice*, uint32_t)
{
}
void RocePublishCommit(GM_ADDR dst, GM_ADDR src, uint32_t, uint8_t*, AscendTimerDevice*, AscendTimer::RdmaPhase phase,
                       uint32_t qp = 1U)
{
    CheckOwner(qp);
    assert(phase == AscendTimer::RdmaPhase::Combine);
    assert(current.early ? current.readersAcquired && !current.exported : current.hostDone);
    assert(src == reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress({}, HierarchicalSignal::Sequence)));

    assert(dst == window.data() + 4096 + (current.physical / 2U) * 512U);
    assert(current.payloads == current.expectedRows);
    ++current.commits;
}
void RoceWaitCommit(int32_t* flag, uint32_t, AscendTimerDevice*, AscendTimer::RdmaPhase phase, uint32_t, uint8_t*,
                    uint32_t qp = 1U)
{
    CheckOwner(qp);
    assert(current.commits == 1U && (current.early ? current.readersAcquired : current.hostDone));
    assert(phase == AscendTimer::RdmaPhase::Combine);

    assert(reinterpret_cast<GM_ADDR>(flag) == window.data() + 4096 + (current.physical / 2U) * 512U);
    ++current.waits;
}
void RoceFinishOutstanding(uint32_t, uint8_t*, AscendTimerDevice*, AscendTimer::RdmaPhase, uint32_t qp)
{
    CheckOwner(qp);
    assert(current.waits == 1U);
    ++current.quiets;
}
void PublishHierarchicalLocalSignal(const Params&, HierarchicalSignal)
{
    assert(current.quiets == 1U);
}
template <typename... T>
void ExportCompactExpertTokenCounts(T&&...)
{
}
template <typename... T>
void CrossRankSyncInWorldSize(T&&...)
{
    assert(false);
}
template <int, typename, bool, uint32_t, typename... T>
void UnpermuteTokens(T&&...)
{
    assert(!ExpectedOwner());
    assert(!current.early || current.producersAcquired);
    ++current.unpermutes;
}

inline uint32_t GetSubBlockIdx() { return current.physical % 2; }
void PublishCoreClosure(const Params&, uint32_t block, bool producers)
{
    assert(current.early && producers && current.physical == 2U * block + 1U);
    assert(current.routedDone && current.sharedDone);
}
void CloseHost(const Params&, uint32_t, bool producers, int&)
{
    assert(current.early && current.physical == (producers ? 32U : 1U));
}
int32_t* ClosureSlot(const Params&, uint32_t slot)
{
    assert(slot == 96U || slot == 97U);
    return reinterpret_cast<int32_t*>(window.data()) + slot;
}
void WaitUntilGmFlagEquals(int32_t* flag, uint32_t seq)
{
    assert(current.early && seq == 259U);
    if (ExpectedOwner()) {
        assert(flag == reinterpret_cast<int32_t*>(window.data()) + 96 && current.payloads == current.expectedRows);
        current.readersAcquired = true;
    } else {
        assert(flag == reinterpret_cast<int32_t*>(window.data()) + 97);
        current.producersAcquired = true;
    }
}
template <class... T>
void AdvanceInputEpochLocally(T&&...)
{
    assert(current.early);
}

struct Kernel {
    uint32_t blockIdx_ = current.physical / 2;
    static constexpr bool ENABLE_A8W4 = true, TopkWeightsPrefetch = false;
    static constexpr uint32_t GMM1_TILE_M = 256;
    static constexpr int CombineQuantMode = COMBINE_NO_QUANT;
    using TopkWeightsType = float;
    MegaMoeTilingData tiling{};
    Params params_{&tiling, {}};
    AivJobContext aivJob_{current.physical, 56U};
    MoeStageCommonConfig commonConfig_{};
    int hostClosure_ = 0;
    Progress hierarchicalCombine_{};
    AscendTimerDevice timer_{};
    Dump exceptionDump_{};
    uint32_t sharedExpertNum_ = 0, rankId_ = 0, worldSize_ = 32;
    int countWorkspace_ = 0, tokenDispatchScratch_ = 0, tokenUnpermuteConfig_ = 0, tokenUnpermuteScratch_ = 0;

    void ProcessMoeExpertStages(int32_t&)
    {
        assert(current.preDrains == 0U);
        current.routedDone = true;
        current.sharedDone = sharedExpertNum_ == 0U;
        current.drained = true;
    }

    void ProcessMoeExpertStages()
    {
        assert(false);
    }
    void ProcessSharedExpertTail()
    {
        assert(current.routedDone && !current.sharedDone && current.preDrains == 0U);
        current.drained = false;  // Shared prologue still owns UB/events here.
        current.sharedDone = current.drained = true;
    }
    void ProcessSharedExpertGmm2()
    {
        assert(false);
    }
    MegaMoeUnpermuteBufferConfig InitTokenUnpermuteBuffers()
    {
        return {};
    }
    void Run()
    {
        auto& derived = *this;
        int32_t gmm1TileReadySequence = 0;
        /* PRODUCTION_STAGES */
    }
};
}  // namespace MegaMoeImpl

int main()
{
    using namespace MegaMoeImpl;
    for (auto& address : g_winRankAddr_) address = window.data();
    for (bool shared : {false, true})
        for (bool timer : {false, true})
            for (uint32_t rank : {0U, 30U})
                for (uint32_t rows : {0U, 16U})
                    for (uint32_t physical = 0; physical < 56U; ++physical) {
                        current = {};
                        current.physical = physical;
                        current.expectedRows = rows;
                        current.timer = timer;
                        Kernel kernel;
                        kernel.rankId_ = kernel.commonConfig_.rankId = rank;
                        kernel.tiling.topoType = TOPO_TYPE_MTE_ROCE;
                        kernel.tiling.blockAivNum = 56U;
                        kernel.sharedExpertNum_ = shared ? 1U : 0U;
                        kernel.Run();
                        const bool owner = ExpectedOwner();
                        assert(current.preDrains == static_cast<uint32_t>(owner));
                        if (owner) assert(current.payloads == rows);
                        assert(current.commits == static_cast<uint32_t>(owner));
                        assert(current.waits == static_cast<uint32_t>(owner));
                        assert(current.quiets == static_cast<uint32_t>(owner));
                        assert(current.unpermutes == static_cast<uint32_t>(!owner));
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
                        assert(current.hostCqCalls == static_cast<uint32_t>(physical == 0U));
#else
                        assert(current.hostCqCalls == 0U);
#endif
                    }
#ifdef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
    for (uint32_t bs : {32U, 96U, 128U}) {
        for (bool fp8Input : {false, true}) {
            for (uint32_t physical = 0; physical < 56; ++physical) {
                current = {};
                current.physical = physical;
                current.early = fp8Input;
                current.oddOwners = bs == 96;
                current.expectedRows = (physical / 2U) % 3U;  // Include empty QPs.
                Kernel kernel;
                auto& t = kernel.tiling;
                t.topoType = TOPO_TYPE_MTE_ROCE;
                t.epWorldSize = 32;
                t.dispatchRanksPerNode = 16;
                t.aicNum = 28;
                t.blockAivNum = 56;
                t.bs = bs;
                t.h = bs == 32 ? 3584 : 5120;
                t.hiddenDim = bs == 32 ? 2560 : 4608;
                t.topK = bs == 32 ? 8 : 6;
                t.moeExpertPerRank = bs == 32 ? 12 : 13;
                t.sharedFp8 = bs != 32;
                t.sharedExpertNum = t.sharedFp8 ? 1U : 0U;
                kernel.sharedExpertNum_ = t.sharedExpertNum;
                t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
                t.combineQuantMode = COMBINE_NO_QUANT;
                t.hierarchicalSequence = 259U;
                kernel.params_.inputScaleGmAddr = fp8Input ? window.data() : nullptr;
                kernel.Run();
                assert(current.preDrains == static_cast<uint32_t>(ExpectedOwner()));
                assert(current.commits == static_cast<uint32_t>(ExpectedOwner()));
                assert(current.quiets == current.commits && current.hostDone == !fp8Input);
                assert(current.hostCqCalls ==
                       static_cast<uint32_t>(!fp8Input && physical == static_cast<uint32_t>(current.oddOwners)));
            }
        }
    }
#endif
}
