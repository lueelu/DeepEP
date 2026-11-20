// Copyright (c) 2026, Lu Lu

// The production function is inserted by test_count_startup.py. Device copies
// are deferred until MTE3_S; the gateway double observes what AIC/AIV0 can read
// while payload is still unavailable. This does not emulate NPU cache coherence.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
constexpr int AIC = 0;
#ifdef COUNT_TEST_AIC
constexpr int g_coreType = AIC;
#else
constexpr int g_coreType = 1;
#endif
constexpr uint32_t INT_CACHELINE = 128U;
constexpr uint32_t RANK_SYNC_COUNTER_OFFSET_BYTES = 0U;
constexpr uint32_t SYNC_EVENT_ID2 = 2U;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 1;
using GM_ADDR = uint8_t*;

struct Tensor {
    int32_t* data = nullptr;
    bool global = false;
    int32_t GetValue(uint32_t i) const
    {
        return data[i];
    }
    void SetValue(uint32_t i, int32_t value)
    {
        data[i] = value;
    }
    uintptr_t GetPhyAddr() const
    {
        return reinterpret_cast<uintptr_t>(data);
    }
    void SetGlobalBuffer(int32_t* pointer)
    {
        data = pointer;
        global = true;
    }
    Tensor operator[](uint64_t i) const
    {
        return {data + i, global};
    }
};
template <typename>
using GlobalTensor = Tensor;
struct CopyParams {
    uint32_t blocks, bytes, srcStride, dstStride, reserved;
};
struct PadParams {
    bool enabled;
    uint32_t left, right, value;
};
inline std::vector<std::function<void()>> pending;
inline bool published = false, acquired = false, publishTimerActive = false, reloadExpected = false;
inline uint32_t lane = 1U, maskBuilds = 0U;
inline std::array<int32_t, 64> raw{}, prefixGm{};
inline std::array<int32_t, 4> totalsGm{};
inline std::array<int32_t, 2U * INT_CACHELINE> flags{};
inline GM_ADDR g_winRankAddr_[32]{};

namespace AscendC {
enum class HardEvent { MTE2_V, V_S, S_V, V_MTE3, MTE3_S };
inline int32_t ReadGmByPassDCache(const int32_t* pointer)
{
    return *pointer;
}
inline int64_t GetSystemCycle()
{
    static int64_t tick = 0;
    return ++tick;
}
}  // namespace AscendC
using AscendC::ReadGmByPassDCache;
template <AscendC::HardEvent Event, uint32_t>
void SyncFuncStatic()
{
    if constexpr (Event == AscendC::HardEvent::MTE3_S) {
        for (auto& copy : pending) copy();
        pending.clear();
    }
}
inline void DataCopyPad(Tensor dst, Tensor src, CopyParams params, PadParams = {})
{
    auto copy = [=] { std::copy_n(src.data, params.bytes / sizeof(int32_t), dst.data); };
    if (dst.global)
        pending.push_back(copy);
    else
        copy();
}
inline void WriteGmByPassDCache(int32_t* pointer, int32_t value)
{
    assert(pointer == flags.data() + INT_CACHELINE && value == 1);
    assert(pending.empty() && totalsGm[2] == 5 && totalsGm[3] == 4);
    assert(!reloadExpected || prefixGm[0] == 999);
    assert(publishTimerActive && !acquired);
    *pointer = value;
    published = true;
}
inline uint32_t GetSubBlockIdx()
{
    return lane;
}
struct MoeStageCommonConfig {
    uint32_t worldSize = 32U, moeExpertPerRank = 2U;
};
struct BlockWorkspaceContext {
    uint32_t blockIdx = 1U, blockNum = 2U;
    bool shared = false;
};
inline void WaitForMoeExpertTokenCountReady(GM_ADDR, const BlockWorkspaceContext&, uint32_t) { assert(false); }
struct Tiling {
    bool hierarchical = true, decode = true;
    uint32_t maxOutputSize = 32U;
};
struct Params {
    Tiling* tilingData;
    struct {
        GM_ADDR rankSyncInWorldPtr, expertCountRecvPtr;
    } peermemInfo;
    struct {
        GM_ADDR flagSendCntCalToUpdParamsPtr;
    } workspaceInfo;
};
template <typename>
struct TokenDispatchScratch {
    int peerRowLimitsCache = 7;
    Tensor cumsumInfoTensor, expertTokenNumsOutTensor, expertRevNumsGlobalTensor, cumsumInfoGlobalTensor;
};
template <class... T> void LoadGatheredRouteCounts(T&&...) { assert(false); }
inline bool UseHierarchicalRoce(const Tiling* t)
{
    return t->hierarchical;
}
inline bool UseDecodeRailRouting(const Tiling* t)
{
    return t->decode;
}
inline uint64_t GetExpertCountWorkspaceOffset(const BlockWorkspaceContext& w, uint32_t experts, uint32_t, bool)
{
    return w.blockIdx * experts;
}
class AscendTimerDevice;
template <typename T>
void LoadMteExpertCounts(const MoeStageCommonConfig&, const Params&, TokenDispatchScratch<T>& s, uint32_t epoch,
                         AscendTimerDevice*)
{
    assert(epoch == 0x81U);
    std::copy(raw.begin(), raw.end(), s.cumsumInfoTensor.data);
}
inline void CheckExpertCountEpochVF(uint32_t* counts, uint32_t* output, uint32_t n, uint32_t epoch)
{
    for (uint32_t i = 0; i < n; ++i) {
        assert((counts[i] >> 24) == epoch);
        counts[i] &= 0xffffffU;
    }
    output[0] = 0U;
}
template <auto Function, typename... Args>
void asc_vf_call(Args... args)
{
    Function(args...);
}
inline void ComputeExpertCountTables(Tensor counts, Tensor totals, uint32_t experts, uint32_t world, uint32_t)
{
    for (uint32_t e = 0; e < experts; ++e) {
        int32_t sum = 0;
        for (uint32_t s = 0; s < world; ++s) sum += counts.GetValue(e * world + s);
        totals.SetValue(e, sum);
    }
    // Model the destructive prefix transform before the local readiness acquire.
    std::fill_n(counts.data, experts * world, 999);
}
class AscendTimerDevice {};
namespace AscendTimer {
enum Event {
    AIV_COUNT_TABLE_PREPARE,
    AIV_COUNT_METADATA_WAIT,
    AIV_COUNT_PREFIX_BUILD,
    AIV_COUNT_TABLE_PUBLISH,
    AIV_GATEWAY_PAYLOAD_CHECK
};
}
struct AscendTimerFixedScope {
    AscendTimer::Event event;
    AscendTimerFixedScope(AscendTimerDevice*, AscendTimer::Event e) : event(e)
    {
        if (event == AscendTimer::AIV_COUNT_TABLE_PUBLISH) publishTimerActive = true;
    }
    ~AscendTimerFixedScope()
    {
        if (event == AscendTimer::AIV_COUNT_TABLE_PUBLISH) publishTimerActive = false;
    }
};
inline void AcquireHierarchicalGatewayPayload(const BlockWorkspaceContext&, const Params&)
{
    assert(maskBuilds == 0U);
    assert(published && flags[INT_CACHELINE] == 1 && pending.empty());
    assert(totalsGm[2] == 5 && totalsGm[3] == 4 && !publishTimerActive);
    assert(!reloadExpected || prefixGm[0] == 999);
    acquired = true;
}

/* PRODUCTION_COUNT_PREPARATION */

template <bool Reload>
void Run(bool decode, bool hierarchical, uint32_t subblock)
{
    pending.clear();
    published = acquired = publishTimerActive = false;
    maskBuilds = 0U;
    reloadExpected = Reload;
    lane = subblock;
    raw.fill(0);
    raw[0] = 3;
    raw[16] = 2;
    raw[33] = 4;
    prefixGm.fill(-1);
    totalsGm.fill(-1);
    flags.fill(0);
    std::array<int32_t, 64> wire = raw, ub{};
    for (auto& count : wire) count |= static_cast<int32_t>(0x81000000U);
    std::array<int32_t, 2> totals{};
    int32_t epoch = 1;
    Tiling tiling{hierarchical, decode, 32U};
    Params params{&tiling,
                  {reinterpret_cast<GM_ADDR>(&epoch), reinterpret_cast<GM_ADDR>(wire.data())},
                  {reinterpret_cast<GM_ADDR>(flags.data())}};
    TokenDispatchScratch<int> scratch{
        7, {ub.data()}, {totals.data()}, {totalsGm.data(), true}, {prefixGm.data(), true}};
    PrepareMoeExpertTokenCountTable<Reload>({}, {}, params, scratch);
    if (g_coreType == AIC || lane == 0U) {
        assert(!published && !acquired && maskBuilds == 0U && scratch.peerRowLimitsCache == 7);
        return;
    }
    assert(published && !acquired && scratch.peerRowLimitsCache == 0);
    assert(maskBuilds == 0U);
    assert(totalsGm[0] == -1 && totalsGm[1] == -1 && flags[0] == 0);
    assert(Reload || prefixGm[0] == -1);
}
int main()
{
    for (uint32_t subblock : {0U, 1U}) {
        for (bool reload : {false, true}) {
            for (bool decode : {false, true}) {
                for (bool hierarchical : {false, true}) {
                    if (hierarchical && !decode) continue;
                    if (reload)
                        Run<true>(decode, hierarchical, subblock);
                    else
                        Run<false>(decode, hierarchical, subblock);
                }
            }
        }
    }
}
