// Copyright (c) 2026, Lu Lu
// Compile the production acquisition loop. DMA/clock/VF doubles supply actual
// old, future and ready epochs; the diagnostics may not change acquisition.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#include "ascend_timer_v2.hpp"

inline std::array<int32_t, 8192> ub;
inline std::vector<std::vector<int32_t>> samples;
inline std::vector<uint32_t> addresses;
inline uint32_t copies = 0, checks = 0, syncs = 0, writes = 0, captures = 0, scalarReads = 0;
inline int64_t cycle = 100;
namespace AscendC {
int64_t GetSystemCycle()
{
    return ++cycle;
}
enum class HardEvent { V_MTE2, MTE2_V, V_S, V_MTE3, MTE3_S };
}  // namespace AscendC
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 5;
enum class TPosition { VECCALC };
template <typename T>
struct LocalTensor {
    uint32_t addr = 0;
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t) : addr(offset) {}
    T GetValue(uint32_t i) const
    {
        ++cycle;
        ++scalarReads;
        return reinterpret_cast<T*>(ub.data())[addr / sizeof(T) + i];
    }
    void SetValue(uint32_t i, T value)
    {
        reinterpret_cast<T*>(ub.data())[addr / sizeof(T) + i] = value;
    }
    uintptr_t GetPhyAddr() const
    {
        return addr;
    }
};
// Production VF calls convert offsets to UB pointers. Resolve them in the VF
// double; offsets deliberately remain 32-bit like the device API.
template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* value)
    {
        data = value;
    }
};
struct CopyParams {
    uint32_t blocks, bytes, srcStride, dstStride, reserved;
};
struct PadParams {
    bool pad;
    uint32_t left, right, value;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T>, CopyParams params, PadParams)
{
    assert(copies < samples.size() && params.blocks == 1U);
    const auto& sample = samples[copies++];
    assert(params.bytes == sample.size() * sizeof(T));
    std::copy(sample.begin(), sample.end(), ub.begin() + dst.addr / sizeof(T));
    addresses.push_back(dst.addr);
    cycle += 50;
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, CopyParams params)
{
    ++captures;
    assert(params.blocks == 1U);
    std::copy_n(ub.data() + src.addr / sizeof(T), params.bytes / sizeof(T), dst.data);
    cycle += 12;
}
template <AscendC::HardEvent, uint32_t>
void SyncFuncStatic()
{
    ++syncs;
    cycle += 3;
}
void CheckTransposeExpertCountsVF(uint32_t* sourceOffset, uint32_t* countOffset, uint32_t* resultOffset,
                                  uint32_t sources, uint32_t experts, uint32_t stride, uint32_t epoch)
{
    auto* source = ub.data() + reinterpret_cast<uintptr_t>(sourceOffset) / 4U;
    auto* counts = ub.data() + reinterpret_cast<uintptr_t>(countOffset) / 4U;
    auto* result = ub.data() + reinterpret_cast<uintptr_t>(resultOffset) / 4U;
    uint32_t stale = 0;
    for (uint32_t s = 0; s < sources; ++s)
        for (uint32_t e = 0; e < experts; ++e) {
            const uint32_t value = static_cast<uint32_t>(source[s * stride + e]);
            stale += (value >> 24) != epoch;
            counts[e * sources + s] = static_cast<int32_t>(value & 0xffffffU);
        }
    result[0] = static_cast<int32_t>(stale);
    ++checks;
    cycle += 25;
}
template <auto Function, typename... Args>
void asc_vf_call(Args... args)
{
    Function(args...);
}
int32_t ReadGmByPassDCache(int32_t* p)
{
    return *p;
}
struct MoeStageCommonConfig {
    uint32_t worldSize = 32, moeExpertPerRank = 12;
};
struct Buffers {
    uint32_t routeItemsPerBatch = 512, bufferCount = 1, copyBufferBytes = 2048;
};
struct Tiling {
    Buffers dispatchBufferConfig;
};
struct Params {
    Tiling* tilingData;
    struct {
        uint8_t* expertCountRecvPtr;
    } peermemInfo;
};
template <typename T>
struct TokenDispatchScratch {
    LocalTensor<int32_t> validTopkIndexTensor{TPosition::VECCALC, 0, 512};
    LocalTensor<int32_t> cumsumInfoTensor{TPosition::VECCALC, 16384, 384};
    LocalTensor<int32_t> expertTokenNumsOutTensor{TPosition::VECCALC, 8192, 16};
};
uint32_t CalcMteExpertCountRankStride(uint32_t n)
{
    return (n + 7U) / 8U * 8U;
}
struct AscendTimerDevice {
    bool enabled = true;
    std::array<uint64_t, AscendTimer::COUNT_DIAGNOSTIC_COUNT> values{};
    std::array<int32_t, AscendTimer::COUNT_SNAPSHOT_MAX_WORDS> snapshot{};
    bool Enabled() const
    {
        return enabled;
    }
    void WriteCountDiagnostics(const uint64_t* data)
    {
        ++writes;
        std::copy_n(data, values.size(), values.begin());
    }
    uint8_t* CountSnapshotAddress()
    {
        return reinterpret_cast<uint8_t*>(snapshot.data());
    }
};

/* PRODUCTION_COUNT_SNAPSHOT */

void Run(bool doubleBuffer, bool failed, bool enabled)
{
    using namespace AscendTimer;
    copies = checks = syncs = writes = captures = scalarReads = 0;
    addresses.clear();
    ub.fill(-1);
    MoeStageCommonConfig common;
    std::vector<int32_t> ready(512, 0);  // Padding is stale but must be ignored.
    for (uint32_t s = 0; s < 32; ++s)
        for (uint32_t e = 0; e < 12; ++e) ready[s * 16 + e] = static_cast<int32_t>(0x81000000U | (s + e + 1U));
    samples.clear();
    if (failed) {
        samples.push_back(std::vector<int32_t>(512, 0));
        auto partial = ready;
        partial[7 * 16 + 5] = static_cast<int32_t>(0x80000006U);
        partial[23 * 16 + 11] = static_cast<int32_t>(0x82000009U);
        samples.push_back(partial);
    }
    samples.push_back(ready);
    Tiling tiling;
    tiling.dispatchBufferConfig.copyBufferBytes = doubleBuffer ? 2048U : 0U;
    Params params{&tiling, {reinterpret_cast<uint8_t*>(ready.data())}};
    TokenDispatchScratch<int32_t> scratch;
    AscendTimerDevice timer;
    timer.enabled = enabled;
    LoadMteExpertCounts(common, params, scratch, 0x81U, &timer);
    assert(copies == samples.size() && checks == copies && syncs == copies * 3U + captures * 2U);
    assert(scalarReads == copies);  // Only the VF status, never a scalar scan of the snapshot.
    for (uint32_t s = 0; s < 32; ++s)
        for (uint32_t e = 0; e < 12; ++e)
            assert(scratch.cumsumInfoTensor.GetValue(e * 32 + s) == static_cast<int32_t>(s + e + 1U));
#ifdef DEEPEP_MEGAMOE_TIMER
    if (enabled) {
        assert(writes == 1 && timer.values[COUNT_SNAPSHOT_ATTEMPTS] == samples.size());
        assert(timer.values[COUNT_SNAPSHOT_FAILURES] == (failed ? 2U : 0U));
        assert(timer.values[COUNT_DMA_CYCLES] >= copies * 50U && timer.values[COUNT_CHECK_CYCLES] >= copies * 25U);
        assert(timer.values[COUNT_SUCCESS_DMA_CYCLES] >= 50U && timer.values[COUNT_SUCCESS_CHECK_CYCLES] >= 25U);
        assert(timer.values[COUNT_EXPECTED_EPOCH] == 0x81U && timer.values[COUNT_SNAPSHOT_BYTES] == 2048U);
        if (failed) {
            assert(timer.values[COUNT_LAST_STALE_WORDS] == 2U);
            assert(captures == (doubleBuffer ? 1U : 0U));
            assert(timer.values[COUNT_CAPTURED_SNAPSHOT_WORDS] == (doubleBuffer ? 512U : 0U));
            if (doubleBuffer) {
                assert(std::equal(samples[1].begin(), samples[1].end(), timer.snapshot.begin()));
                assert(timer.values[COUNT_DIAGNOSTIC_CYCLES] > 0U);
            }
            assert(timer.values[COUNT_BACKOFF_CYCLES] >= 10U);
            assert(addresses[1] == (doubleBuffer ? 2048U : 0U));
        } else {
            assert(timer.values[COUNT_LAST_STALE_SOURCE_MASK] == 0U && timer.values[COUNT_DIAGNOSTIC_CYCLES] == 0U);
        }
    } else
#endif
    {
        assert(writes == 0U && captures == 0U);
        assert(std::all_of(addresses.begin(), addresses.end(), [](uint32_t p) { return p == 0U; }));
    }
}
int main()
{
    for (bool buffers : {false, true})
        for (bool failed : {false, true})
            for (bool enabled : {false, true}) Run(buffers, failed, enabled);
}
