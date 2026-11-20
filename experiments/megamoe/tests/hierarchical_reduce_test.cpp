// Copyright (c) 2026, Lu Lu
// Execute the production layout, readiness scheduler, FP32 reduction, and final
// unpermute against CPU device primitives. NIC delivery is explicitly deferred.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*
#define MEGA_MOE_ROCE_H
#define MEGA_MOE_UTILS_H
#define MEGA_MOE_MXFP8_UTILS_H
constexpr int AIC = 0, AIV = 1, g_coreType = AIV;
constexpr int PIPE_V = 1;
constexpr uint32_t EVENT_ID0 = 0;

struct bfloat16_t {
    uint16_t bits = 0;
    bfloat16_t() = default;
    bfloat16_t(float value)
    {
        uint32_t word;
        std::memcpy(&word, &value, 4);
        word += 0x7fffU + ((word >> 16) & 1U);
        bits = static_cast<uint16_t>(word >> 16);
    }
    operator float() const
    {
        uint32_t word = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &word, 4);
        return value;
    }
};
struct fp8_e4m3fn_t {};
struct fp8_e5m2_t {};
namespace Std {
template <class A, class B>
using IsSame = std::is_same<A, B>;
}
namespace ReduceTest {
inline void Check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
alignas(512) inline std::array<uint8_t, 256U * 1024U> ub{};
inline std::array<std::array<int, 8>, 16> events{};
inline uint64_t cycle = 0;
inline uint32_t activeLane = 0;
inline uint32_t expectedPeer = 0;
inline uint8_t* localBase = nullptr;
inline uint8_t* remoteBase = nullptr;
inline size_t heapBytes = 0;
inline bool recordRouteReads = false;
inline std::vector<std::pair<const int32_t*, uint32_t>> routeReads;
struct InputLoad {
    bfloat16_t* destination;
    std::vector<bfloat16_t> values;
};
inline bool deferInputs = false;
inline std::vector<InputLoad> queuedInputs;
inline std::array<std::vector<InputLoad>, 8> inputEvents;
inline std::array<bfloat16_t*, 8> releaseInputs{};
inline std::map<bfloat16_t*, bool> busyInputs;
inline bfloat16_t* lastCastInput = nullptr;
inline uint32_t overlappedCasts = 0;
inline uint32_t scalarReads = 0, scalarFloatReads = 0, flagDmas = 0, readyVfs = 0, rowVfs = 0;
inline const uint32_t* readyResultAddress = nullptr;
inline bool readyResultPending = false;
struct FlagLoad {
    uint8_t* destination;
    std::vector<uint8_t> bytes;
};
inline std::vector<FlagLoad> queuedFlags;
inline std::array<std::vector<FlagLoad>, 8> flagEvents;
inline bool checkSnapshotReads = false;
inline uint32_t snapshotMask = 0, snapshotPackets = 0;
inline const uint8_t* snapshotBase = nullptr;
inline const uint8_t* snapshotResult = nullptr;
inline const uint8_t* flagSourceBegin = nullptr;
inline size_t flagSourceBytes = 0;
inline uint32_t flagSourceMask = 0, flagSourcePackets = 0;
inline bool checkRowReads = false;
inline size_t mainUbLimit = ub.size() - 512U;
inline const uint8_t *rowBegin = nullptr, *sumBegin = nullptr, *weightBegin = nullptr;
inline uint32_t rowReadBytes = 0, sumReadBytes = 0;
inline void CheckUbAccess(const void* address, size_t bytes, bool write)
{
    const auto* ptr = static_cast<const uint8_t*>(address);
    Check(ptr >= ub.data() && ptr + bytes <= ub.data() + mainUbLimit, "VF escaped declared main UB budget");
    if (checkSnapshotReads && write)
        Check(ptr == snapshotResult && bytes == 4U, "ready VF wrote outside its result word");
    if (checkSnapshotReads && !write) {
        const size_t offset = ptr - snapshotBase;
        Check(bytes == 4U && offset % 32U == 28U && offset / (snapshotPackets * 32U) < 32U &&
                  (snapshotMask & (uint32_t{1} << (offset / (snapshotPackets * 32U)))) != 0U,
              "VF read an inactive row or a non-marker word");
    }
    if (checkRowReads) {
        const bool input = ptr >= rowBegin && ptr + bytes <= rowBegin + rowReadBytes;
        const bool sum = ptr >= sumBegin && ptr + bytes <= sumBegin + sumReadBytes;
        const bool weight = ptr == weightBegin && bytes == sizeof(float);
        Check(write ? sum : input || sum || weight, "row VF read/write escaped its padded slots");
    }
}
struct Post {
    uint32_t qp;
    uint8_t* source;
    uint64_t destination;
    std::vector<uint8_t> bytes;
};
inline std::vector<Post> posts;
inline void Deliver()
{
    for (auto& post : posts) {
        Check(std::equal(post.bytes.begin(), post.bytes.end(), post.source), "source reused before completion");
        std::copy(post.bytes.begin(), post.bytes.end(), remoteBase + post.destination);
    }
}
}  // namespace ReduceTest
namespace AscendC {
using TEventID = uint32_t;
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT, UNKNOWN };
enum class HardEvent { S_MTE2, MTE2_S, MTE2_V, V_MTE2, S_V, V_S, V_MTE3, MTE3_S, MTE3_V, MTE2_MTE3, MTE3_MTE2, S_MTE3 };
enum class PaddingMode { Normal, Compact };
template <typename T>
struct LocalTensor {
    T* data = nullptr;
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t count)
    {
        ReduceTest::Check(offset % 32 == 0 && offset + count * sizeof(T) <= ReduceTest::ub.size(),
                          "UB bounds/alignment");
        data = reinterpret_cast<T*>(ReduceTest::ub.data() + offset);
    }
    explicit LocalTensor(T* address) : data(address) {}
    LocalTensor operator[](uint64_t offset) const
    {
        return LocalTensor(data + offset);
    }
    T GetValue(uint32_t index) const
    {
        if (static_cast<const void*>(data + index) == ReduceTest::readyResultAddress)
            ReduceTest::Check(!ReduceTest::readyResultPending, "Scalar read before V_S acquired ready result");
        ++ReduceTest::scalarReads;
        if constexpr (std::is_same_v<T, float>) ++ReduceTest::scalarFloatReads;
        return data[index];
    }
    T* GetPhyAddr() const
    {
        return data;
    }
    void SetValue(uint32_t index, T value)
    {
        data[index] = value;
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        return LocalTensor<U>(reinterpret_cast<U*>(data));
    }
};
template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* address)
    {
        data = address;
    }
    const T* GetPhyAddr() const
    {
        return data;
    }
    T* GetPhyAddr(uint64_t offset) const
    {
        return data + offset;
    }
    GlobalTensor operator[](uint64_t offset) const
    {
        return {data + offset};
    }
};
struct DataCopyParams {
    uint16_t blockCount, blockLen, srcStride, dstStride;
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride, reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool pad;
    uint32_t left, right;
    T value;
};
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        if (ReduceTest::recordRouteReads) ReduceTest::routeReads.emplace_back(src.data, count);
    }
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        if (ReduceTest::deferInputs) {
            ReduceTest::Check(!ReduceTest::busyInputs[dst.data], "input buffer reused before vector release");
            ReduceTest::busyInputs[dst.data] = true;
            ReduceTest::queuedInputs.push_back({dst.data, {src.data, src.data + count}});
            return;
        }
    }
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void Duplicate(LocalTensor<T> dst, T value, uint32_t count)
{
    std::fill_n(dst.data, count, value);
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    std::memmove(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyParams p)
{
    if constexpr (std::is_same_v<T, uint32_t>) {
        ReduceTest::Check(p.blockLen == 1U && p.srcStride == 15U && p.dstStride == 0U,
                          "packet tails not gathered in 32B blocks");
        ++ReduceTest::flagDmas;
        std::vector<uint8_t> bytes(p.blockCount * 32U);
        for (uint32_t i = 0; i < p.blockCount; ++i) {
            auto* from = reinterpret_cast<uint8_t*>(src.data) + i * 512U;
            if (ReduceTest::flagSourceBegin != nullptr) {
                using namespace ReduceTest;
                Check(from >= flagSourceBegin && from + 32U <= flagSourceBegin + flagSourceBytes,
                      "flag DMA escaped owned token");
                const size_t offset = from - flagSourceBegin;
                Check(offset % 512U == 480U &&
                          (flagSourceMask & (uint32_t{1} << (offset / (flagSourcePackets * 512U)))) != 0U,
                      "flag DMA read an inactive row");
            }
            std::memcpy(bytes.data() + i * 32U, from, 32U);
        }
        ReduceTest::queuedFlags.push_back({reinterpret_cast<uint8_t*>(dst.data), std::move(bytes)});
        return;
    }
    for (uint32_t i = 0; i < p.blockCount; ++i) {
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data) + i * (p.blockLen + p.dstStride) * 32U,
                    reinterpret_cast<uint8_t*>(src.data) + i * (p.blockLen + p.srcStride) * 32U, p.blockLen * 32U);
    }
}
template <typename T, PaddingMode Mode = PaddingMode::Normal>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams p, DataCopyPadExtParams<T> pad)
{
    const uint32_t blockBytes = Mode == PaddingMode::Compact ? p.blockLen : (p.blockLen + 31U) / 32U * 32U;
    std::vector<uint8_t> result((p.blockCount * blockBytes + 31U) / 32U * 32U, 0xA5);
    for (uint32_t i = 0; i < p.blockCount; ++i) {
        std::memcpy(result.data() + i * blockBytes,
                    reinterpret_cast<uint8_t*>(src.data) + i * (p.blockLen + p.srcStride), p.blockLen);
        if (pad.pad) {
            for (uint32_t offset = p.blockLen; offset < blockBytes; offset += sizeof(T))
                std::memcpy(result.data() + i * blockBytes + offset, &pad.value, sizeof(T));
        }
    }
    if constexpr (Mode == PaddingMode::Compact) {
        auto* target = reinterpret_cast<bfloat16_t*>(dst.data);
        if (ReduceTest::deferInputs) {
            ReduceTest::Check(!ReduceTest::busyInputs[target], "packed input reused before vector release");
            ReduceTest::busyInputs[target] = true;
            std::vector<bfloat16_t> values(result.size() / sizeof(bfloat16_t));
            std::memcpy(values.data(), result.data(), result.size());
            ReduceTest::queuedInputs.push_back({target, std::move(values)});
            return;
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        if (ReduceTest::deferInputs) {
            // Weights must be acquired by the first input's MTE2_V too.
            ReduceTest::queuedFlags.push_back({reinterpret_cast<uint8_t*>(dst.data), std::move(result)});
            return;
        }
    }
    std::memcpy(dst.data, result.data(), result.size());
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p)
{
    ReduceTest::Check(p.blockCount == 1, "unexpected packed DMA");
    std::memcpy(dst.data, src.data, p.blockLen);
}
template <typename D, typename S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode, uint32_t count)
{
    if constexpr (std::is_same_v<S, bfloat16_t>) {
        if (ReduceTest::deferInputs) {
            ReduceTest::Check(ReduceTest::queuedInputs.empty(), "input load lacks a completion event");
            bool prefetched = false;
            for (const auto& loads : ReduceTest::inputEvents) {
                for (const auto& load : loads) {
                    ReduceTest::Check(load.destination != src.data, "Cast before input DMA completed");
                    prefetched = true;
                }
            }
            ReduceTest::overlappedCasts += prefetched;
            ReduceTest::lastCastInput = src.data;
        }
    }
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = D(static_cast<float>(src.data[i]));
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> src, float weight, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = src.data[i] * weight;
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = a.data[i] + b.data[i];
}
template <HardEvent E>
void SetFlag(TEventID id)
{
    auto& flag = ReduceTest::events[static_cast<size_t>(E)][id];
    ReduceTest::Check(flag == 0, "reusing a live event");
    flag = 1;
    if constexpr (E == HardEvent::MTE2_V) {
        ReduceTest::Check(ReduceTest::flagEvents[id].empty(), "flag event still owns DMA");
        ReduceTest::flagEvents[id].swap(ReduceTest::queuedFlags);
    }
    if (ReduceTest::deferInputs) {
        if constexpr (E == HardEvent::MTE2_V) {
            ReduceTest::Check(ReduceTest::inputEvents[id].empty(), "input event still owns a load");
            ReduceTest::inputEvents[id].swap(ReduceTest::queuedInputs);
        } else if constexpr (E == HardEvent::V_MTE2) {
            if (id != 7U) {
                ReduceTest::Check(ReduceTest::lastCastInput != nullptr, "release before vector input read");
                ReduceTest::releaseInputs[id] = ReduceTest::lastCastInput;
                ReduceTest::lastCastInput = nullptr;
            }
        }
    }
}
template <HardEvent E>
void WaitFlag(TEventID id)
{
    auto& flag = ReduceTest::events[static_cast<size_t>(E)][id];
    ReduceTest::Check(flag == 1, "waiting on an unpublished event");
    flag = 0;
    if constexpr (E == HardEvent::V_S) {
        if (id == 7U) ReduceTest::readyResultPending = false;
    }
    if constexpr (E == HardEvent::MTE2_V) {
        for (const auto& load : ReduceTest::flagEvents[id])
            std::copy(load.bytes.begin(), load.bytes.end(), load.destination);
        ReduceTest::flagEvents[id].clear();
    }
    if (ReduceTest::deferInputs) {
        if constexpr (E == HardEvent::MTE2_V) {
            for (const auto& load : ReduceTest::inputEvents[id])
                std::copy(load.values.begin(), load.values.end(), load.destination);
            ReduceTest::inputEvents[id].clear();
        } else if constexpr (E == HardEvent::V_MTE2) {
            if (id != 7U) {
                ReduceTest::Check(ReduceTest::releaseInputs[id] != nullptr, "missing input release");
                ReduceTest::busyInputs[ReduceTest::releaseInputs[id]] = false;
                ReduceTest::releaseInputs[id] = nullptr;
            }
        }
    }
}
template <int>
void PipeBarrier()
{
}
inline int64_t GetSystemCycle()
{
    return ++ReduceTest::cycle;
}
}  // namespace AscendC

template <auto Function>
void asc_vf_call(uint32_t* ids, uint32_t* masks, int32_t* tokens, int32_t* count, uint32_t rows, uint32_t topK,
                 uint32_t first, uint32_t width)
{
    Function(ids, masks, tokens, count, rows, topK, first, width);
}

template <auto Function>
void asc_vf_call(uint32_t* tails, uint32_t* result, uint32_t packets, uint32_t topk, uint32_t mask, uint32_t sequence)
{
    using namespace ReduceTest;
    Check(queuedFlags.empty(), "VF before flag DMA submitted");
    for (const auto& loads : flagEvents) Check(loads.empty(), "VF before flag DMA completed");
    checkSnapshotReads = true;
    snapshotBase = reinterpret_cast<uint8_t*>(tails);
    snapshotResult = reinterpret_cast<uint8_t*>(result);
    snapshotMask = mask;
    snapshotPackets = packets;
    ++readyVfs;
    Function(tails, result, packets, topk, mask, sequence);
    readyResultAddress = result;
    readyResultPending = true;
    checkSnapshotReads = false;
}

template <auto Function>
void asc_vf_call(bfloat16_t* input, float* sum, float* weight, uint32_t hidden)
{
    using namespace ReduceTest;
    Check(queuedFlags.empty(), "row VF before weight DMA submitted");
    for (const auto& loads : flagEvents) Check(loads.empty(), "row VF before weight DMA completed");
    if (deferInputs) {
        Check(queuedInputs.empty(), "input load lacks a completion event");
        bool prefetched = false;
        for (const auto& loads : inputEvents)
            for (const auto& load : loads) {
                Check(load.destination != input, "row VF before input DMA completed");
                prefetched = true;
            }
        overlappedCasts += prefetched;
        lastCastInput = input;
    }
    checkRowReads = true;
    rowBegin = reinterpret_cast<uint8_t*>(input);
    sumBegin = reinterpret_cast<uint8_t*>(sum);
    weightBegin = reinterpret_cast<uint8_t*>(weight);
    rowReadBytes = (hidden + 63U) / 64U * 128U;
    sumReadBytes = (hidden + 63U) / 64U * 256U;
    ++rowVfs;
    Function(input, sum, weight, hidden);
    checkRowReads = false;
}

#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc_ep32/kernels/mega_moe/common/ascend_timer_v2.hpp"
class AscendTimerDevice {
public:
    bool Enabled() const
    {
        return true;
    }
    std::vector<uint32_t> acquired;
    uint32_t reduced = 0;
    uint32_t readyRecords = 0, syncRecords = 0;
    void RecordDetail(AscendTimer::DynamicTimingType type, int64_t start, int64_t end, uint64_t first, uint64_t second)
    {
        ReduceTest::Check(end > start, "empty trace interval");
        if (type == AscendTimer::AIV_GATEWAY_PAYLOAD_WAIT) {
            auto source = static_cast<uint32_t>(first >> 32) - 1U;
            ReduceTest::Check(static_cast<uint32_t>(first) == (source + 16U) % 32U + 1U && second == 259U,
                              "gateway acquire trace lost its source, gateway, or sequence");
            acquired.push_back(source);
        } else if (type == AscendTimer::AIV_COMBINE_REDUCE) {
            ReduceTest::Check((first >> 32) == ReduceTest::activeLane + 1U && (second >> 32) != 0U,
                              "wrong reduction trace");
            ++reduced;
        } else if (type == AscendTimer::AIV_COMBINE_READY) {
            ReduceTest::Check((first >> 32) == ReduceTest::activeLane + 1U &&
                                  (static_cast<uint32_t>(second) & ~static_cast<uint32_t>(second >> 32)) == 0U,
                              "wrong ready snapshot / bits outside pending mask");
            ++readyRecords;
        } else if (type == AscendTimer::AIV_COMBINE_SYNC) {
            ReduceTest::Check((first >> 32) == ReduceTest::activeLane + 1U && static_cast<uint32_t>(first) >= 1U &&
                                  static_cast<uint32_t>(first) <= 4U,
                              "wrong synchronization marker");
            ++syncRecords;
        }
    }
};
struct AscendTimerFixedScope {
    AscendTimerFixedScope(AscendTimerDevice*, AscendTimer::FixedTiming) {}
};
struct AscendTimerCombineProgressScope {
    AscendTimerCombineProgressScope(AscendTimerDevice*, AscendTimer::DynamicTimingType) {}
    void StartScan() {}
    void StopScan() {}
    void Finish(uint32_t, uint32_t, uint32_t, uint32_t) {}
};
namespace MegaMoeImpl {
using namespace AscendC;
constexpr uint32_t MEGAMOE_ROCE_SCRATCH_BYTES = 256U, MEGAMOE_ROCE_EVENT_ID = 7U;
struct MoeStageCommonConfig {
    uint32_t rankId, tokenNum, topK, tokenHiddenDim, moeExpertPerRank, sharedExpertNum = 0, worldSize = 32;
};
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
};
struct WorkRange {
    uint32_t start, count;
};
inline WorkRange TilingByJobContext(uint32_t count, uint32_t job, uint32_t jobs, uint32_t)
{
    const uint32_t chunk = (count + jobs - 1U) / jobs, first = chunk * job;
    return {first, first < count ? std::min(chunk, count - first) : 0U};
}
struct Params {
    MegaMoeTilingData* tilingData;
    PeermemInfo peermemInfo;
    GM_ADDR probsGmAddr = nullptr;
    GM_ADDR expertIdxGmAddr = nullptr;
    GM_ADDR y2GmAddr = nullptr;
    struct {
        GM_ADDR sharedExpertGmm2TileCounterPtr = nullptr;
        GM_ADDR sharedExpertResultPtr = nullptr;
    } workspaceInfo{};
    GM_ADDR inputScaleGmAddr = nullptr;
};
template <HardEvent E, uint32_t Id>
void SyncFuncStatic()
{
    SetFlag<E>(Id);
    WaitFlag<E>(Id);
}
inline int32_t ReadGmByPassDCache(const int32_t* p) { return *p; }
template <class... T> void RoceWaitCommit(T&&...) { throw std::runtime_error("unexpected early commit"); }
struct RoceWaitState {};
std::function<void()> deliverLateRows;
enum class RoceWaitKind { CombineRows, LocalReadiness };
inline void RoceCheckWait(RoceWaitState&, uint32_t, uint8_t*, RoceWaitKind kind, uint32_t, uint32_t, int32_t* = nullptr,
                          uint32_t = 1U)
{
    if (kind == RoceWaitKind::LocalReadiness && deliverLateRows) {
        deliverLateRows();
        return;
    }
    throw std::runtime_error("drain waited on an incomplete producer");
}
inline void RocePutBytes(uint8_t* dst, uint8_t* src, uint32_t bytes, uint32_t peer, uint8_t*, AscendTimerDevice*,
                         AscendTimer::RdmaPhase, bool commit = false, uint32_t qp = 1U)
{
    using namespace ReduceTest;
    Check(!commit && qp == activeLane + 1U && peer == expectedPeer, "foreign QP owner/destination");
    Check(queuedInputs.empty(), "input DMA not submitted before POST");
    Check(queuedFlags.empty(), "flag DMA not submitted before POST");
    for (const auto& loads : flagEvents) Check(loads.empty(), "flag DMA still pending before POST");
    for (const auto& loads : inputEvents) Check(loads.empty(), "input DMA still pending before POST");
    for (const auto& input : busyInputs) Check(!input.second, "input buffer not released before POST");
    for (const auto& type : events)
        for (int live : type) Check(live == 0, "reduction leaked an event");
    Check(src >= localBase && src + bytes <= localBase + heapBytes && dst >= localBase &&
              dst + bytes <= localBase + heapBytes,
          "non-symmetric operand");
    posts.push_back({qp, src, static_cast<uint64_t>(dst - localBase), {src, src + bytes}});
}
namespace Mxfp8 {
template <typename A, typename B, typename... Args>
void DeQuantMxFp8(Args&&...)
{
}
}  // namespace Mxfp8
inline int32_t* GetCombineSyncCounterAddress(int32_t* base, uint32_t group)
{
    return base + group * INT_CACHELINE;
}
inline void WaitUntilGmFlagEquals(int32_t* address, int32_t expected)
{
    ReduceTest::Check(*address == expected, "shared result not ready");
}
}  // namespace MegaMoeImpl
#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_hierarchical_reduce.hpp"
#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_route_histogram.hpp"
#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_dispatch_packet_vf.hpp"
#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_unpermute.hpp"

namespace MegaMoeImpl {
/* PRODUCTION_WEIGHT_PACKER */
/* BF16_PACKER */
/* DISPATCH_ROW_TYPES */
struct TokenDispatchConfig {
    uint32_t quantTokenScaleAlignBytes;
    MegaMoeDispatchBufferConfig bufferConfig{};
    uint32_t routeIndexAlignSize = 256;
};
template <class T>
struct TokenDispatchScratch {
    LocalTensor<int32_t> validTopkIndexTensor;
    uint32_t gatewayReady = 0;
    LocalTensor<int32_t> cumsumInfoTensor{nullptr};
    AscendTimerDevice* trace = nullptr;
    uint32_t traceRank = 0;
};
std::vector<DispatchRow> copiedRows, pendingRows;
template <class A, class Q, bool Prefetch>
void IssueDispatchRow(const TokenDispatchConfig&, const Params&, GM_ADDR*, TokenDispatchScratch<A>&,
                      DispatchCopyProgress&, const DispatchRow& row)
{
    pendingRows.push_back(row);
}
template <class A, class Q, bool Prefetch>
void DrainDispatchCopyBuffers(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<A>&,
                              DispatchCopyProgress&)
{
    copiedRows.insert(copiedRows.end(), pendingRows.begin(), pendingRows.end());
    pendingRows.clear();
}
struct MoeSyncWorkspaceLayout {};
static std::vector<std::pair<int, int>> publishedRanges;
void PublishGmm1TileReady(const MoeSyncWorkspaceLayout&, const Params&, uint32_t, int32_t, int32_t begin, int32_t end)
{
    ReduceTest::Check(pendingRows.empty() && copiedRows.size() == static_cast<size_t>(end),
                      "published tile before its rows and metadata retired");
    publishedRanges.emplace_back(begin, end);
}
#define ACLSHMEMI_RDMA_K_BACKEND_HNS_1825 1
#ifdef TEST_DISPATCH_TIMER
#define DEEPEP_MEGAMOE_TIMER 1
/* DISPATCH_RELAY_SCOPE */
#endif
/* PRODUCTION_DISPATCH_QUEUE */
#ifdef TEST_DISPATCH_TIMER
#undef DEEPEP_MEGAMOE_TIMER
#endif
#undef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
}
using namespace MegaMoeImpl;

void CheckDeferredRows()
{
    using namespace ReduceTest;
    MegaMoeTilingData tiling{};
    tiling.epWorldSize = 32;
    tiling.dispatchRanksPerNode = 16;
    tiling.topoType = TOPO_TYPE_MTE_ROCE;
    tiling.topK = 2;
    tiling.hierarchicalSequence = 259;
    tiling.roceScratchUbOffset = 8192;
    constexpr uint32_t packets = 2;
    std::vector<uint8_t> heap(96 * 1024);
    std::vector<uint8_t> second(heap.size());
    Params params{&tiling, {}};
    params.peermemInfo.rankSyncInWorldPtr = heap.data();
    params.peermemInfo.relayPayloadPtr = heap.data() + 8192;
    params.peermemInfo.gatewayPayloadReadyPtr = heap.data();
    GM_ADDR peers[32]{};
    peers[0] = heap.data();
    peers[1] = second.data();
    auto mark = [&](auto& memory, int token, int packet) {
        auto* tail = memory.data() + 8192 + ((token * packets + packet + 1) * 512) - 4;
        std::memcpy(tail, &tiling.hierarchicalSequence, 4);
    };
    TokenDispatchConfig config{640U};
    std::array<int32_t, 64> routes;
    TokenDispatchScratch<int> scratch{LocalTensor<int32_t>(routes.data())};
    MoeStageCommonConfig common{};
    std::vector<int32_t> indices(2 * 32 * 64);
    params.peermemInfo.maskRecvPtr = reinterpret_cast<GM_ADDR>(indices.data());
    // Expert 1, ordinal offset 2. The local and already acquired sources come after the slow range.
    for (uint32_t i = 0; i < indices.size(); ++i) indices[i] = 2 * (i % 64);
    // Resident indices from four independent gateways form one vector check. Vary both
    // the final batch length and the hint mask after the initial source traversal.
    std::vector<uint8_t> third(heap.size()), fourth(heap.size());
    peers[2] = third.data();
    peers[3] = fourth.data();
    std::vector<uint8_t>* gateways[] = {&heap, &second, &third, &fourth};
    for (uint32_t mixed : {0U, 1U}) {
        const uint32_t total = 8U - mixed;
        scratch.gatewayReady = 0U;
        config.bufferConfig.routeItemsPerBatch = 32U;
        DispatchCopyProgress progress{};
        DispatchRouteBatch batch{};
        for (uint32_t i = 0; i < 4; ++i) {
            std::fill(gateways[i]->begin(), gateways[i]->end(), 0);
            const int rows = i == 3 && mixed ? 1 : 2;
            for (int j = 0; j < rows; ++j) {
                const int token = 2 * i + j;
                if (i == 1 && mixed) continue;  // This gateway will supply the only hint.
                mark(*gateways[i], token, 0);
                mark(*gateways[i], token, 1);
            }
            DispatchRankTokens<int, int, false>(config, common, params, peers, scratch, 1, 0, 16 + i,
                                                2 * i, 2 * i, 2 * i + rows, batch, progress);
        }
        Check(batch.deferredSources == 4, "expected four initially delayed sources");
        if (mixed) std::memcpy(heap.data() + HIERARCHICAL_FLAG_BYTES, &tiling.hierarchicalSequence, 4);
        const uint32_t beforeVfs = readyVfs, beforeDmas = flagDmas;
        FinishDispatchSources<int, int, false>(config, common, params, peers, scratch, 1, batch, progress);
        Check(readyVfs == beforeVfs + 1 && flagDmas == beforeDmas + total - 2 * mixed,
              "resident sources must share one VF and skip acquired lanes");
        Check(copiedRows.size() == total, "incomplete or duplicate cross-source copies");
        for (uint32_t j = 0; j < total; ++j)
            Check(copiedRows[j].source == 16 + j / 2 && copiedRows[j].route == static_cast<int>(2 * j) &&
                      copiedRows[j].destination == static_cast<int>(j), "wrong mixed-source dispatch coordinates");
        copiedRows.clear();
    }
    for (uint32_t capacity : {16U, 64U})
        for (uint32_t release : {0U, 1U, 2U}) {
            std::fill(second.begin(), second.end(), 0);
            std::fill(heap.begin(), heap.end(), 0);
            scratch.gatewayReady = 0;
            config.bufferConfig.routeItemsPerBatch = capacity;
            DispatchCopyProgress progress{};
            DispatchRouteBatch batch{};
            uint32_t stale = tiling.hierarchicalSequence - 128U;
            std::memcpy(heap.data(), &stale, 4);
            if (release == 0) std::memcpy(heap.data() + HIERARCHICAL_FLAG_BYTES, &tiling.hierarchicalSequence, 4);
            // Exercise both directions: the last packet alone and every packet except the last.
            const uint32_t latePacket = release == 2 ? 1U : 0U;
            for (uint32_t i = 2; i < 43; ++i) {
                mark(heap, i, 1U - latePacket);
                const uint32_t oldEpoch = tiling.hierarchicalSequence - 256U;
                std::memcpy(heap.data() + 8192 + ((i * packets + latePacket + 1) * 512) - 4, &oldEpoch, 4);
            }
            for (uint32_t token : {3U, 7U}) mark(heap, token, latePacket);
            auto submit = [&](uint32_t source, int dst, int count) {
                DispatchRankTokens<int, int, false>(config, common, params, peers, scratch, 1, 5, source, dst, 2,
                                                    2 + count, batch, progress);
            };
            submit(16, 100, 41);
            submit(0, 10, 1);
            submit(17, 50, 3);
            Check(batch.deferredSources == (release == 0 ? 1U : 2U) && batch.deferred[0].count == 41,
                  "deferred source split during traversal");
            uint32_t polls = 0;
            deliverLateRows = [&] {
                if (release && !polls++) {
                    Check(copiedRows.size() == 3 && copiedRows.front().source == 0,
                          "initially deferred source delivered early");
                    if (release == 1)
                        std::memcpy(heap.data() + HIERARCHICAL_FLAG_BYTES, &tiling.hierarchicalSequence, 4);
                    else
                        for (int token = 2; token < 5; ++token)
                            for (int packet = 0; packet < 2; ++packet) mark(second, token, packet);
                    return;
                }
                Check(copiedRows.size() == 6 && copiedRows[0].destination == 15,
                      "local and later remote source must retire before releasing the 41 slow rows");
                const uint32_t startB = release ? 3U : 1U;
                for (uint32_t j = 0; j < 3; ++j) {
                    const auto& row = copiedRows[startB + j];
                    Check(row.source == 17 && row.destination == static_cast<int>(55 + j) &&
                              row.route == static_cast<int>(4 + 2 * j),
                          "late B lost its coordinates");
                }
                for (uint32_t i = 2; i < 43; ++i) mark(heap, i, latePacket);
            };
            FinishDispatchSources<int, int, false>(config, common, params, peers, scratch, 1, batch, progress);
            Check(copiedRows.size() == 45, "missing or duplicate rows");
            std::array<bool, 41> seen{};
            for (uint32_t i = 0; i < 45; ++i) {
                const auto& r = copiedRows[i];
                if (r.source != 16) continue;
                const int j = r.destination - 105;
                Check(r.source == 16 && j >= 0 && j < 41 && !seen[j] && r.route == 2 * (j + 2),
                      "wrong deferred coordinates");
                seen[j] = true;
            }
            Check(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "missing slow row");
            deliverLateRows = {};
            copiedRows.clear();
        }
    // Derive the compact table and source prefix from concrete top-k slots, then run the production
    // worker-range traversal. Worker 0 hits both delayed ranks. Worker 1 owns the remaining peer.
    std::array<int32_t, 64> prefix{};
    std::vector<std::pair<uint32_t, int32_t>> expected;
    for (uint32_t rank = 0; rank < 32; ++rank) {
        int32_t count = 0;
        for (int token = 0; token < 64; ++token) {
            const bool hit = (rank == 0 && token == 9) || (rank == 16 && token >= 2 && token < 43) ||
                             (rank == 17 && token % 20 == 3 && token < 60) || (rank == 18 && token < 2);
            const std::array<int, 2> ids =
                (rank + token) % 2 ? std::array<int, 2>{hit ? 1 : 2, 3} : std::array<int, 2>{3, hit ? 1 : 2};
            for (int k = 0; k < 2; ++k)
                if (ids[k] == 1) {
                    indices[(32 + rank) * 64 + count++] = 2 * token + k;
                    expected.emplace_back(rank, 2 * token + k);
                }
        }
        prefix[32 + rank] = expected.size();
    }
#ifdef TEST_DISPATCH_TIMER
    AscendTimerDevice timer;
    scratch.trace = &timer;
#endif
    scratch.cumsumInfoTensor = LocalTensor<int32_t>(prefix.data());
    scratch.gatewayReady = 0;
    std::fill(heap.begin(), heap.end(), 0);
    std::fill(second.begin(), second.end(), 0);
    // The third gateway is already ready, and is owned by the second worker.
    peers[2] = second.data();
    std::memcpy(heap.data() + 2 * HIERARCHICAL_FLAG_BYTES, &tiling.hierarchicalSequence, 4);
    config.bufferConfig.routeItemsPerBatch = 16;
    uint32_t polls = 0;
    deliverLateRows = [&] {
        Check(publishedRanges.empty(), "premature tile ready");
        if (polls++ == 0) {
            Check(copiedRows.size() == 1 && copiedRows[0].source == 0, "local worker row delayed");
            std::memcpy(heap.data() + HIERARCHICAL_FLAG_BYTES, &tiling.hierarchicalSequence, 4);
        } else {
            Check(copiedRows.size() == 4 && copiedRows[1].source == 17 && copiedRows[3].source == 17,
                  "second source stalled");
            std::memcpy(heap.data(), &tiling.hierarchicalSequence, 4);
        }
    };
    DispatchOwnedExpertRows<int, int, 256, false>(config, common, {}, params, peers, scratch, 1, {0, 0, 45});
    DispatchOwnedExpertRows<int, int, 256, false>(config, common, {}, params, peers, scratch, 1, {0, 45, 47});
    Check(publishedRanges == std::vector<std::pair<int, int>>{{0, 45}, {45, 47}} && polls == 2, "wrong worker ranges");
    std::sort(copiedRows.begin(), copiedRows.end(),
              [](const auto& x, const auto& y) { return x.destination < y.destination; });
    for (size_t i = 0; i < expected.size(); ++i)
        Check(copiedRows[i].destination == static_cast<int>(i) && copiedRows[i].source == expected[i].first &&
                  copiedRows[i].route == expected[i].second,
              "skewed worker row mapping");
    copiedRows.clear();
#ifdef TEST_DISPATCH_TIMER
    Check(timer.acquired == std::vector<uint32_t>{17, 16, 18}, "missing or duplicate gateway acquire");
#endif
    publishedRanges.clear();
    deliverLateRows = {};
}

void CheckCase(uint32_t sourceRank, uint32_t bs, uint32_t capacity, uint32_t topk, uint32_t hidden, uint32_t pattern,
               uint32_t sequence, bool shared)
{
    using namespace ReduceTest;
    constexpr uint32_t qps = 16U;
    ub.fill(0);
    events = {};
    posts.clear();
    queuedInputs.clear();
    inputEvents = {};
    releaseInputs = {};
    busyInputs.clear();
    lastCastInput = nullptr;
    overlappedCasts = 0;
    scalarFloatReads = 0;
    rowVfs = 0;
    MegaMoeTilingData t{};
    t.bs = bs;
    t.numMaxTokensPerRank = capacity;
    t.topK = topk;
    t.h = hidden;
    t.epWorldSize = 32U;
    t.dispatchRanksPerNode = 16U;
    t.moeExpertPerRank = 12U;
    t.blockAivNum = 56U;
    t.aicNum = 28U;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.roceScratchUbOffset = 256U * 1024U - 512U;
    t.hierarchicalSequence = sequence;
    t.sharedExpertNum = shared ? 1U : 0U;
    mainUbLimit = CalcGatewayCombineReduceUbBytes(&t);
    const PeermemSizeParams sizes{capacity,           topk, hidden, 12U, 32U, 2U, 1U, false, false,
                                  TOPO_TYPE_MTE_ROCE, true, 16U};
    heapBytes = CalcPeermemLeastSize(sizes);
    std::vector<uint8_t> sourceHeap(heapBytes, 0xA5), gatewayHeap(heapBytes, 0xA5);
    PeermemInfo source(sourceHeap.data(), &t), gateway(gatewayHeap.data(), &t);
    Check(gateway.combineReduceCommitPtr + qps * 512U == gatewayHeap.data() + heapBytes, "layout budget mismatch");
    localBase = gatewayHeap.data();
    remoteBase = sourceHeap.data();
    expectedPeer = sourceRank;
    const uint32_t gatewayRank = HierarchicalGateway(sourceRank);
    MoeStageCommonConfig srcCommon{sourceRank, bs, topk, hidden, 12U, shared ? 1U : 0U};
    MoeStageCommonConfig gwCommon{gatewayRank, bs, topk, hidden, 12U};
    std::vector<int32_t> ids(bs * topk);
    std::vector<float> weights(bs * topk);
    std::vector<bfloat16_t> expertRows(static_cast<size_t>(bs) * topk * hidden);
    std::vector<bfloat16_t> output(bs * hidden), sharedRows(bs * hidden);
    std::vector<float> expectedPartial(bs * hidden, 0.0f);
    std::vector<bfloat16_t> expectedOutput(bs * hidden);
    std::vector<uint32_t> masks(bs, 0U);
    const uint32_t sourceHost = HierarchicalHost(sourceRank);
    auto* originalLocal = reinterpret_cast<bfloat16_t*>(source.combineSendPtr);
    const uint32_t gatewayRowBytes = (hidden * 2U + 507U) / 508U * 512U;
    const uint32_t packetCount = gatewayRowBytes / 512U;
    const bfloat16_t poison(std::nanf(""));
    std::fill_n(originalLocal, capacity * topk * hidden, poison);
    for (uint32_t token = 0; token < bs; ++token) {
        for (uint32_t k = 0; k < topk; ++k) {
            const bool remote = pattern == 1 || (pattern == 2 && (token + k) % 3 == 0) ||
                                (pattern == 3 && (token % 4) >= 2) || (pattern == 4 && token == bs - 1U);
            ids[token * topk + k] = static_cast<int32_t>((remote ? 1U - sourceHost : sourceHost) * 192U + k);
            weights[token * topk + k] = static_cast<float>(static_cast<int>((token * 11U + k * 7U) % 17U) - 8) / 19.0f;
            if (remote) masks[token] |= uint32_t{1} << k;
            for (uint32_t h = 0; h < hidden; ++h) {
                const size_t index = (static_cast<size_t>(token) * topk + k) * hidden + h;
                expertRows[index] =
                    bfloat16_t(static_cast<float>(static_cast<int>((token * 31U + k * 13U + h) % 61U) - 30) / 23.0f);
                if (remote) {
                    const uint32_t byte = h * 2U;
                    const uint64_t offset =
                        static_cast<uint64_t>(token * topk + k) * gatewayRowBytes + (byte / 508U * 512U + byte % 508U);
                    std::memcpy(gateway.gatewayCombinePtr + offset, &expertRows[index], 2U);
                } else {
                    originalLocal[index] = expertRows[index];
                }
            }

            auto* row = reinterpret_cast<uint32_t*>(gateway.gatewayCombinePtr +
                                                    static_cast<uint64_t>(token * topk + k) * gatewayRowBytes);
            for (uint32_t p = 0; p < packetCount; ++p) row[p * 128U + 127U] = sequence - 1U;
        }
        for (uint32_t h = 0; h < hidden; ++h) {
            float remoteSum = 0.0f, localSum = 0.0f;
            bool firstRemote = true, firstLocal = true;
            for (uint32_t k = 0; k < topk; ++k) {
                const float value =
                    static_cast<float>(expertRows[(static_cast<size_t>(token) * topk + k) * hidden + h]) *
                    weights[token * topk + k];
                const bool remote = (masks[token] & (uint32_t{1} << k)) != 0U;
                float& sum = remote ? remoteSum : localSum;
                bool& first = remote ? firstRemote : firstLocal;
                sum = first ? value : sum + value;
                first = false;
            }
            expectedPartial[token * hidden + h] = remoteSum;
            float total = firstLocal ? remoteSum : firstRemote ? localSum : localSum + remoteSum;
            sharedRows[token * hidden + h] = bfloat16_t(static_cast<float>(h % 7U) / 13.0f);
            if (shared) total += static_cast<float>(sharedRows[token * hidden + h]);
            expectedOutput[token * hidden + h] = bfloat16_t(total);
        }
    }
    Params sourceParams{&t, source, reinterpret_cast<uint8_t*>(weights.data()), reinterpret_cast<uint8_t*>(ids.data()),
                        reinterpret_cast<uint8_t*>(output.data())};
    // Execute the production weight packer, then model one combined metadata Put.
    const bool batched = bs == 96 && topk == 6;
    if (batched) PackBf16Bs96RailMetadata(srcCommon, sourceParams);
    else PackHierarchicalCombineWeights(srcCommon, sourceParams);
    const uint64_t slot = sourceHost * CalcRailTopkIdsSlotBytes(&t);
    auto* sourceRail = reinterpret_cast<int32_t*>(source.railTopkIdsRecvPtr + slot);
    for (uint32_t row = 0; row < bs * topk; ++row) {
        auto& id = sourceRail[(row / 127U) * 128U + row % 127U];
        if (batched) Check(id == ids[row], "packed ID changed");
        else id = ids[row];
    }
    for (uint32_t packet = 0; packet < (bs * topk + 126U) / 127U; ++packet) {
        auto& marker = sourceRail[packet * 128U + 127U];
        if (batched) Check(static_cast<uint32_t>(marker) == sequence, "stale rail packet");
        else marker = static_cast<int32_t>(sequence);
    }
    const uint64_t prefix = HierarchicalMetadataPrefixBytes(&t);
    const uint64_t bytes = prefix + CalcRailTopkIdsSlotDataBytes(bs, topk);
    std::memcpy(gateway.railTopkIdsRecvPtr + slot - prefix, source.railTopkIdsRecvPtr + slot - prefix, bytes);
    Check(std::memcmp(gateway.railTopkIdsRecvPtr + slot - prefix, weights.data(), weights.size() * 4U) == 0,
          "packed weights differ");
    Params params{&t, gateway};
    std::array<HierarchicalCombineProgress, qps> state{};
    AscendTimerDevice timer;
    auto* scratch = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(t.roceScratchUbOffset));
    for (activeLane = 0; activeLane < qps; ++activeLane) {
        const uint32_t first = bs * activeLane / qps, end = bs * (activeLane + 1U) / qps;
        routeReads.clear();
        recordRouteReads = true;
        InitializeHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, &timer);
        recordRouteReads = false;
        uint32_t expectedRows = 0;
        for (uint32_t token = 0; token < bs; ++token) {
            const uint32_t expected = token >= first && token < end ? masks[token] : 0U;
            Check(state[activeLane].pendingTopkMasks[token] == expected, "wrong lane mask / parsed packet marker");
            expectedRows += CombineMaskCount(expected);
        }
        Check(state[activeLane].expectedRows == expectedRows, "wrong lane row count");
        std::set<uint32_t> expectedChunks, actualChunks;
        for (uint32_t row = first * topk; row < end * topk; ++row)
            expectedChunks.insert(row / RAIL_PACKET_DATA_WORDS * RAIL_PACKET_WORDS +
                                  (row % RAIL_PACKET_DATA_WORDS) / 64U * 64U);
        const auto* routes = reinterpret_cast<int32_t*>(gateway.railTopkIdsRecvPtr + slot);
        for (auto [address, words] : routeReads) {
            Check(words == 64U, "route DMA changed reserved scratch size");
            actualChunks.insert(static_cast<uint32_t>(address - routes));
        }
        Check(actualChunks == expectedChunks && routeReads.size() == expectedChunks.size(),
              "lane initializer read unrelated or duplicate route chunks");
        Check(!TryProgressHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer),
              "stale flags released data");
        TraceHierarchicalCombineBoundary(gwCommon, params, state[activeLane], activeLane, &timer, 1U);
        TraceHierarchicalCombineBoundary(gwCommon, params, state[activeLane], activeLane, &timer, 2U);
    }
    // Publish expert rows in reverse order. A partial token must never be posted.
    deferInputs = true;
    std::vector<uint32_t> published(bs, 0U);
    for (uint32_t row = bs * topk; row-- > 0;) {
        const uint32_t token = row / topk, k = row % topk;
        if ((masks[token] & (uint32_t{1} << k)) == 0) continue;

        auto* words =
            reinterpret_cast<uint32_t*>(gateway.gatewayCombinePtr + static_cast<uint64_t>(row) * gatewayRowBytes);
        // A last-packet marker cannot release an earlier missing packet.
        for (uint32_t p = 0; p < packetCount; ++p) words[p * 128U + 127U] = sequence;
        const uint32_t missing = packetCount / 2U;
        words[missing * 128U + 127U] = sequence - 1U;
        Check(GatewayCombinePacketReadyMask(params, token, uint32_t{1} << k) == 0U, "partial packet row released");
        const size_t beforeIncomplete = posts.size();
        for (activeLane = 0; activeLane < qps; ++activeLane)
            TryProgressHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer);
        for (size_t p = beforeIncomplete; p < posts.size(); ++p) {
            const auto sent = (posts[p].source - gateway.combinePartialSendPtr) / (hidden * 4U);
            Check(published[sent] == masks[sent] && sent != token, "partial packet row reduced or posted");
        }
        words[missing * 128U + 127U] = sequence;
        Check(GatewayCombinePacketReadyMask(params, token, uint32_t{1} << k) == (uint32_t{1} << k),
              "complete packet row not ready");

        published[token] |= uint32_t{1} << k;
        const size_t before = posts.size();
        for (activeLane = 0; activeLane < qps; ++activeLane)
            TryProgressHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer);
        for (size_t p = before; p < posts.size(); ++p) {
            const auto& post = posts[p];
            const uint32_t sent = static_cast<uint32_t>((post.source - gateway.combinePartialSendPtr) / (hidden * 4U));
            Check(published[sent] == masks[sent], "incomplete token was sent");
        }
    }
    for (activeLane = 0; activeLane < qps; ++activeLane) {
        DrainHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer);
        TraceHierarchicalCombineBoundary(gwCommon, params, state[activeLane], activeLane, &timer, 3U);
        TraceHierarchicalCombineBoundary(gwCommon, params, state[activeLane], activeLane, &timer, 4U);
    }
    Check(timer.syncRecords == (qps * 4U), "missing synchronization diagnostics");
    Check(timer.readyRecords >= bs * 4U, "missing/unsolicited ready diagnostics");
    std::vector<uint32_t> sent(bs, 0U);
    for (const auto& post : posts) {
        const uint32_t token = static_cast<uint32_t>((post.source - gateway.combinePartialSendPtr) / (hidden * 4U));
        Check(post.bytes.size() == hidden * 4U, "partial was not FP32");
        Check(token >= bs * (post.qp - 1U) / qps && token < bs * post.qp / qps, "token owned by wrong QP");
        ++sent[token];
        Check(std::memcmp(post.bytes.data(), expectedPartial.data() + token * hidden, post.bytes.size()) == 0,
              "FP32 weighted sum differs");
    }
    for (uint32_t token = 0; token < bs; ++token)
        Check(sent[token] == static_cast<uint32_t>(masks[token] != 0), "missing or duplicate token");
    uint32_t expectedOverlap = 0;
    for (const auto mask : masks)
        if (mask != 0U) expectedOverlap += CombineMaskCount(mask) - 1U;
    Check(overlappedCasts == expectedOverlap, "next expert load did not overlap current Cast");
    uint32_t expectedRowVfs = 0U;
    for (const auto mask : masks) expectedRowVfs += CombineMaskCount(mask);
    Check(rowVfs == expectedRowVfs && scalarFloatReads == 0U, "row fusion skipped a row or read a Scalar weight");
    mainUbLimit = ub.size() - 512U;
    deferInputs = false;
    Check(timer.reduced == posts.size(), "missing reduction timing record");
    // UB can be reused while the NIC owns immutable symmetric GM sources.
    ub.fill(0xCC);
    Deliver();
    if (!posts.empty()) {
        posts.front().source[0] ^= 1U;
        bool failed = false;
        try {
            Deliver();
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Check(failed, "source lifetime negative control failed");
        posts.front().source[0] ^= 1U;
    }
    // Production output consumption must skip poisoned remote BF16 slots and
    // include the FP32 partial once, then the optional shared result once.
    const uint32_t sharedN = SelectSharedExpertPolicy(true, shared ? 1U : 0U, bs).gmm2TileN;
    std::vector<int32_t> sharedReady(INT_CACHELINE, static_cast<int32_t>((hidden + sharedN - 1U) / sharedN));
    sourceParams.workspaceInfo.sharedExpertGmm2TileCounterPtr = reinterpret_cast<uint8_t*>(sharedReady.data());
    sourceParams.workspaceInfo.sharedExpertResultPtr = reinterpret_cast<uint8_t*>(sharedRows.data());
    for (uint32_t physical = 0; physical < t.blockAivNum; ++physical) {
        const uint32_t logical = UnpermuteJobIndex(&t, physical);
        if (logical >= UnpermuteJobCount(&t)) continue;
        events = {};
        MegaMoeUnpermuteBufferConfig buffer{};
        buffer.tokensPerBatch = 2;
        buffer.inputBufferCount = 2;
        buffer.bf16SlotElementCount = hidden;
        buffer.fp32SlotElementCount = hidden;
        buffer.topKWeightsBufferBytes = Ops::Base::CeilAlign(topk * 2U * 4U, 32U);
        TokenUnpermuteConfig context{{logical, UnpermuteJobCount(&t)}, 0U, UnpermuteJobCount(&t), buffer, {}};
        TokenUnpermuteScratch buffers;
        auto actual = CreateTokenUnpermuteBuffers<float, COMBINE_NO_QUANT>(context, hidden, buffers);
        UnpermuteTokens<COMBINE_NO_QUANT, float, false, 256U>(context, srcCommon, sourceParams, buffers, actual);
        for (const auto& type : events)
            for (int live : type) Check(live == 0, "unpermute leaked an event");
    }
    for (size_t i = 0; i < output.size(); ++i)
        Check(output[i].bits == expectedOutput[i].bits, "final output differs / remote weighted twice");
}

void CheckBatchSnapshots()
{
    using namespace ReduceTest;
    for (uint32_t hidden : {1024U, 1056U, 3584U, 8128U, 8192U}) {
        for (uint32_t topk : {1U, 7U, 8U, 32U}) {
            MegaMoeTilingData t{};
            t.h = hidden;
            t.topK = topk;
            t.epWorldSize = 32U;
            t.dispatchRanksPerNode = 16U;
            t.topoType = TOPO_TYPE_MTE_ROCE;
            const uint32_t packets = (hidden * 2U + 507U) / 508U;
            const size_t tokenBytes = topk * packets * 512U;
            std::vector<uint32_t> heap(3U * tokenBytes / 4U, 0xDEADBEEFU);
            Params params{&t, {}};
            params.peermemInfo.gatewayCombinePtr = reinterpret_cast<uint8_t*>(heap.data());
            const uint32_t full = topk == 32U ? 0xffffffffU : (uint32_t{1} << topk) - 1U;
            for (uint32_t sequence : {1U, 129U, 0x80000101U, 0xffffffffU}) {
                t.hierarchicalSequence = sequence;
                for (uint32_t mask : {0U, full, 0x55555555U & full, 0x80000001U & full, 0x1BU & full}) {
                    auto* words = heap.data() + tokenBytes / 4U;
                    for (uint32_t k = 0; k < topk; ++k)
                        for (uint32_t p = 0; p < packets; ++p)
                            words[(k * packets + p) * 128U + 127U] =
                                (mask & (uint32_t{1} << k)) != 0U ? sequence : sequence ^ 0x80000080U;
                    flagSourceBegin = reinterpret_cast<uint8_t*>(words);
                    flagSourceBytes = tokenBytes;
                    flagSourceMask = mask;
                    flagSourcePackets = packets;
                    const uint32_t expectedDmas = CombineMaskCount(mask & ~(mask << 1U));
                    auto check = [&](uint32_t expected) {
                        ub.fill(0xA5);
                        flagDmas = scalarReads = readyVfs = 0U;
                        Check(GatewayCombinePacketReadyMask(params, 1U, mask) == expected,
                              "batch ready mask differs / stale packet accepted");
                        Check(flagDmas == expectedDmas && scalarReads == static_cast<uint32_t>(mask != 0U) &&
                                  readyVfs == static_cast<uint32_t>(mask != 0U),
                              "batch fragmented DMA or repeated Scalar/VF handoff");
                        const size_t snapshotBytes = topk * packets * 32U;
                        // Only the first result word may be written; later UB
                        // belongs to other stages and the reserved QP scratch.
                        for (size_t i = snapshotBytes + 4U; i < snapshotBytes + 64U; ++i)
                            Check(ub[i] == 0xA5, "batch readiness escaped its UB budget");
                        for (size_t i = ub.size() - 512U; i < ub.size(); ++i)
                            Check(ub[i] == 0xA5, "batch readiness overwrote QP scratch");
                        for (const auto& type : events)
                            for (int live : type) Check(live == 0, "batch leaked an event");
                    };
                    check(mask);
                    for (uint32_t k = 0; k < topk; ++k) {
                        const uint32_t bit = uint32_t{1} << k;
                        if ((mask & bit) == 0U) continue;
                        for (uint32_t p : {0U, packets / 2U, packets - 1U}) {
                            auto& marker = words[(k * packets + p) * 128U + 127U];
                            marker = sequence ^ (p == 0U ? 1U : p == packets - 1U ? 0x80000000U : 0x80U);
                            check(mask & ~bit);
                            marker = sequence;
                        }
                    }
                    check(mask);  // A retry must refresh GM after a failed snapshot.
                }
            }
        }
    }
    flagSourceBegin = nullptr;
}

void CheckRouteHistograms()
{
    using namespace ReduceTest;
    auto* ids = reinterpret_cast<uint32_t*>(ub.data());
    auto* bins = reinterpret_cast<uint16_t*>(ub.data() + 8192);
    for (uint32_t experts : {12U, 13U}) {
        for (uint32_t count : {0U, 63U, 64U, 65U, 576U, 768U}) {
            for (uint32_t rank : {0U, 17U, 31U}) {
                const uint32_t first = experts * rank;
                for (uint32_t source = 0; source < 32U; ++source) {
                    std::fill_n(ids, 1024U, first);
                    std::array<uint16_t, 13> expected{};
                    for (uint32_t i = 0; i < count; ++i) {
                        ids[i] = i % 7U == 0U ? 0xffffffffU :
                                 (source == rank ? first + (i % experts) : (31U - rank) * experts);
                        if (ids[i] >= first && ids[i] < first + experts) ++expected[ids[i] - first];
                    }
                    CountRouteSourceVF(ids, bins, count, first);
                    Check(std::equal(bins, bins + experts, expected.begin()), "incorrect per-source route bins");
                }
            }
        }
    }
}

void CheckHostRouteHistograms()
{
    using namespace ReduceTest;
    auto* ids = reinterpret_cast<uint32_t*>(ub.data());
    constexpr uint32_t stride = 256, items = 193;
    auto* bins = reinterpret_cast<uint16_t*>(ids + 16 * stride);
    auto* table = reinterpret_cast<uint32_t*>(bins + 16 * 128);
    for (uint32_t experts : {12U, 13U}) {
        std::vector<uint32_t> expected(experts * 32);
        for (uint32_t host : {1U, 0U}) {
            for (uint32_t s = 0; s < 16; ++s) {
                auto* record = ids + s * stride;
                for (uint32_t j = 0; j < stride; ++j) {
                    record[j] = j % 11 == 0 ? ~0U : (j * 3 + s * host + s) % (experts + 3);
                    if (j < items && record[j] < experts) ++expected[record[j] * 32 + host * 16 + s];
                }
            }
            CountRouteSourceVF(ids, bins, items, 0U, 16U, stride);
            TransposeRouteBinsVF(reinterpret_cast<uint32_t*>(bins), table, host * 16, experts);
        }
        Check(std::equal(expected.begin(), expected.end(), table), "incorrect 16-source histogram transpose");
    }
}

void CheckRoutePlansAndPackets()
{
    using namespace ReduceTest;
    auto* ids = reinterpret_cast<uint32_t*>(ub.data());
    auto* masks = ids + 1024;
    auto* list = reinterpret_cast<int32_t*>(masks + 256);
    auto* count = list + 256;
    for (uint32_t rows : {32U, 65U, 96U, 128U}) {
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t k = 0; k < 6; ++k) ids[row * 6 + k] = row % 3 == 0 ? 0xffffffffU : (row + k) * 7U % 416U;
        RouteHostMasksVF<true>(ids, masks, list, count, rows, 6, 208, 208);
        int32_t n = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t expected = 0;
            for (uint32_t k = 0; k < 6; ++k)
                if (ids[row * 6 + k] >= 208 && ids[row * 6 + k] < 416) expected |= 1U << k;
            Check(masks[row] == expected, "incorrect host mask");
            if (expected) Check(list[n++] == static_cast<int32_t>(row), "unstable compact token order");
        }
        Check(*count == n, "incorrect compact count");
        RouteHostMasksVF<false>(ids, masks, nullptr, nullptr, rows, 6, 208, 208);
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t expected = 0;
            for (uint32_t k = 0; k < 6; ++k)
                if (ids[row * 6 + k] < 208) expected |= 1U << k;
            Check(masks[(rows + 7U) / 8U * 8U + row] == expected, "incorrect local mask for output");
        }
    }
    for (uint32_t words : {120U, 121U, 912U, 1320U}) {
        for (uint32_t i = 0; i < words; ++i) ids[i] = i + 1;
        const uint32_t packets = (words + 119U) / 120U;
        auto* dst = ids + 2048;
        std::fill_n(dst, packets * 128U, ~0U);
        PackDispatchVF(ids, dst, words, packets, 259U);
        for (uint32_t p = 0; p < packets; ++p)
            for (uint32_t j = 0; j < 128; ++j)
                Check(dst[p * 128 + j] == (j == 127                         ? 259U
                                           : j < 120 && p * 120 + j < words ? ids[p * 120 + j]
                                                                            : 0U),
                      "incorrect dispatch payload, padding or sequence");
    }
}

void CheckFusedRowNumerics()
{
    using namespace ReduceTest;
    for (uint32_t hidden : {32U, 64U, 96U, 1024U, 1056U, 3584U, 8128U, 8192U}) {
        ub.fill(0xA5);
        auto* input = reinterpret_cast<bfloat16_t*>(ub.data());
        auto* sum = reinterpret_cast<float*>(ub.data() + 32U * 1024U);
        auto* weight = reinterpret_cast<float*>(ub.data() + 96U * 1024U);
        std::vector<float> reference(hidden);
        auto verify = [&] {
            Check(std::memcmp(sum, reference.data(), hidden * 4U) == 0, "fused row changed FP32 rounding/order");
            const auto* end = reinterpret_cast<uint8_t*>(sum) + hidden * 4U;
            for (size_t i = 0; i < 256U; ++i) Check(end[i] == 0xA5, "masked VF store overwrote sum tail");
        };
        std::fill_n(input, hidden, bfloat16_t(-0.0f));
        weight[0] = 1.0f;
        asc_vf_call<AccumulateGatewayCombineRowVF<true>>(input, sum, weight, hidden);
        std::fill(reference.begin(), reference.end(), -0.0f);
        verify();  // First product must not be added to +0.

        std::fill_n(input, hidden, bfloat16_t(-1.5f));
        asc_vf_call<AccumulateGatewayCombineRowVF<true>>(input, sum, weight, hidden);
        std::fill(reference.begin(), reference.end(), -1.5f);
        verify();
        std::fill_n(input, hidden, bfloat16_t(1.5f));
        weight[0] = std::nextafter(1.0f, 2.0f);
        volatile float product = 1.5f * weight[0];
        for (auto& value : reference) value += product;
        Check(reference[0] != std::fma(1.5f, weight[0], -1.5f), "FMA negative control not sensitive");
        asc_vf_call<AccumulateGatewayCombineRowVF<false>>(input, sum, weight, hidden);
        verify();
        std::mt19937 random(hidden);
        for (uint32_t row = 0U; row < 32U; ++row) {
            weight[0] = static_cast<int>(random() % 2001U) / 997.0f - 1.0f;
            for (uint32_t h = 0; h < hidden; ++h) {
                input[h] = bfloat16_t((static_cast<int>(random() % 8193U) - 4096) / 31.0f);
                volatile float term = static_cast<float>(input[h]) * weight[0];
                reference[h] += term;
            }
            asc_vf_call<AccumulateGatewayCombineRowVF<false>>(input, sum, weight, hidden);
            verify();
        }
    }
}

int main()
{
    try {
        CheckDeferredRows();
        CheckRoutePlansAndPackets();
        CheckRouteHistograms();
        CheckHostRouteHistograms();
        CheckBatchSnapshots();
        CheckFusedRowNumerics();
        for (uint32_t rank : {0U, 21U})
            for (uint32_t pattern = 0; pattern < 5; ++pattern) {
                CheckCase(rank, 32U, 64U, 8U, 3584U, pattern, 0x80000101U, false);
                CheckCase(rank, 7U, 32U, 1U, 1024U, pattern, 0xffffffffU, true);
                CheckCase(rank, 1U, 8U, 32U, 8192U, pattern, 129U, false);
            }
        CheckCase(21U, 96U, 96U, 6U, 5120U, 2U, 0x80000201U, true);
        CheckCase(0U, 256U, 256U, 32U, 1024U, 2U, 257U, true);
        CheckCase(0U, 3U, 8U, 7U, 1024U, 2U, 3U, false);
        CheckCase(16U, 37U, 64U, 7U, 1024U, 2U, 3U, false);
        CheckCase(0U, 3U, 8U, 7U, 8128U, 2U, 129U, false);
        CheckCase(16U, 3U, 8U, 7U, 1056U, 2U, 257U, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
