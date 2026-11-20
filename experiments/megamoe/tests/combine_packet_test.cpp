// Copyright (c) 2026, Lu Lu
// Deferred MTE2/MTE3 primitives execute production packing and ring submission.
// Guard source extents, UB reuse, event lifetimes and exact destination bytes.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#define __aicore__
#define __gm__
using bfloat16_t = uint16_t;
struct fp8_e4m3fn_t {};
struct fp8_e5m2_t {};
constexpr int AIC = 0, AIV = 1, g_coreType = AIV;
constexpr uint32_t EVENT_ID0 = 0U;
namespace PacketTest {
alignas(512) std::array<uint8_t, 256U * 1024U> ub{};
std::array<std::array<bool, 8>, 10> events{};
std::vector<std::function<void()>> loads;
struct Write {
    uint8_t* target;
    const uint8_t* source;
    std::vector<uint8_t> expected;
};
std::vector<Write> queued;
std::array<std::vector<Write>, 8> inFlight;
const uint8_t* sourceBegin = nullptr;
const uint8_t* sourceEnd = nullptr;
const uint8_t* metaBegin = nullptr;
const uint8_t* metaEnd = nullptr;
uint32_t stores = 0U, sends = 0U, mte3ScalarWaits = 0U;
void CheckWritable(const void* pointer, uint32_t bytes)
{
    const auto* first = static_cast<const uint8_t*>(pointer);
    for (const auto& pending : inFlight)
        for (const auto& write : pending)
            assert(first + bytes <= write.source || first >= write.source + write.expected.size());
}
void Finish(std::vector<Write>& writes)
{
    for (const auto& write : writes) {
        assert(std::equal(write.expected.begin(), write.expected.end(), write.source));
        std::copy(write.expected.begin(), write.expected.end(), write.target);
    }
    writes.clear();
}
}  // namespace PacketTest
namespace AscendC {
using TEventID = uint32_t;
enum class TPosition { VECCALC, VECIN };
enum class PaddingMode { Normal, Compact };
enum class HardEvent { MTE2_S, S_MTE3, MTE2_MTE3, MTE3_MTE2, MTE3_S, MTE2_V, V_MTE3 };
template <typename T>
struct LocalTensor {
    T* data = nullptr;
    LocalTensor() = default;
    explicit LocalTensor(T* pointer) : data(pointer) {}
    LocalTensor(TPosition, uint32_t offset, uint32_t count)
    {
        assert(offset % 32U == 0U && offset + count * sizeof(T) <= PacketTest::ub.size());
        data = reinterpret_cast<T*>(PacketTest::ub.data() + offset);
    }
    LocalTensor operator[](uint64_t offset) const
    {
        return LocalTensor(data + offset);
    }
    T GetValue(uint32_t index) const
    {
        return data[index];
    }
    void SetValue(uint32_t index, T value)
    {
        assert(PacketTest::loads.empty());
        PacketTest::CheckWritable(data + index, sizeof(T));
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
    void SetGlobalBuffer(T* pointer)
    {
        data = pointer;
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
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride, reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool enabled;
    uint32_t left, right;
    T value;
};
template <typename T, PaddingMode Mode = PaddingMode::Normal>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T> pad)
{
    using namespace PacketTest;
    assert(reinterpret_cast<uintptr_t>(dst.data) % 32U == 0U && copy.dstStride == 0U && pad.left == 0U);
    const uint32_t block =
        Mode == PaddingMode::Normal ? (copy.blockLen + pad.right * sizeof(T) + 31U) / 32U * 32U : copy.blockLen;
    const uint32_t bytes = (copy.blockCount * block + 31U) / 32U * 32U;
    CheckWritable(dst.data, bytes);
    assert(reinterpret_cast<uint8_t*>(dst.data) + bytes <= ub.data() + ub.size());
    for (uint32_t i = 0U; i < copy.blockCount; ++i) {
        const auto* first = reinterpret_cast<const uint8_t*>(src.data) + i * (copy.blockLen + copy.srcStride);
        assert((first >= sourceBegin && first + copy.blockLen <= sourceEnd) ||
               (metaBegin && first >= metaBegin && first + copy.blockLen <= metaEnd));
    }
    loads.push_back([=]() {
        auto* target = reinterpret_cast<uint8_t*>(dst.data);
        std::memset(target, 0xCE, bytes);
        for (uint32_t i = 0U; i < copy.blockCount; ++i) {
            std::memcpy(target + i * block,
                        reinterpret_cast<const uint8_t*>(src.data) + i * (copy.blockLen + copy.srcStride),
                        copy.blockLen);
            if (pad.enabled)
                for (uint32_t j = copy.blockLen; j < block; j += sizeof(T))
                    std::memcpy(target + i * block + j, &pad.value, sizeof(T));
        }
    });
}
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    DataCopyPad(dst, src, {1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    using namespace PacketTest;
    assert(loads.empty() && copy.blockCount == 1U);
    const auto* begin = reinterpret_cast<const uint8_t*>(src.data);
    queued.push_back({reinterpret_cast<uint8_t*>(dst.data), begin, {begin, begin + copy.blockLen}});
    ++sends;
}
template <HardEvent E>
void SetFlag(TEventID id)
{
    using namespace PacketTest;
    auto& event = events[static_cast<uint32_t>(E)][id];
    assert(!event);
    event = true;
    if constexpr (E == HardEvent::MTE3_MTE2) {
        assert(inFlight[id].empty());
        inFlight[id].swap(queued);
    }
}
template <HardEvent E>
void WaitFlag(TEventID id)
{
    using namespace PacketTest;
    auto& event = events[static_cast<uint32_t>(E)][id];
    assert(event);
    event = false;
    if constexpr (E == HardEvent::MTE2_S || E == HardEvent::MTE2_MTE3) {
        for (auto& load : loads) load();
        loads.clear();
    } else if constexpr (E == HardEvent::MTE3_MTE2) {
        Finish(inFlight[id]);
    } else if constexpr (E == HardEvent::MTE3_S) {
        ++mte3ScalarWaits;
        Finish(queued);
        for (auto& writes : inFlight) Finish(writes);
    }
}
uint32_t GetSubBlockIdx()
{
    return 1U;
}
void WriteGmByPassDCache(int32_t* ptr, int32_t value)
{
    assert(PacketTest::queued.empty());
    ++PacketTest::stores;
    *ptr = value;
}
}  // namespace AscendC
#include "../csrc_ep32/kernels/mega_moe/common/mega_moe_peermem.hpp"
namespace MegaMoeImpl {
using namespace AscendC;
constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
struct MoeStageCommonConfig {
    uint32_t tokenHiddenDim, rankId;
};
struct Params {
    MegaMoeTilingData* tilingData;
    PeermemInfo peermemInfo;
    struct {
        uint8_t* gmm2MmadResPtr = nullptr;
        uint8_t* metaInfoPtr = nullptr;
    } workspaceInfo{};
};
uint8_t* windows[32]{};
uint8_t* GetRankWinAddrWithOffset(uint32_t rank, uint64_t offset)
{
    assert(windows[rank]);
    return windows[rank] + offset;
}
template <HardEvent E, uint32_t Id>
void SyncFuncStatic()
{
    SetFlag<E>(Id);
    WaitFlag<E>(Id);
}
namespace Mxfp8 {
template <uint8_t, typename T, typename... Args>
void QuantMxFp8(Args&&...)
{
    assert(false);
}
}  // namespace Mxfp8
namespace CombineImpl {
/* PRODUCTION_ROUTES */
/* PRODUCTION_SENDER */
}  // namespace CombineImpl
}  // namespace MegaMoeImpl
#include "../csrc_ep32/kernels/mega_moe/stage/mega_moe_combine_packet.hpp"
namespace MegaMoeImpl {
/* PRODUCTION_CONFIG */
/* PRODUCTION_INIT */
/* PRODUCTION_DRAIN */
/* PRODUCTION_SEND */
/* PRODUCTION_PRELOAD */
struct ExpertLoopState {
    uint32_t expertIdx;
    uint64_t globalTokenStartIndex;
};
struct WorkRange {
    uint32_t start, count;
};
namespace AscendTimer {
enum class CombineRowPhase { Lifetime = 1, Issue = 2, ReuseWait = 3 };
}
struct AscendTimerDevice {
    uint32_t begins = 0U, ends = 0U;
    int64_t CurrentWave() const
    {
        return 0;
    }
    int BeginCombineRowAtWave(uint32_t expert, uint32_t row, uint32_t slot, AscendTimer::CombineRowPhase phase, int64_t)
    {
        return BeginCombineRow(expert, row, slot, phase);
    }
    int BeginCombineRow(uint32_t, uint32_t, uint32_t, AscendTimer::CombineRowPhase)
    {
        return begins++;
    }
    void EndCombineRow(int handle)
    {
        assert(handle >= 0);
        ++ends;
    }
};
/* PRODUCTION_PIPELINE */
}  // namespace MegaMoeImpl
using namespace MegaMoeImpl;

void CheckCase(uint32_t hidden, uint32_t sequence, uint32_t rank)
{
    using namespace PacketTest;
    ub.fill(0xA5);
    events = {};
    stores = sends = mte3ScalarWaits = 0U;
    MegaMoeTilingData t{};
    t.h = hidden;
    t.bf16Bs96 = hidden == 5120U;
    t.topK = 4U;
    t.hierarchicalSequence = sequence;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    std::vector<uint8_t> local(1024U * 1024U, 0xA5), peer(local.size(), 0xA5);
    const uint32_t host = rank / 16U * 16U;
    windows[host] = local.data();
    windows[host + 7U] = peer.data();
    PeermemInfo memory;
    memory.rankSyncInWorldPtr = local.data();
    memory.combineSendPtr = local.data();
    memory.gatewayCombinePtr = local.data() + 256U * 1024U;
    Params params{&t, memory};
    MoeStageCommonConfig common{hidden, rank};
    WaveCombineScratch scratch;
    const auto config = InitWaveCombineBuffers<COMBINE_NO_QUANT>(common, scratch, UsePacketizedGatewayCombine(&t));
    assert(config.rowBufferCount == 6U);
    const uint32_t packedBytes = (hidden * 2U + 507U) / 508U * 512U;
    assert(config.slotStrideBytes == packedBytes);
    assert(WAVE_COMBINE_UB_BASE + config.slotStrideBytes * config.rowBufferCount <= WAVE_COMBINE_UB_LIMIT);
    const uint32_t count = 19U;
    std::vector<uint16_t> rows(count * hidden);
    for (uint32_t i = 0U; i < rows.size(); ++i) rows[i] = (i * 131U + i / 7U) & 0xFFFFU;
    GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(rows.data());
    CrossExpertCombinePipeline pipeline;
    AscendTimerDevice timer;
    std::vector<int32_t> metadata(count * META_INFO_SIZE);
    params.workspaceInfo.gmm2MmadResPtr = reinterpret_cast<uint8_t*>(rows.data());
    params.workspaceInfo.metaInfoPtr = reinterpret_cast<uint8_t*>(metadata.data());
    metaBegin = reinterpret_cast<uint8_t*>(metadata.data());
    metaEnd = metaBegin + metadata.size() * sizeof(int32_t);
    assert(pipeline.Enabled(params));
    for (uint32_t i = 0U; i < count; ++i) {
        const bool gateway = i % 3U != 1U;
        // Include local gateway, another same-host gateway, and direct peer output.
        const uint32_t destination = i % 3U == 0U ? host : host + 7U;
        auto meta = scratch.metaInfoTensor[i * META_INFO_SIZE];
        meta.SetValue(RANK_ID, gateway ? destination ^ 16U : destination);
        meta.SetValue(TOKEN_ID, i / 4U);
        meta.SetValue(TOPK_INDEX, i % 4U);
        sourceBegin = reinterpret_cast<uint8_t*>(rows.data() + i * hidden);
        sourceEnd = sourceBegin + hidden * 2U;

        std::copy(meta.data, meta.data + META_INFO_SIZE, metadata.data() + i * META_INFO_SIZE);
        pipeline.IssueSlice(common, params, {i / 2U, i / 2U * 2U}, i % 2U, {0U, 1U}, timer);
        // Two rows from different calls remain in flight simultaneously.
        if (i == 1U) assert(!inFlight[0].empty() && !inFlight[1].empty());
        if (t.bf16Bs96 && i == 3U)
            for (uint32_t slot = 0; slot < 4U; ++slot) assert(!inFlight[slot].empty());
        if (i % 7U == 5U) {
            pipeline.Drain(timer);
            for (const auto& pending : inFlight) assert(pending.empty());
            CheckWritable(ub.data(), ub.size());
            ub.fill(0xA5);  // activation/Dispatch can reuse ALL stage UB
        }
        assert(sends == i + 1U);  // one MTE3 submission per row, never per packet
        if (true) assert(stores == 0U && mte3ScalarWaits == 0U);
    }

    pipeline.Drain(timer);
    pipeline.Drain(timer);  // no duplicate event waits on empty drain
    assert(timer.begins == count * 3U && timer.ends == timer.begins);

    assert(loads.empty() && queued.empty());
    for (const auto& writes : inFlight) assert(writes.empty());
    for (const auto& type : events)
        for (bool event : type) assert(!event);
    for (uint32_t i = 0U; i < count; ++i) {
        const bool gateway = i % 3U != 1U;
        const uint8_t* base = (i % 3U == 0U ? local.data() : peer.data()) + (gateway ? 256U * 1024U : 0U);
        const bool packet = gateway;
        const uint32_t stride = packet ? packedBytes : hidden * 2U;
        const auto* bytes = base + i * stride;
        const auto* expected = reinterpret_cast<const uint8_t*>(rows.data() + i * hidden);
        for (uint32_t byte = 0U; byte < hidden * 2U; ++byte)
            assert(bytes[packet ? byte / 508U * 512U + byte % 508U : byte] == expected[byte]);
        if (packet) {
            for (uint32_t p = 0U; p < packedBytes / 512U; ++p) {
                uint32_t marker;
                std::memcpy(&marker, bytes + p * 512U + 508U, 4U);
                assert(marker == sequence);
            }
        }
    }
    const uint32_t ringEnd = WAVE_COMBINE_UB_BASE + config.rowBufferCount * config.slotStrideBytes;
    for (uint32_t i = ringEnd; i < WAVE_COMBINE_UB_LIMIT; ++i) assert(ub[i] == 0xA5);
    metaBegin = metaEnd = nullptr;
    ub.fill(0xCC);  // a subsequent stage may safely take every UB slot
}
int main()
{
    static_assert(WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT == 1U);
    for (uint32_t hidden : {1024U, 3584U, 8128U, 8192U})
        for (uint32_t sequence : {1U, 129U, 0xffffffffU})
            for (uint32_t rank : {0U, 30U}) CheckCase(hidden, sequence, rank);
    for (uint32_t hidden = 1024U; hidden <= 8192U; hidden += 32U) CheckCase(hidden, 0x80000101U, 14U);
}
