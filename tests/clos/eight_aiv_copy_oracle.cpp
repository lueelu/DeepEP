// CPU mocks for the production bodies extracted by test_eight_aiv_copy.py.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#define __aicore__
#define __gm__
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/platform.h"
#include "common/udma_schedule.h"

using GM_ADDR = uint8_t*;
using namespace ascend_deepep;
using namespace ascend_deepep::udma;
constexpr int EVENT_ID0 = 0, EVENT_ID2 = 2, EVENT_ID5 = 5, EVENT_ID6 = 6, EVENT_ID7 = 7;
constexpr uint32_t kHeadersOffset = kBatchWqes * (kWqeBytes + 4U * sizeof(uint64_t)), kOutputRecordBytes = 64;
constexpr unsigned kOutputSrcWord=6, kOutputWeightWord=7;
static unsigned list_loads;
void trap() { assert(false); }
namespace AscendC {
enum class HardEvent { V_S, MTE2_MTE3, MTE3_MTE2, MTE3_S, S_MTE2, MTE2_S, S_MTE3 };
inline uint64_t GetSystemCycle()
{
    static uint64_t tick = 1000;
    return ++tick;
}
template <HardEvent>
void SetFlag(int)
{
}
template <HardEvent>
void WaitFlag(int)
{
}
template <class T>
struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
    T GetValue(uint64_t i) const
    {
        return data[i];
    }
    GlobalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
template <class T>
struct LocalTensor {
    T* data{};
    T GetValue(uint64_t i) const
    {
        return data[i];
    }
    void SetValue(uint64_t i, T value)
    {
        data[i] = value;
    }
    LocalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
struct DataCopyExtParams {
    uint32_t blocks, bytes, src_stride, dst_stride, reserved;
};
template <class T>
struct DataCopyPadExtParams {
    bool pad;
    uint8_t left, right;
    T value;
};
template <class T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(copy.blocks == 1);
    if constexpr (sizeof(T) == 4) ++list_loads;
    std::memcpy(dst.data, src.data, copy.bytes);
}
template <class T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    assert(copy.blocks == 1);
    std::memcpy(dst.data, src.data, copy.bytes);
}
}  // namespace AscendC
struct Resource {
    std::vector<uint64_t> ub = std::vector<uint64_t>(platform::kKernelUbBytes / 8, 0);
    template <class T>
    AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset)
    {
        assert(offset % 32 == 0 && offset < platform::kKernelUbBytes);
        return {reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ub.data()) + offset)};
    }
};

static unsigned sender_phase;
static bool pending_rows;
struct NoDispatchOverlapWork {
    void BeforePublish(uint32_t offset) {
        assert(sender_phase == 0 && !pending_rows && offset % 32 == 0);
        sender_phase = 1;
    }
    void operator()(uint32_t) { assert(sender_phase == 2); sender_phase = 3; }
};

using Row = std::tuple<uint32_t, uint32_t, uint32_t>;  // peer, input token, output row
static std::vector<Row> sent;
static uint32_t max_visit, used_visit;
template <uint32_t Capacity, uint32_t Qps, class Trace>
struct DispatchSenderT {
    DispatchTilingData td;
    DispatchSenderT(const DispatchTilingData& t, uint32_t, AscendC::LocalTensor<uint8_t>, GM_ADDR, GM_ADDR, GM_ADDR)
        : td(t)
    {
        sender_phase = 0;
        pending_rows = false;
    }
    uint32_t Count() const
    {
        return 3;
    }
    uint32_t Rank(uint32_t p) const
    {
        return p == 0 ? 1U : (p == 1 ? td.world_size / 2U : td.world_size - 1U);
    }
    void SetTrace(Trace*) {}
    void BeginVisit(uint32_t)
    {
        used_visit = 0;
    }
    uint32_t PutRows(uint32_t p, const int32_t* plan, uint32_t row, uint32_t field, uint32_t n)
    {
        assert(sender_phase == 0);
        pending_rows = true;
        assert(field == 0);
        n = std::min(n, 7U);  // Exercise partial progress, not only full visits.
        used_visit += n;
        assert(used_visit <= td.udma_wqes_per_visit);
        max_visit = std::max(max_visit, used_visit);
        for (uint32_t i = 0; i < n; ++i) sent.emplace_back(Rank(p), plan[i], row + i);
        return n;
    }
    void Flush() { assert(sender_phase == 0); pending_rows = false; }
    void PutPackedScales(uint32_t, uint32_t, uint32_t, uint32_t)
    {
        assert(false);
    }
    void Publish(GM_ADDR, uint64_t, uint32_t, uint32_t) {
        assert(sender_phase == 1 && !pending_rows); sender_phase = 2;
    }
    void PutChannelOutput(uint32_t, uint32_t, uint64_t, uint64_t, uint32_t) { assert(false); }
    void PutChannelField(uint32_t, GM_ADDR, uint64_t, uint64_t, uint32_t)
    {
        assert(false);
    }
    void Quiet() { assert(sender_phase == 4); sender_phase = 5; }
};
void WaitPackedScaleReady(const DispatchTilingData&, uint32_t, uint32_t)
{
    assert(false);
}
template <bool>
void WaitIncoming(GM_ADDR, const DispatchTilingData&, uint32_t, uint32_t, uint32_t)
{
    assert(sender_phase == 3); sender_phase = 4;
}
void aclshmemx_sync_vec_all() { assert(sender_phase == 5); }

// PRODUCTION_TRACE
// PRODUCTION_FUNCTIONS

void CheckCopyAndReduction()
{
    for (unsigned world : {4U, 8U, 16U, 32U, 64U, 128U, 256U})
        for (unsigned channels_count : {3U, 8U, 32U})
            for (unsigned tokens : {0U, 1U, 3U, 65U, 257U, 4099U})
                for (unsigned rank : {0U, world / 2U, world - 1U})
                    for (unsigned first : {0U, 4U, 7U})
                        for (bool weighted : {false, true}) {
                            if (world > 16 && tokens > 257) continue;
                            CombineTilingData td{};
                            td.rank = rank;
                            td.world_size = world;
                            td.block_dim = 8;
                            td.num_channels = channels_count;
                            td.num_tokens = tokens;
                            td.token_capacity = tokens + 32;
                            td.x_row_stride = 64;
                            td.topk_weights_row_stride = 32;
                            td.x_buffer_offset = 32;
                            td.topk_weights_buffer_offset = td.x_buffer_offset + uint64_t(world) * td.token_capacity * td.x_row_stride;
                            td.has_topk_weights = weighted;
                            td.compact_self_rows = 1;
                            td.udma_self_copy_first_aiv = first;
                            const unsigned channel_tokens = (tokens + channels_count - 1) / channels_count;
                            std::vector<int32_t> rp(world * world), cp(world * channels_count), ordinal(tokens, -1);
                            std::vector<unsigned> local_tokens;
                            for (unsigned ch = 0; ch < channels_count; ++ch) {
                                cp[rank * channels_count + ch] = local_tokens.size();  // receive-side EXCLUSIVE prefix
                                unsigned n = 0;
                                for (unsigned t = ch * channel_tokens; t < std::min(tokens, (ch + 1) * channel_tokens); ++t)
                                    if ((t + rank) % 3 == 0 && ch != 2) {
                                        ordinal[t] = n++;
                                        local_tokens.push_back(t);
                                    }
                            }
                            const unsigned base = rank ? 5 : 0;
                            if (rank) rp[(rank - 1) * world + rank] = base;
                            rp[rank * world + rank] = base + local_tokens.size();
                            std::vector<uint8_t> x((base + local_tokens.size() + 1) * 64), weights(x.size());
                            for (unsigned i = 0; i < x.size(); ++i) {
                                x[i] = (i * 17) % 251;
                                weights[i] = (i * 19) % 253;
                            }
                            std::vector<uint8_t> output(td.topk_weights_buffer_offset + uint64_t(world) * td.token_capacity * 32 + 32,
                                                        0xAB);
                            auto expected = output;
                            auto scatter = output;
                            for (unsigned j = 0; j < local_tokens.size(); ++j) {
                                auto compact = uint64_t(rank) * td.token_capacity + j;
                                auto original = uint64_t(rank) * td.token_capacity + local_tokens[j];
                                for (unsigned field = 0; field < (weighted ? 2U : 1U); ++field) {
                                    const unsigned stride = field ? 32 : 64;
                                    const auto offset = field ? td.topk_weights_buffer_offset : td.x_buffer_offset;
                                    const auto& input = field ? weights : x;
                                    std::copy_n(input.data() + (base + j) * stride, stride,
                                                expected.data() + offset + compact * stride);
                                    std::copy_n(input.data() + (base + j) * stride, stride,
                                                scatter.data() + offset + original * stride);
                                }
                            }
                            Resource resource;
                            for (unsigned core = 0; core < 8; ++core)
                                CopyCompactSelf(resource, x.data(), weights.data(), reinterpret_cast<GM_ADDR>(rp.data()),
                                                output.data(), td, core);
                            assert(output == expected);  // Includes untouched remote partitions and guard bytes.
                            AscendC::GlobalTensor<int32_t> channels;
                            channels.SetGlobalBuffer(cp.data());
                            for (unsigned reduce_cores : {8U, 32U, 48U, 64U}) {
                                td.block_dim = reduce_cores;
                                const auto step = (tokens + reduce_cores - 1) / reduce_cores;
                                for (unsigned core = 0; core < reduce_cores; ++core)
                                    for (unsigned t = core * step; t < std::min(tokens, (core + 1) * step); ++t) {
                                        for (unsigned peer : {0U, world / 2U, world - 1U})
                                            if (peer != rank)
                                                assert(ContributorSlot(td, channels, peer, t, 0) ==
                                                       uint64_t(peer) * td.token_capacity + t);
                                        if (ordinal[t] < 0) continue;
                                        const auto slot = ContributorSlot(td, channels, rank, t, ordinal[t]);
                                        const auto old_slot = uint64_t(rank) * td.token_capacity + t;
                                        for (unsigned field = 0; field < (weighted ? 2U : 1U); ++field) {
                                            const unsigned stride = field ? 32 : 64;
                                            const auto offset = field ? td.topk_weights_buffer_offset : td.x_buffer_offset;
                                            assert(std::memcmp(output.data() + offset + slot * stride,
                                                               scatter.data() + offset + old_slot * stride, stride) == 0);
                                        }
                                    }
                            }
                        }
}

void CheckCachedDispatch()
{
    for (unsigned world : {8U, 16U, 32U, 64U, 128U, 256U})
        for (bool opt_in : {false, true})
            for (unsigned cores : {8U, 16U})
                for (unsigned tokens : {0U, 3U, 513U, 2049U})
                    for (unsigned visit : {4U, 8U, 16U, 32U})
                        for (unsigned workers : {1U, 2U}) {
                            DispatchTilingData td{};
                            td.world_size = world;
                            td.cached_channel = opt_in;
                            td.num_tokens = tokens;
                            td.num_channels = 8;
                            td.x_element_bytes = 2;
                            td.udma_workers_per_peer = workers;
                            td.udma_wqes_per_visit = visit;
                            const unsigned width = (tokens + 7) / 8, stride = (width + 7) / 8 * 8;
                            std::vector<int32_t> plan(world * 8 * stride, -1), rp(world * world), cp(world * 8);
                            std::vector<Row> expected;
                            for (unsigned peer : {1U, world / 2U, world - 1U}) {
                                unsigned row = 0;
                                for (unsigned ch = 0; ch < 8; ++ch) {
                                    unsigned index = 0;
                                    for (unsigned t = ch * width; t < std::min(tokens, (ch + 1) * width); ++t)
                                        if (peer != world / 2U && ch != 3 && t % (peer + 1) != 0) {
                                            plan[(peer * 8 + ch) * stride + index++] = t;
                                            expected.emplace_back(peer, t, row++);
                                        }
                                    cp[peer * 8 + ch] = row;  // send-side INCLUSIVE prefix
                                }
                            }
                            std::sort(expected.begin(), expected.end());
                            for (bool cached : {false, true}) {
                                td.cached_mode = cached;
                                sent.clear();
                                list_loads = max_visit = 0;
                                for (unsigned core = 0; core < workers; ++core) {
                                    Resource resource;
                                    DisabledDispatchDeviceTrace trace(0);
                                    if (world >= 128) {
                                        DispatchRemoteImpl<64, 2>(
                                            resource, nullptr, nullptr, reinterpret_cast<GM_ADDR>(plan.data()),
                                            reinterpret_cast<GM_ADDR>(rp.data()), reinterpret_cast<GM_ADDR>(cp.data()),
                                            nullptr, td, core, cores, trace);
                                    } else {
                                        DispatchRemoteImpl<8, 1>(
                                            resource, nullptr, nullptr, reinterpret_cast<GM_ADDR>(plan.data()),
                                            reinterpret_cast<GM_ADDR>(rp.data()), reinterpret_cast<GM_ADDR>(cp.data()),
                                            nullptr, td, core, cores, trace);
                                    }
                                }
                                std::sort(sent.begin(), sent.end());
                                assert(sent == expected);
                                if (cached && opt_in)
                                    assert(list_loads == 0);
                                else if (tokens > 3)
                                    assert(list_loads > 0);
                            }
                        }
}

namespace torch {
constexpr int kBFloat16 = 1;
}
void CheckHostGate()
{
    // Flip every exclusion separately; independent reduce core count is not a gate.
    for (unsigned exclusion = 0; exclusion <= 14; ++exclusion) {
        CombineTilingData tiling{};
        bool use_udma = exclusion != 1, epilogue = exclusion != 2, aggregate = exclusion == 4;
        tiling.udma_token_schedule = exclusion == 3;
        unsigned pipeline_chunk_tokens = exclusion == 5 ? 64 : 0;
        std::optional<int> expanded_x, topk_weights;
        if (exclusion == 13) topk_weights = 1;
        if (exclusion == 6) expanded_x = 1;
        const bool compact_self_enabled = exclusion != 14;
        unsigned world_size = exclusion == 7 ? 16 : 8, resolved_num_aiv = exclusion == 8 ? 16 : 8;
        unsigned num_channels = exclusion == 9 ? 16 : 8;
        struct Tensor {
            int dtype;
            int scalar_type() const
            {
                return dtype;
            }
        } x{exclusion == 10 ? 0 : 1};
        unsigned hidden_bytes = 5632, topk_weights_row_bytes = 32;
        struct {
            unsigned x_row_stride, topk_weights_row_stride;
        } workspace{5632, 32};
        if (exclusion == 11) workspace.x_row_stride += 32;
        if (exclusion == 12) {
            topk_weights = 1;
            workspace.topk_weights_row_stride += 32;
        }
        // PRODUCTION_GATE
        assert(tiling.compact_self_rows == (exclusion == 0 || exclusion == 7 || exclusion == 8 ||
                                          exclusion == 9 || exclusion == 13));
    }
}
void CheckTrace()
{
    Resource resource;
    std::vector<uint64_t> output(3 * 64, 0xdead);
    DispatchDeviceTrace trace(reinterpret_cast<uint64_t>(output.data()));
    trace.Set(DispatchDeviceTrace::Entry, trace.Now());
    trace.Set(DispatchDeviceTrace::Core, 1);
    trace.Mark(DispatchDeviceTrace::BeginDone);
    auto started = trace.Now();
    trace.Add(DispatchDeviceTrace::SrcCycles, started);
    trace.Count(DispatchDeviceTrace::PutRowsCalls);
    trace.Save(resource, 1);
    assert(output[64 + DispatchDeviceTrace::Core] == 1);
    assert(output[64 + DispatchDeviceTrace::BeginDone] > 0);
    assert(output[64 + DispatchDeviceTrace::SrcCycles] == 1);
    assert(output[64 + DispatchDeviceTrace::PutRowsCalls] == 1);
    assert(output[64 + DispatchDeviceTrace::WorkDone] > output[64 + DispatchDeviceTrace::BeginDone]);
    for (unsigned i = 0; i < output.size(); ++i)
        if (i < 64 || i >= 128) assert(output[i] == 0xdead);
    DisabledDispatchDeviceTrace off(0);
    off.Save(resource, 0);
    assert(output[0] == 0xdead);
}
int main()
{
    CheckTrace();
    CheckCopyAndReduction();
    CheckCachedDispatch();
    CheckHostGate();
}
