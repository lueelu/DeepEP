// CPU mocks for the production bodies extracted by test_eight_aiv_copy.py.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#ifdef NDEBUG
#error "CPU oracle assertions must remain enabled"
#endif

#define __aicore__
#define __gm__
#define __ubuf__
#define __simt_vf__
#define __launch_bounds__(n)
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/platform.h"
#include "common/udma_schedule.h"

using GM_ADDR = uint8_t*;
using namespace ascend_deepep;
using namespace ascend_deepep::udma;
constexpr int EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID2 = 2, EVENT_ID5 = 5, EVENT_ID6 = 6, EVENT_ID7 = 7;
static bool audit_copy_events = false;
static bool copy_credit[2]{};
constexpr uint32_t kHeadersOffset = kBatchWqes * (kWqeBytes + 4U * sizeof(uint64_t));
static unsigned list_loads;
static unsigned list_elements;
void trap() { assert(false); }
namespace AscendC {
enum class HardEvent { V_S, MTE2_MTE3, MTE3_MTE2, MTE3_S, S_MTE2, MTE2_S, S_MTE3, MTE2_V, S_V };
template <bool> void SyncAll() {}
inline uint64_t GetSystemCycle()
{
    static uint64_t tick = 1000;
    return ++tick;
}
template <HardEvent Event>
void SetFlag(int id)
{
    if constexpr (Event == HardEvent::MTE3_MTE2) {
        if (audit_copy_events) { assert(id < 2 && !copy_credit[id]); copy_credit[id] = true; }
    }
}
template <HardEvent Event>
void WaitFlag(int id)
{
    if constexpr (Event == HardEvent::MTE3_MTE2) {
        if (audit_copy_events) { assert(id < 2 && copy_credit[id]); copy_credit[id] = false; }
    }
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
    T* GetPhyAddr() { return data; }
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
    if constexpr (sizeof(T) == 4) { ++list_loads; list_elements += copy.bytes / 4; }
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
static bool metadata_packed;
namespace ascend_deepep {
template <uint32_t Capacity, class R, class Sender, class Post>
void PackRemoteChannelMetadata(R&, Sender&, GM_ADDR, GM_ADDR, const DispatchTilingData& td,
                               uint32_t core, uint32_t, Post& post) {
    assert(sender_phase == 0 && !pending_rows && !metadata_packed);
    metadata_packed = true;
    for (uint32_t ch=core%td.udma_workers_per_peer;ch<td.num_channels;ch+=td.udma_workers_per_peer)post(ch);
}
}
static bool in_combine;
static unsigned combine_phase, combine_rows, combine_copies;
static bool combine_expects_copy;
static bool token_combine;
static unsigned token_windows, token_published;
struct dim3 { unsigned x; explicit dim3(unsigned n) : x(n) {} };
static dim3 threadIdx(0);
template <auto Fn, class... Args>
void asc_vf_call(dim3 threads, Args... args) {
    for (threadIdx.x = 0; threadIdx.x < threads.x; ++threadIdx.x) Fn(args...);
}
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
    uintptr_t ub_begin;
    struct PendingIndex {
        const int32_t* index;
        uint32_t peer, row;
        int32_t saved = 0;
        bool retained = false;
    };
    std::vector<PendingIndex> pending;
    DispatchSenderT(const DispatchTilingData& t, uint32_t, AscendC::LocalTensor<uint8_t> ub, GM_ADDR, GM_ADDR, GM_ADDR)
        : td(t), ub_begin(reinterpret_cast<uintptr_t>(ub.data))
    {
        sender_phase = 0;
        pending_rows = false;
        metadata_packed = false;
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
    // Defer index reads until a full WQE batch or explicit flush, as the real
    // sender does. Sharing one batch across peers exercises short UB spans.
    uint32_t PutRowsUb(uint32_t p, const int32_t* plan, uint32_t row, uint32_t field, uint32_t n)
    {
        assert(sender_phase == 0 && !metadata_packed);
        pending_rows = true;
        assert(field == 0);
        n = std::min(n, 7U);  // Exercise partial progress, not only full visits.
        n = std::min(n, 32U - unsigned(pending.size()));
        assert(reinterpret_cast<uintptr_t>(plan) >= ub_begin);
        assert(reinterpret_cast<uintptr_t>(plan + n) <= ub_begin + platform::kKernelUbBytes);
        used_visit += n;
        assert(used_visit <= td.udma_wqes_per_visit);
        max_visit = std::max(max_visit, used_visit);
        for (uint32_t i = 0; i < n; ++i) pending.push_back({plan + i, Rank(p), row + i});
        if (pending.size() == 32) DrainWqes();
        return n;
    }
    // The separate sender oracle tests the production preservation method.
    // Here verify the scheduler calls it before overwriting any live index.
    void PreserveIndices()
    {
        for (auto& entry : pending) if (!entry.retained) {
            entry.saved = *entry.index;
            entry.retained = true;
        }
    }
    void DrainWqes() {
        for (const auto& entry : pending)
            sent.emplace_back(entry.peer, entry.retained ? entry.saved : *entry.index, entry.row);
        pending.clear();
    }
    void Flush() { assert(sender_phase == 0); DrainWqes(); pending_rows = false; }
    void PutPackedScales(uint32_t, uint32_t, uint32_t, uint32_t)
    {
        assert(false);
    }
    void Publish(GM_ADDR, uint64_t, uint32_t, uint32_t) {
        assert(sender_phase == 1 && !pending_rows); sender_phase = 2;
    }
    void PutChannelOutput(uint32_t, uint32_t, uint64_t, uint64_t, uint32_t) {
        assert(sender_phase == 0 && metadata_packed);
    }
    void PutChannelField(uint32_t, GM_ADDR, uint64_t, uint64_t, uint32_t)
    {
        assert(sender_phase == 0 && metadata_packed);
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
void aclshmemx_sync_vec_all() { assert(in_combine ? combine_phase == 5 : sender_phase == 5); }

bool CombineSimtWeightsEnabled(const CombineTilingData&) { return false; }
void CombineRemoteWeights(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const CombineTilingData&, uint32_t) { assert(false); }
void WaitAggregateReady(const CombineTilingData&, uint32_t, uint32_t) { assert(combine_phase == 0); }
template <bool>
void WaitIncoming(GM_ADDR, const CombineTilingData&, uint32_t, uint32_t, uint32_t) {
    if (token_combine) {
        assert(combine_copies == 1 && combine_phase == (token_windows ? 5U : 3U));
    } else {
        assert(combine_phase == (combine_expects_copy ? 3U : 2U));
    }
    combine_phase = 4;
}
template <class R>
void CombineSelfCopy(R&, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                     const CombineTilingData&, int32_t) {
    assert(combine_phase == 2); // Fail if even one copy precedes final publication.
    if (token_combine) assert(token_published == token_windows);
    ++combine_copies;
    combine_phase = 3;
}
template <uint32_t Capacity, uint32_t Qps, class Trace>
struct CombineSenderT {
    CombineSenderT(const CombineTilingData&, uint32_t, uint8_t*, GM_ADDR, GM_ADDR, GM_ADDR,
                   uint32_t, uint32_t, uint32_t, uint32_t, bool) { combine_phase = 0; }
    uint32_t Count() const { return 1; }
    uint32_t Rank(uint32_t) const { return 1; }
    void SetTrace(Trace*) {}
    void BeginVisit(uint32_t) { assert(combine_phase == 0); }
    template <uint32_t Phase>
    uint32_t PutSelectedRows(uint32_t, int32_t*, uint32_t, uint32_t, uint32_t count) {
        assert(combine_phase == 0);
        combine_rows += count;
        return count;
    }
    void Flush() { assert(combine_phase == 0); }
    void FlushForIndex() { Flush(); }
    void Publish(GM_ADDR, uint64_t, uint32_t, uint32_t) { assert(combine_phase == 0); combine_phase = 2; }
    void Quiet() { assert(combine_phase == 4); combine_phase = 5; }
};

constexpr uint32_t kCombineSenderBytes = kHeadersOffset + kMaxPeers * (2U * kWqeBytes + sizeof(uint64_t));
struct CombineSender {
    unsigned first, count;
    CombineSender(const CombineTilingData& td, uint32_t, uint8_t*, GM_ADDR, GM_ADDR, GM_ADDR,
                  uint32_t start, uint32_t, uint32_t, uint32_t)
        : first(start), count(std::min(kMaxPeers, td.world_size - 1 - start)) { combine_phase = 0; }
    unsigned Count() const { return count; }
    unsigned Rank(unsigned p) const { return first + p + 1; }
    template <uint32_t Phase>
    unsigned PutSelectedRows(unsigned, int32_t*, unsigned, unsigned, unsigned count) {
        assert(combine_phase == 0 && combine_copies == 0);
        combine_rows += count;
        return count;
    }
    void Flush() { assert(combine_phase == 0); }
    void Publish(GM_ADDR, uint64_t, uint32_t, uint32_t) {
        assert(combine_phase == 0);
        ++token_published;
        combine_phase = 2;
    }
    void Quiet() {
        assert(combine_phase == (token_published == token_windows ? 3U : 2U));
        combine_phase = 5;
    }
};

// PRODUCTION_TRACE
// PRODUCTION_FUNCTIONS

template <unsigned Phase>
void CheckTokenCombineOrdering() {
    for (unsigned world : {2U, 16U, 64U, 128U, 256U})
        for (unsigned rows : {0U, 3U, 65U})
            for (bool weighted : {false, true}) {
                CombineTilingData td{};
                td.world_size = world;
                td.block_dim = 8;
                td.udma_wqes_per_visit = 8;
                td.has_topk_weights = weighted;
                std::vector<int32_t> indices(world * rows + 32), prefix(world * world);
                for (unsigned peer = 0; peer < world; ++peer) prefix[peer * world] = (peer + 1) * rows;
                for (unsigned core = 0; core < td.block_dim; ++core) {
                    Resource resource;
                    token_combine = true;
                    token_published = combine_copies = combine_rows = 0;
                    token_windows = core < TokenWorkers(td.block_dim) ? (world - 2) / kMaxPeers + 1 : 0;
                    combine_phase = 2; // Helpers own no payload windows.
                    CombineTokenRemote<Phase>(resource, nullptr, nullptr,
                        reinterpret_cast<GM_ADDR>(indices.data()), reinterpret_cast<GM_ADDR>(prefix.data()),
                        nullptr, nullptr, td, core);
                    assert(combine_copies == unsigned(Phase != 1));
                    token_combine = false;
                }
            }
}

template <unsigned Phase>
void CheckCombineOrdering()
{
    for (unsigned rows : {0U, 3U, 65U, 513U})
        for (bool weighted : {false, true})
            for (bool legacy_after : {false, true}) {
                CombineTilingData td{};
                td.world_size = 2;
                td.block_dim = 8;
                td.udma_workers_per_peer = 1;
                td.udma_wqes_per_visit = 8;
                td.pipeline_chunk_tokens = 32;
                td.has_topk_weights = weighted;
                td.udma_self_copy_after_submit = legacy_after;
                std::vector<int32_t> indices(rows + 8), prefix{0, 0, int32_t(rows), 0};
                Resource resource;
                DisabledDispatchDeviceTrace trace(0);
                in_combine = true;
                combine_rows = combine_copies = 0;
                combine_expects_copy = Phase == 0 || Phase == 2;
                CombineRemoteImpl<64, Phase, 2>(resource, nullptr, nullptr,
                    reinterpret_cast<GM_ADDR>(indices.data()), reinterpret_cast<GM_ADDR>(prefix.data()),
                    nullptr, nullptr, td, 0, trace);
                assert(combine_rows == rows * (weighted ? 2U : 1U));
                assert(combine_copies == unsigned(combine_expects_copy));
                assert(combine_phase == (Phase == 1 ? 0U : 5U));
                in_combine = false;
            }
}

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
                            for (bool db : {false, true}) {
                                td.combine_self_double_buffer = db;
                                std::fill(output.begin(), output.end(), 0xAB);
                                for (unsigned core = 0; core < 8; ++core)
                                    CopyCompactSelf(resource, x.data(), weights.data(), reinterpret_cast<GM_ADDR>(rp.data()),
                                                    output.data(), td, core);
                                assert(output == expected);  // Includes untouched remote partitions and guard bytes.
                            }
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
                for (unsigned tokens : {0U, 3U, 513U, 2049U, 4099U})
                    for (unsigned visit : {4U, 8U, 16U, 31U, 32U, 257U})
                        for (unsigned workers : {1U, 2U}) for (bool metadata : {false, true}) {
                            DispatchTilingData td{};
                            td.world_size = world;
                            td.cached_channel = opt_in;
                            td.num_tokens = tokens;
                            td.num_channels = 8;
                            td.x_element_bytes = 2;
                            td.udma_workers_per_peer = workers;
                            td.udma_wqes_per_visit = visit;
                            td.channel_metadata = metadata;
                            td.has_topk_weights = metadata;
                            td.num_topk = 8;
                            td.metadata_direct_output = opt_in;
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
                                list_loads = list_elements = max_visit = 0;
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
                                if (cached && opt_in) {
                                    // Independent whole-channel, per-peer visit order;
                                    // UB refill must not truncate a visit or reorder peers.
                                    std::vector<Row> ordered;
                                    unsigned expected_loads = 0;
                                    for (unsigned core = 0; core < workers; ++core)
                                        for (unsigned ch = core; ch < 8; ch += workers) {
                                            unsigned cursor[3]{}, counts[3]{}, starts[3]{};
                                            const unsigned peers[3] = {1U, world / 2U, world - 1U};
                                            for (unsigned p = 0; p < 3; ++p) {
                                                starts[p] = ch ? cp[peers[p] * 8 + ch - 1] : 0;
                                                counts[p] = cp[peers[p] * 8 + ch] - starts[p];
                                                expected_loads += (counts[p] + 255) / 256;
                                            }
                                            while (cursor[0] < counts[0] || cursor[1] < counts[1] || cursor[2] < counts[2])
                                                for (unsigned p = 0; p < 3; ++p) {
                                                    const unsigned stop = std::min(counts[p], cursor[p] + visit);
                                                    for (; cursor[p] < stop; ++cursor[p])
                                                        ordered.emplace_back(peers[p], plan[(peers[p] * 8 + ch) * stride + cursor[p]],
                                                                             starts[p] + cursor[p]);
                                                }
                                        }
                                    assert(sent == ordered);
                                    assert(list_elements == expected.size());
                                    assert(list_loads == expected_loads);
                                } else if (tokens > 3)
                                    assert(list_loads > 0);
                                std::sort(sent.begin(), sent.end());
                                assert(sent == expected);
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
        for (const char* wqe_fast : {static_cast<const char*>(nullptr), "0", "1"})
            for (const char* self_db : {static_cast<const char*>(nullptr), "0", "1"}) {
                // PRODUCTION_OPT_GATES
                assert(tiling.combine_wqe_fast == (wqe_fast && wqe_fast[0]=='1' &&
                       (exclusion==0 || exclusion>=7)));
                assert(tiling.combine_self_double_buffer ==
                       (self_db && self_db[0]=='1' && tiling.compact_self_rows));
            }
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
void CheckCopyTiles()
{
    constexpr unsigned tile = 64U * 1024U;
    for (bool db : {false, true})
        for (unsigned bytes : {0U, 1U, 31U, tile-1, tile, tile+1, 2*tile, 2*tile+17, 5*tile+31}) {
            Resource resource;
            std::fill(resource.ub.begin(), resource.ub.end(), 0x12345678);
            std::vector<uint8_t> input(bytes+64), output(bytes+64, 0xAB);
            for (unsigned i=0;i<input.size();++i) input[i]=(i*17)%251;
            auto expected=output;
            std::copy_n(input.data()+3, bytes, expected.data()+7);
            audit_copy_events=true;
            // Reuse the same slots and event IDs for the next field/call.
            for (unsigned repeat=0;repeat<3;++repeat) {
                CopySelfField(resource,input.data()+3,output.data()+7,bytes,db);
                assert(output==expected && !copy_credit[0] && !copy_credit[1]);
            }
            audit_copy_events=false;
            for (unsigned i=0;i<kBatchWqes*kWqeBytes/8;++i) assert(resource.ub[i]==0x12345678);
        }
}

int main()
{
    CheckCopyTiles();
    CheckTrace();
    CheckCopyAndReduction();
    CheckCachedDispatch();
    CheckCombineOrdering<0>();
    CheckCombineOrdering<1>();
    CheckCombineOrdering<2>();
    CheckCombineOrdering<3>();
    CheckTokenCombineOrdering<0>();
    CheckTokenCombineOrdering<1>();
    CheckTokenCombineOrdering<2>();
    CheckHostGate();
}
