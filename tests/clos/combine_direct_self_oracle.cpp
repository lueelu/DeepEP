// Runs the production reduction with synchronous CPU DMA/vector stand-ins.
// Verifies addresses, values and balanced events, not device concurrency.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>
#define __aicore__
#define __gm__
#define GM_ADDR uint8_t*
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/platform.h"

template <bool Bfloat>
struct Word {
    uint16_t bits;
};
using bfloat16_t = Word<true>;
using half = Word<false>;
template <bool B>
float Decode(Word<B> v)
{
    if constexpr (B) {
        uint32_t bits = uint32_t(v.bits) << 16;
        float value;
        std::memcpy(&value, &bits, 4);
        return value;
    } else {
        unsigned exp = (v.bits >> 10) & 31, frac = v.bits & 1023;
        if (exp == 31) return NAN;
        float value = exp ? std::ldexp(float(1024 + frac), int(exp) - 25) : std::ldexp(float(frac), -24);
        return v.bits & 0x8000 ? -value : value;
    }
}
template <bool B>
Word<B> Encode(float value)
{
    if constexpr (B) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        return {uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)};
    } else {
        // Test values are bounded, exact multiples of 1/16; no subnormals.
        if (value == 0) return {0};
        int exp;
        float mantissa = std::frexp(std::fabs(value), &exp);
        unsigned sig = unsigned(std::nearbyint(mantissa * 2048));
        if (sig == 2048) {
            sig = 1024;
            ++exp;
        }
        assert(exp >= -13 && exp <= 16);
        return {uint16_t((value < 0 ? 0x8000 : 0) | ((exp + 14) << 10) | (sig - 1024))};
    }
}
constexpr int PIPE_V = 0;
constexpr int EVENT_ID0 = 0, EVENT_ID1 = 1, EVENT_ID2 = 2, EVENT_ID3 = 3, EVENT_ID5 = 5, EVENT_ID6 = 6;
static int current_core, core_count;
int get_block_idx()
{
    return current_core;
}
int get_subblockdim()
{
    return 1;
}
int get_subblockid()
{
    return 0;
}
int get_block_num()
{
    return core_count;
}
namespace AscendC {
enum class HardEvent { V_MTE2, MTE3_V, MTE2_V, V_MTE3, MTE2_S, S_MTE2 };
static int events[6][8]{};
template <HardEvent E>
void SetFlag(int id)
{
    assert(events[int(E)][id]++ == 0);
}
template <HardEvent E>
void WaitFlag(int id)
{
    assert(--events[int(E)][id] == 0);
}
template <int>
void PipeBarrier()
{
}
template <typename T>
struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
    T GetValue(uint64_t n) const
    {
        assert(data);
        return data[n];
    }
    GlobalTensor operator[](uint64_t n) const
    {
        return {data + n};
    }
};
template <typename T>
struct LocalTensor {
    T* data{};
    T GetValue(uint64_t n) const
    {
        return data[n];
    }
    LocalTensor operator[](uint64_t n) const
    {
        return {data + n};
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        return {reinterpret_cast<U*>(data)};
    }
};
struct DataCopyExtParams {
    uint32_t blocks, bytes, src_gap, dst_gap, reserved;
    DataCopyExtParams(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
        : blocks(a), bytes(b), src_gap(c), dst_gap(d), reserved(e)
    {
    }
};
template <typename T>
struct DataCopyPadExtParams {
    bool pad;
    uint8_t left, right;
    T value;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams p, DataCopyPadExtParams<T> pad)
{
    assert(p.blocks == 1 && uintptr_t(dst.data) % 32 == 0);
    std::memcpy(dst.data, src.data, p.bytes);
    if (pad.pad) std::fill(dst.data + p.bytes / sizeof(T), dst.data + p.bytes / sizeof(T) + pad.right, pad.value);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p)
{
    assert(p.blocks == 1 && uintptr_t(src.data) % 32 == 0);
    std::memcpy(dst.data, src.data, p.bytes);
}
void Duplicate(LocalTensor<float> out, float value, unsigned n)
{
    std::fill(out.data, out.data + n, value);
}
enum class RoundMode { CAST_NONE, CAST_RINT };
template <bool B>
void Cast(LocalTensor<float> out, LocalTensor<Word<B>> in, RoundMode, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) out.data[i] = Decode(in.data[i]);
}
template <bool B>
void Cast(LocalTensor<Word<B>> out, LocalTensor<float> in, RoundMode, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) out.data[i] = Encode<B>(in.data[i]);
}
void Add(LocalTensor<float> out, LocalTensor<float> a, LocalTensor<float> b, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) out.data[i] = a.data[i] + b.data[i];
}
}  // namespace AscendC
namespace ascend_deepep {
struct UbWorkspace {
    alignas(32) uint8_t bytes[platform::kKernelUbBytes]{};
    template <typename T>
    AscendC::LocalTensor<T> GetBufferByByte(unsigned n)
    {
        assert(n < sizeof(bytes));
        return {reinterpret_cast<T*>(bytes + n)};
    }
};
namespace udma {
struct DisabledDispatchDeviceTrace {
    static constexpr bool enabled = false;
    enum { Version, Operation, Rank, Core, Entry, InitDone, BeginDone, MetadataDone, OutputDone, HandleDone };
    explicit DisabledDispatchDeviceTrace(uint64_t) {}
    void Set(int, uint64_t) {}
    void Mark(int) {}
    uint64_t Now()
    {
        return 0;
    }
};
template <class... A>
void Begin(A...)
{
    assert(false);
}
inline unsigned TokenWorkers(unsigned n)
{
    return n;
}
template <unsigned, class... A>
void CombineTokenRemote(A...)
{
    assert(false);
}
template <unsigned, class... A>
void CombineRemote(A...)
{
    assert(false);
}
}  // namespace udma
}  // namespace ascend_deepep
// PRODUCTION_CONSTANTS
// PRODUCTION_REDUCTION

float Value(unsigned peer, unsigned token, unsigned col, unsigned generation)
{
    return float(int((peer * 13 + token * 7 + col * 3 + generation * 5) % 31) - 15) / 16;
}
template <bool B>
void Check(unsigned world, unsigned rank, unsigned tokens, unsigned hidden, unsigned channels, unsigned topk,
           unsigned reduce_cores, unsigned route_mode)
{
    using T = Word<B>;
    ascend_deepep::CombineTilingData td{};
    td.world_size = world;
    td.rank = rank;
    td.num_tokens = tokens;
    td.num_channels = channels;
    td.block_dim = reduce_cores;
    td.hidden = hidden;
    td.x_element_bytes = 2;
    td.num_topk = topk;
    td.token_capacity = tokens + 7;
    td.has_topk_weights = topk != 0;
    td.x_row_stride = (hidden * 2 + 31) / 32 * 32;
    td.x_buffer_offset = 32;
    td.topk_weights_buffer_offset = td.x_buffer_offset + uint64_t(world) * td.token_capacity * td.x_row_stride;
    std::vector<int32_t> heads(tokens * world, -1), starts(world * channels), ends(world * channels),
        ranks(world * world);
    std::vector<std::vector<unsigned>> routes(world);
    const unsigned channel_tokens = (tokens + channels - 1) / channels;
    for (unsigned peer = 0; peer < world; ++peer) {
        for (unsigned ch = 0; ch < channels; ++ch) {
            starts[peer * channels + ch] = routes[peer].size();
            int ordinal = 0;
            for (unsigned t = ch * channel_tokens; t < std::min(tokens, (ch + 1) * channel_tokens); ++t) {
                bool present = route_mode == 0   ? ((peer * 11 + t * 7) % 5 != 0)
                               : route_mode == 1 ? false
                               : route_mode == 2 ? peer == rank
                                                 : peer != rank;
                if (!present) continue;
                heads[t * world + peer] = ordinal++;
                routes[peer].push_back(t);
            }
            ends[peer * channels + ch] = routes[peer].size();
        }
    }
    unsigned total = 0, base = 0;
    for (unsigned source = 0; source < world; ++source) {
        if (source == rank) base = total;
        total += source == rank ? routes[rank].size() : (source + 3) % 7;
        ranks[source * world + rank] = total;
    }
    td.num_recv_tokens = total;
    td.combine_send_channel_prefix = reinterpret_cast<uint64_t>(ends.data());
    // Tail guards also make empty receives valid pointers without dereferencing them.
    std::vector<T> input((total + 1) * hidden), bias0(tokens * hidden), bias1(tokens * hidden),
        output(tokens * hidden + 16);
    std::vector<float> input_weights((total + 1) * topk), output_weights(tokens * topk + 16);
    core_count = reduce_cores;
    for (unsigned generation : {0U, 1U})
      for (unsigned dense_payload : {0U, 1U})
        for (unsigned compact : {0U, 1U})
            for (unsigned local_compact : {0U, 1U}) {
                // Production enables dense payload only with aligned rows and
                // compact (or absent) weights. Sparse routes still pack densely.
                if (dense_payload && (hidden * 2 != td.x_row_stride || (topk && !compact))) continue;
                td.combine_payload_max_rows = dense_payload ? 2U : 1U;
                td.combine_compact_weights = compact && topk;
                td.compact_self_rows = local_compact;
                td.has_bias_0 = generation;
                td.has_bias_1 = generation;
                td.topk_weights_row_stride = compact ? topk * 4 : (topk * 4 + 31) / 32 * 32;
                const uint64_t workspace_bytes = td.topk_weights_buffer_offset +
                                                 uint64_t(world) * td.token_capacity * td.topk_weights_row_stride + 32;
                std::vector<uint8_t> workspace(workspace_bytes, 0xff);
                for (unsigned r = 0; r < routes[rank].size(); ++r) {
                    for (unsigned h = 0; h < hidden; ++h)
                        input[(base + r) * hidden + h] = Encode<B>(Value(rank, routes[rank][r], h, generation));
                    for (unsigned k = 0; k < topk; ++k)
                        input_weights[(base + r) * topk + k] = Value(rank, routes[rank][r], k + 7, generation);
                }
                std::vector<float> expected(tokens * hidden), expected_weights(tokens * topk);
                for (unsigned t = 0; t < tokens; ++t)
                    for (unsigned h = 0; h < hidden; ++h) {
                        bias0[t * hidden + h] = Encode<B>(Value(1, t, h, generation));
                        bias1[t * hidden + h] = Encode<B>(Value(2, t, h, generation));
                        if (generation)
                            expected[t * hidden + h] = Decode(bias0[t * hidden + h]) + Decode(bias1[t * hidden + h]);
                    }
                for (unsigned peer = 0; peer < world; ++peer)
                    for (unsigned r = 0; r < routes[peer].size(); ++r) {
                        unsigned t = routes[peer][r];
                        uint64_t legacy_slot =
                            uint64_t(peer) * td.token_capacity + (peer == rank && local_compact ? r : t);
                        uint64_t payload_slot = dense_payload ? uint64_t(peer) * td.token_capacity + r : legacy_slot;
                        uint64_t weight_slot = compact ? uint64_t(peer) * td.token_capacity + r : legacy_slot;
                        for (unsigned h = 0; h < hidden; ++h) {
                            T v = Encode<B>(Value(peer, t, h, generation));
                            expected[t * hidden + h] += Decode(v);
                            std::memcpy(workspace.data() + td.x_buffer_offset + payload_slot * td.x_row_stride + h * 2,
                                        &v, 2);
                        }
                        for (unsigned k = 0; k < topk; ++k) {
                            float v = Value(peer, t, k + 7, generation);
                            expected_weights[t * topk + k] += v;
                            std::memcpy(workspace.data() + td.topk_weights_buffer_offset +
                                            weight_slot * td.topk_weights_row_stride + k * 4,
                                        &v, 4);
                        }
                    }
                for (unsigned direct : {0U, 1U}) {
                    if (dense_payload && !direct) continue;
                    td.combine_direct_self = direct;
                    if (direct) {
                        // Any accidental SHMEM self read now produces NaN; input is authoritative.
                        std::memset(workspace.data() + td.x_buffer_offset +
                                        uint64_t(rank) * td.token_capacity * td.x_row_stride,
                                    0xff, td.token_capacity * td.x_row_stride);
                        std::memset(workspace.data() + td.topk_weights_buffer_offset +
                                        uint64_t(rank) * td.token_capacity * td.topk_weights_row_stride,
                                    0xff, td.token_capacity * td.topk_weights_row_stride);
                    }
                    std::fill(output.begin(), output.end(), Encode<B>(7));
                    std::fill(output_weights.begin(), output_weights.end(), 123.0F);
                    for (current_core = 0; current_core < int(reduce_cores); ++current_core) {
                        V1CombineUdmaImpl<T, false, true>(
                            reinterpret_cast<GM_ADDR>(input.data()), reinterpret_cast<GM_ADDR>(input_weights.data()),
                            reinterpret_cast<GM_ADDR>(bias0.data()), reinterpret_cast<GM_ADDR>(bias1.data()), nullptr,
                            reinterpret_cast<GM_ADDR>(ranks.data()), reinterpret_cast<GM_ADDR>(starts.data()),
                            reinterpret_cast<GM_ADDR>(heads.data()), workspace.data(),
                            reinterpret_cast<GM_ADDR>(output.data()), reinterpret_cast<GM_ADDR>(output_weights.data()),
                            td);
                        for (auto& row : AscendC::events)
                            for (int event : row) assert(event == 0);
                    }
                    for (unsigned i = 0; i < tokens * hidden; ++i)
                        assert(output[i].bits == Encode<B>(expected[i]).bits);
                    for (unsigned i = 0; i < tokens * topk; ++i) assert(output_weights[i] == expected_weights[i]);
                    for (unsigned i = tokens * hidden; i < output.size(); ++i)
                        assert(output[i].bits == Encode<B>(7).bits);
                    for (unsigned i = tokens * topk; i < output_weights.size(); ++i)
                        assert(output_weights[i] == 123.0F);
                }
            }
}

int main()
{
    for (unsigned exclusion = 0; exclusion < 7; ++exclusion) {
        ascend_deepep::CombineTilingData tiling{};
        bool use_udma = exclusion != 1, epilogue = exclusion != 2, aggregate = exclusion == 4;
        tiling.udma_token_schedule = exclusion == 3;
        unsigned pipeline_chunk_tokens = exclusion == 5 ? 32 : 0;
        std::optional<int> expanded_x;
        if (exclusion == 6) expanded_x = 1;
        // PRODUCTION_GATE
        assert(bool(tiling.combine_direct_self) == (exclusion == 0));
    }
    for (unsigned world : {4U, 8U, 16U, 32U, 64U, 128U})
        for (unsigned rank : {0U, world / 2, world - 1})
            for (unsigned tokens : {0U, 3U, 65U})
                for (unsigned channels : {8U, 32U})
                    for (unsigned reduce_cores : {8U, 32U, 48U})
                        for (unsigned topk : {0U, 6U, 8U, 10U}) {
                            Check<true>(world, rank, tokens, 17, channels, topk, reduce_cores, 0);
                            Check<false>(world, rank, tokens, 32, channels, topk, reduce_cores, 0);
                        }
    Check<true>(16, 3, 65, 32, 8, 128, 48, 0);
    Check<true>(16, 3, 17, 6144, 8, 8, 48, 0);
    Check<true>(8, 3, 19, 2816, 32, 0, 8, 0);
    Check<false>(4, 0, 3, 33008, 8, 6, 32, 0);  // Dense multi-chunk payload reduction.
    Check<false>(4, 0, 17, 17, 32, 1, 32, 0);
    for (unsigned route : {1U, 2U, 3U}) {
        Check<true>(8, 3, 19, 17, 8, 10, 48, route);
        Check<false>(16, 0, 3, 33000, 32, 6, 8, route);  // Multiple payload chunks.
    }
}
