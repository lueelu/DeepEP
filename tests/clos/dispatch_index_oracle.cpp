// The test inserts production window helpers and sender methods. Constant
// evaluation exercises their decisions even when Windows blocks test binaries.
#include <array>
#include <cstdint>
#define __aicore__ constexpr
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
constexpr uint32_t kBatchWqes = 32, kOutputRecordWords = 12;
constexpr uint32_t kTileTokens = 256, EVENT_ID6 = 6;
void trap();
namespace __asc_simt_vf {
void __trap();
}
namespace AscendC {
template <class T>
struct LocalTensor {
    T* data;
    constexpr T GetValue(unsigned i) const
    {
        return data[i];
    }
    constexpr LocalTensor operator[](unsigned i) const
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
constexpr void DataCopyPad(LocalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p, DataCopyPadExtParams<T>)
{
    if (p.blocks != 1 || p.src_stride || p.dst_stride) trap();
    for (unsigned i = 0; i < p.bytes / sizeof(T); ++i) dst.data[i] = src.data[i];
}
enum class HardEvent { S_MTE2, MTE2_S };
template <HardEvent>
constexpr void SetFlag(unsigned)
{
}
template <HardEvent>
constexpr void WaitFlag(unsigned)
{
}
}  // namespace AscendC
// WINDOW_HELPERS

constexpr bool WindowCheck(unsigned density, unsigned tile)
{
    std::array<int32_t, 700> plan{}, cache{};
    unsigned total = 0;
    for (unsigned i = 0; i < 700; ++i)
        if ((i * 13 + 5) % 11 < density) plan[total++] = i;
    unsigned consumed = 0, loaded_first = 0, loaded_count = 0, reads = 0, old_reads = 0;
    for (unsigned start = 0; start < 700; start += tile) {
        unsigned size = 700 - start < tile ? 700 - start : tile;
        unsigned left = total - consumed, load = left < size ? left : size;
        old_reads += load;
        unsigned available = loaded_first + loaded_count - consumed;
        unsigned last = available ? cache[loaded_count - 1] : 0;
        if (!ReuseDispatchIndexWindow(available, left, size, last, start + size)) {
            for (unsigned i = 0; i < load; ++i) cache[i] = plan[consumed + i];
            loaded_first = consumed;
            loaded_count = load;
            reads += load;
        }
        unsigned offset = consumed - loaded_first;
        unsigned count = DispatchIndexLowerBound(AscendC::LocalTensor<int32_t>{cache.data()}, offset,
                                                 loaded_count - offset, start + size);
        unsigned expected = 0;
        while (consumed + expected < total && unsigned(plan[consumed + expected]) < start + size) ++expected;
        if (count != expected) return false;
        for (unsigned i = 0; i < count; ++i)
            if (cache[offset + i] != plan[consumed + i]) return false;
        consumed += count;
    }
    return consumed == total && reads <= old_reads && (density != 1 || reads < old_reads);
}
constexpr bool Windows()
{
    for (unsigned d = 0; d <= 11; ++d)
        for (unsigned tile : {32U, 64U, 256U})
            if (!WindowCheck(d, tile)) return false;
    return true;
}
static_assert(Windows());

struct Trace {
    enum { StateFlushes, DescriptorCycles, ListWindows, ListCycles, PutRowsCalls };
    constexpr uint64_t Now() const
    {
        return 0;
    }
    constexpr void Count(unsigned) const {}
    constexpr void Add(unsigned, uint64_t) const {}
};
using DispatchDeviceTrace = Trace;
struct Base {
    static constexpr unsigned QueuePeer(unsigned p)
    {
        return p;
    }
    static constexpr void TraceCount(unsigned) {}
    static constexpr void Admit(unsigned) {}
    static constexpr uint64_t TraceNow()
    {
        return 0;
    }
    static constexpr void TraceAdd(unsigned, uint64_t) {}
};
struct Model {
    static constexpr unsigned saved_base = 1536;
    struct {
        unsigned has_x_scales, num_scales, scale_element_bytes, packed_peer_scales;
    } td_{};
    struct Peer {
        unsigned state;
    };
    std::array<Peer, 4> peers_{};
    std::array<uint64_t, 128> spans{};
    std::array<int32_t, 1600> memory{};
    std::array<uint64_t, 64> headers{};
    std::array<uint64_t, 4 * kOutputRecordWords> bases{};
    std::array<uint64_t, 1024> actual_src{}, expected_src{}, actual_dst{}, expected_dst{};
    std::array<unsigned, 128> actual_batches{}, expected_batches{};
    unsigned rows_pending_ = 0, spans_ = 0, saved_indices_ = 0, last_ = 0, active_state_ = 0;
    unsigned actual = 0, expected = 0, nb = 0, ne = 0, expected_pending = 0, expected_state = 0;
    unsigned fields;
    constexpr Model(unsigned f, bool shared) : fields(f)
    {
        td_ = {f - 1, f - 1, f == 2 ? 4U : 0U, 0};
        for (unsigned p = 0; p < 4; ++p) {
            peers_[p].state = shared ? p / 2 : p;
            bases[p * kOutputRecordWords] = 100000 + p * 10000;
            bases[p * kOutputRecordWords + 1] = 500000 + p * 10000;
            bases[p * kOutputRecordWords + 2] = 1024;
        }
        for (unsigned i = 0; i < headers.size(); ++i) headers[i] = i + 900;
    }
    // SENDER_METHODS
    constexpr unsigned Count() const
    {
        return 2;
    }
    constexpr void BeginVisit(unsigned) {}
    constexpr unsigned PutRowsUb(unsigned p, uint64_t address, unsigned row, unsigned field, unsigned count)
    {
        return PutRowsImpl(p, address, row, field, count, true);
    }
    constexpr void Flush()
    {
        if (!rows_pending_) return;
        uint64_t wqes[32 * 8]{};
        for (unsigned tid = 0; tid < 32; ++tid)
            BuildDispatchWqes(wqes, spans.data(), headers.data(), bases.data(), 1000, 5000, 64, 4, 4096, fields, spans_,
                              rows_pending_, tid);
        for (unsigned i = 0; i < rows_pending_; ++i) {
            actual_src[actual] = wqes[i * 8 + 7];
            actual_dst[actual++] = wqes[i * 8 + 5];
        }
        actual_batches[nb++] = rows_pending_;
        rows_pending_ = spans_ = saved_indices_ = 0;
    }
    constexpr void Expect(unsigned p, unsigned row, unsigned token, unsigned field)
    {
        unsigned state = peers_[p].state;
        if (expected_pending && expected_state != state) {
            expected_batches[ne++] = expected_pending;
            expected_pending = 0;
        }
        expected_state = state;
        unsigned bytes = field ? 4 : 64;
        expected_src[expected] = (field ? 5000 : 1000) + uint64_t(token) * bytes;
        expected_dst[expected++] = bases[p * kOutputRecordWords + field] + uint64_t(row) * bytes;
        if (++expected_pending == 32) {
            expected_batches[ne++] = 32;
            expected_pending = 0;
        }
    }
    constexpr bool Run(unsigned visit, bool gm)
    {
        unsigned dest[2]{};
        // Includes empty windows, odd tails, and partial scale pairs at a batch
        // boundary. Rewriting both list windows also models a channel transition.
        for (unsigned window = 0; window < 8; ++window) {
            PreserveIndices();
            // Idempotence: retained spans must not consume more saved storage.
            unsigned saved = saved_indices_;
            PreserveIndices();
            if (saved != saved_indices_) return false;
            unsigned length[2]{(window * 17 + 1) % 33, (window * 7 + 3) % 31};
            unsigned cursor[2]{}, field[2]{};
            for (unsigned p = 0; p < 2; ++p)
                for (unsigned t = 0; t < 64; ++t) {
                    memory[p * 64 + t] = int(window * 64 + p * 32 + t);
                    memory[256 + window * 128 + p * 64 + t] = memory[p * 64 + t];
                }
            while (cursor[0] < length[0] || cursor[1] < length[1]) {
                for (unsigned p = 0; p < 2; ++p) {
                    unsigned budget = visit;
                    while (budget && cursor[p] < length[p]) {
                        unsigned available = (length[p] - cursor[p]) * fields - field[p];
                        unsigned request = available < budget ? available : budget;
                        unsigned address = (gm ? 256 + window * 128 : 0) + p * 64 + cursor[p];
                        unsigned count = PutRowsImpl(p, address, dest[p], field[p], request, !gm);
                        for (unsigned i = 0; i < count; ++i) {
                            unsigned position = field[p] + i;
                            Expect(p, dest[p] + position / fields, window * 64 + p * 32 + cursor[p] + position / fields,
                                   position % fields);
                        }
                        unsigned rows = (field[p] + count) / fields;
                        field[p] = (field[p] + count) % fields;
                        cursor[p] += rows;
                        dest[p] += rows;
                        budget -= count;
                    }
                }
            }
        }
        PreserveIndices();
        for (unsigned t = 0; t < 128; ++t) memory[t] = -1;
        Flush();
        if (expected_pending) expected_batches[ne++] = expected_pending;
        if (actual != expected || nb != ne) return false;
        for (unsigned i = 0; i < actual; ++i)
            if (actual_src[i] != expected_src[i] || actual_dst[i] != expected_dst[i]) return false;
        for (unsigned i = 0; i < nb; ++i)
            if (actual_batches[i] != expected_batches[i]) return false;
        return true;
    }
    constexpr bool RunCached(unsigned visit_budget)
    {
        auto& sender = *this;
        const struct {
            unsigned udma_wqes_per_visit;
        } td{visit_budget};
        const Trace trace;
        const bool cached_channel = true;
        unsigned row[2]{};
        for (unsigned channel = 0; channel < 3; ++channel) {
            unsigned total[2]{channel == 0 ? 271U : (channel == 1 ? 17U : 0U),
                              channel == 0 ? 39U : (channel == 1 ? 9U : 3U)};
            unsigned consumed[2]{total[0], total[1]}, counts[2]{total[0], total[1]};
            unsigned loaded_first[2]{}, loaded_count[2]{}, cursor[2]{}, field[2]{};
            uint64_t plan_base[2]{0, 300};
            std::array<int32_t, 600> plan{};
            for (unsigned p = 0; p < 2; ++p)
                for (unsigned i = 0; i < total[p]; ++i) plan[plan_base[p] + i] = channel * 1000 + p * 500 + i;
            AscendC::LocalTensor<int32_t> lists{memory.data()}, plan_gm{plan.data()};
            // Independent old whole-channel ordering and physical-SQ batch oracle.
            unsigned expected_cursor[2]{};
            while (expected_cursor[0] < total[0] || expected_cursor[1] < total[1])
                for (unsigned p = 0; p < 2; ++p) {
                    unsigned budget = visit_budget;
                    while (budget && expected_cursor[p] < total[p]) {
                        const unsigned i = expected_cursor[p]++;
                        Expect(p, row[p] + i, plan[plan_base[p] + i], 0);
                        --budget;
                    }
                }
            unsigned remaining = total[0] + total[1];
            // CACHED_SENDER_LOOP
        }
        Flush();
        if (expected_pending) expected_batches[ne++] = expected_pending;
        if (actual != expected || nb != ne) return false;
        for (unsigned i = 0; i < actual; ++i)
            if (actual_src[i] != expected_src[i] || actual_dst[i] != expected_dst[i]) return false;
        for (unsigned i = 0; i < nb; ++i)
            if (actual_batches[i] != expected_batches[i]) return false;
        return true;
    }
};
// Separate evaluations keep compiler constexpr limits bounded.
static_assert(Model(1, true).Run(1, false));
static_assert(Model(1, true).Run(4, false));
static_assert(Model(1, false).Run(8, false));
static_assert(Model(1, true).Run(16, false));
static_assert(Model(1, false).Run(32, false));
static_assert(Model(2, true).Run(4, false));
static_assert(Model(2, false).Run(8, false));
static_assert(Model(2, true).Run(16, false));
static_assert(Model(2, false).Run(32, false));
static_assert(Model(1, true).Run(64, false));
static_assert(Model(1, false).Run(128, false));
static_assert(Model(2, false).Run(64, false));
static_assert(Model(2, true).Run(128, false));
static_assert(Model(1, true).Run(16, true));
static_assert(Model(2, true).Run(16, true));
static_assert(Model(1, true).RunCached(4));
static_assert(Model(1, true).RunCached(31));
static_assert(Model(1, false).RunCached(32));
static_assert(Model(1, true).RunCached(64));
static_assert(Model(1, false).RunCached(128));
static_assert(Model(1, true).RunCached(257));
