// CPU data/ownership oracle; synchronous DMA mocks do not prove device ordering.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/fused_layout.h"
#include "ops/dispatch_host_timing.h"
using namespace ascend_deepep;
using GM_ADDR = uint8_t*;
constexpr int EVENT_ID0 = 0, PIPE_V = 0;
struct {
    unsigned x;
} threadIdx;
struct dim3 {
    unsigned x;
    explicit dim3(unsigned n) : x(n) {}
};
inline void asc_threadfence() {}
inline void trap()
{
    std::abort();
}
namespace __asc_simt_vf {
inline void __trap()
{
    std::abort();
}
}  // namespace __asc_simt_vf
template <auto Fn, class... Args>
void asc_vf_call(dim3 grid, Args... args)
{
    for (threadIdx.x = 0; threadIdx.x < grid.x; ++threadIdx.x) Fn(args...);
}
namespace AscendC {
enum class HardEvent { V_S, S_MTE2, MTE2_V, V_MTE3, MTE3_S };
template <HardEvent>
void SetFlag(int)
{
}
template <HardEvent>
void WaitFlag(int)
{
}
template <int>
void PipeBarrier()
{
}
template <class T>
struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p)
    {
        data = p;
    }
    GlobalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
template <class T>
struct LocalTensor {
    T* data{};
    T* GetPhyAddr() const
    {
        return data;
    }
};
struct DataCopyExtParams {
    uint32_t blocks, bytes, src_stride, dst_stride, reserved;
    DataCopyExtParams(uint32_t n, uint32_t b, uint32_t s, uint32_t d, uint32_t r)
        : blocks(n), bytes(b), src_stride(s), dst_stride(d), reserved(r)
    {
    }
};
template <class T>
struct DataCopyPadExtParams {
    bool pad;
    uint8_t left, right;
    T value;
};
template <class T>
void Duplicate(LocalTensor<T> dst, T value, uint32_t n)
{
    std::fill_n(dst.data, n, value);
}
template <class T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(copy.blocks == 1 && copy.src_stride == 0 && copy.dst_stride == 0);
    std::memcpy(dst.data, src.data, copy.bytes);
}
template <class T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    assert(copy.blocks == 1 && copy.src_stride == 0 && copy.dst_stride == 0);
    std::memcpy(dst.data, src.data, copy.bytes);
}
}  // namespace AscendC
struct Resource {
    std::vector<uint64_t> ub = std::vector<uint64_t>(platform::kKernelUbBytes / 8);
    template <class T>
    AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset)
    {
        assert(offset % 32 == 0 && offset < platform::kKernelUbBytes);
        return {reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ub.data()) + offset)};
    }
};
// PRODUCTION_FUNCTIONS

int main()
{
    // Exercise actual fused/deferred builders across tails, empty channels and skew.
    for (uint32_t world : {4U, 8U, 16U, 128U, 256U})
        for (uint32_t tokens : {0U, 3U, 129U, 4099U})
            for (uint32_t routing : {0U, 1U, 2U}) {
                DispatchTilingData td{};
                td.world_size = world;
                td.num_tokens = tokens;
                td.num_channels = 8;
                td.num_topk = 8;
                td.num_experts = 256;
                td.expert_stride = 256;
                td.rank_stride = (world + 7U) / 8U * 8U;
                uint32_t length = (tokens + 7U) / 8U, stride = (length + 7U) / 8U * 8U;
                std::vector<topk_idx_t> idx(tokens * 8);
                std::vector<uint8_t> mask(tokens * world, 99), expected_mask(tokens * world, 0);
                std::vector<int32_t> expected_hist(8 * 256, 0), expected_counts(8 * td.rank_stride, 0);
                std::vector<int32_t> expected_plan(world * 8 * stride, -99), expected_head(tokens * world, -1);
                for (uint32_t t = 0; t < tokens; ++t) {
                    for (uint32_t j = 0; j < 8; ++j) {
                        auto e =
                            routing == 2 || (t + j) % 11 == 0 ? -1 : int(routing == 1 ? j : (t * 17 + j * 13) % 256);
                        idx[t * 8 + j] = e;
                        if (e >= 0) {
                            expected_mask[t * world + e / (256 / world)] = 1;
                            ++expected_hist[(t / length) * 256 + e];
                        }
                    }
                    for (uint32_t p = 0; p < world; ++p)
                        if (expected_mask[t * world + p]) {
                            auto ch = t / length;
                            auto ordinal = expected_counts[ch * td.rank_stride + p]++;
                            expected_plan[(p * 8 + ch) * stride + ordinal] = t;
                            expected_head[t * world + p] = ordinal;
                        }
                }
                td.notify_topk_idx = reinterpret_cast<uint64_t>(idx.data());
                for (uint32_t defer : {0U, 1U}) {
                    Resource resource;
                    std::fill(mask.begin(), mask.end(), 99);
                    std::vector<int32_t> plan(expected_plan.size(), -99), head(expected_head.size(), -99);
                    std::vector<int32_t> hist(expected_hist.size(), -99), counts(expected_counts.size(), -99);
                    td.notify_defer_plan = defer;
                    td.notify_expert_partials = reinterpret_cast<uint64_t>(hist.data());
                    for (uint32_t ch = 0; ch < 8; ++ch) {
                        BuildFusedDispatchPlan(resource, mask.data(), reinterpret_cast<GM_ADDR>(plan.data()),
                                               reinterpret_cast<GM_ADDR>(head.data()), td, ch);
                        std::copy_n(resource.GetBufferByByte<int32_t>(0).data, td.rank_stride,
                                    counts.data() + ch * td.rank_stride);
                    }
                    assert(mask == expected_mask && hist == expected_hist && counts == expected_counts);
                    if (defer) {
                        assert(std::all_of(plan.begin(), plan.end(), [](auto v) { return v == -99; }));
                        assert(std::all_of(head.begin(), head.end(), [](auto v) { return v == -99; }));
                        for (uint32_t ch = 0; ch < 8; ++ch) {
                            // Later notify stages overwrite the old UB counts. Rebuild must not depend on them.
                            std::fill(resource.ub.begin(), resource.ub.end(), uint64_t(-1));
                            BuildDispatchPlan(resource, mask.data(), reinterpret_cast<GM_ADDR>(plan.data()),
                                              reinterpret_cast<GM_ADDR>(head.data()), td, ch);
                        }
                    }
                    assert(plan == expected_plan && head == expected_head);
                }
            }
    using namespace ascend_deepep::ops;
    DispatchHostTiming disabled;
    disabled.Mark(NotifySubmitted);
    disabled.Finish();
    assert(disabled.TotalSlot() == nullptr && TakeDispatchHostTimings().empty());
    BeginDispatchHostTiming();
    DispatchHostTiming enabled;
    for (int i = NotifySubmitBegin; i <= CommSubmitted; ++i) enabled.Mark(static_cast<HostStage>(i));
    enabled.Value(Fused, 1);
    enabled.Finish();
    MarkDispatchHostTiming(ApiReturn);
    MarkDispatchHostTiming(Complete);
    auto rows = TakeDispatchHostTimings();
    assert(rows.size() == 1 && rows[0][Fused] == 1 && rows[0][HostPrep] == 0);
    assert(rows[0][ApiEntry] > 0);
    for (int i = ApiEntry + 1; i <= Complete; ++i) assert(rows[0][i] >= rows[0][i - 1]);
    assert(TakeDispatchHostTimings().empty());
}
