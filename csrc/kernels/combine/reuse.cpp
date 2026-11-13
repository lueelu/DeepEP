// Copyright (c) 2026, Lu Lu

#include "kernel_operator.h"
#include "shmem.h"
#include "launch.hpp"
#include "tiling.hpp"

using namespace AscendC;
namespace {
constexpr uint32_t kElements = 8192U;
constexpr uint32_t kUbBytes = kElements * 12U + 32768U + 32U;
static_assert(kUbBytes <= ascend_deepep::combine_config::kLaunchUbBytes);
// Reuse owns these otherwise idle control slots; each slot has one remote writer.
constexpr uint32_t kGatherReady = 0U, kReleased = 8U, kReturned = 16U;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
constexpr aclshmemx_udma_op_config_t kOrdered{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
struct Scratch {
    LocalTensor<bfloat16_t> input, output;
    LocalTensor<float> converted, sum;
    LocalTensor<uint8_t> copy;
    LocalTensor<uint64_t> flag;
    __ubuf__ uint8_t* wqe;

    __aicore__ inline void StoreFlag(GM_ADDR address, uint64_t value)
    {
        flag.SetValue(0, value);
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        DataCopyPad(gm, flag, DataCopyExtParams(1, sizeof(uint64_t), 0, 0, 0));
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void Wait(GM_ADDR workspace, uint32_t slot, uint64_t epoch)
    {
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
            workspace + uint64_t(slot) * ascend_deepep::combine_control::kSlotBytes));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        do {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
            DataCopy(flag, gm, 4U);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
        } while (flag.GetValue(0) < epoch);
    }

    __aicore__ inline void Add(GM_ADDR address, uint32_t count, bool first)
    {
        GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(address));
        // Scalar credit polling must finish before the DMA reads reused rows.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopy(input, gm, count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (first) {
            Cast(sum, input, RoundMode::CAST_NONE, count);
        } else {
            Cast(converted, input, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sum, sum, converted, count);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void Store(GM_ADDR address, uint32_t count, bool empty = false)
    {
        if (empty) {
            Duplicate(output, static_cast<bfloat16_t>(0.0F), count);
        } else {
            Cast(output, sum, RoundMode::CAST_RINT, count);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(address));
        DataCopy(gm, output, count);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }

    __aicore__ inline void Put(GM_ADDR destination, GM_ADDR source, uint32_t bytes, uint32_t peer, uint32_t self)
    {
        if (peer == self ||
            (peer / 8U == self / 8U && (aclshmemi_get_state()->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE))) {
            if (peer != self) destination = reinterpret_cast<GM_ADDR>(aclshmem_ptr(destination, peer));
            GlobalTensor<uint8_t> src, dst;
            src.SetGlobalBuffer(source);
            dst.SetGlobalBuffer(destination);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(copy, src, DataCopyExtParams(1, bytes, 0, 0, 0), DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            DataCopyPad(dst, copy, DataCopyExtParams(1, bytes, 0, 0, 0));
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        } else {
            // Core 0 exclusively owns QP0. At most one SO is outstanding.
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kOrdered>(destination, source, wqe, bytes, peer, 0U,
                                                                    EVENT_ID0);
            aclshmemx_udma_qp_quiet(peer, 0U);
        }
    }

    __aicore__ inline void Publish(GM_ADDR workspace, uint32_t slot, uint64_t epoch, uint32_t peer,
                                   const ascend_deepep::PutCombineTiling& t)
    {
        auto* source = workspace + uint64_t(kReturned + t.rank_num) * ascend_deepep::combine_control::kSlotBytes;
        StoreFlag(source, epoch);
        // Put drains payload and flag transfers before their source can be overwritten.
        Put(workspace + uint64_t(slot) * ascend_deepep::combine_control::kSlotBytes, source, sizeof(uint64_t), peer,
            t.rank_id);
    }
};

__aicore__ inline uint32_t LowerToken(GlobalTensor<int32_t>& table, uint64_t base, uint32_t count, uint32_t token)
{
    uint32_t low = 0, high = count;
    while (low < high) {
        const uint32_t mid = low + (high - low) / 2U;
        if (uint32_t(table.GetValue((base + mid) * 6U + 1U)) < token)
            low = mid + 1U;
        else
            high = mid;
    }
    return low;
}

__aicore__ inline void Reuse(GM_ADDR workspace, GM_ADDR input, GM_ADDR forward, GM_ADDR counts, GM_ADDR backward,
                             GM_ADDR offsets, GM_ADDR mask, GM_ADDR weights, GM_ADDR weight_meta,
                             const ascend_deepep::PutCombineTiling& t, uint32_t window)
{
    const uint32_t core = GetBlockIdx(), lanes = t.rank_num < 8U ? t.rank_num : 8U;
    const uint32_t slots = ascend_deepep::CombineReturnSlots(t.rank_num, t.weight_topk);
    const uint64_t row_bytes = uint64_t(t.hidden_size) * 2U;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> in, out, converted, sum, copy, flag;
    TBuf<TPosition::VECOUT> wqe;
    pipe.InitBuffer(in, kElements * 2U);
    pipe.InitBuffer(out, kElements * 2U);
    pipe.InitBuffer(converted, kElements * 4U);
    pipe.InitBuffer(sum, kElements * 4U);
    pipe.InitBuffer(copy, 16384U);
    pipe.InitBuffer(wqe, 16384U);
    pipe.InitBuffer(flag, 32U);
    Scratch s{in.Get<bfloat16_t>(),
              out.Get<bfloat16_t>(),
              converted.Get<float>(),
              sum.Get<float>(),
              copy.Get<uint8_t>(),
              flag.Get<uint64_t>(),
              reinterpret_cast<__ubuf__ uint8_t*>(wqe.Get<uint8_t>().GetPhyAddr())};
    GlobalTensor<int32_t> f, n, ids;
    GlobalTensor<int64_t> b, boundaries;
    GlobalTensor<uint64_t> masks;
    f.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(forward));
    n.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(backward));
    boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(offsets));
    masks.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(mask));
    uint64_t cursor[8]{}, end[8]{};
    // The wrapper joins the previous invocation before this local reset. No peer
    // may publish new flags until every rank has finished resetting its slots.
    for (uint32_t slot = core; slot <= kReturned + t.rank_num; slot += 64U) {
        s.StoreFlag(workspace + uint64_t(slot) * ascend_deepep::combine_control::kSlotBytes, 0U);
    }
    aclshmemx_barrier_all_vec();
    if (core == 0U) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            cursor[lane] = boundaries.GetValue(lane);
            end[lane] = boundaries.GetValue(lane + 1U);
        }
        if (t.with_weights) {
            ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(weight_meta));
            for (uint32_t row = 0; row < t.weight_rows; ++row) {
                const uint32_t route = ids.GetValue(row);
                const uint32_t peer = route / (t.token_num * t.weight_topk);
                s.Put(workspace + t.weight_recv_offset_bytes + uint64_t(route % (t.token_num * t.weight_topk)) * 4U,
                      weights + uint64_t(row) * 4U, 4U, peer, t.rank_id);
            }
        }
    }
    for (uint32_t token_begin = 0; token_begin < t.token_num; token_begin += window) {
        const uint32_t token_end = token_begin + window < t.token_num ? token_begin + window : t.token_num;
        for (uint32_t source = 0; source < t.rank_num; ++source) {
            const uint32_t epoch = token_begin / window * t.rank_num + source + 1U;
            if (core == 0U) {
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    const uint32_t peer = t.rank_id / 8U * 8U + lane;
                    if (epoch > 1U) s.Wait(workspace, kReleased + lane, epoch - 1U);
                    while (cursor[lane] < end[lane]) {
                        const uint64_t item = cursor[lane] * 3U;
                        const uint64_t target = b.GetValue(item + 2U);
                        if (uint32_t(target >> 32U) != epoch) break;
                        s.Put(workspace + t.gather_input_offset_bytes + uint64_t(uint32_t(target)) * row_bytes,
                              input + uint64_t(b.GetValue(item + 1U)) * row_bytes, uint32_t(row_bytes), peer,
                              t.rank_id);
                        ++cursor[lane];
                    }
                    // Empty peers publish too, so sparse routes cannot strand a waiter.
                    s.Publish(workspace, kGatherReady + t.rank_id % 8U, epoch, peer, t);
                }
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    s.Wait(workspace, kGatherReady + lane, epoch);
                }
            }
            SyncAll<true>();
            const uint64_t base = uint64_t(source) * t.forward_row_capacity;
            const uint32_t count = n.GetValue(source);
            const uint32_t first = LowerToken(f, base, count, token_begin);
            const uint32_t last = LowerToken(f, base, count, token_end);
            const uint32_t slice_begin = first + uint64_t(last - first) * core / 64U;
            const uint32_t slice_end = first + uint64_t(last - first) * (core + 1U) / 64U;
            uint32_t task = slice_begin == first ? first : uint32_t(f.GetValue((base + slice_begin - 1U) * 6U));
            for (; task < slice_end;) {
                const uint64_t row = (base + task) * 6U;
                const uint32_t next = f.GetValue(row);
                if (f.GetValue(row + 3U) >= 0) {
                    const uint32_t token = f.GetValue(row + 1U);
                    s.Add(input + uint64_t(f.GetValue(row + 2U)) * row_bytes, t.hidden_size, true);
                    for (uint32_t j = task; j < next; ++j) {
                        const uint64_t at = (base + j) * 6U;
                        const bool local = uint32_t(f.GetValue(at + 3U)) == t.rank_id;
                        const uint32_t input_row = local ? uint32_t(f.GetValue(at + 4U))
                                                         : uint32_t(f.GetValue(at + 5U)) % t.gather_row_capacity;
                        s.Add(
                            (local ? input : workspace + t.gather_input_offset_bytes) + uint64_t(input_row) * row_bytes,
                            t.hidden_size, false);
                    }
                    s.Store(workspace + t.server_partial_offset_bytes + uint64_t(token % window) * row_bytes,
                            t.hidden_size);
                }
                task = next;
            }
            SyncAll<true>();
            if (core == 0U) {
                for (uint32_t i = first; i < last;) {
                    const uint64_t row = (base + i) * 6U;
                    const uint32_t token = f.GetValue(row + 1U);
                    const bool reduced = f.GetValue(row + 3U) >= 0;
                    auto* payload =
                        reduced ? workspace + t.server_partial_offset_bytes + uint64_t(token % window) * row_bytes
                                : input + uint64_t(f.GetValue(row + 2U)) * row_bytes;
                    s.Put(workspace + t.returned_partial_offset_bytes +
                              (uint64_t(token) * slots + t.rank_id / 8U) * row_bytes,
                          payload, uint32_t(row_bytes), source, t.rank_id);
                    i = f.GetValue(row);
                }
                s.Publish(workspace, kReturned + t.rank_id, epoch, source, t);
                // All local readers and outgoing sends are done before granting reuse.
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    s.Publish(workspace, kReleased + t.rank_id % 8U, epoch, t.rank_id / 8U * 8U + lane, t);
                }
                if (t.rank_id == source) {
                    for (uint32_t peer = 0; peer < t.rank_num; ++peer) {
                        s.Wait(workspace, kReturned + peer, epoch);
                    }
                }
            }
            SyncAll<true>();
            if (t.rank_id == source) {
                for (uint32_t token = token_begin + core; token < token_end; token += 64U) {
                    const uint64_t bits = masks.GetValue(token);
                    bool first_partial = true;
                    for (uint32_t server = 0; server < (t.rank_num + 7U) / 8U; ++server) {
                        if (bits & (1ULL << server)) {
                            s.Add(workspace + t.returned_partial_offset_bytes +
                                      (uint64_t(token) * slots + server) * row_bytes,
                                  t.hidden_size, first_partial);
                            first_partial = false;
                        }
                    }
                    s.Store(workspace + t.output_offset_bytes + uint64_t(token) * row_bytes, t.hidden_size,
                            first_partial);
                }
            }
            SyncAll<true>();
        }
    }
}
#endif
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_reuse_kernel(
    GM_ADDR workspace, GM_ADDR input, GM_ADDR forward, GM_ADDR counts, GM_ADDR backward, GM_ADDR offsets, GM_ADDR mask,
    GM_ADDR weights, GM_ADDR weight_meta, ascend_deepep::PutCombineTiling tiling, uint32_t window)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Reuse(workspace, input, forward, counts, backward, offsets, mask, weights, weight_meta, tiling, window);

}
#endif
