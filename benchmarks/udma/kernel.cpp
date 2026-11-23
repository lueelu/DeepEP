#include "contract.h"
#include "kernels/common/ub_workspace.h"
#include "kernels/common/udma_shared_sender.h"

using namespace AscendC;
using namespace udma_bench;
namespace u = ascend_deepep::udma;

// One lane owns each word. Source/destination/epoch/offset all affect the value;
// validation cannot pass by reading a stale payload from a preceding iteration.
template <bool Verify>
__simt_vf__ __launch_bounds__(32) inline void Payload(__gm__ uint64_t* send, __gm__ uint64_t* recv,
                                                      __gm__ uint64_t* errors, uint64_t words, uint32_t src,
                                                      uint32_t dst, uint64_t epoch, uint32_t core, uint32_t cores,
                                                      bool write_send, bool write_recv)
{
    uint64_t bad = 0;
    for (uint64_t i = uint64_t(core) * kLanes + threadIdx.x; i < words; i += uint64_t(cores) * kLanes) {
        uint64_t value = i ^ (uint64_t(src) << 48) ^ (uint64_t(dst) << 32) ^ (epoch * 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        if constexpr (Verify) {
            bad += recv[i] != value;
        } else {
            if (write_send) send[i] = value;
            if (write_recv) recv[i] = ~value;
        }
    }
    if constexpr (Verify) errors[core * kLanes + threadIdx.x] += bad;
    asc_threadfence();
}

__aicore__ inline void DrainSimt()
{
    SetFlag<HardEvent::V_S>(EVENT_ID6);
    WaitFlag<HardEvent::V_S>(EVENT_ID6);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void bench_prepare(GM_ADDR workspace, Config c,
                                                                                  uint64_t ffts)
{
    util_set_ffts_config(ffts);
    const uint32_t core = get_block_idx();
    for (uint32_t peer = 0; peer < c.world; ++peer) {
        if (Edge(c, c.rank, peer)) {
            auto* row = reinterpret_cast<__gm__ uint64_t*>(workspace + SendOffset(c) + peer * c.capacity);
            asc_vf_call<Payload<false>>(dim3(kLanes), row, row, row, c.bytes / 8, c.rank, peer, c.epoch, core, c.cores,
                                        true, false);
            DrainSimt();
        }
        if (Edge(c, peer, c.rank)) {
            auto* row = reinterpret_cast<__gm__ uint64_t*>(workspace + RecvOffset(c) + peer * c.capacity);
            asc_vf_call<Payload<false>>(dim3(kLanes), row, row, row, c.bytes / 8, peer, c.rank, c.epoch, core, c.cores,
                                        false, true);
            DrainSimt();
        }
    }
}

// Keep original Clos WQE construction, SQ admission, shared-Jetty mapping and
// per-batch CQE requests. Only payload selection and marker publication differ.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
class Sender : public u::Sender {
public:
    __aicore__ Sender(Config c, uint32_t core, __ubuf__ uint8_t* scratch, GM_ADDR workspace)
        : u::Sender(c.rank, c.world, core, c.peers, scratch, workspace)
    {
    }

    __aicore__ void PublishActive(GM_ADDR workspace, const Config& c)
    {
        Flush();
        auto* one = u::Signal(workspace, 0, c.rank);
        for (uint32_t p = 0; p < count_; ++p) {
            if (Edge(c, c.rank, peers_[p].peer)) {
                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, u::kDone>(
                    one, one, reinterpret_cast<__ubuf__ uint64_t*>(scratch_), 1U, peers_[p].peer, qp_base_, EVENT_ID5);
            }
        }
    }
};
#endif

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void bench_transfer(GM_ADDR workspace, GM_ADDR timings,
                                                                                   Config c, uint64_t ffts)
{
    util_set_ffts_config(ffts);
    const uint32_t core = get_block_idx();
    ascend_deepep::UbWorkspace ub;
    // Reset only the disjoint control header. All AIVs/all ranks participate,
    // including inactive pair-test ranks. Entry alignment is outside the timer.
    u::Begin(workspace, 0, c.rank, c.world, core, c.cores, ub.GetBufferByByte<uint64_t>(0));
    uint64_t t[kTimes];
    t[0] = GetSystemCycle();
    t[1] = t[2] = t[3] = t[0];
    bool active = false;
    const uint32_t count = u::WorkerCount(c.rank, c.world, core, c.peers);
    for (uint32_t p = 0; p < count; ++p) active |= Edge(c, c.rank, u::WorkerPeer(c.rank, c.world, core, c.peers, p));
    if (active) {
        Sender sender(c, core, reinterpret_cast<__ubuf__ uint8_t*>(ub.GetBufferByByte<uint8_t>(0).GetPhyAddr()),
                      workspace);
        t[1] = GetSystemCycle();
        uint64_t cursors[u::kMaxPeers]{};
        bool remaining = true;
        while (remaining) {
            remaining = false;
            for (uint32_t p = 0; p < sender.Count(); ++p) {
                const uint32_t peer = sender.Rank(p);
                if (!Edge(c, c.rank, peer)) continue;
                for (uint32_t visit = 0; visit < c.visit && cursors[p] < c.bytes; ++visit) {
                    const uint64_t n = c.bytes - cursors[p] < c.chunk ? c.bytes - cursors[p] : c.chunk;
                    sender.Put(
                        p,
                        reinterpret_cast<__gm__ uint8_t*>(workspace + RecvOffset(c) + c.rank * c.capacity + cursors[p]),
                        reinterpret_cast<__gm__ uint8_t*>(workspace + SendOffset(c) + peer * c.capacity + cursors[p]),
                        static_cast<uint32_t>(n));
                    cursors[p] += n;
                }
                remaining |= cursors[p] < c.bytes;
            }
        }
        sender.PublishActive(workspace, c);
        t[2] = GetSystemCycle();
        sender.Quiet();  // Release payload and marker sources before reuse.
        t[3] = GetSystemCycle();
    }
    const uint64_t deadline = GetSystemCycle() + u::kIncomingTimeoutCycles;
    for (uint32_t src = core; src < c.world; src += c.cores) {
        if (!Edge(c, src, c.rank)) continue;
        auto* address = u::Signal(workspace, 0, src);
        GlobalTensor<uint64_t> marker;
        marker.SetGlobalBuffer(address);
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
            if (marker.GetValue(0) == 1U) break;
            if (uint64_t(GetSystemCycle()) >= deadline) {
                AscendC::printf("[UDMA_BENCH_TIMEOUT] rank=%u core=%u source=%u mode=%u\n", c.rank, core, src, c.mode);
                trap();
            }
        }
    }
    t[4] = GetSystemCycle();
    SyncAll<true>();
    t[5] = GetSystemCycle();
    // Timing record copy and final global rendezvous are outside all intervals.
    // Each core owns a 64-byte row. Sender scratch has already been drained.
    auto record = ub.GetBufferByByte<uint64_t>(0);
    for (uint32_t i = 0; i < kTimes; ++i) record.SetValue(i, t[i]);
    record.SetValue(6, active ? 1U : 0U);
    record.SetValue(7, 0);
    GlobalTensor<uint64_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(timings) + core * 8);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID4);
    DataCopy(out, record, 8);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
    aclshmemi_sync_npu_v4<true>(0);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void bench_verify(GM_ADDR workspace, GM_ADDR errors,
                                                                                 Config c, uint64_t ffts)
{
    util_set_ffts_config(ffts);
    const uint32_t core = get_block_idx();
    for (uint32_t src = 0; src < c.world; ++src) {
        if (!Edge(c, src, c.rank)) continue;
        auto* row = reinterpret_cast<__gm__ uint64_t*>(workspace + RecvOffset(c) + src * c.capacity);
        asc_vf_call<Payload<true>>(dim3(kLanes), row, row, reinterpret_cast<__gm__ uint64_t*>(errors), c.bytes / 8, src,
                                   c.rank, c.epoch, core, c.cores, false, false);
        DrainSimt();
    }
}

extern "C" void udma_bench_prepare(void* stream, void* workspace, Config c, uint64_t ffts)
{
    bench_prepare<<<c.cores, nullptr, stream>>>(static_cast<uint8_t*>(workspace), c, ffts);
}
extern "C" void udma_bench_transfer(void* stream, void* workspace, void* times, Config c, uint64_t ffts)
{
    bench_transfer<<<c.cores, ascend_deepep::platform::kKernelUbBytes, stream>>>(static_cast<uint8_t*>(workspace),
                                                                                 static_cast<uint8_t*>(times), c, ffts);
}
extern "C" void udma_bench_verify(void* stream, void* workspace, void* errors, Config c, uint64_t ffts)
{
    bench_verify<<<c.cores, nullptr, stream>>>(static_cast<uint8_t*>(workspace), static_cast<uint8_t*>(errors), c,
                                               ffts);
}
