#pragma once
#include "../common/ll_rdma_layout.h"
#include "../common/ll_group_policy.h"
#include "kernel_operator.h"
#ifdef DEEPEP_LL_RDMA
#include "device/gm2gm/engine/shmem_device_rdma.h"
#endif
namespace LlTransport {
using namespace AscendC;
using Args = DispatchDedup::CommArgs;
template<HardEvent E> __aicore__ inline void Sync(uint32_t id) {
    SetFlag<E>(id);
    WaitFlag<E>(id);
}
__aicore__ inline bool Rdma(const __gm__ Args* a) { return a->reservedConfig[0] == LlRdma::kBackend; }
__aicore__ inline bool SameHost(const __gm__ Args* a, uint32_t peer) {
    return DispatchDedup::LlSameHost(uint32_t(a->rank), peer,
        a->reserved0 ? a->reserved0 : DispatchDedup::kLlPhysicalRanksPerHost);
}
__aicore__ inline bool RemoteWrite(const __gm__ Args* a, uint32_t peer) { return Rdma(a) && !SameHost(a, peer); }
__aicore__ inline void Check(const __gm__ Args* a, uint32_t peer, uint64_t offset, uint64_t bytes) {
    if (peer >= uint32_t(a->rankSize) || offset > a->peerMemBytes || bytes > a->peerMemBytes - offset) Trap();
}
__aicore__ inline GM_ADDR RemoteAddress(const __gm__ Args* a, uint32_t peer, uint64_t offset) {
    return a->reservedAddresses[0] + uint64_t(peer) * 1024U + offset;
}
__aicore__ inline GM_ADDR Address(const __gm__ Args* a, uint32_t peer, uint64_t offset) {
    return RemoteWrite(a, peer) ? RemoteAddress(a, peer, offset) : a->peerMems[peer] + offset;
}
__aicore__ inline GlobalTensor<uint8_t> Tensor(GM_ADDR p, bool uncached=false) {
    GlobalTensor<uint8_t> t; t.SetGlobalBuffer(p);
    if (uncached) t.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return t;
}
__aicore__ inline GM_ADDR Stage(const __gm__ Args* a, uint32_t bytes) {
    if (GetBlockNum() != a->reservedConfig[1] || GetBlockIdx() >= a->reservedConfig[1] || bytes > LlRdma::kCoreBytes) Trap();
    return a->reservedAddresses[1] + uint64_t(GetBlockIdx()) * LlRdma::kCoreBytes;
}
__aicore__ inline void Put(const __gm__ Args* a, uint32_t peer, GM_ADDR dst, GM_ADDR src,
                           uint32_t bytes, LocalTensor<uint8_t> scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_put_nbi(dst, src, (__ubuf__ uint8_t*)scratch.GetPhyAddr(), bytes, peer, GetBlockIdx(), syncId);
#else
    Trap();
#endif
}
__aicore__ inline void Quiet(uint32_t peer, LocalTensor<uint8_t> scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_quiet(peer, GetBlockIdx(), (__ubuf__ uint8_t*)scratch.GetPhyAddr(), syncId);
#else
    Trap();
#endif
}
// Target RoCE SO orders all preceding fragments before the final fragment;
// each fragment is itself ordered. The last existing 32B flag is the tail.
// All-flag receive checks include it, so one WQE publishes the complete packet.
// Quiet still owns registered source reuse, not receiver consumption.
// See ROCE_SO_CONTRACT.md. No cross-WQE/QP ordering is assumed.
__aicore__ inline void Packet(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                              LocalTensor<uint8_t> packet, uint32_t bytes, uint32_t syncId=7U,
                              bool preservePacket=true) {
    Check(a, peer, offset, bytes);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {

        DataCopy(Tensor(dst,Rdma(a)), packet, bytes);

        return;
    }
    if (bytes == 0U || bytes % 512U || syncId >= 8U) { Trap(); return; }
    auto stage = Stage(a, bytes);

    PipeBarrier<PIPE_ALL>();
    DataCopy(Tensor(stage,true), packet, bytes);
    Sync<HardEvent::MTE3_S>(syncId); // registered GM complete before Scalar posts NIC WQEs

    Put(a, peer, dst, stage, bytes, packet, syncId);

    Quiet(peer, packet, syncId);

    if (preservePacket) {

        // Multi-destination callers retain the packet. Single-use callers
        // overwrite its body before reuse and need no scratch restoration.
        DataCopy(packet, Tensor(stage,true), 128U);
        Sync<HardEvent::MTE2_S>(syncId);
        PipeBarrier<PIPE_ALL>();

    }
}
// Strided 32B records. Count records publish only word 0 after words 1..7
// complete. Ready records carry no payload and are published after local stage
// producers have completed. scratch is a retired, independent >=128B UB view.
__aicore__ inline void Records(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                               LocalTensor<uint8_t> records, uint32_t count, uint32_t stride,
                               bool counts, LocalTensor<uint8_t> scratch) {
    if (!count) return;
    Check(a, peer, offset, uint64_t(count - 1U) * stride + 32U);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {
        DataCopyPad(Tensor(dst,Rdma(a)), records, {uint16_t(count), 32U, 0U, stride - 32U, 0U}); return;
    }
    auto stage = Stage(a, count * 32U);
    PipeBarrier<PIPE_ALL>();
    DataCopy(Tensor(stage,true), records, count * 32U);
    Sync<HardEvent::MTE3_S>(7U);
    if (counts) {
        for (uint32_t i = 0; i < count; ++i) Put(a, peer, dst + uint64_t(i)*stride + 4U, stage + i*32U + 4U, 28U, scratch);
        Quiet(peer, scratch);
    }
    for (uint32_t i = 0; i < count; ++i) Put(a, peer, dst + uint64_t(i)*stride, stage + i*32U, counts ? 4U : 32U, scratch);
    Quiet(peer, scratch);
}
// Partial members belong to one aligned logical group (4 or 16), always
// within one physical host. Cross-host reads are forbidden, including fallback.
__aicore__ inline void LoadLocalPeer(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                                    LocalTensor<uint16_t> output, uint32_t bytes) {
    Check(a, peer, offset, bytes);
    if (!SameHost(a, peer) || a->peerMems[peer] == nullptr) { Trap(); return; }
    DataCopyPad(output.ReinterpretCast<uint8_t>(), Tensor(a->peerMems[peer] + offset, Rdma(a)),
                {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
}
}
