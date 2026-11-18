#pragma once
#include "../common/ll_rdma_layout.h"
#include "../common/ll_group_policy.h"
#include "ll_count_c_api.h"
#include "ll_dispatch_c_api.h"
#ifdef DEEPEP_LL_RDMA
#include "device/gm2gm/engine/shmem_device_rdma.h"
#endif
namespace LlTransport {
using Args = DispatchDedup::CommArgs;
__aicore__ inline bool Rdma(const __gm__ Args* a) { return a->reservedConfig[0] == LlRdma::kBackend; }
__aicore__ inline bool SameHost(const __gm__ Args* a, uint32_t peer) {
    return DispatchDedup::LlSameHost(uint32_t(a->rank), peer,
        a->reserved0 ? a->reserved0 : DispatchDedup::kLlPhysicalRanksPerHost);
}
__aicore__ inline bool RemoteWrite(const __gm__ Args* a, uint32_t peer) { return Rdma(a) && !SameHost(a, peer); }
__aicore__ inline void Check(const __gm__ Args* a, uint32_t peer, uint64_t offset, uint64_t bytes) {
    if (peer >= uint32_t(a->rankSize) || offset > a->peerMemBytes || bytes > a->peerMemBytes - offset) trap();
}
__aicore__ inline GM_ADDR RemoteAddress(const __gm__ Args* a, uint32_t peer, uint64_t offset) {
    return a->reservedAddresses[0] + uint64_t(peer) * 1024U + offset;
}
__aicore__ inline GM_ADDR Address(const __gm__ Args* a, uint32_t peer, uint64_t offset) {
    return RemoteWrite(a, peer) ? RemoteAddress(a, peer, offset) : a->peerMems[peer] + offset;
}
__aicore__ inline GM_ADDR Stage(const __gm__ Args* a, uint32_t bytes) {
    if (asc_get_block_num() != a->reservedConfig[1] || asc_get_block_idx() >= a->reservedConfig[1] ||
        bytes > LlRdma::kCoreBytes) trap();
    return a->reservedAddresses[1] + uint64_t(asc_get_block_idx()) * LlRdma::kCoreBytes;
}
__aicore__ inline void Put(const __gm__ Args* a, uint32_t peer, GM_ADDR dst, GM_ADDR src,
                           uint32_t bytes, __ubuf__ uint8_t* scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_put_nbi(dst, src, scratch, bytes, peer, asc_get_block_idx(), syncId);
#else
    trap();
#endif
}
__aicore__ inline void Quiet(uint32_t peer, __ubuf__ uint8_t* scratch, uint32_t syncId=7U) {
#ifdef DEEPEP_LL_RDMA
    aclshmemx_roce_qp_quiet(peer, asc_get_block_idx(), scratch, syncId);
#else
    trap();
#endif
}
// Raw dispatch path. The SDK borrows the first 128B after the staged DMA completes.
__aicore__ inline void Packet(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                              __ubuf__ uint8_t* packet, uint32_t bytes, uint32_t syncId = 7U,
                              bool preservePacket = true) {
    Check(a, peer, offset, bytes);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {
        LlDispatchC::Store(dst, packet, bytes, 1U, 0U, 0U, Rdma(a));
        return;
    }
    if (!bytes || bytes % 512U || syncId >= 8U) { trap(); return; }
    auto stage = Stage(a, bytes);
    asc_sync_pipe(PIPE_ALL);
    LlDispatchC::Store(stage, packet, bytes, 1U, 0U, 0U, true);
    LlDispatchC::Sync<PIPE_MTE3, PIPE_S>(event_t(syncId));
    Put(a, peer, dst, stage, bytes, packet, syncId);
    Quiet(peer, packet, syncId);
    if (preservePacket) {
        LlDispatchC::Load(packet, stage, 128U, 1U, 0U, 0U, true);
        LlDispatchC::Sync<PIPE_MTE2, PIPE_S>(event_t(syncId));
        asc_sync_pipe(PIPE_ALL);
    }
}
// Count-only raw path, preserving the body-before-ready fallback for RDMA.
__aicore__ inline void Records(const __gm__ Args* a, uint32_t peer, uint64_t offset,
                               __ubuf__ uint8_t* records, uint32_t count, uint32_t stride,
                               bool counts, __ubuf__ uint8_t* scratch) {
    if (!count) return;
    Check(a, peer, offset, uint64_t(count - 1U) * stride + 32U);
    auto dst = Address(a, peer, offset);
    if (!RemoteWrite(a, peer)) {
        LlCountC::Store(dst, records, 32U, count, stride - 32U, Rdma(a));
        return;
    }
    auto stage = Stage(a, count * 32U);
    asc_sync_pipe(PIPE_ALL);
    LlCountC::Store(stage, records, count * 32U, 1U, 0U, true);
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
    if (counts) {
        for (uint32_t i = 0; i < count; ++i)
            Put(a, peer, dst + uint64_t(i)*stride + 4U, stage + i*32U + 4U, 28U, scratch);
        Quiet(peer, scratch);
    }
    for (uint32_t i = 0; i < count; ++i)
        Put(a, peer, dst + uint64_t(i)*stride, stage + i*32U, counts ? 4U : 32U, scratch);
    Quiet(peer, scratch);
}
}
