// Copyright (c) 2026, Lu Lu

#pragma once
#include "ll_rdma_layout.h"
#include "ll_group_policy.h"
#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
namespace UbxTransport {
using namespace AscendC;
using Args = UbxDispatch::CommArgs;
template <HardEvent E>
__aicore__ inline void Sync(uint32_t id)
{
    SetFlag<E>(id);
    WaitFlag<E>(id);
}
__aicore__ inline bool Rdma(const __gm__ Args* a)
{
    return a->reservedConfig[0] == UbxRdma::kBackend;
}
__aicore__ inline bool SameHost(const __gm__ Args* a, uint32_t peer)
{
    return UbxDispatch::LlSameHost(uint32_t(a->rank), peer,
                                   a->reserved0 ? a->reserved0 : UbxDispatch::kLlPhysicalRanksPerHost);
}
__aicore__ inline bool RemoteWrite(const __gm__ Args* a, uint32_t peer)
{
    return Rdma(a) && !SameHost(a, peer);
}
__aicore__ inline void Check(const __gm__ Args* a, uint32_t peer, uint64_t offset, uint64_t bytes)
{
    if (peer >= uint32_t(a->rankSize) || offset > a->peerMemBytes || bytes > a->peerMemBytes - offset) Trap();
}
__aicore__ inline GM_ADDR RemoteAddress(const __gm__ Args* a, uint32_t peer, uint64_t offset)
{
    return a->reservedAddresses[0] + uint64_t(peer) * 1024U + offset;
}
__aicore__ inline GM_ADDR Address(const __gm__ Args* a, uint32_t peer, uint64_t offset)
{
    return RemoteWrite(a, peer) ? RemoteAddress(a, peer, offset) : a->peerMems[peer] + offset;
}
__aicore__ inline GlobalTensor<uint8_t> Tensor(GM_ADDR p, bool uncached = false)
{
    GlobalTensor<uint8_t> t;
    t.SetGlobalBuffer(p);
    if (uncached) t.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return t;
}
__aicore__ inline GM_ADDR Stage(const __gm__ Args* a, uint32_t bytes)
{
    if (GetBlockNum() != a->reservedConfig[1] || GetBlockIdx() >= a->reservedConfig[1] || bytes > UbxRdma::kCoreBytes)
        Trap();
    return a->reservedAddresses[1] + uint64_t(GetBlockIdx()) * UbxRdma::kCoreBytes;
}
__aicore__ inline void Put(const __gm__ Args* a, uint32_t peer, GM_ADDR dst, GM_ADDR src, uint32_t bytes,
                           LocalTensor<uint8_t> scratch, uint32_t syncId = 7U)
{
    aclshmemx_roce_qp_put_nbi(dst, src, (__ubuf__ uint8_t*)scratch.GetPhyAddr(), bytes, peer, GetBlockIdx(), syncId);
}
__aicore__ inline void Quiet(uint32_t peer, LocalTensor<uint8_t> scratch, uint32_t syncId = 7U)
{
    aclshmemx_roce_qp_quiet(peer, GetBlockIdx(), (__ubuf__ uint8_t*)scratch.GetPhyAddr(), syncId);
}
}  // namespace UbxTransport
