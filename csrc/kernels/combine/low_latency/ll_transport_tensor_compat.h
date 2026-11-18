#pragma once
#include "kernel_operator.h"
#include "../../dispatch/low_latency/device/ll_transport.h"
// Only Combine constructs tensor views. Dispatch includes the raw transport directly.
namespace LlTransport {
template <AscendC::HardEvent E> __aicore__ inline void Sync(uint32_t id) {
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}
__aicore__ inline AscendC::GlobalTensor<uint8_t> Tensor(GM_ADDR address, bool uncached = false) {
    AscendC::GlobalTensor<uint8_t> result;
    result.SetGlobalBuffer(address);
    if (uncached) result.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    return result;
}
__aicore__ inline void Put(const __gm__ Args* args, uint32_t peer, GM_ADDR dst, GM_ADDR src,
                           uint32_t bytes, AscendC::LocalTensor<uint8_t> scratch, uint32_t id = 7U) {
    Put(args, peer, dst, src, bytes, reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), id);
}
__aicore__ inline void Quiet(uint32_t peer, AscendC::LocalTensor<uint8_t> scratch, uint32_t id = 7U) {
    Quiet(peer, reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), id);
}
__aicore__ inline void Packet(const __gm__ Args* args, uint32_t peer, uint64_t offset,
    AscendC::LocalTensor<uint8_t> packet, uint32_t bytes, uint32_t id = 7U, bool preserve = true) {
    Packet(args, peer, offset, reinterpret_cast<__ubuf__ uint8_t*>(packet.GetPhyAddr()), bytes, id, preserve);
}
__aicore__ inline void Records(const __gm__ Args* args, uint32_t peer, uint64_t offset,
    AscendC::LocalTensor<uint8_t> records, uint32_t count, uint32_t stride, bool counts,
    AscendC::LocalTensor<uint8_t> scratch) {
    Records(args, peer, offset, reinterpret_cast<__ubuf__ uint8_t*>(records.GetPhyAddr()), count, stride,
            counts, reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()));
}
__aicore__ inline void LoadLocalPeer(const __gm__ Args* args, uint32_t peer, uint64_t offset,
                                    AscendC::LocalTensor<uint16_t> output, uint32_t bytes) {
    Check(args, peer, offset, bytes);
    if (!SameHost(args, peer) || !args->peerMems[peer]) { trap(); return; }
    LlDispatchC::Load(reinterpret_cast<__ubuf__ uint8_t*>(output.GetPhyAddr()),
        args->peerMems[peer] + offset, bytes, 1U, 0U, 0U, Rdma(args));
}
}  // namespace LlTransport
