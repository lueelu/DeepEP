#pragma once
#include "kernel_operator.h"
#include <deep_ep/comm/comm_args.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/comm/transport.hpp>
#include <deep_ep/common/ub.hpp>

// Only Combine constructs tensor views. Dispatch includes the raw transport directly.
namespace LlTransport {
using Args = deep_ep::CommArgs;
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
    deep_ep::comm::Put(deep_ep::comm::MakeTransportView(args), peer, dst, src, bytes,
                       reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), id);
}
__aicore__ inline void Quiet(uint32_t peer, AscendC::LocalTensor<uint8_t> scratch, uint32_t id = 7U) {
    deep_ep::comm::Quiet(peer, reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), id);
}
__aicore__ inline void Packet(const __gm__ Args* args, uint32_t peer, uint64_t offset,
    AscendC::LocalTensor<uint8_t> packet, uint32_t bytes, uint32_t id = 7U, bool preserve = true) {
    deep_ep::comm::Packet(deep_ep::comm::MakeTransportView(args), peer, offset,
                          reinterpret_cast<__ubuf__ uint8_t*>(packet.GetPhyAddr()), bytes, id, preserve);
}
__aicore__ inline void Records(const __gm__ Args* args, uint32_t peer, uint64_t offset,
    AscendC::LocalTensor<uint8_t> records, uint32_t count, uint32_t stride, bool counts,
    AscendC::LocalTensor<uint8_t> scratch) {
    deep_ep::comm::Records(deep_ep::comm::MakeTransportView(args), peer, offset,
                           reinterpret_cast<__ubuf__ uint8_t*>(records.GetPhyAddr()), count, stride,
                           counts, reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()));
}
__aicore__ inline void LoadLocalPeer(const __gm__ Args* args, uint32_t peer, uint64_t offset,
                                    AscendC::LocalTensor<uint16_t> output, uint32_t bytes) {
    const auto view = deep_ep::comm::MakeTransportView(args);
    deep_ep::comm::Check(view, peer, offset, bytes);
    auto* source = view.peer_ptr(peer);
    if (!view.same_host(peer) || !source) { trap(); return; }
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(output.GetPhyAddr())), reinterpret_cast<__gm__ uint8_t*>(source + offset), 1U, bytes, 0U, 0U, false, (view.is_rdma() ? asc_load_l2_cache_mode::NOTALLOC_CLEAN : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM), bytes + 0U, deep_ep::math::align(bytes, 32U) + 0U);
}
}  // namespace LlTransport
