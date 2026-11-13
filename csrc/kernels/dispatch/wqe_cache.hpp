// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Clos-only scratch. Cache one 32-row descriptor batch per destination server;
// each visit still submits its original DB batch before rotating to the next peer.
// No payload or queue state is cached, and valid bits reset on every invocation.
constexpr uint32_t kClosCacheBase = 2048U, kClosCacheStride = 384U;
constexpr uint32_t kClosCacheEntries = 32U, kClosBuildRows = 32U;
constexpr uint32_t kClosBuiltWqesBase = 16384U;
constexpr uint32_t kClosBuiltWqesStride = kClosBuildRows * kUdmaWqeBytes;
constexpr uint32_t kClosBuiltWqesEnd = kClosBuiltWqesBase + kClosCacheEntries * kClosBuiltWqesStride;
static_assert(kClosCacheBase + kClosCacheEntries * kClosCacheStride <= kClosBuiltWqesBase);
static_assert(kClosBuiltWqesEnd <= kSlotSelectUbBytes);

template <bool Profile>
__aicore__ inline LocalTensor<uint64_t> PrepareClosBuiltWqes(DeepepSecondaryState& s, uint32_t peer, uint32_t qp,
                                                             __gm__ uint32_t* first, uint32_t remaining,
                                                             uint32_t emit_count, uint64_t source, uint64_t target,
                                                             __ubuf__ uint8_t* scratch,
                                                             TBuf<QuePosition::VECCALC>& slots)
{
    // EP512 keeps the bounded per-visit builder; doubling this cache exceeds UB.
    if (s.world > kClosCacheEntries * 8U) {
        const uint64_t load_start = DetailClock<Profile>(s.detail_mode);
        auto rows = slots.Get<uint32_t>();
        NativeLoad(rows, first, emit_count * 2U);
        const uint64_t build_start = DetailClock<Profile>(s.detail_mode);
        DetailAdd<Profile>(s.detail, 7U, build_start - load_start);
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
        deepep::comm::tile::FillUdmaRemoteSqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), reinterpret_cast<__gm__ uint8_t*>(target),
            aclshmemi_udma_get_mem_info(aclshmemi_udma_qp_info_fetch(), slot, qp), 0U);
        auto output = slots.GetWithOffset<uint64_t>(128U, 256U);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        asc_vf_call<DispatchBuildPeerTableWqesVf>(dim3(32U), reinterpret_cast<__ubuf__ uint64_t*>(output.GetPhyAddr()),
                                                  header, reinterpret_cast<__ubuf__ uint32_t*>(rows.GetPhyAddr()),
                                                  emit_count, s.row_bytes, source, target);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        DetailAdd<Profile>(s.detail, 8U, DetailClock<Profile>(s.detail_mode) - build_start);
        return output;
    }
    const uint32_t entry = peer / 8U;
    auto storage = slots.GetWithOffset<uint64_t>(kClosCacheStride / 8U, kClosCacheBase + entry * kClosCacheStride);
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(storage.GetPhyAddr());
    auto* tag = header + 8U;
    const uint64_t key = (uint64_t(qp) << 32U) | peer;
    const uint32_t bit = 1U << entry;
    // The peer can change without its server index changing. QP and live target
    // are also part of the key; phase changes invalidate the row range below.
    if (!(s.wqe_cache_valid & bit) || tag[0] != key || header[5] != target) {
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        deepep::comm::tile::FillUdmaRemoteSqe<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), reinterpret_cast<__gm__ uint8_t*>(target),
            aclshmemi_udma_get_mem_info(aclshmemi_udma_qp_info_fetch(), slot, qp), 0U);
        tag[0] = key;
        tag[1] = 0U;
        tag[2] = 0U;
        tag[3] = 0U;
        tag[4] = 0U;
        s.wqe_cache_valid |= bit;
    }
    const uint64_t address = reinterpret_cast<uint64_t>(first);
    auto output = slots.GetWithOffset<uint64_t>(kClosBuildRows * 8U, kClosBuiltWqesBase + entry * kClosBuiltWqesStride);
    if (address < tag[1] || address + uint64_t(emit_count) * 8U > tag[2] || tag[3] != source || tag[4] != s.row_bytes) {
        const uint32_t count = remaining < kClosBuildRows ? remaining : kClosBuildRows;
        auto rows = storage[16U].ReinterpretCast<uint32_t>();
        const uint64_t load_start = DetailClock<Profile>(s.detail_mode);
        NativeLoad(rows, first, count * 2U);
        const uint64_t build_start = DetailClock<Profile>(s.detail_mode);
        DetailAdd<Profile>(s.detail, 7U, build_start - load_start);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        asc_vf_call<DispatchBuildPeerTableWqesVf>(dim3(32U), reinterpret_cast<__ubuf__ uint64_t*>(output.GetPhyAddr()),
                                                  header, reinterpret_cast<__ubuf__ uint32_t*>(rows.GetPhyAddr()),
                                                  count, s.row_bytes, source, target);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        DetailAdd<Profile>(s.detail, 8U, DetailClock<Profile>(s.detail_mode) - build_start);
        tag[1] = address;
        tag[2] = address + uint64_t(count) * 8U;
        tag[3] = source;
        tag[4] = s.row_bytes;
    }
    // One 8-byte row pair maps to eight uint64 words in the WQE cache.
    return output[address - tag[1]];
}
