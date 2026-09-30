// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
namespace DeepepWeightAggregate {
template <bool Profile>
__aicore__ inline uint64_t Clock()
{
    if constexpr (Profile) {
        return AscendC::GetSystemCycle();
    }
    return 0U;
}

__aicore__ inline void WaitVector()
{
    // UB-only decode/compaction need V_S, not a full DDR barrier per scan tile.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

// Protect local UB reuse. Remote readiness requires every 512B packet flag;
// neither a local completion nor header arrival proves payload readiness.
__aicore__ inline void WaitLocalPut()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    // Subsequent GatherMask writes use V; order those after the scalar wait too.
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
}

// The caller owns this UB exclusively and drains both payload planes together.
template <typename T>
__aicore__ inline void Put(__gm__ T* destination, AscendC::LocalTensor<T> source, uint32_t count, uint32_t peer)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
    if (peer == uint32_t(aclshmem_my_pe())) {
        NativeStore(destination, source, count);
    } else {
        aclshmemx_mte_put_nbi<uint8_t>(reinterpret_cast<__gm__ uint8_t*>(destination),
                                       reinterpret_cast<__ubuf__ uint8_t*>(source.GetPhyAddr()), count * sizeof(T),
                                       peer, EVENT_ID1);
    }
}

}  // namespace DeepepWeightAggregate
