// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

// Ascend 950 GetSystemCycle is a system counter (1000 ticks/us), not core MHz.
// All intervals are subtracted within one AIV. Cross-device clocks are never joined.
// Separate template instantiations keep reads/stores out of the clean kernel.
template <bool Enabled>
__aicore__ inline void DispatchProfileMark(uint64_t address, uint32_t word)
{
    if constexpr (Enabled) {
        reinterpret_cast<__gm__ uint64_t*>(address)[word] = AscendC::GetSystemCycle();
    }
}
template <bool Enabled>
__aicore__ inline uint64_t DispatchProfileClock()
{
    if constexpr (Enabled) return AscendC::GetSystemCycle();
    return 0U;
}

// Detail words: 0 data WQEs, 1 data DBs, 2 short batches, 3 SQ wraps,
// 4 SO submissions, 5 planned WQEs, 6 planned DBs; 7 route load ticks,
// 8 WQE build+existing fences, 9 SQ copy, 10 DB helper, 11 peer-ready;
// 12 finish quiet, 13 local join, 14 done publish, 15 remote wait, 16 final join;
// 17 batch, 18 mode (1 counters, 2 timed), 19 generation, 20 magic,
// 21 rank, 22 core, 23 ABI; 24/25 relay/direct WQEs, 26/27 relay/direct DBs.
template <bool Enabled>
struct DispatchDetail {
    __aicore__ uint64_t* data()
    {
        return nullptr;
    }
};
template <>
struct DispatchDetail<true> {
    uint64_t words[32]{};
    __aicore__ uint64_t* data()
    {
        return words;
    }
};
template <bool Enabled>
__aicore__ inline void DetailAdd(uint64_t* d, uint32_t word, uint64_t value)
{
    if constexpr (Enabled) {
        d[word] += value;
    }
}
template <bool Enabled>
__aicore__ inline uint64_t DetailClock(uint32_t mode)
{
    if constexpr (Enabled) {
        if (mode == 2U) return AscendC::GetSystemCycle();
    }
    return 0U;
}
