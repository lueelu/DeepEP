// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

// Remote ordering is only assumed inside one aligned 512B MTE block. The
// generation at bytes 504..511 protects that block, never earlier blocks.
namespace DispatchAggregatePacket {
constexpr uint32_t kBytes = 512U, kFlag = 504U, kPollBlocks = 32U;
constexpr uint32_t kScaleRows = 2U, kWeightRows = 60U;

__aicore__ inline uint32_t ScaleData(uint32_t row)
{
    return row / 2U * kBytes + row % 2U * 224U;
}
__aicore__ inline uint32_t ScaleSlot(uint32_t row)
{
    return row / 2U * 128U + 112U + row % 2U;
}
__aicore__ inline uint32_t WeightMeta(uint32_t row)
{
    return row / kWeightRows * 128U + row % kWeightRows * 2U;
}
__aicore__ inline void Stamp(AscendC::LocalTensor<uint8_t> packet, uint32_t blocks, uint64_t generation)
{
    auto words = packet.ReinterpretCast<uint64_t>();
    for (uint32_t i = 0; i < blocks; ++i) words.SetValue(i * 64U + 63U, generation);
}

// Scratch is idle WQE UB after all sends have drained; no new TPipe allocation.
// Compare both halves of the generation, then vector-sum matching 1.0 flags.
// This avoids raw-generation overflow, FP32 rounding and stale-flag cancellation.
template <typename Scratch>
__aicore__ inline uint32_t ReadyCount(GM_ADDR blocks, uint32_t count, uint64_t generation, Scratch& scratch)
{
    auto tails = scratch.template GetWithOffset<int32_t>(256U, 0U);
    auto low = scratch.template GetWithOffset<int32_t>(64U, 1024U);
    auto high = scratch.template GetWithOffset<int32_t>(64U, 1280U);
    auto pick_low = scratch.template GetWithOffset<uint32_t>(8U, 1536U);
    auto pick_high = scratch.template GetWithOffset<uint32_t>(8U, 1568U);
    auto cmp_low = scratch.template GetWithOffset<uint8_t>(32U, 1600U);
    auto cmp_high = scratch.template GetWithOffset<uint8_t>(32U, 1632U);
    auto ones = scratch.template GetWithOffset<float>(64U, 1664U);
    auto matched = scratch.template GetWithOffset<float>(64U, 1920U);
    auto sum = scratch.template GetWithOffset<float>(8U, 2176U);
    auto work = scratch.template GetWithOffset<float>(64U, 2208U);
    // Padding lanes are initialized, but GatherMask below only counts real blocks.
    NativeFill(low, 0, 64U);
    NativeFill(high, 0, 64U);
    NativeFill(cmp_low.template ReinterpretCast<int32_t>(), 0, 8U);
    NativeFill(cmp_high.template ReinterpretCast<int32_t>(), 0, 8U);
    for (uint32_t i = 0; i < 8U; ++i) {
        pick_low.SetValue(i, 0x40404040U);
        pick_high.SetValue(i, 0x80808080U);
    }
    NativeFill(ones.template ReinterpretCast<int32_t>(), 0x3f800000, 64U);
    NativeLoadPacketTails(tails, blocks, count);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    uint64_t selected = 0;
    AscendC::GatherMask(low, tails, pick_low, true, count * 8U, {1U, 1U, 0U, 0U}, selected);
    AscendC::GatherMask(high, tails, pick_high, true, count * 8U, {1U, 1U, 0U, 0U}, selected);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compares(cmp_low, low, int32_t(uint32_t(generation)), AscendC::CMPMODE::EQ, 64U);
    AscendC::Compares(cmp_high, high, int32_t(uint32_t(generation >> 32U)), AscendC::CMPMODE::EQ, 64U);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::And(cmp_low.template ReinterpretCast<uint16_t>(), cmp_low.template ReinterpretCast<uint16_t>(),
                 cmp_high.template ReinterpretCast<uint16_t>(), 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GatherMask(matched, ones, cmp_low.template ReinterpretCast<uint32_t>(), true, count, {1U, 1U, 0U, 0U},
                        selected);
    DeepepWeightAggregate::WaitVector();
    uint32_t ready = 0;
    if (selected) {
        AscendC::ReduceSum(sum, matched, work, int32_t(selected));
        DeepepWeightAggregate::WaitVector();
        ready = uint32_t(sum.GetValue(0));
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    return ready;
}

template <typename Scratch>
__aicore__ inline uint64_t Wait(GM_ADDR region, uint32_t blocks, uint64_t generation, Scratch& scratch)
{
    uint64_t polls = 0;
    if (!blocks) return polls;
    uint32_t ready;
    do {
        ready = 0;
        ++polls;
        for (uint32_t i = 0; i < blocks; i += kPollBlocks) {
            const uint32_t n = blocks - i < kPollBlocks ? blocks - i : kPollBlocks;
            ready += ReadyCount(region + uint64_t(i) * kBytes, n, generation, scratch);
        }
    } while (ready != blocks);
    // The caller reloads payload AFTER this observation; polling snapshots are
    // never expanded (data might have arrived while a flag load was in flight).
    return polls;
}
}  // namespace DispatchAggregatePacket
