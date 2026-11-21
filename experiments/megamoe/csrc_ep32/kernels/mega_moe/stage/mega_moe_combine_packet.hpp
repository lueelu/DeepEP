// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_COMBINE_PACKET_H
#define MEGA_MOE_COMBINE_PACKET_H

#include "kernels/mega_moe_common/common/mega_moe_peermem.hpp"

namespace MegaMoeImpl {
using namespace AscendC;

// Treat BF16 pairs as raw uint32 words; no numerical conversion. Full packets
// acquire their marker through MTE2 padding. Only the short final packet needs
// a local scalar stamp, after MTE2 has finished touching its final 32B block.
// The caller owns a drained row slot and retires MTE3 before reusing that slot.
__aicore__ inline void PackGatewayCombineRow(LocalTensor<uint32_t> packed, GlobalTensor<uint32_t> source,
                                             uint32_t rowBytes, uint32_t sequence, TEventID event)
{
    constexpr uint32_t dataBytes = RAIL_PACKET_DATA_WORDS * sizeof(uint32_t);
    const uint32_t full = rowBytes / dataBytes;
    const uint32_t tail = rowBytes % dataBytes;
    if (full != 0U) {
        DataCopyPad<uint32_t, PaddingMode::Normal>(packed, source, {static_cast<uint16_t>(full), dataBytes, 0U, 0U, 0U},
                                                   {true, 0U, 1U, sequence});
    }
    if (tail != 0U) {
        DataCopyPad(packed[full * RAIL_PACKET_WORDS], source[full * RAIL_PACKET_DATA_WORDS], {1U, tail, 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        // Unused tail payload is ignored by the consumer, including any dummy
        // padding. Never read past the producer's actual BF16 source row.
        packed.SetValue(full * RAIL_PACKET_WORDS + RAIL_PACKET_DATA_WORDS, sequence);
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        // MTE2 -> S -> MTE3 already orders the full row and the final stamp.
        return;
    }
    SetFlag<HardEvent::MTE2_MTE3>(event);
    WaitFlag<HardEvent::MTE2_MTE3>(event);
}

// Compact MTE2 skips each marker, producing consecutive BF16 values. Its final
// block also copies unused tail bytes; the input slot reserves a full packet-row
// stride, while Cast/reduction consume exactly hidden elements.
__aicore__ inline void LoadGatewayCombineRow(LocalTensor<bfloat16_t> input, GlobalTensor<bfloat16_t> rows, uint64_t row,
                                             uint32_t hidden)
{
    const uint32_t rowBytes = CalcGatewayCombinePacketRowBytes(hidden);
    GlobalTensor<uint32_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rows.GetPhyAddr(0U)) +
                           row * (rowBytes / sizeof(uint32_t)));
    DataCopyPad<uint32_t, PaddingMode::Compact>(input.template ReinterpretCast<uint32_t>(), source,
                                                {static_cast<uint16_t>(rowBytes / RAIL_PACKET_BYTES),
                                                 RAIL_PACKET_DATA_WORDS * sizeof(uint32_t), sizeof(uint32_t), 0U, 0U},
                                                {false, 0U, 0U, 0U});
}
}  // namespace MegaMoeImpl
#endif
