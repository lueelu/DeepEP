// Copyright (c) 2026, Lu Lu

#pragma once
#include <ubx/kernels/ll_transport.h>
namespace UbxCombine {
// The caller initializes every 32B flag once and owns MTE3->MTE2 reuse.
// GM->UB dstStride is in 32B blocks. Payload DMA never targets a flag.
__aicore__ inline void LoadDirectWire(AscendC::LocalTensor<uint16_t> packet,
                                      AscendC::GlobalTensor<uint16_t> expert, uint32_t bytes)
{
    using namespace AscendC;
    const uint32_t full = bytes / 480U, tail = bytes % 480U;
    if (full) DataCopyPad(packet, expert, {uint16_t(full), 480U, 0U, 1U, 0U},
                          {false, 0U, 0U, uint16_t(0)});
    if (tail) DataCopyPad(packet[full * 256U], expert[full * 240U], {1U, tail, 0U, 0U, 0U},
                          {false, 0U, 0U, uint16_t(0)});
}
__aicore__ inline void LoadStageWire(GM_ADDR address,
                                     AscendC::LocalTensor<uint16_t> out, uint32_t bytes)
{
    using namespace AscendC;
    const uint32_t full = bytes / 480U, tail = bytes % 480U;
    auto src = UbxTransport::Tensor(address, true);
    auto dst = out.ReinterpretCast<uint8_t>();
    if (full) DataCopyPad(dst, src, {uint16_t(full), 480U, 32U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
    if (tail) DataCopyPad(dst[full * 480U], src[full * 512U], {1U, tail, 0U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
}
}  // namespace UbxCombine
