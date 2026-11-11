// Copyright (c) 2026, Lu Lu

#pragma once
#include "ll_transport.h"
namespace UbxCombine {
// Shared workspace may later be scanned before dispatch prefix is known.
// Never put raw expert bytes into the universal 32B flag positions.
__aicore__ inline void StoreStageWire(AscendC::GlobalTensor<uint16_t> dst, AscendC::LocalTensor<uint16_t> src,
                                      AscendC::LocalTensor<float> zeros, uint32_t bytes)
{
    using namespace AscendC;
    const uint32_t full = bytes / 480U, tail = bytes % 480U, blocks = (bytes + 479U) / 480U;
    if (full) DataCopyPad(dst, src, {uint16_t(full), 480U, 0U, 32U, 0U});
    if (tail) DataCopyPad(dst[full * 256U], src[full * 240U], {1U, tail, 0U, 0U, 0U});
    DataCopyPad(dst[240U], zeros.ReinterpretCast<uint16_t>(), {uint16_t(blocks), 32U, 0U, 480U, 0U});
}
__aicore__ inline void LoadStageWire(const __gm__ UbxDispatch::CommArgs* args, uint32_t peer, uint64_t offset,
                                     AscendC::LocalTensor<uint16_t> out, uint32_t bytes)
{
    using namespace AscendC;
    const uint32_t full = bytes / 480U, tail = bytes % 480U;
    UbxTransport::Check(args, peer, offset, uint64_t((bytes + 479U) / 480U) * 512U);
    if (!UbxTransport::SameHost(args, peer) || args->peerMems[peer] == nullptr) {
        Trap();
        return;
    }
    auto src = UbxTransport::Tensor(args->peerMems[peer] + offset, false);
    auto dst = out.ReinterpretCast<uint8_t>();
    if (full) DataCopyPad(dst, src, {uint16_t(full), 480U, 32U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
    if (tail) DataCopyPad(dst[full * 480U], src[full * 512U], {1U, tail, 0U, 0U, 0U}, {false, 0U, 0U, uint8_t(0)});
}
}  // namespace UbxCombine
