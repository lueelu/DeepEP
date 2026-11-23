// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"
#include "common/mega_moe_bf16_saved_meta.hpp"

namespace MegaMoeForwardSave {

constexpr uint32_t kCopyBufferBytes = 16 * 1024;

// Backward P0 uses static UB without a TPipe. Reserve event 3 for the synchronous
// copy fences; the Dispatch ping-pong slots independently use events 0 and 1.
template <AscendC::HardEvent event>
__aicore__ inline void SyncCopyStage()
{
    AscendC::SetFlag<event>(EVENT_ID3);
    AscendC::WaitFlag<event>(EVENT_ID3);
}

__aicore__ inline void CopyContiguous(GM_ADDR dstAddr, GM_ADDR srcAddr, uint64_t byteCount, uint32_t coreIdx,
                                      uint32_t coreCount, AscendC::LocalTensor<uint8_t> copyBuffer)
{
    const uint64_t unalignedBytesPerCore = (byteCount + coreCount - 1) / coreCount;
    const uint64_t bytesPerCore = (unalignedBytesPerCore + MegaMoeMeta::kMetaAlignment - 1) /
                                  MegaMoeMeta::kMetaAlignment * MegaMoeMeta::kMetaAlignment;
    const uint64_t coreOffset = coreIdx * bytesPerCore;
    if (coreOffset >= byteCount) {
        return;
    }
    uint64_t remaining = byteCount - coreOffset;
    if (remaining > bytesPerCore) {
        remaining = bytesPerCore;
    }

    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcAddr));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstAddr));

    uint64_t copied = 0;
    while (copied < remaining) {
        const uint32_t current =
            static_cast<uint32_t>(remaining - copied > kCopyBufferBytes ? kCopyBufferBytes : remaining - copied);
        if (current % 32U == 0) {
            AscendC::DataCopy(copyBuffer, src[coreOffset + copied], current);
            SyncCopyStage<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(dst[coreOffset + copied], copyBuffer, current);
        } else {
            const AscendC::DataCopyExtParams span{1U, current, 0U, 0U, 0U};
            AscendC::DataCopyPad(copyBuffer, src[coreOffset + copied], span,
                                 AscendC::DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
            SyncCopyStage<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopyPad(dst[coreOffset + copied], copyBuffer, span);
        }
        SyncCopyStage<AscendC::HardEvent::MTE3_MTE2>();
        copied += current;
    }
}

}  // namespace MegaMoeForwardSave
