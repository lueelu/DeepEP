// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "kernel_operator.h"
#include "layout/mega_moe_saved_meta.hpp"

namespace MegaMoeForwardSave {

constexpr uint32_t kCopyBufferBytes = 16 * 1024;

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
        AscendC::DataCopy(copyBuffer, src[coreOffset + copied], current);
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::DataCopy(dst[coreOffset + copied], copyBuffer, current);
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        copied += current;
    }
}

}  // namespace MegaMoeForwardSave
