#pragma once

#include "kernels/mega_moe_common/common/mega_moe_peermem.hpp"
#include "kernel_operator.h"

namespace MegaMoeImpl {
// Emit complete aligned packets from the same UB row as the local contiguous copy.
__simd_vf__ inline void PackDispatchVF(__ubuf__ uint32_t* src, __ubuf__ uint32_t* dst, uint32_t words, uint32_t packets,
                                       uint32_t sequence)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    constexpr uint32_t data = DISPATCH_PACKET_DATA / sizeof(uint32_t);
    auto full = CreateMask<uint32_t, MaskPattern::ALL>();
    for (uint32_t p = 0; p < packets; ++p) {
        for (uint32_t off = 0; off < RAIL_PACKET_WORDS; off += lanes) {
            RegTensor<uint32_t> v;
            RegTensor<int32_t> index;
            Arange(index, static_cast<int32_t>(off));
            Duplicate(v, 0U, full);
            uint32_t count = words - p * data;
            count = count > data ? data : count;
            count = count > off ? count - off : 0U;
            auto active = UpdateMask<uint32_t>(count);
            RegTensor<uint32_t> loaded;
            Gather(loaded, src + p * data, reinterpret_cast<RegTensor<uint32_t>&>(index), active);
            Add<uint32_t, MaskMergeMode::MERGING>(v, v, loaded, active);
            MaskReg marker;
            Compares<int32_t, AscendC::CMPMODE::EQ>(marker, index, static_cast<int32_t>(RAIL_PACKET_WORDS - 1), full);
            RegTensor<uint32_t> epoch;
            Duplicate(epoch, sequence, full);
            Add<uint32_t, MaskMergeMode::MERGING>(v, v, epoch, marker);
            StoreAlign(dst + p * RAIL_PACKET_WORDS + off, v, full);
        }
    }
}

}  // namespace MegaMoeImpl
