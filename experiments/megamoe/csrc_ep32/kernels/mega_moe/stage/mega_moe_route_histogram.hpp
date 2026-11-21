// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_ROUTE_HISTOGRAM_H
#define MEGA_MOE_ROUTE_HISTOGRAM_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// For a 32-bit route ID, only its low byte should count. Map out-of-range
// IDs to bin 127, and all three unused bytes to 255 (outside BIN0).
__simd_vf__ inline void CountRouteSourceVF(__ubuf__ uint32_t* ids, __ubuf__ uint16_t* output, uint32_t count,
                                           uint32_t firstExpert, uint32_t sources = 1U, uint32_t stride = 0U)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = CreateMask<uint32_t, MaskPattern::ALL>();
    auto binsMask = CreateMask<uint16_t, MaskPattern::ALL>();
    for (uint32_t source = 0; source < sources; ++source) {
        RegTensor<uint16_t> bins;
        Duplicate(bins, uint16_t(0), binsMask);
        for (uint32_t start = 0; start < count; start += lanes) {
            RegTensor<uint32_t> values;
            LoadAlign(values, ids + source * stride + start);
            Adds(values, values, 0U - firstExpert, full);
            Mins(values, values, 127U, full);
            Adds(values, values, 0xffffff00U, full);
            uint32_t activeBytes = (count - start) * 4U;
            auto active = UpdateMask<uint8_t>(activeBytes);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(
                bins, reinterpret_cast<RegTensor<uint8_t>&>(values), active);
        }
        StoreAlign(output + source * 128U, bins, binsMask);
    }
}

__simd_vf__ inline void TransposeRouteBinsVF(__ubuf__ uint32_t* pairs, __ubuf__ uint32_t* table, uint32_t first,
                                             uint32_t experts)
{
    using namespace AscendC::Reg;
    uint32_t n = 16U;
    auto active = UpdateMask<uint32_t>(n);
    RegTensor<int32_t> offsets;
    Arange(offsets, 0);
    Muls(offsets, offsets, 64, active);
    RegTensor<uint32_t> low;
    Duplicate(low, 65535U, active);
    for (uint32_t e = 0; e < experts; ++e) {
        RegTensor<uint32_t> values;
        Gather(values, pairs + e / 2U, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
        if (e & 1U)
            ShiftRights(values, values, static_cast<int16_t>(16), active);
        else
            And(values, values, low, active);
        StoreAlign(table + e * 32U + first, values, active);
    }
}

}  // namespace MegaMoeImpl
#endif
