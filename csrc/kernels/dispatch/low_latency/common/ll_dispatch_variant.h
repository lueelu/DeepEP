// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>

namespace DispatchDedup {
// Internal dispatch classification, independent of the shared-memory
// communication ABI. Numeric format values are checked against the public
// enums by the Host binding.
constexpr uint8_t kLlDispatchFp16 = 3;
constexpr uint8_t kLlDispatchBf16 = 11;
constexpr uint8_t kLlDispatchE4M3 = 15;
constexpr uint8_t kLlDispatchE5M2 = 16;
constexpr uint8_t kLlDispatchPlain16 = 0;
constexpr uint8_t kLlDispatchQuantize16 = 1;
constexpr uint8_t kLlDispatchPrequantized = 2;
constexpr uint32_t kLlDispatchInvalidVariant = 0xffffffffU;

// The entry kernel has two compile-time input element classes; the group size
// is the second axis. Plain16, Quantize16 with E4M3/E5M2 and
// PrequantizedFp8Packs are runtime tiling fields of that one kernel, so the
// format matrix never multiplies translation units. Prequantized rows are
// opaque FP8 byte packs whose element type is never read; they use the
// FP16-class kernel because both 16-bit families have the same width.
constexpr uint32_t kLlDispatchInputFp16 = 0U;
constexpr uint32_t kLlDispatchInputBf16 = 1U;

// Returns the compile-time kernel class for a tiling format, or
// kLlDispatchInvalidVariant when the format is not admitted. Host and device
// evaluate the same table before launching or running the kernel.
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
constexpr uint32_t LlDispatchInputClass(uint8_t dtype, uint8_t quantMode,
                                        uint8_t expandXOutDtype, uint8_t inputMode)
{
    if (dtype == kLlDispatchE4M3 && quantMode == 4U &&
        expandXOutDtype == kLlDispatchE4M3 && inputMode == kLlDispatchPrequantized) {
        return kLlDispatchInputFp16;
    }
    if (dtype == kLlDispatchFp16 || dtype == kLlDispatchBf16) {
        const uint32_t inputClass = dtype == kLlDispatchFp16 ? kLlDispatchInputFp16 : kLlDispatchInputBf16;
        if (quantMode == 0U && inputMode == kLlDispatchPlain16 &&
            (expandXOutDtype == dtype || expandXOutDtype == 255U)) {
            return inputClass;
        }
        if (quantMode == 4U && inputMode == kLlDispatchQuantize16 &&
            (expandXOutDtype == kLlDispatchE4M3 || expandXOutDtype == kLlDispatchE5M2)) {
            return inputClass;
        }
    }
    return kLlDispatchInvalidVariant;
}
}  // namespace DispatchDedup
