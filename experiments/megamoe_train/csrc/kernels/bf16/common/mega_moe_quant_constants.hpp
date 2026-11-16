// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#ifndef MEGA_MOE_QUANT_CONSTANTS_HPP
#define MEGA_MOE_QUANT_CONSTANTS_HPP

// Shared by host tiling and device kernel code. Keep this header free of
// AscendC host-only and device-only dependencies.
#define DISPATCH_QUANT_MODE_NO_QUANT 0
#define DISPATCH_QUANT_MODE_MXFP 4
#define DISPATCH_QUANT_OUT_TYPE_E5M2 3
#define DISPATCH_QUANT_OUT_TYPE_E4M3FN 4

#endif  // MEGA_MOE_QUANT_CONSTANTS_HPP
