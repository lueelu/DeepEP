// Copyright (c) 2026, Lu Lu
// Modified by lishaoxun 2026

#pragma once
#include "../notify/launch.hpp"
extern "C" ASCEND_DEEPEP_EXPORT void combine_prepare_kernel_do(void* stream, uint8_t* workspace, uint8_t* tiling);
extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(void* stream, uint8_t* workspace, uint8_t* forward,
                                                       uint8_t* counts, uint8_t* backward, uint8_t* backward_counts,
                                                       uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights,
                                                       uint8_t* weight_meta, uint8_t* tiling);
