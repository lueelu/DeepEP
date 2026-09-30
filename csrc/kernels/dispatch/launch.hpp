// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once
#include <cstdint>
#include "../notify/launch.hpp"
extern "C" ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(void* stream, uint8_t* workspace, uint8_t* destinations,
                                                        uint8_t* forward, uint8_t* counts, uint8_t* weights,
                                                        uint8_t* scales, uint8_t* tiling);
