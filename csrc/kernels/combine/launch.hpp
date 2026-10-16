// Copyright (c) 2026, Lu Lu

#pragma once
#include "../notify/launch.hpp"
extern "C" ASCEND_DEEPEP_EXPORT void combine_reuse_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                             uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                             uint8_t* offsets, uint8_t* mask, uint8_t* weights,
                                                             uint8_t* weight_meta, uint8_t* tiling, uint32_t window);
extern "C" ASCEND_DEEPEP_EXPORT void combine_ep8_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                           uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                           uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges,
                                                           uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
                                                           uint8_t* tiling);
extern "C" ASCEND_DEEPEP_EXPORT void combine_ep8_profile_kernel_do(
    void* stream, uint8_t* workspace, uint8_t* input, uint8_t* forward, uint8_t* counts, uint8_t* backward,
    uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
    uint8_t* profile, uint64_t sample, uint8_t* chunk_profile, uint8_t* peer_profile, uint8_t* tiling);
extern "C" ASCEND_DEEPEP_EXPORT void combine_prepare_kernel_do(void* stream, uint8_t* workspace, uint8_t* tiling);
extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                       uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                       uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges,
                                                       uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
                                                       uint8_t* tiling);
extern "C" ASCEND_DEEPEP_EXPORT void combine_profile_kernel_do(
    void* stream, uint8_t* workspace, uint8_t* input, uint8_t* forward, uint8_t* counts, uint8_t* backward,
    uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
    uint8_t* profile, uint64_t sample, uint8_t* chunk_profile, uint8_t* peer_profile, uint8_t* tiling);
