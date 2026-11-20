// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once
#include <cstdint>

extern "C" void ep32_benchmark_sleep_do(uint32_t blocks, void* stream, uint64_t cycles);
extern "C" void ep32_benchmark_roce_barrier_do(void* stream, uint64_t ffts_addr);

// Host 和 Bisheng 共享唯一入口声明；FP8 scale 非空时绕过输入量化。
extern "C" void ep32_mega_moe_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context,
                                   uint8_t* x, uint8_t* x_scales, uint8_t* topk_ids, uint8_t* topk_weights,
                                   uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1, uint8_t* weight_scales2,
                                   uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
                                   uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums,
                                   uint8_t* workspace, uint8_t* timer, const uint8_t* tiling_host, uint64_t ffts_addr);
