// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once
#include <cstdint>

namespace MegaMoeUrma {
struct Arguments;
}
namespace deepep::megamoe {
enum class Backend : uint8_t { Mte, Urma };
// Host-only launch description; the device argument blocks are assembled by
// the runtime JIT launcher from these fields.
struct MteLaunchArgs {
    bool is_a8w4 = false;
    uint8_t *context = nullptr, *x = nullptr, *x_scales = nullptr, *topk_ids = nullptr, *topk_weights = nullptr;
    uint8_t *weight1 = nullptr, *weight2 = nullptr, *weight_scales1 = nullptr, *weight_scales2 = nullptr;
    uint8_t *shared_weight1 = nullptr, *shared_weight2 = nullptr, *shared_weight_scales1 = nullptr,
            *shared_weight_scales2 = nullptr;
    uint8_t *y = nullptr, *expert_token_nums = nullptr, *workspace = nullptr, *timer = nullptr;
    const uint8_t* tiling = nullptr;
    uint64_t ffts = 0;
};
void launch_kernel(Backend backend, uint32_t block_dim, uint32_t ub_bytes, void* stream, const MteLaunchArgs& mte,
                   const MegaMoeUrma::Arguments* urma);
}  // namespace deepep::megamoe
