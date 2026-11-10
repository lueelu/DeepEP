// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#include "kernel_launch.hpp"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include <stdexcept>

namespace deepep::megamoe {
void launch_kernel(Backend backend, uint32_t block_dim, uint32_t ub_bytes, void* stream, const PodLaunchArgs& p,
                   const MegaMoeUrma::Arguments* ubx)
{
    if (backend == Backend::Ubx) {
        if (!ubx) throw std::invalid_argument("Missing UBX device arguments");
        mega_moe_urma_kernel_do(block_dim, ub_bytes, stream, ubx);
        return;
    }
    mega_moe_kernel_do(block_dim, ub_bytes, stream, p.is_a8w4, p.context, p.x, p.x_scales, p.topk_ids, p.topk_weights,
                       p.weight1, p.weight2, p.weight_scales1, p.weight_scales2, p.shared_weight1, p.shared_weight2,
                       p.shared_weight_scales1, p.shared_weight_scales2, p.y, p.expert_token_nums, p.workspace, p.timer,
                       p.tiling, p.ffts);
}
}  // namespace deepep::megamoe
