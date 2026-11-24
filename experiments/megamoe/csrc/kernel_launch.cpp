// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#include "kernel_launch.hpp"
#include "jit/mega_moe_jit.hpp"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include <stdexcept>

namespace deepep::megamoe {
void launch_kernel(Backend backend, uint32_t block_dim, uint32_t ub_bytes, void* stream, const MteLaunchArgs& p,
                   const MegaMoeUrma::Arguments* urma)
{
    // Both backends compile their device code at runtime; no prebuilt MegaMoE
    // device library is packaged any more.
    if (backend == Backend::Urma) {
        launch_mega_moe_urma_jit(block_dim, ub_bytes, stream, urma);
        return;
    }
    launch_mega_moe_jit(block_dim, ub_bytes, stream, p);
}
}  // namespace deepep::megamoe
