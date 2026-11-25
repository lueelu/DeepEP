// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

// Runtime MegaMoE device compilation. The wheel no longer carries prebuilt
// MegaMoE device libraries: every variant is compiled through DeepJIT from the
// kernel sources bundled under deep_ep/_kernel_src, and each binary also
// publishes its SuperKernel bind record.
#pragma once

#include <cstdint>
#include <string>

namespace MegaMoeUrma {
struct Arguments;
}

namespace deepep::megamoe {

struct MteLaunchArgs;

// Called once from Python after the extension is imported. `package_root` is
// the deep_ep package directory that owns `_kernel_src`; `ascend_home` may be
// empty, in which case the active CANN installation is discovered.
void init_mega_moe_jit(const std::string& package_root, const std::string& ascend_home);

void launch_mega_moe_jit(uint32_t block_dim, uint32_t ub_bytes, void* stream, const MteLaunchArgs& args);
void launch_mega_moe_urma_jit(uint32_t block_dim, uint32_t ub_bytes, void* stream,
                              const MegaMoeUrma::Arguments* args);

}  // namespace deepep::megamoe
