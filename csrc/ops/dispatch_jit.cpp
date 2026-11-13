// Copyright (c) 2026, Lu Lu

// JIT-backed implementations of the legacy dispatch entry points. The public
// Host call sites and Python API are unchanged; only the device launch changes.
#include "runtime/jit_runtime.hpp"
#include "kernels/dispatch/launch.hpp"
#include "kernels/dispatch/tiling.hpp"

#include <cstdint>
#include <string>
#include <vector>

#ifndef DEEPEP_JIT_PI_ON_CHIP
#define DEEPEP_JIT_PI_ON_CHIP 0
#endif

namespace {

std::vector<std::string> flags()
{
    return {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        std::string("-DACLSHMEMI_HNS_1825_PI_ON_CHIP=") + std::to_string(DEEPEP_JIT_PI_ON_CHIP),
        "--cce-auto-infer-kernel-type=false",
    };
}

std::string source(const char* file)
{
    return std::string("#include <kernels/dispatch/") + file + ">\n";
}

}  // namespace

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(void* stream, uint8_t* workspace, uint8_t* destinations,
                                                        uint8_t* forward, uint8_t* counts, uint8_t* weights,
                                                        uint8_t* scales, uint8_t* tiling)
{
    const auto t = *reinterpret_cast<ascend_deepep::DispatchTiling*>(tiling);
    const bool profile = t.profile;
    const auto kernel =
        deepep::jit_compile(profile ? "dispatch_profile" : "dispatch",
                    source(profile ? "dispatch_profile_kernel.cpp" : "dispatch_kernel.cpp"), flags());
    deepep::jit_launch(kernel, stream, 64, 192 * 1024, workspace, destinations, forward, counts, weights, scales, t);
}
