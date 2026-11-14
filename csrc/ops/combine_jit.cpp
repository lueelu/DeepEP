// Copyright (c) 2026, Lu Lu

// JIT-backed implementations of the legacy combine entry points. Host call
// signatures stay unchanged; only the device launch is runtime-compiled.
#include "runtime/jit_runtime.hpp"
#include "kernels/combine/launch.hpp"
#include "kernels/combine/tiling.hpp"

#include <cstdint>
#include <memory>
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
    return std::string("#include <") + file + ">\n";
}

std::shared_ptr<deepep::JitKernel> compile(const char* name, const char* file)
{
    return deepep::jit_compile(name, source(file), flags());
}

}  // namespace

extern "C" ASCEND_DEEPEP_EXPORT void combine_prepare_kernel_do(void* stream, uint8_t* workspace, uint8_t* tiling)
{
    const auto kernel =
        compile("combine_prepare", "kernels/combine/combine_prepare_kernel.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, 0, workspace, t);
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                       uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                       uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges,
                                                       uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
                                                       uint8_t* tiling)
{
    const auto kernel = compile("combine", "kernels/combine/combine_kernel.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, ascend_deepep::combine_config::kLaunchUbBytes, workspace, input, forward,
                       counts, backward, backward_counts, mask, ranges, masks, weights, weight_meta, t);
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_profile_kernel_do(
    void* stream, uint8_t* workspace, uint8_t* input, uint8_t* forward, uint8_t* counts, uint8_t* backward,
    uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
    uint8_t* profile, uint64_t sample, uint8_t* chunk_profile, uint8_t* peer_profile, uint8_t* tiling)
{
    const auto kernel = compile("combine_profile", "kernels/combine/combine_profile_kernel.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, ascend_deepep::combine_config::kLaunchUbBytes, workspace, input, forward,
                       counts, backward, backward_counts, mask, ranges, masks, weights, weight_meta, profile, sample,
                       chunk_profile, peer_profile, t);
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_ep8_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                           uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                           uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges,
                                                           uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
                                                           uint8_t* tiling)
{
    const auto kernel = compile("combine_ep8", "kernels/combine/combine_ep8_kernel.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, ascend_deepep::combine_config::kLaunchUbBytes, workspace, input, forward,
                       counts, backward, backward_counts, mask, ranges, masks, weights, weight_meta, t);
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_ep8_profile_kernel_do(
    void* stream, uint8_t* workspace, uint8_t* input, uint8_t* forward, uint8_t* counts, uint8_t* backward,
    uint8_t* backward_counts, uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* weights, uint8_t* weight_meta,
    uint8_t* profile, uint64_t sample, uint8_t* chunk_profile, uint8_t* peer_profile, uint8_t* tiling)
{
    const auto kernel = compile("combine_ep8_profile", "kernels/combine/combine_ep8_profile_kernel.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, ascend_deepep::combine_config::kLaunchUbBytes, workspace, input, forward,
                       counts, backward, backward_counts, mask, ranges, masks, weights, weight_meta, profile, sample,
                       chunk_profile, peer_profile, t);
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_reuse_kernel_do(void* stream, uint8_t* workspace, uint8_t* input,
                                                             uint8_t* forward, uint8_t* counts, uint8_t* backward,
                                                             uint8_t* offsets, uint8_t* mask, uint8_t* weights,
                                                             uint8_t* weight_meta, uint8_t* tiling, uint32_t window)
{
    const auto kernel = compile("combine_reuse", "kernels/combine/reuse.cpp");
    const auto t = *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling);
    deepep::jit_launch(kernel, stream, 64, ascend_deepep::combine_config::kLaunchUbBytes, workspace, input, forward,
                       counts, backward, offsets, mask, weights, weight_meta, t, window);
}
