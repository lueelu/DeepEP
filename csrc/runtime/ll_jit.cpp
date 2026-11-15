// Copyright (c) 2026, Lu Lu

// LL Dispatch/Combine source generation on top of the shared DeepJIT runtime.
#include "ll_jit.hpp"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jit_runtime.hpp"
#include "../kernels/combine/low_latency/launch.hpp"
#include "../kernels/dispatch/low_latency/launch.hpp"
#include "../ubx/launch.hpp"

#ifndef DEEPEP_JIT_PI_ON_CHIP
#define DEEPEP_JIT_PI_ON_CHIP 0
#endif

namespace deepep {
namespace {

const char* entry_header(LlJitOperation operation)
{
    switch (operation) {
        case LlJitOperation::Dispatch:
            return "kernels/dispatch/low_latency/dispatch_entry.h";
        case LlJitOperation::MeshCombine:
            return "kernels/combine/low_latency/combine_entry.h";
        case LlJitOperation::UbxCombine:
            return "ubx/kernels/combine_entry.h";
    }
    throw std::invalid_argument("Invalid LL JIT operation");
}

std::string source(LlJitOperation operation)
{
    return std::string("\n#include <") + entry_header(operation) + ">\n";
}

std::string dispatch_source(const LlJitSignature& signature)
{
    std::ostringstream text;
    text << "#define DEEPEP_LL_JIT_SPECIALIZED 1\n"
         << "#define DEEPEP_LL_JIT_SPEC_BS " << signature.bs << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_H " << signature.hidden << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_TOPK " << signature.topK << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_EXPERTS " << signature.experts << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_CAPACITY " << signature.capacity << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_CORES " << signature.cores << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_WORLD " << signature.world << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_RANK " << signature.rank << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_INPUT_CLASS " << signature.inputClass << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_GROUP " << signature.group << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_INPUT_MODE " << signature.inputMode << "U\n"
         << "#define DEEPEP_LL_JIT_SPEC_E4M3 " << signature.e4m3 << "\n"
         << "#define DEEPEP_LL_JIT_SPEC_DEDUP " << signature.dedup << "\n";
    return text.str() + source(LlJitOperation::Dispatch);
}

std::string combine_source(const LlJitSignature& signature)
{
    const bool ubx = signature.operation == LlJitOperation::UbxCombine;
    const char* prefix = ubx ? "DEEPEP_UBX_JIT" : "DEEPEP_LL_JIT";
    std::ostringstream text;
    text << "#define " << prefix << "_SPECIALIZED 1\n"
         << "#define " << prefix << "_SPEC_BS " << signature.bs << "U\n"
         << "#define " << prefix << "_SPEC_H " << signature.hidden << "U\n"
         << "#define " << prefix << "_SPEC_TOPK " << signature.topK << "U\n"
         << "#define " << prefix << "_SPEC_EXPERTS " << signature.experts << "U\n"
         << "#define " << prefix << "_SPEC_CAPACITY " << signature.capacity << "U\n"
         << "#define " << prefix << "_SPEC_CORES " << signature.cores << "U\n"
         << "#define " << prefix << "_SPEC_WORLD " << signature.world << "U\n"
         << "#define " << prefix << "_SPEC_RANK " << signature.rank << "U\n"
         << "#define " << prefix << "_SPEC_BF16 " << signature.bf16 << "\n"
         << "#define " << prefix << "_SPEC_DEDUP " << signature.dedup << "\n";
    if (ubx) text << "#define DEEPEP_UBX_JIT_SPEC_GROUP " << signature.group << "U\n";
    return text.str() + source(signature.operation);
}

std::vector<std::string> flags(const LlJitSignature& signature)
{
    const bool ubx = signature.operation == LlJitOperation::UbxCombine;
    std::vector<std::string> values = {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        ubx ? "-DDEEPEP_UBX_JIT_TU=1" : "-DDEEPEP_LL_JIT_TU=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        std::string("-DACLSHMEMI_HNS_1825_PI_ON_CHIP=") + std::to_string(DEEPEP_JIT_PI_ON_CHIP),
        "--cce-auto-infer-kernel-type=false",
    };
#ifdef DEEPEP_JIT_REFERENCE_PROFILE
    if (signature.operation == LlJitOperation::Dispatch) values.emplace_back("-DDEEPEP_LL_REFERENCE_PROFILE=1");
#endif
#ifdef DEEPEP_JIT_COUNT_HISTOGRAM
    if (signature.operation == LlJitOperation::Dispatch) values.emplace_back("-DDEEPEP_LL_COUNT_HISTOGRAM=1");
#endif
    return values;
}

void launch_dispatch(const LlJitSignature& signature, const EpServerDedupKernelArgs& args, void* stream)
{
    const auto kernel = jit_compile("dispatch", dispatch_source(signature), flags(signature));
    jit_launch(kernel, stream, static_cast<int>(signature.cores), 0, args.commArgs, args.x, args.expertIds,
               args.expertScales, args.expandXOut, args.dynamicScalesOut, args.expertTokenNumsOut, args.sendCountsOut,
               args.tokenTypeOut, args.destinationIndexOut, args.relayReadIndexOut, args.sourceMaskOut,
               args.expertScalesOut, args.tilingData, args.reservedMagic, args.inputScales, args.stageTimer,
               args.timerMode, args.expertTimer);
}

void launch_combine(const LlJitSignature& signature, const EpServerDedupCombineKernelArgs& args, void* stream)
{
    const auto kernel = jit_compile("mesh_combine", combine_source(signature), flags(signature));
    jit_launch(kernel, stream, static_cast<int>(signature.cores), 0, args.commArgs, args.expertOut,
               args.destinationIndex, args.sendCounts, args.expertScales, args.tokenType, args.relayReadIndex,
               args.sourceMask, args.yOut, args.tilingData);
}

void launch_ubx_combine(const LlJitSignature& signature, const ubx::CombineArgs& args, void* stream)
{
    const auto kernel = jit_compile("ubx_combine", combine_source(signature), flags(signature));
    jit_launch(kernel, stream, static_cast<int>(signature.cores), 0, args.comm, args.expert, args.destination,
               args.counts, args.weights, args.types, args.relay, args.mask, args.output, args.tiling);
}

}  // namespace

void ll_jit_launch(const LlJitSignature& signature, const void* arguments, std::size_t bytes, void* stream)
{
    switch (signature.operation) {
        case LlJitOperation::Dispatch:
            if (bytes != sizeof(EpServerDedupKernelArgs))
                throw std::invalid_argument("Dispatch launch argument size mismatch");
            launch_dispatch(signature, *static_cast<const EpServerDedupKernelArgs*>(arguments), stream);
            break;
        case LlJitOperation::MeshCombine:
            if (bytes != sizeof(EpServerDedupCombineKernelArgs))
                throw std::invalid_argument("Mesh combine launch argument size mismatch");
            launch_combine(signature, *static_cast<const EpServerDedupCombineKernelArgs*>(arguments), stream);
            break;
        case LlJitOperation::UbxCombine:
            if (bytes != sizeof(ubx::CombineArgs)) throw std::invalid_argument("UBX combine argument size mismatch");
            launch_ubx_combine(signature, *static_cast<const ubx::CombineArgs*>(arguments), stream);
            break;
    }
}

}  // namespace deepep
