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

#ifndef DEEPEP_JIT_PI_ON_CHIP
#define DEEPEP_JIT_PI_ON_CHIP 0
#endif

namespace deepep {
namespace {

std::string dispatch_source(const LlJitSignature& s)
{
    std::ostringstream args;
    args << s.world << ", " << s.rank << ", " << s.bs << ", " << s.hidden << ", " << s.topK << ", " << s.experts << ", "
         << s.capacity << ", " << s.cores << ", " << s.inputClass << ", " << s.group << ", " << s.inputMode << ", "
         << s.e4m3 << ", " << s.dedup;
    const std::string shape = args.str();
    std::ostringstream text;
    text << "#include <kernels/dispatch/low_latency/dispatch_entry.h>\n"
         << "extern \"C\" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void "
            "dispatch_server_dedup_kernel_jit(DispatchDedupEntry::ArgsBlock block)\n"
         << "{\n"
         << "    DispatchDedupEntry::Run<" << shape
         << ">(block, deepep::OrdinaryCoreContext(), false);\n"
         << "}\n"
         << "static void __instantiate_kernel()\n"
         << "{\n"
         << "    auto ptr = reinterpret_cast<void*>(&dispatch_server_dedup_kernel_jit);\n"
         << "    (void)ptr;\n"
         << "}\n";
    for (uint32_t i = 0; i < 4U; ++i) {
        text << "__sk__ __vector__ void dispatch_server_dedup_kernel_sk" << i
             << "(const DispatchDedupEntry::ArgsBlock* block, sk::SkSystemArgs* sysArgs)\n"
             << "{\n"
             << "    sysArgs->SkInit();\n"
             << "    const deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};\n"
             << "    const DispatchDedupEntry::ArgsBlock local = *block;\n"
             << "    DispatchDedupEntry::Run<" << shape << ">(local, core, true);\n"
             << "}\n";
    }
    text << "SK_BIND(dispatch_server_dedup_kernel_jit, 0, dispatch_server_dedup_kernel_sk0, "
            "dispatch_server_dedup_kernel_sk1, dispatch_server_dedup_kernel_sk2, "
            "dispatch_server_dedup_kernel_sk3);\n";
    return text.str();
}

std::string combine_source(const LlJitSignature& s)
{
    std::ostringstream args;
    args << s.bs << ", " << s.hidden << ", " << s.topK << ", " << s.experts << ", " << s.world << ", " << s.rank << ", "
         << s.cores << ", " << s.capacity << ", " << s.bf16 << ", " << s.dedup;
    const std::string shape = args.str();
    std::ostringstream text;
    text << "#include <kernels/combine/low_latency/combine_entry.h>\n"
         << "extern \"C\" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void "
            "combine_server_dedup_kernel_jit(CombineDedupEntry::ArgsBlock block)\n"
         << "{\n"
         << "    CombineDedupEntry::Run<" << shape
         << ">(block, deepep::OrdinaryCoreContext(), false);\n"
         << "}\n"
         << "static void __instantiate_kernel()\n"
         << "{\n"
         << "    auto ptr = reinterpret_cast<void*>(&combine_server_dedup_kernel_jit);\n"
         << "    (void)ptr;\n"
         << "}\n";
    for (uint32_t i = 0; i < 4U; ++i) {
        text << "__sk__ __vector__ void combine_server_dedup_kernel_sk" << i
             << "(const CombineDedupEntry::ArgsBlock* block, sk::SkSystemArgs* sysArgs)\n"
             << "{\n"
             << "    sysArgs->SkInit();\n"
             << "    const deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};\n"
             << "    const CombineDedupEntry::ArgsBlock local = *block;\n"
             << "    CombineDedupEntry::Run<" << shape << ">(local, core, true);\n"
             << "}\n";
    }
    text << "SK_BIND(combine_server_dedup_kernel_jit, 0, combine_server_dedup_kernel_sk0, "
            "combine_server_dedup_kernel_sk1, combine_server_dedup_kernel_sk2, "
            "combine_server_dedup_kernel_sk3);\n";
    return text.str();
}

std::vector<std::string> flags(const LlJitSignature& signature)
{
    (void)signature;
    std::vector<std::string> values = {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DDEEPEP_LL_JIT_TU=1",
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
    const auto kernel = jit_compile("combine", combine_source(signature), flags(signature));
    jit_launch(kernel, stream, static_cast<int>(signature.cores), 0, args.commArgs, args.expertOut,
               args.destinationIndex, args.sendCounts, args.expertScales, args.tokenType, args.relayReadIndex,
               args.sourceMask, args.yOut, args.tilingData);
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
        case LlJitOperation::Combine:
            if (bytes != sizeof(EpServerDedupCombineKernelArgs))
                throw std::invalid_argument("Combine launch argument size mismatch");
            launch_combine(signature, *static_cast<const EpServerDedupCombineKernelArgs*>(arguments), stream);
            break;
    }
}

}  // namespace deepep
