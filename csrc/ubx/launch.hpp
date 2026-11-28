// Copyright (c) 2026, Lu Lu
#pragma once
#include "kernels/ep/dispatch.hpp"

namespace deep_ep {
inline void launch_combine_put(void* stream, uint32_t cores, uint32_t group, uint32_t ub_bytes, int h, int k, int e,
                               void* comm, void* expert, void* destination, void* counts, void* weights,
                               void* types, void* relay, void* mask, void* output, void* tiling)
{
    const auto source = std::format(R"(
#include <ubx/kernels/combine_entry.h>
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_put_jit(
    GM_ADDR comm, GM_ADDR expert, GM_ADDR destination, GM_ADDR counts, GM_ADDR weights,
    GM_ADDR types, GM_ADDR relay, GM_ADDR mask, GM_ADDR output, GM_ADDR tiling)
{{
    CombinePut<{}, {}, {}, {}, {}>(comm, expert, destination, counts, weights, types, relay, mask, output, tiling);
}}
static void __instantiate_kernel() {{ (void)&combine_put_jit; }}
)", group, h, k, e, ub_bytes);
    deep_jit::ascend::CompilerOptions options;
    options.extra_bisheng_flags = dispatch_flags();
    options.extra_bisheng_flags.emplace_back("-ffp-contract=off");
    options.extra_bisheng_flags.emplace_back("-I" + jit->backend.toolkit.asc_devkit_include.parent_path().string());
    const auto kernel = jit->compile("combine_put", source, options);
    auto launch = jit->default_launch_options;
    launch.stream = static_cast<aclrtStream>(stream);
    launch.num_blocks = cores;
    launch.num_ubuf_bytes = ub_bytes;
    jit->launch(kernel, launch, comm, expert, destination, counts, weights, types, relay, mask, output, tiling);
}
}  // namespace deep_ep
