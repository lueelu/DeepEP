// Copyright (c) 2026, Lu Lu

// Host JIT entry of the table-driven clock wait. The generated source contains
// no deadline value and no call index: both are runtime arguments, so all
// captured wait nodes share one compiled kernel.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <deep_ep/common/clock_wait.hpp>

#include "runtime/jit.hpp"

namespace deep_ep {

inline std::vector<std::string> clock_wait_jit_flags()
{
#ifndef DEEPEP_JIT_PI_ON_CHIP
#error "DEEPEP_JIT_PI_ON_CHIP must be forwarded from the SHMEM SDK identity"
#endif
    static_assert(DEEPEP_JIT_PI_ON_CHIP == 0 || DEEPEP_JIT_PI_ON_CHIP == 1,
                  "DEEPEP_JIT_PI_ON_CHIP must be the SDK's recorded 0/1 doorbell mode");
    // The device header shares the Dispatch include closure, so it must see the
    // same SDK identity as the library it will be launched against.
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

inline void launch_clock_wait_table(uint32_t cores, void* stream,
                                    void* deadlines, int64_t index,
                                    void* records, int64_t batch, bool diagnostic)
{
    if (cores == 0U || stream == nullptr || deadlines == nullptr || index < 0)
        throw std::invalid_argument("Invalid LL clock wait-table launch");
    static const std::string source = R"(
#include <deep_ep/impls/ep/clock_wait.hpp>

using namespace deep_ep;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void clock_wait_table(
    __gm__ uint8_t* deadlines, int64_t index, __gm__ uint8_t* records, int64_t batch,
    int32_t cores, int32_t diagnostic)
{
    clock_wait_body(ClockWaitArgs{deadlines, index, records, batch, cores, diagnostic},
                    OrdinaryCoreContext());
}
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&clock_wait_table);
    (void)ptr;
}
extern "C" __sk__ __vector__ void clock_wait_table_sk0(
    const ClockWaitArgs* args, sk::SkSystemArgs* system_args)
{
    clock_wait_impl_sk<0>(args, system_args);
}
extern "C" __sk__ __vector__ void clock_wait_table_sk1(
    const ClockWaitArgs* args, sk::SkSystemArgs* system_args)
{
    clock_wait_impl_sk<1>(args, system_args);
}
extern "C" __sk__ __vector__ void clock_wait_table_sk2(
    const ClockWaitArgs* args, sk::SkSystemArgs* system_args)
{
    clock_wait_impl_sk<2>(args, system_args);
}
extern "C" __sk__ __vector__ void clock_wait_table_sk3(
    const ClockWaitArgs* args, sk::SkSystemArgs* system_args)
{
    clock_wait_impl_sk<3>(args, system_args);
}
SK_BIND(clock_wait_table, 0, clock_wait_table_sk0, clock_wait_table_sk1,
        clock_wait_table_sk2, clock_wait_table_sk3);
)";
    deep_jit::ascend::CompilerOptions options;
    options.extra_bisheng_flags = clock_wait_jit_flags();
    const auto kernel = jit->compile("clock_wait_table", source, options);

    deep_jit::ascend::LaunchOptions launch_options;
    launch_options.stream = static_cast<aclrtStream>(stream);
    launch_options.num_blocks = static_cast<int>(cores);
    launch_options.num_ubuf_bytes = 0;
    jit->launch(kernel, launch_options, deadlines, index, records, batch,
                static_cast<int32_t>(cores), static_cast<int32_t>(diagnostic ? 1 : 0));
}

}  // namespace deep_ep
