// Copyright (c) 2026, Lu Lu

// JIT host entry for the MegaMoE device template. The Python/extension API and
// mega_moe_kernel_do signature stay unchanged.
#include "runtime/jit_runtime.hpp"
#include "kernels/mega_moe/mega_moe_tiling.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"
#include "kernel_launch.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::string> flags()
{
    std::vector<std::string> values = {
        "-std=c++17",
        "--cce-auto-sync=off",
        "-DCATLASS_ARCH=3510",
        "-DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1",
        "-DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1",
        "-DENABLE_TENSOR_API",
        "-DNOT_DYNAMIC_COMPILE",
        "-DHIGH_PERFORMANCE=1",
        "-DDETERMINISTIC_MODE=0",
        "-DL2_CACHE_HINT",
        "--cce-auto-infer-kernel-type=false",
    };
#if defined(DEEPEP_MEGAMOE_TIMER)
    values.emplace_back("-DDEEPEP_MEGAMOE_TIMER=1");
#endif
#if defined(DEEPEP_MEGAMOE_ASCENDC_API_VERSION)
    values.emplace_back(std::string("-DASCENDC_API_VERSION=") + std::to_string(DEEPEP_MEGAMOE_ASCENDC_API_VERSION));
#endif
    return values;
}

std::string source(const MegaMoeTilingData& tiling, bool is_a8w4, bool fuse_gate_up)
{
    const std::string shape = std::to_string(tiling.bs) + ", " + std::to_string(tiling.h) + ", " +
                              std::to_string(tiling.epWorldSize) + ", " + std::to_string(tiling.topK) + ", " +
                              std::to_string(tiling.blockAivNum) + ", " + (is_a8w4 ? "true" : "false") + ", " +
                              (fuse_gate_up ? "true" : "false");
    const std::string params =
        "GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids, GM_ADDR topk_weights, GM_ADDR weight1, "
        "GM_ADDR weight2, GM_ADDR weight_scales1, GM_ADDR weight_scales2, GM_ADDR shared_weight1, "
        "GM_ADDR shared_weight2, GM_ADDR shared_weight_scales1, GM_ADDR shared_weight_scales2, GM_ADDR y, "
        "GM_ADDR expert_token_nums, GM_ADDR workspace, GM_ADDR timer, MegaMoeTilingData tiling, uint64_t ffts_addr";
    const std::string args =
        "context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2, "
        "shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums, "
        "workspace, timer, tiling, ffts_addr";
    const std::string call =
        "    MegaMoeRun<" + shape + ">(MegaMoeArgs{" + args + "}, deepep::OrdinaryCoreContext());\n";
    const std::string sk_call =
        "    sysArgs->SkInit();\n"
        "    const deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};\n"
        "    MegaMoeRun<" + shape + ">(*block, core);\n";
    return "#include <kernels/mega_moe/mega_moe_common.hpp>\n"
           "extern \"C\" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void mega_moe_entry(" + params +
           ")\n{\n" + call + "}\n"
           "static void __instantiate_kernel() {\n"
           "    auto ptr = reinterpret_cast<void*>(&mega_moe_entry);\n"
           "    (void)ptr;\n"
           "}\n"
           "__sk__ __mix__(1, 2) void mega_moe_sk0(const MegaMoeArgs* block, sk::SkSystemArgs* sysArgs)\n{\n" +
           sk_call + "}\n"
           "__sk__ __mix__(1, 2) void mega_moe_sk1(const MegaMoeArgs* block, sk::SkSystemArgs* sysArgs)\n{\n" +
           sk_call + "}\n"
           "__sk__ __mix__(1, 2) void mega_moe_sk2(const MegaMoeArgs* block, sk::SkSystemArgs* sysArgs)\n{\n" +
           sk_call + "}\n"
           "__sk__ __mix__(1, 2) void mega_moe_sk3(const MegaMoeArgs* block, sk::SkSystemArgs* sysArgs)\n{\n" +
           sk_call + "}\n"
           "SK_BIND(mega_moe_entry, 0, mega_moe_sk0, mega_moe_sk1, mega_moe_sk2, mega_moe_sk3);\n";
}

}  // namespace


extern "C" void mega_moe_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    const bool small_w4a8 = is_a8w4 && tiling.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    const char* name = is_a8w4 ? (small_w4a8 ? "mega_moe_a8w4_gateup" : "mega_moe_a8w4") : "mega_moe_a8w8";
    const auto kernel = deepep::jit_compile(name, source(tiling, is_a8w4, small_w4a8), flags());
    deepep::jit_launch(kernel, stream, static_cast<int>(block_dim), static_cast<int>(ub_bytes), context, x, x_scales,
                       topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2, shared_weight1,
                       shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums, workspace,
                       timer, tiling, ffts_addr);
}
