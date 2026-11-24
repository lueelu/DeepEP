// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "jit/mega_moe_jit.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/lazy.hpp>

#include "kernel_launch.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#include "kernels/mega_moe_common/mega_moe_tiling.hpp"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"

#ifndef DEEPEP_MEGAMOE_PI_ON_CHIP
#error "DEEPEP_MEGAMOE_PI_ON_CHIP must be forwarded from the SHMEM SDK identity"
#endif
#ifndef DEEPEP_MEGAMOE_JIT_SIGNATURE
#error "DEEPEP_MEGAMOE_JIT_SIGNATURE must carry the bundled kernel source identity"
#endif

namespace deepep::megamoe {
namespace {

namespace fs = std::filesystem;
using Kernel = deep_jit::ascend::Kernel;
using LaunchOptions = deep_jit::ascend::LaunchOptions;

// This module is a separate shared object from deep_ep._C, so it owns its own
// DeepJIT runtime instead of reusing the core singleton. The two backends keep
// separate include orders: shared headers resolve their epilogue and GMM
// implementations from the backend directory that is first on the search path,
// exactly like the former per-backend device translation units.
deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>> mte_jit(nullptr);
deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>> urma_jit(nullptr);
bool mega_jit_ready = false;

void AppendInclude(const fs::path& path, std::vector<fs::path>& directories)
{
    fs::path normal = path.lexically_normal();
    if (fs::is_directory(normal)) directories.emplace_back(std::move(normal));
}

fs::path AscendHome(const std::string& requested)
{
    std::vector<fs::path> candidates;
    if (!requested.empty()) candidates.emplace_back(requested);
    const char* home = std::getenv("ASCEND_HOME_PATH");
    if (home != nullptr && *home != '\0') candidates.emplace_back(home);
    home = std::getenv("ASCEND_TOOLKIT_HOME");
    if (home != nullptr && *home != '\0') candidates.emplace_back(home);
    candidates.emplace_back("/usr/local/Ascend/ascend-toolkit/latest");
    candidates.emplace_back("/usr/local/Ascend/cann");
    for (const auto& candidate : candidates)
        if (fs::is_directory(candidate)) return fs::absolute(candidate).lexically_normal();
    throw std::runtime_error("MegaMoE JIT cannot find the CANN installation; set ASCEND_HOME_PATH");
}

// Same search order as experiments/megamoe/cmake/Headers.cmake; the Tensor API
// lives in the installed operator packages, so it stays external to the wheel.
fs::path TensorApiCommon(const fs::path& home)
{
    std::vector<fs::path> candidates = {
        home / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common",
        home / "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        home / "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common",
    };
    for (const auto& candidate : candidates)
        if (fs::is_regular_file(candidate / "tensor_api/tensor.h") ||
            fs::is_regular_file(candidate / "tensor_api/include/tensor_api/tensor.h"))
            return candidate;
    throw std::runtime_error(
        "MegaMoE JIT cannot find the Tensor API header tensor_api/tensor.h in the active CANN installation");
}

fs::path AscendArchDir(const fs::path& home)
{
    for (const char* sibling : {"aarch64-linux", "x86_64-linux", "arm64-linux"}) {
        const fs::path candidate = home / sibling;
        if (fs::is_directory(candidate / "asc/include")) return candidate;
    }
    return home / "aarch64-linux";
}

std::vector<std::string> MegaMoEJitFlags(bool urma)
{
    std::vector<std::string> flags = {
        "-std=c++17",
        "--cce-auto-infer-kernel-type=false",
        "--cce-auto-sync=off",
        "-DCATLASS_ARCH=3510",
        "-DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1",
        "-DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1",
        "-DENABLE_TENSOR_API",
        "-DNOT_DYNAMIC_COMPILE",
        "-DHIGH_PERFORMANCE=1",
        "-DDETERMINISTIC_MODE=0",
        "-DL2_CACHE_HINT",
    };
    // The device code calls into the SDK it was built against, so the recorded
    // doorbell mode must travel with the compiled kernel.
    flags.emplace_back("-DACLSHMEMI_HNS_1825_PI_ON_CHIP=" + std::to_string(DEEPEP_MEGAMOE_PI_ON_CHIP));
    if (urma) flags.emplace_back("-DDEEPEP_MEGAMOE_URMA=1");
#ifdef DEEPEP_MEGAMOE_TIMER
    flags.emplace_back("-DDEEPEP_MEGAMOE_TIMER=1");
#endif
    return flags;
}

std::vector<fs::path> MegaMoEIncludeDirs(const std::string& package_root, const std::string& ascend_home, bool urma)
{
    const fs::path root = fs::absolute(package_root).lexically_normal();
    const fs::path kernel_src = root / "_kernel_src";
    const fs::path mega_src = kernel_src / "megamoe";
    for (const char* required : {"mte/kernels/mega_moe/mega_moe_entry.hpp",
                                 "urma/kernels/mega_moe_urma/mega_moe_urma_entry.hpp", "catlass/include",
                                 "catlass/examples/common"})
        if (!fs::exists(mega_src / required))
            throw std::runtime_error("MegaMoE JIT is missing bundled kernel sources: " +
                                     (mega_src / required).string() +
                                     "; reinstall the complete Beta wheel built with DEEPEP_BUILD_MEGAMOE=ON");
    const fs::path home = AscendHome(ascend_home);
    const fs::path arch = AscendArchDir(home);
    const fs::path common = TensorApiCommon(home);
    std::vector<fs::path> directories;
    // The bundled view already resolved every shared header against this
    // backend's original include order, so it is the only project include root.
    AppendInclude(mega_src / (urma ? "urma" : "mte"), directories);
    AppendInclude(mega_src / "catlass/include", directories);
    AppendInclude(mega_src / "catlass/examples/common", directories);
    AppendInclude(kernel_src / "shmem/include", directories);
    AppendInclude(kernel_src / "shmem/src", directories);
    AppendInclude(kernel_src / "shmem/src/device", directories);
    AppendInclude(kernel_src / "shmem/src/device_simt", directories);
    AppendInclude(home / "include", directories);
    AppendInclude(home / "pkg_inc", directories);
    AppendInclude(home / "include/experiment/runtime", directories);
    AppendInclude(home / "pkg_inc/runtime", directories);
    AppendInclude(home / "include/op_common", directories);
    AppendInclude(home / "pkg_inc/op_common", directories);
    AppendInclude(home / "compiler/ascendc/act", directories);
    AppendInclude(arch / "asc/atcos", directories);
    AppendInclude(arch / "asc/include", directories);
    AppendInclude(arch / "asc/include/adv_api", directories);
    AppendInclude(arch / "asc/impl", directories);
    AppendInclude(common, directories);
    AppendInclude(common / "tensor_api", directories);
    AppendInclude(common / "tensor_api/include", directories);
    return directories;
}

void EnsureMegaJit()
{
    if (!mega_jit_ready)
        throw std::runtime_error(
            "MegaMoE JIT is not initialized; import deep_ep_experimental.megamoe before the first launch");
}

std::shared_ptr<Kernel> CompiledKernel(const std::string& key, const std::string& source, bool urma)
{
    // The generated source and the compiler options are part of DeepJIT's
    // content identity, so its own memory/disk caches distinguish every
    // specialization. An outer routed/shared map would silently reuse a
    // binary compiled for different capacities or weight modes.
    deep_jit::ascend::CompilerOptions options;
    options.extra_bisheng_flags = MegaMoEJitFlags(urma);
    return (urma ? urma_jit : mte_jit)->compile(key, source, options);
}

LaunchOptions MegaLaunchOptions(uint32_t block_dim, uint32_t ub_bytes, void* stream)
{
    if (block_dim == 0 || ub_bytes == 0)
        throw std::invalid_argument("MegaMoE JIT requires a positive block count and UB budget");
    LaunchOptions options;
    options.stream = static_cast<aclrtStream>(stream);
    options.num_blocks = static_cast<int>(block_dim);
    options.num_ubuf_bytes = static_cast<int>(ub_bytes);
    return options;
}

const char* MteParameters()
{
    return "__gm__ uint8_t* context, __gm__ uint8_t* x, __gm__ uint8_t* xScales, "
           "__gm__ uint8_t* topkIds, __gm__ uint8_t* topkWeights, "
           "__gm__ uint8_t* weight1, __gm__ uint8_t* weight2, __gm__ uint8_t* weightScales1,\n"
           "    __gm__ uint8_t* weightScales2, __gm__ uint8_t* sharedWeight1, __gm__ uint8_t* sharedWeight2,\n"
           "    __gm__ uint8_t* sharedWeightScales1, __gm__ uint8_t* sharedWeightScales2, "
           "__gm__ uint8_t* output,\n"
           "    __gm__ uint8_t* expertTokenNums, __gm__ uint8_t* workspace, __gm__ uint8_t* timer, "
           "MegaMoeTilingData tiling, uint64_t ffts";
}

std::string MteSource(bool is_a8w4, bool fuse_gate_up, bool early_weights)
{
    const char* w4 = is_a8w4 ? "true" : "false";
    const char* flag = fuse_gate_up ? "true" : "false";
    const char* early = early_weights ? "true" : "false";
    std::string source =
        "#include <kernels/mega_moe/mega_moe_entry.hpp>\n\n"
        "extern \"C\" __schedmode__(1) __global__ __mix__(1, 2) void mega_moe_jit(";
    source += MteParameters();
    source +=
        ")\n{\n"
        "    MegaMoeEntry::MegaMoeArgs args{context, x, xScales, topkIds, topkWeights, weight1, weight2,\n"
        "        weightScales1, weightScales2, sharedWeight1, sharedWeight2, sharedWeightScales1,\n"
        "        sharedWeightScales2, output, expertTokenNums, workspace, timer, tiling, ffts};\n"
        "    MegaMoeEntry::MegaMoeBody<" + std::string(w4) + ", " + flag + ", " + early + ">(&args);\n"
        "}\n";
    for (uint32_t slot = 0; slot < 4U; ++slot) {
        source += "extern \"C\" __sk__ __mix__(1, 2) void mega_moe_jit_sk" + std::to_string(slot) +
                  "(const MegaMoeEntry::MegaMoeArgs* args, sk::SkSystemArgs* system_args)\n{\n"
                  "    MegaMoeEntry::MegaMoeSk<" + std::string(w4) + ", " + flag + ", " + early + ", " +
                  std::to_string(slot) + ">(args, system_args);\n}\n";
    }
    source += "SK_BIND(mega_moe_jit, 0, mega_moe_jit_sk0, mega_moe_jit_sk1, mega_moe_jit_sk2, mega_moe_jit_sk3);\n";
    return source;
}

// Fixed plan capacity and model geometry become template arguments, so the
// device compiler folds every stride and bound derived from them. Per-invocation
// values stay in the Arguments block.
std::string UrmaSpec(const MegaMoeUrma::Config& config)
{
    return std::to_string(config.sourceRowCapacity) + ", " + std::to_string(config.hidden) + ", " +
           std::to_string(config.intermediate) + ", " + std::to_string(config.topk) + ", " +
           std::to_string(config.localExperts) + ", " + std::to_string(config.world) + ", " +
           std::to_string(config.topkWeightsType);
}

std::string UrmaSource(bool shared, const std::string& spec)
{
    const char* flag = shared ? "true" : "false";
    std::string source =
        "#include <kernels/mega_moe_urma/mega_moe_urma_entry.hpp>\n\n"
        "extern \"C\" __schedmode__(1) __global__ __mix__(1, 2) void mega_moe_urma_jit("
        "MegaMoeUrma::Arguments args)\n{\n"
        "    MegaMoeUrmaEntry::" + std::string(shared ? "SharedBody" : "RoutedBody") + "<" + spec + ">(args);\n}\n";
    for (uint32_t slot = 0; slot < 4U; ++slot) {
        source += "extern \"C\" __sk__ __mix__(1, 2) void mega_moe_urma_jit_sk" + std::to_string(slot) +
                  "(const MegaMoeUrma::Arguments* args, sk::SkSystemArgs* system_args)\n{\n"
                  "    MegaMoeUrmaEntry::MegaMoeUrmaSk<" + std::string(flag) + ", " + std::to_string(slot) + ", " +
                  spec + ">(args, system_args);\n}\n";
    }
    source +=
        "SK_BIND(mega_moe_urma_jit, 0, mega_moe_urma_jit_sk0, mega_moe_urma_jit_sk1, mega_moe_urma_jit_sk2, "
        "mega_moe_urma_jit_sk3);\n";
    return source;
}

}  // namespace

void init_mega_moe_jit(const std::string& package_root, const std::string& ascend_home)
{
    if (mega_jit_ready) return;
    const fs::path root = fs::absolute(package_root).lexically_normal();
    // Validate both include closures before any launch; the runtimes themselves
    // are created on the first launch so importing this module never touches a
    // device.
    const deep_jit::Config mte_config(root, "MEGAMOE", DEEPEP_MEGAMOE_JIT_SIGNATURE,
                                      MegaMoEIncludeDirs(package_root, ascend_home, false), {"kernels/"});
    const deep_jit::Config urma_config(root, "MEGAMOE_URMA", DEEPEP_MEGAMOE_JIT_SIGNATURE,
                                       MegaMoEIncludeDirs(package_root, ascend_home, true), {"kernels/"});
    mte_jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>>(
        [mte_config] { return std::make_shared<deep_jit::Runtime<deep_jit::Ascend>>(mte_config); });
    urma_jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>>(
        [urma_config] { return std::make_shared<deep_jit::Runtime<deep_jit::Ascend>>(urma_config); });
    mega_jit_ready = true;
}

void launch_mega_moe_jit(uint32_t block_dim, uint32_t ub_bytes, void* stream, const MteLaunchArgs& args)
{
    EnsureMegaJit();
    if (args.tiling == nullptr) throw std::invalid_argument("MegaMoE launch requires its tiling data");
    const MegaMoeTilingData& tiling = *reinterpret_cast<const MegaMoeTilingData*>(args.tiling);
    static_assert(sizeof(MegaMoeTilingData) == 224U, "MegaMoE tiling ABI changed");
    bool fuse_gate_up = false;
    bool early_weights = false;
    std::string variant;
    if (tiling.topkWeightsPrefetch) {
        early_weights = true;
        if (args.is_a8w4 && tiling.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
            fuse_gate_up = true;
            variant = "mega_moe_pf_w4_small";
        } else if (args.is_a8w4) {
            variant = "mega_moe_pf_w4";
        } else {
            variant = "mega_moe_pf_w8";
        }
    } else if (args.is_a8w4 && tiling.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        fuse_gate_up = true;
        variant = "mega_moe_w4_small";
    } else if (args.is_a8w4) {
        variant = "mega_moe_w4";
    } else {
        variant = "mega_moe_w8";
    }
    auto kernel = CompiledKernel(variant, MteSource(args.is_a8w4, fuse_gate_up, early_weights), false);
    mte_jit->launch(kernel, MegaLaunchOptions(block_dim, ub_bytes, stream), args.context, args.x, args.x_scales,
                     args.topk_ids, args.topk_weights, args.weight1, args.weight2, args.weight_scales1,
                     args.weight_scales2, args.shared_weight1, args.shared_weight2, args.shared_weight_scales1,
                     args.shared_weight_scales2, args.y, args.expert_token_nums, args.workspace, args.timer, tiling,
                     args.ffts);
}

void launch_mega_moe_urma_jit(uint32_t block_dim, uint32_t ub_bytes, void* stream,
                              const MegaMoeUrma::Arguments* args)
{
    EnsureMegaJit();
    if (args == nullptr) throw std::invalid_argument("Missing URMA device arguments");
    const bool shared = args->sharedExperts != 0U;
    const std::string variant = shared ? "mega_moe_urma_shared" : "mega_moe_urma_routed";
    auto kernel = CompiledKernel(variant, UrmaSource(shared, UrmaSpec(args->config)), true);
    urma_jit->launch(kernel, MegaLaunchOptions(block_dim, ub_bytes, stream), *args);
}

}  // namespace deepep::megamoe
