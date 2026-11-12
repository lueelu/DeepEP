// Copyright (c) 2026, Lu Lu

#include "ll_jit.hpp"

#ifdef DEEPEP_LL_JIT
#include <acl/acl.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

#include <deep_jit/backend/ascend/backend.hpp>

#define DEEPEP_JIT_STRINGIFY_IMPL(value) #value
#define DEEPEP_JIT_STRINGIFY(value) DEEPEP_JIT_STRINGIFY_IMPL(value)

namespace deepep {
namespace {

constexpr const char* kDispatchEntry = "kernels/dispatch/low_latency/dispatch_entry.h";
constexpr const char* kMeshCombineEntry = "kernels/combine/low_latency/combine_entry.h";
constexpr const char* kUbxCombineEntry = "ubx/kernels/combine_entry.h";
constexpr const char* kShmemHeader = "shmem.h";

enum class Operation { Dispatch, MeshCombine, UbxCombine };

// One process-wide mutex: DeepJIT allocates temporary directories and its
// compiles spawn external tools.
std::mutex& launch_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<deep_jit::Runtime<deep_jit::Ascend>>& runtime()
{
    static std::shared_ptr<deep_jit::Runtime<deep_jit::Ascend>> instance;
    return instance;
}

std::string env_text(const char* name)
{
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string required_env(const char* name)
{
    const std::string value = env_text(name);
    if (value.empty()) throw std::runtime_error(std::string("LL JIT requires the ") + name + " environment variable");
    return value;
}

std::vector<std::string> split(const std::string& text, const char separator)
{
    std::vector<std::string> parts;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator))
        if (!part.empty()) parts.push_back(part);
    return parts;
}

std::filesystem::path module_dir()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ll_jit_available), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("LL JIT cannot locate the loaded extension module");
    return std::filesystem::absolute(std::filesystem::path(info.dli_fname)).lexically_normal().parent_path();
}

// Packaged wheels ship the device headers next to the extension; a source
// checkout builds against csrc directly.
std::filesystem::path source_root()
{
    const auto packaged = module_dir() / "_kernel_src" / "csrc";
    if (std::filesystem::is_directory(packaged)) return packaged;
    return std::filesystem::path(DEEPEP_JIT_SOURCE_ROOT);
}

// SHMEM stays an external SDK at run time, exactly like the AOT device
// library: reuse the root recorded at build time unless the deployment moved.
std::filesystem::path shmem_root()
{
    const std::string configured = env_text("DEEPEP_LL_JIT_SHMEM_ROOT");
    if (!configured.empty()) return configured;
    const std::filesystem::path built(DEEPEP_JIT_SHMEM_ROOT);
    if (std::filesystem::is_regular_file(built / "include" / kShmemHeader)) return built;
    const auto checkout = source_root().parent_path() / "third-party/shmem/install/shmem";
    if (std::filesystem::is_regular_file(checkout / "include" / kShmemHeader)) return checkout;
    throw std::runtime_error(
        "LL JIT cannot find the SHMEM Device headers; set DEEPEP_LL_JIT_SHMEM_ROOT "
        "to the installed SDK (the one that provides lib/libshmem.so)");
}

// One root for everything the JIT writes: DeepJIT's kernel cache and the
// toolkit compatibility links. DEEPEP_JIT_CACHE_DIR is DeepJIT's
// ':'-separated list; only its writable first entry is used here.
std::filesystem::path cache_root()
{
    const std::string configured = env_text("DEEPEP_JIT_CACHE_DIR");
    if (!configured.empty()) {
        const auto roots = split(configured, ':');
        if (roots.empty()) throw std::runtime_error("DEEPEP_JIT_CACHE_DIR has no usable cache root: " + configured);
        return roots.front();
    }
    return std::filesystem::path(required_env("HOME")) / ".dj";
}

// CANN names its per-architecture directories after the host (aarch64-linux,
// x86_64-linux), while DeepJIT always probes the aarch64 name.
std::string host_arch()
{
    utsname name{};
    if (uname(&name) == 0 && name.machine[0] != '\0') return name.machine;
    return "aarch64";
}

// DeepJIT discovers the toolkit at "$ASCEND_HOME/bin/bisheng" and
// "$ASCEND_HOME/<arch>-linux/asc/include/adv_api", where the architecture
// name is hard-coded to aarch64. Never hand it the real CANN root: current
// releases put bisheng under tools/bisheng_compiler and some installs have no
// aarch64-linux tree at all. The compatibility root exposes both names.
std::filesystem::path toolkit_root()
{
    const std::filesystem::path ascend_home = required_env("ASCEND_HOME_PATH");
    const auto root = cache_root() / "toolkit";
    std::filesystem::path compiler = ascend_home / "tools/bisheng_compiler/bin";
    if (!std::filesystem::is_regular_file(compiler / "bisheng")) compiler = ascend_home / "bin";
    for (const char* name : {"bisheng", "ld.lld"})
        if (!std::filesystem::is_regular_file(compiler / name))
            throw std::runtime_error("LL JIT cannot find " + std::string(name) + " under " + compiler.string());
    std::filesystem::create_directories(root / "bin");
    // Every rank builds this root on first use and the mapping is deterministic,
    // so an already correct link is success. Publish replacements through a
    // per-process staging name and rename, so no rank observes a missing link.
    auto link = [](const std::filesystem::path& target, const std::filesystem::path& at) {
        std::error_code error;
        if (std::filesystem::equivalent(at, target, error)) return;
        const auto staging =
            std::filesystem::path(at.string() + ".link." + std::to_string(static_cast<long>(getpid())));
        std::filesystem::remove(staging, error);
        std::filesystem::create_symlink(target, staging, error);
        if (error) throw std::runtime_error("LL JIT cannot link " + staging.string() + " to " + target.string());
        std::filesystem::rename(staging, at, error);
        if (error) std::filesystem::remove(staging, error);
        if (!std::filesystem::equivalent(at, target, error))
            throw std::runtime_error("LL JIT cannot link " + at.string() + " to " + target.string());
    };
    link(compiler / "bisheng", root / "bin/bisheng");
    link(compiler / "ld.lld", root / "bin/ld.lld");
    // Prefer the host layout, then the aarch64 one DeepJIT expects, then the
    // arch-independent devkit root this repository also accepts.
    const std::string arch = host_arch();
    std::filesystem::path adv_api = ascend_home / "include";
    for (const auto& candidate : {ascend_home / (arch + "-linux") / "asc/include/adv_api",
                                  ascend_home / "aarch64-linux/asc/include/adv_api",
                                  ascend_home / "asc/include/adv_api"})
        if (std::filesystem::is_directory(candidate)) {
            adv_api = candidate;
            break;
        }
    if (!std::filesystem::is_directory(adv_api))
        throw std::runtime_error("LL JIT cannot find the AscendC devkit headers; expected " +
                                 (ascend_home / (arch + "-linux") / "asc/include/adv_api").string());
    for (const auto& layout : std::vector<std::string>{arch + "-linux", "aarch64-linux"}) {
        std::filesystem::create_directories(root / layout / "asc/include");
        link(adv_api, root / layout / "asc/include/adv_api");
    }
    return root;
}

std::vector<std::filesystem::path> include_dirs()
{
    const std::filesystem::path ascend_home = required_env("ASCEND_HOME_PATH");
    const std::string arch = host_arch();
    const auto shmem = shmem_root();
    std::vector<std::filesystem::path> dirs = {
        source_root(),
        shmem / "include",
        shmem / "src",
        shmem / "src/device",
        shmem / "src/device_simt",
        // Nonexistent -I entries are ignored by Bisheng; both layouts are
        // listed so the same Host works on either CANN packaging.
        ascend_home / "include",
        ascend_home / (arch + "-linux") / "include",
        ascend_home / (arch + "-linux") / "asc/include",
        ascend_home / "aarch64-linux/include",
        ascend_home / "include/experiment/runtime",
        ascend_home / "compiler/tikcpp",
        ascend_home / "compiler/tikcpp/tikcfw",
        ascend_home / "compiler/tikcpp/tikcfw/impl",
        ascend_home / "compiler/tikcpp/tikcfw/interface",
    };
    for (const auto& extra : split(env_text("DEEPEP_LL_JIT_INCLUDE_DIRS"), ':')) dirs.emplace_back(extra);
    return dirs;
}

// DeepJIT only hashes the entry text it is handed, and its include parser
// rejects quoted includes. The entry is therefore named with angle brackets and
// resolved through the include list, while everything below it keeps the same
// quoted includes as the AOT build. Because that entry text is emitted by the
// Host, the kernel and SHMEM header trees are folded into the runtime
// signature instead, so any header change still invalidates the disk cache.
void collect_headers(const std::filesystem::path& root, std::vector<std::filesystem::path>& files)
{
    std::error_code ignored;
    if (!std::filesystem::is_directory(root, ignored)) return;
    for (const auto& item : std::filesystem::recursive_directory_iterator(
             root, std::filesystem::directory_options::skip_permission_denied, ignored)) {
        const auto extension = item.path().extension();
        if (item.is_regular_file() && (extension == ".h" || extension == ".hpp" || extension == ".inc"))
            files.push_back(item.path());
    }
}

std::string header_fingerprint()
{
    std::vector<std::filesystem::path> files;
    collect_headers(source_root(), files);
    const auto shmem = shmem_root();
    collect_headers(shmem / "include", files);
    collect_headers(shmem / "src", files);
    std::sort(files.begin(), files.end());
    uint64_t digest = 1469598103934665603ULL;
    const auto update = [&digest](const char* data, const size_t size) {
        for (size_t index = 0; index < size; ++index) {
            digest ^= static_cast<unsigned char>(data[index]);
            digest *= 1099511628211ULL;
        }
    };
    for (const auto& file : files) {
        const std::string path = file.generic_string();
        update(path.data(), path.size());
        std::ifstream stream(file, std::ios::binary);
        const std::string content{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
        update(content.data(), content.size());
    }
    std::ostringstream text;
    text << std::hex << digest;
    return text.str();
}

std::string entry_source(const LlDispatchJitSpec& spec)
{
    if (!std::filesystem::is_regular_file(source_root() / kDispatchEntry))
        throw std::runtime_error("LL JIT cannot find " + (source_root() / kDispatchEntry).string());
    std::ostringstream source;
    source << "#define DEEPEP_LL_JIT_SPECIALIZED 1\n"
           << "#define DEEPEP_LL_JIT_SPEC_BS " << spec.bs << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_H " << spec.h << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_TOPK " << spec.topK << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_EXPERTS " << spec.experts << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_CORES " << spec.cores << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_WORLD " << spec.world << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_INPUT_CLASS " << spec.inputClass << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_GROUP " << spec.group << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_INPUT_MODE " << spec.inputMode << "U\n"
           << "#define DEEPEP_LL_JIT_SPEC_E4M3 " << (spec.e4m3 ? 1 : 0) << "\n"
           << "#define DEEPEP_LL_JIT_SPEC_DEDUP " << (spec.dedup ? 1 : 0) << "\n"
           << "#include <" << kDispatchEntry << ">\n";
    return source.str();
}

std::string entry_source(const LlCombineJitSpec& spec, Operation operation)
{
    const char* entry = operation == Operation::UbxCombine ? kUbxCombineEntry : kMeshCombineEntry;
    if (!std::filesystem::is_regular_file(source_root() / entry))
        throw std::runtime_error("LL JIT cannot find " + (source_root() / entry).string());
    const char* prefix = operation == Operation::UbxCombine ? "DEEPEP_UBX_JIT" : "DEEPEP_LL_JIT";
    std::ostringstream source;
    source << "#define " << prefix << "_SPECIALIZED 1\n"
           << "#define " << prefix << "_SPEC_BS " << spec.bs << "U\n"
           << "#define " << prefix << "_SPEC_H " << spec.h << "U\n"
           << "#define " << prefix << "_SPEC_TOPK " << spec.topK << "U\n"
           << "#define " << prefix << "_SPEC_EXPERTS " << spec.experts << "U\n"
           << "#define " << prefix << "_SPEC_CORES " << spec.cores << "U\n"
           << "#define " << prefix << "_SPEC_WORLD " << spec.world << "U\n";
    if (operation == Operation::UbxCombine)
        source << "#define DEEPEP_UBX_JIT_SPEC_GROUP " << spec.group << "U\n";
    source << "#define " << prefix << "_SPEC_BF16 " << (spec.bf16 ? 1 : 0) << "\n"
           << "#define " << prefix << "_SPEC_DEDUP " << (spec.dedup ? 1 : 0) << "\n"
           << "#include <" << entry << ">\n";
    return source.str();
}

// Bisheng options for the real JIT compile. Mirrors the Ahead-of-Time Device
// build, including -std=c++17, which CMake adds there but DeepJIT does not.
std::vector<std::string> jit_flags(Operation operation)
{
    std::vector<std::string> flags = {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        // Keep the JIT translation unit kernel-only: the SK symbols and the
        // in-library launch helpers belong to the Ahead-of-Time library, and
        // DeepJIT requires exactly one kernel entry per ELF.
        operation == Operation::UbxCombine ? "-DDEEPEP_UBX_JIT=1" : "-DDEEPEP_LL_JIT=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        std::string("-DACLSHMEMI_HNS_1825_PI_ON_CHIP=") + DEEPEP_JIT_STRINGIFY(DEEPEP_JIT_PI_ON_CHIP),
        "--cce-auto-infer-kernel-type=false",
    };
#ifdef DEEPEP_JIT_REFERENCE_PROFILE
    if (operation == Operation::Dispatch) flags.emplace_back("-DDEEPEP_LL_REFERENCE_PROFILE=1");
#endif
#ifdef DEEPEP_JIT_COUNT_HISTOGRAM
    if (operation == Operation::Dispatch) flags.emplace_back("-DDEEPEP_LL_COUNT_HISTOGRAM=1");
#endif
    for (const auto& extra : split(env_text("DEEPEP_LL_JIT_DEFINES"), ';')) flags.push_back(extra);
    return flags;
}

std::string spec_tag(const LlDispatchJitSpec& spec)
{
    std::ostringstream tag;
    tag << "ll_dispatch_b" << spec.bs << "_h" << spec.h << "_k" << spec.topK << "_e" << spec.experts << "_n"
        << spec.cores << "_w" << spec.world << "_c" << spec.inputClass << "_g" << spec.group << "_i"
        << spec.inputMode << "_q" << (spec.e4m3 ? 1 : 0) << "_d" << (spec.dedup ? 1 : 0);
    return tag.str();
}

std::string spec_tag(const LlCombineJitSpec& spec, Operation operation)
{
    std::ostringstream tag;
    tag << (operation == Operation::UbxCombine ? "ll_combine_ubx" : "ll_combine_mesh") << "_b" << spec.bs << "_h"
        << spec.h << "_k" << spec.topK << "_e" << spec.experts << "_n" << spec.cores << "_w" << spec.world << "_g"
        << spec.group << "_t" << (spec.bf16 ? 1 : 0) << "_d" << (spec.dedup ? 1 : 0);
    return tag.str();
}

std::shared_ptr<deep_jit::Runtime<deep_jit::Ascend>> create_runtime()
{
    const auto includes = include_dirs();
    const std::string extra = env_text("DEEPEP_LL_JIT_SIGNATURE");
    const std::string signature = (extra.empty() ? std::string() : extra + ':') + header_fingerprint();
    deep_jit::Config config(module_dir(), "DEEPEP", signature, includes, {});
    // The Ascend backend resolves its toolkit from ASCEND_HOME_PATH when the
    // runtime is constructed, so expose the compatibility root for that call.
    const std::string original = env_text("ASCEND_HOME_PATH");
    const std::string shim = toolkit_root().string();
    setenv("ASCEND_HOME_PATH", shim.c_str(), 1);
    try {
        auto instance = std::make_shared<deep_jit::Runtime<deep_jit::Ascend>>(config);
        setenv("ASCEND_HOME_PATH", original.c_str(), 1);
        return instance;
    } catch (...) {
        setenv("ASCEND_HOME_PATH", original.c_str(), 1);
        throw;
    }
}

struct LoadedKernel {
    std::shared_ptr<deep_jit::ascend::Kernel> kernel;
    std::filesystem::path artifact;
};

// The registry keeps the module alive while an asynchronous launch, a captured
// graph or a Python plan may still reference its entry. A conservative process
// lifetime matches the AOT library, which also stays loaded.
std::map<std::string, LoadedKernel>& kernel_cache()
{
    static std::map<std::string, LoadedKernel> instances;
    return instances;
}

std::string context_key()
{
    aclrtContext context = nullptr;
    const auto status = aclrtGetCurrentContext(&context);
    if (status != ACL_SUCCESS || context == nullptr)
        throw std::runtime_error("LL JIT requires an initialized ACL context on the creating NPU");
    return std::to_string(reinterpret_cast<uintptr_t>(context));
}

LoadedKernel& load_kernel(const std::string& tag, const std::string& source, Operation operation)
{
    const std::string key = context_key() + ':' + tag;
    auto entry = kernel_cache().find(key);
    if (entry != kernel_cache().end()) return entry->second;

    if (!runtime()) runtime() = create_runtime();
    deep_jit::ascend::CompilerOptions overrides;
    overrides.extra_bisheng_flags = jit_flags(operation);
    const std::filesystem::path artifact = runtime()->compile_without_load(tag, source, overrides);
    LoadedKernel loaded{deep_jit::ascend::Kernel::load(artifact, runtime()->env), artifact};
    entry = kernel_cache().emplace(key, std::move(loaded)).first;
    return entry->second;
}

LoadedKernel& prepared_kernel(const std::string& tag)
{
    const std::string key = context_key() + ':' + tag;
    auto entry = kernel_cache().find(key);
    if (entry == kernel_cache().end())
        throw std::runtime_error("LL JIT kernel " + tag + " was not prepared in this ACL context; call prepare first");
    return entry->second;
}
}  // namespace

bool ll_jit_available()
{
    return true;
}

void ll_jit_prepare_dispatch(const LlDispatchJitSpec& spec)
{
    std::lock_guard<std::mutex> lock(launch_mutex());
    load_kernel(spec_tag(spec), entry_source(spec), Operation::Dispatch);
}

void ll_jit_prepare_combine(const LlCombineJitSpec& spec, bool rdma)
{
    std::lock_guard<std::mutex> lock(launch_mutex());
    const Operation operation = rdma ? Operation::UbxCombine : Operation::MeshCombine;
    load_kernel(spec_tag(spec, operation), entry_source(spec, operation), operation);
}

void ll_jit_launch_dispatch(void* stream, const EpServerDedupKernelArgs& args, const LlDispatchJitSpec& spec)
{
    std::lock_guard<std::mutex> lock(launch_mutex());
    auto& kernel = prepared_kernel(spec_tag(spec));

    // Assign fields instead of using a designated initializer: DeepJIT adds
    // launch options over time, and every omitted field must stay defaulted.
    deep_jit::ascend::LaunchOptions options;
    options.stream = static_cast<aclrtStream>(stream);
    options.num_blocks = static_cast<int>(spec.cores);
    options.num_ubuf_bytes = 0;
    runtime()->launch(kernel.kernel, options, args.commArgs, args.x, args.expertIds, args.expertScales,
                      args.expandXOut, args.dynamicScalesOut, args.expertTokenNumsOut, args.sendCountsOut,
                      args.tokenTypeOut, args.destinationIndexOut, args.relayReadIndexOut, args.sourceMaskOut,
                      args.expertScalesOut, args.tilingData, args.reservedMagic, args.inputScales, args.stageTimer,
                      args.timerMode, args.expertTimer);
}

void ll_jit_launch_combine(void* stream, const EpServerDedupCombineKernelArgs& args, const LlCombineJitSpec& spec)
{
    std::lock_guard<std::mutex> lock(launch_mutex());
    auto& kernel = prepared_kernel(spec_tag(spec, Operation::MeshCombine));

    deep_jit::ascend::LaunchOptions options;
    options.stream = static_cast<aclrtStream>(stream);
    options.num_blocks = static_cast<int>(spec.cores);
    options.num_ubuf_bytes = 0;
    runtime()->launch(kernel.kernel, options, args.commArgs, args.expertOut, args.destinationIndex, args.sendCounts,
                      args.expertScales, args.tokenType, args.relayReadIndex, args.sourceMask, args.yOut, args.tilingData);
}

void ll_jit_launch_combine_ubx(void* stream, const ubx::CombineArgs& args, const LlCombineJitSpec& spec)
{
    std::lock_guard<std::mutex> lock(launch_mutex());
    auto& kernel = prepared_kernel(spec_tag(spec, Operation::UbxCombine));

    deep_jit::ascend::LaunchOptions options;
    options.stream = static_cast<aclrtStream>(stream);
    options.num_blocks = static_cast<int>(spec.cores);
    options.num_ubuf_bytes = 0;
    runtime()->launch(kernel.kernel, options, args.comm, args.expert, args.destination, args.counts, args.weights,
                      args.types, args.relay, args.mask, args.output, args.tiling);
}
}  // namespace deepep
#endif
