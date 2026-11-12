// Copyright (c) 2026, Lu Lu

// Optional runtime compilation of the LL Dispatch/Combine kernels through
// DeepJIT. Only the Host module uses this bridge. The Ahead-of-Time device
// library compiles the same entry headers without the specialization macros,
// so the device body is shared and only the compile-time axes differ.
#pragma once
#include <cstdint>
#include <stdexcept>

#include "../kernels/combine/low_latency/launch.hpp"
#include "../kernels/dispatch/low_latency/launch.hpp"
#include "../ubx/launch.hpp"

namespace deepep {
// Launch identity baked into one JIT dispatch translation unit. The generated
// entry defines DEEPEP_LL_JIT_SPEC_* from these fields; the device rechecks
// them against the launch description before running, and the Host adds them
// to the cache key.
struct LlDispatchJitSpec {
    uint32_t bs = 0;
    uint32_t h = 0;
    uint32_t topK = 0;
    uint32_t experts = 0;
    uint32_t cores = 0;       // active AIV count of this launch
    uint32_t world = 0;       // EP world size (CommArgs.rankSize)
    uint32_t inputClass = 0;  // DispatchDedup::kLlDispatchInputFp16/Bf16
    uint32_t group = 0;       // LlTilingKey: 4 or 16
    uint32_t inputMode = 0;   // EpServerDedupInputMode
    bool e4m3 = false;
    bool dedup = false;
};

// Launch identity of one Combine entry. `group` selects the UBX route-plan
// instantiation; the mesh implementation ignores it but keeps the field so the
// cache key covers the same source of truth.
struct LlCombineJitSpec {
    uint32_t bs = 0;
    uint32_t h = 0;
    uint32_t topK = 0;
    uint32_t experts = 0;
    uint32_t cores = 0;
    uint32_t world = 0;
    uint32_t group = 0;
    bool bf16 = false;
    bool dedup = false;
};

#ifdef DEEPEP_LL_JIT
bool ll_jit_available();
// Compiles (or hits the disk cache for) this exact identity and loads it in the
// current ACL context. Call before graph capture so a compiler/runtime error
// can still be reported to Python.
void ll_jit_prepare_dispatch(const LlDispatchJitSpec& spec);
void ll_jit_prepare_combine(const LlCombineJitSpec& spec, bool rdma);
// Launches a prepared kernel. Throws with a diagnostic when the environment is
// incomplete or the identity was never prepared.
void ll_jit_launch_dispatch(void* stream, const EpServerDedupKernelArgs& args, const LlDispatchJitSpec& spec);
void ll_jit_launch_combine(void* stream, const EpServerDedupCombineKernelArgs& args, const LlCombineJitSpec& spec);
void ll_jit_launch_combine_ubx(void* stream, const ubx::CombineArgs& args, const LlCombineJitSpec& spec);
#else
inline bool ll_jit_available()
{
    return false;
}
inline void ll_jit_prepare_dispatch(const LlDispatchJitSpec&)
{
    throw std::runtime_error("LL DeepJIT support is not compiled into this extension");
}
inline void ll_jit_prepare_combine(const LlCombineJitSpec&, bool)
{
    throw std::runtime_error("LL DeepJIT support is not compiled into this extension");
}
inline void ll_jit_launch_dispatch(void*, const EpServerDedupKernelArgs&, const LlDispatchJitSpec&)
{
    throw std::runtime_error("LL DeepJIT support is not compiled into this extension");
}
inline void ll_jit_launch_combine(void*, const EpServerDedupCombineKernelArgs&, const LlCombineJitSpec&)
{
    throw std::runtime_error("LL DeepJIT support is not compiled into this extension");
}
inline void ll_jit_launch_combine_ubx(void*, const ubx::CombineArgs&, const LlCombineJitSpec&)
{
    throw std::runtime_error("LL DeepJIT support is not compiled into this extension");
}
#endif
}  // namespace deepep
