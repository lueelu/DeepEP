// Copyright (c) 2026, Lu Lu

// Minimal host-side identity for one DeepJIT LL specialization. DeepEP-Ascend
// uses the generated source string itself as the cache key; this struct only
// carries the values that become template/macro arguments.
#pragma once

#include <cstddef>
#include <cstdint>

namespace deepep {

enum class LlJitOperation : uint32_t {
    Dispatch = 0U,
    MeshCombine = 1U,
    UbxCombine = 2U,
};

struct LlJitSignature {
    LlJitOperation operation = LlJitOperation::Dispatch;
    uint32_t bs = 0U;
    uint32_t hidden = 0U;
    uint32_t topK = 0U;
    uint32_t experts = 0U;
    uint32_t capacity = 0U;
    uint32_t world = 0U;
    uint32_t rank = 0U;
    uint32_t cores = 0U;
    uint32_t group = 0U;
    uint32_t inputClass = 0U;
    uint32_t inputMode = 0U;
    uint32_t e4m3 = 0U;
    uint32_t bf16 = 0U;
    uint32_t dedup = 0U;
};

// Compiles (DeepJIT mem/disk cache) and launches one operation. The caller owns
// the tensor arguments; the launch is asynchronous.
void ll_jit_launch(const LlJitSignature& signature, const void* arguments, std::size_t bytes, void* stream);

}  // namespace deepep
