// Copyright (c) 2026, Lu Lu

// Shared DeepJIT runtime used by every operator. Following DeepEP-Ascend, one
// lazy Runtime owns the cache and compiler; operators only provide a source
// string and launch arguments.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <acl/acl.h>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/env.hpp>

namespace deepep {

using JitKernel = deep_jit::ascend::Kernel;

deep_jit::Runtime<deep_jit::Ascend>& jit_runtime();

std::shared_ptr<JitKernel> jit_compile(const std::string& name, const std::string& source,
                                       const std::vector<std::string>& extra_flags);

template <typename... Args>
void jit_launch(const std::shared_ptr<JitKernel>& kernel, void* stream, int blocks, int ubuf_bytes, Args&&... args)
{
    deep_jit::ascend::LaunchOptions options;
    options.stream = static_cast<aclrtStream>(stream);
    options.num_blocks = blocks;
    options.num_ubuf_bytes = ubuf_bytes;
    options.num_launch_timeout_secs = deep_jit::get_env<int>("DEEPEP_JIT_LAUNCH_TIMEOUT", 300);
    jit_runtime().launch(kernel, options, std::forward<Args>(args)...);
}

}  // namespace deepep
