// Copyright (c) 2026, Lu Lu

// Shared DeepJIT runtime, following DeepEP-Ascend: one lazily constructed
// Runtime owns the cache and compiler; operators only provide a source string
// and launch arguments. The Python package initializes the include roots once.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_ep {

// Set once by init_jit() before the first compile or device query.
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>> jit(nullptr);

// `root` is the deep_ep package directory. Every build mode bundles the device
// kernel sources into `<root>/_kernel_src`, so the include roots are fixed.
inline void init_jit(const std::string& root_path)
{
    const auto root = std::filesystem::absolute(root_path).lexically_normal();
    const auto kernel_root = root / "_kernel_src";
    const std::vector<std::filesystem::path> include_dirs = {
        root / "include",
        kernel_root / "csrc",
        kernel_root / "shmem/include",
        kernel_root / "shmem/src",
        kernel_root / "shmem/src/device",
        kernel_root / "shmem/src/device_simt",
    };
    const deep_jit::Config config(root, "EP", "", include_dirs, {"deep_ep/"});
    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>>([config] {
        return std::make_shared<deep_jit::Runtime<deep_jit::Ascend>>(config);
    });
}

inline int jit_num_vec_cores()
{
    return jit->device.get_num_vec_cores();
}

inline int64_t jit_num_ubuf_bytes_per_vec_core()
{
    return jit->device.get_num_ubuf_bytes_per_vec_core();
}

}  // namespace deep_ep
