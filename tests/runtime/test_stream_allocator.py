"""Execute the production Host allocator header with stream/allocator stand-ins."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_same_stream_allocation_and_foreign_input_protection(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for StreamAllocator Host oracle")
    includes = {
        "torch/extension.h": r"""
#pragma once
#include <cstdint>
#include <optional>
#include <initializer_list>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) if (!(condition)) throw std::runtime_error("device mismatch")
namespace c10 {
template<class T> using optional = std::optional<T>;
enum class DeviceType { PrivateUse1 };
}
namespace at { using IntArrayRef = std::initializer_list<int>; struct Scalar {}; }
namespace torch {
inline int active = 1, next = 0, records = 0;
struct TensorOptions { int device() const { return 0; } };
struct Storage { int id; int data_ptr() const { return id; } };
struct Tensor {
    int id, allocation;
    bool defined() const { return true; }
    int device() const { return 0; }
    Storage storage() const { return {id}; }
    void zero_() const { if (active != 1) throw std::runtime_error("wrong execution stream"); }
    void fill_(at::Scalar) const { zero_(); }
};
inline Tensor empty(at::IntArrayRef, const TensorOptions&) { return {++next, active}; }
}
""",
        "torch_npu/csrc/core/npu/NPUGuard.h": r"""
#pragma once
#include <torch/extension.h>
namespace c10_npu {
struct NPUStream {
    int id;
    int device_index() const { return 0; }
    int device() const { return 0; }
    static NPUStream unpack3(int id, int, c10::DeviceType) { return {id}; }
    bool operator!=(const NPUStream& other) const { return id != other.id; }
};
inline NPUStream getCurrentNPUStream() { return {torch::active}; }
struct NPUStreamGuard {
    int previous;
    explicit NPUStreamGuard(NPUStream stream): previous(torch::active) { torch::active = stream.id; }
    ~NPUStreamGuard() { torch::active = previous; }
};
}
""",
        "torch_npu/csrc/core/npu/NPUCachingAllocator.h": r"""
#pragma once
#include <torch_npu/csrc/core/npu/NPUGuard.h>
namespace c10_npu::NPUCachingAllocator {
inline void recordStream(int, NPUStream stream) {
    if (stream.id != 1) throw std::runtime_error("wrong recorded stream");
    ++torch::records;
}
}
""",
    }
    for name, content in includes.items():
        header = tmp_path / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(content, encoding="utf-8")
    source = tmp_path / "probe.cpp"
    source.write_text(
        r"""
#include <cassert>
#include "ops/stream_allocator.h"
int main() {
    using ascend_deepep::ops::StreamAllocator;
    torch::TensorOptions options;
    StreamAllocator same(std::nullopt);
    auto output = same.Empty({4}, options);
    same.Zeros({4}, options);
    same.Full({4}, {}, options);
    assert(output.allocation == 1 && torch::records == 0 && torch::active == 1);
    // Incoming/cached storage remains protected regardless of this allocator's setting.
    same.Record({torch::Tensor{100, 2}, std::nullopt});
    assert(torch::records == 1);
    StreamAllocator different(2);
    auto other = different.Empty({4}, options);
    different.Zeros({4}, options);
    different.Full({4}, {}, options);
    assert(other.allocation == 2 && torch::records == 4 && torch::active == 1);
    StreamAllocator explicit_same(1);
    explicit_same.Empty({4}, options);
    assert(torch::records == 4);
}
""",
        encoding="utf-8",
    )
    target = tmp_path / ("probe.exe" if os.name == "nt" else "probe")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(root / "csrc/direct"), str(source), "-o", str(target)],
        check=True,
        timeout=120,
    )
    try:
        subprocess.run([str(target)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("Windows application control blocked compiled Host oracle")
        raise
