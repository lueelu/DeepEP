"""Compile the production mapping-range check against a CPU Runtime double."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_mte_mapping_range_failures(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-only mapping test")
    root = Path(__file__).resolve().parents[2]
    acl = tmp_path / "acl"
    acl.mkdir()
    (acl / "acl.h").write_text("""#pragma once
#include <cstddef>
constexpr int ACL_SUCCESS = 0;
int aclrtMemGetAddressRange(void*, void**, std::size_t*);
""")
    # Compile the actual helper after it was folded into BufferRuntime, without
    # pulling Torch/NPU runtime dependencies into this CPU boundary test.
    runtime = (root / "csrc/clos/runtime/buffer_runtime.cpp").read_text(encoding="utf-8")
    begin = runtime.index("inline std::string MteMappingError(")
    end = runtime.index("\n}\n", begin) + 3
    (tmp_path / "mapping_check.h").write_text(
        "#include <acl/acl.h>\n#include <cstdint>\n#include <string>\n" + runtime[begin:end],
        encoding="utf-8",
    )
    source = tmp_path / "check.cpp"
    source.write_text("""
#include "mapping_check.h"
#include <cassert>
#include <limits>
int status = 0, calls = 0;
uintptr_t range_base = 4096;
std::size_t range_size = 8192;
int aclrtMemGetAddressRange(void* address, void** base, std::size_t* size) {
    assert(address != nullptr);
    ++calls;
    *base = reinterpret_cast<void*>(range_base);
    *size = range_size;
    return status;
}
int main() {
    assert(!MteMappingError(nullptr, 32).empty());
    assert(!MteMappingError(reinterpret_cast<void*>(4096), 0).empty());
    assert(calls == 0);
    assert(MteMappingError(reinterpret_cast<void*>(4096), 8192).empty());
    assert(MteMappingError(reinterpret_cast<void*>(8192), 4096).empty());
    assert(!MteMappingError(reinterpret_cast<void*>(8192), 4097).empty());
    assert(!MteMappingError(reinterpret_cast<void*>(4095), 1).empty());
    assert(!MteMappingError(reinterpret_cast<void*>(12288), 1).empty());
    assert(!MteMappingError(reinterpret_cast<void*>(16384), 1).empty());
    status = 507000;
    assert(MteMappingError(reinterpret_cast<void*>(4096), 32).find("507000") != std::string::npos);
    status = 0;
    range_base = 0;
    assert(!MteMappingError(reinterpret_cast<void*>(4096), 32).empty());
    range_base = 4096;
    range_size = 0;
    assert(!MteMappingError(reinterpret_cast<void*>(4096), 32).empty());
    range_size = 8192;
    assert(!MteMappingError(reinterpret_cast<void*>(4096), std::numeric_limits<std::size_t>::max()).empty());
}
""")
    binary = tmp_path / "check"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(tmp_path),
            "-I" + str(root / "csrc/clos"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True, timeout=10)
