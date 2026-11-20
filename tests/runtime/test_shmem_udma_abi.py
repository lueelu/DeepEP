# Copyright (c) 2026, Lu Lu

"""Compile real pinned SHMEM WQE builders; emulate only device types and queue I/O."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def function(source, name, *, template=True):
    position = source.index(name + "(")
    start = source.rfind("\ntemplate <" if template else "\nACLSHMEM_DEVICE", 0, position) + 1
    opening = source.index("\n{", position) + 1
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def sdk_source(path):
    checkout = Path(os.environ.get("DEEPEP_SHMEM_TEST_SOURCE", ROOT / "third-party/shmem"))
    revision = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]["revision"]
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{revision}:{path}"], capture_output=True, encoding="utf-8"
    )
    if result.returncode:
        pytest.skip("Pinned SHMEM Git sources required; initialize submodule or set DEEPEP_SHMEM_TEST_SOURCE")
    return result.stdout


def contract_source():
    common = sdk_source("src/device/shmemi_device_common.hpp")
    udma = sdk_source("src/device/gm2gm/engine/shmem_device_udma.hpp")
    names = (
        "aclshmemi_udma_check_config",
        "aclshmemi_udma_require_submit_cqe",
        "aclshmemi_udma_build_flag",
        "aclshmemi_fill_sge_header_ctx",
        "aclshmemi_udma_get_reduce_attr",
        "aclshmemi_udma_fill_sqe_base_ctx",
        "aclshmemi_udma_fill_sqe_ctx",
        "aclshmemi_udma_stage_send_wqe",
    )
    production = "\n".join(
        [
            function(common, "aclshmemi_debug_assert_remote_mr_addr", template=False),
            function(common, "aclshmemi_remote_mr_addr", template=False),
        ]
        + [function(udma, name) for name in names]
    )
    combine = (ROOT / "csrc/kernels/combine/combine.cpp").read_text(encoding="utf-8")
    batch = combine[combine.index("struct CrossPutBatch") :]
    start = batch.index("    __aicore__ inline void Add(")
    body = batch.index("\n    {", start) + 5
    depth, end = 1, body + 1
    while depth:
        depth += (batch[end] == "{") - (batch[end] == "}")
        end += 1
    fixture = (Path(__file__).with_name("shmem_udma_abi_test.cpp")).read_text(encoding="utf-8")
    return fixture.replace("/* PINNED_SHMEM_BUILDERS */", production).replace(
        "/* PRODUCTION_COMBINE_ADD */", batch[start:end]
    )


@pytest.mark.parametrize("debug", [False, True], ids=["release", "sdk-debug"])
def test_live_and_symmetric_wqe_addresses_with_pinned_sdk(tmp_path, debug):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = tmp_path / "contract.cpp"
    source.write_text(contract_source(), encoding="utf-8")
    binary = tmp_path / ("contract.exe" if os.name == "nt" else "contract")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-D__CCE_AICORE__=1",
            f"-DTEST_DEBUG={int(debug)}",
            "-iquote",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device_macro", ["__CCE_AICORE__", "__CCE_KT_TEST__"])
def test_adapters_preserve_distinct_device_address_spaces(tmp_path, device_macro):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang required to enforce distinct pointer address spaces")
    fixture = Path(__file__).with_name("shmem_udma_address_space_test.cpp")
    command = [
        compiler,
        "-std=c++17",
        f"-D{device_macro}=1",
        "-iquote",
        str(ROOT),
        "-c",
        str(fixture),
        "-o",
        str(tmp_path / "spaces.o"),
    ]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    # Ensure this compiler actually rejects the conversion that Bisheng rejected.
    invalid = subprocess.run(
        command + ["-DTEST_ILLEGAL_CAST=1"], capture_output=True, encoding="utf-8", errors="replace", timeout=90
    )
    assert invalid.returncode != 0
    assert "reinterpret_cast" in invalid.stderr and "not allowed" in invalid.stderr


def test_host_pass_does_not_require_device_udma_types(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = tmp_path / "host.cpp"
    # Like shmem.h's Host branch, provide no Device UDMA types or address qualifiers.
    source.write_text('#include "csrc/kernels/dispatch/udma_live_address.hpp"\nint host_pass() { return 0; }\n')
    result = subprocess.run(
        [compiler, "-std=c++17", "-iquote", str(ROOT), "-c", str(source), "-o", str(tmp_path / "host.o")],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fm_publisher_waits_for_wrap_copies_without_requesting_cqe(tmp_path):
    """Compile the production FM publisher against the pinned SDK signatures."""
    import re

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    sdk = sdk_source("src/device/gm2gm/engine/shmem_device_udma.hpp")
    declarations = []
    for name in (
        "aclshmemi_udma_copy_wqe_from_ub_async",
        "aclshmemi_udma_wait_wqe_copy_from_ub",
        "aclshmemi_udma_post_send_update_info",
    ):
        match = re.search(r"ACLSHMEM_DEVICE void " + name + r"\([^{}]*?\)\s*\{", sdk)
        assert match is not None, name
        declarations.append(match[0].rstrip().removesuffix("{").replace("ACLSHMEM_DEVICE", "") + ";")
    production = (ROOT / "csrc/kernels/dispatch/fm_queue.hpp").read_text(encoding="utf-8")
    start = production.index("__aicore__ inline void PublishFmSimtWqes(")
    production = production[start : production.index("\n}", start) + 2]
    source = tmp_path / "publisher.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#define __aicore__
#define __gm__
#define __ubuf__
constexpr uint32_t EVENT_ID0 = 0;
namespace AscendC {
enum class TPosition { VECOUT };
template<class T> struct LocalTensor {
    struct { uint8_t logicPos; uint64_t bufferAddr; } address_{};
};
}
struct aclshmemi_udma_wq_ctx_t { uint32_t depth = 32, wqe_size = 64; uintptr_t buf_addr; };
struct aclshmemi_udma_queue_state_t { uint32_t sq_head, cqe_cnt; };
"""
        + "\n".join(declarations)
        + production
        + r"""
bool pending = false;
uint8_t* destination;
const uint8_t* origin;
uint32_t length, copies, doorbells;
void aclshmemi_udma_copy_wqe_from_ub_async(uint8_t* dst, AscendC::LocalTensor<uint8_t>& src,
                                         uint32_t bytes, uint32_t event) {
    assert(event == EVENT_ID0 && !pending);
    pending = true; destination = dst;
    origin = reinterpret_cast<const uint8_t*>(src.address_.bufferAddr); length = bytes;
}
void aclshmemi_udma_wait_wqe_copy_from_ub(uint32_t event) {
    assert(event == EVENT_ID0 && pending);
    std::memcpy(destination, origin, length); pending = false; ++copies;
}
void aclshmemi_udma_post_send_update_info(uint32_t head, aclshmemi_udma_wq_ctx_t*,
                                        aclshmemi_udma_queue_state_t* state) {
    assert(!pending && state->cqe_cnt == 7); // FM data WQEs do not request CQEs.
    state->sq_head = head; ++doorbells;
}
int main() {
    uint8_t input[32 * 64], ring[32 * 64];
    for (uint32_t i = 0; i < sizeof(input); ++i) input[i] = i % 251;
    for (uint32_t head : {0U, 31U, 0xfffffff0U})
    for (uint32_t count : {1U, 4U, 32U}) {
        std::memset(ring, 0xa5, sizeof(ring)); copies = doorbells = 0;
        aclshmemi_udma_wq_ctx_t ctx{32, 64, reinterpret_cast<uintptr_t>(ring)};
        aclshmemi_udma_queue_state_t state{head, 7};
        PublishFmSimtWqes(input, count, &ctx, &state);
        assert(!pending && doorbells == 1 && state.sq_head == uint32_t(head + count));
        assert(copies == (head % 32 + count > 32 ? 2U : 1U));
        for (uint32_t i = 0; i < count; ++i)
            assert(std::memcmp(ring + ((head + i) % 32) * 64, input + i * 64, 64) == 0);
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "publisher.exe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-include", "initializer_list", str(source), "-o", str(binary)],
        capture_output=True,
        encoding="utf-8",
        timeout=90,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stderr
