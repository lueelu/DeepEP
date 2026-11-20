# Copyright (c) 2026, Lu Lu

"""Exercise the real CQ progress adapter without requiring NPU hardware."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_roce_transport import compile_and_run


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE", None], ids=["hns-1825", "xscale", "mte-only"])
def test_roce_progress_ready_only_and_error_contract(tmp_path, backend):
    compile_and_run(
        tmp_path,
        backend,
        "roce_progress_test.cpp",
        stub_directories=("roce_progress_stubs",),
    )


def test_roce_progress_host_pass_does_not_require_device_sdk_internals(tmp_path):
    compile_and_run(
        tmp_path,
        "HNS_1825",
        "roce_progress_host_pass_test.cpp",
        stub_directories=("roce_progress_stubs",),
    )


def _cpp_block(source, start, end):
    begin = source.index(start)
    return source[begin : source.index(end, begin) + len(end)]


def test_roce_progress_with_pinned_sdk_layouts_and_owner_helper(tmp_path):
    """Use the locked Git objects when an SDK source repository is available.

    Set MEGAMOE_SHMEM_SOURCE for an external checkout. Its checked-out revision
    need not match: read the exact lockfile revision without changing that tree.
    This checks pinned layouts/signatures, not a full CANN device compilation.
    """
    repository = Path(__file__).resolve().parents[3]
    sdk = Path(os.environ.get("MEGAMOE_SHMEM_SOURCE", repository / "third-party" / "shmem"))
    revision = json.loads((repository / "dependencies.lock.json").read_text(encoding="utf-8"))["shmem"]["revision"]
    if not sdk.is_dir():
        pytest.skip("Pinned SHMEM source checkout is not available")
    available = subprocess.run(
        ["git", "-C", str(sdk), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        timeout=10,
    )
    if available.returncode != 0 and "MEGAMOE_SHMEM_SOURCE" not in os.environ:
        pytest.skip("Pinned SHMEM Git object is not available")
    assert available.returncode == 0, available.stderr

    def show(path):
        return subprocess.check_output(
            ["git", "-C", str(sdk), "show", f"{revision}:{path}"], encoding="utf-8", timeout=10
        )

    metadata = show("src/device/gm2gm/engine/shmemi_device_rdma.h")
    layout = show("src/host_device/shmemi_rdma_cqe_layout.h")
    backend = show("src/device/gm2gm/engine/rdma_backends/rdma_device_backend_hns_1825.hpp")
    declarations = []
    for name in ("aclshmemi_rdma_backend_t", "aclshmemi_rdma_db_mode_t"):
        declarations.append(_cpp_block(metadata, f"enum class {name}", ";"))
    for name in ("aclshmemi_rdma_info", "aclshmemi_rdma_sq_ctx", "aclshmemi_rdma_cq_ctx"):
        declarations.append(_cpp_block(metadata, f"struct {name} {{", "\n};"))
    declarations.append(_cpp_block(layout, "struct aclshmemi_hns_1825_cqe_t {", "\n};"))

    # The production adapter is compiled against the actual pinned layouts,
    # and executes the actual owner helper, rather than their test doubles.
    stubs = Path(__file__).with_name("roce_progress_stubs")
    shim = (stubs / "shmem.h").read_text(encoding="utf-8")
    start = shim.index("enum class aclshmemi_rdma_backend_t")
    end = shim.index("static_assert(sizeof(aclshmemi_hns_1825_cqe_t)", start)
    shim = shim[:start] + "\n".join(declarations) + "\n" + shim[end:]
    old_owner = _cpp_block(shim, "inline bool aclshmemi_roce_hns_1825_check_cqe_owner", "\n}")
    owner = _cpp_block(backend, "ACLSHMEM_DEVICE bool aclshmemi_roce_hns_1825_check_cqe_owner", "\n}")
    shim = shim.replace(old_owner, owner.replace("ACLSHMEM_DEVICE", "inline", 1))

    # Compile exact declarations alongside the doubles to detect signature
    # drift of the remaining SDK helpers called by the adapter.
    for marker in (
        "ACLSHMEM_DEVICE uint32_t aclshmemi_hns_1825_read_u32_gm",
        "ACLSHMEM_DEVICE void aclshmemi_roce_hns_1825_dump_cqe",
    ):
        begin = backend.index(marker)
        declaration = backend[begin : backend.index("\n{", begin)].replace("ACLSHMEM_DEVICE", "inline", 1)
        shim += "\n" + declaration + ";\n"
    (tmp_path / "shmem.h").write_text(shim, encoding="utf-8")

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Pinned SDK adapter check requires g++ or clang++")
    source = Path(__file__).with_name("roce_progress_test.cpp")
    binary = tmp_path / "roce_progress_pinned"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
            f"-I{tmp_path}",
            f"-I{stubs}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
