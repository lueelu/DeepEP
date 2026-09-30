# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""CPU regressions for UB budgets, Tensor overlap and route-dependent Meta shapes."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

COMPONENT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiling():
    path = COMPONENT / "python/deep_ep_experimental/ep_gmm_fused/_tiling.py"
    spec = importlib.util.spec_from_file_location("ep_gmm_tiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reported_ub", [248 * 1024, 256 * 1024])
@pytest.mark.parametrize("rows", [29, 30])
def test_inverse_copy_budget(tiling, reported_ub, rows):
    case = dict(operator=tiling.GROUPED_MATMUL, shape=dict(m=128, k=256, n=2176), world_size=2, expert_num=2)
    profile = tiling.default_tiling(case)
    assert profile["local_copy_tile_rows"] == 29
    profile["local_copy_tile_rows"] = rows
    arguments = dict(n=2176, m=128, k=256, world_size=2, aic_cores=32, ub_bytes=reported_ub)
    if rows == 30:
        with pytest.raises(ValueError, match="UB capacity"):
            tiling.validate_tiling(tiling.GROUPED_MATMUL, profile, **arguments)
    else:
        tiling.validate_tiling(tiling.GROUPED_MATMUL, profile, **arguments)


def test_forward_copy_budget(tiling):
    case = dict(operator=tiling.ALL_TO_ALLV, shape=dict(m=128, k=256, n=256), world_size=2, expert_num=2)
    profile = tiling.default_tiling(case)
    for rows in (247, 248):
        profile["local_copy_tile_rows"] = rows
        if rows == 248:
            with pytest.raises(ValueError, match="UB capacity"):
                tiling.validate_tiling(tiling.ALL_TO_ALLV, profile, ub_bytes=256 * 1024)
        else:
            tiling.validate_tiling(tiling.ALL_TO_ALLV, profile)


def test_native_ub_budget(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for production Host validation")
    source = tmp_path / "budget.cpp"
    source.write_text(
        """#include "tiling_validation.hpp"
#include <cassert>
using namespace deepep::ep_gmm_fused::TilingValidation;
int main() {
    assert(LocalCopyRowCapacity(2176, 256 * 1024) == 29);
    assert(LocalCopyRowCapacity(2176, 248 * 1024) == 29);
    assert(LocalCopyRowCapacity(2176, 128 * 1024) == 15);
    assert(LocalCopyRowCapacity(2176, 256) == 0);
    assert(LocalCopyRowCapacity(0, 256 * 1024) == 0);
    GroupedMatMul inverse{128, 256, 256, 10, 29};
    assert(ValidateGroupedMatMul(inverse, 2176, 2, 32, 256 * 1024) == nullptr);
    inverse.localCopyTileRows = 30;
    assert(ValidateGroupedMatMul(inverse, 2176, 2, 32, 256 * 1024) != nullptr);
    AllToAllV forward{128, 256, 256, 64, 1280, 2, 247, 1, 0, 1};
    assert(ValidateAllToAllV(forward, 128, 256, 2, 32, 256 * 1024) == nullptr);
    forward.localCopyTileRows = 248;
    assert(ValidateAllToAllV(forward, 128, 256, 2, 32, 256 * 1024) != nullptr);
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("budget.exe" if sys.platform == "win32" else "budget")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(COMPONENT / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


@pytest.fixture(scope="module")
def torch_contracts(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if sys.platform != "linux" or not (shutil.which("c++") or shutil.which("g++")):
        pytest.skip("Linux C++ compiler required for CPU ATen contract tests")
    from torch.utils.cpp_extension import load_inline

    extension = load_inline(
        name="ep_gmm_fused_contract_tests",
        cpp_sources="""#include "tensor_validation.hpp"
void overlap(const at::Tensor& out, const at::Tensor& input) {
    deepep::ep_gmm_fused::CheckNoOverlap(out, input, "out", "input");
}
at::Tensor inverse_meta(const at::Tensor& x, const at::Tensor& weight,
                        const at::Tensor& local, const at::Tensor& global,
                        const at::Tensor& out, bool preallocated) {
    if (preallocated) {
        return deepep::ep_gmm_fused::GroupedMatMulAllToAllVUdmaOutMeta(x, weight, local, global, 2, 2, out, {});
    }
    return deepep::ep_gmm_fused::GroupedMatMulAllToAllVUdmaMeta(x, weight, local, global, 2, 2, {});
}
""",
        functions=["overlap", "inverse_meta"],
        extra_include_paths=[str(COMPONENT / "csrc")],
        extra_cflags=["-O0"],
        build_directory=str(tmp_path_factory.mktemp("aten-contracts")),
    )
    return torch, extension


def test_contiguous_view_overlap(torch_contracts):
    torch, extension = torch_contracts
    storage = torch.empty(131072, dtype=torch.bfloat16)
    output = storage[:65536].view(256, 256)
    for other in (storage[256:65792].view(256, 256), output):
        with pytest.raises(RuntimeError, match="must not overlap"):
            extension.overlap(output, other)
        with pytest.raises(RuntimeError, match="must not overlap"):
            extension.overlap(other, output)
    extension.overlap(output, storage[65536:].view(256, 256))
    extension.overlap(output, torch.empty_like(output))
    extension.overlap(storage[:0], storage[:0])
    extension.overlap(output, storage[:0])
    # Route tensors may be reinterpreted views of the same allocation.
    with pytest.raises(RuntimeError, match="must not overlap"):
        extension.overlap(output, storage.view(torch.int64))


@pytest.mark.parametrize("fake", [False, True])
def test_inverse_meta_requires_known_output_rows(torch_contracts, fake):
    torch, extension = torch_contracts
    x = torch.empty((2, 256), device="meta", dtype=torch.bfloat16)
    weight = torch.empty((1, 256, 256), device="meta", dtype=torch.bfloat16)
    local = torch.tensor([1, 3], dtype=torch.int64, device="meta")
    global_counts = torch.tensor([1, 1], dtype=torch.int64, device="meta")
    out = torch.empty((4, 256), device="meta", dtype=torch.bfloat16)
    if fake:
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode()
        x, weight, local, global_counts, out = map(mode.from_tensor, (x, weight, local, global_counts, out))
    with pytest.raises(RuntimeError, match="require out="):
        extension.inverse_meta(x, weight, local, global_counts, out, False)
    result = extension.inverse_meta(x, weight, local, global_counts, out, True)
    assert result.shape == (4, 256)


def test_version_uses_distribution_metadata(monkeypatch):
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0.dev99" if name == "ascend-deepep" else None)
    path = COMPONENT / "python/deep_ep_experimental/ep_gmm_fused/_version.py"
    spec = importlib.util.spec_from_file_location("component_version", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__version__ == "0.1.0.dev99"
