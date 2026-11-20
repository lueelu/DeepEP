# Copyright (c) 2026, Lu Lu
"""Execute production phase ordering and host-barrier CQ ownership on CPU."""

from pathlib import Path
import shutil
import subprocess

import pytest


def body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start : end + 1]
    raise AssertionError(f"Unclosed function: {signature}")


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_aiv0_reduction_waits_for_all_prologues_and_keeps_exclusive_qp_ownership(tmp_path, backend):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("AIV0 owner checks require a C++ compiler")
    directory = Path(__file__).parent
    kernels = directory.parent / "csrc_ep32/kernels/mega_moe"
    kernel = (kernels / "mega_moe_arch35.hpp").read_text(encoding="utf-8")
    wave = body(kernel, "::ProcessWave(Derived& derived)")
    # Execute the actual routed/shared stage calls and the entire output tail.
    # Input preparation is outside this harness, but its existing barrier must
    # precede every prologue and the QP1 handoff.
    assert wave.index("ProcessInputPreparationStage();") < wave.index("SyncAll<false>();")
    assert wave.index("SyncAll<false>();") < wave.index("ProcessSharedExpertGmm1(")
    assert (
        wave.index("SyncAll<false>();")
        < wave.index("FinishHierarchicalDispatch(")
        < wave.index("derived.ProcessMoeExpertStages(")
    )
    stages = wave[wave.index("    // 阶段 3：") : wave.index("    // 恢复入口时")]
    dispatch = (kernels / "stage/mega_moe_hierarchical_dispatch.hpp").read_text(encoding="utf-8")
    fixture = (directory / "aiv0_combine_owner_test.cpp").read_text(encoding="utf-8")
    source = tmp_path / "aiv0_combine_owner.cpp"
    source.write_text(
        fixture.replace("/* PRODUCTION_STAGES */", stages).replace(
            "/* PRODUCTION_HOST_BARRIER */", body(dispatch, "void CrossRankSyncInHierarchicalHost(")[1:-1]
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "aiv0_combine_owner"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-UNDEBUG",
            f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1",
            f"-I{directory}",
            f"-I{directory / 'layout_stubs'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_input_metadata_order(tmp_path, backend):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Rail metadata checks require a C++ compiler")
    directory = Path(__file__).parent
    kernel = (directory.parent / "csrc_ep32/kernels/mega_moe/mega_moe_arch35.hpp").read_text(encoding="utf-8")
    pack = body(kernel, "::PackOrPostRailMetadata()")
    stage = body(kernel, "::ProcessInputPreparationStage()").split("QuantizeLocalTokens<")[0] + "}"
    fixture = (directory / "rail_metadata_order.cpp").read_text(encoding="utf-8")
    source, binary = tmp_path / "metadata.cpp", tmp_path / "metadata"
    source.write_text(fixture.replace("/* PACK */", pack).replace("/* STAGE */", stage), encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}", str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
