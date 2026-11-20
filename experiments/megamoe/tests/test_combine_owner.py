# Copyright (c) 2026, Lu Lu

"""Exercise production Combine ownership and its scheduler with CPU device doubles."""

from pathlib import Path
import shutil
import subprocess

import pytest


DIRECTORY = Path(__file__).parent
KERNEL = DIRECTORY.parent / "csrc_ep32/kernels/mega_moe"


def function_body(source, name):
    opening = source.index("{", source.index(name))
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[opening:end]


def combine_owner_source():
    wave = (KERNEL / "mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    combine = (KERNEL / "stage/mega_moe_gmm2_combine.hpp").read_text(encoding="utf-8")
    utils = (KERNEL / "common/mega_moe_utils.hpp").read_text(encoding="utf-8")
    fixture = (DIRECTORY / "combine_owner_test.cpp").read_text(encoding="utf-8")
    for marker, source, name in [
        ("READY_COMBINE", wave, "::TryRunReadyCombine("),
        ("OWNED_RANGE", combine, "WorkRange GetWaveCombineOwnedRange("),
        ("PRODUCER_COUNT", combine, "uint32_t GetTokenCombineProducerCount("),
        ("BALANCED_RANGE", utils, "WorkRange GetBalancedTokenRange("),
        ("ROTATED_RANGE", utils, "WorkRange GetRotatedBalancedTokenRange("),
        ("DISPATCH_SUBSET", utils, "uint32_t GetDispatchSubsetJobIndex("),
        ("M_GROUP_COUNT", utils, "uint32_t GetMGroupCountForRows("),
    ]:
        fixture = fixture.replace(f"/* {marker} */", function_body(source, name)[1:-1])
    # Match the kernel's tile sizes instead of baking test-only producer counts in.
    constants = (KERNEL / "common/mega_moe_constants.hpp").read_text(encoding="utf-8")
    declarations = "\n".join(
        line
        for line in constants.splitlines()
        if line.startswith(("constexpr uint32_t L1_TILE_N =", "constexpr uint32_t L1_TILE_M_256 ="))
    )
    return fixture.replace("/* TILE_CONSTANTS */", declarations)


@pytest.fixture(scope="module")
def combine_owner_binary(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Combine ownership compilation requires g++ or clang++")
    directory = tmp_path_factory.mktemp("combine-owner")
    source = directory / "combine_owner.cpp"
    source.write_text(combine_owner_source(), encoding="utf-8")
    binary = directory / "combine_owner"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{DIRECTORY.parents[2] / 'csrc'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("scenario", ["coverage", "readiness", "dispatch"])
def test_combine_owner_contract(combine_owner_binary, scenario):
    result = subprocess.run([str(combine_owner_binary), scenario], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
