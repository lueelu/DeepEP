# Copyright (c) 2026, Lu Lu

"""Compile the production layout and its budget against CANN arithmetic doubles."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def production_layout_binary(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Production layout compilation requires g++ or clang++")
    source = Path(__file__).with_name("peermem_layout_test.cpp")
    stubs = Path(__file__).with_name("layout_stubs")
    binary = tmp_path_factory.mktemp("peermem-layout") / "peermem_layout_test"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
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
    return binary


@pytest.mark.parametrize("mode", ["a8w4", "a8w8"])
def test_peermem_size_address_workspace_ready_and_legacy_compatibility(production_layout_binary, mode):
    result = subprocess.run([str(production_layout_binary), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
