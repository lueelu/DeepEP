# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the production lifecycle against link-time fault-injection doubles."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def lifecycle_binary(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None or not (root / "csrc").is_dir():
        pytest.skip("Host lifecycle compilation requires Linux, g++ and source tree")
    binary = tmp_path_factory.mktemp("native-lifecycle") / "lifecycle"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-I" + str(root / "tests/runtime/stubs"),
            "-I" + str(root / "csrc"),
            str(root / "csrc/runtime/session.cpp"),
            str(root / "tests/runtime/session_test.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "invalid",
        "init-fault",
        "allocation-fault",
        "memset-fault",
        "foreign",
        "abandoned",
        "close-fault",
        "normal",
        "udma",
        "megamoe",
        "low-latency",
        "ep16-small-workspace",
    ],
)
def test_native_lifecycle_faults(lifecycle_binary, case):
    result = subprocess.run([lifecycle_binary, case], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
