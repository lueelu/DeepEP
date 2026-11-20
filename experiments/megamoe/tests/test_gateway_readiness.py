# Copyright (c) 2026, Lu Lu

"""Compile the production gateway helper against concurrent flag primitives.

These tests cover batched DMA, epoch/lifetime rules and consumer ordering,
not device cache coherence or hardware DMA visibility.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def gateway_readiness_binaries(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Gateway readiness compilation requires g++ or clang++")
    directory = Path(__file__).parent
    output = tmp_path_factory.mktemp("gateway-readiness")
    binaries = {}
    for engine in ("aiv", "aic"):
        binary = output / f"gateway_readiness_{engine}"
        command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread"]
        if engine == "aic":
            command.append("-DGATEWAY_TEST_AIC=1")
        command += [str(directory / "gateway_readiness_test.cpp"), "-o", str(binary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        binaries[engine] = binary
    return binaries


@pytest.mark.parametrize(
    "scenario",
    ["distribution", "late-arrival", "epochs", "late-consumer", "guards"],
)
def test_gateway_readiness_protocol(gateway_readiness_binaries, scenario):
    result = subprocess.run(
        [str(gateway_readiness_binaries["aiv"]), scenario],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_gateway_readiness_aic_is_inert(gateway_readiness_binaries):
    result = subprocess.run(
        [str(gateway_readiness_binaries["aic"]), "guards"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
