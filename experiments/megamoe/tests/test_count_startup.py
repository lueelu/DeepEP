# Copyright (c) 2026, Lu Lu

"""Execute production count preparation with deferred GM writes and payload readiness."""

from pathlib import Path
import shutil
import subprocess

import pytest


def count_startup_source():
    directory = Path(__file__).parent
    production = (directory.parent / "csrc_ep32/kernels/mega_moe/stage/mega_moe_token_dispatch.hpp").read_text(
        encoding="utf-8"
    )
    start = production.index("template <bool NeedCumsumReload = false, typename ActivationType>")
    opening = production.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (production[end] == "{") - (production[end] == "}")
        end += 1
    fixture = (directory / "count_startup_test.cpp").read_text(encoding="utf-8")
    return fixture.replace("/* PRODUCTION_COUNT_PREPARATION */", production[start:end])


@pytest.mark.parametrize("engine", ["aiv", "aic"])
def test_count_ready_precedes_payload_wait_after_gm_completion(tmp_path, engine):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Count preparation compilation requires g++ or clang++")
    source = tmp_path / "count_startup.cpp"
    source.write_text(count_startup_source(), encoding="utf-8")
    binary = tmp_path / "count_startup"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    if engine == "aic":
        command.append("-DCOUNT_TEST_AIC=1")
    result = subprocess.run(command + [str(source), "-o", str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
