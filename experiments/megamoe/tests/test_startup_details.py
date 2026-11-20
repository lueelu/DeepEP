# Copyright (c) 2026, Lu Lu

"""Compile production snapshot acquisition, keeping diagnostics observable."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_count_snapshot_profile_preserves_acquisition_and_last_failed_sources(tmp_path, timer):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Snapshot compilation requires g++ or clang++")
    directory = Path(__file__).parent
    kernels = directory.parent / "csrc_ep32/kernels/mega_moe"
    production = (kernels / "stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    declaration = production.index("__aicore__ inline void LoadMteExpertCounts(")
    start = production.rfind("template <", 0, declaration)
    opening = production.index("{", declaration)
    depth, end = 1, opening + 1
    while depth:
        depth += (production[end] == "{") - (production[end] == "}")
        end += 1
    fixture = (directory / "count_snapshot_test.cpp").read_text(encoding="utf-8")
    source = tmp_path / "snapshot.cpp"
    source.write_text(fixture.replace("/* PRODUCTION_COUNT_SNAPSHOT */", production[start:end]), encoding="utf-8")
    binary = tmp_path / "snapshot"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{kernels / 'common'}"]
    if timer:
        command.append("-DDEEPEP_MEGAMOE_TIMER=1")
    result = subprocess.run(command + [str(source), "-o", str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
