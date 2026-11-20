# Copyright (c) 2026, Lu Lu

"""Run the production AIV1 scheduler against deterministic CPU-ready events."""

from pathlib import Path
import shutil
import subprocess

import pytest


def aiv1_priority_source(production_source=None):
    directory = Path(__file__).parent
    if production_source is None:
        production_source = (directory.parent / "csrc_ep32/kernels/mega_moe/mega_moe_wave_a8w4.hpp").read_text(
            encoding="utf-8"
        )
    start = production_source.index("::ProcessAiv1PriorityStages(")
    opening = production_source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (production_source[end] == "{") - (production_source[end] == "}")
        end += 1
    # Keep the actual scheduler control flow; the C++ fixture only supplies
    # ready/not-ready stage operations and observes their externally visible order.
    fixture = (directory / "aiv1_priority_test.cpp").read_text(encoding="utf-8")
    return fixture.replace("/* PRODUCTION_SCHEDULER_BODY */", production_source[opening:end])


def test_aiv1_preserves_activation_priority_and_cross_expert_ring_lifetime(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("AIV1 scheduler compilation requires g++ or clang++")
    source = tmp_path / "aiv1_priority.cpp"
    source.write_text(aiv1_priority_source(), encoding="utf-8")
    binary = tmp_path / "aiv1_priority"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
