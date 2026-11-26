# Copyright (c) 2026, Lu Lu
# Modified by xixing, guofanfeng 2026

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fast_path", [0, 1])
def test_peer_row_mapping_round_trip(tmp_path, fast_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler is required")
    root = Path(__file__).resolve().parents[4]
    output = tmp_path / "peer_rows"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-DMEGAMOE_PREFILL_PEER_FAST_PATH={fast_path}",
            f"-iquote{root}",
            f"-iquote{root / 'experiments/megamoe/csrc/kernels/mega_moe'}",
            str(Path(__file__).with_name("megamoe_peer_rows_test.cpp")),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = subprocess.run([str(output)], check=True, capture_output=True, text=True, timeout=90)
    assert "PASS peer rows:" in result.stdout
