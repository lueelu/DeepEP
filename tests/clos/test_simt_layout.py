"""V2 layout routing and production DMA/VF CPU oracle; no NPU performance claims."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bits", [32, 64])
def test_production_simt_layout_dma_and_counts(tmp_path, bits):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    executable = tmp_path / ("layout.exe" if os.name == "nt" else "layout")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-DEP_NUM_TOPK_IDX_BITS={bits}",
            "-I",
            str(ROOT / "csrc/direct"),
            str(Path(__file__).with_name("simt_layout_oracle.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    try:
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked oracle execution (4551)")
        raise
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SIMT layout CPU oracle passed" in result.stdout
