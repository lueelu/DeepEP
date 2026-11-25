"""Execute production dense-weight addresses/descriptors with a scatter oracle."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corrupt_prefix", [False, True])
def test_compact_weight_production_mapping(tmp_path, corrupt_prefix):
    source = ROOT / "csrc/direct"
    mapping = cpp_functions((source / "kernels/combine/combine_udma.cpp").read_text(), "CompactWeightSlot")
    if corrupt_prefix:
        mapping = mapping.replace("channel - 1U", "channel")
    sender = cpp_functions((source / "kernels/combine/udma_combine_sender.h").read_text(), "PutCompactWeights")
    copy = cpp_functions(
        (source / "kernels/combine/udma_combine_self_copy.h").read_text(), "CopySelfField", "CopyCompactSelf"
    )
    harness = Path(__file__).with_name("compact_combine_weights_oracle.cpp").read_text()
    harness = harness.replace("// PRODUCTION_MAPPING", mapping).replace("// PRODUCTION_SENDER", sender)
    harness = harness.replace("// PRODUCTION_COPY", copy)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for production-body CPU oracle")
    path = tmp_path / "compact.cpp"
    path.write_text(harness)
    executable = tmp_path / ("compact.exe" if os.name == "nt" else "compact")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(source), str(path), "-o", str(executable)],
        check=True,
        timeout=180,
    )
    try:
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise
    if corrupt_prefix:
        assert result.returncode != 0, "oracle must reject the wrong channel prefix"
    else:
        assert result.returncode == 0, result.stderr
