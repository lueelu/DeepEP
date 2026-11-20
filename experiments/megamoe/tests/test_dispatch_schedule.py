# Copyright (c) 2026, Lu Lu

"""Check production Dispatch continuations, ownership, and ready publication."""

from pathlib import Path
import shutil
import subprocess

import pytest

from test_combine_owner import function_body


def test_dispatch14_preserves_rows_and_ready_publication(tmp_path):
    directory = Path(__file__).parent
    kernel = directory.parent / "csrc_ep32/kernels/mega_moe"
    wave = (kernel / "mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    dispatch = (kernel / "stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    utils = (kernel / "common/mega_moe_utils.hpp").read_text(encoding="utf-8")
    source = (directory / "dispatch_schedule_test.cpp").read_text(encoding="utf-8")
    for marker, text, name in [
        ("REVERSE_START", wave, "uint32_t ReverseWaveStart("),
        ("WAVE_GROUPS", wave, "uint32_t WaveGroups("),
        ("DISPATCH_STEP", wave, "::DispatchNextExpertSlice("),
        ("DISPATCH_RANGE", dispatch, "void DispatchTokenRange("),
        ("DISPATCH_OWNED", dispatch, "void DispatchOwnedExpertRows("),
        ("PUBLISH_READY", dispatch, "void PublishGmm1TileReady("),
        ("BALANCED_RANGE", utils, "WorkRange GetBalancedTokenRange("),
        ("ROTATED_RANGE", utils, "WorkRange GetRotatedBalancedTokenRange("),
        ("DISPATCH_SUBSET", utils, "uint32_t GetDispatchSubsetJobIndex("),
    ]:
        source = source.replace(f"/* {marker} */", function_body(text, name)[1:-1])
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Dispatch scheduling requires g++ or clang++")
    cpp = tmp_path / "dispatch_schedule.cpp"
    cpp.write_text(source, encoding="utf-8")
    binary = tmp_path / "dispatch_schedule"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
