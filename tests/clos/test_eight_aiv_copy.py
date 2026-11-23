"""CPU execution of the transplanted copy, slot mapping and cached scheduler.

The DMA/sender mocks check data ownership and addressing, not device ordering.
Run the fused-notify device precision suite as well when an NPU is available.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corrupt_cached_retention", [False, True])
def test_eight_aiv_production_bodies(tmp_path, corrupt_cached_retention):
    source = ROOT / "csrc/direct"
    functions = cpp_functions(
        (source / "kernels/combine/udma_combine_self_copy.h").read_text(), "CopySelfField", "CopyCompactSelf"
    )
    functions += cpp_functions((source / "kernels/combine/combine_udma.cpp").read_text(), "ContributorSlot")
    schedule = (source / "kernels/dispatch/udma_dispatch_schedule.h").read_text()
    if corrupt_cached_retention:
        marker = schedule.index("// A short batch may still reference this slot")
        call = schedule.index("sender.PreserveIndices();", marker)
        schedule = schedule[:call] + schedule[call:].replace("sender.PreserveIndices();", "", 1)
    begin = schedule.index("template <class Sender, class Trace>")
    functions += schedule[begin : schedule.index("// Each producer", begin)]
    functions += cpp_functions(schedule, "ReuseDispatchIndexWindow", "DispatchIndexLowerBound", "DispatchRemoteImpl")
    functions += cpp_functions((source / "kernels/combine/udma_combine_schedule.h").read_text(), "CombineRemoteImpl")
    functions += cpp_functions(
        (source / "kernels/combine/udma_token_combine_schedule.h").read_text(),
        "GatherTokenIndices",
        "CombineTokenRemote",
    )
    host = (source / "ops/combine.cpp").read_text()
    begin = host.index("    tiling.compact_self_rows =")
    gate = host[begin : host.index(";", begin) + 1]
    harness = (Path(__file__).with_name("eight_aiv_copy_oracle.cpp")).read_text()
    harness = harness.replace("using GM_ADDR = uint8_t*;", "#define GM_ADDR uint8_t*")
    trace = (source / "kernels/dispatch/dispatch_device_trace.h").read_text()
    trace = "\n".join(line for line in trace.splitlines() if not line.startswith(("#include", "#pragma")))
    harness = harness.replace("// PRODUCTION_TRACE", trace)
    harness = harness.replace("// PRODUCTION_FUNCTIONS", functions).replace("// PRODUCTION_GATE", gate)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for production-body CPU oracle")
    path = tmp_path / "oracle.cpp"
    path.write_text(harness)
    executable = tmp_path / ("oracle.exe" if os.name == "nt" else "oracle")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(source), str(path), "-o", str(executable)], check=True, timeout=180
    )
    try:
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked oracle execution (4551)")
        raise
    if corrupt_cached_retention:
        assert result.returncode != 0 and "sent == ordered" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
