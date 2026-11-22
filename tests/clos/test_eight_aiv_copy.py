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


def test_eight_aiv_production_bodies(tmp_path):
    source = ROOT / "csrc/direct"
    functions = cpp_functions(
        (source / "kernels/combine/udma_combine_self_copy.h").read_text(), "CopySelfField", "CopyCompactSelf"
    )
    functions += cpp_functions((source / "kernels/combine/combine_udma.cpp").read_text(), "ContributorSlot")
    functions += cpp_functions((source / "kernels/dispatch/udma_dispatch_schedule.h").read_text(), "DispatchRemoteImpl")
    host = (source / "ops/combine.cpp").read_text()
    begin = host.index("    tiling.compact_self_rows =")
    gate = host[begin : host.index(";", begin) + 1]
    harness = (Path(__file__).with_name("eight_aiv_copy_oracle.cpp")).read_text()
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
        subprocess.run([str(executable)], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked oracle execution (4551)")
        raise
