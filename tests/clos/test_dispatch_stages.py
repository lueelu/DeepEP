"""Validate diagnostic accounting independently of device clocks/profiler runtime."""

import csv
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/clos"))
try:
    spec = importlib.util.spec_from_file_location("dispatch_stages", ROOT / "benchmarks/clos/bench_dispatch_stages.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("drop_peer_trace", [False, True])
def test_dispatch_calls_compile_against_actual_helper_signatures(tmp_path, drop_peer_trace):
    """Compile both routing branches; no NPU SDK or executable launch required."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for dispatch call-signature regression")
    kernels = ROOT / "csrc/direct/kernels/dispatch"
    declarations = []
    for filename, name in (
        ("udma_dispatch_metadata.h", "DispatchMetadata"),
        ("udma_dispatch_schedule.h", "DispatchRemote"),
        ("udma_token_dispatch_schedule.h", "DispatchTokenRemote"),
    ):
        definition = cpp_functions((kernels / filename).read_text(), name)
        declarations.append(definition.split("{", 1)[0] + " {}")
    source = (kernels / "dispatch_udma.cpp").read_text()
    start = source.index("    ascend_deepep::udma::DispatchMetadata(")
    calls = source[start : source.index("    const int32_t actual_recv_tokens", start)]
    if drop_peer_trace:
        assert "tiling, core_idx, core_num, trace);" in calls
        calls = calls.replace("tiling, core_idx, core_num, trace);", "tiling, core_idx, core_num);")
    program = r"""
#include <cstdint>
#define __aicore__
using GM_ADDR = uint8_t*;
namespace ascend_deepep::udma {
struct DispatchTilingData { bool udma_token_schedule; };
// DECLARATIONS
}
struct EnabledTrace { enum { MetadataDone }; void Mark(int) {} };
struct DisabledTrace { enum { MetadataDone }; void Mark(int) {} };
template<class Trace> void check(bool token) {
    int resource = 0;
    GM_ADDR x{}, dispatch_scales{}, topk_idx{}, topk_weights{}, dispatch_plan{}, rank_prefix_matrix{},
        channel_prefix_matrix{}, symmetric_buffer{}, send_head{}, recv_x{}, recv_x_scales{};
    ascend_deepep::udma::DispatchTilingData tiling{token};
    uint32_t core_idx = 0, core_num = 8, kDispatchStageBytes = 65536;
    Trace trace;
    // CALLS
}
template void check<EnabledTrace>(bool);
template void check<DisabledTrace>(bool);
"""
    path = tmp_path / "dispatch_calls.cpp"
    path.write_text(program.replace("// DECLARATIONS", "\n".join(declarations)).replace("// CALLS", calls))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "dispatch_calls.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if drop_peer_trace:
        assert result.returncode != 0
        assert "DispatchRemote" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def fixture_trace():
    fields = [
        "version",
        "rank",
        "core",
        "cached",
        "whole_channel",
        "entry_cycles",
        *bench.TIMELINE,
        "src_cycles",
        "topk_cycles",
        "self_cycles",
        "list_cycles",
    ]
    rows = []
    for core in range(8):
        rows.append([1, 3, core, 1, 1, 123456, *range(1000, 12000, 1000), 1500, 2500, 3000, 0])
    return fields, rows


def test_trace_intervals_do_not_add_overlapping_metadata_twice():
    fields, rows = fixture_trace()
    decoded = bench.decode_trace(rows, fields, 3, 8, True, True, 1000)
    for row in decoded:
        intervals = row["intervals"]
        assert all(intervals[name + "_interval_us"] == 1 for name in bench.TIMELINE)
        assert intervals["src_cycles_us"] == 1.5
        assert intervals["topk_cycles_us"] == 2.5
        assert sum(intervals[name + "_interval_us"] for name in bench.TIMELINE) == 11


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 0),
        ("rank", 0),
        ("core", 7),
        ("cached", 0),
        ("whole_channel", 0),
        ("entry_cycles", 0),
        ("quiet_done", 1),
        ("init_done", -1),
    ],
)
def test_reject_wrong_launch_or_incomplete_trace(field, value):
    fields, rows = fixture_trace()
    rows[0][fields.index(field)] = value
    with pytest.raises(ValueError):
        bench.decode_trace(rows, fields, 3, 8, True, True, 1000)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_capture_excludes_alignment_and_guards(tmp_path, cached, corrupt):
    names = ["cached_notify_combine_kernel"]
    if not cached:
        names.append("dispatch_layout_kernel")
    names += ["cached_notify_dispatch_kernel" if cached else "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    rows = []
    for sample in range(7):  # 2 guards + 3 samples + 2 guards
        for name in names:
            rows.append({"Name": name, "Start Time(us)": len(rows) * 10, "Duration(us)": sample + 1})
    if corrupt:
        rows.pop(2)
    with (tmp_path / "kernel_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if corrupt:
        with pytest.raises(RuntimeError):
            bench.capture_stages(tmp_path, 3, cached)
    else:
        result = bench.capture_stages(tmp_path, 3, cached)
        assert "cached_notify_combine_kernel" not in result
        assert result["v1_dispatch_udma_kernel"] == [3, 4, 5]
