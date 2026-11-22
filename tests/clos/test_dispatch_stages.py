"""Validate diagnostic accounting independently of device clocks/profiler runtime."""

import csv
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/clos"))
try:
    spec = importlib.util.spec_from_file_location("dispatch_stages", ROOT / "benchmarks/clos/bench_dispatch_stages.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
finally:
    sys.path.pop(0)


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
