"""Profiling accounting for fused and separate layout/notify kernels."""

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fused_bench", ROOT / "benchmarks/clos/bench_fused_notify.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def host_fields():
    return [
        "api_entry",
        "native_entry",
        "notify_submit_begin",
        "notify_submitted",
        "count_wait_begin",
        "total_observed",
        "counts_observed",
        "output_ready",
        "comm_submit_begin",
        "comm_submitted",
        "native_return",
        "api_return",
        "complete",
        "fused",
        "host_prep",
        "deferred_plan",
        "mapped_counts",
    ]


@pytest.mark.parametrize("mode", list(bench.MODES))
def test_host_intervals_are_same_call_and_native_flags_are_checked(mode):
    fields = host_fields()
    row = [1000 * (i + 1) for i in range(13)] + list(bench.MODES[mode]) + [1]
    result = bench.host_durations([row], fields, 1, mode)
    assert result["api_to_notify_submit"] == [3.0]
    assert result["post_launch_prepare"] == [1.0]
    assert result["wait_total"] == [1.0]
    assert result["wait_remaining_counts"] == [1.0]
    assert result["comm_submit_to_complete"] == [3.0]
    assert result["api_complete_diagnostic"] == [12.0]
    with pytest.raises(RuntimeError, match="expected"):
        bench.host_durations([row], fields, 2, mode)
    with pytest.raises(RuntimeError, match="schema"):
        bench.host_durations([row[:-1]], fields, 1, mode)
    wrong = row.copy()
    wrong[13] = 1 - wrong[13]
    with pytest.raises(RuntimeError, match="flags"):
        bench.host_durations([wrong], fields, 1, mode)
    for timestamp in (0, 999999):
        wrong = row.copy()
        wrong[5] = timestamp
        with pytest.raises(RuntimeError, match="timestamp"):
            bench.host_durations([wrong], fields, 1, mode)


@pytest.mark.parametrize("fused,simt", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("guards", [0, 2])
def test_profile_sequence_excludes_outer_barrier(tmp_path, fused, simt, expanded, guards):
    names = ["cached_notify_combine_kernel"]
    if not fused:
        names.append("dispatch_layout_simt_kernel" if simt else "dispatch_layout_kernel")
    names += ["notify_dispatch_kernel", "direct_dispatch_udma_kernel"]
    if expanded:
        names.append("expand_kernel")
    path = tmp_path / "kernel_details.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Start Time(us)", "Duration(us)"])
        writer.writeheader()
        for i, name in enumerate(names * (3 + 2 * guards)):
            measured = guards <= i // len(names) < guards + 3
            writer.writerow(
                {
                    "Name": name,
                    "Start Time(us)": i * 1000,
                    "Duration(us)": 900 if "combine" in name else (10 if measured else 777),
                }
            )
    result = bench.parse_capture(tmp_path, fused, 3, expanded, guards, simt_layout=simt)
    assert result["notify_dispatch_kernel"] == [10] * 3
    assert ("dispatch_layout_kernel" in result) != fused
    with pytest.raises(RuntimeError, match="sequence"):
        bench.parse_capture(tmp_path, fused, 2, expanded, guards, simt_layout=simt)
    if simt:
        with pytest.raises(RuntimeError, match="sequence"):
            bench.parse_capture(tmp_path, False, 3, expanded, guards)
    if fused:
        with pytest.raises(RuntimeError, match="sequence"):
            bench.parse_capture(tmp_path, False, 3, expanded, guards)
