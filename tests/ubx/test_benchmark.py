# Copyright (c) 2026, Lu Lu
"""CPU checks for the independent V2 Combine benchmark's measurement contract."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def load(path):
    spec = importlib.util.spec_from_file_location("benchmark_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load(Path(__file__).with_name("test_combine.py"))
reference = load(Path(__file__).resolve().parents[1] / "elastic/test_ep_low_latency.py")


def test_each_replay_averages_all_ranks_before_tail_selection():
    rows = [[1000] * 50 + [4, 12] * 25, [1000] * 50 + [8, 4] * 25]
    report = benchmark.rank_mean_report(rows, 50)
    assert report["iterations_us"][-4:] == [6, 8, 6, 8]
    assert report["mean_us"] == report["p50_us"] == 7
    assert report["samples"] == 50 and report["first_iteration"] == 51
    system = reference.summarize_kernel(rows, "combine_put_jit", 50)
    assert system["kernel_samples"] == 50 and system["mean_kernel_us"] == report["mean_us"]
    assert system["p50_kernel_us"] == report["p50_us"]


@pytest.mark.parametrize("rows", [[], [[]], [[1], [1, 2]], [[float("nan")]], [[0]]])
def test_invalid_samples(rows):
    with pytest.raises(ValueError):
        benchmark.rank_mean_report(rows, 50)


def test_short_run_and_defaults():
    assert benchmark.rank_mean_report([[4], [6]], 50)["mean_us"] == 5
    args = benchmark.parser().parse_args(["--output", "unused"])
    assert (args.warmup, args.iters, args.perf_tail, args.timer) == (5, 100, 50, "profiler")


def test_combine_profiler_ignores_clock_and_collective_tasks(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Name,Duration(us),Start Time(us)\n"
        "combine_put_jit,4,5\nclock_wait_table,90,1\nHcomAllReduce,20,0\n"
        "combine_put_jit,3,2\nclock_wait_table,90,4\n"
    )
    assert benchmark.graph_kernel_durations(reference, tmp_path, 2) == [3, 4]
    with pytest.raises(ValueError, match="one device wait"):
        benchmark.graph_kernel_durations(reference, tmp_path, 3)


def test_help_without_npu():
    result = subprocess.run([sys.executable, benchmark.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--clock-lead-us" in result.stdout and "--perf-tail" in result.stdout


def test_route_digest_detects_slot_and_rank_changes():
    routes = [[[1, 2]], [[3, 4]]]
    assert benchmark.route_digest(routes) == benchmark.route_digest([[[1, 2]], [[3, 4]]])
    assert benchmark.route_digest(routes) != benchmark.route_digest([[[2, 1]], [[3, 4]]])
    assert benchmark.route_digest(routes) != benchmark.route_digest([[[3, 4]], [[1, 2]]])


def test_put_route_preflight_distinguishes_capacity_and_actual_bs():
    # Source 31 and expert rank 2 wrap to region 2 with two local experts.
    entries = [[(31, 2), (31, 5)], []]
    physical, slot = 2 * 256 + 1, (31 * 8 + 5) * 4 + 1
    destinations = [[31, 2, 0], [physical, 3, slot]]
    benchmark.check_put_destinations(entries, [1, 3], destinations, 2, 32, 8, 256, 4)
    for broken in ([physical, -1, -1], [physical, 17, slot], [physical, 3, (31 * 256 + 5) * 4 + 1],
                   [2 * 8 + 1, 3, slot]):
        with pytest.raises(AssertionError, match="terminal PUT"):
            benchmark.check_put_destinations(entries, [1, 3], [destinations[0], broken], 2, 32, 8, 256, 4)


def test_clock_windows_cover_the_last_rank_before_first_release():
    benchmark.check_clock_windows([[(1, 4), (6, 10)], [(2, 3), (7, 9)]])
    with pytest.raises(ValueError, match="wait 2"):
        benchmark.check_clock_windows([[(1, 4), (6, 10)], [(2, 3), (11, 12)]])
