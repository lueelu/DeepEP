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


def test_slowest_rank_can_change_each_iteration():
    result = benchmark.rank_max_summary([[2, 8, 3], [7, 1, 9]])
    assert result == {"mean_us": 8, "min_us": 7, "p50_us": 8, "max_us": 9, "samples": 3}


def test_temporal_report_uses_chronological_global_maxima():
    # The worst rank changes. The early peak must remain in the full result,
    # while the last 50 are selected by iteration, never by latency or rank.
    gathered = [[20] * 50 + [30, 10] * 25, [10] * 50 + [10, 40] * 25]
    gathered[1][2] = 100
    report = benchmark.rank_max_report(gathered)
    summary = report["summary"]
    assert summary["mean_us"] == 28.3 and summary["max_us"] == 100
    assert summary["max_iteration"] == 3 and summary["max_rank"] == 1
    assert summary["first50_mean_us"] == 21.6
    assert summary["last50_start_iteration"] == 51 and summary["last50_samples"] == 50
    assert summary["last50_mean_us"] == summary["last50_p50_us"] == 35
    assert summary["last50_min_us"] == 30 and summary["last50_max_us"] == 40
    assert report["iterations"][50:52] == [
        {"iteration": 51, "rank": 0, "us": 30},
        {"iteration": 52, "rank": 1, "us": 40},
    ]
    assert report["slowest5"][0] == {"iteration": 3, "rank": 1, "us": 100}
    assert report["blocks10"][0] == {"first_iteration": 1, "last_iteration": 10, "mean_us": 28}
    assert report["blocks10"][-1] == {"first_iteration": 91, "last_iteration": 100, "mean_us": 35}


def test_late_peak_is_not_removed_and_ties_keep_first_rank_and_iteration():
    report = benchmark.rank_max_report([[5] * 100 + [80], [5] * 100 + [80]])
    summary = report["summary"]
    assert summary["max_iteration"] == 101 and summary["max_rank"] == 0
    assert summary["last50_start_iteration"] == 52 and summary["last50_max_us"] == 80
    assert report["slowest5"][1] == {"iteration": 1, "rank": 0, "us": 5}
    assert report["blocks10"][-1] == {"first_iteration": 101, "last_iteration": 101, "mean_us": 80}


def test_short_run_reports_actual_window_length():
    summary = benchmark.rank_max_report([[4]])["summary"]
    assert summary["samples"] == summary["last50_samples"] == summary["last50_start_iteration"] == 1
    assert summary["mean_us"] == summary["first50_mean_us"] == summary["last50_mean_us"] == 4


@pytest.mark.parametrize("samples", [[], [[]], [[1], [1, 2]], [[float("nan")]], [[0]]])
def test_reject_invalid_or_missing_samples(samples):
    with pytest.raises(ValueError):
        benchmark.rank_max_summary(samples)


def test_combine_kernel_selection_and_exact_sample_count(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPEP_LL_BACKEND", "rdma")
    assert reference.kernel_name("combine") == "ubx_combine_route_plan"
    assert reference.kernel_name("dispatch") == "dispatch_server_dedup_kernel"
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us),Start Time(us)\n"
        "ubx_combine_route_plan,3,2\nHcomAllReduce,90,1\n"
        "combine_server_dedup_kernel,88,1\nubx_combine_route_plan,4,3\n"
    )
    assert reference.kernel_durations(tmp_path, reference.kernel_name("combine"), 2) == [3, 4]
    with pytest.raises(ValueError, match="expected 3"):
        reference.kernel_durations(tmp_path, reference.kernel_name("combine"), 3)
    monkeypatch.setenv("DEEPEP_LL_BACKEND", "mte")
    assert reference.kernel_name("combine") == "combine_server_dedup_kernel"


def test_help_does_not_initialize_torch_or_npu():
    result = subprocess.run([sys.executable, benchmark.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--tokens" in result.stdout and "--iters" in result.stdout
    assert "--launch-gate" in result.stdout


def test_gate_enabled_by_default_and_explicit_ungated_comparison():
    assert benchmark.parser().parse_args(["--output", "unused"]).launch_gate == "notify"
    assert benchmark.parser().parse_args(["--output", "unused", "--launch-gate", "none"]).launch_gate == "none"


def test_clean_event_default_and_explicit_profiler():
    assert benchmark.parser().parse_args(["--output", "unused"]).timer == "acl-event"
    assert benchmark.parser().parse_args(["--output", "unused", "--timer", "profiler"]).timer == "profiler"


def test_default_sampling_window_and_short_run_override():
    args = benchmark.parser().parse_args(["--output", "unused"])
    assert args.warmup == 10 and args.iters == 100
    assert benchmark.parser().parse_args(["--output", "unused", "--iters", "50"]).iters == 50


@pytest.mark.parametrize("warmup", [0, 2])
def test_event_sampling_retains_all_measured_iterations_after_completion(warmup):
    from types import SimpleNamespace

    operations = []
    counter = iter(range(warmup + 3))

    def elapsed():
        assert operations[-2:] == ["invoke", "complete"]
        operations.append("elapsed")
        return next(counter) + 1

    samples = benchmark.event_samples(
        lambda: operations.append("invoke"),
        lambda: operations.append("complete"),
        SimpleNamespace(elapsed_us=elapsed),
        warmup,
        3,
    )
    assert samples == [warmup + 1, warmup + 2, warmup + 3]
    assert operations == ["invoke", "complete", "elapsed"] * (warmup + 3)


def test_route_digest_detects_slot_and_rank_changes():
    routes = [[[1, 2]], [[3, 4]]]
    assert benchmark.route_digest(routes) == benchmark.route_digest([[[1, 2]], [[3, 4]]])
    assert benchmark.route_digest(routes) != benchmark.route_digest([[[2, 1]], [[3, 4]]])
    assert benchmark.route_digest(routes) != benchmark.route_digest([[[3, 4]], [[1, 2]]])
