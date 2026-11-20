# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.benchmark import load_config
from deep_ep_experimental.megamoe.performance import (
    alignment_config,
    collect_profile,
    enqueue_batch,
    read_kernel_samples,
    summarize_rounds,
    SLEEP_CYCLES,
)


def write_samples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "Stream ID", "Device_id"])
        writer.writerows(rows)


def kernel(start, duration, name="mega_moe_kernel<true>"):
    return [name, "AI_MIX_AIC", start, duration, 3, 0]


def test_samples_merge_shards_sort_and_keep_slow_round(tmp_path):
    write_samples(tmp_path / "one/kernel_details.csv", [kernel(300, 10000), kernel(100, 200)])
    write_samples(
        tmp_path / "two/kernel_details_1.csv",
        [
            kernel(200, 210),
            kernel(50, 100000, "Exp"),
            kernel(90, 50000, "HcclAllReduce"),
            kernel(91, 20000, "benchmark_sleep_kernel"),
            kernel(92, 300, "benchmark_roce_barrier_kernel"),
            kernel(93, 100, "barrier_on_stream_kernel"),
        ],
    )
    samples = read_kernel_samples(tmp_path, 3)
    assert [s["duration_us"] for s in samples] == [200, 210, 10000]
    assert samples[-1]["row"] == 2 and "one" in samples[-1]["file"]


@pytest.mark.parametrize(
    "rows,iterations,message",
    [
        ([], 1, "expected 1"),
        ([kernel(1, 2)], 2, "expected 2"),
        ([kernel(1, 2), kernel(1, 2)], 2, "Duplicate"),
        ([kernel(1, 0)], 1, "Nonfinite"),
        ([kernel(1, "nan")], 1, "Nonfinite"),
        ([kernel("bad", 2)], 1, "Invalid"),
    ],
)
def test_bad_profiler_export_is_never_a_success(tmp_path, rows, iterations, message):
    write_samples(tmp_path / "kernel_details.csv", rows)
    with pytest.raises(ValueError, match=message):
        read_kernel_samples(tmp_path, iterations)


def test_read_only_profile_requires_one_sum_before_each_kernel(tmp_path):
    directory = tmp_path / "profile"
    csv_path = directory / "kernel_details.csv"
    write_samples(csv_path, [kernel(1, 2, "ReduceSum"), kernel(4, 2), kernel(7, 2, "ReduceSum"), kernel(10, 2)])
    assert len(read_kernel_samples(directory, 2, require_read_sum=True)) == 2
    write_samples(csv_path, [kernel(1, 2, "ReduceSum"), kernel(4, 2, "ReduceSum"), kernel(7, 2), kernel(10, 2)])
    with pytest.raises(ValueError, match="alternating"):
        read_kernel_samples(directory, 2, require_read_sum=True)


def test_round_policy_node_scope_and_no_device_timestamp_subtraction():
    per_rank = [[1000, 900, 3, 9], [1100, 800, 8, 4], [1200, 700, 20, 2], [1300, 600, 12, 15]]
    report = summarize_rounds(per_rank, 2)
    assert report["mean_us"] == 2.5  # min per round, NOT min of rank means
    assert report["rank_max_mean_us"] == 17.5
    assert [node["mean_us"] for node in report["per_node"]] == [3.5, 7]
    assert report["metric"] == "megamoe_kernel_duration_us"
    assert report["per_rank_rounds_us"] == [[3, 9], [8, 4], [20, 2], [12, 15]]
    assert report["recorded_rounds"] == 4 and report["rounds"] == 2
    assert per_rank[0] == [1000, 900, 3, 9]  # Selection does not mutate the raw input.
    with pytest.raises(ValueError, match="divide"):
        summarize_rounds(per_rank, 3)


@pytest.mark.parametrize(
    "iterations,first,selected,mean",
    [
        (1, 0, 1, 1),
        (2, 1, 1, 2),
        (3, 1, 2, 2.5),
        (5, 2, 3, 4),
        (20, 10, 10, 15.5),
        (40, 20, 20, 30.5),
    ],
)
def test_tail_half_selection_even_odd_and_single_round(iterations, first, selected, mean):
    report = summarize_rounds([list(range(1, iterations + 1)), list(range(2, 2 * iterations + 1, 2))])
    assert report["iteration_selection"] == "tail_half_ceil"
    assert report["recorded_rounds"] == iterations
    assert report["first_selected_iteration"] == first
    assert report["rounds"] == selected
    assert report["mean_us"] == mean
    assert report["rank_max_mean_us"] == 2 * mean
    assert report["rank_min_rounds_us"][0] == first + 1
    assert report["rank_min_rounds_us"][-1] == iterations


def test_exp_once_per_batch_continuous_launches_and_only_final_round_has_timer():
    log, prefix, timer = [], object(), object()
    torch = SimpleNamespace(exp=lambda data, out: log.append(("exp", data, out)))
    enqueue_batch(
        lambda t: log.append(("call", t)),
        torch,
        prefix,
        dict(repeats=2),
        3,
        timer,
    )
    assert log == [("exp", prefix, prefix)] * 2 + [
        ("call", None),
        ("call", None),
        ("call", timer),
    ]


def test_profiler_host_barrier_before_batch_without_per_round_synchronize(tmp_path, monkeypatch):
    log = []

    class Profile:
        def __enter__(self):
            log.append("enter")
            return self

        def step(self):
            log.append("step")

        def __exit__(self, *args):
            log.append("export")
            write_samples(tmp_path / "kernel_details.csv", [kernel(10, 2), kernel(20, 3)])

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw,
        ExportType=SimpleNamespace(Text="text"),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipes"),
        ProfilerLevel=SimpleNamespace(Level1=1),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=lambda **kw: Profile(),
        schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: log.append("sync")), exp=lambda *a, **kw: log.append("exp")
    )
    group = SimpleNamespace(barrier=lambda stage: log.append(stage))
    samples = collect_profile(
        lambda t: log.append(("call", t)),
        group,
        torch,
        dict(alignment=dict(repeats=2), iters=2),
        object(),
        tmp_path,
        96,
    )
    assert log == [
        "enter",
        "sync",
        "profiler ready BS=96",
        "exp",
        "exp",
        ("call", None),
        ("call", None),
        "sync",
        "step",
        "export",
    ]
    assert [s["duration_us"] for s in samples] == [2, 3]


def test_alignment_is_explicit_validated_and_recorded(tmp_path):
    config = load_config("deepseek_v4_pro")
    assert config["alignment"] == dict(rows=100000, cols=30000, repeats=20)
    for value in ({}, dict(rows=1, cols=2, repeats=0), dict(rows=True, cols=2, repeats=1)):
        with pytest.raises(ValueError, match="alignment"):
            alignment_config(dict(alignment=value))
    config["alignment"] = dict(rows=1000, cols=2000, repeats=10)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["alignment"] == config["alignment"]


def test_shmem_rounds_flush_sleep_barrier_then_kernel_without_exp_or_host_waits():
    log, timer = [], object()
    enqueue_batch(
        lambda value: log.append(("call", value)),
        SimpleNamespace(),
        None,
        dict(repeats=20),
        3,
        timer,
        method="shmem",
        flush_buffer=SimpleNamespace(zero_=lambda: log.append("flush")),
        sleep=lambda cycles: log.append(("sleep", cycles)),
        barrier=lambda: log.append("shmem"),
    )
    assert log == sum(
        (["flush", ("sleep", SLEEP_CYCLES), "shmem", ("call", value)] for value in (None, None, timer)), []
    )


@pytest.mark.parametrize("method", ["shmem", "unknown"])
def test_invalid_method_or_missing_shmem_hooks_fail_before_launch(method):
    with pytest.raises(ValueError):
        enqueue_batch(lambda _: pytest.fail("invalid batch launched"), SimpleNamespace(), None, {}, 2, method=method)


def test_read_sum_can_use_a_different_serial_stream(tmp_path):
    path = tmp_path / "kernel_details.csv"
    reduce = kernel(1, 2, "ReduceSum")
    reduce[4] = 1
    write_samples(path, [reduce, kernel(3, 3)])
    assert len(read_kernel_samples(tmp_path, 1, require_read_sum=True)) == 1
    write_samples(path, [reduce, kernel(2, 3)])
    with pytest.raises(ValueError, match="serial"):
        read_kernel_samples(tmp_path, 1, require_read_sum=True)
