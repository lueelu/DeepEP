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
    write_samples(tmp_path / "two/kernel_details_1.csv", [kernel(200, 210), kernel(50, 100000, "Exp")])
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


def test_aligned_batch_only_final_round_has_timer():
    log, prefix, timer = [], object(), object()
    torch = SimpleNamespace(exp=lambda data, out: log.append(("exp", data, out)))
    enqueue_batch(lambda t: log.append(("call", t)), torch, prefix, dict(repeats=2), 3, timer)
    assert log == [("exp", prefix, prefix)] * 2 + [("call", None)] * 2 + [("call", timer)]


def test_profiler_barrier_after_enter_no_per_round_synchronize(tmp_path, monkeypatch):
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
