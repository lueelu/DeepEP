# Copyright (c) 2026, Lu Lu

"""DSV41 case selection, launch contract and all-rank aggregation without NPUs."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.performance import summarize_rounds

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("dsv41_perf", ROOT / "experiments/megamoe/scripts/dsv41_perf.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def options(**kwargs):
    return SimpleNamespace(
        **dict(
            dict(
                bs_values=list(launcher.BATCHES),
                input="fp8",
                routing="balanced",
                shared_experts=1,
                perf_method="legacy",
                warmup=20,
                iters=4,
                seed=20261029,
                ep_size=32,
                node_rank=1,
                master_addr="192.0.2.1",
                master_port=29654,
            ),
            **kwargs,
        )
    )


def test_plan_has_exact_48_cases_and_only_one_bs_multiplier(capsys):
    assert launcher.main(["plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["case_count"] == 48
    for item, ep, experts in zip(plan["plans"], (16, 32), (400, 416)):
        assert item["ep_size"] == ep
        assert item["original_bs"] == list(range(1, 25))
        assert item["config"]["bs_values"] == list(range(6, 145, 6))
        assert item["config"]["model"] == dict(
            hidden=5120,
            intermediate_hidden=2304,
            num_experts=experts,
            num_topk=6,
            num_shared_experts=1,
        )
    assert launcher.batches("144,6") == [6, 144]
    with pytest.raises(argparse.ArgumentTypeError):
        launcher.batches("24,24")


def test_launch_runs_installed_full_benchmark_in_one_collective_job(tmp_path):
    command = launcher.command(options(), tmp_path)
    assert command[:4] == [launcher.sys.executable, "-I", "-m", "torch.distributed.run"]
    for flag, expected in (
        ("--nnodes", "2"),
        ("--nproc-per-node", "16"),
        ("--node-rank", "1"),
        ("--module", "deep_ep_experimental.megamoe.benchmark"),
        ("--mode", "perf"),
        ("--max-restarts", "0"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--timer" not in command
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "measurements")


def sample_run(directory, ep=32, local_rank=16):
    config = launcher.specification(options(bs_values=[6, 144]), ep)
    directory.mkdir()
    launcher.write_json(directory / "launcher_node_1.json", {"specification": config})
    rank_dir = directory / f"measurements/rank_{local_rank}"
    rank_dir.mkdir(parents=True)
    launcher.write_json(rank_dir / "manifest.json", dict(config, mode="perf"))
    case = rank_dir / "bs_6"
    case.mkdir()
    # Fastest/slowest rank alternates: max(mean(rank)) = 7, mean(max(round)) = 10.
    values = [[100, 100, 10, 2], [100, 100, 2, 10]] + [[100, 100, 7, 7]] * (ep - 2)
    report = summarize_rounds(values, 16)
    report["perf_method"] = "legacy"
    path = case / "profiling.json"
    launcher.write_json(path, report)
    return path


def test_summary_uses_all_ranks_tail_samples_and_preserves_missing_cases(tmp_path):
    directory = tmp_path / "ep32"
    sample_run(directory)
    _, rows = launcher.collect(directory)
    assert rows[0]["max_rank_mean_us"] == 7
    assert rows[0]["per_round_max_mean_us"] == 10
    assert rows[0]["per_round_min_mean_us"] == 2
    assert rows[0]["recorded_rounds"] == 4 and rows[0]["selected_rounds"] == 2
    assert rows[1]["status"] == "missing" and rows[1]["max_rank_mean_us"] is None
    assert not launcher.summarize([directory], tmp_path / "summary")
    assert len(json.loads((tmp_path / "summary/summary.json").read_text(encoding="utf-8"))) == 2
    with pytest.raises(ValueError, match="Duplicate EP/BS"):
        launcher.summarize([directory, directory], tmp_path / "duplicate")


@pytest.mark.parametrize("failure", ["ranks", "rounds", "nan", "timer"])
def test_summary_rejects_incomplete_or_wrong_measurements(tmp_path, failure):
    directory = tmp_path / "run"
    path = sample_run(directory)
    report = json.loads(path.read_text())
    if failure == "ranks":
        report["per_rank_rounds_us"].pop()
    elif failure == "rounds":
        report["per_rank_rounds_us"][0].pop()
    elif failure == "nan":
        report["per_rank_rounds_us"][0][0] = float("nan")
    else:
        report["timer_pass"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    _, rows = launcher.collect(directory)
    assert rows[0]["status"] == "invalid" and rows[0]["max_rank_mean_us"] is None
