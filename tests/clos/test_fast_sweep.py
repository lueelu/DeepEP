"""Candidate admissibility, profiler accounting and real QP-mask coverage."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("clos_fast_sweep", ROOT / "benchmarks/clos/bench_fast_sweep.py")
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("aiv", [8, 20, 32, 64])
def test_admission_and_unique_candidates(ep, aiv):
    group = min(8, ep)
    configs = sweep.candidates(ep, aiv, group)
    assert configs
    assert len({c["id"] for c in configs}) == len(configs)
    for c in configs:
        owners = ((group if c["order"] == "group_round_robin" else ep - 1) + c["peers"] - 1) // c["peers"]
        assert c["payload_aivs"] == owners * c["workers"] <= aiv
        assert owners * c["workers"] * c["qps"] <= 32
        assert c["workers"] == 1 or c["qps"] == 1
        if c["order"] == "group_round_robin":
            assert ((group + owners - 1) // owners) * (ep // group) <= 64


def test_quick_subset_and_auto_baseline():
    configs = sweep.candidates(64, 32, 8)
    assert configs[0]["id"] == "rank_p4_w1_q1_v32"
    quick = sweep.candidates(64, 32, 8, quick=True)
    assert {c["id"] for c in quick} < {c["id"] for c in configs}
    assert {c["order"] for c in quick} == {"rank", "group_round_robin"}


@pytest.mark.parametrize("ep", [4, 8, 16, 32])
@pytest.mark.parametrize("aiv", [8, 20])
def test_modes_keep_identity_and_prune_only_equivalent_spare_copy(ep, aiv):
    base = sweep.candidates(ep, aiv, min(8, ep))
    combined = sweep.combine_candidates(base, sweep.SELF_COPY_MODES, aiv)
    assert len({c["id"] for c in combined}) == len(combined)
    assert len([c for c in combined if c["self_copy_mode"] == "default"]) == len(base)
    assert len([c for c in combined if c["self_copy_mode"] == "self_copy_after_submit"]) == len(base)
    assert len([c for c in combined if c["self_copy_mode"] == "parallel_self_copy"]) == sum(
        c["payload_aivs"] < aiv for c in base
    )
    assert len(sweep.combine_candidates(base, ["parallel_self_copy"], aiv)) == len(base)


def test_mode_switch_drains_work_and_clears_previous_flags(monkeypatch):
    events = []
    monkeypatch.setenv(sweep.COMBINE_SELF_COPY_ENV, "1")
    monkeypatch.setenv(sweep.COMBINE_AFTER_SUBMIT_ENV, "1")
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("drain")))
    buffer = object()

    def configure(actual_buffer, config):
        assert actual_buffer is buffer
        assert events[-1] == "drain"
        events.append((os.environ[sweep.COMBINE_SELF_COPY_ENV], os.environ[sweep.COMBINE_AFTER_SUBMIT_ENV]))

    monkeypatch.setattr(sweep, "configure", configure)
    for mode in ("self_copy_after_submit", "parallel_self_copy", "default", "self_copy_after_submit"):
        sweep.activate_candidate(torch, buffer, {"self_copy_mode": mode})
    sweep.activate_candidate(torch, buffer, {"id": "dispatch"})
    assert events[1::2] == [("0", "1"), ("1", "0"), ("0", "0"), ("0", "1"), ("0", "0")]


def test_multi_mode_cli_and_conflicts():
    args = sweep.parse_args(["--combine-self-copy-modes", *sweep.SELF_COPY_MODES])
    assert args.combine_self_copy_modes == list(sweep.SELF_COPY_MODES)
    for argv in (
        ["--combine-self-copy-modes", "default", "default"],
        ["--combine-self-copy-modes", "default", "--combine-self-copy-after-submit"],
        ["--combine-parallel-self-copy", "--combine-self-copy-after-submit"],
    ):
        with pytest.raises(SystemExit):
            sweep.parse_args(argv)


def test_cli_plan_without_framework():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/clos/bench_fast_sweep.py"),
            "--ep-size",
            "4",
            "--case",
            "2048,6144,8,20",
            "--plan-only",
            "--smoke",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert '"hidden": 6144' in result.stdout


@pytest.mark.parametrize("case", ["0,2816,8,32", "8192,2816,8,3", "8192,2817,8,32", "1,2,3"])
def test_bad_shapes(case):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--case", case])


def test_replay_rejects_incomplete_or_illegal_strategy(tmp_path, monkeypatch, capsys):
    shape = sweep.parse_shape("2048,6144,8,20")
    configs = sweep.candidates(4, 20, 4)
    selected = {
        "cached_dispatch": {"winner": {"config": configs[0]}},
        "combine_unweighted": {"winner": {"config": configs[-1]}},
    }
    report = dict(
        ep=4,
        experts=256,
        seed=20260823,
        ranks_per_group=4,
        huge_pages=True,
        flush_l2=False,
        results=[dict(shape=shape, operations=selected)],
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    argv = ["sweep", "--ep-size", "4", "--case", "2048,6144,8,20", "--plan-only", "--replay-report", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    sweep.main()
    plan = json.loads(capsys.readouterr().out)["plans"][0]
    assert len(plan["configs"]) == 2
    assert plan["operations"]["cached_dispatch"] == [configs[0]]
    assert plan["operations"]["combine_unweighted"] == sweep.combine_candidates([configs[-1]], ["default"], 20)
    selected["combine_unweighted"]["winner"]["config"] = configs[-1] | {"qps": 99}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid candidate"):
        sweep.main()
    report["flush_l2"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="settings must match"):
        sweep.main()


def test_replay_preserves_old_and_new_modes():
    config = sweep.candidates(4, 8, 4)[0]
    legal = {config["id"]: config}
    for mode in sweep.SELF_COPY_MODES:
        old = dict(
            combine_parallel_self_copy=mode == "parallel_self_copy",
            combine_self_copy_after_submit=mode == "self_copy_after_submit",
        )
        expected = sweep.combine_candidates([config], [mode], 8)[0]
        assert sweep.replay_config(config, "combine_unweighted", old, legal) == expected
        assert sweep.replay_config(expected, "combine_unweighted", {}, legal) == expected
        overridden = sweep.replay_config(expected, "combine_unweighted", {}, legal, "default")
        assert overridden["self_copy_mode"] == "default"
        with pytest.raises(ValueError, match="invalid candidate"):
            sweep.replay_config(expected | {"id": config["id"]}, "combine_unweighted", {}, legal)
        with pytest.raises(ValueError, match="invalid candidate"):
            sweep.replay_config(expected, "cached_dispatch", {}, legal)


def test_full_runner_reuses_one_buffer_across_aiv_and_modes(tmp_path, monkeypatch):
    """Exercise orchestration without NPU; this is not kernel validation."""
    from types import ModuleType

    class Tensor:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def __ne__(self, other):
            return self

        def __floordiv__(self, other):
            return self

        def __int__(self):
            return 0

    tensor = Tensor()
    instances, measured, configured = [], [], []

    class Buffer:
        get_buffer_size_hint = staticmethod(lambda *args: 1024)

        def __init__(self, *args, **kwargs):
            instances.append(self)
            self.destroyed = False

        def dispatch(self, *args, **kwargs):
            return tensor, None, None, SimpleNamespace(num_recv_tokens=8192), tensor

        def combine(self, *args, **kwargs):
            return tensor, tensor

        def destroy(self):
            self.destroyed = True

    group = SimpleNamespace(agree=lambda value: None)
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(set_device=lambda d: None, synchronize=lambda: None, empty_cache=lambda: None)
    torch.Generator = lambda: tensor
    torch.rand = torch.randn = lambda *a, **kw: tensor
    torch.bfloat16 = "bf16"
    dist = ModuleType("torch.distributed")
    dist.TCPStore = lambda *a, **kw: object()
    torch.distributed = dist
    names = [
        "deep_ep",
        "deep_ep.backends",
        "deep_ep.backends.direct",
        "deep_ep.backends.direct.buffer",
        "deep_ep.backends.direct.elastic",
        "deep_ep.backends.direct.dtypes",
        "torch_npu",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    elastic = sys.modules["deep_ep.backends.direct.elastic"]
    elastic.ElasticBuffer = Buffer
    elastic.__file__ = str(tmp_path / "elastic.py")
    (tmp_path / "elastic.py").write_text("# fake")
    native = tmp_path / "_clos_C.so"
    native.write_bytes(b"fake")
    (tmp_path / "libdeepep_clos_kernels.so").write_bytes(b"fake")
    (tmp_path / ".libs").mkdir()
    (tmp_path / ".libs/libshmem.so").write_bytes(b"fake")
    sys.modules["deep_ep.backends.direct.dtypes"].topk_idx_t = "int64"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(native)))
    monkeypatch.setattr(sweep, "StoreGroup", lambda *a: group)
    monkeypatch.setattr(sweep, "configure", lambda b, c: configured.append((b, c)))
    for name, value in {
        "RANK": "0",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        sweep.COMBINE_SELF_COPY_ENV: "0",
        sweep.COMBINE_AFTER_SUBMIT_ENV: "0",
        "DEEPEP_CLOS_TUNING_SESSION": "0",
    }.items():
        monkeypatch.setenv(name, value)

    def measure(torch, torch_npu, buffer, group, args, directory, op, configs, invoke, logical_bytes, samples):
        measured.append((op, directory.name, configs))
        for c in configs:
            sweep.activate_candidate(torch, buffer, c)
            invoke()
        return [
            dict(config=c, GBps=100 + i, mean_us=10, samples_per_rank=samples)
            for i, c in reversed(list(enumerate(configs)))
        ]

    monkeypatch.setattr(sweep, "measure", measure)
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--ep-size",
            "4",
            "--case",
            "8192,2816,8,8",
            "--case",
            "8192,2816,8,20",
            "--smoke",
            "--output",
            str(out),
            "--combine-self-copy-modes",
            *sweep.SELF_COPY_MODES,
        ],
    )
    sweep.main()
    assert len(instances) == 1 and instances[0].destroyed
    assert all(b is instances[0] for b, c in configured)
    coarse = [(op, cs) for op, name, cs in measured if name.endswith("_coarse")]
    assert len(coarse) == 4  # two AIVs, two operators; dispatch is not repeated per mode
    for op, cs in coarse:
        if op == "cached_dispatch":
            assert all("self_copy_mode" not in c for c in cs)
        else:
            assert {c["self_copy_mode"] for c in cs} == set(sweep.SELF_COPY_MODES)
    report = json.loads((out / "report.json").read_text())
    assert report["buffer_initializations"] == 1 and len(report["results"]) == 2
    for result in report["results"]:
        winner = result["operations"]["combine_unweighted"]["winner"]
        assert "self_copy_mode" in winner["config"] and len(winner["repeats"]) == 2


@pytest.mark.parametrize("combine", [False, True])
def test_profile_stage_separation_and_missing_sample(tmp_path, combine):
    op = "combine_unweighted" if combine else "cached_dispatch"
    names = ["cached_notify_combine_kernel"]
    names += (
        ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
        if combine
        else ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    )
    warmup, samples = 2, 3
    calls = warmup + samples + 2 + 4
    path = tmp_path / "kernel_details.csv"
    rows = []
    for call in range(calls):
        for stage, name in enumerate(names):
            rows.append(
                {
                    "Name": name,
                    "Start Time(us)": call * 10000 + stage * 1000,
                    "Duration(us)": [100, 200, 300, 400][stage],
                }
            )

    def save():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    save()
    result = sweep.parse_capture(tmp_path, op, [{"id": "test"}], warmup, samples)[0]
    assert result["mean_us"] == 300
    assert result["barrier_us"] == 100
    assert result["notify_us"] == 200
    assert result["reduce_us"] == (400 if combine else 0)
    rows.pop()
    save()
    with pytest.raises(RuntimeError, match="incomplete capture"):
        sweep.parse_capture(tmp_path, op, [{"id": "test"}], warmup, samples)


def test_production_qp_union_covers_all_schedules(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for production QP union execution")
    harness = tmp_path / "coverage.cpp"
    harness.write_text(
        r"""
#include "csrc/direct/common/udma_schedule.h"
#include <cassert>
#include <cstdio>
using namespace ascend_deepep::udma;
int main() {
  for (uint32_t world : {2U,4U,8U,16U,32U,64U,128U,256U})
  for (uint32_t group : {1U,2U,4U,8U,16U}) {
    if (group > world || world % group) continue;
    for (uint32_t rank=0; rank<world; ++rank)
    for (uint32_t peer=0; peer<world; ++peer) {
      const auto mask=TuningPeerMask(rank,peer,world,group);
      assert(mask==TuningPeerMask(peer,rank,world,group));
      if (rank==peer) { assert(mask==0); continue; }
      assert(mask & 1);
      for (uint32_t order=0; order<=3; ++order)
      for (uint32_t peers : {1U,2U,4U,8U})
      for (uint32_t lanes : {1U,2U}) {
        if (order==3 && peers==8) continue;
        const auto workers=Workers(world,peers,order,group);
        if (workers*lanes>32) continue;
        if (order==3 && ((group+workers-1)/workers)*(world/group)>64) continue;
        const auto owner=Owner(rank,peer,world,peers,order,group);
        assert(owner<workers);
        for (uint32_t lane=0;lane<lanes;++lane)
          assert(mask & (uint64_t{1} << (owner*lanes+lane)));
      }
    }
  }
  std::puts("QP_UNION_OK");
}
""".replace("#include <cstdio>", "#include <cstdio>\n#include <initializer_list>")
    )
    binary = tmp_path / "coverage"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I" + str(ROOT), str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60, check=True)
    assert "QP_UNION_OK" in result.stdout
