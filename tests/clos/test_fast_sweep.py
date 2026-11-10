"""Candidate admissibility, profiler accounting and real QP-mask coverage."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

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
    selected["combine_unweighted"]["winner"]["config"] = configs[-1] | {"qps": 99}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid candidate"):
        sweep.main()
    report["flush_l2"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="settings must match"):
        sweep.main()


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
