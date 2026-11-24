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
        ["--include-weighted-combine", "--combine-parallel-self-copy"],
        ["--strategies", "group_p1_w1_q1_v32", "--smoke"],
    ):
        with pytest.raises(SystemExit):
            sweep.parse_args(argv)


def test_weighted_fixed_strategy_plan(tmp_path, monkeypatch, capsys):
    argv = [
        "sweep",
        "--ep-size",
        "8",
        "--case",
        "8192,2816,8,32",
        "--include-weighted-combine",
        "--strategies",
        "group_p1_w1_q1_v32",
        "--combine-self-copy-modes",
        *sweep.SELF_COPY_MODES,
        "--plan-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    sweep.main()
    plan = json.loads(capsys.readouterr().out)["plans"][0]
    ops = plan["operations"]
    assert len(ops["cached_dispatch"]) == 1
    assert len(ops["combine_unweighted"]) == 3
    assert {c["self_copy_mode"] for c in ops["combine_weighted"]} == {"default", "self_copy_after_submit"}
    assert all(
        c["order"] == "group" and c["peers"] == c["workers"] == c["qps"] == 1 and c["visit"] == 32
        for cs in ops.values()
        for c in cs
    )
    argv[argv.index("group_p1_w1_q1_v32")] = "group_p1_w99_q1_v32"
    with pytest.raises(ValueError, match="invalid strategies"):
        sweep.main()


def test_weighted_replay_modes():
    c = sweep.candidates(8, 32, 8)[0]
    legal = {c["id"]: c}
    for mode in ("default", "self_copy_after_submit"):
        candidate = sweep.combine_candidates([c], [mode], 32, weighted=True)[0]
        assert sweep.replay_config(candidate, "combine_weighted", {}, legal) == candidate
        with pytest.raises(ValueError, match="does not support"):
            sweep.replay_config(candidate, "combine_weighted", {}, legal, "parallel_self_copy")


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


@pytest.mark.parametrize("weighted", [False, True])
def test_full_runner_reuses_one_buffer_across_aiv_and_modes(tmp_path, monkeypatch, weighted):
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
            weights = tensor if kwargs.get("topk_weights") is not None else None
            return tensor, None, weights, SimpleNamespace(num_recv_tokens=8192), tensor

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
    torch.float32 = "fp32"
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
        sweep.QP_PLANS_ENV: "",
    }.items():
        monkeypatch.setenv(name, value)

    def measure(torch, torch_npu, buffer, group, args, directory, op, configs, invoke, logical_bytes, samples):
        measured.append((op, directory.name, configs))
        if op != "cached_dispatch":
            assert logical_bytes == 8192 * (2816 * 2 + (8 * 4 if op == "combine_weighted" else 0))
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
        ]
        + (["--include-weighted-combine"] if weighted else []),
    )
    sweep.main()
    assert len(instances) == 1 and instances[0].destroyed
    assert all(b is instances[0] for b, c in configured)
    coarse = [(op, cs) for op, name, cs in measured if name.endswith("_coarse")]
    assert len(coarse) == (6 if weighted else 4)  # dispatch is not repeated per mode
    for op, cs in coarse:
        if op == "cached_dispatch":
            assert all("self_copy_mode" not in c for c in cs)
        elif op == "combine_unweighted":
            assert {c["self_copy_mode"] for c in cs} == set(sweep.SELF_COPY_MODES)
        else:
            assert {c["self_copy_mode"] for c in cs} == {"default", "self_copy_after_submit"}
    report = json.loads((out / "report.json").read_text())
    assert report["buffer_initializations"] == 1 and len(report["results"]) == 2
    for result in report["results"]:
        winner = result["operations"]["combine_unweighted"]["winner"]
        assert "self_copy_mode" in winner["config"] and len(winner["repeats"]) == 2
        assert ("combine_weighted" in result["operations"]) == weighted
        if weighted:
            assert result["operations"]["combine_weighted"]["logical_row_bytes"] == 5664


@pytest.mark.parametrize("op", ["cached_dispatch", "combine_unweighted", "combine_weighted"])
def test_profile_stage_separation_and_missing_sample(tmp_path, op):
    combine = op != "cached_dispatch"
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
#include "common/udma_schedule.h"
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
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I" + str(ROOT / "csrc/direct"),
            str(harness),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    try:
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60, check=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise
    assert "QP_UNION_OK" in result.stdout


def test_qp_reservation_uses_all_operations_but_not_visit_or_copy_mode():
    a = dict(order="rank", peers=8, workers=1, qps=1, visit=8)
    b = dict(a, visit=32, self_copy_mode="parallel_self_copy")
    c = dict(a, order="group_round_robin", peers=2, qps=2)
    plans = [dict(operations={"cached_dispatch": [a, b], "combine_weighted": [c]})]
    assert sweep.qp_plan_spec(plans) == "group_round_robin:2:1:2;rank:8:1:1"


def test_exact_qp_reservation_matches_worker_assignment(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    harness = tmp_path / "reservation.cpp"
    harness.write_text(r"""
#include "runtime/udma_qp_reservation.h"
#include <cassert>
#include <iostream>
using namespace ascend_deepep::udma;
// These checks execute during compilation even when Windows blocks the EXE.
constexpr bool pair_coverage(uint32_t world,uint32_t group,uint32_t order,uint32_t p,uint32_t lanes,
                             uint32_t rank,uint32_t peer) {
  uint64_t expected=rank==peer ? 0 : 1;
  for (uint32_t src : {rank,peer})
  for (uint32_t owner=0;owner<Workers(world,p,order,group);++owner)
  for (uint32_t i=0;i<WorkerCount(src,world,owner,p,order,group);++i) {
    const auto dst=WorkerPeer(src,world,owner,p,i,order,group);
    if (dst==(src==rank ? peer : rank)) for(uint32_t lane=0;lane<lanes;++lane)
      expected |= uint64_t{1}<<(owner*lanes+lane);
  }
  const auto actual=QpReservation::Mask(rank,peer,world,group,order,p,lanes);
  return actual==expected && actual==QpReservation::Mask(peer,rank,world,group,order,p,lanes);
}
static_assert(pair_coverage(4,4,0,1,2,0,3));
static_assert(pair_coverage(8,8,0,2,1,0,7));
static_assert(pair_coverage(16,16,2,2,2,3,15));
static_assert(pair_coverage(32,16,0,4,1,0,31));
static_assert(pair_coverage(64,16,0,8,1,0,63));
static_assert(pair_coverage(128,16,3,2,2,17,127));
static_assert(pair_coverage(256,16,1,8,1,127,255));
static_assert(pair_coverage(64,16,0,8,1,0,0));
int main() {
  for (uint32_t world : {4U,8U,16U,32U,64U,128U,256U}) {
    const uint32_t group = world < 16 ? world : 16;
    for (uint32_t order=0; order<4; ++order)
    for (uint32_t p : {1U,2U,4U,8U})
    for (uint32_t lanes : {1U,2U}) {
      const auto owners=Workers(world,p,order,group);
      if (owners*lanes>32 || (order==3 && (p>4 || ((group+owners-1)/owners)*(world/group)>64))) continue;
      const char* names[]={"rank","group","group_staggered","group_round_robin"};
      const auto spec=std::string(names[order])+":"+std::to_string(p)+":1:"+std::to_string(lanes);
      std::vector<QpReservation> reservations;
      for (uint32_t rank=0; rank<world; ++rank) reservations.emplace_back(spec,rank,world,group);
      for (uint32_t rank=0; rank<world; ++rank) {
        std::vector<uint64_t> expected(world,1); expected[rank]=0;
        // Oracle enumerates real sender assignments at BOTH ends. It does not
        // reconstruct masks with Owner(), the function used by the reservation.
        for (uint32_t src=0; src<world; ++src)
        for (uint32_t owner=0; owner<owners; ++owner)
        for (uint32_t i=0; i<WorkerCount(src,world,owner,p,order,group); ++i) {
          const auto dst=WorkerPeer(src,world,owner,p,i,order,group);
          if (src==rank || dst==rank) for (uint32_t lane=0; lane<lanes; ++lane)
            expected[src==rank ? dst : src] |= uint64_t{1}<<(owner*lanes+lane);
        }
        assert(reservations[rank].Masks()==expected);
        assert(reservations[rank].Allows(order,p,1,lanes));
        assert(reservations[rank].Allows(order,p,lanes,1));
        for (uint32_t peer=0; peer<world; ++peer) {
          assert(expected[peer]==reservations[peer].Masks()[rank]);
          assert((expected[peer]&~TuningPeerMask(rank,peer,world,group))==0);
        }
      }
    }
  }
  const QpReservation narrow("rank:8:1:1",0,64,16);
  assert(!narrow.Allows(0,4,1,1));
  assert(!narrow.Allows(0,8,1,2));
  const QpReservation both("rank:8:1:1;rank:4:1:1",0,64,16);
  assert(both.Allows(0,8,1,1) && both.Allows(0,4,1,1));
  for (const auto* bad : {"rank:0:1:1","rank:8:2:2","rank:8:1:1;","rank:8:1", "rank:8:1:1:1", "rank:-1:1:1", "unknown:8:1:1", ";"}) {
    bool rejected=false;
    try { QpReservation unused(bad,0,64,16); } catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  assert(QpReservation().Masks().empty());
  assert(QpReservation().Allows(3,4,2,1));
  for (uint32_t world : {32U,64U}) {
    QpReservation r(world==32 ? "rank:4:1:1" : "rank:8:1:1",0,world,16);
    uint32_t before=0,after=0;
    for (uint32_t peer=0;peer<world;++peer) for (uint32_t bit=0;bit<32;++bit) {
      before+=(TuningPeerMask(0,peer,world,16)>>bit)&1;
      after+=(r.Masks()[peer]>>bit)&1;
    }
    assert(after<before);
    std::cout << "EP" << world << " peer-QP entries " << before << " -> " << after << '\n';
  }
  std::cout << "QP_RESERVATION_OK\n";
}
""")
    binary = tmp_path / "reservation.exe"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-I" + str(ROOT / "csrc/direct"), str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=90,
    )
    try:
        result = subprocess.run([str(binary)], capture_output=True, text=True, check=True, timeout=90)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise
    assert "QP_RESERVATION_OK" in result.stdout
    print(result.stdout)
