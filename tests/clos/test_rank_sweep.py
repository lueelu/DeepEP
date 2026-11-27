"""Staged search admission, native observations and independent timing accounting."""

import csv
import importlib.util
import json
from pathlib import Path
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/clos"
SPEC = importlib.util.spec_from_file_location("rank_sweep", BENCH / "rank_sweep.py")
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


@pytest.mark.parametrize(
    "ep,cores,expected",
    [
        (4, 8, [(1, 1, 1, 3), (1, 1, 2, 3), (1, 2, 1, 6)]),
        (4, 32, [(1, 1, 1, 3), (1, 1, 2, 3), (1, 2, 1, 6)]),
        (8, 8, [(1, 1, 1, 7), (1, 1, 2, 7), (2, 2, 1, 8)]),
        (8, 32, [(1, 1, 1, 7), (1, 1, 2, 7), (1, 2, 1, 14)]),
        (16, 8, [(2, 1, 1, 8), (2, 1, 2, 8), (4, 2, 1, 8)]),
        (16, 32, [(1, 1, 1, 15), (1, 1, 2, 15), (1, 2, 1, 30)]),
        (32, 8, [(4, 1, 1, 8), (4, 1, 2, 8), (8, 2, 1, 8)]),
        (32, 32, [(1, 1, 1, 31), (2, 1, 2, 16), (2, 2, 1, 32)]),
    ],
)
def test_rank_shortlist_core_allocation(ep, cores, expected):
    actual = sweep.shortlist(ep, cores, visits=(32,))
    assert [(c["peers"], c["workers"], c["qps"], c["payload_aivs"]) for c in actual] == expected


@pytest.mark.parametrize("ep,group", [(4, 4), (8, 8), (16, 16), (32, 16), (32, 8)])
def test_matrix_policy_and_qp_reservation(ep, group):
    from tests.api.test_clos_compatibility import run_host

    run_host(f"""
        sys.path.insert(0, 'benchmarks/clos')
        from rank_sweep import cases, shortlist, SweepPolicy, policy_config, OPS, op_kind, VISITS
        from bench_fast_sweep import qp_plan_spec
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        from dataclasses import asdict, replace
        base = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
        base = replace(base, ranks_per_group={group}, topology_known=True)
        wrapper = SweepPolicy(base)
        shapes = cases()
        assert len(shapes) == 12
        assert all(s['tokens'] == 8192 and s['hidden'] in (2816,6144) for s in shapes[:4])
        configs = []
        for s in shapes:
            a, h = s['aiv'], s['hidden']
            assert len(shortlist({ep}, a)) == 6
            for op in OPS:
                original = base.for_launch(a, h*2, op_kind(op), bf16=True)
                configs.append(policy_config(original, {ep}))
                for c in shortlist({ep}, a, VISITS):
                    wrapper.config, wrapper.operation = c, op_kind(op)
                    plan = wrapper.for_launch(a, h*2, op_kind(op), bf16=True)
                    plan.validate_num_aiv(a)
                    assert plan.send_workers == c['payload_aivs'] <= a
                    assert plan.send_workers * c['qps'] <= 32
                    before, after = asdict(original), asdict(plan)
                    for field in ('peer_order','peers_per_worker','workers_per_peer','qps_per_worker','wqes_per_visit'):
                        before.pop(field); after.pop(field)
                    assert before == after
                    configs.append(c)
                wrapper.config = policy_config(original, {ep}, True)
                assert wrapper.for_launch(a, h*2, op_kind(op), bf16=True) == original
        spec = qp_plan_spec([dict(operations={{'all':configs}})])
        assert len(spec.split(';')) <= 8
        assert all(v.startswith('rank:') for v in spec.split(';'))
    """)


@pytest.mark.parametrize("ep", [4, 8, 16, 32])
def test_cli_plan_and_rejected_selectors(monkeypatch, capsys, ep):
    monkeypatch.syspath_prepend(str(BENCH))
    import bench_fast_sweep as entry

    args = entry.parse_args(["--small-rank-sweep", "--ep-size", str(ep), "--plan-only"])
    assert args.seed == 20261122 and not args.huge_pages and args.reduce_cores == 48
    sweep.run(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["ep"] == ep
    assert len(plan["cases"]) == 12 and len(plan["operators"]) == 6
    for extra in (
        ["--ep-size", "64"],
        ["--huge-pages"],
        ["--case", "4096,2816,8,8"],
        ["--seed=1"],
        ["--visits=32"],
        ["--ranks-per-group=4"],
        ["--reduce-cores", "7"],
    ):
        with pytest.raises(SystemExit):
            entry.parse_args(["--small-rank-sweep", "--ep-size", "8", *extra])


def test_aggregation_is_max_per_sample_not_max_rank_mean():
    result = sweep.slowest([{"notify": [10, 100], "comm": [1, 9]}, {"notify": [100, 10], "comm": [9, 1]}])
    assert sweep.stats(result["notify"])["mean"] == 100
    assert sweep.stats(result["comm"])["p95"] == 9
    with pytest.raises(ValueError):
        sweep.slowest([{"comm": [1]}, {"comm": [1, 2]}])


@pytest.mark.parametrize("ep", [4, 8, 16, 32])
def test_combine_defaults_cli_no_search(monkeypatch, capsys, ep):
    monkeypatch.syspath_prepend(str(BENCH))
    import bench_fast_sweep as entry

    mode = ["--combine-defaults-only", "--skip-precision", "--single-pass", "--ep-size", str(ep)]
    args = entry.parse_args([*mode, "--plan-only"])
    assert args.combine_payload_max_rows is None and args.combine_visits is None and args.combine_wqe_fast is None
    assert len(args.case) == 6 and all(c["tokens"] == 8192 for c in args.case)
    sweep.run(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["operators"] == ["combine", "combine_weighted"] and plan["candidates"] == []
    assert plan["pre_combine_notify"] and not plan["precision_checked"] and plan["final_passes"] == 1
    for extra in (["--small-rank-sweep"], ["--combine-payload-max-rows", "1", "2"], ["--combine-visits", "8"]):
        with pytest.raises(SystemExit):
            entry.parse_args([*mode, *extra])
    with pytest.raises(SystemExit):
        entry.parse_args(["--skip-precision"])


def test_default_run_case_skips_all_golden_checks_and_reverse_pass(monkeypatch):
    import sys

    class Tensor:
        def npu(self):
            return self

        def to(self, **kwargs):
            return self

        def clone(self):
            return self

        def __lt__(self, other):
            return self

        def __floordiv__(self, other):
            return self

        def __eq__(self, other):
            return self

        def any(self, dim):
            return self

        def sum(self):
            return 3

    def forbidden(*args, **kwargs):
        raise AssertionError("precision check must be skipped")

    x = Tensor()
    monkeypatch.setitem(
        sys.modules,
        "bench_dispatch_stages",
        SimpleNamespace(cpu_inputs=lambda *a: (x, x, x), check_cpu_reference=forbidden),
    )
    monkeypatch.setitem(sys.modules, "deep_ep.backends.direct.dtypes", SimpleNamespace(topk_idx_t=None))
    monkeypatch.setattr(sweep, "precision", forbidden)
    plan = SimpleNamespace(peer_order=0, peers_per_worker=2, workers_per_peer=1, qps_per_worker=1, wqes_per_visit=8)
    measured, launches = [], []

    def measure(s, shape, op, configs, invoke, logical_bytes, phase, samples):
        assert len(configs) == 1 and configs[0]["auto"] and "payload_max_rows" not in configs[0]
        invoke()
        measured.append((op, phase, logical_bytes))
        return [
            dict(
                config=configs[0],
                samples={"comm": [1, 2], "notify": [3, 4]},
                metrics={"comm": {"mean": 1.5}},
                payload_bytes_per_rank=logical_bytes,
            )
        ]

    monkeypatch.setattr(sweep, "measure", measure)
    s = SimpleNamespace(
        torch=SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)),
        rank=0,
        args=SimpleNamespace(
            combine_defaults_only=True,
            combine_payload_max_rows=None,
            skip_precision=True,
            single_pass=True,
            reduce_cores=48,
            final_samples=2,
        ),
        base=SimpleNamespace(for_launch=lambda *a, **kw: plan),
        group=SimpleNamespace(size=lambda: 16, agree=lambda *a: None),
        buffer=SimpleNamespace(
            dispatch=lambda *a, **kw: [x, x, x, object()], combine=lambda *a, **kw: launches.append(kw)
        ),
        activate=lambda *a: None,
        prepare=lambda: None,
        done=lambda x: x,
        write=lambda *a: None,
        out=Path("unused"),
        completed=[],
    )
    results = sweep.run_case(s, dict(tokens=8192, hidden=2816, topk=8, aiv=8), {})
    assert [(op, phase) for op, phase, _ in measured] == [("combine", "final0"), ("combine_weighted", "final0")]
    assert all(not r["results"][0]["precision_checked"] for r in results)
    assert len(launches) == 2 and all(r["reduce_num_sms"] == 48 for r in launches)


def test_payload_sweep_cli_and_frozen_plan(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(BENCH))
    import bench_fast_sweep as entry

    argv = ["--ep-size", "16", "--case", "4096,2816,8,8", "--combine-payload-max-rows", "2", "4", "8", "--plan-only"]
    args = entry.parse_args(argv)
    assert args.combine_payload_max_rows == [2, 4, 8]
    assert len(args.case) == 1 and not args.huge_pages
    sweep.run(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["operators"] == ["combine", "combine_weighted"]
    assert plan["payload_max_rows"] == [2, 4, 8]
    assert plan["qp_plans"] is None  # Resolve only actual default plans at runtime.
    auto = sweep.candidate(2, 1, 1, 8, 16) | {"id": "auto", "auto": True}
    candidates = sweep.payload_candidates(auto, args.combine_payload_max_rows)
    policy = SimpleNamespace(config=None, operation=None)
    monkeypatch.delenv(sweep.PAYLOAD_ROWS_ENV, raising=False)
    for config in candidates + list(reversed(candidates)):
        sweep.activate_config(policy, config, "combine_weighted")
        assert os.environ[sweep.PAYLOAD_ROWS_ENV] == str(config["payload_max_rows"])
        assert all(config[k] == auto[k] for k in ("order", "peers", "workers", "qps", "visit", "payload_aivs"))
        native = SimpleNamespace(
            last_combine_payload_max_rows=lambda: int(os.environ[sweep.PAYLOAD_ROWS_ENV]),
            last_combine_wqe_fast=lambda: int(os.environ[sweep.WQE_FAST_ENV]),
        )
        assert sweep.check_payload_limit(native, config)["payload_max_rows_submitted"] == config["payload_max_rows"]
    sweep.activate_config(policy, auto, "fresh_weighted")
    assert sweep.PAYLOAD_ROWS_ENV not in os.environ
    with pytest.raises(RuntimeError, match="limit mismatch"):
        sweep.check_payload_limit(SimpleNamespace(last_combine_payload_max_rows=lambda: 1), candidates[1])
    for extra in (["0"], ["33"], ["2", "2"], ["2", "--small-rank-sweep"], ["2", "--visits", "32"]):
        with pytest.raises(SystemExit):
            entry.parse_args(["--ep-size", "16", "--combine-payload-max-rows", *extra])


def test_payload_sweep_axes_and_native_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH))
    import bench_fast_sweep as entry

    args = entry.parse_args(
        [
            "--ep-size",
            "32",
            "--combine-payload-max-rows",
            "1",
            "2",
            "4",
            "--combine-visits",
            "4",
            "8",
            "32",
            "--combine-wqe-fast",
            "0",
            "1",
        ]
    )
    auto = sweep.candidate(4, 1, 1, 8, 32) | {"id": "auto", "auto": True}
    configs = sweep.payload_candidates(auto, args.combine_payload_max_rows, args.combine_visits, args.combine_wqe_fast)
    assert len(configs) == 12 and len({c["id"] for c in configs}) == 12
    assert sum(c["auto"] for c in configs) == 1
    assert all(c["combine_wqe_fast"] == 0 for c in configs if c["payload_max_rows"] > 1)
    assert all((c["order"], c["peers"], c["workers"], c["qps"]) == ("rank", 4, 1, 1) for c in configs)
    monkeypatch.delenv(sweep.PAYLOAD_ROWS_ENV, raising=False)
    monkeypatch.delenv(sweep.WQE_FAST_ENV, raising=False)
    policy = SimpleNamespace(config=None, operation=None)
    for config in configs + list(reversed(configs)):
        sweep.activate_config(policy, config, "combine")
        native = SimpleNamespace(
            last_combine_payload_max_rows=lambda: int(os.environ[sweep.PAYLOAD_ROWS_ENV]),
            last_combine_wqe_fast=lambda: int(os.environ[sweep.WQE_FAST_ENV]),
        )
        assert sweep.check_payload_limit(native, config)["combine_wqe_fast_submitted"] == config["combine_wqe_fast"]
    wrong = SimpleNamespace(last_combine_payload_max_rows=lambda: 1, last_combine_wqe_fast=lambda: 0)
    with pytest.raises(RuntimeError, match="fast mismatch"):
        sweep.check_payload_limit(wrong, next(c for c in configs if c["combine_wqe_fast"]))
    sweep.activate_config(policy, auto, "fresh_weighted")
    assert sweep.WQE_FAST_ENV not in os.environ and sweep.PAYLOAD_ROWS_ENV not in os.environ
    for extra in (
        ["--combine-visits", "3"],
        ["--combine-visits", "8", "8"],
        ["--combine-wqe-fast", "2"],
        ["--combine-wqe-fast", "0", "0"],
    ):
        with pytest.raises(SystemExit):
            entry.parse_args(["--ep-size", "16", "--combine-payload-max-rows", "1", *extra])
    with pytest.raises(SystemExit):
        entry.parse_args(["--combine-visits", "8"])


def test_payload_summary_without_implicit_baseline(tmp_path):
    auto = sweep.candidate(2, 1, 1, 8, 16) | {"id": "auto", "auto": True}
    configs = sweep.payload_candidates(auto, [2, 4])
    results = [
        dict(config=c, metrics={"comm": {"mean": 80}}, GBps=300, rounds=[], precision_checked=False) for c in configs
    ]
    state = SimpleNamespace(
        completed=[dict(shape={}, operator="combine", results=results, winner=configs[0])],
        args=SimpleNamespace(output=tmp_path, reduce_cores=48),
        group=SimpleNamespace(size=lambda: 16),
    )
    sweep.write_summary(state)
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["payload_max_rows"] for r in rows] == ["2", "4"]
    assert all(
        r[k] == ""
        for r in rows
        for k in ("comm_change_vs_auto", "bandwidth_change_vs_auto", "comm_worse_in_either_round")
    )
    assert not any("api" in k for k in rows[0])
    assert all(r["precision"] == "skipped" for r in rows)
    # An unset limit is now selected by row bytes; never report it as explicit rows=1.
    for config in configs:
        config.pop("payload_max_rows")
    sweep.write_summary(state)
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        assert all(row["payload_max_rows"] == "auto" for row in csv.DictReader(f))


def test_finalists_select_communication_winners_without_api():
    rows = [
        dict(config={"id": name, "auto": name == "auto"}, metrics={"comm": {"mean": comm}})
        for name, comm in [("auto", 1), ("slow", 4), ("comm", 2), ("middle", 3)]
    ]
    assert [c["id"] for c in sweep.finalists(rows, 2)] == ["comm", "middle", "auto"]


def test_auto_winner_is_frozen_for_held_out_shapes():
    measured = sweep.candidate(1, 2, 1, 8, 8) | {"id": "auto", "auto": True}
    transferred = sweep.transfer_candidate(measured, 8)
    assert not transferred.get("auto")
    assert transferred["id"] == "rank_p1_w2_q1_v8"
    assert transferred["payload_aivs"] == 14


@pytest.mark.parametrize("op", sweep.OPS)
def test_profile_candidates_and_guards_are_not_mixed(tmp_path, op):
    barrier = "cached_notify_combine_kernel"
    if op.startswith("combine"):
        names = [barrier, barrier, "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
    elif op.startswith("fresh"):
        names = [barrier, "dispatch_layout_simt_kernel", "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    else:
        names = [barrier, "cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    rows = []
    # Two global guards, then each candidate's warmup+samples+two guards, then two guards.
    for iteration in range(2 * (1 + 3 + 2) + 4):
        for field, name in enumerate(names):
            rows.append({"Name": name, "Start Time(us)": len(rows), "Duration(us)": iteration * 10 + field + 1})

    def save():
        with (tmp_path / "kernel_details.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    save()
    values = sweep.parse_profile(tmp_path, op, [{}, {}], 1, 3)
    comm_offset = (
        names.index("v1_combine_udma_comm_kernel" if op.startswith("combine") else "v1_dispatch_udma_kernel") + 1
    )
    assert values[0]["comm"] == [30 + comm_offset, 40 + comm_offset, 50 + comm_offset]
    assert values[1]["comm"] == [90 + comm_offset, 100 + comm_offset, 110 + comm_offset]
    if op.startswith("combine"):
        assert values[0]["notify"] == [32, 42, 52]  # second barrier only
    rows.pop()
    save()
    with pytest.raises(RuntimeError, match="sequence"):
        sweep.parse_profile(tmp_path, op, [{}, {}], 1, 3)


def test_cached_launch_checks_native_not_creation_handle():
    plan = SimpleNamespace(peers_per_worker=1, workers_per_peer=1, qps_per_worker=2, wqes_per_visit=64)
    native = SimpleNamespace(
        last_dispatch_launch=lambda: [0, 1, 1, 2, 64, 8, 1],
        last_dispatch_channel_metadata=lambda: True,
        last_dispatch_metadata_options=lambda: 7,
        last_dispatch_overlap_options=lambda: 8,
    )
    # Result intentionally has no handle policy: cached handles keep their creation plan.
    result = sweep.check_launch(native, None, plan, "cached_weighted", 8, 48)
    assert result["dispatch_launch"][4] == 64
    native.last_dispatch_launch = lambda: [0, 1, 1, 2, 32, 8, 1]
    with pytest.raises(RuntimeError, match="policy"):
        sweep.check_launch(native, None, plan, "cached_weighted", 8, 48)


@pytest.mark.parametrize("defaults_only", [False, True])
def test_measure_profiles_device_stages_without_api_sampling(tmp_path, monkeypatch, defaults_only):
    monkeypatch.syspath_prepend(str(BENCH))
    calls, files, boundaries = [], {}, []
    configs = [{"id": "a"}, {"id": "b"}]
    samples = {stage: [10.0, 20.0] for stage in ("comm", "notify", "reduce", "layout")}
    monkeypatch.setattr(sweep, "parse_profile", lambda *a: [dict(samples), dict(samples)])
    monkeypatch.setattr(sweep, "check_launch", lambda *a: {"observed": True})
    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU=0, NPU=1),
        ExportType=SimpleNamespace(Text=0),
        ProfilerLevel=SimpleNamespace(Level0=0),
        AiCMetrics=SimpleNamespace(AiCoreNone=0),
        profile=lambda **kw: nullcontext(),
        tensorboard_trace_handler=lambda path: None,
        _ExperimentalConfig=lambda **kw: None,
    )
    session = SimpleNamespace(
        torch_npu=SimpleNamespace(profiler=profiler),
        native=SimpleNamespace(last_combine_payload_max_rows=lambda: 4, last_combine_wqe_fast=lambda: 0),
        out=tmp_path,
        args=SimpleNamespace(warmup=1, reduce_cores=48, combine_defaults_only=defaults_only),
        rank=0,
        group=SimpleNamespace(agree=lambda x: None, gather=lambda x: [x, x], size=lambda: 2),
        write=lambda p, v: files.__setitem__(p.name, v),
        activate=lambda c, o: None,
        prepare=lambda: boundaries.append("prepare"),
        done=lambda x: boundaries.append("done"),
        policy=SimpleNamespace(for_launch=lambda *a, **kw: None),
    )

    def invoke():
        boundaries.append("invoke")
        calls.append(os.environ["DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY"])

    rows = sweep.measure(
        session, {"tokens": 4096, "hidden": 2816, "aiv": 8}, "combine", configs, invoke, 15000, "test", 2
    )
    # One untimed launch audit per config, then profiler guards/warmups/samples only.
    assert calls == (["1"] if defaults_only else ["0"]) * 2 + ["1"] * 14
    assert boundaries == ["prepare", "invoke", "done"] * 16
    assert "DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY" not in os.environ
    assert rows[0]["GBps"] == 1
    assert set(files["rank_samples.json"]["stages"][0]) == ({"notify", "comm"} if defaults_only else set(samples))
    assert "api" not in rows[0]["metrics"]
    if defaults_only:
        assert all(v["payload_max_rows_submitted"] == 4 for v in files["rank_samples.json"]["launches"])
    assert not rows[0]["precision_checked"]


@pytest.mark.parametrize("ep", [4, 8, 16, 32])
@pytest.mark.parametrize("payload_only", [False, True, "defaults"])
def test_session_reserves_once_and_reuses_buffer_for_all_shapes(ep, payload_only):
    from tests.api.test_clos_compatibility import run_host

    run_host(
        """
        import os, tempfile, json
        ep = TEST_EP
        payload_only = TEST_PAYLOAD
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0, 'benchmarks/clos')
        import rank_sweep as runner
        import bench_fast_sweep as entry
        import stage_metrics
        from deep_ep.backends.direct import elastic, _native
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        os.environ.update(RANK='0',WORLD_SIZE=str(ep),LOCAL_RANK='0',MASTER_ADDR='localhost')
        for key in list(os.environ):
            if key.startswith('DEEPEP_CLOS_'): del os.environ[key]
        dist = ModuleType('torch.distributed')
        dist.TCPStore = lambda *a,**k:None
        sys.modules['torch.distributed'] = torch.distributed = dist
        sys.modules['torch_npu'] = ModuleType('torch_npu')
        torch.npu = SimpleNamespace(set_device=lambda n:None,synchronize=lambda:None,empty_cache=lambda:None)
        group = SimpleNamespace(size=lambda:ep,rank=lambda:0,gather=lambda x:[x]*ep,agree=lambda x:None)
        entry.StoreGroup = lambda *a:group
        stage_metrics.L2CacheFlush = lambda *a:None
        stage_metrics.prepare_v2_sample = lambda *a:None
        _native.load_native_module = lambda:SimpleNamespace(**{name:lambda:None for name in
            ['last_dispatch_channel_metadata','last_dispatch_metadata_options','last_combine_launch',
             'last_dispatch_launch','last_dispatch_overlap_options','last_combine_payload_max_rows','last_combine_wqe_fast']})
        initialized, executed, destroyed = [], [], []
        class Buffer:
            get_buffer_size_hint = staticmethod(lambda g,t,h,k: t*h*k)
            def __init__(self,g,**kw):
                assert not kw['use_1g_huge_pages'] and kw['num_max_tokens_per_rank']==8192
                assert os.environ['DEEPEP_CLOS_TUNING_SESSION']=='1'
                if not payload_only:
                    for a in (8,32):
                        for c in runner.shortlist(ep,a):
                            assert f"rank:{c['peers']}:{c['workers']}:{c['qps']}" in os.environ['DEEPEP_CLOS_QP_PLANS']
                initialized.append(self)
                self._udma_auto_policy = resolve_udma_policy(True,ep,'udma',None,None,None,None,None).bind_group(g)
                self.runtime = SimpleNamespace(use_1g_huge_pages=lambda:False)
            @property
            def udma_auto_policy(self): return self._udma_auto_policy
            def destroy(self): destroyed.append(self)
        elastic.ElasticBuffer = Buffer
        def run_case(s,shape,winners):
            assert s.buffer is initialized[0] and len(initialized)==1
            if payload_only and payload_only != 'defaults':
                c = runner.policy_config(s.base.for_launch(shape['aiv'],shape['hidden']*2,'combine',bf16=True),ep,True)
                for config in runner.payload_candidates(c,s.args.combine_payload_max_rows,s.args.combine_visits,s.args.combine_wqe_fast):
                    s.activate(config,'combine')
                    assert os.environ[runner.PAYLOAD_ROWS_ENV]==str(config['payload_max_rows'])
                    plan=s.policy.for_launch(shape['aiv'],shape['hidden']*2,'combine',bf16=True)
                    assert (plan.peers_per_worker,plan.workers_per_peer,plan.qps_per_worker,plan.wqes_per_visit)==(config['peers'],config['workers'],config['qps'],config['visit'])
                    assert os.environ[runner.WQE_FAST_ENV]==str(config['combine_wqe_fast'])
            executed.append(shape)
            return []
        runner.run_case = run_case
        runner.write_summary = lambda s:None
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            native = root/'_clos_C.so'; native.write_bytes(b'host')
            (root/'libdeepep_clos_kernels.so').write_bytes(b'kernel')
            (root/'.libs').mkdir(); (root/'.libs/libshmem.so').write_bytes(b'shmem')
            mode = ['--combine-defaults-only','--skip-precision','--single-pass'] if payload_only == 'defaults' else ['--combine-payload-max-rows','1','2','4','8','--combine-visits','8','32','--combine-wqe-fast','0','1'] if payload_only else ['--small-rank-sweep']
            args = entry.parse_args([*mode,'--ep-size',str(ep),'--output',str(root/'result')])
            with patch.object(runner.importlib.util,'find_spec',return_value=SimpleNamespace(origin=str(native))):
                runner.run(args)
            report = json.loads((args.output/'report.json').read_text())
            assert report['buffer_initializations']==1 and not report['huge_pages']
            if payload_only == 'defaults':
                assert report['mode']=='combine_defaults' and report['final_passes']==1
                assert not report['precision_checked'] and report['pre_combine_notify']
                assert not json.loads((args.output/'rank0/exit.json').read_text())['precision_checked']
        assert len(initialized)==1 and destroyed==initialized and executed==args.case
        assert runner.PAYLOAD_ROWS_ENV not in os.environ and runner.WQE_FAST_ENV not in os.environ
    """.replace("TEST_EP", str(ep)).replace("TEST_PAYLOAD", repr(payload_only))
    )
