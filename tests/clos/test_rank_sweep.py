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


@pytest.mark.parametrize("ep", [4, 8])
def test_matrix_policy_and_qp_reservation(ep):
    from tests.api.test_clos_compatibility import run_host

    run_host(f"""
        sys.path.insert(0, 'benchmarks/clos')
        from rank_sweep import cases, shortlist, SweepPolicy, policy_config, OPS, op_kind, VISITS
        from bench_fast_sweep import qp_plan_spec
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        from dataclasses import asdict
        base = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
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
                    before, after = asdict(original), asdict(plan)
                    for field in ('peer_order','peers_per_worker','workers_per_peer','qps_per_worker','wqes_per_visit'):
                        before.pop(field); after.pop(field)
                    assert before == after
                    configs.append(c)
                wrapper.config = policy_config(original, {ep}, True)
                assert wrapper.for_launch(a, h*2, op_kind(op), bf16=True) == original
        spec = qp_plan_spec([dict(operations={{'all':configs}})])
        assert len(spec.split(';')) <= 4
        assert all(v.startswith('rank:') for v in spec.split(';'))
    """)


def test_cli_plan_and_rejected_selectors(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(BENCH))
    import bench_fast_sweep as entry

    args = entry.parse_args(["--small-rank-sweep", "--ep-size", "8", "--plan-only"])
    assert args.seed == 20261122 and not args.huge_pages and args.reduce_cores == 48
    sweep.run(args)
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["cases"]) == 12 and len(plan["operators"]) == 6
    for extra in (
        ["--ep-size", "16"],
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
    result = sweep.slowest([{"api": [10, 100], "comm": [1, 9]}, {"api": [100, 10], "comm": [9, 1]}])
    assert sweep.stats(result["api"])["mean"] == 100
    assert sweep.stats(result["comm"])["p95"] == 9
    with pytest.raises(ValueError):
        sweep.slowest([{"comm": [1]}, {"comm": [1, 2]}])


def test_finalists_keep_api_and_communication_winners():
    rows = [
        dict(config={"id": name, "auto": name == "auto"}, metrics={"api": {"mean": api}, "comm": {"mean": comm}})
        for name, api, comm in [("auto", 1, 1), ("api", 2, 4), ("comm", 4, 2), ("middle", 3, 3)]
    ]
    assert [c["id"] for c in sweep.finalists(rows, 2)] == ["api", "comm", "auto"]


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


def test_measure_keeps_profiler_notify_outside_full_api(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH))
    calls, files = [], {}
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
        native=None,
        out=tmp_path,
        args=SimpleNamespace(warmup=1, reduce_cores=48),
        rank=0,
        group=SimpleNamespace(agree=lambda x: None, gather=lambda x: [x, x], size=lambda: 2),
        write=lambda p, v: files.__setitem__(p.name, v),
        activate=lambda c, o: None,
        prepare=lambda: None,
        done=lambda x: x,
        policy=SimpleNamespace(for_launch=lambda *a, **kw: None),
    )

    def invoke():
        calls.append(os.environ["DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY"])

    rows = sweep.measure(
        session, {"tokens": 4096, "hidden": 2816, "aiv": 8}, "combine", configs, invoke, 15000, "test", 2
    )
    assert calls == ["0"] * 8 + ["1"] * 14
    assert "DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY" not in os.environ
    assert rows[0]["GBps"] == 1
    assert len(files["rank_samples.json"]["stages"][0]["api"]) == 2
    assert not rows[0]["precision_checked"]


def test_session_reserves_once_and_reuses_buffer_for_all_shapes():
    from tests.api.test_clos_compatibility import run_host

    run_host("""
        import os, tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0, 'benchmarks/clos')
        import rank_sweep as runner
        import bench_fast_sweep as entry
        import stage_metrics
        from deep_ep.backends.direct import elastic, _native
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        os.environ.update(RANK='0',WORLD_SIZE='4',LOCAL_RANK='0',MASTER_ADDR='localhost')
        for key in list(os.environ):
            if key.startswith('DEEPEP_CLOS_'): del os.environ[key]
        dist = ModuleType('torch.distributed')
        dist.TCPStore = lambda *a,**k:None
        sys.modules['torch.distributed'] = torch.distributed = dist
        sys.modules['torch_npu'] = ModuleType('torch_npu')
        torch.npu = SimpleNamespace(set_device=lambda n:None,synchronize=lambda:None,empty_cache=lambda:None)
        group = SimpleNamespace(size=lambda:4,rank=lambda:0,gather=lambda x:[x]*4,agree=lambda x:None)
        entry.StoreGroup = lambda *a:group
        stage_metrics.L2CacheFlush = lambda *a:None
        stage_metrics.prepare_v2_sample = lambda *a:None
        _native.load_native_module = lambda:SimpleNamespace(**{name:lambda:None for name in
            ['last_dispatch_channel_metadata','last_dispatch_metadata_options','last_combine_launch',
             'last_dispatch_launch','last_dispatch_overlap_options']})
        initialized, executed, destroyed = [], [], []
        class Buffer:
            get_buffer_size_hint = staticmethod(lambda g,t,h,k: t*h*k)
            def __init__(self,g,**kw):
                assert not kw['use_1g_huge_pages'] and kw['num_max_tokens_per_rank']==8192
                assert os.environ['DEEPEP_CLOS_TUNING_SESSION']=='1'
                assert 'rank:1:2:1' in os.environ['DEEPEP_CLOS_QP_PLANS']
                initialized.append(self)
                self._udma_auto_policy = resolve_udma_policy(True,4,'udma',None,None,None,None,None).bind_group(g)
                self.runtime = SimpleNamespace(use_1g_huge_pages=lambda:False)
            @property
            def udma_auto_policy(self): return self._udma_auto_policy
            def destroy(self): destroyed.append(self)
        elastic.ElasticBuffer = Buffer
        def run_case(s,shape,winners):
            assert s.buffer is initialized[0] and len(initialized)==1
            executed.append(shape)
            return []
        runner.run_case = run_case
        runner.write_summary = lambda s:None
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            native = root/'_clos_C.so'; native.write_bytes(b'host')
            (root/'libdeepep_clos_kernels.so').write_bytes(b'kernel')
            (root/'.libs').mkdir(); (root/'.libs/libshmem.so').write_bytes(b'shmem')
            args = entry.parse_args(['--small-rank-sweep','--ep-size','4','--output',str(root/'result')])
            with patch.object(runner.importlib.util,'find_spec',return_value=SimpleNamespace(origin=str(native))):
                runner.run(args)
            report = json.loads((args.output/'report.json').read_text())
            assert report['buffer_initializations']==1 and not report['huge_pages']
        assert len(initialized)==1 and destroyed==initialized and executed==runner.cases()
    """)
