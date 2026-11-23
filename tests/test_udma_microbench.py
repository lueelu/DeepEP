"""Offline traffic/accounting checks; no simulated bandwidth results."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "udma_bench", Path(__file__).resolve().parents[1] / "benchmarks/udma/bench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_default_suite_and_capacity():
    args = bench.parser().parse_args(["--ep-size", "32"])
    suite = bench.cases(args)
    assert len(suite) == (62 + 31 + 1) * 2
    assert {c["chunk"] for c in suite} == {5632, 262144}
    assert {c["bytes"] for c in suite} == {16 * 1024**2}


def test_exhaustive_directed_pairs_and_permutation():
    args = bench.parser().parse_args(["--ep-size", "4", "--pair-scope", "all", "--chunk-bytes", "32"])
    suite = bench.cases(args)
    pair_edges = [edge for c in suite if c["mode"] == "pair" for edge in bench.edges(c, 4)]
    assert set(pair_edges) == {(s, d) for s in range(4) for d in range(4) if s != d}
    assert len(pair_edges) == 12
    for c in suite:
        if c["mode"] == "permutation":
            graph = bench.edges(c, 4)
            assert len({d for _, d in graph}) == 4
            assert all(s != d for s, d in graph)


@pytest.mark.parametrize(
    "argv",
    [
        ["--bytes-per-peer", "33"],
        ["--chunk-bytes", "0"],
        ["--chunk-bytes", "2097152"],
        ["--bytes-per-peer", "32", "32"],
        ["--aiv", "1"],
        ["--ep-size", "257"],
        ["--iterations", "0"],
        ["--warmup", "-1"],
        ["--deltas", "0"],
        ["--deltas", "1", "1"],
    ],
)
def test_bad_config_rejected(argv):
    with pytest.raises(ValueError):
        bench.cases(bench.parser().parse_args(argv))


def test_metrics_use_one_device_clock_and_include_local_join():
    raw = dict(
        ticks=[1000, 2000, 5000, 6000, 8000, 10000, 1, 0, 1500, 1500, 1500, 1500, 9000, 11000, 0, 0], mismatched_words=0
    )
    result = bench.sample_metrics(raw)
    assert result["elapsed_us"] == 10
    assert result["post_us"] == 3
    assert result["incoming_us"] == 7.5
    assert result["local_join_us"] == 2
    shifted = dict(raw, ticks=[v + 10000000 if i % 8 < 6 else v for i, v in enumerate(raw["ticks"])])
    assert bench.sample_metrics(shifted) == result


def test_per_iteration_max_not_max_of_rank_means():
    case = dict(mode="alltoall", bytes=1000000, chunk=32, source=0, destination=0, delta=0)
    ranks = [
        dict(rank=0, samples=[dict(elapsed_us=10, mismatched_words=0), dict(elapsed_us=30, mismatched_words=0)]),
        dict(rank=1, samples=[dict(elapsed_us=30, mismatched_words=0), dict(elapsed_us=10, mismatched_words=0)]),
    ]
    result = bench.summarize(case, 2, ranks)
    assert result["mean_us"] == 30
    assert result["total_remote_bytes"] == 2000000
    assert result["per_sender_GBps"] == pytest.approx(1000000 / 30 / 1000)


def test_idle_pair_ranks_do_not_dilute_link_bandwidth():
    case = dict(mode="pair", bytes=1000000, chunk=32, source=1, destination=2, delta=0)
    ranks = [dict(rank=r, samples=[dict(elapsed_us=20 if r in (1, 2) else 900, mismatched_words=0)]) for r in range(4)]
    result = bench.summarize(case, 4, ranks)
    assert result["mean_us"] == 20
    assert result["aggregate_GBps"] == result["per_sender_GBps"] == 50
    assert result["directed_edges"] == 1
    ranks[2]["samples"][0]["mismatched_words"] = 1
    assert bench.summarize(case, 4, ranks)["mismatched_words"] == 1


def test_invalid_timings_and_missing_ranks_rejected():
    with pytest.raises(ValueError):
        bench.sample_metrics(dict(ticks=[0] * 8, mismatched_words=0))
    with pytest.raises(ValueError):
        bench.sample_metrics(dict(ticks=[1, 2, 1, 4, 5, 6, 1, 0], mismatched_words=0))
    with pytest.raises(ValueError):
        bench.summarize({}, 4, [dict(rank=0, samples=[])])


def test_runner_reuses_one_session_and_writes_all_cases(monkeypatch, tmp_path):
    """Exercise lifecycle and output wiring with an explicitly offline runtime."""
    calls = []

    class FakeSession:
        def __init__(self, *args):
            calls.append(("init", args))

        def sample(self, mode, src, dst, delta, size, chunk, epoch, corrupt=False):
            calls.append(("sample", epoch))
            return dict(ticks=[1000, 2000, 3000, 4000, 5000, 6000, 1, 0], epoch=epoch, mismatched_words=int(corrupt))

        def close(self):
            calls.append(("close",))

    def gather(out, value):
        out[:] = [value, value]

    def gather_samples(value, out, dst):
        out[:] = [dict(value, rank=r) for r in range(2)]

    dist = SimpleNamespace(
        init_process_group=lambda *a, **k: None,
        all_gather_object=gather,
        broadcast_object_list=lambda *a, **k: None,
        gather_object=gather_samples,
        barrier=lambda: None,
        destroy_process_group=lambda: None,
        finish=lambda: None,
    )
    torch = SimpleNamespace(
        distributed=dist,
        __version__="offline-test",
        npu=SimpleNamespace(set_device=lambda d: None, get_device_name=lambda d: "offline-test"),
    )
    native = SimpleNamespace(Session=FakeSession, unique_id=lambda: [0], __file__=__file__)
    for name, module in (
        ("torch", torch),
        ("torch.distributed", dist),
        ("torch_npu", SimpleNamespace(__version__="offline-test")),
        ("_udma_microbench", native),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in (("WORLD_SIZE", "2"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(bench, "fingerprint", lambda p: "offline-test")
    monkeypatch.setattr(bench, "runtime_libraries", lambda: [])
    monkeypatch.setattr(bench, "Control", lambda *args: dist)
    bench.main(
        [
            "--ep-size",
            "2",
            "--bytes-per-peer",
            "32",
            "--chunk-bytes",
            "32",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--check-validator",
            "--output",
            str(tmp_path),
        ]
    )
    assert sum(c[0] == "init" for c in calls) == 1
    assert sum(c[0] == "close" for c in calls) == 1
    assert [c[1] for c in calls if c[0] == "sample"] == [1, 2, 3, 4, 5]
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["summaries"]) == 4
    assert report["precision_passed"] and report["destroyed"]
    assert len((tmp_path / "rank0.jsonl").read_text().splitlines()) == 5


def test_control_rounds_and_shutdown_with_staggered_clients():
    class Store:
        def __init__(self):
            self.values = {}
            self.changed = threading.Condition()

        def set(self, key, value):
            with self.changed:
                self.values[key] = value
                self.changed.notify_all()

        def get(self, key):
            with self.changed:
                assert self.changed.wait_for(lambda: key in self.values, timeout=5)
                return self.values[key]

        def wait(self, keys):
            for key in keys:
                self.get(key)

        def delete_key(self, key):
            with self.changed:
                del self.values[key]

    store = Store()

    def run(rank):
        control = bench.Control.__new__(bench.Control)
        control.rank, control.world, control.round, control.store = rank, 4, 0, store
        values = []
        control.all_gather_object(values, rank)
        assert values == [0, 1, 2, 3]
        uid = ["uid"] if rank == 0 else [None]
        control.broadcast_object_list(uid, src=0)
        assert uid == ["uid"]
        for iteration in range(8):
            result = [] if rank == 0 else None
            control.gather_object(dict(rank=rank, iteration=iteration), result, dst=0)
            if rank == 0:
                assert result == [dict(rank=r, iteration=iteration) for r in range(4)]
        control.barrier()
        control.finish()

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, range(4)))
    assert sum(key.startswith("udma/closed/") for key in store.values) == 4
