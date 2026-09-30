# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Offline protocol tests. Synthetic runtimes here are never production backends."""

import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import deep_ep

from benchmarks._backend import BackendUnavailable, resolve_backend
from benchmarks._config import (
    Config,
    hidden_value,
    load_config,
    route_for_token,
    traffic,
    weight_value,
)
from benchmarks._utils import (
    CSV_FIELDS,
    aggregate_samples,
    measure,
    measured_result,
    plan,
    summary_csv,
    write_result,
)
from benchmarks.bench_ep import main
from tests.utils.cases import Case
from tests.utils.reference import dispatch

ROOT = Path(__file__).resolve().parents[1]
METADATA = dict(
    backend="test-only",
    device="test-only",
    software="test-only",
    topology="test-only",
    variant="test-only",
)


@pytest.mark.parametrize(
    "change",
    [
        {"world_size": True},
        {"iterations": 0},
        {"warmup": -1},
        {"hidden": 0},
        {"seed": -1},
        {"experts": 7},
        {"topk": 9},
        {"tokens_per_rank": (1,)},
        {"tokens_per_rank": (1, -1)},
        {"tokens_per_rank": (0, 0)},
        {"tokens_per_rank": (True, 1)},
        {"with_weights": "true"},
        {"alignment": 2},
        {"dtype": "float32"},
        {"api": "unknown"},
        {"operation": "moe-step"},
        {"route": "unknown"},
        {"dtype": "fp8", "hidden": 127},
        {"operation": "combine", "hidden": 128},
        {"operation": "combine", "dispatch_mode": "cached"},
        {"api": "legacy", "layout": "expanded"},
        {"api": "legacy", "layout": "ordinary", "dtype": "fp8"},
        {"tokens_per_rank": (1048576, 1)},
        {
            "iterations": 10000,
            "world_size": 1024,
            "tokens_per_rank": (1,) * 1024,
            "experts": 1024,
        },
        {"hidden_typo": 256},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        Config.from_dict(Config().to_dict() | change)


@pytest.mark.parametrize("filename", ["smoke.json", "upstream_bf16.json", "upstream_fp8.json"])
def test_presets_have_no_measured_metrics(filename):
    config = Config.from_dict(load_config(ROOT / "benchmarks/configs" / filename))
    report = plan(config)
    assert report["status"] == "planned"
    assert "statistics" not in report
    assert "rank_samples_us" not in report
    assert report["correctness_result"] == "not_run"
    assert "bandwidth_gbps" not in json.dumps(report)
    assert "latency_mean_us" not in json.dumps(report)


@pytest.mark.parametrize(
    "text",
    [
        '{"world_size": 2, "world_size": 8}',
        '{"world_size": NaN}',
        "[]",
        '{"seed": Infinity}',
        '{"hidden": "256"}',
        '{"bad": 1}',
    ],
)
def test_bad_json_is_rejected(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        Config.from_dict(load_config(path))


def test_config_size_bound(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(" " * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        load_config(path)


@pytest.mark.parametrize("layout", ["ordinary", "expanded"])
@pytest.mark.parametrize("route", ["balanced", "skew", "duplicate", "masked"])
@pytest.mark.parametrize("operation", ["dispatch", "combine", "roundtrip"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_bytes_match_independent_route_reference(layout, route, operation, dtype):
    cfg = Config(
        layout=layout,
        route=route,
        operation=operation,
        dtype=dtype,
        tokens_per_rank=(3, 0),
        topk=4,
    )
    # Use an independent, hand-readable layout oracle instead of repeating the
    # implementation's destination-set calculation in this test.
    routes = tuple(tuple(route_for_token(cfg, r, t) for t in range(n)) for r, n in enumerate(cfg.tokens_per_rank))
    x = tuple(tuple((float(t),) for t in range(n)) for n in cfg.tokens_per_rank)
    case = Case("byte_check", cfg.experts, 1, cfg.topk, x, routes, None)
    oracle = dispatch(case, expanded=layout == "expanded")
    rows = sum(len(rank) for rank in oracle.ranks)
    forward = rows * cfg.hidden * (1 if dtype == "fp8" else 2)
    backward = rows * cfg.hidden * 2
    expected = {
        "dispatch": forward,
        "combine": backward,
        "roundtrip": forward + backward,
    }[operation]
    assert traffic(cfg)["effective_bytes"] == expected


def test_duplicate_slots_are_not_rank_deduplicated_in_expanded_mode():
    cfg = Config(tokens_per_rank=(1, 0), topk=2, route="duplicate", hidden=256)
    assert traffic(cfg)["effective_bytes"] == 1024
    assert traffic(replace(cfg, layout="ordinary"))["effective_bytes"] == 512
    assert route_for_token(cfg, 0, 0) == (0, 0)
    assert hidden_value(cfg, 0, 0, 0) == -63 / 16
    assert weight_value(cfg, 1, 0, 0) == 19 / 128


def test_rank_max_is_taken_before_percentiles():
    raw = [[100, 1], [1, 100]]
    maximums, result = aggregate_samples(raw, 2, 2)
    assert maximums == [100, 100]
    assert result["latency_mean_us"] == result["latency_p50_us"] == 100
    _, result = aggregate_samples([list(range(1, 21))], 1, 20)
    assert result["latency_p50_us"] == 10
    assert result["latency_p95_us"] == 19


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [[1]],
        [[1], []],
        [[1], [0]],
        [[-1], [1]],
        [[True], [1]],
        [[math.nan], [1]],
        [[math.inf], [1]],
    ],
)
def test_incomplete_or_invalid_samples_rejected(samples):
    with pytest.raises(ValueError):
        aggregate_samples(samples, 2, 1)


class TraceRuntime:
    """Deterministic lifecycle fixture, not a device emulator."""

    rank = 0
    world_size = 2
    metadata = METADATA

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.lease = None
        self.verifications = 0

    def agree_config(self, fingerprint):
        self.calls.append("agree")
        assert len(fingerprint) == 64

    def prepare(self):
        self.calls.append("prepare")

    def verify(self):
        self.calls.append("verify")
        self.verifications += 1
        if self.failure == "verify" or self.failure == "post_verify" and self.verifications == 2:
            raise AssertionError("injected mismatch")

    def all_ranks_ok(self, local_ok):
        self.calls.append("vote")
        return local_ok and self.failure != "peer_verify"

    def barrier(self):
        self.calls.append("barrier")

    def synchronize(self):
        self.calls.append("sync")

    def invoke(self):
        self.calls.append("invoke")
        assert self.lease is None
        if self.failure == "invoke":
            raise RuntimeError("native failure")
        self.lease = object()
        return self.lease

    def wait(self, output):
        assert output is self.lease
        self.calls.append("wait")

    def release(self, output):
        assert output is self.lease
        self.calls.append("release")
        self.lease = None

    def gather_samples(self, samples):
        self.calls.append("gather")
        return [samples, [n * 2 for n in samples]]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.calls.append("close")
        if self.failure == "close":
            raise RuntimeError("teardown failure")


def test_timing_waits_for_output_and_excludes_release_and_collectives():
    runtime = TraceRuntime()
    timestamps = iter([0, 10000, 20000, 40000])

    def clock():
        runtime.calls.append("clock")
        return next(timestamps)

    report = measure(Config(warmup=1, iterations=2), runtime, clock=clock)
    assert report["rank_samples_us"] == [[10.0, 20.0], [20.0, 40.0]]
    assert report["rank_max_samples_us"] == [20.0, 40.0]
    assert runtime.calls == [
        "agree",
        "prepare",
        "verify",
        "sync",
        "vote",
        "barrier",
        "sync",
        "invoke",
        "wait",
        "sync",
        "release",
        "barrier",
        "sync",
        "clock",
        "invoke",
        "wait",
        "sync",
        "clock",
        "release",
        "barrier",
        "sync",
        "clock",
        "invoke",
        "wait",
        "sync",
        "clock",
        "release",
        "verify",
        "sync",
        "vote",
        "gather",
    ]


@pytest.mark.parametrize("failure", ["verify", "post_verify", "peer_verify", "invoke"])
def test_measurement_aborts_on_correctness_or_runtime_error(failure):
    runtime = TraceRuntime(failure)
    with pytest.raises(RuntimeError):
        measure(Config(warmup=0, iterations=1), runtime)
    assert "gather" not in runtime.calls


def test_capability_gate_happens_before_factory(monkeypatch):
    cfg = Config(dtype="fp8")
    descriptor = dict(
        protocol_version=1,
        capabilities=["elastic", "dispatch", "expanded"],
        shape_supported=True,
    )
    native = SimpleNamespace(
        describe_benchmark=lambda _: descriptor,
        create_benchmark=lambda _: pytest.fail("must not create device session"),
    )
    monkeypatch.setattr("benchmarks._backend.import_module", lambda _: native)
    with pytest.raises(BackendUnavailable, match="fp8"):
        resolve_backend(cfg)


@pytest.mark.parametrize("mode", ["--help", "--dry-run", "--list-scenarios", "run"])
def test_cli_does_not_import_framework_or_initialize_devices(mode):
    code = """
import sys
sys.path.insert(0, sys.argv.pop(1))
class RejectFramework:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'numpy', 'mpi4py'):
            raise RuntimeError('Unexpected framework import: ' + fullname)
sys.meta_path.insert(0, RejectFramework())
from benchmarks.bench_ep import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(Path(deep_ep.__file__).resolve().parent.parent)]
        + ([] if mode == "run" else [mode]),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (2 if mode == "run" else 0), result.stderr
    assert "Unexpected framework import" not in result.stderr
    if mode == "run":
        report = json.loads(result.stdout)
        assert report["status"] == "failed"
        assert "adapter is not implemented" in result.stderr
        assert "statistics" not in report


def test_csv_and_raw_report_are_reproducible_and_do_not_overwrite(tmp_path):
    cfg = Config(warmup=0, iterations=2)
    report = measured_result(
        cfg,
        {"rank_samples_us": [[10, 20], [20, 40]]},
        revision="a" * 40,
        metadata=METADATA,
    )
    assert report["statistics"]["bandwidth_gbps"] == traffic(cfg)["effective_bytes"] / 30000
    output = tmp_path / "run"
    write_result(output, report)
    restored = json.loads((output / "result.json").read_text())
    assert restored == report
    assert Config.from_dict(load_config(output / "config.json")) == cfg
    rows = list(csv.DictReader(io.StringIO(summary_csv(report))))
    assert len(rows) == 1
    assert rows[0]["latency_mean_us"] == "30.0"
    assert rows[0]["baseline_latency_us"] == rows[0]["speedup"] == ""
    assert set(rows[0]) == set(CSV_FIELDS)
    with pytest.raises(FileExistsError):
        write_result(output, report)
    assert json.loads((output / "result.json").read_text()) == report


def test_dry_run_directory_has_no_performance_files(tmp_path, capsys):
    output = tmp_path / "plan"
    assert main(["--dry-run", "--output-dir", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {"plan.json", "config.json"}
    assert "statistics" not in json.loads(capsys.readouterr().out)


def test_failed_teardown_does_not_publish_performance(tmp_path, monkeypatch, capsys):
    runtime = TraceRuntime("close")
    monkeypatch.setattr("benchmarks._backend.resolve_backend", lambda _: lambda _: runtime)
    output = tmp_path / "failed"
    assert (
        main(
            [
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--revision",
                "a" * 40,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert runtime.calls[-1] == "close"
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_csv_template_stays_in_sync():
    header = (ROOT / "benchmarks/results/template.csv").read_text().strip()
    assert header.split(",") == CSV_FIELDS


@pytest.mark.parametrize("rank,world", [(True, 2), (-1, 2), (2, 2), (0, True), (0, 1)])
def test_bad_runtime_identity_fails_before_prepare(rank, world):
    runtime = TraceRuntime()
    runtime.rank, runtime.world_size = rank, world
    with pytest.raises(ValueError, match="rank/world"):
        measure(Config(), runtime)
    assert not runtime.calls


@pytest.mark.parametrize("world,iterations", [(0, 1), (1, 0), (True, 1), (1, True)])
def test_aggregation_rejects_empty_dimensions(world, iterations):
    with pytest.raises(ValueError, match="positive integers"):
        aggregate_samples([], world, iterations)


@pytest.mark.parametrize(
    "update",
    [
        {"protocol_version": True},
        {"protocol_version": 2},
        {"capabilities": "all"},
        {"capabilities": [True]},
        {"shape_supported": 1},
        {"shape_supported": False},
    ],
)
def test_invalid_backend_descriptor_is_rejected(monkeypatch, update):
    cfg = Config()
    descriptor = (
        dict(
            protocol_version=1,
            capabilities=sorted(cfg.required_capabilities()),
            shape_supported=True,
        )
        | update
    )
    native = SimpleNamespace(
        describe_benchmark=lambda _: descriptor,
        create_benchmark=lambda _: pytest.fail("must not initialize"),
    )
    monkeypatch.setattr("benchmarks._backend.import_module", lambda _: native)
    with pytest.raises(BackendUnavailable):
        resolve_backend(cfg)


def test_complete_capabilities_only_resolve_factory_without_calling_it(monkeypatch):
    cfg = Config(
        operation="roundtrip",
        dtype="fp8",
        dispatch_mode="cached",
        alignment=128,
        tokens_per_rank=(16, 0),
    )
    expected = {
        "elastic",
        "dispatch",
        "combine",
        "roundtrip",
        "fp8",
        "expanded",
        "cached",
        "alignment128",
        "weights",
        "unequal_tokens",
        "zero_tokens",
        "route_balanced",
    }
    assert cfg.required_capabilities() == expected
    assert "single_rank" in Config(world_size=1, tokens_per_rank=(16,)).required_capabilities()

    def factory(_):
        pytest.fail("resolution must not initialize")

    native = SimpleNamespace(
        describe_benchmark=lambda _: dict(protocol_version=1, capabilities=sorted(expected), shape_supported=True),
        create_benchmark=factory,
    )
    monkeypatch.setattr("benchmarks._backend.import_module", lambda _: native)
    assert resolve_backend(cfg) is factory


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        METADATA | {"hostname": "unexpected"},
        METADATA | {"device": ""},
        METADATA | {"device": "x" * 257},
        METADATA | {"device": "line\nbreak"},
        METADATA | {"device": "=1+1"},
        METADATA | {"topology": "  @formula"},
        METADATA | {"variant": 1},
    ],
)
def test_invalid_or_unsafe_metadata_is_rejected(metadata):
    with pytest.raises(ValueError, match="metadata"):
        measured_result(
            Config(iterations=1),
            {"rank_samples_us": [[1], [2]]},
            revision="a" * 40,
            metadata=metadata,
        )


@pytest.mark.parametrize("revision", [None, "abc", "A" * 40, "z" * 40])
def test_revision_is_a_full_sha(revision):
    with pytest.raises(ValueError, match="revision"):
        measured_result(
            Config(iterations=1),
            {"rank_samples_us": [[1], [2]]},
            revision=revision,
            metadata=METADATA,
        )


def test_result_recomputes_statistics_instead_of_trusting_summary():
    report = measured_result(
        Config(iterations=1),
        {"rank_samples_us": [[10], [20]], "statistics": {"latency_mean_us": 1}},
        revision="a" * 40,
        metadata=METADATA,
    )
    assert report["statistics"]["latency_mean_us"] == 20
    assert report["rank_max_samples_us"] == [20]


@pytest.mark.parametrize("rank", [0, 1])
def test_cli_success_publishes_only_after_teardown_on_rank_zero(tmp_path, monkeypatch, capsys, rank):
    runtime = TraceRuntime()
    runtime.rank = rank
    monkeypatch.setattr("benchmarks._backend.resolve_backend", lambda _: lambda _: runtime)
    from benchmarks import bench_ep

    original_write = bench_ep.write_result

    def write_after_exit(directory, report):
        assert runtime.calls[-1] == "close"
        original_write(directory, report)

    monkeypatch.setattr(bench_ep, "write_result", write_after_exit)
    output = tmp_path / "protocol-only"
    assert (
        main(
            [
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--revision",
                "a" * 40,
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    if rank == 0:
        report = json.loads(printed)
        assert report["correctness_result"] == "passed_pre_and_post"
        assert {p.name for p in output.iterdir()} == {
            "config.json",
            "result.json",
            "summary.csv",
        }
    else:
        assert not printed and not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "agree_config",
        "prepare",
        "barrier",
        "synchronize",
        "wait",
        "release",
        "gather_samples",
    ],
)
def test_cli_runtime_failures_always_close_without_publishing(tmp_path, monkeypatch, capsys, failure):
    runtime = TraceRuntime()

    def fail(*args):
        raise RuntimeError("injected private diagnostic")

    setattr(runtime, failure, fail)
    monkeypatch.setattr("benchmarks._backend.resolve_backend", lambda _: lambda _: runtime)
    output = tmp_path / "failure"
    assert (
        main(
            [
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--revision",
                "a" * 40,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert runtime.calls[-1] == "close"
    assert not output.exists()
    captured = capsys.readouterr()
    assert "injected private diagnostic" not in captured.out
    assert json.loads(captured.out)["status"] == "failed"


@pytest.mark.parametrize(
    "argv",
    [
        ["--world-size", "3"],
        ["--tokens-per-rank", "1,,2"],
        ["--operation", "unknown"],
        ["--with-weights", "false"],
        ["--config", "not-used.json", "--preset", "smoke"],
    ],
)
def test_cli_rejects_bad_arguments_without_performance(argv):
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench_ep", "--dry-run", *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert '"statistics"' not in result.stdout


def test_cli_overrides_config_and_roundtrips_fingerprint(tmp_path, capsys):
    config = tmp_path / "input.json"
    config.write_text(json.dumps(dict(operation="roundtrip", dtype="fp8", layout="ordinary")))
    assert (
        main(
            [
                "--dry-run",
                "--config",
                str(config),
                "--layout",
                "expanded",
                "--no-with-weights",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    cfg = Config.from_dict(report["config"])
    assert cfg.layout == "expanded" and cfg.dtype == "fp8" and not cfg.with_weights
    assert report["config_fingerprint"] == cfg.fingerprint()


def test_zero_logical_traffic_is_distinct_from_unmeasured_latency():
    cfg = Config(tokens_per_rank=(1, 0), topk=1, route="masked", warmup=0, iterations=1)
    assert traffic(cfg)["effective_bytes"] == 0
    result = measured_result(cfg, {"rank_samples_us": [[1], [2]]}, revision="a" * 40, metadata=METADATA)
    assert result["statistics"]["bandwidth_gbps"] == 0
    assert result["statistics"]["latency_mean_us"] == 2
