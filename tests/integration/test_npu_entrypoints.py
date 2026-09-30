# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Device-free contract tests; these are not NPU execution evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from .npu_backend import NativeV1, REQUIRED_OPERATIONS, installed_native, require_tensor, wait_completion
from .workflow import smoke_cases


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    if not path.is_file():
        pytest.skip("Launcher source tests execute before package staging")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher():
    return load_script("validate_ep")


@pytest.mark.parametrize("value", (None, "", "0", "0,0", "0,1,2", "-1,0", "0;exit,1", "0,256", "x,1"))
def test_devices_are_explicit_bounded_and_distinct(launcher, value):
    with pytest.raises(ValueError):
        launcher.validate_devices(value)


def test_valid_devices(launcher):
    assert launcher.validate_devices("2,3") == "2,3"


def rank_report(rank, backend="npu-stub", name="mixed_k1"):
    return {
        "schema_version": 1,
        "rank": rank,
        "world_size": 2,
        "backend": backend,
        "passed": True,
        "npu_tensor_roundtrip_tested": True,
        "native_ep_verified": backend == "native",
        "wheel_api_verified": backend == "native",
        "device_reference_captured": False,
        "second_reference": "cpu-bf16-fp32-path-simulation",
        "build_identity_sha256": "a" * 64,
        "data_transport": "installed-deepep" if backend == "native" else "hccl-reference-collectives",
        "results": [
            {
                "name": name,
                "passed": True,
                "npu_tensors_checked": True,
                "dispatch_checked": True,
                "precision": {"passed": True},
            }
        ],
    }


def save_reports(path, reports):
    for rank, report in enumerate(reports):
        (path / f"rank-{rank}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("backend", ("native", "npu-stub"))
def test_both_rank_reports_required(launcher, tmp_path, backend):
    save_reports(tmp_path, [rank_report(0, backend)])
    with pytest.raises(FileNotFoundError):
        launcher.read_results(tmp_path, backend=backend, names=("mixed_k1",))
    save_reports(tmp_path, [rank_report(rank, backend) for rank in range(2)])
    assert len(launcher.read_results(tmp_path, backend=backend, names=("mixed_k1",))) == 2


@pytest.mark.parametrize(
    "fault",
    (
        "rank",
        "backend",
        "passed",
        "native_ep_verified",
        "npu_tensor_roundtrip_tested",
        "cases",
        "precision",
        "build_identity",
        "second_iteration",
    ),
)
def test_bad_or_incomplete_reports_never_pass(launcher, tmp_path, fault):
    reports = [rank_report(rank, "native") for rank in range(2)]
    names = ("mixed_k1",)
    bad = reports[1]
    if fault == "cases":
        bad["results"] = []
    elif fault == "precision":
        bad["results"][0]["precision"]["passed"] = False
    elif fault == "build_identity":
        bad["build_identity_sha256"] = "b" * 64
    elif fault == "second_iteration":
        names = ("fresh_iterations",)
        for report in reports:
            report["results"][0]["name"] = "fresh_iterations"
    else:
        bad[fault] = "wrong"
    save_reports(tmp_path, reports)
    with pytest.raises(RuntimeError):
        launcher.read_results(tmp_path, backend="native", names=names)


def test_cpu_tensor_is_rejected():
    class Tensor:
        device = "cpu"
        dtype = "bf16"

    torch = SimpleNamespace(Tensor=Tensor)
    with pytest.raises(AssertionError, match="Expected"):
        require_tensor(torch, Tensor(), device="npu:0", dtype="bf16")


@pytest.mark.parametrize("event", (None, SimpleNamespace(event=None), SimpleNamespace(event=object())))
def test_async_completion_cannot_be_omitted(event):
    with pytest.raises(AssertionError, match="recorded"):
        wait_completion(event)


def test_native_adapter_calls_public_api_and_waits():
    calls = []
    event = SimpleNamespace(event=object(), current_stream_wait=lambda: calls.append("wait"))

    class Config:
        def get_scaleup_buffer_size_hint(self, hidden_bytes, ranks):
            return hidden_bytes * ranks

        def get_scaleout_buffer_size_hint(self, hidden_bytes, ranks):
            return 0

    class Buffer:
        get_dispatch_config = get_combine_config = staticmethod(lambda ranks: Config())

        def __init__(self, group, *, num_scaleup_bytes, num_scaleout_bytes, explicitly_destroy):
            calls.append(("create", group, num_scaleup_bytes, num_scaleout_bytes, explicitly_destroy))

        def get_dispatch_layout(self, idx, experts, **kwargs):
            calls.append(("layout", kwargs))
            return "per_rank", "per_scaleout", "per_expert", "mask", event

        def dispatch(self, x, **kwargs):
            calls.append(("dispatch", kwargs))
            return "x", "idx", "weights", [1], "handle", event

        def combine(self, x, handle, **kwargs):
            calls.append(("combine", kwargs))
            return "out", "values", event

        def destroy(self):
            calls.append("destroy")

    case = smoke_cases()[0]
    adapter = NativeV1(SimpleNamespace(Buffer=Buffer), "group", case)
    assert calls[0] == ("create", "group", case.hidden * 2 * 2, 0, True)
    result = adapter.dispatch("input", "indices", "gates", 8)
    assert result.handle == "handle" and result.counts == [1]
    assert calls[1] == ("layout", {"async_finish": True}) and calls[2] == "wait"
    assert calls[3] == (
        "dispatch",
        {
            "topk_idx": "indices",
            "topk_weights": "gates",
            "num_tokens_per_rank": "per_rank",
            "num_tokens_per_scaleout_rank": "per_scaleout",
            "num_tokens_per_expert": "per_expert",
            "is_token_in_rank": "mask",
            "async_finish": True,
        },
    )
    adapter.combine("payload", result.handle, "values")
    assert calls[-1] == ("combine", {"topk_weights": "values", "async_finish": True})
    adapter.destroy()
    assert calls[-1] == "destroy"


@pytest.mark.parametrize(
    "installed, operations, expected",
    ((False, [], RuntimeError), (True, [], NotImplementedError), (True, list(REQUIRED_OPERATIONS), None)),
)
def test_installed_native_preflight(monkeypatch, tmp_path, installed, operations, expected):
    site = tmp_path / "site"
    package = (site if installed else tmp_path / "checkout") / "deep_ep"
    package.mkdir(parents=True)
    (package / "_build_info.json").write_text(json.dumps({"communication_operations": operations}), encoding="utf-8")
    module = SimpleNamespace(__file__=str(package / "__init__.py"))
    monkeypatch.setitem(sys.modules, "deep_ep", module)
    monkeypatch.setitem(
        sys.modules,
        "deep_ep._native",
        SimpleNamespace(require_native=lambda name: SimpleNamespace(__file__=str(package / "_C.so"))),
    )
    monkeypatch.setattr("sysconfig.get_path", lambda name: str(site))
    if expected:
        with pytest.raises(expected):
            installed_native()
    else:
        assert installed_native()[0] is module


def test_launcher_command_is_bounded_and_explicit(launcher, tmp_path):
    command = launcher.worker_command("python", "native", "mixed_k1", tmp_path)
    assert "--nproc-per-node=2" in command and "--max-restarts=0" in command
    assert command[command.index("--backend") + 1] == "native"
    assert "functional-stub" not in command
    assert tuple(launcher.CASE_NAMES) == tuple(case.name for case in smoke_cases())


@pytest.mark.parametrize("backend, expected_jobs", (("npu-stub", 1), ("native", 6)))
def test_outer_launcher_runs_every_case_once_without_install(monkeypatch, launcher, tmp_path, backend, expected_jobs):
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:19091")
    jobs = []

    def fake_launch(command, *, cwd, env, timeout):
        jobs.append(command)
        assert "pip" not in command and "PYTHONPATH" not in env
        assert not (cwd / "deep_ep").exists()
        name = command[command.index("--case") + 1]
        names = launcher.CASE_NAMES if name == "all" else (name,)
        directory = Path(command[-1])
        reports = []
        for rank in range(2):
            report = rank_report(rank, backend)
            report["results"] = [rank_report(rank, backend, case)["results"][0] for case in names]
            for record in report["results"]:
                if record["name"] == "fresh_iterations":
                    record["second_iteration"] = copy.deepcopy(record)
            reports.append(report)
        save_reports(directory, reports)

    monkeypatch.setattr(launcher, "launch", fake_launch)
    output = tmp_path / "results"
    launcher.main(["--backend", backend, "--devices", "0,1", "--output", str(output)])
    assert len(jobs) == expected_jobs
    report = json.loads((output / "summary.json").read_text())
    assert report["native_ep_verified"] is (backend == "native")


@pytest.mark.parametrize(
    "operations, accepted",
    (
        ([], True),
        (list(REQUIRED_OPERATIONS), True),
        # The Elastic operations emitted by CMakeLists.txt into capabilities.json.
        (["ElasticBuffer.dispatch", "ElasticBuffer.combine"], True),
        (["unknown"], False),
        (["Buffer.dispatch"] * 2, False),
        (None, False),
    ),
)
def test_wheel_capability_schema(tmp_path, operations, accepted):
    validator = load_script("validate_package")
    path = tmp_path / "test.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("deep_ep/_C.fake.so", b"fixture")
        archive.writestr("deep_ep/_build_info.json", json.dumps({"communication_operations": operations}))
        archive.writestr("licenses/deepseek-deepep.LICENSE", "fixture")
    if accepted:
        assert validator.inspect_wheel(path)["communication_operations"] == operations
    else:
        with pytest.raises(RuntimeError, match="communication_operations"):
            validator.inspect_wheel(path)


def test_top_level_option_is_discoverable_without_framework():
    for script, option in (("validate_package", "--ep-roundtrip"), ("validate_ep", "--backend")):
        if not (ROOT / "scripts" / (script + ".py")).is_file():
            pytest.skip("Launcher source test")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / (script + ".py")), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert option in result.stdout and "npu-stub" in result.stdout and "native" in result.stdout
