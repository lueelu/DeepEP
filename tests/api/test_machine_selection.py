"""Backend selection and keyword translation, without loading an NPU runtime."""

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from deep_ep import _machine
from deep_ep._compat import keyword_aliases


def topology(kind="clos"):
    levels = [{"net_type": "CLOS", "rank_addr_list": [{"addr_type": "EID", "addr": "00" * 16, "ports": ["0/0"]}]}]
    edges = [{"local_a": str(i), "local_a_ports": ["0/0"], "topo_type": "CLOS"} for i in range(16)]
    if kind == "mesh_clos":
        levels.append(dict(levels[0], net_type="MESH"))
        edges.append({"local_a": "8", "local_b": "9", "local_a_ports": ["0/1"]})
    root = {"rank_list": [{"device_id": 0, "local_id": 8, "level_list": levels}]}
    return root, {"edge_list": edges}


def test_clos_sixteen_does_not_assume_eight_card_servers():
    root, topo = topology()
    assert _machine.classify_root(root, topo, 0) == "global_clos"
    assert _machine.classify_root(*topology("mesh_clos"), 0) == "mesh_and_clos"
    root["rank_list"][0]["level_list"][0]["net_type"] = "TOPO_FILE_DESC"
    assert _machine.classify_root(root, topo, 0) == "global_clos"


@pytest.mark.parametrize("fault", ["device", "duplicate", "empty", "eid", "mismatch", "pcie"])
def test_incomplete_or_ambiguous_evidence_is_rejected(fault):
    root, topo = topology()
    rank = root["rank_list"][0]
    if fault == "device":
        rank["device_id"] = 4
    elif fault == "duplicate":
        root["rank_list"].append(copy.deepcopy(rank))
    elif fault == "empty":
        rank["level_list"] = []
    elif fault == "eid":
        rank["level_list"][0]["rank_addr_list"][0]["addr"] = "invalid"
    elif fault == "mismatch":
        rank["level_list"][0]["net_type"] = "MESH"
    else:
        topo["edge_list"] = [{"local_a": "8", "local_a_ports": ["PCIe"]}]
    with pytest.raises(ValueError):
        _machine.classify_root(root, topo, 0)


def test_override_needs_no_runtime_or_rootinfo(monkeypatch):
    monkeypatch.setattr(_machine, "detect", lambda _: pytest.fail("unexpected detection"))
    assert _machine.select("global_clos")["source"] == "explicit"
    assert _machine.select("mesh_clos")["source"] == "explicit"


@pytest.mark.parametrize("old,new", [("pod", "mesh_clos"), ("clos_only", "global_clos"), ("card_4pmesh", "mesh_pcie")])
def test_legacy_machine_names_are_input_only_aliases(monkeypatch, old, new):
    monkeypatch.setattr(_machine, "detect", lambda _: pytest.fail("explicit alias must not probe"))
    assert _machine.select(old) == _machine.select(new) == {"machine_type": new, "source": "explicit"}
    assert old not in _machine.KINDS


def test_direct_is_implementation_not_machine_type():
    with pytest.raises(RuntimeError, match="machine_type must be one of"):
        _machine.select("direct")


def test_consensus_rejects_mixed_backends_and_local_errors(monkeypatch):
    pytest.importorskip("torch.distributed")

    class Group:
        def allgather(self, record):
            return [record, {"selection": {"machine_type": "global_clos"}, "error": None}]

    monkeypatch.setitem(sys.modules, "mpi4py.MPI", SimpleNamespace(Comm=Group))
    with pytest.raises(ValueError, match="different"):
        _machine.select("mesh_clos", Group())
    monkeypatch.setattr(_machine, "detect", lambda _: (_ for _ in ()).throw(ValueError("bad topology")))
    with pytest.raises(RuntimeError, match="collectively"):
        _machine.select("auto", Group())


def test_torch_process_group_uses_object_collective(monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    group = dist.ProcessGroup(0, 1)
    assert hasattr(group, "allgather")  # The real type that exposed the MPI misclassification.
    calls = []

    def gather(output, record, *, group):
        calls.append(group)
        output[:] = [record]

    monkeypatch.setattr(dist, "get_world_size", lambda actual: 1 if actual is group else pytest.fail("wrong group"))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert _machine.select("global_clos", group)["machine_type"] == "global_clos"
    assert calls == [group]


def test_unrelated_allgather_object_is_rejected(monkeypatch):
    pytest.importorskip("torch.distributed")
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)

    class NotACommunicator:
        def allgather(self, record):
            pytest.fail("must not call an unrelated allgather method")

    with pytest.raises(TypeError, match="ProcessGroup or MPI"):
        _machine.select("mesh_clos", NotACommunicator())
    assert "mpi4py.MPI" not in sys.modules


def test_driver_timeout_is_not_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(_machine.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        _machine.board_id(0)
    assert len(calls) == 1 and calls[0]["timeout"] == 5


def test_unknown_board_requires_override(monkeypatch, tmp_path):
    monkeypatch.setattr(_machine, "physical_device", lambda: 0)
    monkeypatch.setattr(_machine, "board_id", lambda _: 0x46)
    # Explicit absent files are errors, never silently replaced by board guesses.
    with pytest.raises(RuntimeError, match="explicit rootinfo"):
        _machine.select("auto", rootinfo_path=tmp_path / "absent")


def test_aliases_preserve_defaults_and_reject_conflicts():
    def original(a=20, *, flag=False):
        return a, flag

    call = keyword_aliases(original, {"num_sms": "a", "num_aiv": "a"})
    assert call(num_sms=32) == (32, False)
    assert call(32, num_aiv=32, flag=True) == (32, True)
    with pytest.raises(TypeError, match="conflicting"):
        call(a=20, num_sms=32)
    with pytest.raises(TypeError, match="conflicting"):
        call(num_sms=32, num_aiv=64)
    with pytest.raises(TypeError, match="unexpected"):
        call(unknown=True)
    assert inspect.signature(call) == inspect.signature(original)


def test_backend_is_selected_once_and_constructor_defaults_are_preserved(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "require_native", lambda _: None, raising=False)
    calls = []
    monkeypatch.setattr(
        _machine, "select", lambda *args, **kw: calls.append((args, kw)) or {"machine_type": "mesh_clos"}
    )

    class ElasticBuffer(metaclass=_machine.MachineSelector):
        from deep_ep._compat import public_keywords

        def __init__(self, group, num_device_timeout_secs=7):
            self.timeout = num_device_timeout_secs

        __init__ = public_keywords(__init__, {"num_gpu_timeout_secs": "num_device_timeout_secs"})

        def operation(self):
            return self.timeout

    instance = ElasticBuffer(None, num_gpu_timeout_secs=9)
    assert instance.operation() == instance.operation() == 9
    assert len(calls) == 1
    assert instance.machine_type == "mesh_clos"
    monkeypatch.setitem(
        sys.modules, "deep_ep.backends.direct.api", SimpleNamespace(buffer_class=lambda _: ElasticBuffer)
    )
    assert isinstance(instance, ElasticBuffer)


def test_operation_aliases_do_not_inspect_signatures_on_each_call(monkeypatch):
    from deep_ep import _compat

    call = keyword_aliases(lambda a=32: a, {"num_sms": "a"})
    monkeypatch.setattr(_compat.inspect, "signature", lambda _: pytest.fail("hot-path reflection"))
    assert call(a=64) == 64
    assert call(num_sms=32) == 32


def test_runtime_detection_delegates_to_physical_probe(monkeypatch):
    calls, selection = [], {"machine_type": "global_clos"}
    monkeypatch.setattr(_machine, "physical_device", lambda: 13)

    def probe(physical, rootinfo):
        calls.append((physical, rootinfo))
        return selection

    monkeypatch.setattr(_machine, "detect_physical", probe)
    assert _machine.detect("current.json") is selection
    assert calls == [(13, "current.json")]


@pytest.mark.parametrize("physical", [-1, 2**32, True, "0", None])
def test_physical_probe_rejects_invalid_ids_before_queries(monkeypatch, physical):
    monkeypatch.setattr(_machine, "board_id", lambda _: pytest.fail("unexpected driver query"))
    with pytest.raises(ValueError, match="physical device"):
        _machine.detect_physical(physical)


def machine_script():
    path = Path(__file__).resolve().parents[2] / "scripts/check_machine.py"
    spec = importlib.util.spec_from_file_location("machine_check_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_clos_files(tmp_path):
    root, topo = topology()
    topo_path, root_path = tmp_path / "topo.json", tmp_path / "root.json"
    topo_path.write_text(json.dumps(topo), encoding="utf-8")
    root["topo_file_path"] = str(topo_path)
    root_path.write_text(json.dumps(root), encoding="utf-8")
    return root_path


@pytest.mark.parametrize("expected,code", [("global_clos", 0), ("mesh_clos", 1), ("mesh_pcie", 1)])
def test_machine_cli_checks_expectation_without_overriding(tmp_path, expected, code):
    script = machine_script()
    root_path, output = write_clos_files(tmp_path), tmp_path / "report.json"
    assert (
        script.main(
            [
                "--implementation",
                "source",
                "--physical-device",
                "0",
                "--rootinfo",
                str(root_path),
                "--expect",
                expected,
                "--json",
                str(output),
            ]
        )
        == code
    )
    report = json.loads(output.read_text())
    assert report["passed"] is (code == 0)
    assert report["results"][0]["selection"]["machine_type"] == "global_clos"
    assert report["results"][0]["selection"]["source"] == "rootinfo"
    assert len(report["selector"]["sha256"]) == 64


def test_machine_cli_installed_path_does_not_import_package(monkeypatch, tmp_path):
    script = machine_script()
    package = tmp_path / "deep_ep"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('package must not execute')\n")
    (package / "_machine.py").write_bytes(Path(_machine.__file__).read_bytes())
    monkeypatch.syspath_prepend(str(tmp_path))
    root_path, output = write_clos_files(tmp_path), tmp_path / "report.json"
    assert script.main(["--physical-devices", "0", "--rootinfo", str(root_path), "--json", str(output)]) == 0
    report = json.loads(output.read_text())
    assert Path(report["selector"]["path"]) == package / "_machine.py"
    assert report["implementation"] == "installed"


def test_machine_cli_has_no_framework_runtime_or_shmem_import(tmp_path):
    root = Path(__file__).resolve().parents[2]
    root_path = write_clos_files(tmp_path)
    code = """
import ctypes, importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torch_npu', 'deep_ep', 'ascend_deepep'):
            raise AssertionError('unexpected package import: ' + name)
sys.meta_path.insert(0, Block())
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected dynamic library load')
ctypes.CDLL = forbidden
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(root / "scripts/check_machine.py"),
            "--implementation",
            "source",
            "--physical-devices",
            "0",
            "--rootinfo",
            str(root_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MACHINE_CHECK_PASSED" in result.stdout


def test_machine_cli_preserves_missing_evidence_and_checks_all_devices(tmp_path):
    script = machine_script()
    root_path, output = write_clos_files(tmp_path), tmp_path / "report.json"
    assert (
        script.main(
            [
                "--implementation",
                "source",
                "--physical-devices",
                "1",
                "0",
                "--rootinfo",
                str(root_path),
                "--json",
                str(output),
            ]
        )
        == 1
    )
    records = json.loads(output.read_text())["results"]
    assert records[0]["selection"] is None and records[0]["error"]
    assert records[1]["passed"]


def test_machine_cli_rejects_mixed_machines(monkeypatch, tmp_path):
    script, output = machine_script(), tmp_path / "report.json"
    selector = SimpleNamespace(
        detect_physical=lambda device, _: {
            "physical_device": device,
            "machine_type": "mesh_clos" if device else "global_clos",
            "source": "stub",
        }
    )
    monkeypatch.setattr(script, "load_selector", lambda _: (selector, {"path": "CPU stub"}))
    assert script.main(["--physical-devices", "0", "1", "--json", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["passed"] and "disagree" in report["errors"][0]


def test_machine_cli_does_not_overwrite_report(monkeypatch, tmp_path):
    script, output = machine_script(), tmp_path / "report.json"
    output.write_text("previous evidence")
    monkeypatch.setattr(script, "load_selector", lambda _: pytest.fail("must reject before querying"))
    with pytest.raises(SystemExit) as error:
        script.main(["--physical-devices", "0", "--json", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "previous evidence"
