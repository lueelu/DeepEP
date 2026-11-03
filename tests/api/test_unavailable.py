# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""An API-only checkout must never report successful communication or completion."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import deep_ep
from importlib.machinery import EXTENSION_SUFFIXES


ROOT = Path(deep_ep.__file__).resolve().parent.parent
INSTALLED_NATIVE = any((ROOT / "deep_ep" / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES)
RUNTIME_METHODS = {
    "Buffer.__init__",
    "Buffer.destroy",
    "Buffer.capture",
    "Buffer.get_comm_stream",
    "ElasticBuffer.__init__",
    "ElasticBuffer.destroy",
    "ElasticBuffer.capture",
    "ElasticBuffer.get_comm_stream",
    "ElasticBuffer.barrier",
    "ElasticBuffer.get_buffer_size_hint",
    "ElasticBuffer.dispatch",
    "ElasticBuffer.combine",
    "EPHandle.__init__",
}
BASELINE = json.loads(Path(__file__).with_name("api_contract.json").read_text(encoding="utf-8"))
HOST_METHODS = {
    "Buffer.set_num_compute_units",
    "Buffer.set_num_sms",
    "Buffer.get_dispatch_config",
    "Buffer.get_combine_config",
}
METHODS = [
    (name, method)
    for name, spec in BASELINE["objects"].items()
    for method in spec["methods"]
    if name != "Config" and f"{name}.{method}" not in HOST_METHODS
]


class Uninspectable:
    """Fail if a stub touches a tensor, process group, handle, or event."""

    def __getattr__(self, name):
        raise AssertionError(f"Stub unexpectedly inspected input attribute {name}")


@pytest.mark.parametrize("num_qps", [0, 1, 32])
@pytest.mark.parametrize("method", ["dispatch", "combine"])
def test_reserved_qp_budget_does_not_enable_an_algorithm(method, num_qps):
    if INSTALLED_NATIVE and method in ("dispatch", "combine"):
        pytest.skip("Installed notify parameter rejection is covered by notify tests")
    operation = getattr(deep_ep.ElasticBuffer, method)
    args = [Uninspectable(), Uninspectable()]
    if method == "combine":
        args.append(Uninspectable())
    with pytest.raises(NotImplementedError, match=rf"ElasticBuffer\.{method} is not implemented"):
        operation(*args, num_qps=num_qps)


@pytest.mark.parametrize("name,method", METHODS, ids=[f"{n}.{m}" for n, m in METHODS])
def test_every_declared_operation_fails_before_touching_inputs(name, method):
    if name.startswith("Event") or (INSTALLED_NATIVE and f"{name}.{method}" in RUNTIME_METHODS):
        pytest.skip("Implemented runtime operation covered by runtime tests")
    operation = getattr(getattr(deep_ep, name), method)
    args = [
        Uninspectable()
        for p in inspect.signature(operation).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    canonical = "get_theoretical_num_compute_units" if method == "get_theoretical_num_sms" else method
    if method == "get_low_latency_rdma_size_hint":
        canonical = "get_low_latency_scaleout_size_hint"
    with pytest.raises(NotImplementedError, match=rf"{name}\.{canonical} is not implemented"):
        operation(*args)


@pytest.mark.parametrize("name", [name for name in BASELINE["exports"] if name != "Config"])
def test_constructors_cannot_create_usable_objects(name):
    if name.startswith("Event") or (
        INSTALLED_NATIVE
        and name
        in {
            "Buffer",
            "ElasticBuffer",
            "EventHandle",
            "EventOverlap",
            "EPHandle",
        }
    ):
        pytest.skip("Implemented runtime constructor covered by runtime tests")
    cls = getattr(deep_ep, name)
    args = [Uninspectable() for p in inspect.signature(cls).parameters.values() if p.default is inspect.Parameter.empty]
    with pytest.raises(NotImplementedError, match=f"{name}.__init__"):
        cls(*args)


def run_isolated(code):
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1")
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_imports_and_failure_paths_do_not_load_framework_or_native_runtime():
    if INSTALLED_NATIVE:
        pytest.skip("Source-only failure path; installed import isolation is tested separately")
    run_isolated(
        """
import importlib.abc, sys
class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'numpy', 'mpi4py'} or fullname == 'deep_ep._C':
            raise AssertionError('Unexpected optional dependency import: ' + fullname)
sys.meta_path.insert(0, BlockFramework())
import deep_ep
from deep_ep.buffers.legacy import Buffer
from deep_ep.buffers.elastic import ElasticBuffer, EPHandle
from deep_ep.utils.event import EventHandle, EventOverlap
assert all(getattr(deep_ep, name) is not None for name in deep_ep.__all__)
for operation in (lambda: Buffer(None), lambda: ElasticBuffer(None), Buffer.capture,
                  ElasticBuffer.capture):
    try:
        operation()
    except NotImplementedError:
        continue
    raise AssertionError('Stub unexpectedly succeeded')
assert 'deep_ep._C' not in sys.modules
assert 'torch' not in sys.modules
"""
    )


def test_fake_event_does_not_gain_completion_or_context_manager_success():
    if INSTALLED_NATIVE:
        pytest.skip("Installed events are covered by runtime tests")
    event = object.__new__(deep_ep.EventOverlap)
    with pytest.raises(AttributeError):
        event.current_stream_wait(release_handle=True)
    with pytest.raises(AttributeError):
        with event:
            pass  # Upstream checks on context exit, not entry.
