"""Public entry points and legacy imports must share one backend implementation."""

import importlib
import inspect
import subprocess
import sys

import pytest

import deep_ep
from deep_ep.backends.pod import buffer as pod_v1, elastic as pod_v2


def test_public_entries_and_legacy_identity():
    from deep_ep.buffer import Buffer
    from deep_ep.buffers import Buffer as PackageBuffer
    from deep_ep.buffers.legacy import Buffer as LegacyBuffer
    from deep_ep.buffers.elastic import ElasticBuffer

    assert Buffer is PackageBuffer is LegacyBuffer is deep_ep.Buffer
    assert ElasticBuffer is deep_ep.ElasticBuffer
    assert Buffer.__module__ == "deep_ep.buffer"
    assert ElasticBuffer.__module__ == "deep_ep.buffers.elastic"


@pytest.mark.parametrize("name", ["dispatch", "combine", "destroy"])
@pytest.mark.parametrize(
    "public,backend", [(deep_ep.Buffer, pod_v1.Buffer), (deep_ep.ElasticBuffer, pod_v2.ElasticBuffer)]
)
def test_pod_operations_are_inherited_without_per_call_routing(public, backend, name):
    assert getattr(public, name) is getattr(backend, name)


def test_all_backend_packages_are_in_distribution():
    from setuptools import find_packages

    packages = set(find_packages(include=["deep_ep", "deep_ep.*", "ascend_deepep", "ascend_deepep.*"]))
    assert {"deep_ep.backends", "deep_ep.backends.pod", "deep_ep.backends.clos", "ascend_deepep"} <= packages


@pytest.mark.parametrize("first", ["ascend_deepep", "deep_ep.backends.clos"])
def test_clos_import_orders_share_module_state(first):
    pytest.importorskip("torch")
    code = f"""
import importlib
importlib.import_module({first!r})
import deep_ep
import ascend_deepep
from deep_ep.backends import clos
for name in ("buffer", "elastic", "config", "_native", "event", "resources", "dtypes", "utils"):
    old = importlib.import_module("ascend_deepep." + name)
    new = importlib.import_module("deep_ep.backends.clos." + name)
    assert old is new, name
    sentinel = object()
    old._compat_test_state = sentinel
    assert new._compat_test_state is sentinel, name
assert ascend_deepep.Buffer is clos.Buffer
assert ascend_deepep.ElasticBuffer is clos.ElasticBuffer
assert isinstance(object.__new__(clos.EPHandle), deep_ep.EPHandle)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_clos_adapter_operations_still_use_migrated_functions():
    pytest.importorskip("torch")
    from deep_ep import _clos_adapter
    from deep_ep.backends.clos import buffer, elastic

    for adapter, implementation in (
        (_clos_adapter.Buffer, buffer.Buffer),
        (_clos_adapter.ElasticBuffer, elastic.ElasticBuffer),
    ):
        for name in ("dispatch", "combine"):
            assert inspect.unwrap(getattr(adapter, name)) is inspect.unwrap(getattr(implementation, name))
        assert adapter.destroy is implementation.destroy


def test_old_pod_layout_import_is_same_module():
    old = importlib.import_module("deep_ep.buffers._elastic_layout")
    new = importlib.import_module("deep_ep.backends.pod._elastic_layout")
    assert old is new
