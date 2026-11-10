"""Public entry points and legacy imports must share one backend implementation."""

import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

import deep_ep
from deep_ep.backends.mesh_clos import buffer as pod_v1, elastic as pod_v2


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

    root = Path(__file__).resolve().parents[2]
    packages = set(find_packages(where=root))
    assert {
        "deep_ep.backends",
        "deep_ep.backends.mesh_clos",
        "deep_ep.backends.direct",
        "deep_ep.backends.mesh_pcie",
        "deep_ep.backends.global_clos",
    } <= packages
    assert {name for name in packages if name.startswith("ascend_deepep")} == {"ascend_deepep"}
    assert not (root / "ascend_deepep/lib").exists()


@pytest.mark.parametrize("first", ["deep_ep", "deep_ep.backends.direct"])
def test_public_and_backend_imports_do_not_require_removed_package(first):
    pytest.importorskip("torch")
    code = f"""
import importlib
import importlib.abc
import sys
class RejectRemovedPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ascend_deepep" or fullname.startswith("ascend_deepep."):
            raise AssertionError("imported removed package: " + fullname)
sys.meta_path.insert(0, RejectRemovedPackage())
importlib.import_module({first!r})
import deep_ep
from deep_ep.backends.direct import api as _clos_adapter
from deep_ep.backends import direct as clos
for name in ("buffer", "elastic", "config", "_native", "event", "resources", "dtypes", "utils"):
    importlib.import_module("deep_ep.backends.direct." + name)
assert issubclass(_clos_adapter.Buffer, clos.Buffer)
assert issubclass(_clos_adapter.ElasticBuffer, clos.ElasticBuffer)
assert deep_ep.ExpandHandle is clos.ExpandHandle
assert deep_ep.Config(num_sms=32).num_sms == 32
assert isinstance(object.__new__(clos.EPHandle), deep_ep.EPHandle)
assert not any(n == "ascend_deepep" or n.startswith("ascend_deepep.") for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_clos_adapter_operations_still_use_migrated_functions():
    pytest.importorskip("torch")
    from deep_ep.backends.direct import api as _clos_adapter
    from deep_ep.backends.direct import buffer, elastic

    for adapter, implementation in (
        (_clos_adapter.Buffer, buffer.Buffer),
        (_clos_adapter.ElasticBuffer, elastic.ElasticBuffer),
    ):
        for name in ("dispatch", "combine"):
            assert inspect.unwrap(getattr(adapter, name)) is inspect.unwrap(getattr(implementation, name))
        assert adapter.destroy is implementation.destroy


def test_old_pod_layout_import_is_same_module():
    old = importlib.import_module("deep_ep.buffers._elastic_layout")
    new = importlib.import_module("deep_ep.backends.mesh_clos._elastic_layout")
    assert old is new
