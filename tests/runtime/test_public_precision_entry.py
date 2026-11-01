"""Check precision-entry routing without importing Torch or starting NPU jobs."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def entry(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "clos/source/test_v1_v2_shared_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "buffer_entrypoints"
    )
    namespace = {"Buffer": object(), "ElasticBuffer": object()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    package, selector = ModuleType("deep_ep"), ModuleType("deep_ep._machine")
    package.Buffer, package.ElasticBuffer = object(), object()
    calls = []

    def select(*args, **kwargs):
        calls.append((args, kwargs))
        return {"machine_type": selector.kind, "source": "rootinfo"}

    selector.kind, selector.select = "clos_only", select
    monkeypatch.setitem(sys.modules, "deep_ep", package)
    monkeypatch.setitem(sys.modules, "deep_ep._machine", selector)
    return namespace, package, selector, calls


def test_public_entry_uses_auto_and_passes_no_engine_override(entry):
    namespace, public, _, calls = entry
    group = object()
    result = namespace["buffer_entrypoints"](
        SimpleNamespace(public_auto=True, public_clos=False, transport=None), group
    )
    assert result == (public.Buffer, public.ElasticBuffer, {})
    assert calls == [((), {"group": group})]


@pytest.mark.parametrize("kind", ["pod", "card_4pmesh"])
def test_public_entry_rejects_other_topologies_without_forcing_clos(entry, kind):
    namespace, _, selector, _ = entry
    selector.kind = kind
    with pytest.raises(RuntimeError, match="requires auto-detected clos_only"):
        namespace["buffer_entrypoints"](SimpleNamespace(public_auto=True, public_clos=False, transport=None), object())


@pytest.mark.parametrize("transport", ["mte", "udma"])
def test_public_entry_rejects_explicit_transport(entry, transport):
    namespace, _, _, calls = entry
    with pytest.raises(ValueError, match="omit --transport"):
        namespace["buffer_entrypoints"](
            SimpleNamespace(public_auto=True, public_clos=False, transport=transport), object()
        )
    assert calls == []


@pytest.mark.parametrize("transport", [None, "mte", "udma"])
def test_direct_backend_entry_preserves_original_engine_default(entry, transport):
    namespace, _, _, calls = entry
    result = namespace["buffer_entrypoints"](
        SimpleNamespace(public_auto=False, public_clos=False, transport=transport), object()
    )
    assert result == (namespace["Buffer"], namespace["ElasticBuffer"], {"transport": transport or "udma"})
    assert calls == []


def test_explicit_public_clos_is_separate_from_detection(entry):
    namespace, public, _, calls = entry
    result = namespace["buffer_entrypoints"](
        SimpleNamespace(public_auto=False, public_clos=True, transport=None), object()
    )
    assert result == (public.Buffer, public.ElasticBuffer, {"machine_type": "clos_only"})
    assert calls == []
