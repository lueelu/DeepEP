# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Native identity checks with local test doubles; no framework/driver loading."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from deep_ep import _native


def test_require_native_reuses_success_across_operations(monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock

    module = object()
    probe, load = Mock(return_value=True), Mock(return_value=module)
    monkeypatch.setattr(Path, "is_file", probe)
    monkeypatch.setattr(_native, "_load_native", load)
    _native._require_native_cached.cache_clear()
    try:
        for operation in ("Dispatch", "Combine", "Notify", "Dispatch"):
            assert _native.require_native(operation) is module
        probe.assert_called_once()
        load.assert_called_once()
    finally:
        _native._require_native_cached.cache_clear()


@pytest.mark.parametrize("fault", [None, "origin", "protocol", "external", "missing", "source_link"])
def test_native_loader_identity(monkeypatch, tmp_path, fault):
    library = tmp_path / "libshmem.so"
    library.write_bytes(b"test SDK identity")
    expected = hashlib.sha256(library.read_bytes()).hexdigest()
    (tmp_path / "_build_info.json").write_text(json.dumps({"shmem": {"sha256": {"lib/libshmem.so": expected}}}))
    module = SimpleNamespace(
        runtime_protocol=2 if fault == "protocol" else 1,
        shmem_library_path=lambda: str(library if fault != "external" else tmp_path.parent / "libshmem.so"),
    )
    spec = SimpleNamespace(origin=str((tmp_path.parent if fault == "origin" else tmp_path) / "_C.so"))
    if fault == "source_link":
        target = tmp_path / "build" / "_C.so"
        target.parent.mkdir()
        target.write_bytes(b"native test double")
        from pathlib import Path

        try:
            Path(spec.origin).symlink_to(target)
        except OSError:
            pytest.skip("Source builds require symlink support")
    monkeypatch.setattr(_native, "__file__", str(tmp_path / "_native.py"))
    monkeypatch.setattr(
        _native.importlib.util,
        "find_spec",
        lambda name: None if fault == "missing" else spec,
    )
    monkeypatch.setattr(_native.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(_native, "_load_bundled_shmem", lambda directory, info: tmp_path)
    _native._load_native.cache_clear()
    try:
        if fault and fault != "source_link":
            with pytest.raises(ImportError):
                _native._load_native()
        else:
            assert _native._load_native() is module
    finally:
        _native._load_native.cache_clear()


@pytest.mark.parametrize("fault", [None, "manifest", "missing", "checksum", "external", "dependency"])
def test_bundled_shmem_is_complete_and_never_falls_back(monkeypatch, tmp_path, fault):
    import ctypes

    bundle = tmp_path / ".libs"
    bundle.mkdir()
    hashes = {}
    for name in _native._SHMEM_LIBRARIES:
        path = bundle / name
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = bundle / "aclshmem_bootstrap_uid.so"
    if fault == "manifest":
        hashes.pop(target.name)
    elif fault == "missing":
        target.unlink()
    elif fault == "checksum":
        target.write_bytes(b"replaced library")
    external = tmp_path / "external" / "libshmem.so"
    monkeypatch.setattr(_native, "_loaded_shmem_paths", lambda: {external} if fault == "external" else set())
    monkeypatch.setattr(_native, "_shmem_handles", {})
    calls = []

    def load(path, mode):
        calls.append(path)
        if fault == "dependency":
            raise OSError("libruntime.so: not found")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", load)
    if fault:
        with pytest.raises(
            ImportError,
            match={
                "manifest": "Incomplete bundled",
                "missing": "Missing bundled",
                "checksum": "checksum mismatch",
                "external": "Conflicting SHMEM",
                "dependency": "libruntime.so: not found",
            }[fault],
        ):
            _native._load_bundled_shmem(tmp_path, {"shmem_runtime": hashes})
        assert len(calls) == (1 if fault == "dependency" else 0)
    else:
        for _ in range(2):
            assert _native._load_bundled_shmem(tmp_path, {"shmem_runtime": hashes}) == bundle
        assert calls == [str(bundle / name) for name in _native._SHMEM_LIBRARIES]
