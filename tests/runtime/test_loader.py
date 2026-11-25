# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Native identity checks with local test doubles; no framework/driver loading."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from deep_ep import _native


def test_dynamic_preload_reuses_handles_after_partial_failure(monkeypatch, tmp_path):
    import ctypes

    names = ("libdeepep_kernels.so", "libdeepep_clos_kernels.so")
    for name in names:
        (tmp_path / name).write_bytes(b"device")
    calls = []
    fail = True

    def load(path, mode):
        calls.append(path)
        if path.endswith(names[1]) and fail:
            raise OSError("load failed")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(_native, "_device_handles", {})
    _native._preload_dynamic_device_libraries(tmp_path, {"shmem": {"metadata_mode": "fixed"}})
    assert not calls
    info = {"shmem": {"metadata_mode": "dynamic"}}
    with pytest.raises(ImportError, match="Cannot preload"):
        _native._preload_dynamic_device_libraries(tmp_path, info)
    fail = False
    _native._preload_dynamic_device_libraries(tmp_path, info)
    _native._preload_dynamic_device_libraries(tmp_path, info)
    assert calls == [str(tmp_path / names[0]), str(tmp_path / names[1]), str(tmp_path / names[1])]


def test_dynamic_preload_checks_all_files_before_loading(monkeypatch, tmp_path):
    import ctypes

    (tmp_path / "libdeepep_kernels.so").write_bytes(b"device")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("must preflight both libraries"))
    with pytest.raises(ImportError, match="Missing dynamic"):
        _native._preload_dynamic_device_libraries(tmp_path, {"shmem": {"metadata_mode": "dynamic"}})


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
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_loader_identity(monkeypatch, tmp_path, fault, dynamic):
    library = tmp_path / "libshmem.so"
    library.write_bytes(b"test SDK identity")
    expected = hashlib.sha256(library.read_bytes()).hexdigest()
    (tmp_path / "_build_info.json").write_text(
        json.dumps(
            {"shmem": {"metadata_mode": "dynamic" if dynamic else "fixed", "sha256": {"lib/libshmem.so": expected}}}
        )
    )
    loaded = []
    if dynamic:
        import ctypes

        for name in ("libdeepep_kernels.so", "libdeepep_clos_kernels.so"):
            (tmp_path / name).write_bytes(b"device")
        monkeypatch.setattr(_native, "_device_handles", {})
        monkeypatch.setattr(ctypes, "CDLL", lambda path, **kwargs: loaded.append(path) or object())
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

    def import_native(name):
        assert len(loaded) == (2 if dynamic else 0)
        return module

    monkeypatch.setattr(_native.importlib, "import_module", import_native)
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
