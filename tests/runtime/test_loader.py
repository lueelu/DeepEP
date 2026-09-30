# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Native identity checks with local test doubles; no framework/driver loading."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from deep_ep import _native


@pytest.mark.parametrize("fault", [None, "origin", "protocol", "checksum", "missing"])
def test_native_loader_identity(monkeypatch, tmp_path, fault):
    library = tmp_path / "libshmem.so"
    library.write_bytes(b"test SDK identity")
    expected = hashlib.sha256(library.read_bytes()).hexdigest()
    (tmp_path / "_build_info.json").write_text(
        json.dumps({"shmem": {"sha256": {"lib/libshmem.so": expected if fault != "checksum" else "0" * 64}}})
    )
    module = SimpleNamespace(
        runtime_protocol=2 if fault == "protocol" else 1,
        shmem_library_path=lambda: str(library),
    )
    spec = SimpleNamespace(origin=str((tmp_path.parent if fault == "origin" else tmp_path) / "_C.so"))
    monkeypatch.setattr(_native, "__file__", str(tmp_path / "_native.py"))
    monkeypatch.setattr(
        _native.importlib.util,
        "find_spec",
        lambda name: None if fault == "missing" else spec,
    )
    monkeypatch.setattr(_native.importlib, "import_module", lambda name: module)
    _native._load_native.cache_clear()
    try:
        if fault == "checksum":
            with pytest.warns(RuntimeWarning, match="Loaded SHMEM library differs"):
                assert _native._load_native() is module
        elif fault:
            with pytest.raises(ImportError):
                _native._load_native()
        else:
            assert _native._load_native() is module
    finally:
        _native._load_native.cache_clear()
