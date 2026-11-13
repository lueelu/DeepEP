# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Lazy native loading. Importing deep_ep never imports a framework or driver."""

import importlib
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import NoReturn

_SHMEM_LIBRARIES = (
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "aclshmem_bootstrap_uid.so",
    "libshmem.so",
)
_shmem_lock = RLock()
_shmem_handles = {}  # Keep successfully loaded objects alive, including after a partial failure.


def _loaded_shmem_paths():
    # Native builds target Linux. Inspect mappings without loading another SDK.
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)"))
            if path.name in _SHMEM_LIBRARIES:
                paths.add(path.resolve())
    return paths


def _load_bundled_shmem(directory, info):
    import ctypes

    bundle = (directory / ".libs").resolve()
    hashes = info.get("shmem_runtime", {})
    with _shmem_lock:
        if set(hashes) != set(_SHMEM_LIBRARIES):
            raise ImportError("Incomplete bundled SHMEM manifest; rebuild and reinstall DeepEP")
        # Check the entire closure before loading anything; never fall back to
        # SHMEM_ROOT or LD_LIBRARY_PATH when a wheel is damaged.
        for name in _SHMEM_LIBRARIES:
            path = bundle / name
            if not path.is_file() or path.resolve().parent != bundle:
                raise ImportError(f"Missing bundled SHMEM library: {path}; reinstall DeepEP")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hashes[name]:
                raise ImportError(f"Bundled SHMEM checksum mismatch: {name}; reinstall DeepEP")
        for path in _loaded_shmem_paths():
            if path != bundle / path.name:
                raise ImportError(
                    f"Conflicting SHMEM library already loaded: {path}; "
                    "start a new process without preloading another SHMEM SDK"
                )
        for name in _SHMEM_LIBRARIES:
            path = str(bundle / name)
            if path not in _shmem_handles:
                try:
                    # Absolute paths select this wheel. Preloading utils also
                    # resolves the SDK's transitive dependency without an SDK RPATH.
                    mode = ctypes.RTLD_LOCAL if name.startswith("aclshmem_bootstrap_") else ctypes.RTLD_GLOBAL
                    _shmem_handles[path] = ctypes.CDLL(path, mode=mode)
                except OSError as exc:
                    raise ImportError(
                        f"Cannot load bundled SHMEM library {name}: {exc}. "
                        "Load the matching CANN environment and check its driver dependencies."
                    ) from exc
    return bundle


def _unavailable(operation: str) -> NoReturn:
    raise NotImplementedError(
        f"{operation} is not implemented in DeepEP. "
        "No communication operation or simulated result has been performed. "
        "See docs/api.md for implementation status."
    )


class _MissingNativeExtension(ImportError):
    pass


@lru_cache(maxsize=1)
def _require_native_cached():
    # Cache only a successfully validated module, shared by all operation names.
    # Exceptions are not cached, so missing files or failed imports can be retried.
    directory = Path(__file__).resolve().parent
    if not any((directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES):
        raise _MissingNativeExtension
    return _load_native()


def require_native(operation):
    try:
        return _require_native_cached()
    except _MissingNativeExtension:
        _unavailable(operation)


@lru_cache(maxsize=1)
def _load_native():
    spec = importlib.util.find_spec("deep_ep._C")
    if spec is None:
        raise ImportError("Native extension file exists but cannot be resolved")
    # A source build exposes its own native files through package-local links.
    # Check the importing directory, not the target of the extension symlink.
    if not spec.origin or Path(spec.origin).parent.resolve() != Path(__file__).resolve().parent:
        raise ImportError("Native extension must belong to this deep_ep installation")
    directory = Path(__file__).resolve().parent
    info = json.loads((directory / "_build_info.json").read_text(encoding="utf-8"))
    bundle = _load_bundled_shmem(directory, info)
    # _C always links the mandatory DeepJIT bridge (and therefore torch_npu);
    # import the Python framework first, like DeepEP-Ascend does.
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    try:
        module = importlib.import_module("deep_ep._C")
    except ImportError as exc:
        raise ImportError(
            "Cannot load deep_ep._C; check the matching CANN environment and DeepEP installation"
        ) from exc
    if module.runtime_protocol != 1:
        raise ImportError("Unsupported DeepEP runtime protocol")
    library = Path(module.shmem_library_path()).resolve()
    if library != bundle / "libshmem.so":
        raise ImportError(
            f"Native extension resolved an external SHMEM library: {library}; restart with a clean environment"
        )
    return module


def npu_api():
    import torch
    import torch_npu  # noqa: F401

    return torch.npu
