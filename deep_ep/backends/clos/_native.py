"""Lazy loading helpers for the optional Ascend native extension."""

from __future__ import annotations

import importlib
from threading import Lock
from types import ModuleType
from typing import Optional


_NATIVE_MODULE: Optional[ModuleType] = None
_NATIVE_LOCK = Lock()


def _load_package_shmem_libs() -> None:
    # Both native modules resolve the same SDK SONAME. Never ship a second SDK.
    from deep_ep._native import require_native

    require_native("Clos SHMEM dependency")


def load_native_module() -> ModuleType:
    """Load and return ``deep_ep._clos_C`` on first native API use."""

    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE

    with _NATIVE_LOCK:
        if _NATIVE_MODULE is not None:
            return _NATIVE_MODULE
        _load_package_shmem_libs()

        import torch

        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise ImportError("torch_npu must be installed and loadable before using deep_ep native APIs") from exc

        module = importlib.import_module("deep_ep._clos_C")

        from .dtypes import topk_idx_t

        compiled_topk_idx_t = getattr(module, "topk_idx_t", None)
        if compiled_topk_idx_t != topk_idx_t:
            raise ImportError(
                "Python topk_idx_t does not match the native extension: "
                f"python={topk_idx_t}, native={compiled_topk_idx_t}"
            )

        # Importing the extension registers the torch.classes definitions.
        _resolve_native_class(torch, "ShmemInstance")
        _resolve_native_class(torch, "BufferRuntime")
        _NATIVE_MODULE = module
        return module


def get_native_class(name: str):
    """Resolve a registered low-level ``torch.classes.AscendDeepEP`` class."""

    load_native_module()
    import torch

    return _resolve_native_class(torch, name)


def _resolve_native_class(torch_module: ModuleType, name: str):
    """Resolve one registered class and normalize torch lookup failures."""

    try:
        namespace = getattr(torch_module.classes, "AscendDeepEP")
        return getattr(namespace, name)
    except (AttributeError, RuntimeError) as exc:
        raise ImportError(f"native AscendDeepEP class is not registered: {name}") from exc
