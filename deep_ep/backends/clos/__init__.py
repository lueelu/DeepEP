"""Internal implementation and legacy import path for deep_ep on Ascend."""

from __future__ import annotations

from ._native import get_native_class
from .buffer import Buffer
from .config import Config
from .dtypes import topk_idx_t
from .elastic import EPHandle, ElasticBuffer
from .event import EventHandle, EventOverlap
from .expand import ExpandHandle
from .resources import Handle, Slot, SlotState


_NATIVE_EXPORTS = frozenset(("BufferRuntime", "ShmemInstance"))


def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        native_class = get_native_class(name)
        globals()[name] = native_class
        return native_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = [
    "Config",
    "Buffer",
    "ElasticBuffer",
    "EPHandle",
    "topk_idx_t",
    "EventHandle",
    "EventOverlap",
    "ExpandHandle",
    "Handle",
    "Slot",
    "SlotState",
    "BufferRuntime",
    "ShmemInstance",
]
