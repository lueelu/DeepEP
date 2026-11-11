# Copyright (c) 2026, Lu Lu
"""UBX-only prepared Combine API; importing this module does not initialize an NPU."""

__all__ = ["Session", "Combine", "routes", "ranks_per_host"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from ..._native import require_native

    require_native("UBX Combine")
    from ... import _ubx_C

    return getattr(_ubx_C, name)
