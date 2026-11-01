"""Compatibility exports for the Clos implementation now under deep_ep.backends."""

from deep_ep.backends import clos as _implementation

__all__ = _implementation.__all__


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
