"""Public V1 Buffer entry point; machine selection runs once at construction."""

from ._machine import MachineSelector
from ._native import require_native as require_native
from .backends.pod.buffer import Buffer as PodBuffer


class Buffer(PodBuffer, metaclass=MachineSelector):
    """Select POD, Clos or CARD_4PMESH while preserving the V1 interface."""


def _is_blackwell():
    return False


__all__ = ["Buffer", "_is_blackwell"]
