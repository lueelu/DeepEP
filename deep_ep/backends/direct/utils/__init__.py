"""Utilities shared by the Ascend Python API."""

from __future__ import annotations


def align_up(value: int, alignment: int) -> int:
    """Round a non-negative integer up to an alignment boundary."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("value must be a non-negative integer")
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    return ((value + alignment - 1) // alignment) * alignment


__all__ = ["align_up"]
