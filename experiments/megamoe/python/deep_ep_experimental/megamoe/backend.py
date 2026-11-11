# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""Internal MegaMoE product selection; public signatures remain in api.py."""

from .validation import collective_check


def select_backend(group):
    # Product identity is distinct from fabric topology. Unknown products keep
    # the POD entry; its existing shape/platform/peer checks still apply.
    try:
        from deep_ep._machine import board_id, physical_device

        product = board_id(physical_device())
    except Exception:
        product = None
    # Board 0x44 identifies the UBX product; its implementation uses URMA.
    selected = "urma" if product == 0x44 else "pod"
    collective_check(group, "MegaMoE product selection", None, selected)
    return selected
