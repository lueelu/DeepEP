"""Internal MegaMoE product selection; public signatures remain in api.py."""

from .validation import collective_check


def cv_rows(world, bs):
    return 128 if world == 16 and bs in (1024, 2048) else 256


def select_backend(group):
    # Product identity is distinct from fabric topology. Unknown products keep
    # the POD entry; its existing shape/platform/peer checks still apply.
    try:
        from deep_ep._machine import board_id, physical_device

        product = board_id(physical_device())
    except Exception:
        product = None
    selected = "ubx" if product == 0x44 else "pod"
    collective_check(group, "MegaMoE product selection", None, selected)
    return selected
