# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""Internal MegaMoE product selection; public signatures remain in api.py."""

from .validation import collective_check

DECODE_MAX_TOKENS_PER_RANK = 256


def _ubx_backend(world, capacity, ranks_per_node):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("num_max_tokens_per_rank must be a positive integer")
    if world in (4, 8, 16):
        return "mte" if capacity <= DECODE_MAX_TOKENS_PER_RANK else "urma"
    if world == 32:
        if ranks_per_node != 16 or capacity > DECODE_MAX_TOKENS_PER_RANK:
            raise ValueError(
                f"UBX EP32 requires ranks_per_node=16 and num_max_tokens_per_rank <= {DECODE_MAX_TOKENS_PER_RANK}"
            )
        return "mte"  # The MTE entry creates a hierarchical RoCE session for 2 x 16 ranks.
    raise ValueError("UBX MegaMoE supports EP4/EP8/EP16, or EP32 for hierarchical decode")


def select_backend(group, num_max_tokens_per_rank, ranks_per_node, *, base=None):
    error = None
    ubx = None
    if base is not None:
        selected = getattr(base, "_backend_name", None)
        if selected is None:
            error = "base must be an initialized SymmBuffer"
    else:
        try:
            from deep_ep._machine import board_id, physical_device

            ubx = board_id(physical_device()) == 0x44
        except Exception:
            ubx = False
        selected = "mte"
        if ubx:
            try:
                selected = _ubx_backend(group.size(), num_max_tokens_per_rank, ranks_per_node)
            except ValueError as failure:
                error = str(failure)
    signature = (selected, num_max_tokens_per_rank, ranks_per_node, ubx)
    collective_check(group, "MegaMoE product selection", error, signature)
    return selected
