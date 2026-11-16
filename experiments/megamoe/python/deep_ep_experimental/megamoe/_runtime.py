# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""One internal MegaMoE resource owner for MTE and URMA backends."""

from deep_ep._runtime import Runtime


class _MegaRuntime(Runtime):
    _megamoe = True

    def __init__(self, group, capacity, timeout=120, *, backend="mte"):
        self._megamoe_backend = backend
        super().__init__(group, capacity, timeout)
