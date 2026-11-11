# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man

"""One internal MegaMoE resource owner for POD and URMA backends."""

from deep_ep._runtime import Runtime


class _MegaRuntime(Runtime):
    _megamoe = True

    def __init__(self, group, capacity, timeout=120, *, backend="pod"):
        self._megamoe_backend = backend
        super().__init__(group, capacity, timeout)
