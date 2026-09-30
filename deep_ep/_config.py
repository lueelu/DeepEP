# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Legacy Config declarations; no device sizing policy is implemented."""

from ._native import _unavailable


class Config:
    """Platform-neutral names with reference defaults, not a device tuning policy."""

    def __init__(
        self,
        num_compute_units: int = 20,
        num_max_scaleup_chunked_send_tokens: int = 6,
        num_max_scaleup_chunked_recv_tokens: int = 256,
        num_max_scaleout_chunked_send_tokens: int = 6,
        num_max_scaleout_chunked_recv_tokens: int = 256,
    ) -> None:
        _unavailable("Config.__init__")

    def get_scaleup_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, /) -> int:
        _unavailable("Config.get_scaleup_buffer_size_hint")

    def get_scaleout_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, /) -> int:
        _unavailable("Config.get_scaleout_buffer_size_hint")
