# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""V1 configuration with upstream names and platform-neutral aliases."""

from ._compat import public_keywords


class Config:
    """AIV tuning and conservative Clos workspace sizing."""

    __slots__ = ("_impl",)

    def __init__(
        self,
        num_compute_units: int = 32,
        num_max_scaleup_chunked_send_tokens: int = 6,
        num_max_scaleup_chunked_recv_tokens: int = 256,
        num_max_scaleout_chunked_send_tokens: int = 6,
        num_max_scaleout_chunked_recv_tokens: int = 256,
        *,
        udma_peers_per_worker: int | None = None,
        udma_wqes_per_visit: int | None = None,
    ) -> None:
        from .backends.clos.config import Config as ClosConfig

        values = [
            num_compute_units,
            num_max_scaleup_chunked_send_tokens,
            num_max_scaleup_chunked_recv_tokens,
            num_max_scaleout_chunked_send_tokens,
            num_max_scaleout_chunked_recv_tokens,
        ]
        self._impl = ClosConfig(
            *values, udma_peers_per_worker=udma_peers_per_worker, udma_wqes_per_visit=udma_wqes_per_visit
        )

    @property
    def num_compute_units(self):
        return self._impl.num_aiv

    def __getattr__(self, name):
        return getattr(
            object.__getattribute__(self, "_impl"), name.replace("scaleup", "nvl").replace("scaleout", "rdma")
        )

    def get_nvl_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, num_tokens: int = 8192) -> int:
        return self._impl.get_nvl_buffer_size_hint(hidden_bytes, num_ranks, num_tokens)

    def get_rdma_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, /) -> int:
        return self._impl.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)

    get_scaleup_buffer_size_hint = get_nvl_buffer_size_hint
    get_scaleout_buffer_size_hint = get_rdma_buffer_size_hint


Config.__init__ = public_keywords(
    Config.__init__,
    {
        "num_sms": "num_compute_units",
        "num_max_nvl_chunked_send_tokens": "num_max_scaleup_chunked_send_tokens",
        "num_max_nvl_chunked_recv_tokens": "num_max_scaleup_chunked_recv_tokens",
        "num_max_rdma_chunked_send_tokens": "num_max_scaleout_chunked_send_tokens",
        "num_max_rdma_chunked_recv_tokens": "num_max_scaleout_chunked_recv_tokens",
    },
    aliases={
        "num_aiv": "num_compute_units",
        "num_max_shmem_chunked_send_tokens": "num_max_scaleup_chunked_send_tokens",
        "num_max_shmem_chunked_recv_tokens": "num_max_scaleup_chunked_recv_tokens",
    },
)
