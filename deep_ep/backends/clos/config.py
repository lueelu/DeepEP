"""Configuration values used by Ascend communication tiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._build_config import CATLASS_ARCH as _BUILD_CATLASS_ARCH
from .dtypes import topk_idx_bytes
from .utils import align_up


_DEFAULT_NUM_AIV_FOR_SIZE_HINT = 32
_KERNEL_PLATFORM_CONFIG = Path(__file__).resolve().parent / "lib" / "platform_config.txt"
if _KERNEL_PLATFORM_CONFIG.exists():
    _platform_values = dict(
        line.split("=", 1) for line in _KERNEL_PLATFORM_CONFIG.read_text(encoding="ascii").splitlines() if "=" in line
    )
    _CATLASS_ARCH = int(_platform_values["catlass_arch"])
else:
    _CATLASS_ARCH = _BUILD_CATLASS_ARCH
if _CATLASS_ARCH == 3510:
    _MAX_NUM_AIV = 64
elif _CATLASS_ARCH == 2201:
    _MAX_NUM_AIV = 48
else:
    raise RuntimeError(f"unsupported build-time CATLASS_ARCH: {_CATLASS_ARCH}")
_DEFAULT_NUM_TOKENS_FOR_SIZE_HINT = 8192
_MAX_SHMEM_INTRANODE_RANKS = 1024
_MAX_LOCAL_EXPERTS_FOR_SIZE_HINT = 1024
_MAX_TOPK_FOR_SIZE_HINT = 128
_MAX_SCALES_FOR_SIZE_HINT = 128
_TOPK_IDX_BYTES = topk_idx_bytes
_INT_BYTES = 4
_FLOAT_BYTES = 4
_COUNT_ALIGNMENT_ELEMS = 8
_WORKSPACE_ALIGNMENT = 32
_NOTIFY_WORKSPACE_ALIGNMENT = 512
_UDMA_COMPLETION_SIGNAL_BYTES = 512
_SHMEM_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024


@dataclass(frozen=True, init=False)
class Config:
    """Host-selected fields shared by dispatch/combine tiling.

    A zero-valued AIV count requests the backend default. NVL and RDMA chunk
    fields follow the public names used by DeepEP.
    The RDMA fields are reserved for future inter-node kernels; the current
    intra-node implementation does not allocate RDMA workspace.
    """

    num_aiv: int = 0
    udma_peers_per_worker: int = 4
    udma_wqes_per_visit: int = 8
    num_max_nvl_chunked_send_tokens: int = 6
    num_max_nvl_chunked_recv_tokens: int = 256
    num_max_rdma_chunked_send_tokens: int = 6
    num_max_rdma_chunked_recv_tokens: int = 256

    def __init__(
        self,
        num_aiv: int = 0,
        num_max_nvl_chunked_send_tokens: int = 6,
        num_max_nvl_chunked_recv_tokens: int = 256,
        num_max_rdma_chunked_send_tokens: int = 6,
        num_max_rdma_chunked_recv_tokens: int = 256,
        *,
        udma_peers_per_worker: int = 4,
        udma_wqes_per_visit: int = 8,
        num_sms: int | None = None,
        num_max_shmem_chunked_send_tokens: int | None = None,
        num_max_shmem_chunked_recv_tokens: int | None = None,
    ) -> None:
        """Create a communication tuning configuration.

        The first five positional arguments represent AIV, NVL send/receive
        chunk capacity and reserved RDMA send/receive chunk capacity. The
        positional order and NVL field names match DeepEP. ``num_sms`` is
        accepted as an alias for ``num_aiv``.
        """

        num_aiv = _resolve_alias("num_aiv", num_aiv, 0, "num_sms", num_sms)
        num_max_nvl_chunked_send_tokens = _resolve_alias(
            "num_max_nvl_chunked_send_tokens",
            num_max_nvl_chunked_send_tokens,
            6,
            "num_max_shmem_chunked_send_tokens",
            num_max_shmem_chunked_send_tokens,
        )
        num_max_nvl_chunked_recv_tokens = _resolve_alias(
            "num_max_nvl_chunked_recv_tokens",
            num_max_nvl_chunked_recv_tokens,
            256,
            "num_max_shmem_chunked_recv_tokens",
            num_max_shmem_chunked_recv_tokens,
        )
        values = {
            "num_aiv": num_aiv,
            "udma_peers_per_worker": udma_peers_per_worker,
            "udma_wqes_per_visit": udma_wqes_per_visit,
            "num_max_nvl_chunked_send_tokens": num_max_nvl_chunked_send_tokens,
            "num_max_nvl_chunked_recv_tokens": num_max_nvl_chunked_recv_tokens,
            "num_max_rdma_chunked_send_tokens": num_max_rdma_chunked_send_tokens,
            "num_max_rdma_chunked_recv_tokens": num_max_rdma_chunked_recv_tokens,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name, allowed in (("udma_peers_per_worker", (1, 2, 4, 8)), ("udma_wqes_per_visit", (4, 8, 16, 32))):
            value = getattr(self, name)
            _validate_positive_int(name, value)
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        _validate_nonnegative_int("num_aiv", self.num_aiv)
        if self.num_aiv % 2:
            raise ValueError("num_aiv must be even")
        if self.num_aiv > _MAX_NUM_AIV:
            raise ValueError(f"num_aiv must not exceed {_MAX_NUM_AIV}")

        self._validate_chunk_pair(
            "nvl",
            self.num_max_nvl_chunked_send_tokens,
            self.num_max_nvl_chunked_recv_tokens,
        )
        self._validate_chunk_pair(
            "rdma",
            self.num_max_rdma_chunked_send_tokens,
            self.num_max_rdma_chunked_recv_tokens,
        )

    @property
    def uses_platform_defaults(self) -> bool:
        return self.num_aiv == 0

    @property
    def num_sms(self) -> int:
        """Return the configured AIV task-block count."""

        return self.num_aiv

    @property
    def num_max_shmem_chunked_send_tokens(self) -> int:
        """Compatibility alias for the former Ascend-specific public name."""

        return self.num_max_nvl_chunked_send_tokens

    @property
    def num_max_shmem_chunked_recv_tokens(self) -> int:
        """Compatibility alias for the former Ascend-specific public name."""

        return self.num_max_nvl_chunked_recv_tokens

    def get_nvl_buffer_size_hint(
        self,
        hidden_bytes: int,
        num_ranks: int,
        num_tokens: int = _DEFAULT_NUM_TOKENS_FOR_SIZE_HINT,
    ) -> int:
        """Estimate the full-capacity intra-node workspace.

        ``hidden_bytes`` is the byte width of one token and ``num_tokens`` is
        the maximum number of source tokens on each rank. The default reserves
        capacity for 8192 tokens per rank. Top-k indices, weights and scales
        use the V1 maximum widths so optional payloads cannot under-size the
        symmetric workspace. The returned size satisfies the 2 MiB CANN SHMEM
        allocation alignment and can be passed directly to ``Buffer``.
        """

        _validate_size_hint_inputs(hidden_bytes, num_ranks)
        _validate_positive_int("num_tokens", num_tokens)
        num_aiv = self.num_aiv or _DEFAULT_NUM_AIV_FOR_SIZE_HINT
        num_channels = num_aiv

        rank_stride = align_up(num_ranks, _COUNT_ALIGNMENT_ELEMS)
        channel_stride = align_up(num_channels, _COUNT_ALIGNMENT_ELEMS)
        expert_stride = align_up(
            num_ranks * _MAX_LOCAL_EXPERTS_FOR_SIZE_HINT,
            _COUNT_ALIGNMENT_ELEMS,
        )
        persistent_notify_elems = num_ranks * num_ranks + num_ranks * channel_stride
        normal_notify_elems = persistent_notify_elems + rank_stride + expert_stride + num_channels * rank_stride
        num_bytes = align_up(
            normal_notify_elems * _INT_BYTES,
            _NOTIFY_WORKSPACE_ALIGNMENT,
        )

        channel_prefix_row_bytes = num_channels * _INT_BYTES
        num_bytes = align_up(
            num_bytes + num_ranks * channel_prefix_row_bytes,
            _WORKSPACE_ALIGNMENT,
        )

        recv_buffer_tokens = num_tokens * num_ranks
        num_bytes = align_up(
            num_bytes + recv_buffer_tokens * hidden_bytes,
            _WORKSPACE_ALIGNMENT,
        )
        # Source indices share x's compact receive-slot order. GM does not
        # require each channel segment to start at a 32-byte boundary.
        src_idx_bytes = recv_buffer_tokens * _INT_BYTES
        num_bytes = align_up(
            num_bytes + src_idx_bytes,
            _WORKSPACE_ALIGNMENT,
        )
        optional_row_bytes = (
            _MAX_TOPK_FOR_SIZE_HINT * _TOPK_IDX_BYTES,
            _MAX_TOPK_FOR_SIZE_HINT * _FLOAT_BYTES,
            _MAX_SCALES_FOR_SIZE_HINT * _FLOAT_BYTES,
        )
        for row_bytes in optional_row_bytes:
            num_bytes = align_up(
                num_bytes + recv_buffer_tokens * row_bytes,
                _WORKSPACE_ALIGNMENT,
            )
        # Non-contiguous FP8 scale views are normalized into a compact local
        # source region before MTE/UDMA sends. The region is reserved for every
        # FP8 dispatch so all ranks retain identical SHMEM offsets.
        num_bytes = align_up(
            num_bytes + num_tokens * _MAX_SCALES_FOR_SIZE_HINT * _FLOAT_BYTES,
            _WORKSPACE_ALIGNMENT,
        )
        # Reserve the optional UDMA completion slots as well, so one hint is
        # valid for both MTE and UDMA Buffer transports.
        # One incoming completion marker per source, independent of QP/worker count.
        signal_slots = num_ranks
        num_bytes = align_up(num_bytes, _UDMA_COMPLETION_SIGNAL_BYTES)
        num_bytes = align_up(
            num_bytes + signal_slots * _UDMA_COMPLETION_SIGNAL_BYTES,
            _WORKSPACE_ALIGNMENT,
        )
        return align_up(num_bytes, _SHMEM_ALLOCATION_ALIGNMENT)

    def get_rdma_buffer_size_hint(self, hidden_bytes: int, num_ranks: int) -> int:
        """Return the reserved RDMA workspace requirement.

        Inputs are validated even though the current intra-node path does not
        consume an RDMA workspace and therefore returns zero.
        """

        _validate_size_hint_inputs(hidden_bytes, num_ranks)
        return 0

    @staticmethod
    def _validate_chunk_pair(name: str, send_tokens: int, recv_tokens: int) -> None:
        for field_name, value in (
            (f"num_max_{name}_chunked_send_tokens", send_tokens),
            (f"num_max_{name}_chunked_recv_tokens", recv_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if send_tokens >= recv_tokens:
            raise ValueError(
                f"num_max_{name}_chunked_send_tokens must be smaller than num_max_{name}_chunked_recv_tokens"
            )
        if name == "rdma" and send_tokens > recv_tokens // 2:
            raise ValueError(
                "num_max_rdma_chunked_send_tokens must not exceed half of num_max_rdma_chunked_recv_tokens"
            )


def _validate_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_size_hint_inputs(hidden_bytes: int, num_ranks: int) -> None:
    for name, value in (("hidden_bytes", hidden_bytes), ("num_ranks", num_ranks)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if hidden_bytes <= 0:
        raise ValueError("hidden_bytes must be greater than zero")
    if num_ranks < 2 or num_ranks > _MAX_SHMEM_INTRANODE_RANKS:
        raise ValueError(f"num_ranks must be between 2 and {_MAX_SHMEM_INTRANODE_RANKS}")


def _resolve_alias(
    primary_name: str,
    primary_value: int,
    primary_default: int,
    alias_name: str,
    alias_value: int | None,
) -> int:
    if alias_value is None:
        return primary_value
    if primary_value != primary_default and primary_value != alias_value:
        raise ValueError(f"{primary_name} and {alias_name} must not conflict")
    return alias_value
