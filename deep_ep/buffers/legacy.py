# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""V1 API with native resource primitives; EP algorithms remain unavailable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from .._native import _unavailable, require_native
from .._runtime import Runtime, runtime_of, warn_unclosed

if TYPE_CHECKING:
    import mpi4py.MPI
    import torch
    import torch.distributed as dist
    from .._config import Config
    from ..utils.event import EventOverlap


class Buffer:
    """Own a restricted native resource session; see docs/api.md."""

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        num_scaleup_bytes: int = 0,
        num_scaleout_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_memory_transport_for_low_latency_mode: bool = True,
        allow_multinode_memory_transport: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        comm: Optional["mpi4py.MPI.Comm"] = None,
    ) -> None:
        """Create explicit-size MTE resources, not an EP algorithm backend.

        Scale-up/out describe domains, not transport engines. Memory-transport
        flags do not enable every scale-up backend. num_qps_per_rank is a
        reserved compatibility value, not a shared DMA queue/channel count.
        See docs/api.md for the transport configuration boundary.
        """
        require_native("Buffer.__init__")
        if (
            num_scaleout_bytes != 0
            or low_latency_mode is not False
            or allow_multinode_memory_transport is not False
            or enable_shrink is not False
            or comm is not None
            or num_qps_per_rank != 24
            or allow_memory_transport_for_low_latency_mode is not True
        ):
            raise NotImplementedError("Buffer supports only the single-domain MTE resource runtime")
        if explicitly_destroy is not True:
            raise NotImplementedError("Runtime currently requires explicitly_destroy=True")
        self._runtime = Runtime(group, num_scaleup_bytes)

    def __del__(self):
        warn_unclosed(self)

    def destroy(self):
        """Collectively release owned resources; successful repetition is safe."""
        require_native("Buffer.destroy")
        runtime_of(self).close()

    @staticmethod
    def set_num_compute_units(new_num_compute_units: int) -> None:
        """Declare Buffer.set_num_compute_units; execution is not implemented."""
        _unavailable("Buffer.set_num_compute_units")

    @staticmethod
    def capture() -> EventOverlap:
        """Capture a real event on the current framework stream."""
        require_native("Buffer.capture")
        from ..utils.event import EventHandle, EventOverlap

        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_scaleout_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """Declare Buffer.get_low_latency_scaleout_size_hint; execution is not implemented."""
        _unavailable("Buffer.get_low_latency_scaleout_size_hint")

    def get_comm_stream(self) -> torch.Stream:
        """Return the live framework communication stream."""
        require_native("Buffer.get_comm_stream")
        runtime = runtime_of(self)
        runtime.ensure_alive()
        return runtime.stream

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """Declare Buffer.get_dispatch_config; execution is not implemented."""
        _unavailable("Buffer.get_dispatch_config")

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """Declare Buffer.get_combine_config; execution is not implemented."""
        _unavailable("Buffer.get_combine_config")

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """Declare Buffer.get_dispatch_layout; execution is not implemented."""
        _unavailable("Buffer.get_dispatch_layout")

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_scaleout_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Declare Buffer.dispatch; execution is not implemented."""
        _unavailable("Buffer.dispatch")

    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Declare Buffer.combine; execution is not implemented."""
        _unavailable("Buffer.combine")

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """Declare Buffer.clean_low_latency_buffer; execution is not implemented."""
        _unavailable("Buffer.clean_low_latency_buffer")

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """Declare Buffer.low_latency_dispatch; execution is not implemented."""
        _unavailable("Buffer.low_latency_dispatch")

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """Declare Buffer.low_latency_combine; execution is not implemented."""
        _unavailable("Buffer.low_latency_combine")

    def get_next_low_latency_combine_buffer(self, handle: object):
        """Declare Buffer.get_next_low_latency_combine_buffer; execution is not implemented."""
        _unavailable("Buffer.get_next_low_latency_combine_buffer")
