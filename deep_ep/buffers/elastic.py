# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""V2 expanded dispatch/combine and their Notify metadata orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from .._native import _unavailable, npu_api, require_native
from .._runtime import Runtime, rank_and_world, runtime_of, warn_unclosed
from ._elastic_layout import combine_layout, dispatch_layout, elastic_size_hint, integer, workspace_layout

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    from ..utils.event import EventHandle, EventOverlap


_PERSISTENT = (
    "server_mask",
    "dst",
    "forward_list",
    "forward_counts",
    "backward_list",
    "backward_counts",
    "chunk_ranges",
    "chunk_masks",
    "rank_rows",
    "gather_rows",
    "expert_totals",
    "weight_return_meta",
)


def _collective_check(runtime, error, signature):
    """Propagate local prelaunch failures before any rank enters a device collective."""
    if runtime.group is None:
        if error is not None:
            raise error
        return
    import torch
    import torch.distributed as dist

    values = [int(error is not None), *signature]
    device = f"npu:{runtime.device}" if str(dist.get_backend(runtime.group)) == "hccl" else "cpu"
    local = torch.tensor(values, dtype=torch.int64, device=device)
    peers = [torch.empty_like(local) for _ in range(runtime.world_size)]
    dist.all_gather(peers, local, group=runtime.group)
    gathered = [peer.cpu().tolist() for peer in peers]
    failed = [rank for rank, row in enumerate(gathered) if row[0]]
    if failed:
        raise ValueError(f"Notify preflight failed on ranks {failed}: {error or 'see the failing rank log'}")
    if any(row[1:] != gathered[0][1:] for row in gathered):
        raise ValueError("Notify ranks disagree on T/K/E/hidden or configured token capacity")


class EPHandle:
    """Routing metadata owner; metadata_only distinguishes notify from payload dispatch."""

    def __init__(
        self,
        do_expand: bool,
        num_experts: int,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        num_compute_units: int,
        topk_idx: torch.Tensor,
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_recv_tokens_per_expert_list: list,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        dst_buffer_slot_idx: torch.Tensor,
        token_metadata_at_forward: Optional[torch.Tensor],
        channel_linked_list: Optional[torch.Tensor],
        is_low_latency: bool = False,
    ):
        require_native("EPHandle.__init__")
        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_compute_units = num_compute_units
        self.topk_idx = topk_idx
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list
        self.is_low_latency = is_low_latency
        self.token_type = None
        self.destination_index = None
        self.relay_read_index = None
        self.source_mask = None
        self.source_weights = None
        self.send_counts = None
        self.expert_recv_counts = None
        self.server_mask = None
        self.dst = None
        self.forward_list = None
        self.forward_counts = None
        self.backward_list = None
        self.backward_counts = None
        self.chunk_ranges = None
        self.chunk_masks = None
        self.rank_rows = None
        self.gather_rows = None
        self.expert_totals = None
        self.weight_return_meta = None
        self.dispatch_dst = None
        self.address_stride = None
        self.rank = None
        self.world_size = None
        self.num_tokens = None
        self.num_topk = None
        self.chunk_tokens = None
        self.trace = None
        self.hidden = None
        self.metadata_only = True
        self.metadata_version = 6
        self._buffer_token = None
        self._routing_versions = {}


class ElasticBuffer:
    """Own a restricted native resource session; see docs/api.md."""

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        num_bytes: Optional[int] = None,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_device_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """Initialize MTE+URMA using a group or explicit rank/world_size.

        Explicit identities bootstrap SHMEM directly and only validate locally.
        All ranks must agree on configuration and collective call order.
        num_bytes is the entire symmetric workspace, cleared at construction.
        Dispatch validates exactly 8 GiB when selecting LL; no extra allocation.
        """
        require_native("ElasticBuffer.__init__")
        if (
            num_cpu_bytes != 0
            or deterministic is not False
            or allow_hybrid_mode is not True
            or allow_multiple_reduction is not True
            or prefer_overlap_with_compute is not True
            or sl_idx != 3
            or type(num_allocated_qps) is not int
            or num_allocated_qps not in (0, 6)
            or num_device_timeout_secs != 100
        ):
            raise NotImplementedError("ElasticBuffer supports default options and num_allocated_qps=0/6")
        if type(use_fp8_dispatch) is not bool:
            raise TypeError("use_fp8_dispatch must be bool")
        if explicitly_destroy is not True:
            raise NotImplementedError("Runtime currently requires explicitly_destroy=True")
        integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 0, 10240)
        integer("num_topk", num_topk, 0, 16)
        if type(hidden) is not int or hidden not in (0, 7168):
            raise ValueError("Dispatch supports hidden=7168")
        if num_bytes is None:
            if not num_max_tokens_per_rank or not num_topk or hidden != 7168:
                raise NotImplementedError("Pass num_bytes or provide token/topk bounds and hidden=7168")
            num_bytes = self.get_buffer_size_hint(
                group, num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch, world_size=world_size
            )
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self._handle_token = object()
        self._runtime = Runtime(group, num_bytes, num_cpu_timeout_secs, udma=True, rank=rank, world_size=world_size)

    def __del__(self):
        warn_unclosed(self)

    def destroy(self) -> None:
        """Collectively release owned resources; successful repetition is safe."""
        require_native("ElasticBuffer.destroy")
        runtime_of(self).close()

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        *,
        world_size: Optional[int] = None,
    ) -> int:
        """Two SHMEM arenas sized for 1024 experts and four times average rows.

        Heavier skew requires explicit num_bytes; dispatch checks actual counts
        before payload launch. Size hints include weights and optional FP8 scales; combine
        uses BF16 expert and gateway rows in the reserved second half. EP256 shapes
        beyond the default Combine UB limit receive a dispatch-only hint; the hint
        does not guarantee that Combine supports the shape.
        """
        require_native("ElasticBuffer.get_buffer_size_hint")
        if type(hidden) is not int or hidden != 7168:
            raise ValueError("Dispatch sizing requires hidden=7168")
        if allow_hybrid_mode is not True or allow_multiple_reduction is not True:
            raise NotImplementedError("Dispatch sizing supports only default algorithm options")
        _, world = rank_and_world(group, 0 if group is None else None, world_size)
        return elastic_size_hint(
            world,
            num_max_tokens_per_rank,
            num_topk,
            1024,
            min(world, 4) * num_max_tokens_per_rank * num_topk,
            min(world, 4) * num_max_tokens_per_rank * num_topk,
            use_fp8_dispatch,
        )

    def barrier(
        self,
        use_comm_stream: bool = True,
        with_cpu_sync: bool = False,
        sequential: bool = True,
    ) -> None:
        """Run the explicitly selected host-synchronous sequential barrier."""
        require_native("ElasticBuffer.barrier")
        if type(use_comm_stream) is not bool or with_cpu_sync is not True or sequential is not True:
            raise NotImplementedError("Only the host-synchronous sequential barrier is implemented")
        runtime_of(self).barrier()

    @staticmethod
    def capture() -> EventHandle:
        """Capture a real event on the current framework stream."""
        require_native("ElasticBuffer.capture")
        from ..utils.event import EventHandle

        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """Return the live framework communication stream."""
        require_native("ElasticBuffer.get_comm_stream")
        runtime = runtime_of(self)
        runtime.ensure_alive()
        return runtime.stream

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_physical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_physical_domain_size")

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_logical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_logical_domain_size")

    def get_theoretical_num_compute_units(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        scaleout_gbs: float = 0,
        scaleup_gbs: float = 0,
        compute_read_gbs: float = 200,
        compute_write_gbs: float = 50,
    ) -> int:
        """Declare ElasticBuffer.get_theoretical_num_compute_units; execution is not implemented."""
        _unavailable("ElasticBuffer.get_theoretical_num_compute_units")

    def get_theoretical_num_qps(self, num_compute_units: int) -> int:
        """Declare a reserved QP sizing query, not a generic DMA resource count."""
        _unavailable("ElasticBuffer.get_theoretical_num_qps")

    def _notify_low_latency(self, indices: torch.Tensor, num_experts: int) -> Optional[int]:
        """Return exact expanded rows using the existing EP group, or None."""
        runtime = runtime_of(self)
        if runtime.group is None:
            return None
        import torch
        import torch.distributed as dist

        if dist.get_rank(runtime.group) != runtime.rank or dist.get_world_size(runtime.group) != runtime.world_size:
            raise ValueError("LL count group must match the buffer EP rank order")
        backend = str(dist.get_backend(runtime.group))
        if backend not in ("hccl", "gloo"):
            raise NotImplementedError("LL counts require the existing HCCL or Gloo EP group")
        ids = indices.reshape(-1)
        valid = (ids >= 0) & (ids < num_experts)
        destinations = torch.where(valid, ids // (num_experts // runtime.world_size), 0)
        counts = torch.zeros(runtime.world_size, dtype=torch.int32, device=indices.device)
        counts.scatter_add_(0, destinations.to(torch.int64), valid.to(torch.int32))
        if backend == "gloo":
            counts = counts.cpu()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=runtime.group)
        return int(counts[runtime.rank].item())

    @staticmethod
    def _upload_low_latency(data: bytes, device: torch.device) -> torch.Tensor:
        """Upload immutable ABI bytes using allocator-owned pinned Host storage."""
        import torch

        source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        pinned = torch.empty(source.shape, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(source)
        return pinned.to(device=device, non_blocking=True)

    def _low_latency_plan(self, x, topk_idx, topk_weights, num_experts, capacity):
        """Check only Host metadata before selecting LL; never read routes."""
        import torch

        runtime = runtime_of(self)
        hidden, scales = x if isinstance(x, tuple) and len(x) == 2 else (x, None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
            return None
        b, h = hidden.shape
        if (
            type(capacity) is not int
            or not b <= capacity <= 256
            or runtime.world_size not in (16, 32, 64, 128)
            or topk_weights is None
        ):
            return None
        if hidden.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            return None
        if not isinstance(topk_idx, torch.Tensor) or topk_idx.ndim != 2:
            return None
        k = topk_idx.shape[1]
        if type(num_experts) is not int or num_experts < runtime.world_size or num_experts % runtime.world_size:
            return None
        if (self.hidden and h != self.hidden) or (self.num_topk and k != self.num_topk):
            return None
        if hidden.dtype == torch.float8_e4m3fn:
            if (
                not isinstance(scales, torch.Tensor)
                or scales.dtype not in (torch.int32, torch.float32)
                or tuple(scales.shape) != (b, (h + 127) // 128)
            ):
                return None
        elif scales is not None:
            return None
        for value, dtype, shape in (
            (hidden, hidden.dtype, (b, h)),
            (topk_idx, torch.int64, (b, k)),
            (topk_weights, torch.float32, (b, k)),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                return None
        if scales is not None and not scales.is_contiguous():
            return None
        return require_native("ElasticBuffer.dispatch").low_latency_layout(
            runtime.world_size, b, h, k, num_experts, str(hidden.dtype).removeprefix("torch.")
        )

    def _dispatch_low_latency(
        self, x, topk_idx, topk_weights, num_experts, capacity, do_handle_copy, do_cpu_sync, plan
    ):
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        native, npu = require_native("ElasticBuffer.dispatch"), npu_api()
        hidden, scales = x if isinstance(x, tuple) else (x, None)
        b, h = hidden.shape
        k = topk_idx.shape[1]
        dtype = str(hidden.dtype).removeprefix("torch.")
        with runtime.notify_lock:
            if runtime.notify_failed:
                raise RuntimeError("A previous communication launch failed; restart the distributed job")
            producer = EventHandle()
            with npu.stream(runtime.stream):
                producer.current_stream_wait()
                indices = topk_idx.clone() if do_handle_copy else topk_idx
                for value in (hidden, topk_idx, indices, topk_weights, scales):
                    if value is not None:
                        value.record_stream(runtime.stream)
                # Every allocation is kept alive on error after the first collective.
                retained = (hidden, indices, topk_weights) + (() if scales is None else (scales,))
                runtime.notify_inflight = retained
                try:
                    rows = self._notify_low_latency(indices, num_experts) if do_cpu_sync is not False else None
                    c = plan["capacity_rows"]
                    output_rows = (
                        runtime.world_size * capacity * min(k, num_experts // runtime.world_size) if rows is None else c
                    )
                    if runtime.low_latency_comm_args is None:
                        runtime.low_latency_comm_args = self._upload_low_latency(
                            native.low_latency_comm_args(runtime.native), hidden.device
                        )
                    tensors = {
                        "input": hidden,
                        "indices": indices.to(torch.int32),
                        "weights": topk_weights,
                        "scales": scales,
                        "output": torch.empty((output_rows, h), dtype=hidden.dtype, device=hidden.device),
                        "output_scales": torch.empty(
                            (output_rows, (h + 127) // 128), dtype=scales.dtype, device=hidden.device
                        )
                        if scales is not None
                        else None,
                        "expert_recv_counts": torch.empty(
                            (num_experts // runtime.world_size,), dtype=torch.int64, device=hidden.device
                        ),
                        "send_counts": torch.empty((num_experts,), dtype=torch.int32, device=hidden.device),
                        "token_type": torch.empty((c,), dtype=torch.int32, device=hidden.device),
                        "destination_index": torch.empty((c, 3), dtype=torch.int32, device=hidden.device),
                        "relay_read_index": torch.empty((c, k, 2), dtype=torch.int32, device=hidden.device),
                        "source_mask": torch.empty((b,), dtype=torch.int32, device=hidden.device),
                        "source_weights": torch.empty((b, k), dtype=torch.float32, device=hidden.device),
                        "comm_args": runtime.low_latency_comm_args,
                        "tiling": self._upload_low_latency(
                            native.low_latency_dispatch_tiling(
                                runtime.world_size, b, h, k, num_experts, dtype, capacity
                            ),
                            hidden.device,
                        ),
                    }
                    retained += tuple(value for value in tensors.values() if value is not None)
                    runtime.notify_inflight = retained
                    for value in retained:
                        value.record_stream(runtime.stream)
                    native.dispatch_low_latency(
                        runtime.native, tensors, b, h, k, num_experts, output_rows, dtype, runtime.stream.npu_stream
                    )
                    result = EPHandle(
                        True,
                        num_experts,
                        1,
                        capacity,
                        0,
                        indices,
                        runtime.world_size * capacity if rows is None else None,
                        output_rows if rows is None else rows,
                        [] if rows is None else None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        is_low_latency=True,
                    )
                    result.token_type = tensors["token_type"]
                    result.destination_index = tensors["destination_index"]
                    result.relay_read_index = tensors["relay_read_index"]
                    result.source_mask = tensors["source_mask"]
                    result.source_weights = tensors["source_weights"]
                    result.send_counts = tensors["send_counts"]
                    # Expert counts remain kernel scratch; handle count tensors stay None.
                    result.metadata_only = False
                    result._buffer_token = self._handle_token
                    result.rank, result.world_size = runtime.rank, runtime.world_size
                    result.num_tokens, result.num_topk, result.hidden = b, k, h
                    out, out_scales = tensors["output"], tensors["output_scales"]
                    if rows is not None:
                        out = out.narrow(0, 0, rows)
                        if out_scales is not None:
                            out_scales = out_scales.narrow(0, 0, rows)
                    event = EventOverlap(EventHandle(), retained)
                except Exception:
                    runtime.notify_failed = True
                    raise
            event.current_stream_wait()
            runtime.notify_inflight = None
            return (out, out_scales) if scales is not None else out, None, None, result, event

    def _combine_low_latency(
        self,
        x,
        handle,
        topk_weights,
        bias,
        num_compute_units,
        num_qps,
        previous_event,
        previous_event_before_epilogue,
        async_with_compute_stream,
        allocate_on_comm_stream,
    ):
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        runtime.ensure_alive()
        if handle._buffer_token is not self._handle_token or handle.metadata_only:
            raise ValueError("LL handle must come from this buffer's payload dispatch")
        if topk_weights is not None or bias is not None:
            raise NotImplementedError("LL combine takes its weights from the handle and does not support bias")
        if (
            type(num_compute_units) is not int
            or num_compute_units != 0
            or type(num_qps) is not int
            or num_qps != 0
            or previous_event is not None
            or previous_event_before_epilogue is not None
            or async_with_compute_stream is not False
            or allocate_on_comm_stream is not False
        ):
            raise NotImplementedError("LL combine supports default scheduling and stream options only")
        if (
            not isinstance(x, torch.Tensor)
            or x.dtype not in (torch.bfloat16, torch.float16)
            or tuple(x.shape) != (handle.num_expanded_tokens, handle.hidden)
            or not x.is_contiguous()
        ):
            raise ValueError("LL combine requires contiguous BF16/FP16 expert output matching its handle")
        native, npu = require_native("ElasticBuffer.combine"), npu_api()
        b, k, h, e = handle.num_tokens, handle.num_topk, handle.hidden, handle.num_experts
        dtype = str(x.dtype).removeprefix("torch.")
        tiling = native.low_latency_combine_tiling(
            runtime.world_size, b, h, k, e, dtype, handle.num_max_tokens_per_rank
        )
        with runtime.notify_lock:
            if runtime.notify_failed:
                raise RuntimeError("A previous communication launch failed; restart the distributed job")
            producer = EventHandle()
            with npu.stream(runtime.stream):
                producer.current_stream_wait()
                # The LL kernel reads selected valid rows, each with exactly 2H
                # bytes. Expert storage need not cover logical capacity C_B.
                tensors = {
                    "input": x,
                    "output": torch.empty((b, h), dtype=x.dtype, device=x.device),
                    "destination_index": handle.destination_index,
                    "send_counts": handle.send_counts,
                    "source_weights": handle.source_weights,
                    "token_type": handle.token_type,
                    "relay_read_index": handle.relay_read_index,
                    "source_mask": handle.source_mask,
                    "comm_args": runtime.low_latency_comm_args,
                    "tiling": self._upload_low_latency(tiling, x.device),
                }
                retained = tuple(tensors.values())
                for value in retained:
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("LL handle is missing a required tensor")
                    value.record_stream(runtime.stream)
                runtime.notify_inflight = retained
                try:
                    native.combine_low_latency(runtime.native, tensors, b, h, k, e, dtype, runtime.stream.npu_stream)
                    event = EventOverlap(EventHandle(), retained)
                except Exception:
                    runtime.notify_failed = True
                    raise
            event.current_stream_wait()
            runtime.notify_inflight = None
            return tensors["output"], None, event

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: Optional[bool] = None,
        do_zero_padding: bool = False,
        use_aligned_col_major_sf: bool = False,
    ) -> Tuple[
        Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
        EventOverlap,
    ]:
        """Return (expanded_x, None, expanded_weights, handle, event).

        Supports BF16 or (FP8 E4M3FN, FP32 [T,H/128] scales), H=7168,
        Fresh calls with resolved num_max_tokens_per_rank <= 256 prefer LL;
        unsupported LL options use HT only before payload launch. HT supports
        fresh/cached expanded routing with expert_alignment=1, num_qps=0/2,
        and num_compute_units=0/32 for BF16 or 0/64 for FP8. These hints do
        not change the peer-table rear-core schedule.
        LL returns no expanded weights: its combine applies the saved weights
        internally, so expert output must not be preweighted. With CPU sync
        and an existing EP group LL returns exact R rows; otherwise it returns
        P*M*min(K,E/P) capacity rows. Default M resolution is unchanged.
        """
        require_native("ElasticBuffer.dispatch")
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        runtime.ensure_alive()
        native, npu = require_native("ElasticBuffer.dispatch"), npu_api()
        if isinstance(handle, EPHandle) and handle.is_low_latency:
            raise NotImplementedError("Cached dispatch is not supported for LL handles")
        if handle is None:
            hidden = x[0] if isinstance(x, tuple) and len(x) == 2 else x
            if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
                capacity = (
                    (self.num_max_tokens_per_rank or hidden.shape[0])
                    if num_max_tokens_per_rank is None
                    else num_max_tokens_per_rank
                )
                within_bound = type(capacity) is int and (
                    not self.num_max_tokens_per_rank or capacity <= self.num_max_tokens_per_rank
                )
                ll_options = (
                    within_bound
                    and type(do_handle_copy) is bool
                    and (do_cpu_sync is None or type(do_cpu_sync) is bool)
                    and (do_expand is None or do_expand is True)
                    and (expert_alignment is None or type(expert_alignment) is int and expert_alignment == 1)
                    and type(num_compute_units) is int
                    and num_compute_units == 0
                    and type(num_qps) is int
                    and num_qps == 0
                    and cumulative_local_expert_recv_stats is None
                    and previous_event is None
                    and previous_event_before_epilogue is None
                    and async_with_compute_stream is False
                    and allocate_on_comm_stream is False
                    and do_zero_padding is False
                    and use_aligned_col_major_sf is False
                )
                plan = self._low_latency_plan(x, topk_idx, topk_weights, num_experts, capacity) if ll_options else None
                if plan is not None:
                    if runtime.capacity != 8 << 30:
                        raise ValueError(
                            "LL dispatch requires Buffer(num_bytes=8589934592), exactly 8 GiB; "
                            f"got {runtime.capacity} bytes for EP{runtime.world_size}"
                        )
                    return self._dispatch_low_latency(
                        x, topk_idx, topk_weights, num_experts, capacity, do_handle_copy, do_cpu_sync, plan
                    )
        with runtime.notify_lock:
            error, signature = None, [0] * 8
            try:
                if runtime.notify_failed:
                    raise RuntimeError("A previous communication launch failed; restart the distributed job")
                for name, value in (
                    ("async_with_compute_stream", async_with_compute_stream),
                    ("allocate_on_comm_stream", allocate_on_comm_stream),
                    ("do_handle_copy", do_handle_copy),
                    ("do_zero_padding", do_zero_padding),
                    ("use_aligned_col_major_sf", use_aligned_col_major_sf),
                ):
                    if type(value) is not bool:
                        raise TypeError(f"{name} must be bool")
                if do_expand is not None and do_expand is not True:
                    raise NotImplementedError("Dispatch supports only the expanded expert layout")
                if do_cpu_sync is not None and do_cpu_sync is not True:
                    raise NotImplementedError("Dispatch preflight requires do_cpu_sync=None/True")
                if expert_alignment is not None and (type(expert_alignment) is not int or expert_alignment != 1):
                    raise NotImplementedError("Dispatch supports expert_alignment=1")
                if (
                    cumulative_local_expert_recv_stats is not None
                    or previous_event_before_epilogue is not None
                    or allocate_on_comm_stream
                    or do_zero_padding
                    or use_aligned_col_major_sf
                ):
                    raise NotImplementedError(
                        "Stats, epilogue events, padding and alternate allocation/scale layouts are unavailable"
                    )
                if previous_event is not None and (
                    not isinstance(previous_event, EventHandle) or previous_event._device != runtime.device
                ):
                    raise ValueError("previous_event must belong to the buffer NPU")
                fp8 = isinstance(x, tuple)
                if fp8 and len(x) != 2:
                    raise ValueError("FP8 input must be (hidden, scales)")
                hidden, scales = x if fp8 else (x, None)

                def check_tensor(value, dtype, shape=None):
                    return (
                        isinstance(value, torch.Tensor)
                        and value.dtype == dtype
                        and value.device.type == "npu"
                        and value.device.index == runtime.device
                        and value.is_contiguous()
                        and not value.requires_grad
                        and (shape is None or tuple(value.shape) == tuple(shape))
                    )

                if not check_tensor(hidden, torch.float8_e4m3fn if fp8 else torch.bfloat16) or hidden.ndim != 2:
                    raise ValueError(
                        "x must be contiguous BF16, or (FP8 E4M3FN, FP32 scales), on the buffer NPU without grad"
                    )
                tokens, width = hidden.shape
                if width != 7168:
                    raise ValueError("Dispatch supports hidden=7168")
                if fp8 and not check_tensor(scales, torch.float32, (tokens, 56)):
                    raise ValueError("FP8 scales must be contiguous FP32 [T,56] on the buffer NPU without grad")
                if type(num_compute_units) is not int or num_compute_units not in (0, 64 if fp8 else 32):
                    raise ValueError("num_compute_units must be 0/32 for BF16 or 0/64 for FP8")
                if type(num_qps) is not int or num_qps not in (0, 2):
                    raise ValueError("Primary dispatch uses two QPs per peer; num_qps=0/2")
                if handle is not None:
                    if (
                        not isinstance(handle, EPHandle)
                        or handle._buffer_token is not self._handle_token
                        or handle.metadata_only
                        or handle.metadata_version != 6
                    ):
                        raise ValueError("Cached handle must come from this buffer's successful payload dispatch")
                    if topk_idx is not None:
                        raise ValueError("Cached dispatch inherits routing; omit topk_idx")
                    if num_experts is not None and num_experts != handle.num_experts:
                        raise ValueError("num_experts differs from cached routing")
                    for name, version in handle._routing_versions.items():
                        tensor = getattr(handle, name)
                        if (tensor.data_ptr(), tensor._version) != version:
                            raise ValueError(f"Cached routing was modified: {name}")
                    num_experts, topk = handle.num_experts, handle.num_topk
                    if tokens != handle.num_tokens:
                        raise ValueError("Cached dispatch requires the original token count")
                else:
                    if not check_tensor(topk_idx, torch.int64) or topk_idx.ndim != 2 or topk_idx.shape[0] != tokens:
                        raise ValueError("topk_idx must be contiguous int64 [T,K] on the buffer NPU")
                    topk = topk_idx.shape[1]
                workspace_layout(runtime.world_size, tokens, topk, num_experts)
                bound = self.num_max_tokens_per_rank
                num_max_tokens_per_rank = (
                    (bound or tokens) if num_max_tokens_per_rank is None else num_max_tokens_per_rank
                )
                integer("num_max_tokens_per_rank", num_max_tokens_per_rank, tokens, 10240)
                if bound and num_max_tokens_per_rank > bound:
                    raise ValueError("Input/capacity exceeds the buffer token bound")
                if self.num_topk and self.num_topk != topk:
                    raise ValueError("topk differs from buffer configuration")
                if topk_weights is not None and not check_tensor(topk_weights, torch.float32, (tokens, topk)):
                    raise ValueError("topk_weights must be contiguous FP32 [T,K] on the buffer NPU without grad")
                native.notify_check_mapping(runtime.native)
                signature = [
                    tokens,
                    topk,
                    num_experts,
                    width,
                    num_max_tokens_per_rank,
                    int(fp8),
                    int(topk_weights is not None),
                    int(handle is not None),
                ]
            except Exception as exc:
                error = exc
            _collective_check(runtime, error, signature)

            if handle is None:
                _, _, _, handle, _ = self._notify(
                    hidden,
                    topk_idx,
                    topk_weights,
                    None,
                    num_experts,
                    num_max_tokens_per_rank,
                    1,
                    32,
                    0,
                    previous_event,
                    None,
                    True,
                    False,
                    None,
                    do_handle_copy,
                    True,
                    True,
                    False,
                    False,
                )
            error, tensors = None, {}
            payload_signature = signature + [0, int(runtime.native.peer_table), int(hasattr(handle, "_peer_plan"))]
            try:
                rows, local_rows = handle.address_stride, handle.num_expanded_tokens
                layout = dispatch_layout(
                    runtime.world_size, tokens, topk, num_experts, rows, fp8, weights=topk_weights is not None
                )
                if layout["workspace_bytes"] > runtime.capacity // 2:
                    raise ValueError(
                        f"Dispatch needs at least {2 * layout['num_bytes']} SHMEM bytes per rank for the actual receive counts"
                    )
                native_layout = native.dispatch_layout(
                    runtime.rank, runtime.world_size, tokens, topk, num_experts, rows, fp8, topk_weights is not None
                )
                if native_layout.get("aggregate_protocol") != layout["aggregate_protocol"] or any(
                    layout[key] != value for key, value in native_layout.items()
                ):
                    raise RuntimeError("Python/native dispatch layouts differ; rebuild the installed package")
                payload_signature[-3] = rows
                if previous_event is not None:
                    previous_event.current_stream_wait()
                producer = EventHandle()
                with npu.stream(runtime.stream):
                    producer.current_stream_wait()
                    if getattr(handle, "dispatch_dst", None) is None:
                        # Preserve Notify's original stride/sentinel ABI for Combine.
                        raw = handle.dst.to(torch.int64)
                        flat = torch.where(raw < 0, ~raw, raw)
                        encoded = flat // rows * layout["address_stride"] + flat % rows
                        encoded = torch.where(raw < 0, ~encoded, encoded)
                        handle.dispatch_dst = torch.where(raw == -(1 << 31), raw, encoded).to(torch.int32)
                    out = torch.empty((local_rows, width), dtype=hidden.dtype, device=hidden.device)
                    out_weights = (
                        torch.empty((local_rows,), dtype=torch.float32, device=hidden.device)
                        if topk_weights is not None
                        else None
                    )
                    out_scales = (
                        torch.empty((local_rows, 56), dtype=torch.float32, device=hidden.device) if fp8 else None
                    )
                    peer_plan = getattr(handle, "_peer_plan", None)
                    if runtime.native.peer_table and peer_plan is None:
                        raise ValueError("Dispatch requires the peer table generated by Notify; recreate the handle")
                    tensors = dict(
                        plan=peer_plan,
                        input=hidden,
                        output=out,
                        dst=handle.dispatch_dst,
                        forward_list=handle.forward_list,
                        forward_counts=handle.forward_counts,
                        weights=topk_weights,
                        output_weights=out_weights,
                        scales=scales,
                        output_scales=out_scales,
                    )
                    # Private benchmark hook; ordinary API calls do not allocate
                    # profile buffers or create device timing events.
                    diagnostics = getattr(self, "_dispatch_diagnostics", None)
                    if diagnostics is not None:
                        tensors["profile"] = diagnostics.get("profile")
                        tensors["weight_profile"] = diagnostics.get("weight_profile")
                        tensors["rotation_wqes"] = diagnostics.get("rotation_wqes", 4)
                        tensors["detail_profile"] = diagnostics.get("detail_profile")
                        tensors["detail_mode"] = diagnostics.get("detail_mode", 0)
                    retained = tuple(value for value in tensors.values() if torch.is_tensor(value)) + (
                        handle.topk_idx,
                        *(getattr(handle, name) for name in _PERSISTENT),
                    )
                    for tensor in retained:
                        tensor.record_stream(runtime.stream)
                    # Catch allocations/input copies and drain the previous call on
                    # every rank before scratch addresses can change or be reused.
                    runtime.stream.synchronize()
            except Exception as exc:
                error = exc
            _collective_check(runtime, error, payload_signature)
            runtime.notify_inflight = retained
            try:
                with npu.stream(runtime.stream):
                    if diagnostics is not None:
                        tensors["kernel_timing"] = True
                    device_times = native.dispatch(
                        runtime.native,
                        tensors,
                        tokens,
                        topk,
                        num_experts,
                        rows,
                        local_rows,
                        fp8,
                        runtime.stream.npu_stream,
                    )
                    if diagnostics is not None:
                        diagnostics["device_ms"] = device_times
                    handle.metadata_only = False
                    handle.num_compute_units = 64 if fp8 else 32
                    handle._routing_versions = {
                        name: (getattr(handle, name).data_ptr(), getattr(handle, name)._version)
                        for name in (
                            "topk_idx",
                            "dispatch_dst",
                            *_PERSISTENT,
                            *(("_peer_plan",) if peer_plan is not None else ()),
                        )
                    }
                    event = EventOverlap(EventHandle(), retained)
                if not async_with_compute_stream:
                    event.current_stream_wait()
                runtime.notify_inflight = None
                return (out, out_scales) if fp8 else out, None, out_weights, handle, event
            except Exception:
                runtime.notify_failed = True
                raise

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Combine expert rows using the backend recorded by the dispatch handle.

        LL applies handle.source_weights internally to unweighted BF16/FP16
        expert results. HT sums BF16 expert results without implicit weighting.
        Both return (output, None, event) and keep the handle reusable. No
        backend fallback is performed by combine.
        """
        require_native("ElasticBuffer.combine")
        import torch
        from ..utils.event import EventHandle, EventOverlap

        if isinstance(handle, EPHandle) and handle.is_low_latency:
            return self._combine_low_latency(
                x,
                handle,
                topk_weights,
                bias,
                num_compute_units,
                num_qps,
                previous_event,
                previous_event_before_epilogue,
                async_with_compute_stream,
                allocate_on_comm_stream,
            )
        runtime = runtime_of(self)
        runtime.ensure_alive()
        native, npu = require_native("ElasticBuffer.combine"), npu_api()
        with runtime.notify_lock:
            error, signature, retained = None, [0] * 7, ()
            try:
                if runtime.notify_failed:
                    raise RuntimeError("A previous communication launch failed; restart the distributed job")
                if type(async_with_compute_stream) is not bool or type(allocate_on_comm_stream) is not bool:
                    raise TypeError("Stream options must be bool")
                if bias is not None:
                    raise NotImplementedError("Combine currently supports bias=None")
                if previous_event_before_epilogue is not None or allocate_on_comm_stream:
                    raise NotImplementedError("Epilogue events and allocate_on_comm_stream are unavailable")
                if type(num_compute_units) is not int or num_compute_units not in (0, 64):
                    raise ValueError("Combine uses 64 AIVs; num_compute_units=0/64")
                if type(num_qps) is not int or num_qps not in (0, 2):
                    raise ValueError("Combine uses two QPs per cross sender; num_qps=0/2")
                if previous_event is not None and (
                    not isinstance(previous_event, EventHandle) or previous_event._device != runtime.device
                ):
                    raise ValueError("previous_event must belong to the buffer NPU")
                if (
                    not isinstance(handle, EPHandle)
                    or handle._buffer_token is not self._handle_token
                    or handle.metadata_only
                    or handle.metadata_version != 6
                ):
                    raise ValueError("Combine needs this buffer's successful expanded payload dispatch handle")
                if not handle.do_expand or handle.expert_alignment != 1:
                    raise ValueError("Combine supports expanded layout without padding")
                if handle.rank != runtime.rank or handle.world_size != runtime.world_size:
                    raise ValueError("Handle topology differs from the buffer")
                for name, version in handle._routing_versions.items():
                    tensor = getattr(handle, name)
                    if (tensor.data_ptr(), tensor._version) != version:
                        raise ValueError(f"Cached routing was modified: {name}")
                local_rows = handle.num_expanded_tokens
                if (
                    not isinstance(x, torch.Tensor)
                    or x.dtype != torch.bfloat16
                    or x.device.type != "npu"
                    or x.device.index != runtime.device
                    or not x.is_contiguous()
                    or x.requires_grad
                    or tuple(x.shape) != (local_rows, 7168)
                ):
                    raise ValueError(
                        "x must be contiguous BF16 [num_expanded_tokens,7168] on the buffer NPU without grad"
                    )
                tokens, topk, chunk = handle.num_tokens, handle.num_topk, handle.chunk_tokens
                if topk_weights is not None and (
                    not isinstance(topk_weights, torch.Tensor)
                    or topk_weights.dtype != torch.float32
                    or topk_weights.device != x.device
                    or tuple(topk_weights.shape) != (local_rows,)
                    or not topk_weights.is_contiguous()
                    or topk_weights.requires_grad
                ):
                    raise ValueError(
                        "topk_weights must be contiguous FP32 [num_expanded_tokens] on the buffer NPU without grad"
                    )
                if previous_event is not None:
                    previous_event.current_stream_wait()
                producer = EventHandle()
                with npu.stream(runtime.stream):
                    producer.current_stream_wait()
                    rows = max(1, max(handle.rank_rows.cpu().tolist()))
                    gathered = max(1, max(handle.gather_rows.cpu().tolist()))
                    layout = combine_layout(runtime.world_size, tokens, topk, rows, gathered, chunk)
                    if layout["workspace_bytes"] > runtime.capacity // 2:
                        raise ValueError("Combine exceeds its reserved SHMEM half; increase num_bytes on every rank")
                    expected = native.combine_layout(
                        runtime.rank, runtime.world_size, tokens, topk, rows, gathered, chunk
                    )
                    if dict(expected) != layout:
                        raise RuntimeError("Python/native combine layouts differ; rebuild the installed package")
                    output = torch.empty((tokens, 7168), dtype=torch.bfloat16, device=x.device)
                    tensors = {
                        name: getattr(handle, name)
                        for name in (
                            "forward_list",
                            "forward_counts",
                            "backward_list",
                            "backward_counts",
                            "server_mask",
                            "chunk_ranges",
                            "chunk_masks",
                        )
                    }
                    tensors.update(input=x, output=output)
                    output_weights = None
                    if topk_weights is not None:
                        output_weights = torch.empty((tokens, topk), dtype=torch.float32, device=x.device)
                        tensors.update(
                            weights=topk_weights,
                            weight_output=output_weights,
                            weight_return_meta=handle.weight_return_meta,
                        )
                    retained = (x, output, handle.topk_idx, *(getattr(handle, name) for name in _PERSISTENT))
                    if output_weights is not None:
                        retained += (topk_weights, output_weights)
                    for tensor in retained:
                        tensor.record_stream(runtime.stream)
                    runtime.stream.synchronize()
                signature = [tokens, topk, chunk, rows, gathered, runtime.capacity, int(topk_weights is not None)]
            except Exception as exc:
                error = exc
            _collective_check(runtime, error, signature)
            runtime.notify_inflight = retained
            try:
                with npu.stream(runtime.stream):
                    native.combine(
                        runtime.native,
                        tensors,
                        tokens,
                        topk,
                        rows,
                        gathered,
                        local_rows,
                        chunk,
                        runtime.stream.npu_stream,
                    )
                    event = EventOverlap(EventHandle(), retained)
                if not async_with_compute_stream:
                    event.current_stream_wait()
                runtime.notify_inflight = None
                return output, output_weights, event
            except Exception:
                runtime.notify_failed = True
                raise

    def _notify(
        self,
        x,
        topk_idx,
        topk_weights,
        stats,
        num_experts,
        capacity,
        alignment,
        cores,
        qps,
        previous_event,
        previous_epilogue,
        async_with_compute,
        allocate_on_comm,
        handle,
        copy_handle,
        cpu_sync,
        expand,
        zero_padding,
        aligned_sf,
    ):
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        runtime.ensure_alive()
        native = require_native("ElasticBuffer.dispatch")
        npu = npu_api()
        with runtime.notify_lock:
            signature = [0] * 5
            error = None
            try:
                if runtime.notify_failed:
                    raise RuntimeError("A previous notify launch failed; recreate the distributed job")
                for name, value in (
                    ("async_with_compute_stream", async_with_compute),
                    ("allocate_on_comm_stream", allocate_on_comm),
                    ("do_handle_copy", copy_handle),
                    ("do_zero_padding", zero_padding),
                    ("use_aligned_col_major_sf", aligned_sf),
                ):
                    if type(value) is not bool:
                        raise TypeError(f"{name} must be bool")
                if expand is not True:
                    raise NotImplementedError("Full notify requires explicit do_expand=True")
                if handle is not None:
                    raise NotImplementedError("Only fresh notify is implemented; pass handle=None")
                if cpu_sync is not None and cpu_sync is not True:
                    raise NotImplementedError("Fresh notify requires do_cpu_sync=None/True")
                if alignment is not None and (type(alignment) is not int or alignment != 1):
                    raise NotImplementedError("Notify supports expert_alignment=1")
                if type(cores) is not int or cores not in (0, 32) or type(qps) is not int or qps != 0:
                    raise ValueError("Notify uses 32 AIVs and MTE; num_compute_units=0/32, num_qps=0")
                if allocate_on_comm or zero_padding or aligned_sf or stats is not None or previous_epilogue is not None:
                    raise NotImplementedError(
                        "Padding, stats, epilogue events and allocation options are not implemented"
                    )
                if previous_event is not None and not isinstance(previous_event, EventHandle):
                    raise TypeError("previous_event must be an EventHandle")
                if previous_event is not None and previous_event._device != runtime.device:
                    raise ValueError("previous_event belongs to another device")
                if (
                    not isinstance(x, torch.Tensor)
                    or x.ndim != 2
                    or x.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                    or x.device.type != "npu"
                    or x.device.index != runtime.device
                    or not x.is_contiguous()
                    or x.requires_grad
                ):
                    raise ValueError("x must be a contiguous BF16/FP8 [T,7168] NPU tensor without grad")
                tokens, hidden = x.shape
                if hidden != 7168:
                    raise ValueError("Notify requires hidden=7168")
                if (
                    not isinstance(topk_idx, torch.Tensor)
                    or topk_idx.ndim != 2
                    or topk_idx.dtype != torch.int64
                    or topk_idx.device != x.device
                    or topk_idx.shape[0] != tokens
                    or not topk_idx.is_contiguous()
                ):
                    raise ValueError("topk_idx must be contiguous int64 [T,K] on the buffer NPU")
                topk = topk_idx.shape[1]
                layout = workspace_layout(runtime.world_size, tokens, topk, num_experts)
                if topk_weights is not None and (
                    not isinstance(topk_weights, torch.Tensor)
                    or topk_weights.dtype != torch.float32
                    or topk_weights.device != x.device
                    or topk_weights.shape != topk_idx.shape
                    or not topk_weights.is_contiguous()
                    or topk_weights.requires_grad
                ):
                    raise ValueError("topk_weights must be contiguous FP32 [T,K] without grad")
                bound = self.num_max_tokens_per_rank
                capacity = (bound or tokens) if capacity is None else capacity
                integer("num_max_tokens_per_rank", capacity, tokens, 10240)
                if bound and (tokens > bound or capacity > bound):
                    raise ValueError("Input/capacity exceeds the buffer token bound")
                if self.num_topk and self.num_topk != topk:
                    raise ValueError("topk differs from buffer configuration")
                if layout["workspace_bytes"] > runtime.capacity // 2:
                    raise ValueError(
                        f"Notify requires at least {2 * layout['num_bytes']} SHMEM bytes per rank (reserved halves)"
                    )
                specification = native.notify_layout(runtime.rank, runtime.world_size, tokens, topk, num_experts)
                if specification["workspace_bytes"] != layout["workspace_bytes"]:
                    raise RuntimeError("Python/native notify layouts differ; rebuild the installed package")
                native.notify_check_mapping(runtime.native)
                signature = [tokens, topk, num_experts, hidden, capacity]
            except Exception as exc:
                error = exc
            _collective_check(runtime, error, signature)

            # All allocations and input dependencies succeed globally before launching the collective kernel.
            tensors = {}
            indices = topk_idx
            error = None
            try:
                if previous_event is not None:
                    previous_event.current_stream_wait()
                producer = EventHandle()
                with npu.stream(runtime.stream):
                    producer.current_stream_wait()
                    indices = topk_idx.clone() if copy_handle else topk_idx
                    topk_idx.record_stream(runtime.stream)
                    indices.record_stream(runtime.stream)
                    tensors = {
                        name: torch.empty(shape, dtype=getattr(torch, dtype), device=x.device)
                        for name, (shape, dtype) in specification["tensors"].items()
                    }
                    for tensor in tensors.values():
                        tensor.record_stream(runtime.stream)
                    # Catch allocation/input copy errors before device collectives.
                    runtime.stream.synchronize()
            except Exception as exc:
                error = exc
            _collective_check(runtime, error, signature)
            runtime.notify_inflight = (indices, tensors)
            try:
                with npu.stream(runtime.stream):
                    routing_diagnostics = getattr(self, "_routing_diagnostics", None)
                    if routing_diagnostics is not None:
                        tensors["kernel_timing"] = True
                    notify_ms, prepare_ms = native.notify(
                        runtime.native, indices, tensors, tokens, topk, num_experts, runtime.stream.npu_stream
                    )
                    if routing_diagnostics is not None:
                        routing_diagnostics["notify_device_ms"] = notify_ms
                        routing_diagnostics["prepare_device_ms"] = prepare_ms
                        routing_diagnostics["plan_in_notify"] = True
                    tensors.pop("kernel_timing", None)
                    runtime.stream.synchronize()
                    local_experts = num_experts // runtime.world_size
                    begin = runtime.rank * local_experts
                    counts = tensors["expert_totals"][begin : begin + local_experts].clone()
                    prefix = counts.cumsum(0, dtype=torch.int32)
                    count_list = counts.cpu().tolist()
                    rows = tensors["rank_rows"].cpu().tolist()
                    # Keep the allocation alive through this contiguous view.
                    # Padding is unwritten; only actual expanded rows are valid.
                    tensors["weight_return_meta"] = tensors["weight_return_meta"][: rows[runtime.rank]]
                    # num_recv_tokens (deduplicated layout) is unavailable; expanded count is exact.
                    result = EPHandle(
                        True,
                        num_experts,
                        1,
                        capacity,
                        32,
                        indices,
                        None,
                        rows[runtime.rank],
                        count_list,
                        None,
                        prefix,
                        counts,
                        None,
                        None,
                        None,
                        None,
                    )
                    for name in _PERSISTENT:
                        setattr(result, name, tensors[name])
                    if "_peer_plan" in tensors:
                        result._peer_plan = tensors["_peer_plan"]
                    result.address_stride = max(1, max(rows))
                    result.rank, result.world_size = runtime.rank, runtime.world_size
                    result.num_tokens, result.num_topk = tokens, topk
                    result.chunk_tokens = 256
                    result._buffer_token = self._handle_token
                    result.trace = None
                    retained = (
                        indices,
                        counts,
                        prefix,
                        *(tensors[name] for name in _PERSISTENT),
                        *((tensors["_peer_plan"],) if "_peer_plan" in tensors else ()),
                    )
                    event = EventOverlap(EventHandle(), retained)
                if not async_with_compute:
                    event.current_stream_wait()
                runtime.notify_inflight = None
                return None, None, None, result, event
            except Exception:
                runtime.notify_failed = True
                # Keep all allocations until successful stream drain during destroy.
                raise
