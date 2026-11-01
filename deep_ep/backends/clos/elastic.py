"""DeepEP V2 elastic intranode API for Ascend NPUs."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Tuple, Union
from deep_ep.utils.event import EventHandle as PublicEventHandle

from ._native import get_native_class
from .buffer import (
    _BOOTSTRAP_LOCK,
    _compact_extra_tensors,
    _create_communication_stream,
    _get_allocation_stream_id,
    _normalize_transport,
    _prepare_shmem_bootstrap,
    _record_recv_outputs_on_current_stream,
    _resolve_group_or_comm,
    _try_record_event,
    _use_comm_stream,
)
from .config import Config
from .config import (
    _MAX_LOCAL_EXPERTS_FOR_SIZE_HINT,
    _MAX_NUM_AIV,
    _MAX_SCALES_FOR_SIZE_HINT,
)
from .dtypes import topk_idx_bytes, topk_idx_t
from .event import EventHandle, EventOverlap


_ELASTIC_BUFFER_ALIGNMENT = 2 * 1024 * 1024
_ROW_ALIGNMENT = 32
_NOTIFY_ALIGNMENT = 512
_COUNT_ALIGNMENT_ELEMS = 8
_DEFAULT_HCCS_GBS = 200.0
_INT_BYTES = 4
_FLOAT_BYTES = 4
_BF16_BYTES = 2
_INT64_MAX = (1 << 63) - 1
_MAX_SHMEM_INTRANODE_RANKS = 1024
_MAX_TOPK = 128


class EPHandle:
    """Routing metadata returned by DeepEP-compatible dispatch APIs."""

    def __init__(
        self,
        do_expand: bool,
        num_experts: int,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        num_sms: int,
        topk_idx: Any,
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_recv_tokens_per_expert_list: List[int],
        psum_num_recv_tokens_per_scaleup_rank: Any,
        psum_num_recv_tokens_per_expert: Any,
        num_unaligned_recv_tokens_per_expert: Any,
        recv_src_metadata: Any,
        dst_buffer_slot_idx: Any,
        token_metadata_at_forward: Optional[Any],
        channel_linked_list: Optional[Any],
    ) -> None:
        if topk_idx is None:
            raise ValueError("topk_idx must not be None")
        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
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

        # Kept for source compatibility. The current Ascend path does not
        # install DeepEP's optional deterministic-sort hook.
        self.cached_recv_src_metadata_before_sort = None
        self._kind = "elastic"
        self._buffer_token: Optional[object] = None

    @property
    def num_compute_units(self):
        """Platform-neutral alias for the retained public AIV-count field."""
        return self.num_sms


class ElasticBuffer:
    """DeepEP V2 intranode buffer with an API independent from V1 ``Buffer``.

    Dispatch and combine share the optimized V1 backend. V2 keeps EPHandle,
    optional expert expansion and independently allocated GM outputs. UDMA
    uses the same shared-Jetty scheduler as V1; MTE retains its own transport.
    ``use_1g_huge_pages`` optionally backs the symmetric workspace with 1 GiB
    HBM pages through CANN SHMEM's user-buffer initialization path.
    """

    def __init__(
        self,
        group: Any,
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
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
        use_1g_huge_pages: bool = False,
        *,
        shmem_ip_port: Optional[str] = None,
        transport: Optional[str] = None,
        udma_schedule: str = "peer",
        udma_peers_per_worker: int = 4,
        udma_wqes_per_visit: int = 8,
    ) -> None:
        rank, num_ranks, resolved_group = _resolve_group_or_comm(group, None)
        transport = _normalize_transport(transport or os.environ.get("ASCEND_DEEPEP_TRANSPORT", "auto"))
        if udma_schedule not in ("peer", "token"):
            raise ValueError("udma_schedule must be 'peer' or 'token'")
        if udma_schedule == "token" and transport != "udma":
            raise ValueError("udma_schedule='token' requires transport='udma'")
        self._udma_config = Config(
            udma_peers_per_worker=udma_peers_per_worker,
            udma_wqes_per_visit=udma_wqes_per_visit,
        )
        self._udma_schedule = udma_schedule
        _validate_elastic_constructor(
            rank=rank,
            num_ranks=num_ranks,
            num_bytes=num_bytes,
            num_cpu_bytes=num_cpu_bytes,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            deterministic=deterministic,
            allow_hybrid_mode=allow_hybrid_mode,
            allow_multiple_reduction=allow_multiple_reduction,
            prefer_overlap_with_compute=prefer_overlap_with_compute,
            sl_idx=sl_idx,
            num_allocated_qps=num_allocated_qps,
            num_cpu_timeout_secs=num_cpu_timeout_secs,
            num_gpu_timeout_secs=num_gpu_timeout_secs,
            explicitly_destroy=explicitly_destroy,
            use_1g_huge_pages=use_1g_huge_pages,
        )
        if shmem_ip_port is not None and (not isinstance(shmem_ip_port, str) or not shmem_ip_port):
            raise ValueError("shmem_ip_port must be a non-empty string when specified")
        if num_bytes is None:
            if num_max_tokens_per_rank <= 0 or hidden <= 0:
                raise ValueError("num_max_tokens_per_rank and hidden must be positive when num_bytes is not provided")
            num_bytes = self.get_buffer_size_hint(
                resolved_group,
                num_max_tokens_per_rank,
                hidden,
                num_topk,
                use_fp8_dispatch,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )

        # SHMEM capacity and routing capacity are symmetric properties. Check
        # them before constructing the native SHMEM runtime so mismatches fail
        # collectively instead of hanging during initialization.
        _validate_group_consensus(
            resolved_group,
            "num_max_tokens_per_rank",
            num_max_tokens_per_rank,
        )
        _validate_group_consensus(resolved_group, "num_bytes", num_bytes)
        _validate_group_consensus(resolved_group, "use_1g_huge_pages", use_1g_huge_pages)

        self.group = resolved_group
        self.rank_idx = rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.group_size = num_ranks
        self.num_scaleout_ranks = 1
        self.num_scaleup_ranks = num_ranks
        self.scaleout_rank_idx = 0
        self.scaleup_rank_idx = rank
        self.num_rdma_ranks = 1
        self.num_nvlink_ranks = num_ranks
        self.num_bytes = num_bytes
        self.num_cpu_bytes = num_cpu_bytes
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch
        # Keep DeepEP's public configuration value for API compatibility. The
        # current Ascend kernel path does not branch on this option.
        self.deterministic = deterministic
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.sl_idx = sl_idx
        self.num_allocated_qps = num_allocated_qps
        self.num_cpu_timeout_secs = num_cpu_timeout_secs
        self.num_gpu_timeout_secs = num_gpu_timeout_secs
        self.explicitly_destroy = explicitly_destroy
        self.use_1g_huge_pages = use_1g_huge_pages
        self._destroyed = False
        self._handle_token = object()

        self.transport = transport
        # Internal baseline switch; not part of the public ElasticBuffer API.
        self._use_single_source_combine = True
        _validate_group_consensus(resolved_group, "udma_peers_per_worker", udma_peers_per_worker)
        _validate_group_consensus(resolved_group, "udma_wqes_per_visit", udma_wqes_per_visit)
        native_type = get_native_class("BufferRuntime")
        with _BOOTSTRAP_LOCK:
            unique_id, bootstrap_key = _prepare_shmem_bootstrap(
                resolved_group,
                None,
                rank,
                num_ranks,
                num_bytes,
                transport,
                use_1g_huge_pages,
                shmem_ip_port,
                num_max_tokens_per_rank or None,
                udma_schedule,
            )
            self._native = native_type(
                rank,
                num_ranks,
                num_bytes,
                transport,
                False,
                shmem_ip_port or "",
                use_1g_huge_pages,
                unique_id,
                bootstrap_key,
                True,
                udma_schedule == "token",
            )
            mapping_check = getattr(self, "_validate_native_mappings", None)
            if mapping_check is not None:
                mapping_check(resolved_group)
        self.num_device_aiv = int(self._native.num_device_aiv())
        if self.num_device_aiv < 2:
            raise RuntimeError("V2 elastic communication requires at least 2 AIVs")
        self.runtime = self._native
        self._communication_stream = _create_communication_stream()

        # Match the V2 initialization visibility contract on the communication
        # stream. The native barrier kernel is reused from the V1 resource layer.
        self.barrier(use_comm_stream=True, with_cpu_sync=True)

    @staticmethod
    def get_buffer_size_hint(
        group: Any,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
    ) -> int:
        """Estimate 2-MiB-aligned intranode dispatch/combine SHMEM capacity.

        An unspecified top-k reserves 128 entries. The estimate reserves up
        to 1024 local experts per rank and 128 FP8 scale columns per token.
        It reserves metadata for all supported AIVs, including explicit
        64-AIV launches on Ascend950. The cached dispatch plan is separate GM.
        """

        _, num_ranks, _ = _resolve_group_or_comm(group, None)
        if num_ranks < 2 or num_ranks > _MAX_SHMEM_INTRANODE_RANKS:
            raise ValueError(f"group size must be between 2 and {_MAX_SHMEM_INTRANODE_RANKS}")
        _validate_positive_int("num_max_tokens_per_rank", num_max_tokens_per_rank)
        _validate_positive_int("hidden", hidden)
        _validate_nonnegative_int("num_topk", num_topk)
        _validate_bool("use_fp8_dispatch", use_fp8_dispatch)
        _validate_bool("allow_hybrid_mode", allow_hybrid_mode)
        _validate_bool("allow_multiple_reduction", allow_multiple_reduction)
        if not allow_multiple_reduction:
            raise NotImplementedError("combine without multiple reduction is not implemented")

        if num_topk > _MAX_TOPK:
            raise ValueError(f"num_topk must not exceed {_MAX_TOPK}")
        effective_topk = _MAX_TOPK if num_topk == 0 else num_topk
        hidden_row_bytes = _checked_align(
            _checked_mul(hidden, 1 if use_fp8_dispatch else _BF16_BYTES, "token row"),
            _ROW_ALIGNMENT,
            "token row",
        )
        topk_idx_row_bytes = _checked_align(
            _checked_mul(effective_topk, topk_idx_bytes, "top-k index row"),
            _ROW_ALIGNMENT,
            "top-k index row",
        )
        topk_weight_row_bytes = _checked_align(
            _checked_mul(effective_topk, _FLOAT_BYTES, "top-k weight row"),
            _ROW_ALIGNMENT,
            "top-k weight row",
        )
        scale_row_bytes = (
            _checked_align(
                _checked_mul(_MAX_SCALES_FOR_SIZE_HINT, _FLOAT_BYTES, "FP8 scale row"),
                _ROW_ALIGNMENT,
                "FP8 scale row",
            )
            if use_fp8_dispatch
            else 0
        )

        num_channels = _MAX_NUM_AIV
        rank_stride = _align_up(num_ranks, _COUNT_ALIGNMENT_ELEMS)
        channel_stride = _align_up(num_channels, _COUNT_ALIGNMENT_ELEMS)
        expert_stride = _align_up(
            _checked_mul(num_ranks, _MAX_LOCAL_EXPERTS_FOR_SIZE_HINT, "expert count"),
            _COUNT_ALIGNMENT_ELEMS,
        )
        notify_elems = _checked_add(
            _checked_mul(num_ranks, num_ranks, "rank-prefix matrix"),
            _checked_mul(num_ranks, channel_stride, "channel-prefix matrix"),
            rank_stride,
            expert_stride,
            _checked_mul(num_channels, rank_stride, "notify channel counters"),
            name="notify metadata",
        )
        notify_bytes = _checked_align(
            _checked_mul(notify_elems, _INT_BYTES, "notify metadata"),
            _NOTIFY_ALIGNMENT,
            "notify metadata",
        )

        recv_slots = _checked_mul(num_max_tokens_per_rank, num_ranks, "dispatch receive slots")
        dispatch_row_bytes = _checked_add(
            hidden_row_bytes,
            _INT_BYTES,
            topk_idx_row_bytes,
            topk_weight_row_bytes,
            scale_row_bytes,
            name="dispatch slot row",
        )
        dispatch_body = _checked_add(
            _checked_mul(
                num_ranks,
                _checked_mul(num_channels, _INT_BYTES, "recv channel row"),
                "recv channel prefixes",
            ),
            _checked_mul(recv_slots, dispatch_row_bytes, "dispatch slots"),
            _checked_mul(
                num_max_tokens_per_rank,
                scale_row_bytes,
                "dispatch input scale staging",
            ),
            name="dispatch workspace",
        )

        # Both optimized UDMA kernels use one incoming marker per source.
        dispatch_signal_slots = num_ranks
        dispatch_body = _checked_add(
            _checked_align(dispatch_body, 512, "dispatch UDMA signal base"),
            _checked_mul(dispatch_signal_slots, 512, "dispatch UDMA signal bytes"),
            name="dispatch with UDMA",
        )

        # Combine reserves one partial for each source rank and original token.
        combine_slots = _checked_mul(num_max_tokens_per_rank, num_ranks, "combine partial slots")
        combine_partial_row = _checked_add(
            _checked_align(
                _checked_mul(hidden, _BF16_BYTES, "combine BF16 row"),
                _ROW_ALIGNMENT,
                "combine BF16 row",
            ),
            topk_weight_row_bytes,
            name="combine partial row",
        )
        combine_body = _checked_mul(combine_slots, combine_partial_row, "combine partials")
        combine_body = _checked_add(
            _checked_align(combine_body, 512, "combine UDMA signal base"),
            _checked_mul(
                num_ranks,
                512,
                "combine UDMA signal bytes",
            ),
            name="combine with UDMA",
        )

        total = _checked_add(
            notify_bytes,
            max(dispatch_body, combine_body),
            name="elastic buffer",
        )
        return _checked_align(total, _ELASTIC_BUFFER_ALIGNMENT, "elastic buffer")

    def destroy(self) -> None:
        """Synchronize and release an explicit-lifetime V2 runtime."""

        if not self.explicitly_destroy:
            raise RuntimeError("destroy() requires explicitly_destroy=True at construction")
        self._destroy_runtime()

    def _destroy_runtime(self) -> None:
        """Run DeepEP's collective teardown path once."""

        with _BOOTSTRAP_LOCK:
            if self._destroyed:
                return
            self.barrier(use_comm_stream=True, with_cpu_sync=True)
            self._native.destroy()
            self._destroyed = True
            self.runtime = None
            self._native = None
            self._communication_stream = None

    def barrier(
        self,
        use_comm_stream: bool = True,
        with_cpu_sync: bool = False,
        sequential: bool = True,
    ) -> None:
        """Order compute around one all-rank device barrier."""

        self._ensure_alive()
        _validate_bool("use_comm_stream", use_comm_stream)
        _validate_bool("with_cpu_sync", with_cpu_sync)
        _validate_bool("sequential", sequential)
        del sequential  # There is only one intra-node scale-up domain.

        compute_stream = _current_npu_stream()
        stream = self._communication_stream if use_comm_stream else compute_stream
        barrier_complete = None
        context = _use_comm_stream(stream) if use_comm_stream else nullcontext()
        with context:
            if use_comm_stream:
                # DeepEP: stream_wait(comm_stream, compute_stream).
                compute_ready = EventHandle(compute_stream)
                compute_ready.current_stream_wait()
            if with_cpu_sync:
                _synchronize_npu_device()
            # One representative AIV per rank invokes a barrier whose scope is
            # all SHMEM PEs/ranks; this is not a one-core-only synchronization.
            self._native.elastic_barrier(1)
            if use_comm_stream:
                barrier_complete = EventHandle(stream)
        if with_cpu_sync:
            _synchronize_npu_device()
        if barrier_complete is not None:
            # DeepEP: stream_wait(compute_stream, comm_stream).
            with _use_comm_stream(compute_stream):
                barrier_complete.current_stream_wait()

    @staticmethod
    def capture() -> EventHandle:
        """Record and return an event on the current compute stream."""

        return EventHandle()

    def get_comm_stream(self) -> Any:
        self._ensure_alive()
        return self._communication_stream

    def get_physical_domain_size(self) -> Tuple[int, int]:
        self._ensure_alive()
        return 1, self.num_ranks

    def get_logical_domain_size(self) -> Tuple[int, int]:
        self._ensure_alive()
        return 1, self.num_ranks

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        """Estimate an even AIV task-block count for intranode communication."""

        _validate_positive_int("num_experts", num_experts)
        _validate_positive_int("num_topk", num_topk)
        _validate_nonnegative_int("num_scaleout_topk", num_scaleout_topk)
        if num_scaleout_topk != 0:
            raise NotImplementedError("scale-out routing is not implemented")
        if num_experts % self.num_ranks:
            raise ValueError("num_experts must be divisible by world_size")
        if num_topk > num_experts:
            raise ValueError("num_topk must not exceed num_experts")
        _validate_nonnegative_real("rdma_gbs", rdma_gbs)
        _validate_nonnegative_real("nvlink_gbs", nvlink_gbs)
        _validate_positive_real("sm_read_gbs", sm_read_gbs)
        _validate_positive_real("sm_write_gbs", sm_write_gbs)

        max_even_aiv = min(_MAX_NUM_AIV, self.num_device_aiv)
        max_even_aiv -= max_even_aiv % 2
        if not self.prefer_overlap_with_compute:
            return max_even_aiv

        # Expected distinct destination ranks under a balanced top-k gate.
        experts_per_rank = num_experts // self.num_ranks
        probability_missed = 1.0
        for index in range(num_topk):
            remaining_nonlocal = num_experts - experts_per_rank - index
            if remaining_nonlocal <= 0:
                probability_missed = 0.0
                break
            probability_missed *= remaining_nonlocal / (num_experts - index)
        expected_ranks = self.num_ranks * (1.0 - probability_missed)

        target_gbs = nvlink_gbs or _DEFAULT_HCCS_GBS
        traffic_fraction = max(1.0 / self.num_ranks, expected_ranks / self.num_ranks)
        read_blocks = math.ceil(target_gbs * traffic_fraction / sm_read_gbs)
        write_blocks = math.ceil(target_gbs * traffic_fraction / sm_write_gbs)
        recommended = max(4, math.ceil(expected_ranks), read_blocks, write_blocks)
        recommended = _align_up(math.ceil(recommended * 1.25), 2)
        return min(recommended, max_even_aiv)

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        _validate_nonnegative_int("num_sms", num_sms)
        return 0

    def dispatch(
        self,
        x: Union[Any, Tuple[Any, Any]],
        topk_idx: Optional[Any] = None,
        topk_weights: Optional[Any] = None,
        cumulative_local_expert_recv_stats: Optional[Any] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: bool = False,
        do_zero_padding: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
    ) -> Tuple[Any, Optional[Any], Optional[Any], EPHandle, EventOverlap]:
        """Dispatch through the intranode kernel, optionally expanding rows."""

        self._ensure_alive()
        _validate_dispatch_scope(
            x,
            do_expand,
            do_zero_padding,
            use_tma_aligned_col_major_sf,
            async_with_compute_stream,
            allocate_on_comm_stream,
            previous_event,
            previous_event_before_epilogue,
        )
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_nonnegative_int("num_qps", num_qps)
        if num_qps != 0:
            raise NotImplementedError("intranode V2 does not use RDMA QPs")
        if cumulative_local_expert_recv_stats is not None:
            raise NotImplementedError("intranode dispatch does not accept cumulative expert stats")
        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError("FP8 input must be (x, x_scales)")
            x_tensor, x_scales = x
        else:
            x_tensor, x_scales = x, None
        if (x_scales is not None) != self.use_fp8_dispatch:
            raise ValueError("FP8 input must match use_fp8_dispatch")

        cached_mode = handle is not None
        if cached_mode:
            self._validate_handle(handle)
            if do_expand != handle.do_expand:
                raise ValueError("cached dispatch do_expand must match EPHandle")
            if topk_idx is not None:
                raise ValueError("cached dispatch requires topk_idx=None")
            if do_cpu_sync not in (None, False):
                raise ValueError("cached dispatch does not support CPU sync")
            resolved_topk_idx = handle.topk_idx
            resolved_num_experts = _value_or(num_experts, handle.num_experts)
            resolved_capacity = _value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
            resolved_alignment = _value_or(expert_alignment, handle.expert_alignment)
            if (
                resolved_num_experts,
                resolved_capacity,
                resolved_alignment,
            ) != (
                handle.num_experts,
                handle.num_max_tokens_per_rank,
                handle.expert_alignment,
            ):
                raise ValueError("cached dispatch arguments must match EPHandle")
            resolved_num_sms = handle.num_sms if num_sms == 0 else num_sms
            if resolved_num_sms != handle.num_sms:
                raise ValueError("cached dispatch num_sms must match EPHandle")
            resolved_do_cpu_sync = False
        else:
            if topk_idx is None:
                raise ValueError("topk_idx is required for non-cached dispatch")
            if num_experts is None:
                raise ValueError("num_experts is required for non-cached dispatch")
            resolved_topk_idx = topk_idx
            resolved_num_experts = num_experts
            if self.num_max_tokens_per_rank:
                if num_max_tokens_per_rank is not None:
                    _validate_positive_int("num_max_tokens_per_rank", num_max_tokens_per_rank)
                    if num_max_tokens_per_rank != self.num_max_tokens_per_rank:
                        raise ValueError("dispatch num_max_tokens_per_rank must match ElasticBuffer")
                resolved_capacity = self.num_max_tokens_per_rank
            else:
                resolved_capacity = num_max_tokens_per_rank
                _validate_group_consensus(self.group, "num_max_tokens_per_rank", resolved_capacity)
                _validate_positive_int("num_max_tokens_per_rank", resolved_capacity)
            resolved_alignment = _value_or(expert_alignment, 1)
            resolved_do_cpu_sync = _value_or(do_cpu_sync, True)
            num_topk = int(resolved_topk_idx.size(1))
            resolved_num_sms = self.get_theoretical_num_sms(resolved_num_experts, num_topk) if num_sms == 0 else num_sms

        _validate_dispatch_tensors(
            x_tensor,
            x_scales,
            resolved_topk_idx,
            topk_weights,
            cumulative_local_expert_recv_stats,
            resolved_num_experts,
            self.num_ranks,
            resolved_capacity,
            resolved_alignment,
            self.hidden,
            self.num_topk,
            do_expand,
        )
        if do_expand and not cached_mode and not resolved_do_cpu_sync:
            raise NotImplementedError("first expanded dispatch requires do_cpu_sync=True for expert counts")
        required_num_bytes = self.get_buffer_size_hint(
            self.group,
            resolved_capacity,
            int(x_tensor.size(1)),
            int(resolved_topk_idx.size(1)),
            self.use_fp8_dispatch,
            self.allow_hybrid_mode,
            True,
        )
        if self.num_bytes < required_num_bytes:
            raise RuntimeError(
                "elastic dispatch requires at least "
                f"{required_num_bytes} bytes of symmetric workspace, got "
                f"{self.num_bytes}"
            )
        _validate_num_sms(resolved_num_sms, self.num_device_aiv)
        _validate_bool("do_handle_copy", do_handle_copy)
        _validate_bool("do_cpu_sync", resolved_do_cpu_sync)
        if not cached_mode and self.num_max_tokens_per_rank == 0:
            self.num_max_tokens_per_rank = resolved_capacity

        single_source_slots = getattr(handle, "_single_source_slots", None) if cached_mode else None
        make_single_source = (
            not cached_mode and do_expand and self.transport == "udma" and self._use_single_source_combine
        )
        if make_single_source:
            import torch

            allocation_stream = self._communication_stream if allocate_on_comm_stream else _current_npu_stream()
        dependency = previous_event or _try_record_event(None)
        with _use_stream_after_event(self._communication_stream, dependency):
            if cached_mode:
                dispatch_layout = handle._dispatch_layout
                rank_prefix_matrix, channel_prefix_matrix, _, _, is_token_in_rank, _ = dispatch_layout
                num_tokens_per_rank = None
                num_tokens_per_expert = None
                kernel_topk_idx = None
                if topk_weights is not None:
                    # V1 needs topk_idx to route weights and derives the expert
                    # count from this tensor's length. Cached notify ignores its
                    # contents, since the prefix matrices are already available.
                    import torch

                    kernel_topk_idx = resolved_topk_idx
                    num_tokens_per_expert = torch.empty(
                        (resolved_num_experts,), dtype=torch.int32, device=x_tensor.device
                    )
            else:
                kernel_topk_idx = resolved_topk_idx
                (
                    num_tokens_per_rank,
                    _,
                    num_tokens_per_expert,
                    is_token_in_rank,
                ) = self._native.get_dispatch_layout(resolved_topk_idx, resolved_num_experts, allocation_stream_id)
                rank_prefix_matrix = None
                channel_prefix_matrix = None
            config = Config(
                num_aiv=resolved_num_sms,
                udma_peers_per_worker=self._udma_config.udma_peers_per_worker,
                udma_wqes_per_visit=self._udma_config.udma_wqes_per_visit,
            )
            dispatch_args = (
                x_tensor,
                x_scales,
                kernel_topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                handle.num_recv_tokens if cached_mode else 0,
                rank_prefix_matrix,
                channel_prefix_matrix,
                resolved_alignment,
                0 if cached_mode or resolved_do_cpu_sync else resolved_capacity * self.num_ranks,
                config.num_aiv,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                False,
                allocation_stream_id,
                resolved_capacity,
                config.udma_peers_per_worker,
                config.udma_wqes_per_visit,
                handle._dispatch_plan if cached_mode else None,
            )
            if cached_mode:
                result = self._native.intranode_dispatch_with_plan(*dispatch_args)
                public_handle = None
            else:
                # Expand writes the final expert prefix before this API returns.
                result, public_handle = self._native.intranode_dispatch_with_handle(*dispatch_args, do_expand)
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                dispatch_plan,
            ) = result
            if not cached_mode:
                dispatch_layout = (
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    recv_channel_prefix_matrix,
                    recv_src_idx,
                    is_token_in_rank,
                    send_head,
                )
            num_recv_tokens = int(recv_x.size(0))
            if previous_event_before_epilogue is not None:
                previous_event_before_epilogue.current_stream_wait()
            if do_expand:
                if cached_mode:
                    offsets, actual_counts, mapping, aligned_counts = handle._expand_layout
                    expand_topk_idx = None
                else:
                    if not num_recv_tokens_per_expert_list:
                        raise RuntimeError("expanded dispatch requires per-expert counts")
                    aligned_counts = list(num_recv_tokens_per_expert_list)
                    offsets = actual_counts = mapping = None
                    expand_topk_idx = recv_topk_idx
                    if make_single_source:
                        with _use_comm_stream(allocation_stream):
                            single_source_slots = torch.empty(
                                (num_recv_tokens,), dtype=torch.int32, device=recv_x.device
                            )
                (
                    recv_x,
                    recv_x_scales,
                    recv_topk_weights,
                    offsets,
                    actual_counts,
                    mapping,
                ) = self._native.expand(
                    recv_x,
                    recv_x_scales,
                    expand_topk_idx,
                    recv_topk_weights,
                    [] if cached_mode else aligned_counts,
                    resolved_alignment,
                    handle.num_expanded_tokens if cached_mode else 0,
                    offsets,
                    actual_counts,
                    mapping,
                    resolved_num_sms,
                    allocation_stream_id,
                    None if cached_mode else public_handle[1],
                    None if cached_mode else public_handle[3],
                    None if cached_mode else single_source_slots,
                )
                recv_topk_idx = None
                num_expanded_tokens = int(recv_x.size(0))
                expand_layout = (offsets, actual_counts, mapping, aligned_counts)
            else:
                num_expanded_tokens = 0
                expand_layout = None
            if cached_mode:
                # The routing indices came from the handle, not this call.
                recv_topk_idx = None
            if not cached_mode:
                handle_topk_idx = resolved_topk_idx.clone() if do_handle_copy else resolved_topk_idx

        if not cached_mode:
            handle = EPHandle(
                do_expand,
                resolved_num_experts,
                resolved_alignment,
                resolved_capacity,
                resolved_num_sms,
                handle_topk_idx,
                num_recv_tokens,
                num_expanded_tokens,
                list(num_recv_tokens_per_expert_list),
                public_handle[0],
                public_handle[1],
                public_handle[2],
                public_handle[3],
                public_handle[4],
                None,
                None,
            )
            _record_recv_outputs_on_current_stream(*public_handle)
            handle._buffer_token = self._handle_token
            handle._dispatch_layout = dispatch_layout
            handle._dispatch_plan = dispatch_plan
            handle._expand_layout = expand_layout
            handle._single_source_slots = single_source_slots
            _record_recv_outputs_on_current_stream(single_source_slots)
            _record_recv_outputs_on_current_stream(handle.topk_idx, *dispatch_layout)
            if expand_layout is not None:
                _record_recv_outputs_on_current_stream(*expand_layout[:3])

        _record_recv_outputs_on_current_stream(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, dispatch_plan)

        if x_scales is not None:
            recv_x = (recv_x, recv_x_scales)

        event = self._finish_comm_event(
            async_with_compute_stream,
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                dispatch_plan,
                x_tensor,
                x_scales,
                topk_weights,
                handle,
            ),
        )
        return recv_x, recv_topk_idx, recv_topk_weights, handle, event

    @staticmethod
    def _unpack_bias(bias: Union[Any, Tuple[Any, Any], None]) -> Tuple[Optional[Any], Optional[Any]]:
        if bias is None:
            return None, None
        if isinstance(bias, tuple):
            if len(bias) != 2:
                raise ValueError("bias tuple must contain exactly two tensors")
            return bias[0], bias[1]
        return bias, None

    def combine(
        self,
        x: Any,
        handle: EPHandle,
        topk_weights: Optional[Any] = None,
        bias: Union[Any, Tuple[Any, Any], None] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[Any, Optional[Any], EventOverlap]:
        """Combine intranode rows, aggregating expanded rows first.

        Communication and final reduction use separate kernels, as in the legacy
        ElasticBuffer API. previous_event_before_epilogue is waited on the
        communication stream after communication and before final reduction.
        """

        self._ensure_alive()
        self._validate_handle(handle)
        _validate_combine_tensors(x, handle, topk_weights, bias)
        _validate_bool("async_with_compute_stream", async_with_compute_stream)
        _validate_bool("allocate_on_comm_stream", allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_event_dependency(previous_event, allocate_on_comm_stream)
        _validate_epilogue_event(previous_event_before_epilogue)
        _validate_nonnegative_int("num_qps", num_qps)
        if num_qps != 0:
            raise NotImplementedError("intranode V2 does not use RDMA QPs")
        resolved_num_sms = handle.num_sms if num_sms == 0 else num_sms
        _validate_num_sms(resolved_num_sms, self.num_device_aiv)
        if resolved_num_sms != handle.num_sms:
            raise ValueError("combine num_sms must match dispatch EPHandle")
        bias_0, bias_1 = self._unpack_bias(bias)
        config = Config(
            num_aiv=resolved_num_sms,
            udma_peers_per_worker=self._udma_config.udma_peers_per_worker,
            udma_wqes_per_visit=self._udma_config.udma_wqes_per_visit,
        )
        (
            rank_prefix_matrix,
            _,
            recv_channel_prefix_matrix,
            recv_src_idx,
            _,
            send_head,
        ) = handle._dispatch_layout

        dependency = previous_event or _try_record_event(None)
        with _use_stream_after_event(self._communication_stream, dependency):
            combine_x, combine_topk_weights = x, topk_weights
            single_source_slots = (
                getattr(handle, "_single_source_slots", None)
                if handle.do_expand and self.transport == "udma" and self._use_single_source_combine
                else None
            )
            if handle.do_expand:
                _, _, recv_to_expanded_idx, _ = handle._expand_layout
                # Preserve combine's separate weight output. Passing these as
                # aggregate's route weights would multiply them into x.
                combine_x, combine_topk_weights = self._native.aggregate(
                    x,
                    None,
                    topk_weights,
                    recv_to_expanded_idx,
                    resolved_num_sms,
                    allocation_stream_id,
                    single_source_slots,
                    rank_prefix_matrix if single_source_slots is not None else None,
                )
            epilogue = self._native._intranode_combine_begin(
                combine_x,
                combine_topk_weights,
                bias_0,
                bias_1,
                recv_src_idx,
                rank_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                config.num_aiv,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                allocation_stream_id,
                handle.num_max_tokens_per_rank,
                config.udma_peers_per_worker,
                config.udma_wqes_per_visit,
                x if single_source_slots is not None else None,
                single_source_slots,
            )
            if previous_event_before_epilogue is not None:
                previous_event_before_epilogue.current_stream_wait()
            combined_x, combined_topk_weights = epilogue.run()

        _record_recv_outputs_on_current_stream(combined_x, combined_topk_weights)
        event = self._finish_comm_event(
            async_with_compute_stream,
            (
                combined_x,
                combined_topk_weights,
                combine_x,
                combine_topk_weights,
                x,
                topk_weights,
                bias,
                handle,
                epilogue,
            ),
        )
        return combined_x, combined_topk_weights, event

    def _finish_comm_event(self, async_with_compute_stream: bool, extra_tensors: Tuple[Any, ...]) -> EventOverlap:
        event = _try_record_event(self._communication_stream)
        if event is None:
            return EventOverlap()
        if async_with_compute_stream:
            return EventOverlap(event, _compact_extra_tensors(extra_tensors))
        event.current_stream_wait()
        return EventOverlap()

    def _validate_handle(self, handle: EPHandle) -> None:
        if not isinstance(handle, EPHandle):
            raise TypeError("handle must be an EPHandle")
        if getattr(handle, "_kind", "elastic") != "elastic":
            raise ValueError("EPHandle belongs to a different dispatch mode")
        if handle._buffer_token is not self._handle_token:
            raise ValueError("EPHandle belongs to a different ElasticBuffer")

    def _ensure_alive(self) -> None:
        if self._destroyed or self._native is None:
            raise RuntimeError("ElasticBuffer has been destroyed")

    def __del__(self) -> None:
        try:
            if getattr(self, "_destroyed", True):
                return
            if not getattr(self, "explicitly_destroy", True):
                try:
                    self._destroy_runtime()
                    return
                except Exception:
                    # Finalizers cannot propagate. Fall back to native local
                    # teardown if collective teardown is no longer possible.
                    pass
            self.runtime = None
            self._native = None
            self._communication_stream = None
            self._destroyed = True
        except Exception:
            pass


@contextmanager
def _use_stream_after_event(stream: Any, event: Optional[EventHandle]):
    with _use_comm_stream(stream):
        if event is not None:
            event.current_stream_wait()
        yield


def _validate_group_consensus(group: Any, name: str, value: Any) -> None:
    """Validate a value shared by all ranks in the group."""

    try:
        import torch.distributed as dist
    except ImportError:
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    _, num_ranks, resolved_group = _resolve_group_or_comm(group, None)
    values = [None] * num_ranks
    dist.all_gather_object(values, value, group=resolved_group)
    if any(type(peer_value) is not type(value) or peer_value != value for peer_value in values):
        raise ValueError(f"all ranks must use the same {name}; got {values}")


def _current_npu_stream() -> Any:
    try:
        import torch_npu  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError("torch_npu is required for ElasticBuffer.barrier") from exc
    return torch.npu.current_stream()


def _synchronize_npu_device() -> None:
    """Wait for all work submitted to the current NPU device."""

    try:
        import torch_npu  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError("torch_npu is required for ElasticBuffer.barrier") from exc
    torch.npu.synchronize()


def _validate_elastic_constructor(**values: Any) -> None:
    rank = values["rank"]
    num_ranks = values["num_ranks"]
    if num_ranks < 2 or num_ranks > _MAX_SHMEM_INTRANODE_RANKS:
        raise ValueError(f"world_size must be between 2 and {_MAX_SHMEM_INTRANODE_RANKS}")
    if rank < 0 or rank >= num_ranks:
        raise ValueError("rank must be in [0, world_size)")
    for name in (
        "num_cpu_bytes",
        "num_max_tokens_per_rank",
        "hidden",
        "num_topk",
        "sl_idx",
        "num_allocated_qps",
    ):
        _validate_nonnegative_int(name, values[name])
    for name in ("num_cpu_timeout_secs", "num_gpu_timeout_secs"):
        _validate_positive_int(name, values[name])
    for name in (
        "use_fp8_dispatch",
        "deterministic",
        "allow_hybrid_mode",
        "allow_multiple_reduction",
        "prefer_overlap_with_compute",
        "explicitly_destroy",
        "use_1g_huge_pages",
    ):
        _validate_bool(name, values[name])
    if values["num_cpu_bytes"] != 0:
        raise NotImplementedError("V2 CPU and hybrid buffers are not implemented")
    if values["num_topk"] > _MAX_TOPK:
        raise ValueError(f"num_topk must not exceed {_MAX_TOPK}")
    if not values["allow_multiple_reduction"]:
        raise NotImplementedError("this V2 implementation requires allow_multiple_reduction=True")
    num_bytes = values["num_bytes"]
    if num_bytes is not None:
        _validate_positive_int("num_bytes", num_bytes)
        if num_bytes % _ELASTIC_BUFFER_ALIGNMENT:
            raise ValueError("num_bytes must be 2 MiB aligned")


def _validate_dispatch_scope(
    x: Any,
    do_expand: bool,
    do_zero_padding: bool,
    use_tma_aligned_col_major_sf: bool,
    async_with_compute_stream: bool,
    allocate_on_comm_stream: bool,
    previous_event: Optional[EventHandle],
    previous_event_before_epilogue: Optional[EventHandle],
) -> None:
    for name, value in (
        ("do_expand", do_expand),
        ("do_zero_padding", do_zero_padding),
        ("use_tma_aligned_col_major_sf", use_tma_aligned_col_major_sf),
        ("async_with_compute_stream", async_with_compute_stream),
        ("allocate_on_comm_stream", allocate_on_comm_stream),
    ):
        _validate_bool(name, value)
    if do_zero_padding:
        raise NotImplementedError("V2 zero padding is not implemented")
    if use_tma_aligned_col_major_sf:
        raise NotImplementedError("CUDA TMA scale-factor layout is not supported")
    _validate_event_dependency(previous_event, allocate_on_comm_stream)
    _validate_epilogue_event(previous_event_before_epilogue)


def _validate_epilogue_event(event: Optional[EventHandle]) -> None:
    if event is not None and not isinstance(event, (EventHandle, PublicEventHandle)):
        raise TypeError("previous_event_before_epilogue must be an EventHandle or None")


def _validate_event_dependency(event: Optional[EventHandle], allocate_on_comm_stream: bool) -> None:
    if event is not None and not isinstance(event, (EventHandle, PublicEventHandle)):
        raise TypeError("previous_event must be an EventHandle or None")
    if event is not None and not allocate_on_comm_stream:
        raise ValueError("allocate_on_comm_stream must be True when a previous event is provided")


def _validate_dispatch_tensors(
    x: Any,
    x_scales: Optional[Any],
    topk_idx: Any,
    topk_weights: Optional[Any],
    cumulative_stats: Optional[Any],
    num_experts: int,
    num_ranks: int,
    capacity: int,
    expert_alignment: int,
    configured_hidden: int,
    configured_topk: int,
    do_expand: bool,
) -> None:
    import torch

    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise TypeError("x must be a 2D torch.Tensor")
    if x_scales is None:
        if x.dtype != torch.bfloat16:
            raise TypeError("BF16 x must have dtype torch.bfloat16")
    else:
        if x.element_size() != 1:
            raise TypeError("FP8 x must use a one-byte dtype")
        if not isinstance(x_scales, torch.Tensor) or x_scales.dim() != 2:
            raise TypeError("x_scales must be a 2D tensor")
        if x_scales.dtype not in (torch.float32, torch.int32):
            raise TypeError("x_scales must have dtype float32 or int32")
        if x_scales.size(0) != x.size(0):
            raise ValueError("x_scales and x must have the same num_tokens")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not isinstance(topk_idx, torch.Tensor) or topk_idx.dim() != 2:
        raise TypeError("topk_idx must be a 2D torch.Tensor")
    if topk_idx.dtype != topk_idx_t:
        raise TypeError("topk_idx must have dtype deep_ep.topk_idx_t")
    if not topk_idx.is_contiguous():
        raise ValueError("topk_idx must be contiguous")
    if topk_idx.size(0) != x.size(0):
        raise ValueError("x and topk_idx must have the same num_tokens")
    if topk_idx.size(1) <= 0:
        raise ValueError("topk_idx must contain at least one selection column")
    if topk_idx.size(1) > _MAX_TOPK:
        raise ValueError(f"topk_idx width must not exceed {_MAX_TOPK}")
    _validate_positive_int("num_experts", num_experts)
    _validate_positive_int("num_max_tokens_per_rank", capacity)
    _validate_positive_int("expert_alignment", expert_alignment)
    if num_experts % num_ranks:
        raise ValueError("num_experts must be divisible by world_size")
    if not do_expand and expert_alignment != 1:
        raise NotImplementedError("non-expanded V2 currently supports expert_alignment=1 only")
    if x.size(0) > capacity:
        raise ValueError("x.size(0) exceeds num_max_tokens_per_rank")
    if configured_hidden and x.size(1) != configured_hidden:
        raise ValueError("x hidden size does not match ElasticBuffer.hidden")
    if configured_topk and topk_idx.size(1) != configured_topk:
        raise ValueError("topk_idx width does not match ElasticBuffer.num_topk")
    if topk_weights is not None:
        if not isinstance(topk_weights, torch.Tensor):
            raise TypeError("topk_weights must be a torch.Tensor or None")
        if topk_weights.dtype != torch.float32:
            raise TypeError("topk_weights must have dtype torch.float32")
        if tuple(topk_weights.shape) != tuple(topk_idx.shape):
            raise ValueError("topk_weights must have the same shape as topk_idx")
        if not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous")
    if cumulative_stats is not None:
        expected_local_experts = num_experts // num_ranks
        if not isinstance(cumulative_stats, torch.Tensor):
            raise TypeError("cumulative_local_expert_recv_stats must be a tensor or None")
        if cumulative_stats.dtype != torch.int32:
            raise TypeError("cumulative_local_expert_recv_stats must have dtype torch.int32")
        if tuple(cumulative_stats.shape) != (expected_local_experts,):
            raise ValueError("cumulative_local_expert_recv_stats has an invalid shape")


def _validate_combine_tensors(
    x: Any,
    handle: EPHandle,
    topk_weights: Optional[Any],
    bias: Union[Any, Tuple[Any, Any], None],
) -> None:
    import torch

    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise TypeError("x must be a 2D torch.Tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must have dtype torch.bfloat16")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    expected_tokens = handle.num_expanded_tokens if handle.do_expand else handle.num_recv_tokens
    if x.size(0) != expected_tokens:
        raise ValueError("x.size(0) must match the handle's token count")
    num_topk = int(handle.topk_idx.size(1))
    if topk_weights is not None:
        if not isinstance(topk_weights, torch.Tensor):
            raise TypeError("topk_weights must be a torch.Tensor or None")
        if topk_weights.dtype != torch.float32:
            raise TypeError("topk_weights must have dtype torch.float32")
        expected_shape = (expected_tokens,) if handle.do_expand else (expected_tokens, num_topk)
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError("topk_weights has an invalid shape")
        if not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous")
    bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)
    expected_bias_shape = (int(handle.topk_idx.size(0)), int(x.size(1)))
    for name, value in (("bias_0", bias_0), ("bias_1", bias_1)):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if tuple(value.shape) != expected_bias_shape:
            raise ValueError(f"{name} has an invalid shape")


def _validate_num_sms(num_sms: int, num_device_aiv: int = _MAX_NUM_AIV) -> None:
    _validate_positive_int("num_sms", num_sms)
    if num_sms % 2:
        raise ValueError("num_sms must be even")
    max_num_aiv = min(_MAX_NUM_AIV, num_device_aiv)
    if num_sms > max_num_aiv:
        raise ValueError(f"num_sms must not exceed {max_num_aiv}")


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_nonnegative_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > _INT64_MAX:
        raise OverflowError(f"{name} exceeds int64")


def _validate_positive_int(name: str, value: Any) -> None:
    _validate_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative_real(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_positive_real(name: str, value: Any) -> None:
    _validate_nonnegative_real(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _checked_mul(lhs: int, rhs: int, name: str) -> int:
    if lhs < 0 or rhs < 0:
        raise ValueError(f"{name} factors must be nonnegative")
    if lhs and rhs > _INT64_MAX // lhs:
        raise OverflowError(f"{name} exceeds int64")
    return lhs * rhs


def _checked_add(*values: int, name: str) -> int:
    total = 0
    for value in values:
        if value < 0 or total > _INT64_MAX - value:
            raise OverflowError(f"{name} exceeds int64")
        total += value
    return total


def _checked_align(value: int, alignment: int, name: str) -> int:
    if value > _INT64_MAX - (alignment - 1):
        raise OverflowError(f"aligned {name} exceeds int64")
    return _align_up(value, alignment)


__all__ = ["ElasticBuffer", "EPHandle"]
