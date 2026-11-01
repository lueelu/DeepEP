"""Ascend expert-parallel communication Buffer."""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from threading import RLock
from typing import Any, List, Optional, Tuple, Union

from deep_ep._backend_lock import LOCK as _BOOTSTRAP_LOCK

from ._native import get_native_class, load_native_module
from .config import Config, _MAX_NUM_AIV
from .event import EventHandle, EventOverlap
from .expand import ExpandHandle
from .resources import Handle, Slot, SlotState


_ORPHANED_NATIVE_RESOURCES: list[tuple[Any, ...]] = []
_MAX_SHMEM_INTRANODE_RANKS = 1024
_SUPPORTED_TRANSPORTS = frozenset(("auto", "mte", "roce", "sdma", "udma"))


class _V1Handle(tuple):
    """Six-tensor compatibility tuple retaining its originating Buffer."""

    def __new__(cls, tensors, owner, dispatch_plan=None):
        handle = super().__new__(cls, tensors)
        handle._buffer_owner = owner
        handle._dispatch_plan = dispatch_plan
        return handle


def _is_blackwell() -> bool:
    return False


class Buffer:
    """Own one CANN SHMEM communication session and symmetric workspace.

    ``num_bytes`` is the usable communication-workspace capacity. The native
    runtime maps it into one CANN SHMEM heap together with its metadata.
    ``transport`` selects the CANN SHMEM data-operation engine; ``auto`` uses
    MTE. ``udma`` enables both MTE and UDMA resources and selects the URMA V1
    dispatch kernel; other transports keep the UBMEM V1 dispatch kernel.
    ``udma_schedule="peer"`` retains the peer-owned UDMA scheduler. Experimental
    ``"token"`` assigns input channels/return rows to half the AIVs (at most 32),
    each with its own QP to all peers. All ranks must select the same schedule
    and launch configuration; changing schedule requires a new Buffer.
    ``use_1g_huge_pages`` registers the workspace as one caller-owned CANN VMM
    buffer backed by 1 GiB HBM pages. It is disabled by default.
    ``shmem_ip_port=None`` selects UID bootstrap within the EP group. An
    IP/port endpoint must be passed explicitly; this class does not read
    ``ASCEND_DEEPEP_SHMEM_IP_PORT`` from the environment.
    ``low_latency_mode`` is retained by the native runtime for subsequent
    kernel/resource selection. The public layout, dispatch and combine
    signatures are identical for both dispatch implementations.
    """

    # ``num_sms`` follows the DeepEP public name for compatibility. On Ascend
    # it is the number of AIV task blocks used by a communication kernel. Zero
    # asks the backend to select the platform default explicitly.
    num_sms: int = 32

    def __init__(
        self,
        group: Any,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        comm: Optional[Any] = None,
        *,
        num_bytes: Optional[int] = None,
        transport: str = "auto",
        udma_schedule: str = "peer",
        use_1g_huge_pages: bool = False,
        shmem_ip_port: Optional[str] = None,
        num_max_tokens_per_rank: Optional[int] = None,
    ) -> None:
        if _is_capturing():
            raise RuntimeError("construct Buffer outside NPU graph capture")
        if num_max_tokens_per_rank is not None:
            if isinstance(num_max_tokens_per_rank, bool) or not isinstance(num_max_tokens_per_rank, int):
                raise TypeError("num_max_tokens_per_rank must be an integer")
            if num_max_tokens_per_rank <= 0:
                raise ValueError("num_max_tokens_per_rank must be positive")
        rank, resolved_num_ranks, resolved_group = _resolve_group_or_comm(group, comm)
        resolved_num_bytes, resolved_nvl_bytes = _resolve_workspace_bytes(num_nvl_bytes, num_bytes)
        resolved_transport = _normalize_transport(transport)
        if udma_schedule not in ("peer", "token"):
            raise ValueError("udma_schedule must be 'peer' or 'token'")
        if udma_schedule == "token" and resolved_transport != "udma":
            raise ValueError("udma_schedule='token' requires transport='udma'")
        _validate_constructor(
            rank,
            resolved_num_ranks,
            resolved_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            num_qps_per_rank,
            allow_nvlink_for_low_latency_mode,
            allow_mnnvl,
            explicitly_destroy,
            enable_shrink,
            use_1g_huge_pages,
        )
        resolved_ip_port = shmem_ip_port
        if resolved_ip_port is not None and (not isinstance(resolved_ip_port, str) or not resolved_ip_port):
            raise ValueError("shmem_ip_port must be a non-empty string when specified")

        self.group = resolved_group
        self.rank = rank
        self.group_size = resolved_num_ranks
        self.num_ranks = resolved_num_ranks
        self.num_bytes = resolved_num_bytes
        self.num_nvl_bytes = resolved_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.transport = resolved_transport
        self._udma_schedule = udma_schedule
        self.low_latency_mode = low_latency_mode
        self.use_1g_huge_pages = use_1g_huge_pages
        self.num_qps_per_rank = num_qps_per_rank
        self.allow_nvlink_for_low_latency_mode = allow_nvlink_for_low_latency_mode
        self.allow_mnnvl = allow_mnnvl
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self._num_max_tokens_per_rank = num_max_tokens_per_rank

        self._lock = RLock()
        self._destroyed = False
        self._failed_event_tensors: Optional[Tuple[Any, ...]] = None
        native_type = get_native_class("BufferRuntime")
        with _BOOTSTRAP_LOCK:
            unique_id, bootstrap_key = _prepare_shmem_bootstrap(
                resolved_group,
                comm,
                rank,
                resolved_num_ranks,
                resolved_num_bytes,
                resolved_transport,
                use_1g_huge_pages,
                resolved_ip_port,
                num_max_tokens_per_rank,
                udma_schedule,
            )
            self._native = native_type(
                rank,
                resolved_num_ranks,
                resolved_num_bytes,
                resolved_transport,
                low_latency_mode,
                resolved_ip_port or "",
                use_1g_huge_pages,
                unique_id,
                bootstrap_key,
                True,  # V1 requests shared Jetty when the resolved transport is UDMA.
                udma_schedule == "token",
            )
            mapping_check = getattr(self, "_validate_native_mappings", None)
            if mapping_check is not None:
                mapping_check(comm if comm is not None else resolved_group)
        # Slot/generation tracks one future layout -> dispatch -> combine
        # session. The native runtime owns the symmetric workspace itself.
        self._slot = Slot(0, owner=self, native_resources=self._native)
        self._communication_stream = _create_communication_stream()

    @property
    def udma_schedule(self) -> str:
        """Immutable schedule; choose before SHMEM creates its peer/QP bindings."""
        return self._udma_schedule

    @property
    def slot(self) -> Slot:
        return self._slot

    @property
    def num_max_tokens_per_rank(self) -> Optional[int]:
        """Immutable common capacity; permits different local T, including zero.

        All EP ranks must agree on this value at construction. Size the SHMEM
        buffer using this capacity. None retains the equal-T V1 convention.
        Graph replay retains shapes/addresses: change values or mask routes
        with -1, rather than resizing tensors. Keep Buffer alive until every
        graph using it has been reset and all replay work has finished.
        Replay overwrites the graph's fixed outputs and handles; copy anything
        needed by a later backward pass before the next replay. Graphs sharing
        this Buffer must serialize communication through its common stream.
        """
        return self._num_max_tokens_per_rank

    @property
    def comm_meta_ptr(self) -> int:
        self._ensure_alive()
        return int(self._native.get_comm_meta_ptr())

    @property
    def native(self) -> Any:
        """Return the native resource owner for later runtime integration."""

        self._ensure_alive()
        return self._native

    @property
    def runtime(self) -> Any:
        """DeepEP-compatible alias for the native runtime object."""

        return self.native

    @property
    def communication_stream(self) -> Any:
        """Return the NPU stream reserved for communication operations."""

        return self.get_comm_stream()

    def is_available(self) -> bool:
        return not self._destroyed and bool(self._native.is_initialized())

    def acquire_slot(self) -> Slot:
        with self._lock:
            self._ensure_alive()
            self._slot.acquire()
            return self._slot

    def create_handle(self, slot: Slot, put_offset: Any) -> Handle:
        """Bind a dispatch ``put_offset`` handle to the active generation."""

        self._validate_slot(slot)
        if slot.state is not SlotState.DISPATCH_READY:
            raise RuntimeError("a Handle can only be created for a DISPATCH_READY Slot")
        if put_offset is None:
            raise ValueError("put_offset must not be None")
        return Handle(self, slot, put_offset)

    def validate_handle(self, handle: Handle) -> None:
        if not isinstance(handle, Handle):
            raise TypeError("handle must be a Handle")
        self._ensure_alive()
        handle.validate(self)

    def barrier(self) -> None:
        self._ensure_alive()
        self._native.barrier()

    def destroy(self) -> None:
        """Free the communication workspace when no operation is active."""

        if _is_capturing():
            raise RuntimeError("destroy Buffer outside NPU graph capture")
        with _BOOTSTRAP_LOCK, self._lock:
            if self._destroyed:
                return
            if self._slot.state is not SlotState.FREE:
                raise RuntimeError(
                    f"cannot destroy Buffer with active Slot: {self._slot.state.value}@{self._slot.generation}"
                )
            # Device work enqueued on the communication stream must finish
            # before its SHMEM queues and symmetric allocations are released.
            self._communication_stream.synchronize()
            self._native.destroy()
            self._destroyed = True
            self._failed_event_tensors = None
            self._communication_stream = None

    @staticmethod
    def is_sm90_compiled() -> bool:
        """Return whether CUDA SM90-specific kernels are available."""

        return False

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """Set communication AIV task-block count using DeepEP's API name.

        ``new_num_sms`` does not change or reserve physical NPU cores. The value is
        copied to :attr:`Config.num_aiv`. Each AIV is one logical channel in
        the Ascend intranode path. Even values retain DeepEP's configuration
        contract, and zero resets selection to the platform default.
        """

        if not isinstance(new_num_sms, int) or isinstance(new_num_sms, bool):
            raise TypeError("new_num_sms must be an integer")
        if new_num_sms < 0:
            raise ValueError("new_num_sms must be greater than or equal to zero")
        if new_num_sms % 2:
            raise ValueError("new_num_sms must be even")
        if new_num_sms > _MAX_NUM_AIV:
            raise ValueError(f"new_num_sms must not exceed {_MAX_NUM_AIV}")
        Buffer.num_sms = new_num_sms

    def get_comm_stream(self) -> Any:
        """Return the internal NPU communication stream."""

        self._ensure_alive()
        return self._communication_stream

    def _finish_comm_event(self, async_finish: bool, extra_tensors: Tuple[Any, ...]) -> EventOverlap:
        try:
            event = _try_record_event(self._communication_stream)
            if async_finish:
                return EventOverlap(event, _compact_extra_tensors(extra_tensors))
            event.current_stream_wait()
        except Exception:
            # Native work has already been submitted. Do not reuse this Buffer
            # or let its outputs/SHMEM be reclaimed without a successful drain.
            # Explicit destroy() waits on the communication stream before
            # releasing these references. No synchronization is added here,
            # including when event recording fails during graph capture.
            self._failed_event_tensors = _compact_extra_tensors(extra_tensors)
            raise
        return EventOverlap()

    def get_dispatch_layout(
        self,
        topk_idx: Any,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[Any, Optional[Any], Any, Any, EventOverlap]:
        """Calculate DeepEP V1 dispatch metadata through the native layout kernel."""

        self._ensure_alive()
        _validate_operation_flags(async_finish, allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_previous_event(previous_event)
        with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
            ) = self._native.get_dispatch_layout(topk_idx, num_experts, allocation_stream_id)
        _record_recv_outputs_on_current_stream(
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        )
        event = self._finish_comm_event(
            async_finish,
            (
                topk_idx,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
            ),
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        )

    def dispatch(
        self,
        x: Union[Any, Tuple[Any, Any]],
        handle: Optional[Tuple[Any, ...]] = None,
        num_tokens_per_rank: Optional[Any] = None,
        num_tokens_per_rdma_rank: Optional[Any] = None,
        is_token_in_rank: Optional[Any] = None,
        num_tokens_per_expert: Optional[Any] = None,
        topk_idx: Optional[Any] = None,
        topk_weights: Optional[Any] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        zero_copy: bool = False,
    ) -> Tuple[
        Union[Any, Tuple[Any, Any]],
        Optional[Any],
        Optional[Any],
        Optional[List[int]],
        Optional[Tuple[Any, ...]],
        EventOverlap,
    ]:
        """Dispatch tokens with the DeepEP V1 Python signature.

        Payload outputs own independent GM storage by default, including in
        cached mode. With zero_copy=True they instead alias this Buffer's
        symmetric workspace: consume or materialize them before the next
        dispatch/combine, and keep this Buffer alive while using the views.
        Non-cached fixed output requires num_worst_tokens == group_size * C,
        where C is num_max_tokens_per_rank (or local T with the equal-T default).
        A smaller bound is rejected before any communication is launched.
        """

        self._ensure_alive()
        _validate_operation_flags(async_finish, allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        if not isinstance(zero_copy, bool):
            raise TypeError("zero_copy must be a bool")
        _validate_previous_event(previous_event)
        x_tensor, x_scales = _split_x_and_scales(x)
        if self.num_max_tokens_per_rank is not None and x_tensor.size(0) > self.num_max_tokens_per_rank:
            raise ValueError("local token count exceeds num_max_tokens_per_rank")
        if _is_capturing() and handle is None and num_worst_tokens <= 0:
            raise RuntimeError("NPU graph dispatch requires num_worst_tokens > 0 (no CPU count readback)")
        if handle is None and num_worst_tokens > 0:
            capacity = self.num_max_tokens_per_rank if self.num_max_tokens_per_rank is not None else x_tensor.size(0)
            if num_worst_tokens != self.group_size * capacity:
                raise ValueError("fixed dispatch output requires num_worst_tokens == group_size * capacity")
        config = self.get_dispatch_config(self.group_size) if config is None else config
        num_aiv, nvl_send, nvl_recv = _config_to_native_args(config)
        del num_tokens_per_rdma_rank

        if handle is not None:
            if isinstance(handle, _V1Handle) and handle._buffer_owner is not self:
                raise ValueError("V1 handle belongs to a different Buffer")
            if topk_idx is not None or topk_weights is not None:
                raise ValueError("cached dispatch requires topk_idx and topk_weights to be None")
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                _recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank_from_handle,
                _send_head,
            ) = _unpack_v1_handle(handle)
            num_recv_tokens = int(recv_src_idx.size(0))
            with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
                (
                    recv_x,
                    recv_x_scales,
                    _recv_topk_idx,
                    _recv_topk_weights,
                    _num_recv_tokens_per_expert_list,
                    _rank_prefix_matrix,
                    _channel_prefix_matrix,
                    _recv_channel_prefix_matrix,
                    _recv_src_idx,
                    _send_head,
                    dispatch_plan,
                ) = self._native.intranode_dispatch_with_plan(
                    x_tensor,
                    x_scales,
                    None,
                    None,
                    None,
                    is_token_in_rank_from_handle,
                    None,
                    num_recv_tokens,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    expert_alignment,
                    num_worst_tokens,
                    num_aiv,
                    nvl_send,
                    nvl_recv,
                    zero_copy,
                    allocation_stream_id,
                    self.num_max_tokens_per_rank,
                    config.udma_peers_per_worker,
                    config.udma_wqes_per_visit,
                    getattr(handle, "_dispatch_plan", None),
                )
            if not zero_copy:
                _record_recv_outputs_on_current_stream(recv_x, recv_x_scales)
            plan_refs = () if dispatch_plan is None else (dispatch_plan,)
            if plan_refs:
                _record_recv_outputs_on_current_stream(*plan_refs)
            event = self._finish_comm_event(async_finish, (recv_x, recv_x_scales, *plan_refs, *handle))
            return (
                _pack_x_and_scales(recv_x, recv_x_scales, x_scales is not None),
                None,
                None,
                None,
                None,
                event,
            )

        if num_tokens_per_rank is None:
            raise ValueError("num_tokens_per_rank is required for non-cached dispatch")
        if is_token_in_rank is None:
            raise ValueError("is_token_in_rank is required for non-cached dispatch")
        if num_tokens_per_expert is None:
            raise ValueError("num_tokens_per_expert is required for non-cached dispatch")
        with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
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
            ) = self._native.intranode_dispatch_with_plan(
                x_tensor,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                None,
                None,
                expert_alignment,
                num_worst_tokens,
                num_aiv,
                nvl_send,
                nvl_recv,
                zero_copy,
                allocation_stream_id,
                self.num_max_tokens_per_rank,
                config.udma_peers_per_worker,
                config.udma_wqes_per_visit,
            )
        if not zero_copy:
            _record_recv_outputs_on_current_stream(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights)
        v1_handle = _V1Handle(
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ),
            self,
            dispatch_plan,
        )
        plan_refs = () if dispatch_plan is None else (dispatch_plan,)
        _record_recv_outputs_on_current_stream(*v1_handle, *plan_refs)
        event = self._finish_comm_event(
            async_finish,
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                *v1_handle,
                *plan_refs,
            ),
        )
        return (
            _pack_x_and_scales(recv_x, recv_x_scales, x_scales is not None),
            recv_topk_idx,
            recv_topk_weights,
            list(num_recv_tokens_per_expert_list),
            v1_handle,
            event,
        )

    def combine(
        self,
        x: Any,
        handle: Tuple[Any, ...],
        topk_weights: Optional[Any] = None,
        bias: Union[Any, Tuple[Any, Any], None] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[Any, Optional[Any], EventOverlap]:
        """Combine tokens using the V1 fused communication/reduction kernel."""

        self._ensure_alive()
        _validate_operation_flags(async_finish, allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_previous_event(previous_event)
        config = self.get_combine_config(self.group_size) if config is None else config
        num_aiv, nvl_send, nvl_recv = _config_to_native_args(config)
        if isinstance(handle, _V1Handle) and handle._buffer_owner is not self:
            raise ValueError("V1 handle belongs to a different Buffer")
        (
            rank_prefix_matrix,
            _send_channel_prefix_matrix,
            channel_prefix_matrix,
            src_idx,
            _is_recv_token_in_rank,
            send_head,
        ) = _unpack_v1_handle(handle)
        bias_0, bias_1 = self._unpack_bias(bias)

        with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
            recv_x, recv_topk_weights = self._native.intranode_combine(
                x,
                topk_weights,
                bias_0,
                bias_1,
                src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                send_head,
                num_aiv,
                nvl_send,
                nvl_recv,
                allocation_stream_id,
                self.num_max_tokens_per_rank,
                config.udma_peers_per_worker,
                config.udma_wqes_per_visit,
            )
        _record_recv_outputs_on_current_stream(recv_x, recv_topk_weights)
        event = self._finish_comm_event(async_finish, (recv_x, recv_topk_weights))
        return recv_x, recv_topk_weights, event

    def expand(
        self,
        x: Union[Any, Tuple[Any, Any]],
        recv_topk_idx: Optional[Any] = None,
        topk_weights: Optional[Any] = None,
        num_tokens_per_expert: Optional[List[int]] = None,
        expert_alignment: Optional[int] = None,
        handle: Optional[ExpandHandle] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Any, Tuple[Any, Any]],
        Optional[Any],
        ExpandHandle,
        EventOverlap,
    ]:
        """Expand rank-deduplicated dispatch rows into expert-major rows."""

        self._ensure_alive()
        _validate_operation_flags(async_finish, allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_previous_event(previous_event)
        x_tensor, x_scales = _split_x_and_scales(x)
        config = Config(Buffer.num_sms) if config is None else config
        num_aiv = _config_to_native_args(config)[0]

        if handle is None:
            if recv_topk_idx is None:
                raise ValueError("recv_topk_idx is required for non-cached expand")
            if num_tokens_per_expert is None:
                raise ValueError("num_tokens_per_expert is required for non-cached expand")
            if expert_alignment is None:
                expert_alignment = 1
            aligned_counts = _validate_expand_counts(num_tokens_per_expert, expert_alignment)
            cached_num_expanded_tokens = 0
            cached_expert_offsets = None
            cached_actual_expert_counts = None
            cached_mapping = None
        else:
            if not isinstance(handle, ExpandHandle):
                raise TypeError("handle must be an ExpandHandle")
            if int(x_tensor.size(0)) != handle.num_recv_tokens:
                raise ValueError("cached expand x must match the handle num_recv_tokens")
            if recv_topk_idx is not None:
                raise ValueError("cached expand requires recv_topk_idx=None")
            if num_tokens_per_expert is not None:
                raise ValueError("cached expand requires num_tokens_per_expert=None")
            if expert_alignment is not None and (expert_alignment != handle.expert_alignment):
                raise ValueError("cached expand expert_alignment must match the handle")
            expert_alignment = handle.expert_alignment
            aligned_counts = []
            cached_num_expanded_tokens = handle.num_expanded_tokens
            cached_expert_offsets = handle.expert_offsets
            cached_actual_expert_counts = handle.actual_expert_counts
            cached_mapping = handle.recv_to_expanded_idx

        with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
            (
                expanded_x,
                expanded_x_scales,
                expanded_topk_weights,
                expert_offsets,
                actual_expert_counts,
                recv_to_expanded_idx,
            ) = self._native.expand(
                x_tensor,
                x_scales,
                recv_topk_idx,
                topk_weights,
                aligned_counts,
                expert_alignment,
                cached_num_expanded_tokens,
                cached_expert_offsets,
                cached_actual_expert_counts,
                cached_mapping,
                num_aiv,
                allocation_stream_id,
            )

        if handle is None:
            handle = ExpandHandle(
                expert_offsets,
                actual_expert_counts,
                recv_to_expanded_idx,
                aligned_counts,
                int(x_tensor.size(0)),
                sum(aligned_counts),
                int(recv_topk_idx.size(1)),
                expert_alignment,
            )
        _record_recv_outputs_on_current_stream(
            expanded_x,
            expanded_x_scales,
            expanded_topk_weights,
            handle.expert_offsets,
            handle.actual_expert_counts,
            handle.recv_to_expanded_idx,
        )
        event = self._finish_comm_event(
            async_finish,
            (
                expanded_x,
                expanded_x_scales,
                expanded_topk_weights,
                handle.expert_offsets,
                handle.actual_expert_counts,
                handle.recv_to_expanded_idx,
            ),
        )
        return (
            _pack_x_and_scales(
                expanded_x,
                expanded_x_scales,
                x_scales is not None,
            ),
            expanded_topk_weights,
            handle,
            event,
        )

    def aggregate(
        self,
        x: Any,
        handle: ExpandHandle,
        topk_weights: Optional[Any] = None,
        expanded_values: Optional[Any] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[Any, Optional[Any], EventOverlap]:
        """Reduce expert-major rows back to the V1 rank-deduplicated layout."""

        self._ensure_alive()
        _validate_operation_flags(async_finish, allocate_on_comm_stream)
        allocation_stream_id = _get_allocation_stream_id(allocate_on_comm_stream)
        _validate_previous_event(previous_event)
        if not isinstance(handle, ExpandHandle):
            raise TypeError("handle must be an ExpandHandle")
        if int(x.size(0)) != handle.num_expanded_tokens:
            raise ValueError("aggregate x must match the handle num_expanded_tokens")
        config = Config(Buffer.num_sms) if config is None else config
        num_aiv = _config_to_native_args(config)[0]
        with _use_comm_stream_after_dependency(self._communication_stream, previous_event):
            aggregated_x, aggregated_values = self._native.aggregate(
                x,
                topk_weights,
                expanded_values,
                handle.recv_to_expanded_idx,
                num_aiv,
                allocation_stream_id,
            )
        _record_recv_outputs_on_current_stream(aggregated_x, aggregated_values)
        event = self._finish_comm_event(
            async_finish,
            (
                aggregated_x,
                aggregated_values,
                handle.expert_offsets,
                handle.actual_expert_counts,
                handle.recv_to_expanded_idx,
            ),
        )
        return aggregated_x, aggregated_values, event

    @staticmethod
    def capture() -> EventOverlap:
        """Record an event on the current NPU stream."""

        return EventOverlap(EventHandle())

    @staticmethod
    def _is_blackwell() -> bool:
        """Compatibility helper for integrations that probe the GPU architecture."""

        return _is_blackwell()

    @staticmethod
    def _unpack_bias(bias: Union[Any, Tuple[Any, Any], None]) -> Tuple[Optional[Any], Optional[Any]]:
        if bias is None:
            return None, None
        if isinstance(bias, tuple):
            if len(bias) > 2:
                raise ValueError("bias tuple must have at most two tensors")
            return bias[0] if len(bias) >= 1 else None, bias[1] if len(bias) == 2 else None
        return bias, None

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        _validate_config_world_size(num_ranks)
        return Config(Buffer.num_sms)

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        _validate_config_world_size(num_ranks)
        return Config(Buffer.num_sms)

    def _validate_slot(self, slot: Slot) -> None:
        self._ensure_alive()
        if not isinstance(slot, Slot):
            raise TypeError("slot must be a Slot")
        if slot is not self._slot or slot.owner is not self:
            raise ValueError("Slot belongs to a different Buffer")

    def _ensure_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Buffer has been destroyed")
        if self._failed_event_tensors is not None:
            raise RuntimeError("Buffer has a failed completion event; call destroy() before replacing it")

    def __del__(self) -> None:
        try:
            if getattr(self, "_destroyed", True):
                return
            slot = getattr(self, "_slot", None)
            native = getattr(self, "_native", None)
            failed_tensors = getattr(self, "_failed_event_tensors", None)
            if failed_tensors is not None and native is not None:
                # Event failure cannot prove that submitted device work has
                # stopped. Never fall through to unsynchronized native cleanup.
                orphaned = globals().get("_ORPHANED_NATIVE_RESOURCES")
                if orphaned is not None:
                    orphaned.append((native, self._communication_stream, failed_tensors))
                warnings.warn(
                    "Buffer was finalized after a completion event failure; "
                    "resources will be retained until process teardown",
                    ResourceWarning,
                    stacklevel=2,
                )
                return
            if slot is not None and native is not None and slot.state is not SlotState.FREE:
                # There is no safe collective cleanup when ranks reach Python
                # finalization independently. Keep in-flight resources alive
                # until process teardown instead of invalidating device work.
                orphaned = globals().get("_ORPHANED_NATIVE_RESOURCES")
                if orphaned is not None:
                    orphaned.append((native, slot))
                warnings.warn(
                    "Buffer was finalized with an active Slot; native resources "
                    "will be retained until process teardown",
                    ResourceWarning,
                    stacklevel=2,
                )
        except Exception:
            # Python may already be partially finalized.
            pass


def _split_x_and_scales(x: Union[Any, Tuple[Any, Any]]) -> Tuple[Any, Optional[Any]]:
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError("FP8 dispatch input tuple must contain (x, x_scales)")
        return x[0], x[1]
    return x, None


def _pack_x_and_scales(x: Any, x_scales: Optional[Any], return_tuple: bool) -> Union[Any, Tuple[Any, Any]]:
    if return_tuple:
        return x, x_scales
    return x


def _record_recv_outputs_on_current_stream(*tensors: Any) -> None:
    # Register owned outputs/handles for caller use, independently of the
    # allocation stream. SHMEM payload views must never enter this path.
    import torch

    for tensor in tensors:
        if tensor is not None and tensor.device.type == "npu":
            tensor.record_stream(torch.npu.current_stream(tensor.device))


def _validate_operation_flags(async_finish: bool, allocate_on_comm_stream: bool) -> None:
    if not isinstance(async_finish, bool):
        raise TypeError("async_finish must be a bool")
    if not isinstance(allocate_on_comm_stream, bool):
        raise TypeError("allocate_on_comm_stream must be a bool")


def _is_capturing() -> bool:
    import torch

    query = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    return bool(query()) if query is not None else False


def _validate_previous_event(previous_event: Any) -> None:
    if previous_event is not None and not hasattr(previous_event, "event"):
        raise TypeError("previous_event must expose an event attribute or be None")


def _get_allocation_stream_id(allocate_on_comm_stream: bool) -> Optional[int]:
    if allocate_on_comm_stream:
        return None
    import torch

    if not hasattr(torch, "npu") or not hasattr(torch.npu, "current_stream"):
        return None  # CPU-only API mocks.
    return torch.npu.current_stream().stream_id


@contextmanager
def _use_comm_stream(stream: Any):
    import torch

    if not hasattr(torch, "npu") or not hasattr(torch.npu, "stream"):
        yield
        return
    # Never catch exceptions thrown by the operation inside this context.
    with torch.npu.stream(stream):
        yield


@contextmanager
def _use_comm_stream_after_dependency(stream: Any, previous_event: Any):
    dependency_event = getattr(previous_event, "event", None)
    if dependency_event is None:
        # Record before switching streams so the communication stream waits
        # for all compute work submitted before this operation.
        dependency_event = _try_record_event(None)

    with _use_comm_stream(stream):
        dependency_event.current_stream_wait()
        yield


def _try_record_event(stream: Any) -> EventHandle:
    # A missing/broken event API must not turn required stream dependencies
    # into successful no-ops. CPU tests provide an explicit event mock.
    return EventHandle(stream)


def _compact_extra_tensors(values: Tuple[Any, ...]) -> Tuple[Any, ...]:
    tensors = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            tensors.extend(item for item in value if item is not None)
        else:
            tensors.append(value)
    return tuple(tensors)


def _config_to_native_args(config: Any) -> Tuple[int, int, int]:
    num_aiv = getattr(config, "num_aiv", getattr(config, "num_sms", None))
    nvl_send = getattr(config, "num_max_nvl_chunked_send_tokens", None)
    nvl_recv = getattr(config, "num_max_nvl_chunked_recv_tokens", None)
    values = {
        "num_aiv": num_aiv,
        "num_max_nvl_chunked_send_tokens": nvl_send,
        "num_max_nvl_chunked_recv_tokens": nvl_recv,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"config.{name} must be an integer")
    return int(num_aiv), int(nvl_send), int(nvl_recv)


def _validate_expand_counts(counts: List[int], expert_alignment: int) -> List[int]:
    if not isinstance(expert_alignment, int) or isinstance(expert_alignment, bool):
        raise TypeError("expert_alignment must be an integer")
    if expert_alignment <= 0:
        raise ValueError("expert_alignment must be positive")
    if not isinstance(counts, (list, tuple)) or not counts:
        raise TypeError("num_tokens_per_expert must be a non-empty list")
    result = []
    for count in counts:
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("expert token counts must be integers")
        if count < 0:
            raise ValueError("expert token counts must be nonnegative")
        if count % expert_alignment:
            raise ValueError("expert token counts must be aligned to expert_alignment")
        result.append(count)
    return result


def _unpack_v1_handle(handle: Tuple[Any, ...]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if not isinstance(handle, (tuple, list)) or len(handle) != 6:
        raise TypeError("handle must be the 6-item tuple returned by dispatch")
    return tuple(handle)  # type: ignore[return-value]


def _validate_constructor(
    rank: int,
    num_ranks: int,
    num_nvl_bytes: int,
    num_rdma_bytes: int,
    low_latency_mode: bool,
    num_qps_per_rank: int,
    allow_nvlink_for_low_latency_mode: bool,
    allow_mnnvl: bool,
    explicitly_destroy: bool,
    enable_shrink: bool,
    use_1g_huge_pages: bool,
) -> None:
    integer_values = {
        "rank": rank,
        "num_ranks": num_ranks,
        "num_nvl_bytes": num_nvl_bytes,
        "num_rdma_bytes": num_rdma_bytes,
        "num_qps_per_rank": num_qps_per_rank,
    }
    for name, value in integer_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_ranks < 2 or num_ranks > _MAX_SHMEM_INTRANODE_RANKS or rank < 0 or rank >= num_ranks:
        raise ValueError(f"Buffer requires 0 <= rank < num_ranks and 2 <= num_ranks <= {_MAX_SHMEM_INTRANODE_RANKS}")
    if num_nvl_bytes < 0:
        raise ValueError("num_nvl_bytes must be greater than or equal to zero")
    if num_rdma_bytes < 0:
        raise ValueError("num_rdma_bytes must be greater than or equal to zero")
    if num_qps_per_rank <= 0:
        raise ValueError("num_qps_per_rank must be greater than zero")
    bool_values = {
        "low_latency_mode": low_latency_mode,
        "allow_nvlink_for_low_latency_mode": allow_nvlink_for_low_latency_mode,
        "allow_mnnvl": allow_mnnvl,
        "explicitly_destroy": explicitly_destroy,
        "enable_shrink": enable_shrink,
        "use_1g_huge_pages": use_1g_huge_pages,
    }
    for name, value in bool_values.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")


def _normalize_transport(transport: str) -> str:
    if not isinstance(transport, str):
        raise TypeError("transport must be a string")
    normalized = transport.lower()
    if normalized not in _SUPPORTED_TRANSPORTS:
        supported = ", ".join(sorted(_SUPPORTED_TRANSPORTS))
        raise ValueError(f"unsupported transport {transport!r}; expected one of: {supported}")
    return normalized


def _bootstrap_all_gather(group: Any, comm: Any, size: int, value: Any) -> list:
    if comm is not None:
        return comm.allgather(value)
    import torch.distributed as dist

    values = [None] * size
    dist.all_gather_object(values, value, group=group)
    return values


def _prepare_shmem_bootstrap(
    group,
    comm,
    rank,
    size,
    num_bytes,
    transport,
    use_1g_huge_pages,
    ip_port,
    num_max_tokens_per_rank=None,
    udma_schedule="peer",
):
    native = load_native_module()
    options = (
        num_bytes,
        transport,
        use_1g_huge_pages,
        ip_port,
        os.environ.get("SHMEM_UID_SESSION_ID") if ip_port is None else None,
        num_max_tokens_per_rank,
        udma_schedule,
    )
    # This key is compared only within this process. Buffer retains group, so
    # its identity cannot be recycled while the associated runtime is alive.
    key = repr((id(group), options))
    state, uid = native.get_shmem_bootstrap(key)
    peers = _bootstrap_all_gather(group, comm, size, (options, state, uid))
    if any(peer[0] != options for peer in peers):
        raise ValueError("all EP ranks must use the same workspace/transport/huge-page/bootstrap options")
    if any(peer[1] < 0 for peer in peers):
        raise RuntimeError("CANN SHMEM already has an active Buffer for a different group or configuration")
    if any(peer[1] != state or peer[2] != uid for peer in peers):
        raise RuntimeError("inconsistent active SHMEM sessions across EP ranks")
    if state or ip_port is not None:
        return uid, key

    # Only EP rank 0 reserves an endpoint. Other ranks receive its opaque UID,
    # never create their own endpoint or interpret a peer's socket descriptor.
    result = None
    if rank == 0:
        try:
            result = (native.get_shmem_unique_id(), None)
        except Exception as exc:
            result = (None, str(exc))
    result = _bootstrap_all_gather(group, comm, size, result)[0]
    if result[1] is not None:
        raise RuntimeError(f"EP rank 0 failed to create a SHMEM unique ID: {result[1]}")
    return result[0], key


def _resolve_workspace_bytes(num_nvl_bytes: int, num_bytes: Optional[int]) -> Tuple[int, int]:
    if not isinstance(num_nvl_bytes, int) or isinstance(num_nvl_bytes, bool):
        raise TypeError("num_nvl_bytes must be an integer")
    if num_nvl_bytes < 0:
        raise ValueError("num_nvl_bytes must be greater than or equal to zero")
    if num_bytes is not None:
        if not isinstance(num_bytes, int) or isinstance(num_bytes, bool):
            raise TypeError("num_bytes must be an integer")
        if num_bytes < 0:
            raise ValueError("num_bytes must be greater than or equal to zero")
        if num_nvl_bytes not in (0, num_bytes):
            raise ValueError("num_bytes and num_nvl_bytes specify different workspace sizes")
        return num_bytes, num_bytes
    return num_nvl_bytes, num_nvl_bytes


def _resolve_group_or_comm(group: Any, comm: Optional[Any]) -> Tuple[int, int, Any]:
    """Resolve rank information from a DeepEP-style process group or MPI comm."""

    if group is None and comm is None:
        raise ValueError("Either 'group' or 'comm' must be provided.")
    if group is not None and comm is not None:
        raise ValueError("Only one of 'group' or 'comm' should be provided.")
    if comm is not None:
        rank_method = getattr(comm, "Get_rank", None)
        size_method = getattr(comm, "Get_size", None)
        if callable(rank_method) and callable(size_method):
            return int(rank_method()), int(size_method()), comm
        raise TypeError("comm must provide Get_rank()/Get_size()")

    rank_method = getattr(group, "rank", None)
    size_method = getattr(group, "size", None)
    if callable(rank_method) and callable(size_method):
        return int(rank_method()), int(size_method()), group

    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise TypeError("group does not provide rank()/size()") from exc
    try:
        return int(dist.get_rank(group)), int(dist.get_world_size(group)), group
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TypeError("group must be a valid torch.distributed process group") from exc


def _create_communication_stream() -> Any:
    """Create a dedicated stream on the current NPU device."""

    try:
        import torch_npu  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError("torch_npu must be installed before creating a Buffer communication stream") from exc
    return torch.npu.Stream(device=torch.npu.current_device())


def _validate_config_world_size(num_ranks: int) -> None:
    if not isinstance(num_ranks, int) or isinstance(num_ranks, bool):
        raise TypeError("num_ranks must be an integer")
    if num_ranks < 2 or num_ranks > _MAX_SHMEM_INTRANODE_RANKS:
        raise ValueError(f"num_ranks must be between 2 and {_MAX_SHMEM_INTRANODE_RANKS}")
