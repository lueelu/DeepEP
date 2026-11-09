"""Experimental fixed-CV, PUT-only URMA MegaMoE API.

Local expert buckets are rebuilt before Pack; the incoming summary is published
separately while Pack/Dispatch progress. ``prepare_routes`` only binds Device
input and capacity-sized storage. Both forward paths use Device routing only.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import threading

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import Runtime, integer, warn_unclosed
from deep_ep._store import StoreGroup
from .urma_routing import DeviceRoutes
from .validation import collective_check

COMMUNICATION_PROTOCOL = "unitflag-peer-cv-epoch-so-v1"


def use_fused_gate_up(bs, world, topk=8, num_experts=384):
    """All routed GMM1 tasks use gate128/up128; shared=True weights stay planar."""
    return True


@lru_cache(maxsize=1)
def _extension():
    core = require_native("URMA MegaMoE")
    if not hasattr(core, "create_megamoe_urma_runtime"):
        raise RuntimeError("Install a wheel built with the MegaMoE URMA runtime")
    extension = importlib.import_module(__package__ + "._C")
    if getattr(extension, "urma_api_protocol", None) != 1:
        raise RuntimeError("URMA Python/native versions differ; rebuild and reinstall the complete wheel")
    if getattr(extension, "urma_tail_schedule", None) != "combine-ready-scan-reduce-world-v1":
        raise RuntimeError("URMA tailCover needs its matching native wheel; rebuild and reinstall the complete wheel")
    if getattr(extension, "urma_compute_schedule", None) != "unitflag-shared-fixub-both-v5":
        raise RuntimeError(
            "URMA M-group lookahead needs its matching native wheel; rebuild and reinstall the complete wheel"
        )
    if getattr(extension, "urma_dimension_contract", None) != "consumer-submit-e2048-cv-policy-v1":
        raise RuntimeError("Rebuild the generalized-dimension MegaMoE extension")
    if getattr(extension, "urma_shared_contract", None) != "unitflag-shared-a8w8-early-reduce-v1":
        raise RuntimeError("Rebuild the shared-expert MegaMoE extension")
    if getattr(extension, "urma_routed_weight_layout", None) != "gate128-up128-v1":
        raise RuntimeError("Rebuild the fused gate128/up128 routed GMM1 extension")
    if getattr(extension, "urma_routing_protocol", None) != 8:
        raise RuntimeError("Device routing requires matching Python/native wheel; rebuild and reinstall")
    return extension


def _trace_extension():
    """Check the optional trace ABI only when a caller requests tracing."""
    extension = _extension()
    expected = {
        "urma_trace_schema": 1,
        "urma_trace_num_engines": 84,
        "urma_trace_record_elements": 16,
        "urma_trace_header_elements": 16,
        "urma_trace_max_capacity": 65536,
    }
    for name, value in expected.items():
        if getattr(extension, name, None) != value:
            raise RuntimeError(
                f"URMA trace Python/native versions differ ({name}); rebuild and reinstall the complete wheel"
            )
    if not getattr(extension, "urma_timer_enabled", False):
        raise RuntimeError("URMA trace requires a wheel built with DEEPEP_MEGAMOE_TIMER=ON")
    return extension


class _UrmaRuntime(Runtime):
    _megamoe = True
    _megamoe_urma = True

    def __init__(self, group, capacity, timeout=120, *, allow_multi_host=False):
        if type(allow_multi_host) is not bool:
            raise TypeError("allow_multi_host must be bool")
        self._megamoe_urma_multi_host = allow_multi_host
        super().__init__(group, capacity, timeout)


@dataclass(frozen=True)
class PreparedUrmaRoutes:
    """Bound Device input plus reusable storage; routes are rebuilt on every launch."""

    owner: object
    native: object
    routing: DeviceRoutes
    bs: int
    topk_idx: object

    @property
    def received_rows(self):
        raise RuntimeError("received_rows is Device-generated; use native.route_stats() after forward for diagnostics")


class UrmaBuffer:
    """EP4/EP8/EP16 owner; multiple hosts require explicit opt-in on every rank."""

    def __init__(
        self,
        group,
        *,
        hidden=3584,
        intermediate_hidden=1280,
        num_experts=384,
        num_topk=8,
        num_shared_experts=0,
        shared_intermediate_hidden=None,
        num_max_tokens_per_rank=16384,
        cv_rows=256,
        gmm1_ahead_mgroups=12,
        max_recv_rows=None,
        timeout=120,
        allow_multi_host=False,
    ):
        import torch

        if not isinstance(group, StoreGroup):
            raise TypeError("URMA experimental API currently requires an explicit StoreGroup")
        error = None
        if max_recv_rows is None:
            max_recv_rows = group.size() * num_max_tokens_per_rank * num_topk
        try:
            if group.size() not in (4, 8, 16):
                raise ValueError("URMA MegaMoE requires EP4/EP8/EP16")
            if type(allow_multi_host) is not bool:
                raise TypeError("allow_multi_host must be bool")
            integer("hidden", hidden, 256, 8192)
            integer("intermediate_hidden", intermediate_hidden, 256, 16384)
            integer("num_experts", num_experts, group.size(), 128 * group.size())
            if num_experts % group.size():
                raise ValueError("num_experts must be divisible by EP")
            integer("num_topk", num_topk, 1, min(8, num_experts))
            integer("num_shared_experts", num_shared_experts, 0, 1)
            if shared_intermediate_hidden is None:
                shared_intermediate_hidden = intermediate_hidden
            integer("shared_intermediate_hidden", shared_intermediate_hidden, 256, 16384)
            if hidden % 256 or intermediate_hidden % 256 or shared_intermediate_hidden % 256:
                raise ValueError("URMA requires H/I aligned256, topk1..8/S0 or S1")
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 16384)
            integer("cv_rows", cv_rows, 128, 4096)
            integer("gmm1_ahead_mgroups", gmm1_ahead_mgroups, 0, 65536)
            if cv_rows % 128:
                raise ValueError("cv_rows must be a multiple of 128")
            if max_recv_rows != group.size() * num_max_tokens_per_rank * num_topk:
                raise ValueError("Device routing requires max_recv_rows=EP*max_tokens*topk (worst-case capacity)")
        except (TypeError, ValueError) as failure:
            error = str(failure)
        collective_check(
            group,
            "URMA configuration",
            error,
            (
                hidden,
                intermediate_hidden,
                num_experts,
                num_topk,
                num_shared_experts,
                shared_intermediate_hidden,
                num_max_tokens_per_rank,
                cv_rows,
                gmm1_ahead_mgroups,
                max_recv_rows,
                timeout,
                allow_multi_host,
                "combine-ready-scan-reduce-world-v1",
                "unitflag-shared-fixub-both-v5",
                COMMUNICATION_PROTOCOL,
            ),
        )
        self.group = group
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_shared_experts = num_shared_experts
        self.shared_intermediate_hidden = shared_intermediate_hidden
        self.num_max_tokens_per_rank, self.max_recv_rows = num_max_tokens_per_rank, max_recv_rows
        self.cv_rows = cv_rows
        self.gmm1_ahead_mgroups = gmm1_ahead_mgroups
        self.allow_multi_host = allow_multi_host
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._lock = threading.RLock()
        self._epoch = 0
        self._failure = None
        self._last_prepared = None
        self._plan = None
        capacity, error = None, None
        try:
            extension = _extension()
            capacity = int(
                extension.urma_buffer_size(
                    group.size(), num_max_tokens_per_rank, num_topk, hidden, max_recv_rows, cv_rows
                )
            )
            self._example = torch.empty(
                (num_max_tokens_per_rank, hidden), device=self._device, dtype=torch.float8_e4m3fn
            )
        except Exception as failure:
            error = str(failure)
        # No rank may enter collective SHMEM initialization after another rank
        # rejected the wheel/layout or failed its ordinary-device allocation.
        collective_check(group, "URMA pre-runtime allocation", error)
        self._runtime = _UrmaRuntime(group, capacity, timeout, allow_multi_host=allow_multi_host)

    @property
    def symmetric_buffer_bytes(self):
        return self._runtime.native.capacity

    @property
    def timer_enabled(self):
        return bool(getattr(_extension(), "urma_timer_enabled", False))

    def allocate_timer(self):
        import torch

        with self._operation("timer allocation"):
            if not self.timer_enabled:
                raise RuntimeError("URMA timer requires a wheel built with its timer support")
            return torch.zeros(int(_extension().urma_timer_numel), dtype=torch.int64, device=self._device)

    def allocate_trace(self, capacity=4096):
        """Allocate bounded per-engine event slices for one diagnostic launch."""
        import torch

        with self._operation("trace allocation"):
            extension = _trace_extension()
            integer("trace capacity", capacity, 1, extension.urma_trace_max_capacity)
            numel = extension.urma_trace_num_engines * (
                extension.urma_trace_header_elements + capacity * extension.urma_trace_record_elements
            )
            return torch.zeros(numel, dtype=torch.int64, device=self._device)

    @contextmanager
    def _operation(self, stage):
        """Fail the collective job, rather than retrying with divergent epochs.

        Store abort wakes peers waiting in host coordination. It cannot cancel
        a kernel already executing; the job launcher must terminate failed jobs.
        Successful cached forwards do not add a host collective to the hot path.
        """
        with self._lock:
            try:
                self._runtime.ensure_alive()
                if self._failure is not None:
                    raise RuntimeError(f"URMA buffer is unusable after a failed operation: {self._failure}")
                yield
            except Exception as error:
                if self._failure is None:
                    self._failure = f"{stage}: {type(error).__name__}: {error}"
                    self.group.abort(self._failure)
                raise

    def _enqueue_forward(self, prepared, operands):
        """Serialize every native launch on the owner's device stream.

        The host RLock alone cannot serialize work on different NPU streams.
        These two event edges also order metadata/input writes and output reads.
        """
        runtime = self._runtime
        npu = npu_api()
        caller = npu.current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        with npu.stream(runtime.stream):
            result = prepared.native.forward(*operands)
        caller.wait_stream(runtime.stream)
        result.record_stream(caller)
        return result

    def prepare_routes(self, topk_idx):
        """Bind a contiguous Device top-k tensor; never inspect its values.

        The caller guarantees legal expert IDs. The kernel performs fresh
        histogram/prefix/exchange/plan construction on every invocation.
        Multiple bindings reuse this owner's stream-serialized workspace.
        """
        import torch

        with self._operation("route binding"):
            if topk_idx.device != self._device or topk_idx.dtype not in (torch.int32, torch.int64):
                raise ValueError("topk_idx must be int32/int64 on the buffer's NPU")
            if topk_idx.ndim != 2 or topk_idx.shape[1] != self.num_topk or not topk_idx.is_contiguous():
                raise ValueError("topk_idx must be contiguous [BS, topk]")
            bs = topk_idx.shape[0]
            if not 0 < bs <= self.num_max_tokens_per_rank:
                raise ValueError(f"BS={bs} exceeds allocated capacity {self.num_max_tokens_per_rank}")
            if self._plan is None:
                address = require_native("URMA MegaMoE").megamoe_workspace_address(self._runtime.native)
                self._plan = _extension().UrmaPlan(
                    self._example,
                    self.group.size(),
                    self.intermediate_hidden,
                    self.num_topk,
                    self.cv_rows,
                    self.max_recv_rows,
                    self.num_max_tokens_per_rank,
                    self.group.rank(),
                    _extension().ffts_config(),
                    address,
                    self._runtime.native.capacity,
                    self.gmm1_ahead_mgroups,
                    self.num_shared_experts,
                    self.shared_intermediate_hidden,
                    self.num_experts,
                )
            routing = DeviceRoutes(*self._plan.routing_tensors(bs))
            return PreparedUrmaRoutes(self, self._plan, routing, bs, topk_idx)

    def forward_prepared(
        self,
        x,
        topk_weights,
        l1_weights,
        l2_weights,
        prepared,
        *,
        timer=None,
        out=None,
        trace=None,
        coarse_trace=False,
        shared_l1_weights=None,
        shared_l2_weights=None,
    ):
        """Expand/Dispatch/GMM1/SwiGLU/GMM2/return PUT and flag-gated weighted reduction."""
        import torch
        from .api import _check_fp4_weight_format, _RoutedWeightPair

        with self._operation("forward"):
            if not isinstance(prepared, PreparedUrmaRoutes) or prepared.owner is not self:
                raise ValueError("prepared routes belong to a different buffer")
            if not isinstance(x, tuple) or len(x) != 2:
                raise TypeError("x must be (FP8 E4M3 data, E8M0 scales)")
            data, scales = x
            if data.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
                raise TypeError("URMA requires FP8 E4M3 data and E8M0 scales")
            if tuple(data.shape) != (prepared.bs, self.hidden) or tuple(scales.shape) not in (
                (prepared.bs, self.hidden // 32),
                (prepared.bs, self.hidden // 64, 2),
            ):
                raise ValueError("input or scale shape differs from the prepared routes")
            scales = scales.reshape(prepared.bs, self.hidden // 64, 2)
            if tuple(topk_weights.shape) != (prepared.bs, self.num_topk) or topk_weights.dtype != torch.float32:
                raise ValueError("top-k weights must be float32 [BS, topk]")
            if not isinstance(l1_weights, _RoutedWeightPair):
                raise ValueError("prepare both GMM1 layouts using transform_weights_for_mega_moe")
            weight1, scale1 = l1_weights.select(
                use_fused_gate_up(prepared.bs, self.group.size(), self.num_topk, self.num_experts)
            )
            weight2, scale2 = l2_weights
            for name, weight in (("l1_weights", weight1), ("l2_weights", weight2)):
                if weight.dtype != torch.uint8:
                    raise TypeError("URMA first profile requires packed FP4 weights")
                _check_fp4_weight_format(weight, name)
            shared = ()
            if self.num_shared_experts:
                if shared_l1_weights is None or shared_l2_weights is None:
                    raise ValueError("S1 needs both shared weight pairs, transformed with shared=True")
                sw1, ss1 = shared_l1_weights
                sw2, ss2 = shared_l2_weights
                if isinstance(shared_l1_weights, _RoutedWeightPair):
                    raise ValueError("Shared GMM1 requires planar shared=True prepared weights")
                import torch_npu

                for sw in (sw1, sw2):
                    if sw.dtype != torch.float8_e4m3fn or not sw.is_contiguous() or torch_npu.get_npu_format(sw) != 2:
                        raise ValueError(
                            "Shared weights require contiguous ND FP8 E4M3 [1,N,K]; routed weights remain FP4"
                        )
                shared = (sw1, sw2, ss1, ss2)
            elif shared_l1_weights is not None or shared_l2_weights is not None:
                raise ValueError("S0 does not accept shared weights")
            if trace is not None:
                extension = _trace_extension()
                row_elements = extension.urma_trace_num_engines * extension.urma_trace_record_elements
                if (
                    trace.dtype != torch.int64
                    or trace.ndim != 1
                    or not trace.is_contiguous()
                    or trace.numel() % row_elements
                    or not 2 * row_elements <= trace.numel() <= (extension.urma_trace_max_capacity + 1) * row_elements
                ):
                    raise ValueError(
                        "trace must be contiguous 1D int64 with "
                        "84 * (16 + capacity * 16) elements, capacity in [1,65536]"
                    )
                if trace.data_ptr() % 128:
                    raise ValueError("trace storage must be 128-byte aligned; use UrmaBuffer.allocate_trace()")
            operands = (data, scales, topk_weights, weight1, weight2, scale1, scale2, timer, out, trace, *shared)
            if any(value is not None and value.device != self._device for value in operands):
                raise ValueError("all tensors must reside on the creating NPU")
            if any(value is not None and value.requires_grad for value in operands):
                raise ValueError("URMA MegaMoE is inference-only")
            if self._epoch >= (1 << 31) - 1:
                raise RuntimeError("URMA epoch exhausted; destroy and recreate the collective runtime")
            self._epoch += 1
            self._last_prepared = prepared
            native_operands = (
                data,
                scales,
                prepared.topk_idx,
                topk_weights,
                weight1,
                weight2,
                scale1,
                scale2,
                timer,
                out,
                self._epoch,
            )
            # Optional instrumentation remains a per-invocation argument.
            if coarse_trace and (trace is None or timer is not None):
                raise ValueError("coarse_trace requires trace storage and no aggregate timer")
            if self.num_shared_experts:
                native_operands += (trace, coarse_trace, *shared)
            elif trace is not None:
                native_operands += (trace,)
                if coarse_trace:
                    native_operands += (True,)
            return self._enqueue_forward(prepared, native_operands)

    def forward(
        self,
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        *,
        timer=None,
        out=None,
        trace=None,
        coarse_trace=False,
        shared_l1_weights=None,
        shared_l2_weights=None,
    ):
        """Complete invocation, including fresh real-input route preparation."""
        with self._lock:
            prepared = self.prepare_routes(topk_idx)
            options = dict(timer=timer, out=out)
            if shared_l1_weights is not None or shared_l2_weights is not None:
                options.update(shared_l1_weights=shared_l1_weights, shared_l2_weights=shared_l2_weights)
            if coarse_trace:
                options["coarse_trace"] = True
            if trace is not None:
                options["trace"] = trace
            return self.forward_prepared(x, topk_weights, l1_weights, l2_weights, prepared, **options)

    def destroy(self):
        with self._lock:
            if self._runtime.closed:
                return
            if self._failure is not None:
                raise RuntimeError(
                    "URMA job was aborted; terminate all workers instead of entering collective teardown"
                )
            try:
                # Runtime.close synchronizes the dedicated execution stream
                # before releasing the shared workspace and communication domain.
                self._runtime.close()
            except Exception as error:
                self._failure = f"destroy: {type(error).__name__}: {error}"
                self.group.abort(self._failure)
                raise
            self._last_prepared = None
            self._plan = None
            self._example = None

    def __del__(self):
        warn_unclosed(self)
