# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Thin inference-only API; all algorithm selection remains in the native layer."""

from __future__ import annotations

from functools import lru_cache, wraps
import importlib
import numbers
import struct
import threading
from typing import TYPE_CHECKING

from deep_ep._runtime import Runtime, integer, warn_unclosed
from deep_ep._native import npu_api, require_native
from deep_ep._store import StoreGroup
from .validation import check_routing, collective_check

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributed import ProcessGroup

    from .weights import WeightHandle

    WeightPair = tuple[Tensor, Tensor]
    WeightInput = (
        WeightPair | list[WeightPair] | tuple[list[Tensor] | tuple[Tensor, ...], list[Tensor] | tuple[Tensor, ...]]
    )


@lru_cache(maxsize=1)
def _extension():
    core = require_native("MegaMoE")
    if not hasattr(core, "create_megamoe_runtime"):
        raise RuntimeError("Install a wheel built with DEEPEP_BUILD_MEGAMOE=ON")
    try:
        extension = importlib.import_module(__package__ + "._ep32_C")
    except ImportError as error:
        raise ImportError("Cannot load MegaMoE Beta: check matching Torch, torch_npu, CANN and SHMEM") from error
    if getattr(extension, "api_protocol", None) != 17:
        raise RuntimeError("MegaMoE Python/native versions differ; rebuild and reinstall the complete Beta wheel")
    return extension


class _MegaRuntime(Runtime):
    _megamoe = True

    def __init__(self, group, capacity, timeout=120, *, ranks_per_node=0):
        self._ranks_per_node = ranks_per_node
        if ranks_per_node == 16 and group.size() == 32:
            self._megamoe_backend = "ep32"
        super().__init__(group, capacity, timeout)


def _activation_clamp(value):
    maximum = float.fromhex("0x1.fffffep127")
    if value is None:
        return maximum
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("activation_clamp must be a nonnegative finite host real")
    if not 0 <= value <= maximum:
        raise ValueError("activation_clamp must be finite and in the FP32 range")
    return struct.unpack("f", struct.pack("f", value))[0]


def _pair(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be (quantized_data, E8M0_scales)")
    return value


def _check_fp4_weight_format(weight, name):
    import torch_npu

    actual = torch_npu.get_npu_format(weight)
    # 请求 NZ(29) 不等于返回码必为29：torch_npu 将 FP4 C0=32 映射成 uint8 C0=16(50)。
    # 不能放行所有 NZ 子格式，uint8 C0=32(51) 并非本路径的打包布局。
    if actual not in (29, 50):
        raise ValueError(
            f"{name}: packed FP4 weights require NZ format 29 or 50 (NZ_C0_16), "
            f"got format={actual}, dtype={weight.dtype}, shape={tuple(weight.shape)}; "
            "call transform_weights_for_mega_moe on raw ND weights once"
        )


class _PlanDomains:
    def __init__(self, envelope, configs):
        self.envelope = envelope
        self.configs = configs
        self.lock = threading.RLock()
        self.next_domain = 0
        self.views = 1
        self.next_view = 1
        self.frozen = False
        self.failed = False


class SymmBuffer:
    """Collective resource owner. All ranks construct/call/destroy in the same order.

    num_experts is global; intermediate_hidden is the post-SwiGLU width.
    This Beta supports nonempty, same-BS calls and one in-flight operation per buffer.
    """

    def __new__(
        cls,
        group: ProcessGroup | StoreGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        *,
        ranks_per_node: int = 0,
        timeout: int = 120,
        topk_weights_type: int = 0,
        base: SymmBuffer | None = None,
        num_max_configs: int | None = None,
    ):
        if group is None or cls is not SymmBuffer:
            return super().__new__(cls)
        from .backend import select_backend
        from . import _platform_api

        if not hasattr(group, "size"):
            return super().__new__(cls)
        backend = select_backend(group, num_max_tokens_per_rank, ranks_per_node, base=base)
        error = None
        try:
            integer("topk_weights_type", topk_weights_type, 0, 1)
            if backend == "mte" and topk_weights_type and ranks_per_node == 16 and group.size() > 16:
                raise ValueError("EP32 hierarchical RoCE supports topk_weights_type=0")
            if (backend == "urma" or topk_weights_type) and (base is not None or num_max_configs not in (None, 1)):
                raise ValueError("Multi-config arenas require hierarchical MTE/RoCE with post-GMM2 routing weights")
        except (TypeError, ValueError) as failure:
            error = str(failure)
        collective_check(group, "buffer implementation", error, (topk_weights_type, ranks_per_node))
        if backend == "urma" or topk_weights_type != 0:
            obj = super().__new__(_platform_api.SymmBuffer)
        else:
            obj = super().__new__(cls)
        obj._backend_name = backend
        return obj

    def __init__(
        self,
        group: ProcessGroup | StoreGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        *,
        ranks_per_node: int = 0,
        timeout: int = 120,
        topk_weights_type: int = 0,
        base: SymmBuffer | None = None,
        num_max_configs: int | None = None,
    ):
        import torch
        import torch.distributed as dist

        if not isinstance(group, (StoreGroup, dist.ProcessGroup)):
            raise TypeError("Pass a StoreGroup or an initialized ProcessGroup explicitly")
        size = group.size() if isinstance(group, StoreGroup) else dist.get_world_size(group)
        error = None
        try:
            if num_max_configs is None:
                num_max_configs = (
                    base._domains.configs
                    if base is not None
                    else (16 if size == 32 and ranks_per_node == 16 and num_max_tokens_per_rank <= 256 else 1)
                )
            integer("num_max_configs", num_max_configs, 1, 4096)
            if num_max_configs > 1 and not (size == 32 and ranks_per_node == 16 and num_max_tokens_per_rank <= 256):
                raise ValueError("multi-domain plans require EP32 with 16 ranks per host and capacity <= 256")
            if base is not None:
                base._ensure_alive()
                if base._device != torch.device(f"npu:{npu_api().current_device()}"):
                    raise ValueError("Shared views must use the same NPU device")
                if (
                    base.group is not group
                    or base.ranks_per_node != ranks_per_node
                    or base.mma_type != mma_type
                    or base._timeout != timeout
                    or base._domains.configs != num_max_configs
                    or base._domains.frozen
                    or num_max_configs == 1
                ):
                    raise ValueError("Shared views require matching runtime options and an open multi-config setup")
                dims = (num_experts // size, num_max_tokens_per_rank, num_topk, hidden)
                if any(v > limit for v, limit in zip(dims, base._domains.envelope)):
                    raise ValueError("view exceeds the root envelope")
            integer("num_experts", num_experts, 1, 16384)
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 65536)
            integer("num_topk", num_topk, 1, min(32, num_experts))
            integer("hidden", hidden, 1024, 8192)
            integer("intermediate_hidden", intermediate_hidden, 256, 4096)
            integer("num_shared_experts", num_shared_experts, 0, 8)
            integer("world_size", size, 2, 128)
            integer("ranks_per_node", ranks_per_node, 0, size)
            integer("timeout", timeout, 1, 300)
            if hidden % 64 or intermediate_hidden % 128:
                raise ValueError("hidden must be divisible by 64; intermediate_hidden by 128")
            if activation != "swiglu" or mma_type not in {"fp8xfp4", "fp8xfp8"}:
                raise ValueError("Only SwiGLU with FP8xFP4/FP8xFP8 is implemented; no BF16xBF16")
            if mma_type != "fp8xfp4" and num_shared_experts:
                raise ValueError("Shared experts currently require FP8xFP4 weights")
            if num_experts % size or num_experts // size > 128:
                raise ValueError("Experts must divide EP, with at most 128 local experts")
            if ranks_per_node and size % ranks_per_node:
                raise ValueError("ranks_per_node must divide EP (contiguous node-major ranks)")
            if ranks_per_node == 16 and size > 16:
                if mma_type != "fp8xfp4":
                    raise ValueError("MTE+RoCE MegaMoE requires FP8xFP4 weights")
                if size != 32:
                    raise ValueError("MTE+RoCE MegaMoE requires EP32 on two 16-rank hosts")
        except (AttributeError, RuntimeError, TypeError, ValueError) as failure:
            error = str(failure)
        # 在进入 SHMEM 之前共同校验布局，不能各 rank 各自选择通信偏移。
        config = (
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            mma_type,
            activation,
            ranks_per_node,
            timeout,
            num_max_configs,
            None if base is None else base._view_id,
        )
        collective_check(group, "configuration", error, config)
        ext = _extension()
        self._domains = (
            base._domains
            if base is not None
            else _PlanDomains((num_experts // size, num_max_tokens_per_rank, num_topk, hidden), num_max_configs)
        )
        capacity = int(ext.buffer_size(size, *self._domains.envelope, ranks_per_node, num_max_configs))
        self._lock = self._domains.lock
        self._timeout = timeout
        self._view_id = 0 if base is None else self._domains.next_view
        self._closed = False
        self._plans = {}
        self._owners = set()
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._plan = self._plan_bs = None
        self.group, self.num_experts = group, num_experts
        self.num_max_tokens_per_rank, self.num_topk = num_max_tokens_per_rank, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts, self.mma_type = num_shared_experts, mma_type
        self.topk_weights_type = topk_weights_type
        self.ranks_per_node = ranks_per_node
        self._hierarchical_roce = ranks_per_node == 16 and size > 16
        self._sequence = 0
        self._roce_config = {}
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        error = None
        config = ()
        try:
            if self._hierarchical_roce:
                native = require_native("MegaMoE")
                fields = ("combine_reduce_version", "combine_qps", "dispatch_qps", "dispatch_workers")
                config = tuple(getattr(ext, f, None) for f in fields)
                if (
                    config[0] != 4
                    or None in config
                    or config != tuple(getattr(native, "megamoe_" + f, None) for f in fields)
                ):
                    raise RuntimeError(
                        "Rebuild the complete wheel: MegaMoE weighted reduction protocol/configuration mismatch"
                    )
                self._roce_config = dict(zip(fields, config))
            self.allocate_inputs()
        except Exception as failure:
            error = str(failure)
        # 输入分配失败先通知所有 rank，再进入 SHMEM，避免其他 rank 已在建链等待。
        collective_check(group, "input allocation", error, config)
        runtime_options = dict(ranks_per_node=ranks_per_node)
        with self._lock:
            if base is None:
                self._runtime = _MegaRuntime(group, capacity, timeout, **runtime_options)
            else:
                base._ensure_alive()
                self._runtime = base._runtime
                self._domains.views += 1
                self._domains.next_view += 1

    def _ensure_alive(self):
        if self._closed or self._domains.failed:
            raise RuntimeError("SymmBuffer view was destroyed or its collective setup failed")
        self._runtime.ensure_alive()

    def allocate_inputs(self):
        """Legacy idempotent helper. Inputs are already allocated by the constructor."""
        import torch

        if hasattr(self, "_runtime"):
            self._ensure_alive()
        if self.x is not None:
            return self
        self.x = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden), dtype=torch.float8_e4m3fn, device=self._device
        )
        self.x_sf = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden // 32), dtype=torch.float8_e8m0fnu, device=self._device
        )
        self.topk_idx = torch.empty(
            (self.num_max_tokens_per_rank, self.num_topk), dtype=torch.int32, device=self._device
        )
        self.topk_weights = torch.empty_like(self.topk_idx, dtype=torch.float32)
        return self

    def _prepare(self, bs):
        integer("BS", bs, 1, self.num_max_tokens_per_rank)
        # Keep a conservative limit if a backend omits the capacity capability.
        limit = getattr(_extension(), "hierarchical_max_tokens", 32)
        if self._hierarchical_roce and bs > limit:
            raise ValueError(f"MTE+RoCE MegaMoE currently supports decode BS in [1,{limit}]; rebuild for larger BS")
        if self._plan_bs == bs or (self._plan is not None and not self._hierarchical_roce):
            return
        shared = self._domains
        if bs in self._plans:
            self._plan, self._plan_bs = self._plans[bs], bs
            return
        domain = 0
        if shared.configs > 1:
            if shared.frozen or shared.next_domain >= shared.configs:
                raise ValueError("plan configuration is frozen or num_max_configs exhausted")
            domain = shared.next_domain
            shared.next_domain += 1
        # Single-domain plans replace the previous workspace. Multi-domain plans retain their addresses.
        self._plan = self._plan_bs = None
        runtime = self._runtime
        address = require_native("MegaMoE").megamoe_workspace_address(runtime.native)
        ext = _extension()
        self._plan = ext.Plan(
            self.x[:bs] if self._hierarchical_roce else self.x,
            self.intermediate_hidden,
            self.num_topk,
            self.num_experts // self.group.size(),
            self.group.size(),
            self.num_max_tokens_per_rank,
            self.mma_type == "fp8xfp4",
            self.num_shared_experts,
            self.ranks_per_node,
            ext.ffts_config(),
            address,
            runtime.native.capacity,
            shared.envelope,
            domain,
            shared.configs,
        )
        if shared.configs > 1:
            self._plans[bs] = self._plan
        self._plan_bs = bs if self._hierarchical_roce else self.num_max_tokens_per_rank

    def prepare(self, bs: int) -> SymmBuffer:
        """Collectively prepare/reuse one BS before timing; all ranks must call together."""
        if npu_api().is_current_stream_capturing():
            raise RuntimeError("collective prepare must run outside graph capture")
        error = None
        with self._lock:
            try:
                self._ensure_alive()
                integer("BS", bs, 1, self.num_max_tokens_per_rank)
            except Exception as failure:
                error = str(failure)
            try:
                collective_check(
                    self.group, "prepare", error, (self._view_id, bs, self._domains.next_domain, self._domains.frozen)
                )
            except Exception:
                if self._domains.configs > 1:
                    self._domains.failed = True
                raise
            try:
                runtime = self._runtime
                runtime.stream.wait_stream(npu_api().current_stream(runtime.device))
                with npu_api().stream(runtime.stream):
                    self._prepare(bs)
            except Exception as failure:
                error = str(failure)
            try:
                collective_check(self.group, f"prepare BS={bs}", error)
            except Exception:
                if self._domains.configs > 1:
                    self._domains.failed = True
                raise
        return self

    def set_wave(self, wave: int):
        """Collectively set the prepared batch's wave override (0 selects auto)."""
        with self._lock:
            error = None
            try:
                self._ensure_alive()
                integer("wave", wave, 0, 64)
                if self._plan is None or npu_api().is_current_stream_capturing():
                    raise ValueError("set_wave requires a prepared plan outside capture")
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, "set_wave", error, (self._view_id, self._plan_bs, wave))
            self._runtime.stream.synchronize()
            self._plan.set_wave(wave)

    @property
    def workspace_bytes(self) -> int:
        """Prepared local scratch only; excludes symmetric heap, inputs and weights."""
        return self._plan.workspace_bytes if self._plan is not None else 0

    @property
    def symmetric_buffer_bytes(self) -> int:
        return self._runtime.native.capacity

    @property
    def timer_enabled(self) -> bool:
        return bool(_extension().timer_enabled and self.mma_type == "fp8xfp4")

    def allocate_timer(self) -> Tensor:
        """Fresh zeroed device trace; pass only to a separate diagnostic invocation."""
        import torch

        self._ensure_alive()
        if not self.timer_enabled:
            raise RuntimeError("W4A8 timer requires DEEPEP_MEGAMOE_TIMER=ON")
        return torch.zeros(_extension().timer_numel, dtype=torch.int64, device=self._device)

    def validate_inputs(self, bs: int, *, x: Tensor | None = None) -> None:
        """Collective, synchronizing routing scan. Run after filling inputs, outside timing."""
        error = None
        try:
            import torch

            self._ensure_alive()
            integer("BS", bs, 1, self.num_max_tokens_per_rank)
            for name, tensor, dtype, shape in (
                ("topk_idx", self.topk_idx, torch.int32, (self.num_max_tokens_per_rank, self.num_topk)),
                ("topk_weights", self.topk_weights, torch.float32, (self.num_max_tokens_per_rank, self.num_topk)),
            ):
                if tensor.device != self._device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(f"{name} must retain its buffer dtype/device/shape")
            if x is None:
                for name, tensor, dtype, shape in (
                    ("x", self.x, torch.float8_e4m3fn, (self.num_max_tokens_per_rank, self.hidden)),
                    ("x_sf", self.x_sf, torch.float8_e8m0fnu, (self.num_max_tokens_per_rank, self.hidden // 32)),
                ):
                    if (
                        tensor.device != self._device
                        or tensor.dtype != dtype
                        or tuple(tensor.shape) != shape
                        or not tensor.is_contiguous()
                    ):
                        raise ValueError(f"{name} must retain its buffer dtype/device/contiguous shape")
            if x is not None and (
                x.dtype != torch.bfloat16 or x.device != self._device or tuple(x.shape) != (bs, self.hidden)
            ):
                raise ValueError("x override must be BF16 [BS,H] on the buffer device")
            check_routing(
                self.topk_idx[:bs].cpu().tolist(),
                self.topk_weights[:bs].cpu().tolist(),
                self.num_experts,
                self.num_topk,
            )
        except Exception as failure:
            error = str(failure)
        collective_check(self.group, "input preflight", error, (bs, x is not None))

    def benchmark_sleep(self, cycles):
        """Queue a profiling delay on the current stream; no host synchronization."""
        with self._lock:
            self._ensure_alive()
            extension = _extension()
            if not hasattr(extension, "benchmark_sleep") or not hasattr(extension, "benchmark_barrier"):
                raise RuntimeError("SHMEM profiling requires rebuilding and reinstalling the MegaMoE wheel")
            extension.benchmark_sleep(cycles)

    def benchmark_barrier(self):
        """Collectively queue SHMEM WORLD synchronization before the next call.

        The current stream joins prior MegaMoE work; the next call waits on this
        stream. Every rank must enqueue the same number of barriers in order.
        """
        with self._lock:
            self._ensure_alive()
            _extension().benchmark_barrier(self._hierarchical_roce)

    def destroy(self):
        with self._lock:
            if self._closed:
                return
            if self._domains.views == 1:
                self._runtime.close()
            else:
                self._runtime.stream.synchronize()
            self._domains.views -= 1
            self._closed = True
            self._plans.clear()
            self._owners.clear()
            self._plan = self._plan_bs = None
            self.x = self.x_sf = self.topk_idx = self.topk_weights = None

    def __del__(self):
        if not getattr(self, "_closed", True):
            warn_unclosed(self)


def _interleave_gate_up_128(tensor):
    """Permute the N axis of raw [E, 2I, ...] storage without numeric conversion."""
    experts, columns, *tail = tensor.shape
    return (
        tensor.reshape((experts, 2, columns // 256, 128, *tail))
        .transpose(1, 2)
        .contiguous()
        .reshape((experts, columns, *tail))
    )


class _SharedFp8Pair(tuple):
    """Prepared ND GMM1 with matching interleaved weight and scale rows."""


class _RoutedWeightPair(tuple):
    """Tuple-compatible immutable prepared weights; both layouts are built off the hot path."""

    def __new__(cls, fused, planar):
        value = super().__new__(cls, fused)
        value._planar = planar
        value._versions = tuple(t._version for t in (*fused, *planar))
        return value

    def select(self, fused):
        if tuple(t._version for t in (*self, *self._planar)) != self._versions:
            raise ValueError("Prepared GMM1 weights/scales were modified in-place; transform raw ND weights again")
        return tuple(self) if fused else self._planar


def transform_weights_for_mega_moe(
    l1_weights: WeightInput, l2_weights: WeightInput, activation: str = "swiglu", *, shared: bool = False
) -> tuple[WeightPair, WeightPair] | tuple[WeightHandle, WeightHandle]:
    """Convert raw Ascend packed weights to NZ once, without quantizing again.

    Per-expert lists return two handles sharing a prepared, updatable owner.
    Raw inputs use [gate | up]. Routed FP4 GMM1 prepares both planar Prefill and
    [gate128, up128] Decode layouts. Keep the returned l1 pair intact and immutable;
    the API selects its layout by actual BS without a timed conversion. Pass shared=True for
    shared experts. Shared FP4 GMM1 stays planar and shared FP8 GMM1 interleaves gate/up with their scales.
    This is not the CUDA UTCCP transform. Old prepared FP4 routed weights must
    be regenerated from raw ND inputs; do not transform prepared NZ twice.
    """
    if activation != "swiglu":
        raise NotImplementedError("Only SwiGLU is supported")
    from .weights import _expert_group, prepare_weights_for_mega_moe

    if _expert_group(l1_weights) is not None or _expert_group(l2_weights) is not None:
        owner = prepare_weights_for_mega_moe(l1_weights, l2_weights, shared=shared)
        return owner.l1, owner.l2
    return _transform_nd_pairs(l1_weights, l2_weights, shared=shared, single_expert=False)


def _transform_nd_pairs(l1_weights, l2_weights, *, shared, single_expert):
    import torch
    import torch_npu

    from .weights import _format_class, _fp4_cast

    def packed_nz(weight):
        if not single_expert:
            return _fp4_cast(weight, 29)
        physical = weight[0].contiguous()
        if physical.storage_offset():
            physical = physical.clone()
        owner = _fp4_cast(physical, 29)
        logical = owner.unsqueeze(0)
        logical._megamoe_nz_owner = owner
        return logical

    transformed = []
    for name, value in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
        weight, scale = _pair(value, name)
        if weight.device.type != "npu" or scale.device != weight.device:
            raise ValueError("Move weights and scales to the same NPU before preparing")
        if weight.ndim != 3:
            raise ValueError(f"{name}: weights must be [E,N,K] or packed [E,N,K/2]")
        if weight.dtype in {torch.int8, torch.uint8} and torch_npu.get_npu_format(weight) in (29, 50):
            raise ValueError(f"{name}: prepare from raw ND weights, not already converted NZ weights")
        if any(_format_class(t) != 2 for t in (weight, scale)):
            raise ValueError(f"{name}: raw weights and scales require linear ND storage")
        if weight.dtype == torch.int8:
            weight = weight.view(torch.uint8)  # 重解释 FP4 的位模式，不能做数值类型转换。
        if weight.dtype not in {torch.uint8, torch.float8_e4m3fn}:
            raise TypeError("Weights must be packed FP4 int8/uint8 or FP8 E4M3")
        k = weight.shape[-1] * (2 if weight.dtype == torch.uint8 else 1)
        expected = (*weight.shape[:2], k // 64, 2)
        if k % 64:
            raise ValueError(f"{name}: K must be divisible by 64")
        if scale.dtype == torch.uint8:
            scale = scale.view(torch.float8_e8m0fnu)
        if scale.dtype == torch.int32 and tuple(scale.shape) == (*weight.shape[:2], k // 128) and k % 128 == 0:
            # 仅接受逻辑行优先、尚未做 CUDA UTCCP 重排的 packed E8M0。
            scale = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape(expected)
        if scale.dtype != torch.float8_e8m0fnu or tuple(scale.shape) != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected} or raw packed int32 [E,N,K/128]")
        scale = scale.contiguous()
        if weight.dtype == torch.uint8:
            if name == "l1_weights" and not shared:
                if weight.shape[1] % 256:
                    raise ValueError("Routed FP4 GMM1 requires gate/up widths divisible by 128")
                # 两种布局一次准备；独立存储并启用版本计数，禁止原地修改后两份权重失配。
                with torch.inference_mode(False):
                    planar_weight = packed_nz(weight)
                    planar_scale = scale.view(torch.uint8).clone().view(torch.float8_e8m0fnu)
                    weight = _interleave_gate_up_128(weight)
                    scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
                    weight = packed_nz(weight)
                    _check_fp4_weight_format(planar_weight, name)
                    _check_fp4_weight_format(weight, name)
                    transformed.append(_RoutedWeightPair((weight, scale), (planar_weight, planar_scale)))
                continue
            # uint8 只是两个 FP4 的存储载体；FP8 布局参数不能代替输入的 FP4 类型声明。
            weight = packed_nz(weight)
            _check_fp4_weight_format(weight, name)
        if shared and weight.dtype == torch.float8_e4m3fn and name == "l1_weights":
            if weight.shape[1] % 256:
                raise ValueError("shared FP8 GMM1 requires 128-column gate/up blocks")
            planar = weight, scale
            weight = _interleave_gate_up_128(weight.view(torch.uint8)).view(torch.float8_e4m3fn)
            scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
            fused = _SharedFp8Pair((weight, scale))
            fused.planar = planar
            transformed.append(fused)
        else:
            transformed.append((weight, scale))
    if transformed[0][0].dtype != transformed[1][0].dtype:
        raise ValueError("Both projections must have the same weight dtype")
    return tuple(transformed)


def mega_moe(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    activation_clamp=None,
    timer=None,
):
    """Explicit-input convenience API; prefer fp8_fp4_mega_moe for DeepGEMM-style use.

    x: BF16 [B,H], or (FP8 E4M3 [B,H], E8M0 [B,H/32]).
    FP8 is copied into the original communication record, never dequantized/requantized.
    Routed work uses the prefill scheduler; W4 shared strategy switches at B=256.
    For graph replay, prepare each view and batch size before capturing the forward.
    """
    return _run(
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=out,
        activation_clamp=activation_clamp,
        timer=timer,
    )


def _run(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, **options):
    from . import _platform_api
    if isinstance(sym_buffer, _platform_api.SymmBuffer):
        return _platform_api._run(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, **options)
    from contextlib import ExitStack
    from .weights import WeightHandle

    owners = [h.owner for h in (l1_weights, options.get("shared_l1_weights")) if isinstance(h, WeightHandle)]
    if not owners:
        return _run_locked(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, **options)
    with ExitStack() as stack:
        for owner in sorted(set(owners), key=id):
            stack.enter_context(owner._lock)
        return _run_locked(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, **options)


def _run_locked(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    activation_clamp=None,
    timer=None,
    cumulative_stats=None,
):
    import torch

    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    clamp = _activation_clamp(activation_clamp)
    data, scales = _pair(x, "x") if isinstance(x, tuple) else (x, None)
    from .weights import WeightHandle

    managed_routed = isinstance(l1_weights, WeightHandle)
    owners = []
    tables = []

    def resolve(l1, l2, shared):
        if not isinstance(l1, WeightHandle) and not isinstance(l2, WeightHandle):
            tables.append(None)
            return l1, l2
        if (
            not isinstance(l1, WeightHandle)
            or l1 is not l1.owner.l1
            or l2 is not l1.owner.l2
            or l1.owner.shared != shared
        ):
            raise ValueError("Supply both prepared handles from the same owner")
        owner = l1.owner
        size = sym_buffer.num_shared_experts if shared else sym_buffer.num_experts // sym_buffer.group.size()
        if size != owner.count:
            raise ValueError("Expert count differs from prepared owner")
        pair1, pair2, table = owner.select(data.shape[0] <= _extension().small_batch_max_tokens)
        tables.append(table)
        owners.append(owner)
        return pair1, pair2

    l1_weights, l2_weights = resolve(l1_weights, l2_weights, False)
    shared_l1_weights, shared_l2_weights = resolve(shared_l1_weights, shared_l2_weights, True)
    w1, s1 = _pair(l1_weights, "l1_weights")
    w2, s2 = _pair(l2_weights, "l2_weights")
    shared = (None, None, None, None)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("Supply both shared weight pairs, or neither")
    if shared_l1_weights is not None:
        sw1, ss1 = _pair(shared_l1_weights, "shared_l1_weights")
        sw2, ss2 = _pair(shared_l2_weights, "shared_l2_weights")
        if sw1.dtype == torch.float8_e4m3fn and not isinstance(shared_l1_weights, _SharedFp8Pair):
            raise ValueError("shared FP8 GMM1 needs a prepared interleaved weight and scale pair")
        shared = (sw1, sw2, ss1, ss2)
    b = sym_buffer
    if data.ndim != 2 or not 0 < data.shape[0] <= b.num_max_tokens_per_rank or data.shape[1] != b.hidden:
        raise ValueError("x must be nonempty [BS,H] within the buffer capacity")
    if data.device != b._device:
        raise ValueError("x must be on the buffer's creating device")
    expected = (b.num_experts // b.group.size(), b.intermediate_hidden * 2)
    if tables[0] is not None:
        expected = (1, expected[1])
    if tuple(w1.shape[:2]) != expected or tuple(topk_idx.shape) != (data.shape[0], b.num_topk):
        raise ValueError("Weight/routing shapes do not match the buffer configuration")
    if tables[1] is None and (shared[0].shape[0] if shared[0] is not None else 0) != b.num_shared_experts:
        raise ValueError("Shared expert count does not match the buffer configuration")
    expected_dtype = torch.uint8 if b.mma_type == "fp8xfp4" else torch.float8_e4m3fn
    if w1.dtype != expected_dtype:
        raise TypeError("Weight dtype differs from buffer mma_type")
    prepared_inputs = ()
    if w1.dtype == torch.uint8:
        fused = data.shape[0] <= _extension().small_batch_max_tokens
        if isinstance(l1_weights, _RoutedWeightPair):
            prepared_inputs = (*l1_weights, *l1_weights._planar)
            w1, s1 = l1_weights.select(fused)
        elif not fused and not managed_routed:
            raise ValueError(
                "Prefill requires both prepared GMM1 layouts; call transform_weights_for_mega_moe "
                "on raw ND weights and keep its returned l1 pair intact"
            )
    if any(t is not None and t.requires_grad for t in (data, scales, topk_weights, w1, s1, w2, s2, out, *shared)):
        raise NotImplementedError("MegaMoE Beta is inference-only")
    if out is not None and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, *prepared_inputs)
        if tensor is not None
    ):
        raise ValueError("out must not overlap any input or timer storage")
    for owner in owners:
        if any(
            t is not None and t.untyped_storage().data_ptr() in owner.storage_bases
            for t in (data, scales, topk_idx, topk_weights, out, timer, cumulative_stats)
        ):
            raise ValueError("call tensors overlap prepared expert storage")
    if topk_idx.dtype != torch.int32:
        raise TypeError("topk_idx must be int32; convert explicitly before timing")
    if w1.dtype == torch.uint8:
        for name, weight in (
            ("l1_weights", w1),
            ("l2_weights", w2),
            ("shared_l1_weights", shared[0]),
            ("shared_l2_weights", shared[1]),
        ):
            if weight is not None and weight.dtype == torch.uint8:
                _check_fp4_weight_format(weight, name)
    if cumulative_stats is not None:
        stats = cumulative_stats
        if (
            stats.dtype != torch.int32
            or tuple(stats.shape) != (b.num_experts // b.group.size(),)
            or stats.device != b._device
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(
            stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, out, timer, *shared, *prepared_inputs)
            if t is not None
        ):
            raise ValueError("Cumulative stats must not alias inputs, output or timer")
    runtime = b._runtime
    with b._lock:
        b._ensure_alive()
        caller = npu_api().current_stream(runtime.device)
        if caller != runtime.stream:
            runtime.stream.wait_stream(caller)
        # 同一 owner 的调用串行进入专属流，禁止不同流覆盖仍在使用的对称空间。
        with npu_api().stream(runtime.stream):
            if b._domains.configs > 1 and data.shape[0] not in b._plans:
                raise RuntimeError("Collectively prepare this view and BS before its first forward")
            b._prepare(data.shape[0])
            sequence = b._sequence + 1 if b._hierarchical_roce else 0
            if sequence > 0xFFFFFFFF:
                raise RuntimeError("RoCE invocation sequence exhausted; recreate SymmBuffer collectively")
            b._owners.update(owners)
            result = b._plan.forward(
                data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, *tables, clamp, sequence, out
            )
            # Keep the sequence in the buffer, not the BS-specific Plan. Failed
            # pre-launch validation does not consume an invocation ID.
            b._sequence = sequence
            b._domains.frozen = True
            if cumulative_stats is not None:
                cumulative_stats.add_(result[1])
                cumulative_stats.record_stream(runtime.stream)
        if caller != runtime.stream:
            caller.wait_stream(runtime.stream)
        for tensor in result:
            tensor.record_stream(caller)
        return result


def _accept_positional_fast_math(forward):
    @wraps(forward)
    def dispatch(*args, **kwargs):
        # The EP32 API accepted a positional bool before activation_clamp was added upstream.
        if len(args) == 10 and isinstance(args[9], bool):
            if "fast_math" in kwargs:
                raise TypeError("multiple values for fast_math")
            kwargs["fast_math"] = args[9]
            args = args[:9]
        return forward(*args, **kwargs)

    return dispatch


@_accept_positional_fast_math
def fp8_fp4_mega_moe(
    y: Tensor,
    l1_weights: WeightPair | WeightHandle,
    l2_weights: WeightPair | WeightHandle,
    sym_buffer: SymmBuffer,
    shared_l1_weights: WeightPair | WeightHandle | None = None,
    shared_l2_weights: WeightPair | WeightHandle | None = None,
    cumulative_local_expert_recv_stats: Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
    *,
    x: Tensor | None = None,
    timer: Tensor | None = None,
) -> None:
    """Write BF16 y in place, return None; accumulate stats if supplied.

    Positional parameters follow DeepGEMM. Ascend-only keyword extensions:
    x is optional BF16 [BS,H] (quantized internally, NOT BF16 GEMM);
    timer is a fresh buffer.allocate_timer() for a separate diagnostic call.
    Routed weights are applied after GMM2; see README numerical compatibility.
    """
    if recipe != (1, 1, 32) or activation != "swiglu" or fast_math is not True:
        raise NotImplementedError("Beta supports recipe=(1,1,32), SwiGLU, default math only")
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    import torch

    sym_buffer._ensure_alive()
    if y.ndim != 2 or y.dtype != torch.bfloat16:
        raise ValueError("y must be BF16 [BS,H]")
    bs = y.shape[0]
    if x is not None and (x.dtype != torch.bfloat16 or tuple(x.shape) != (bs, sym_buffer.hidden)):
        raise ValueError("x override must be BF16 [BS,H]; omit x for buffered FP8 input")
    _run(
        (sym_buffer.x[:bs], sym_buffer.x_sf[:bs]) if x is None else x,
        sym_buffer.topk_idx[:bs],
        sym_buffer.topk_weights[:bs],
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=y,
        activation_clamp=activation_clamp,
        timer=timer,
        cumulative_stats=cumulative_local_expert_recv_stats,
    )
